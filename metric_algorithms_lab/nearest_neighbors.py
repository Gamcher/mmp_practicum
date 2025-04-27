import numpy as np
from sklearn.neighbors import NearestNeighbors
from distances import euclidean_distance, cosine_distance


class KNNClassifier:
    def __init__(self, k: int, strategy: str = 'brute',
                 metric: str = 'euclidean', weights: bool = False,
                 test_block_size: int = 10):
        self.k = k

        if metric != 'euclidean' and metric != 'cosine':
            raise ValueError("Непраильное значение атрибута metric")
        self.metric = metric

        self.strategy = strategy
        if strategy == 'my_own':
            self.model = None
        elif strategy == 'kd_tree' or strategy == 'ball_tree':
            self.model = NearestNeighbors(
                algorithm=strategy, metric='euclidean')
        elif strategy == 'brute':
            self.model = NearestNeighbors(algorithm=strategy, metric=metric)
        else:
            raise ValueError("Неправильное значение атрибута strategy")

        self.weights = weights
        self.test_block_size = test_block_size

        # Atributes for fitted model

        self.fitted_X = None
        self.fitted_y = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        if self.model:
            self.model.fit(X, y)
        else:
            self.fitted_X = X
        self.fitted_y = y

    def find_kneighbors(self, X: np.ndarray, return_distance: bool):
        if self.strategy == 'my_own':
            if self.metric == 'cosine':
                distance_function = cosine_distance
            else:
                distance_function = euclidean_distance

            index_matrix = np.zeros((X.shape[0], self.k), dtype=int)
            if return_distance:
                distance_matrix = np.empty((X.shape[0], self.k))

            for i in range(0, int(np.ceil(X.shape[0]/self.test_block_size))):
                start = i * self.test_block_size
                end = (i+1) * self.test_block_size
                test_block = X[start: end, :]

                dist_block = distance_function(test_block, self.fitted_X)

                k_index_block_matrix = np.argsort(
                    dist_block, axis=1)[:, :self.k]

                if return_distance:
                    distance_matrix[start:end, :] = dist_block[np.arange(
                        k_index_block_matrix.shape[0])[:, None],
                        k_index_block_matrix]
                del dist_block

                index_matrix[start: end, :] = k_index_block_matrix
                del k_index_block_matrix

            if return_distance:
                return distance_matrix, index_matrix
            return index_matrix

        else:
            kneighbors = self.model.kneighbors(X, self.k, return_distance)
            return kneighbors

    def predict(self, X: np.ndarray):
        get_value_index = np.vectorize(lambda x: self.fitted_y[x])

        if self.weights:
            distance_matrix, index_matrix = self.find_kneighbors(
                X, return_distance=True)
            value_matrix = get_value_index(index_matrix)

            del index_matrix

            weights = 1 / (distance_matrix + 10**(-5))
            del distance_matrix

            predicted_classes = []

            for value_row, weight_row in zip(value_matrix, weights):
                values = np.unique(value_row)
                weighted_voices = []
                for value in values:
                    sum_weights = np.sum(
                        weight_row[value_row == value])
                    weighted_voices.append(sum_weights)

                weighted_voices = np.array(weighted_voices)
                predicted_classes.append(values[np.argmax(weighted_voices)])

            return np.array(predicted_classes)
        else:
            index_matrix = self.find_kneighbors(X, return_distance=False)
            value_matrix = get_value_index(index_matrix)
            del index_matrix

            def predict_class_in_row(row):
                values, counts = np.unique(row, return_counts=True)
                index_class = np.argmax(counts)
                return values[index_class]

            predicted_classes = np.apply_along_axis(
                predict_class_in_row, axis=1, arr=value_matrix)
            return predicted_classes
