import numpy as np
from nearest_neighbors import KNNClassifier


np.int = int


def kfold(n: int, n_folds: int):
    x = np.arange(0, n)

    test_folds = np.array_split(x, n_folds)
    mask = np.ones(n, dtype=bool)

    train_folds = []

    for test_fold in test_folds:
        mask[test_fold] = False
        train_folds.append(np.nonzero(mask)[0])
        mask[test_fold] = True

    return list(zip(train_folds, test_folds))


def knn_cross_val_score(X: np.ndarray, y: np.ndarray, k_list: list,
                        score: str, cv: list = None, **kwargs):
    if cv is None:
        cv = kfold(X.shape[0], 3)

    def accuracy(y_pred, y_test):
        return np.sum(y_pred == y_test)/len(y_test)

    if score == 'accuracy':
        function_score = accuracy

    model = KNNClassifier(k=k_list[-1], **kwargs)
    score_dict = {k: np.zeros(len(cv), dtype=float) for k in k_list}

    for i, fold in enumerate(cv):
        X_train = X[fold[0], :]
        y_train = y[fold[0]]

        model.fit(X_train, y_train)

        X_test = X[fold[1], :]
        y_test = y[fold[1]]

        get_value_index = np.vectorize(lambda x: y_train[x])

        if model.weights:
            distance_matrix, index_matrix = model.find_kneighbors(
                X_test, return_distance=True)
            value_matrix = get_value_index(index_matrix)
            del index_matrix

            weights = 1 / (distance_matrix + 10**(-5))
            del distance_matrix

            for k in k_list:
                predicted_classes = []

                for value_row, weight_row in zip(
                        value_matrix[:, :k], weights[:, :k]):
                    values = np.unique(value_row)
                    weighted_voices = []
                    for value in values:
                        sum_weights = np.sum(
                            weight_row[value_row == value])
                        weighted_voices.append(sum_weights)

                    weighted_voices = np.array(weighted_voices)
                    predicted_classes.append(
                        values[np.argmax(weighted_voices)])

                y_pred = np.array(predicted_classes)

                del predicted_classes

                if score == 'accuracy':
                    score_dict[k][i] = function_score(y_pred, y_test)
        else:
            index_matrix = model.find_kneighbors(
                X_test, return_distance=False)

            value_matrix = get_value_index(index_matrix)
            del index_matrix

            def predict_class_in_row(row):
                values, counts = np.unique(row, return_counts=True)
                index_class = np.argmax(counts)
                return values[index_class]

            for k in k_list:
                predicted_classes = np.apply_along_axis(
                    predict_class_in_row, axis=1, arr=value_matrix[:, :k])

                y_pred = predicted_classes
                del predicted_classes

                if score == 'accuracy':
                    score_dict[k][i] = function_score(y_pred, y_test)

    return score_dict
