import numpy as np
import scipy as sp
import time

from oracles import BinaryLogistic


class GDClassifier:
    """
    Реализация метода градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """

    def __init__(
        self, loss_function, step_alpha=1, step_beta=0,
        tolerance=1e-5, max_iter=1000, **kwargs
    ):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора.
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия

        step_alpha - float, параметр выбора шага из текста задания

        step_beta- float, параметр выбора шага из текста задания

        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию.
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход

        max_iter - максимальное число итераций

        **kwargs - аргументы, необходимые для инициализации
        """
        if loss_function == "binary_logistic":
            if 'l2_coef' in kwargs:
                self.loss_function = BinaryLogistic(kwargs["l2_coef"])
            else:
                self.loss_function = BinaryLogistic(0)
        else:
            raise ValueError("Неправильное имя для функции потери")

        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.w = None

    def fit(self, X, y, w_0=None, trace=False):
        """
        Обучение метода по выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w_0 - начальное приближение в методе

        trace - переменная типа bool

        Если trace = True, то метод должен вернуть словарь history, содержащий информацию
        о поведении метода. Длина словаря history = количество итераций + 1 (начальное приближение)

        history['time']: list of floats, содержит интервалы времени между двумя итерациями метода
        history['func']: list of floats, содержит значения функции на каждой итерации
        (0 для самой первой точки)
        history['accuracy']: list of floats, содержит значения точности классификатора на каждой итерации.
        """
        # Инициализация начальных весов
        if w_0 is None:
            self.w = np.zeros(X.shape[1])
        else:
            self.w = w_0
        time_start = time.time()
        history = {'time': [0], 'func': [self.get_objective(X, y)],
                   'accuracy': [self.get_accuracy(X, y)]}

        for i in range(1, self.max_iter+1):
            theta = self.step_alpha/np.power(i, self.step_beta)
            self.w = self.w - theta * self.get_gradient(X, y)

            history['func'].append(self.get_objective(X, y))
            history['time'].append(time.time()-time_start)
            history['accuracy'].append(self.get_accuracy(X, y))
            if np.sum(np.abs(history['func'][-1] - history['func'][-2])) < self.tolerance:
                break
        if trace:
            return history

    def predict(self, X):
        """
        Получение меток ответов на выборке X

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        return: одномерный numpy array с предсказаниями
        """
        sign_vector = np.sign(X.dot(self.w))
        return np.where(sign_vector == 0, 1, sign_vector)

    def predict_proba(self, X):
        """
        Получение вероятностей принадлежности X к классу k

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        return: двумерной numpy array, [i, k] значение соответветствует вероятности
        принадлежности i-го объекта к классу k
        """
        expit_vector = sp.special.expit(X.dot(self.w))
        return np.vstack((expit_vector, -expit_vector + 1))

    def get_objective(self, X, y):
        """
        Получение значения целевой функции на выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array

        return: float
        """
        return self.loss_function.func(X, y, self.w)

    def get_gradient(self, X, y):
        """
        Получение значения градиента функции на выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array

        return: numpy array, размерность зависит от задачи
        """
        return self.loss_function.grad(X, y, self.w)

    def get_weights(self):
        """
        Получение значения весов функционала
        """
        return self.w

    def get_accuracy(self, X, y):
        """
        Получение значения точности
        """
        return np.sum(self.predict(X) == y)/len(y)


class SGDClassifier(GDClassifier):
    """
    Реализация метода стохастического градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """

    def __init__(
        self, loss_function, batch_size, step_alpha=1, step_beta=0,
        tolerance=1e-5, max_iter=1000, random_seed=153, **kwargs
    ):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора.
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия

        batch_size - размер подвыборки, по которой считается градиент

for step_alpha in tqdm(step_alpha_arr):
        step_alpha - float, параметр выбора шага из текста задания

        step_beta- float, параметр выбора шага из текста задания

        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход

        max_iter - максимальное число итераций (эпох)

        random_seed - в начале метода fit необходимо вызвать np.random.seed(random_seed).
        Этот параметр нужен для воспроизводимости результатов на разных машинах.

        **kwargs - аргументы, необходимые для инициализации
        """
        if loss_function == "binary_logistic":
            if 'l2_coef' in kwargs:
                self.loss_function = BinaryLogistic(kwargs["l2_coef"])
            else:
                self.loss_function = BinaryLogistic(0)
        else:
            raise ValueError("Неправильное имя для функции потери")

        self.batch_size = batch_size
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.w = None

        self.random_seed = random_seed

    def fit(self, X, y, w_0=None, trace=False, log_freq=1):
        """
        Обучение метода по выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w_0 - начальное приближение в методе

        Если trace = True, то метод должен вернуть словарь history, содержащий информацию
        о поведении метода. Если обновлять history после каждой итерации, метод перестанет
        превосходить в скорости метод GD. Поэтому, необходимо обновлять историю метода лишь
        после некоторого числа обработанных объектов в зависимости от приближённого номера эпохи.
        Приближённый номер эпохи:
            {количество объектов, обработанных методом SGD} / {количество объектов в выборке}

        log_freq - float от 0 до 1, параметр, отвечающий за частоту обновления.
        Обновление должно проиходить каждый раз, когда разница между двумя значениями приближённого номера эпохи
        будет превосходить log_freq.

        history['epoch_num']: list of floats, в каждом элементе списка будет записан приближённый номер эпохи:
        history['time']: list of floats, содержит интервалы времени между двумя соседними замерами
        history['func']: list of floats, содержит значения функции после текущего приближённого номера эпохи
        history['weights_diff']: list of floats, содержит квадрат нормы разности векторов весов с соседних замеров
        history['accuracy']: list of floats, содержит значения точности классификатора на каждой эпохе.
        (0 для самой первой точки)
        """
        np.random.seed(self.random_seed)

        if w_0 is None:
            self.w = np.zeros(X.shape[1])
        else:
            self.w = w_0

        # количество обработанных объектов
        count_objects, iter_num = 0, 0

        time_start = time.time()
        history = {'epoch_num': [0], 'time': [0], 'func': [
            self.get_objective(X, y)], 'weights_diff': [0],
            'accuracy': [self.get_accuracy(X, y)]}

        while iter_num < self.max_iter:

            theta = self.step_alpha/np.power(iter_num+1, self.step_beta)

            indices = np.random.permutation(X.shape[0])
            X = X[indices]
            y = y.iloc[indices]

            for i in range(0, int(X.shape[0]/self.batch_size)):
                last_w = self.w
                self.w = self.w - theta * self.get_gradient(
                    X[i: i + self.batch_size][:], y[i: i + self.batch_size])

                count_objects += self.batch_size
                epoch = count_objects/X.shape[0]
                if abs(history['epoch_num'][-1] - epoch) > log_freq:
                    history['epoch_num'].append(epoch)
                    history['time'].append(time.time()-time_start)
                    history['func'].append(self.get_objective(X, y))
                    history['weights_diff'].append(
                        np.dot((self.w-last_w),
                               (self.w-last_w)))
                    history['accuracy'].append(self.get_accuracy(X, y))
                    if np.sum(np.abs(history['func'][-1] - history['func'][-2])) < self.tolerance:
                        break
            iter_num += 1
        if trace:
            return history
