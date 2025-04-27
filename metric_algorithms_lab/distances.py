import numpy as np


def euclidean_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    E = np.expand_dims(np.sum(X**2, axis=1), axis=1) - 2 * \
        np.dot(X, Y.T) + np.sum(Y**2, axis=1)
    return np.sqrt(E)


def cosine_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    E = np.dot(X, Y.T)
    size_X = np.sqrt(np.sum(X ** 2, axis=1))
    size_Y = np.sqrt(np.sum(Y ** 2, axis=1))
    size_X = np.expand_dims(size_X, axis=1)
    E = np.divide(E, size_X, out=np.zeros_like(
        E, dtype=float), where=size_X != 0)
    E = np.divide(E, size_Y, out=np.zeros_like(
        E, dtype=float), where=size_Y != 0)
    cosine_E = E - 1.0
    return -1.0 * cosine_E
