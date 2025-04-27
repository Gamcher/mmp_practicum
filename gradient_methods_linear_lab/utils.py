import numpy as np


def grad_finite_diff(function, w, eps=1e-8):
    """
    Возвращает численное значение градиента, подсчитанное по следующией формуле:
        result_i := (f(w + eps * e_i) - f(w)) / eps,
        где e_i - следующий вектор:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    grad = np.zeros_like(w, dtype=float)
    f_w = function(w)
    
    for i in range(len(w)):
        e_i = np.zeros_like(w)
        e_i[i] = 1
        
        w_plus_eps = w + eps * e_i
        f_w_plus_eps = function(w_plus_eps)
        
        grad[i] = (f_w_plus_eps - f_w) / eps
    
    return grad
