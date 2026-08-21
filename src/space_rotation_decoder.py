# space_rotation_decoder.py
import numpy as np

def random_rotation_matrix(p, rng):
    A = rng.normal(size=(p, p))
    Q, R = np.linalg.qr(A)
    d = np.sign(np.diag(R))
    d[d == 0] = 1
    return Q * d

