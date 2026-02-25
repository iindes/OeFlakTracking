import numpy as np

def init_3d_kalman_filter(dt):
    x = np.zeros((6, 1))
    P = np.eye(6) * 1000.0
    
    F = np.array([[1, 0, 0, dt, 0, 0],
                  [0, 1, 0, 0, dt, 0],
                  [0, 0, 1, 0, 0, dt],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1]])
                  
    H = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0]])
                  
    R = np.eye(3) * 50.0
    Q = np.eye(6) * 0.1
    
    return x, P, F, H, R, Q