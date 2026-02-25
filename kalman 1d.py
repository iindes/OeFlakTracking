import numpy as np

def run_1d_kalman_filter():
    dt = 1.0
    x = np.array([[0.0], [0.0]])
    P = np.array([[1000.0, 0.0], [0.0, 1000.0]])
    F = np.array([[1.0, dt], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    R = np.array([[10.0]])
    Q = np.array([[1.0, 0.0], [0.0, 3.0]])
    I = np.eye(2)

    measurements = [11.2, 18.5, 33.1, 41.8, 48.2, 62.5]
    
    for z_val in measurements:
        z = np.array([[z_val]])
        
        x = np.dot(F, x)
        P = np.dot(np.dot(F, P), F.T) + Q
        
        y = z - np.dot(H, x)
        S = np.dot(np.dot(H, P), H.T) + R
        K = np.dot(np.dot(P, H.T), np.linalg.inv(S))
        
        x = x + np.dot(K, y)
        P = np.dot((I - np.dot(K, H)), P)
        
        print(f"{x[0,0]:04.1f} {x[1,0]:04.1f}")

run_1d_kalman_filter()