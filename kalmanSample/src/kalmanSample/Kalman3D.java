package kalmanSample;

public class Kalman3D {

	public static void main(String[] args) {
		RealMatrix x = MatrixUtils.createRealMatrix(6, 1);
        RealMatrix P = MatrixUtils.createRealIdentityMatrix(6).scalarMultiply(1000.0);
        
        RealMatrix F = MatrixUtils.createRealMatrix(new double[][]{
            {1, 0, 0, dt, 0, 0},
            {0, 1, 0, 0, dt, 0},
            {0, 0, 1, 0, 0, dt},
            {0, 0, 0, 1, 0, 0},
            {0, 0, 0, 0, 1, 0},
            {0, 0, 0, 0, 0, 1}
        });
        
        RealMatrix H = MatrixUtils.createRealMatrix(new double[][]{
            {1, 0, 0, 0, 0, 0},
            {0, 1, 0, 0, 0, 0},
            {0, 0, 1, 0, 0, 0}
        });
        
        RealMatrix R = MatrixUtils.createRealIdentityMatrix(3).scalarMultiply(50.0);
        RealMatrix Q = MatrixUtils.createRealIdentityMatrix(6).scalarMultiply(0.1);

	}

}
