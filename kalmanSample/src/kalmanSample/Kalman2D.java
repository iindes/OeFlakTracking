package kalmanSample;

public class Kalman2D {

	public static void main(String[] args) {
		RealMatrix x = MatrixUtils.createRealMatrix(4, 1);
        RealMatrix P = MatrixUtils.createRealIdentityMatrix(4).scalarMultiply(1000.0);
        
        RealMatrix F = MatrixUtils.createRealMatrix(new double[][]{
            {1, 0, dt, 0},
            {0, 1, 0, dt},
            {0, 0, 1, 0},
            {0, 0, 0, 1}
        });
        
        RealMatrix H = MatrixUtils.createRealMatrix(new double[][]{
            {1, 0, 0, 0},
            {0, 1, 0, 0}
        });
        
        RealMatrix R = MatrixUtils.createRealIdentityMatrix(2).scalarMultiply(50.0);
        RealMatrix Q = MatrixUtils.createRealIdentityMatrix(4).scalarMultiply(0.1);

	}

}
