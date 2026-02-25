package kalmanSample;
import org.apache.commons.math3.linear.*;
public class Kalman1D {

	public static void main(String[] args) {
		double dt = 1.0;
        RealMatrix x = MatrixUtils.createRealMatrix(new double[][]{{0.0}, {0.0}});
        RealMatrix P = MatrixUtils.createRealMatrix(new double[][]{{1000.0, 0.0}, {0.0, 1000.0}});
        RealMatrix F = MatrixUtils.createRealMatrix(new double[][]{{1.0, dt}, {0.0, 1.0}});
        RealMatrix H = MatrixUtils.createRealMatrix(new double[][]{{1.0, 0.0}});
        RealMatrix R = MatrixUtils.createRealMatrix(new double[][]{{10.0}});
        RealMatrix Q = MatrixUtils.createRealMatrix(new double[][]{{1.0, 0.0}, {0.0, 3.0}});
        RealMatrix I = MatrixUtils.createRealIdentityMatrix(2);
        
        double[] measurements = {11.2, 18.5, 33.1, 41.8, 48.2, 62.5};
        
        for (double val : measurements) {
            RealMatrix z = MatrixUtils.createRealMatrix(new double[][]{{val}});
            
            x = F.multiply(x);
            P = F.multiply(P).multiply(F.transpose()).add(Q);
            
            RealMatrix y = z.subtract(H.multiply(x));
            RealMatrix S = H.multiply(P).multiply(H.transpose()).add(R);
            RealMatrix K = P.multiply(H.transpose()).multiply(new LUDecomposition(S).getSolver().getInverse());
            
            x = x.add(K.multiply(y));
            P = I.subtract(K.multiply(H)).multiply(P);
            
            System.out.printf("%04.1f %04.1f\n", x.getEntry(0, 0), x.getEntry(1, 0));

	}

}
