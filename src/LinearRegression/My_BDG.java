package LinearRegression;

import java.util.ArrayList;
import java.util.List;


public class My_BDG {
    //求梯度
    public List<Double> sumOfGradient(final List<Double> x,
                                      final List<Double>y,
                                      final List<Double>thetas){
        int m = x.size();
        double sum = 0;
        double sum1 = 0;
        for (int i = 0; i < m; ++i) {
            sum += thetas.get(0) + thetas.get(1) * x.get(i) - y.get(i);
            sum1 += (thetas.get(0) + thetas.get(1) * x.get(i) - y.get(i))*x.get(i);
        }
        double grad0 = 1.0 / m * sum;
        double grad1 = 1.0 / m * sum1;

        List<Double> result = new ArrayList<>();
        result.add(grad0);
        result.add(grad1);
        return result;
    }

    /**
     * 梯度下降 更新参数
     * @param thetas    参数
     * @param direction 梯度
     * @param stepSize  步长
     * @return        更新后的参数
     */
    public List<Double> step(final List<Double> thetas,
                             final List<Double> direction,
                             double stepSize){
        List<Double> result = new ArrayList<>();
        for (int i = 0; i < direction.size(); ++i) {
            result.add(thetas.get(i) + stepSize * direction.get(i));
        }
        return result;
    }

    public double distance(final List<Double> v, final List<Double> w){
        List<Double> subtract = new ArrayList<>();
        for (int i = 0; i < v.size(); ++i) {
            subtract.add(Math.pow(v.get(i) - w.get(i), 2));
        }
        double sum = 0;
        for (int i = 0; i < v.size(); ++i) {
            sum += subtract.get(i);
        }
        return Math.sqrt(sum);
    }

    public List<Double> gradientDescent(double stepSize,
                                        final List<Double> x,
                                        final List<Double> y,
                                        double tolerance, int maxIter){
        int iterNum = 0;
        List<Double> thethas = new ArrayList<>();
        thethas.add(0D);
        thethas.add(0D);
        thethas.add(0D);
        while(true){
            List<Double> gradients = sumOfGradient(x, y, thethas);
            List<Double> nextThetas = step(thethas, gradients, stepSize);
            if(distance(nextThetas, thethas) < tolerance){
                break;
            }
            thethas = nextThetas;
            iterNum += 1;

            if(iterNum == maxIter){
                System.out.println("Max iterations exceeded!");
                break;
            }
        }
        return thethas;
    }

    public static void main(String[] args) {
        My_BDG gradientDescent = new My_BDG();
        List<Double> x = new ArrayList<>();
        x.add(1d);
        x.add(2d);
        x.add(3d);

        List<Double> y = new ArrayList<>();
        y.add(5d);
        y.add(9d);
        y.add(13d);

        double stepSize = 0.001;
        List<Double> result = gradientDescent.gradientDescent(-stepSize, x, y,  0.0000001, 10000000);
        System.out.println("theta0 = "+result.get(0) +"; theta1 = "+result.get(1));
    }
}