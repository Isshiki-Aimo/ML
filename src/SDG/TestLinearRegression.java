package SDG;

public class TestLinearRegression {

    public static void main(String[] args) {
        // TODO Auto-generated method stub
        linearregression m = new linearregression("E:/java/machinelearning/Linearregression/traindata.txt",0.001,1000000);
        m.printTrainData();
        m.trainTheta();
        m.printTheta();
    }

}
