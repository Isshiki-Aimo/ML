/*
 *    LinearRegreession.java
 *    add a bias b
 *    自己实现一个最简单的闭式解   2020 Wenjun Zhang
 *
 */

package LinearRegression_weka;

import java.util.Arrays;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.matrix.Matrix;

public class LinearRegression extends Classifier{
	//数据成员
	private double[] m_Wb;				//参数数组
	
	private Instances m_Instances;		//实例集合
	private int m_NumAtts;				//属性个数
	private int m_NumInstances;			//实例个数

	//训练函数
	public void buildClassifier(Instances train) throws Exception {
		m_Instances = new Instances(train);
		m_NumAtts = m_Instances.numAttributes();
		m_NumInstances = m_Instances.numInstances();
		m_Wb = new double[m_NumAtts];
		//给矩阵赋初值
		Matrix Matrix_X = new Matrix(m_NumInstances, m_NumAtts);
		Matrix Matrix_Y = new Matrix(m_NumInstances,1);
		for(int i=0;i<m_NumInstances;i++) {
			Matrix_Y.set(i, 0, m_Instances.instance(i).classValue());
			for(int j=0;j<m_NumAtts-1;j++) {
				Matrix_X.set(i, j, m_Instances.instance(i).value(j));
			}
			Matrix_X.set(i, m_NumAtts-1, 1);
		}
		//按照最小二乘求解线性回归
	    boolean success = true;
	    double ridge = 0.1;
	    Matrix solution = new Matrix(m_NumAtts, 1);
	    do {
	      Matrix ss = Matrix_X.transpose().times(Matrix_X);
	      // 对角线加上一个值保证满秩
	      for (int i = 0; i < m_NumAtts; i++)
	        ss.set(i, i, ss.get(i, i) + ridge);
	      Matrix bb = Matrix_X.transpose().times(Matrix_Y);
	      try {
	    	solution = ss.solve(bb);
	        success = true;
	      } 
	      catch (Exception ex) {
	        ridge *= 10;
	        success = false;
	      }
	    } while (!success);
		for(int i=0;i<m_NumAtts;i++) {
			m_Wb[i] = solution.get(i, 0);
		}
	}
	
	//预测函数
	public double classifyInstance(Instance instance) throws Exception {
		double temp = 0;
		for(int i=0;i<m_NumAtts-1;i++) {
			temp += m_Wb[i] * instance.value(i);
		}
		temp += m_Wb[m_NumAtts-1];
		return temp;
	}
	
	//主函数
	public static void main(String argv[]) {
		runClassifier(new LinearRegression(), argv);
	}
}