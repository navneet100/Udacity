#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
  TODO:
    * Calculate the RMSE here.
  */
  // TODO: YOUR CODE HERE

  // check the validity of the following inputs:
  //  * the estimation vector size should not be zero
  //  * the estimation vector size should equal ground truth vector size
  // ... your code here
	
	VectorXd rmse(4);
	rmse << 0, 0, 0, 0;

	if (estimations.size() == 0 || ground_truth.size() == 0 || (estimations.size() != ground_truth.size()))
	{
		std::cout << " Input arrays do not have appropriate size";
		return rmse;
	}

	//accumulate squared residuals
	//double res = 0;
	for (int i = 0; i < estimations.size(); ++i) {
		// ... your code here
		VectorXd diff = estimations[i] - ground_truth[i];
		VectorXd diff_sq = diff.array() * diff.array();
		rmse += diff_sq;

		//rmse += diff.cwiseAbs2();

	}

	//calculate the mean
	// ... your code here
	rmse = rmse / estimations.size();
	//calculate the squared root
	// ... your code here
	rmse = rmse.array().sqrt();
	//rmse = rmse.cwiseSqrt();
	//return the result
	return rmse;

}


MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
  TODO:
    * Calculate a Jacobian here.
  */
	MatrixXd Hj(3, 4);
	//recover state parameters
	double px = x_state(0);
	double py = x_state(1);
	double vx = x_state(2);
	double vy = x_state(3);

	//pre-compute a set of terms to avoid repeated calculation
	double c1 = px*px + py*py;
	double c2 = sqrt(c1);
	double c3 = (c1*c2);

	Hj << 0, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0;

	//check division by zero
	/*
	if (fabs(c1) < 0.0001) {
	std::cout << "CalculateJacobian () - Error - Division by Zero \n" ;
	return Hj;
	}

	//compute the Jacobian matrix
	Hj << (px / c2), (py / c2), 0, 0,
	-(py / c1), (px / c1), 0, 0,
	py*(vx*py - vy*px) / c3, px*(px*vy - py*vx) / c3, px / c2, py / c2;

	*/

	if (px == 0 && py == 0) {
		Hj << 0, 0, 0, 0,
			0, 0, 0, 0,
			0, 0, 0, 0;
		return Hj;
	}

	//check division by zero
	if (c2 < 1e-6) {
		std::cerr << "Divide by zero!!" << std::endl;
		return Hj;
	}
	if (abs(c2) < 1e-4) {
		Hj << 0, 0, 0, 0,
			0, 0, 0, 0,
			0, 0, 0, 0;
	}
	else {
		//compute the Jacobian matrix
		/*
		Hj << (px / c2), (py / c2), 0, 0,
			-(py / c1), (px / c1), 0, 0,
			py*(vx*py - vy*px) / c3, px*(px*vy - py*vx) / c3, px / c2, py / c2;
		*/
		double px2py2 = pow(px, 2) + pow(py, 2);

		Hj(0, 0) = px / sqrt(px2py2);
		Hj(0, 1) = py / sqrt(px2py2);

		Hj(1, 0) = -py / px2py2;
		Hj(1, 1) = px / px2py2;

		Hj(2, 0) = py* (vx*py - vy*px) / pow(px2py2, 3 / 2.0);
		Hj(2, 1) = px * (vy*px - vx*py) / pow(px2py2, 3 / 2.0);
		Hj(2, 2) = px / sqrt(px2py2);
		Hj(2, 3) = py / sqrt(px2py2);
	}


	return Hj;
}
