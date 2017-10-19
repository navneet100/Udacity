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

