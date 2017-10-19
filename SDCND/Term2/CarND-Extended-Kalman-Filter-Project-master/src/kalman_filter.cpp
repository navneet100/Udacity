#include "kalman_filter.h"
#include<iostream>
#include <math.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
  TODO:
    * predict the state
  */

  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Kalman Filter equations
  */
	

	
	VectorXd z_pred = H_ * x_ ;
	VectorXd y = z - z_pred;
	MatrixXd Ht = H_.transpose();
	MatrixXd S = H_ * P_ * Ht + R_;
	MatrixXd Si = S.inverse();
	MatrixXd PHt = P_ * Ht;
	MatrixXd K = PHt * Si;

	//new estimate
	x_ = x_ + (K * y);
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H_) * P_;

}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Extended Kalman Filter equations
  */


	float px = x_(0);
	float py = x_(1);
	float vx = x_(2);
	float vy = x_(3);

	float sq = sqrt(px * px + py * py);

	VectorXd z_pred(3);

	//float atanVal = atan2(py, px);
	float atanVal;


	float rhoDot = 0.0;
	if (fabs(sq) > 0.001)
		rhoDot = (px*vx + py*vy) / sq;

	if (px == 0 && py == 0) {
		z_pred << 0, 0, 0;
	}
	else
	{
		if (fabs(px) < 0.0001)
			atanVal = atan2(py, 0.0001);
		else
			atanVal = atan2(py, px);

		rhoDot = (px*vx + py*vy) / sq;
		z_pred << sq, atanVal, rhoDot; //check this
	}
	

	VectorXd y = z - z_pred;

	while (y[1] < -M_PI)
		y[1] += 2 * M_PI;
	while (y[1] > M_PI)
		y[1] -= 2 * M_PI;

	MatrixXd Ht = H_.transpose();
	MatrixXd S = H_ * P_ * Ht + R_;
	MatrixXd Si = S.inverse();
	MatrixXd PHt = P_ * Ht;
	MatrixXd K = PHt * Si;

	//new estimate
	x_ = x_ + (K * y);
	long x_size = x_.rows();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H_) * P_;

}
