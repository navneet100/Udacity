#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.9;//0.75;//30;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.5;//0.75;//30;

 
  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
  is_initialized_ = false;

  ///* State dimension
  n_x_ = 5;

  ///* Augmented state dimension
  n_aug_ = 7;

  ///* Sigma point spreading parameter
  lambda_ = 3 - n_x_;

  weights_ = VectorXd(2 * n_aug_ + 1);
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  for (int i = 1; i < 2 * n_aug_ + 1; i++)
  {
	  weights_(i) = 0.5 / (lambda_ + n_aug_);
  }
	
  Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  n_z_lidar = 2;
  n_z_radar = 3;

  R_lidar = MatrixXd::Zero(n_z_lidar, n_z_lidar);
  R_radar = MatrixXd(n_z_radar, n_z_radar);

  R_lidar(0, 0) = std_laspx_ * std_laspx_;
  R_lidar(1, 1) = std_laspy_ * std_laspy_;

  R_radar << std_radr_*std_radr_, 0, 0,
	  0, std_radphi_*std_radphi_, 0,
	  0, 0, std_radrd_*std_radrd_;
 

}

UKF::~UKF() {}


void UKF::AugmentedSigmaPoints()
{

	//create augmented mean vector
	VectorXd x_aug = VectorXd(n_aug_); //7

	//create augmented state covariance
	MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);//7,7

	//create augmented mean state
	x_aug.head(n_x_) = x_; //5
	x_aug(n_x_) = 0;
	x_aug(n_x_ + 1) = 0;

	//create augmented covariance matrix
	P_aug.fill(0.0);
	P_aug.topLeftCorner(n_x_, n_x_) = P_;
	P_aug(n_x_, n_x_) = std_a_ * std_a_;
	P_aug(n_x_ + 1, n_x_ + 1) = std_yawdd_ * std_yawdd_;

	//create square root matrix
	MatrixXd L = P_aug.llt().matrixL();

	//create augmented sigma points
	Xsig_aug.col(0) = x_aug;
	for (int i = 0; i< n_aug_; i++)
	{
		Xsig_aug.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
		Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
	}


}


void UKF::SigmaPointPrediction(double delta_t)
{

	//create matrix with predicted sigma points as columns
	Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

						  //predict sigma points
	for (int i = 0; i< 2 * n_aug_ + 1; i++)
	{
		//extract values for better readability
		double p_x = Xsig_aug(0, i);
		double p_y = Xsig_aug(1, i);
		double v = Xsig_aug(2, i);
		double yaw = Xsig_aug(3, i);
		double yawd = Xsig_aug(4, i);
		double nu_a = Xsig_aug(5, i);
		double nu_yawdd = Xsig_aug(6, i);

		//predicted state values
		double px_p, py_p;

		//avoid division by zero
		if (fabs(yawd) > 0.001) {
			px_p = p_x + v / yawd * (sin(yaw + yawd*delta_t) - sin(yaw));
			py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd*delta_t));
		}
		else {
			px_p = p_x + v*delta_t*cos(yaw);
			py_p = p_y + v*delta_t*sin(yaw);
		}

		double v_p = v;
		double yaw_p = yaw + yawd*delta_t;
		double yawd_p = yawd;

		//add noise
		px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
		py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
		v_p = v_p + nu_a*delta_t;

		yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
		yawd_p = yawd_p + nu_yawdd*delta_t;

		//write predicted sigma point into right column
		Xsig_pred_(0, i) = px_p;
		Xsig_pred_(1, i) = py_p;
		Xsig_pred_(2, i) = v_p;
		Xsig_pred_(3, i) = yaw_p;
		Xsig_pred_(4, i) = yawd_p;
	}

}


void UKF::PredictMeanAndCovariance()
{
	//predicted state mean
	x_.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
		x_ = x_ + weights_(i) * Xsig_pred_.col(i);
	}

	//predicted state covariance matrix
	P_.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++)
	{  
	   // state difference
		VectorXd x_diff = Xsig_pred_.col(i) - x_;
		//angle normalization
		while (x_diff(3)> M_PI) x_diff(3) -= 2.*M_PI;
		while (x_diff(3)<-M_PI) x_diff(3) += 2.*M_PI;

		P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
	}

}


/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package)
{
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  
*/
	//measurement_pack.raw_measurements_[0]

	if (!is_initialized_)
	{
		
		if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
		{
			use_radar_ = true;
			use_laser_ = false;

			double rho = meas_package.raw_measurements_[0]; //range
			double th = meas_package.raw_measurements_[1]; //bearing
			double rhoRt = meas_package.raw_measurements_[2]; //rangeRt

			double thRt = 0;


			x_[0] = rho * cos(th);
			x_[1] = rho * sin(th);
			x_[2] = 0.0;
			x_[3] = 0.0;
			x_[4] = 0.0;


			R_radar << std_radr_*std_radr_, 0, 0,
				0, std_radphi_*std_radphi_, 0,
				0, 0, std_radrd_*std_radrd_;
		}
		else 
			//if (measurement_pack.sensor_type_ == MeasurementPackage::LASER)
		{
			use_radar_ = false;
			use_laser_ = true;

			double px = meas_package.raw_measurements_[0];
			double py = meas_package.raw_measurements_[1];

			if (fabs(px) < 1e-3)
				px = 1e-5;
			if (fabs(py) < 1e-3)
				py = 1e-5;

			x_(0) = px;
			x_(1) = py;
			x_(2) = 0.0;
			x_(3) = 0.0;
			x_(4) = 0.0;



		}

	
		P_.fill(0.00001);
		
		P_(0, 0) = 0.1;
		P_(1, 1) = 0.1;
		P_(2, 2) = 0.1;
		P_(3, 3) = 0.01;
		P_(4, 4) = 0.1;
		
		previous_timestamp_ = meas_package.timestamp_;
		// done initializing, no need to predict or update
		is_initialized_ = true;
		return;
	}


	double deltaT = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0;

	Prediction(deltaT);

	
	if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
	{
		// Radar updates
		UpdateRadar(meas_package);
	}
	else 
	{
		UpdateLidar(meas_package);
	}
	
	previous_timestamp_ = meas_package.timestamp_;
}


/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */

	AugmentedSigmaPoints();

	SigmaPointPrediction(delta_t);

	PredictMeanAndCovariance();
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */

	VectorXd z_out;
	MatrixXd S_out;



	MatrixXd Zsig = MatrixXd(n_z_lidar, 2 * n_aug_ + 1);

	for (int i = 0; i < 2 * n_aug_ + 1; i++)//2n+1 simga points
	{
		
		// extract values for better readibility
		double p_x = Xsig_pred_(0, i);
		double p_y = Xsig_pred_(1, i);

		// measurement model
		Zsig(0, i) = p_x;     
		Zsig(1, i) = p_y;
	}


	VectorXd z_pred = VectorXd(n_z_lidar);
	z_pred.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++)
	{
		z_pred = z_pred + weights_(i) * Zsig.col(i);
	}

	//measurement covariance matrix S
	MatrixXd S = MatrixXd(n_z_lidar, n_z_lidar);
	S.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 sigma points
												//residual
		VectorXd z_diff = Zsig.col(i) - z_pred;

		//angle normalization
		while (z_diff(1)> M_PI) z_diff(1) -= 2.*M_PI;
		while (z_diff(1)<-M_PI) z_diff(1) += 2.*M_PI;

		S = S + weights_(i) * z_diff * z_diff.transpose();
	}


	S = S + R_lidar;

	MatrixXd Tc = MatrixXd(n_x_, n_z_lidar);
	Tc.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++)
	{  //2n+1 simga points
	   //residual

		VectorXd z_diff = Zsig.col(i) - z_pred;

		//angle normalization
		while (z_diff(1)> M_PI) z_diff(1) -= 2.*M_PI;
		while (z_diff(1)<-M_PI) z_diff(1) += 2.*M_PI;


		// state difference
		VectorXd x_diff = Xsig_pred_.col(i) - x_;

		//angle normalization
		while (x_diff(3)> M_PI) x_diff(3) -= 2.*M_PI;//navneet
		while (x_diff(3)<-M_PI) x_diff(3) += 2.*M_PI;//navneet

		Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
	}

	MatrixXd K = Tc * S.inverse();

	VectorXd z =  meas_package.raw_measurements_;

	//residual
	VectorXd z_diff = z - z_pred;
	
	//angle normalization
	while (z_diff(1)> M_PI) z_diff(1) -= 2.*M_PI;
	while (z_diff(1)<-M_PI) z_diff(1) += 2.*M_PI;


	//update state mean and covariance matrix
	x_ = x_ + K * z_diff;
	P_ = P_ - K * S * K.transpose();

	NIS_laser_ = z_diff.transpose() * S.inverse() * z_diff;

}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */

	VectorXd z_out;
	MatrixXd S_out;


	MatrixXd Zsig = MatrixXd(n_z_radar, 2 * n_aug_ + 1);

	//PredictRadarMeasurement
	//transform sigma points into measurement space
	for (int i = 0; i < 2 * n_aug_ + 1; i++)
	{
		//2n+1 simga points
		// extract values for better readibility
		double p_x = Xsig_pred_(0, i);
		double p_y = Xsig_pred_(1, i);
		double v = Xsig_pred_(2, i);
		double yaw = Xsig_pred_(3, i);

		double v1 = cos(yaw)*v;
		double v2 = sin(yaw)*v;

		double rho = sqrt(p_x*p_x + p_y*p_y);
		if (rho < 0.0001)
			return;

		// measurement model
		Zsig(0, i) = rho;                        //r
		Zsig(1, i) = atan2(p_y, p_x);                                 //phi
		Zsig(2, i) = (p_x*v1 + p_y*v2) / sqrt(p_x*p_x + p_y*p_y);   //r_dot
	}

	VectorXd z_pred = VectorXd(n_z_radar);
	z_pred.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++)
	{
		z_pred = z_pred + weights_(i) * Zsig.col(i);
	}

	//measurement covariance matrix S
	MatrixXd S = MatrixXd(n_z_radar, n_z_radar);
	S.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 sigma points
												//residual
		VectorXd z_diff = Zsig.col(i) - z_pred;

		S = S + weights_(i) * z_diff * z_diff.transpose();
	}


	S = S + R_radar;


	VectorXd z = VectorXd(n_z_radar);

	z << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1],
		meas_package.raw_measurements_[2];
	
	MatrixXd Tc = MatrixXd(n_x_, n_z_radar);
	Tc.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) //2n+1 simga points
	{  
	   //residual
		VectorXd z_diff = Zsig.col(i) - z_pred;

		//angle normalization
		while (z_diff(1)> M_PI) z_diff(1) -= 2.*M_PI;
		while (z_diff(1)<-M_PI) z_diff(1) += 2.*M_PI;

		// state difference
		VectorXd x_diff = Xsig_pred_.col(i) - x_;

		//angle normalization
		while (x_diff(3)> M_PI) x_diff(3) -= 2.*M_PI;
		while (x_diff(3)<-M_PI) x_diff(3) += 2.*M_PI;


		Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
	}


	MatrixXd K = Tc * S.inverse();

	//residual
	VectorXd z_diff = z - z_pred;


	//angle normalization
	while (z_diff(1)> M_PI) z_diff(1) -= 2.*M_PI;
	while (z_diff(1)<-M_PI) z_diff(1) += 2.*M_PI;


	//update state mean and covariance matrix
	x_ = x_ + K * z_diff;
	P_ = P_ - K * S * K.transpose();

	NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;

}
