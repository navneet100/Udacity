#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
	is_initialized_ = false;

	previous_timestamp_ = 0;

	// initializing matrices
	R_laser_ = MatrixXd(2, 2);
	R_radar_ = MatrixXd(3, 3);

	H_laser_ = MatrixXd(2, 4);
	H_laser_ << 1, 0, 0, 0,
			    0, 1, 0, 0;

	Hj_ = MatrixXd(3, 4);

	//measurement covariance matrix - laser
	R_laser_ << 0.0225, 0,
			    0, 0.0225;

	//measurement covariance matrix - radar
	R_radar_ << 0.09, 0, 0,
			0, 0.0009, 0,
			0, 0, 0.09;

	/**
	 * Finish initializing the FusionEKF.
	 * Set the process and measurement noises
	 */
	ekf_.Q_ = MatrixXd(4, 4);

	//state covariance matrix P
	ekf_.P_ = MatrixXd(4, 4);
	ekf_.P_ << 1, 0, 0, 0,
                  0, 1, 0, 0,
                  0, 0, 1000, 0,
				  0, 0, 0, 1000;
	//the initial transition matrix F_
	ekf_.F_ = MatrixXd(4, 4);
	ekf_.F_ << 1, 0, 1, 0,
		   0, 1, 0, 1,
		   0, 0, 1, 0,
		   0, 0, 0, 1;


}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {
}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {

	/*****************************************************************************
	 *  Initialization
	 ****************************************************************************/
 if (!is_initialized_) {
    /**
    TODO:
      * Initialize the state ekf_.x_ with the first measurement.
      * Create the covariance matrix.
      * Remember: you'll need to convert radar from polar to cartesian coordinates.
    */
    // first measurement

    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;
	//ekf_.x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1],0,0;

	ekf_.F_ = MatrixXd(4, 4);

	ekf_.F_ << 1, 0, 1, 0,
		0, 1, 0, 1,
		0, 0, 1, 0,
		0, 0, 0, 1;

	ekf_.P_ = MatrixXd(4, 4);

	ekf_.P_ << 1, 0, 0, 0,
	0, 1, 0, 0,
	0, 0, 1000, 0,
	0, 0, 0, 1000;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {

	  /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
	  //measurement covariance
		ekf_.R_ = R_radar_;

		float rho = measurement_pack.raw_measurements_[0]; //range
		float th = measurement_pack.raw_measurements_[1]; //bearing
		float rhoRt = measurement_pack.raw_measurements_[2]; //rangeRt

		float thRt = 0;


		ekf_.x_[0] = rho * cos(th);
		ekf_.x_[1] = rho * sin(th);
		ekf_.x_[2] = rhoRt * sin(th) - thRt * sin(th) * rho;
		ekf_.x_[3] = rhoRt * cos(th) + thRt * cos(th) * rho;


		Hj_ = tools.CalculateJacobian(ekf_.x_);

		//measurement matrix
		ekf_.H_ = Hj_;

    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
		ekf_.x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;


		//measurement covariance
		ekf_.R_ = R_laser_;

		//measurement matrix
		ekf_.H_ = H_laser_;

    }




	previous_timestamp_ = measurement_pack.timestamp_ ;


    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

	/*****************************************************************************
	 *  Prediction
	 ****************************************************************************/

	/**
	 TODO:
	 * Update the state transition matrix F according to the new elapsed time.
	 - Time is measured in seconds.
	 * Update the process noise covariance matrix.
	 * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
	 */
	//compute the time elapsed between the current and previous measurements
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;	//dt - expressed in seconds
  //float dt = (measurement_pack.timestamp_ / 1000000.0 - previous_timestamp_) ;	//dt - expressed in seconds
  //moved below
  previous_timestamp_ = measurement_pack.timestamp_ ;


  float dt_2 = dt * dt;
  float dt_3 = dt_2 * dt;
  float dt_4 = dt_3 * dt;

  //Modify the F matrix so that the time is integrated
  ekf_.F_(0, 2) = dt;
  ekf_.F_(1, 3) = dt;

  float noise_ax = 9;
  float noise_ay = 9;


  //set the process covariance matrix Q
  ekf_.Q_ = MatrixXd(4, 4);
 
   ekf_.Q_ << (dt_4 / 4.0) * noise_ax, 0, (dt_3 / 2.0) * noise_ax, 0,
	  0, (dt_4 / 4.0) * noise_ay, 0, (dt_3 / 2.0) * noise_ay,
	  (dt_3 / 2.0) * noise_ax, 0, dt_2*noise_ax, 0,
	  0, (dt_3 / 2.0) * noise_ay, 0, dt_2*noise_ay;

 
  //if (dt > 0.001)
	ekf_.Predict();


	/*****************************************************************************
	 *  Update
	 ****************************************************************************/

	/**
	 TODO:
	 * Use the sensor type to perform the update step.
	 * Update the state and covariance matrices.
	 */

	if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
		
		// Radar updates
		
		Hj_ = tools.CalculateJacobian(ekf_.x_);
		ekf_.H_ = Hj_;
		ekf_.R_ = R_radar_;
		ekf_.UpdateEKF(measurement_pack.raw_measurements_);
		
	} else { 


	  //measurement covariance
	  ekf_.R_ = R_laser_;

	  //measurement matrix
	  ekf_.H_ = H_laser_;
	  ekf_.Update(measurement_pack.raw_measurements_);
	}

}
