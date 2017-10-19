#include "PID.h"

using namespace std;

/*
* TODO: Complete the PID class.
*/

PID::PID() {}

PID::~PID() {}

void PID::Init(double _Kp, double _Ki, double _Kd) {

	Kp = _Kp;
	Ki = _Ki;
	Kd = _Kd;

	prev_cte = 0.0;
	sum_cte = 0.0;
}

void PID::UpdateError(double cte) {

	sum_cte += cte;

	p_error = -Kp * cte;
	i_error = -Ki * (sum_cte);
	d_error = -Kd * (cte - prev_cte);
	
	prev_cte = cte;
}

double PID::TotalError() {

	double total_error = p_error + i_error + d_error;

	if (total_error > 1)
		total_error = 1.0;
	else
		if (total_error < -1)
			total_error = -1.0;

	return (total_error);
}

