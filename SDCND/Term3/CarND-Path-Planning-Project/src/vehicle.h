#ifndef VEHICLE_H
#define VEHICLE_H
#include <iostream>
#include <random>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>
#include <map>
#include <string>
#include <iterator>

#include <cassert>
#include <algorithm>
#include <cmath>

using namespace std;

class Vehicle {

public:

  int id;
  double x;
  double y;
  double vx;
  double vy;
  int lane;
  double d;
  double s;
  double v;
  double a;
	double yaw;

	double speed;

	string state;


	Vehicle();
  Vehicle(int id, double x, double y, double vx, double vy, double s, double d, int lane) ;
	//Vehicle(double x, double y, double s, double d, int lane, double angle_deg, double speed, string state, Vehicle * ego_prev) ;

  virtual ~Vehicle();

	vector<double> previous_s;
  vector<double> previous_d;

	void update_car_state(double x, double y, double v, double s, double d, double yaw);
};
#endif

