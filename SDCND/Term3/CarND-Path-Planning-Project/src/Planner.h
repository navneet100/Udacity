#ifndef PLANNER_H
#define PLANNER_H

#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "Eigen-3.3/Eigen/Dense"
#include <vector>

#include "road.h"
#include "spline.h"



using namespace tk;
using namespace std;



class Planner {

private:
  int get_lane_from_d(double d);
  double get_d_value(int lane);
  double get_d_value(double actualD);

  int traj_points;
  string state;
  vector<double> start_s;
  vector<double> end_s;
  vector<double> start_d;
  vector<double> end_d;
  bool generate_points;

public:
  Planner();
  ~Planner(){};


  spline waypoints_spline_x;
  spline waypoints_spline_y;
  spline waypoints_spline_dx;
  spline waypoints_spline_dy;

  vector<double> getXY(double s, double d);

  double PolynomialEquate(vector<double> coefficients, double T);
  vector<double> JMT(vector<double> start, vector <double> end, double T);

  void generate_new_points(vector<vector<double>>& trajectory);
  void create_trajectory(Road& road, Vehicle& car, vector<vector<double>>& trajectory);


  void start_car(Vehicle& car);
  void keep_lane(Vehicle& car);
  void reduce_speed(Vehicle& car);
  void change_lane(Vehicle& car, int target_lane);

};

#endif
