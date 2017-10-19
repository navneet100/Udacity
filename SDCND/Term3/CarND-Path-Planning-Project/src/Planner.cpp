#include "Planner.h"
#include "constants.h"
#include <math.h>

#include "Eigen-3.3/Eigen/Dense"
using Eigen::MatrixXd;
using Eigen::VectorXd;


Planner::Planner()
{
  this->state = "CS";
}

int Planner::get_lane_from_d(double d)
{
  int lane;
  if (d < 4.0) {
    lane = 0;
  }
  else if ((d >= 4.0) && (d < 8.0))
  {
    lane = 1;
  }
  else {
    lane = 2;
  }
  return lane;
}

double Planner::get_d_value(int lane)
{
  double d;
  if (lane == 0)
  {
    d = 2.0;
  }
  else if (lane == 1)
  {
    d = 6.0;
  }
  else
  {
    d = 10.0;
  }
  return d;
}

double Planner::get_d_value(double actualD)
{
  double d;
  if (actualD < 4.0)
  {
    d = 2.0;
  }
  else if ((actualD >= 4.0) && (actualD < 8.0))
  {
    d = 6.0;
  }
  else
  {
    d = 10.0;
  }
  return d;
}

vector<double> Planner::JMT(vector<double> start, vector<double> end, double T)
{

  // Quintic Polynomial Solver

  MatrixXd A = MatrixXd(3, 3);

  A << T * T * T,  T * T * T *  T, T * T *  T * T * T,
       3 * T * T, 4 * T * T * T, 5 * T * T * T * T,
       6 * T, 12 * T * T, 20 * T * T * T;

  MatrixXd B = MatrixXd(3, 1);

  B << end[0] - (start[0] + start[1] * T + .5 * start[2] * T * T),
       end[1] - (start[1] + start[2] * T),
       end[2] - start[2];

  MatrixXd Ai = A.inverse();

  MatrixXd C = Ai * B;

  vector<double> result = { start[0], start[1], .5 * start[2] };

  for (int i = 0; i < C.size(); i++)
  {
    result.push_back(C.data()[i]);
  }
  return result;
}

double Planner::PolynomialEquate(vector<double> coefficients, double T)
{
  double x = 0.0f;
  for (unsigned i = 0; i < coefficients.size(); i++)
  {
    x += coefficients[i] * pow(T,i);
  }
  return x;
}


void Planner::create_trajectory(Road& road, Vehicle& car, vector<vector<double>>& trajectory)
{

  int current_num_points = trajectory[0].size();
  this->generate_points = false;


  if (current_num_points < TRAJ_POINTS)
  {
    this->generate_points = true;



   if (this->state == "CS") //car start
   {
      this->start_car(car);

   }
   else if (this->state == "KL") //keep the lane
   {
      if (road.is_lane_safe(car, car.lane))
      {
        this->keep_lane(car);
      }
      else
      {
        int lane_available = road.get_safe_lane(car);
        if (lane_available == car.lane)
        {
          this->reduce_speed(car);
        }
        else
        {
          this->change_lane(car, lane_available);
        }
      }
    }
    else
    {
      int new_lane = get_lane_from_d(car.previous_d[0]);

      if(road.is_lane_safe(car, new_lane)) //check this
      {
        this->keep_lane(car);
      }
      else
      {
          this->reduce_speed(car);
      }
    }
  }

  if (this->generate_points)
  {
    this->generate_new_points(trajectory);
  }

}




void Planner::generate_new_points(vector<vector<double>>& trajectory)
{

  double T = this->traj_points * TIME_STEP;

  vector<double> traj_s = this->JMT(this->start_s, this->end_s, T);
  vector<double> traj_d = this->JMT(this->start_d, this->end_d, T);

  double t, next_s, next_d, mod_s, mod_d;

  vector <double> XY;

  for(int i = 0; i < this->traj_points; i++)
  {

    t = TIME_STEP * i;


    next_s = 0.0;
    next_d = 0.0;

    next_s = PolynomialEquate(traj_s, t);
    next_d = PolynomialEquate(traj_d, t);


    mod_s = fmod(next_s, TRACK_LENGTH);

    mod_d = next_d;

    XY = getXY(mod_s, mod_d);

    trajectory[0].push_back(XY[0]);
    trajectory[1].push_back(XY[1]);
  }

}

vector<double> Planner::getXY(double s, double d)
{

  double wp_x, wp_y, wp_dx, wp_dy, next_x, next_y;


  wp_x = this->waypoints_spline_x(s);
  wp_y = this->waypoints_spline_y(s);
  wp_dx = this->waypoints_spline_dx(s);
  wp_dy = this->waypoints_spline_dy(s);

  next_x = wp_x + wp_dx * d;
  next_y = wp_y + wp_dy * d;

  return {next_x, next_y};

}

void Planner::start_car(Vehicle& car)
{


  this->traj_points = 2 * TRAJ_POINTS;


  car.s = fmod(car.s, TRACK_LENGTH);

  double target_v = car.v + SPEED_LIMIT * 0.25;
  double target_s = car.s + this->traj_points * TIME_STEP * target_v;;

  this->start_s = {car.s, car.v, 0.0};
  this->end_s = {target_s, target_v, 0.0};

  this->start_d = {get_d_value(car.lane), 0.0, 0.0};
  this->end_d = {get_d_value(car.lane), 0.0, 0.0};


  car.previous_s = this->end_s;
  car.previous_d = this->end_d;

  this->state = "KL";

}

void Planner::keep_lane(Vehicle& car){

  this->traj_points = TRAJ_POINTS;
  this->generate_points = true;//W0824

  car.s = fmod(car.s, TRACK_LENGTH);
  car.previous_s[0] = fmod(car.previous_s[0], TRACK_LENGTH);

  double speedLimit = SPEED_LIMIT;
  if(car.lane == 2)
  {
    speedLimit = SPEED_LIMIT * 0.9;
  }
  else if(car.lane == 1)
  {
    speedLimit = SPEED_LIMIT * 0.95;
  }
  else
  {
    speedLimit = SPEED_LIMIT;
  }

  double target_v = min(car.previous_s[1] * 1.1, speedLimit );
  double target_s = car.previous_s[0] + this->traj_points * TIME_STEP * target_v;

  this->start_s = {car.previous_s[0], car.previous_s[1], car.previous_s[2]};
  this->end_s = {target_s, target_v, 0.0};

  double target_d = get_d_value(car.previous_d[0]);

  this->start_d = {get_d_value(car.previous_d[0]), 0.0, 0.0};
  this->end_d = {target_d, 0.0, 0.0};

  car.previous_s = this->end_s;
  car.previous_d = this->end_d;

  this->state = "KL";
}

void Planner::reduce_speed(Vehicle& car){

  this->traj_points = TRAJ_POINTS ; // * 1.25 gives message outside of lane
  this->generate_points = true;

  car.s = fmod(car.s, TRACK_LENGTH);
  car.previous_s[0] = fmod(car.previous_s[0], TRACK_LENGTH);

  double target_v = max(car.previous_s[1]*0.8, SPEED_LIMIT * 0.5);

  double target_s = car.previous_s[0] + this->traj_points * TIME_STEP * target_v;

  this->start_s = {car.previous_s[0], car.previous_s[1], car.previous_s[2]};
  this->end_s = {target_s, target_v, 0.0};

  double target_d = get_d_value(car.previous_d[0]);

  this->start_d = {get_d_value(car.previous_d[0]), 0.0, 0.0};
  this->end_d = {target_d, 0.0, 0.0};

  car.previous_s = this->end_s;
  car.previous_d = this->end_d;

  this->state = "KL";
}

void Planner::change_lane(Vehicle& car, int target_lane){

  this->traj_points = TRAJ_POINTS;
  this->generate_points = true;

  car.s = fmod(car.s, TRACK_LENGTH);
  car.previous_s[0] = fmod(car.previous_s[0], TRACK_LENGTH);

  double target_v = car.previous_s[1];
  double target_s = car.previous_s[0] + this->traj_points * TIME_STEP * target_v;

  this->start_s = {car.previous_s[0], car.previous_s[1], car.previous_s[2]};
  this->end_s = {target_s, target_v, 0.0};

  double target_d = get_d_value(target_lane);

  this->start_d = {get_d_value(car.previous_d[0]), 0.0, 0.0};
  this->end_d = {target_d, 0.0, 0.0};

  car.previous_s = this->end_s;
  car.previous_d = this->end_d;

  int current_lane = get_lane_from_d(car.previous_d[0]);
  target_lane =  get_lane_from_d(target_d);

  if (current_lane == target_lane)
  {
    this->state = "KL";
  }
   else
   {
     if(current_lane == 0 )
     {
       this->state = "LCR";
     }
     else if(current_lane == 2)
     {
       this->state = "LCL";
     }
     else
     {
       if(target_lane == 0)
       {
         this->state = "LCL";
       }
       else
       {
         this->state = "LCR";
       }
     }
   }

}
