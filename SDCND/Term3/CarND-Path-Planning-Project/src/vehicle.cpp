
#include <iostream>
#include <math.h>
#include <map>
#include <string>
#include <iterator>


#include "vehicle.h"
//#include "costs.h"

/**
 * Initializes Vehicle
 */
//Vehicle::Vehicle() {};

Vehicle::Vehicle()
{
  this->id = -1;
}

Vehicle::Vehicle(int id, double x, double y, double vx, double vy, double s, double d, int lane)
{

  this->id = id;
  this->x = x;
  this->y = y;
  this->vx = vx;
  this->vy = vy;
  this->v = sqrt(vx * vx + vy * vy);

  this->yaw = atan2(vy, vx);
  this->s = s;
  this->d = d;
  this->lane = lane;

}

/*
Vehicle::Vehicle(double x, double y, double s, double d, int lane, double angle_deg, double speed, string state, Vehicle * ego_prev)
{

  this->id = -1;
  this->x = x;
  this->y = y;
  this->yaw =  angle_deg * M_PI /180; //deg2rad(angle_deg);

  this->s = s;
  this->d = d;
  this->lane = lane;

  this->speed = speed;

  this->vx = cos(yaw) * v;
  this->vy = sin(yaw) * v;
  this->state = state;

}
*/

Vehicle::~Vehicle() {}


// TODO - Implement this method.





void Vehicle::update_car_state(double x, double y, double v, double s, double d, double yaw)
{
  this->x = x;
  this->y = y;
  this->v = v;
  this->s = s;
  this->d = d;
  this->yaw = yaw;

  //this->yaw =  yaw * M_PI /180;

  if (this->d < 4.0)
  {
    this->lane = 0;
  }
  else if ((this->d >= 4.0) && (this->d < 8.0))
  {
    this->lane = 1;
  }
  else
  {
    this->lane = 2;
  }


}

