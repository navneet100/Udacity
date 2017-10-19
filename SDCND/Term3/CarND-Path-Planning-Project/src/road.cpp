#include <iostream>
#include "road.h"
#include "vehicle.h"
#include <iostream>
#include <math.h>
#include <map>
#include <string>
#include <iterator>


bool Road::is_lane_safe(Vehicle& car, int lane )
{

  bool safe = true;

  for (int i = 0; i < this->vehicles.size(); i++)
  {
    if(this->vehicles[i].lane == lane)
    {
      double distance = this->vehicles[i].s - car.s;
      if(distance > 0 && distance < FRONT_BUFFER)
      {
        safe = false;
      }
    }


  }

  return safe;
}


bool Road::is_lane_available(Vehicle& car, int lane )
{
  bool safe = true;

  for (int i = 0; i < this->vehicles.size(); i++)
  {
    if(this->vehicles[i].lane == lane)
    {
      double distance = std::abs((this->vehicles[i].s - car.s));
      if(distance < FRONT_BACK_BUFFER)
      {
        safe = false;
      }
    }
  }
  return safe;
}



int Road::get_safe_lane(Vehicle& car)
{
  int car_lane = car.lane;
  int target_lane = car_lane;

  if (car_lane == 0 || car_lane == 2)
  {
    if (this->is_lane_available(car, 1))
    {
      target_lane = 1;
    }
  }
  else
  {
    if (this->is_lane_available(car, 0))
    {
      target_lane = 0;
    }
    else if (this->is_lane_available(car, 2))
    {
      target_lane = 2;
    }
  }
  return target_lane;
}
