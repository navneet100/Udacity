#ifndef ROAD_H
#define ROAD_H

#include <iostream>
#include <random>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>
#include <set>
#include <map>
#include <string>
#include <iterator>
#include "vehicle.h"

#include "constants.h"

using namespace std;

class Road {
public:

	int update_width = 70;
  	string ego_rep = " *** ";
  	int ego_key = -1;
  	int num_lanes;
    vector<int> lane_speeds;

    double density;
    int camera_center;
    //map<int, Vehicle> vehicles;
    int vehicles_added = 0;

		Road() {};
    ~Road() {};


		bool is_lane_safe(Vehicle& car, int lane );
		bool is_lane_available(Vehicle& car, int lane );
		int get_safe_lane(Vehicle& car );

		vector<Vehicle> vehicles;


};
#endif
