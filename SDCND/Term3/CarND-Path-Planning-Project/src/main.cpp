#include <fstream>
#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/QR"
#include "json.hpp"
#include "spline.h"


#include "constants.h"
#include "Planner.h"
#include "vehicle.h"
#include "road.h"
#include "Eigen-3.3/Eigen/Dense"
using Eigen::MatrixXd;
using Eigen::VectorXd;

using namespace std;
using namespace tk;
// for convenience
using json = nlohmann::json;
// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }



//vector<double> JMT(vector< double> start, vector <double> end, double T);
//double PolynomialEquate(vector<double> coefficients, double T);
// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.find_first_of("}");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

double distance(double x1, double y1, double x2, double y2)
{
	return sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1));
}

int ClosestWaypoint(double x, double y, vector<double> maps_x, vector<double> maps_y)
{
	double closestLen = 100000; //large number
	int closestWaypoint = 0;
	for(int i = 0; i < maps_x.size(); i++)
	{
		double map_x = maps_x[i];
		double map_y = maps_y[i];
		double dist = distance(x,y,map_x,map_y);

		if(dist < closestLen)
		{
			closestLen = dist;
			closestWaypoint = i;
		}
	}
	return closestWaypoint;
}

int NextWaypoint(double x, double y, double theta, vector<double> maps_x, vector<double> maps_y)
{
	int closestWaypoint = ClosestWaypoint(x,y,maps_x,maps_y);
	double map_x = maps_x[closestWaypoint];
	double map_y = maps_y[closestWaypoint];
	double heading = atan2( (map_y-y),(map_x-x) );
	double angle = abs(theta-heading);

	if(angle > pi()/4)
	{
		closestWaypoint++;
	}
	return closestWaypoint;
}

// Transform from Cartesian x,y coordinates to Frenet s,d coordinates
vector<double> getFrenet(double x, double y, double theta, vector<double> maps_x, vector<double> maps_y)
{
	int next_wp = NextWaypoint(x,y, theta, maps_x,maps_y);
	int prev_wp;
	prev_wp = next_wp-1;
	if(next_wp == 0)
	{
		prev_wp  = maps_x.size()-1;
	}

	double n_x = maps_x[next_wp]-maps_x[prev_wp];
	double n_y = maps_y[next_wp]-maps_y[prev_wp];
	double x_x = x - maps_x[prev_wp];
	double x_y = y - maps_y[prev_wp];
	// find the projection of x onto n
	double proj_norm = (x_x*n_x+x_y*n_y)/(n_x*n_x+n_y*n_y);
	double proj_x = proj_norm*n_x;
	double proj_y = proj_norm*n_y;
	double frenet_d = distance(x_x,x_y,proj_x,proj_y);
	//see if d value is positive or negative by comparing it to a center point

	double center_x = 1000-maps_x[prev_wp];
	double center_y = 2000-maps_y[prev_wp];
	double centerToPos = distance(center_x,center_y,x_x,x_y);
	double centerToRef = distance(center_x,center_y,proj_x,proj_y);

	if(centerToPos <= centerToRef)
	{
		frenet_d *= -1;
	}

 // calculate s value
	double frenet_s = 0;

	for(int i = 0; i < prev_wp; i++)
  {
		frenet_s += distance(maps_x[i],maps_y[i],maps_x[i+1],maps_y[i+1]);
	}

	frenet_s += distance(0,0,proj_x,proj_y);

	return {frenet_s,frenet_d};
}

// Transform from Frenet s,d coordinates to Cartesian x,y



int main() {
  uWS::Hub h;

  // Load up map values for waypoint's x,y,s and d normalized normal vectors
  vector<double> map_waypoints_x;
  vector<double> map_waypoints_y;
  vector<double> map_waypoints_s;
  vector<double> map_waypoints_dx;
  vector<double> map_waypoints_dy;

  // Waypoint map to read from
  string map_file_ = "../data/highway_map.csv";
  // The max s value before wrapping around the track back to 0


  ifstream in_map_(map_file_.c_str(), ifstream::in);

  string line;

  while (getline(in_map_, line)) {
  	istringstream iss(line);
  	double x;
  	double y;
  	float s;
  	float d_x;
  	float d_y;
  	iss >> x;
  	iss >> y;
  	iss >> s;
  	iss >> d_x;
  	iss >> d_y;

  	map_waypoints_x.push_back(x);
  	map_waypoints_y.push_back(y);
  	map_waypoints_s.push_back(s);
  	map_waypoints_dx.push_back(d_x);
  	map_waypoints_dy.push_back(d_y);
  }


  Road road;
  Vehicle  ego;
  Planner planner;

  planner.waypoints_spline_x.set_points(map_waypoints_s, map_waypoints_x);
  planner.waypoints_spline_y.set_points(map_waypoints_s, map_waypoints_y);
  planner.waypoints_spline_dx.set_points(map_waypoints_s, map_waypoints_dx);
  planner.waypoints_spline_dy.set_points(map_waypoints_s, map_waypoints_dy);

  string stateG = "CS"; //try taking it from prev_ego object

  h.onMessage([&road,&planner, &ego, &map_waypoints_x,&map_waypoints_y,&map_waypoints_s,&map_waypoints_dx,&map_waypoints_dy](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                     uWS::OpCode opCode) {

    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    //auto sdata = string(data).substr(0, length);
        ////cout << sdata << endl;

   //   int lane = 1;
  //double ref_vel = 0.0; //49.5;
    if (length && length > 2 && data[0] == '4' && data[1] == '2') {
      auto s = hasData(data);
      if (s != "")
	  {
        auto j = json::parse(s);
        string event = j[0].get<string>();
        if (event == "telemetry")
		{
          // j[1] is the data JSON object
        	// Main car's localization Data
          	double car_x = j[1]["x"];
          	double car_y = j[1]["y"];
          	double car_s = j[1]["s"];
          	double car_d = j[1]["d"];
          	double car_yaw = j[1]["yaw"];
          	double car_speed = j[1]["speed"];

            //car_speed *= 0.44704; //convert to m/s//0817

          	// Previous path data given to the Planner
          	auto previous_path_x = j[1]["previous_path_x"];
          	auto previous_path_y = j[1]["previous_path_y"];
          	// Previous path's end s and d values
          	double end_path_s = j[1]["end_path_s"];
            double end_path_d = j[1]["end_path_d"];
          	// Sensor Fusion Data, a list of all other cars on the same side of the road.
          	auto sensor_fusion = j[1]["sensor_fusion"];
          	json msgJson;


            vector<double> next_x_vals;
          	vector<double> next_y_vals;

          	// TODO: define a path made up of (x,y) points that the car will visit sequentially every .02 seconds


            int prev_path_size = previous_path_x.size();

            if (prev_path_size > 0)
            {
              car_s = end_path_s;
              car_d = end_path_d;
            }

            ego.update_car_state(car_x, car_y, car_speed, car_s, car_d, car_yaw);

      			vector<Vehicle> vehicles;

      			int num_other_vehicles = sensor_fusion.size();
      			for(int i = 0; i < num_other_vehicles;++i)
      			{
      			  auto vehSensData = sensor_fusion[i];

      			  long veh_id = vehSensData[0];
      			  double veh_x = vehSensData[1];
      			  double veh_y = vehSensData[2];
      			  double veh_vx = vehSensData[3];
      			  double veh_vy = vehSensData[4];
      			  double veh_s = vehSensData[5];
      			  double veh_d = vehSensData[6];
      			  double veh_vel = sqrt(pow(veh_vx,2) + pow(veh_vy,2));
              double veh_dist = distance(car_x,car_y,veh_x,veh_y);


      			  int veh_lane = 0;

      			  if(veh_d < 4.0)
      			  {
      				  veh_lane = 0;
      			  }
      			  else if(veh_d < 8.0)
      			  {
      				  veh_lane = 1;
      			  }
      			  else
      			  {
      				  veh_lane = 2;
      			  }

      				Vehicle vehicle(veh_id,veh_x,veh_y,veh_vx,veh_vy,veh_s,veh_d,veh_lane);
      				vehicles.push_back(vehicle);
      			}


            road.vehicles = vehicles;


            for(int i = 0; i < prev_path_size; i++)
            {
              next_x_vals.push_back(previous_path_x[i]);
              next_y_vals.push_back(previous_path_y[i]);
            }

            vector<vector<double>> trajectory = {next_x_vals, next_y_vals};

            planner.create_trajectory(road, ego, trajectory);

            next_x_vals = trajectory[0];
            next_y_vals = trajectory[1];

          	msgJson["next_x"] = next_x_vals;
          	msgJson["next_y"] = next_y_vals;

          	auto msg = "42[\"control\","+ msgJson.dump()+"]";

          	//this_thread::sleep_for(chrono::milliseconds(1000));

          	ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }
		}
		else
		{
			// Manual driving
			std::string msg = "42[\"manual\",{}]";
			ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
		}
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the
  // program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                     size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";

    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());

    } else {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }

  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;

  });


  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });
  int port = 4567;

  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;

  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }

  h.run();
}
