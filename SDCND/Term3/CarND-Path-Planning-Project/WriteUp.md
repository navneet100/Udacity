# Path Planning Project

### Goal 
In this project, goal is to design a path planner that is able to create smooth, safe paths for the car to follow along a 3 lane highway with traffic. A successful path planner will be able to keep inside its lane, avoid hitting other cars, and pass slower moving traffic all by using localization, sensor fusion, and map data.


## Implementation

### Interpolate map waypoints
The waypoints given in highway_map.csv are about 30m apart. This big gap makes it difficult to plot the actual curve. To fix this, interpolation of waypoints mapping is done
using spline curve. Spline curve mapping is done bu passing s values and corresponding x, y, dx and dy values to 4 separate functions in pairs. These splines will then be used later to 
generate next_x_vals and next_y_vals.


### Vehicle Class
Vehicle object is defined as the Vehicle class in Vehicle.h file. It has properties like x, y, s, d, v, speed values. It also holds the lane in which car is travelling. It also
has its state value - CS - car Start, KL - Keep Lane, LCL - Lane change left, LCR - Lane Change Right.

### Road Class
This class basically keeps collection of all cars and it checks whether a particular lane is safe to travel.
Road has basically three lanes. Left-most lane is labeled as 0, middle as 1 and right-most lane as 2.
Lane 0 has d-value ranges from 0 to 4( not inclusive), lane 1 has - 4(inclusive) to 8(not inclusive) and lane 2 has 8(inclusive) to 12(not inclusive).

### Planner Class
This class is the main route planner. It has functions to check in which lane car can travel and at what speed. It utilizies road class functions to check lane safety.

### Constants
The fixed values used in the application for driving are defined in constants.h file. It has constants like TRAJ_POINTS, TRACK_LENGTH, SPEED_LIMIT, TIME_STEP, FRONT_BUFFER, 
FRONT_BACK_BUFFER.

### Create Ego car and other cars from Simulator Data
Ego ( the main car) is created in the beginning of the program from the simulator data. Then the other cars are also created from sensor_fusion data. These other cars the added 
to the vehicles collection of the Road object.


### Main Driving functionailty
Car starts in the center lane (lane = 1), as the car proceeds, Planner checks Vehicles collection for the cars in the current lane( lane - 1) and see if there is any car with 
s value less than the front-buffer. If there is no car in front safety zone, then car follows the keep_lane strategy and stays in the lane. If there is a car in the safety zone(front buffer area), then planner checks if the lanes available ( for lane 0, available lane is 1, for lane 1, car can move to left ( lane 0 ) or right( lane 2) , and for lane 2, possible available lane is 1.
If there is no other safe lane available, then the planner will try to reduce the speed of car. While checking for lane change, it checks if there is safety zone in front and in back.

After finding the next strategy, Keep_Lane, Change_Lane or Reduce_Speed, the next s and d values are calculated. d values are dependent upon the lanes. After finding target s and d values, the
spline way_points mappings are used to generate next x and y points. These points are then passed to the simulator to set nex x and y positions of the car.