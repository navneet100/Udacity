#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped,TwistStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32, Bool

import tf
import tf.transformations

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number
ONEMPH = 0.44704 # in mps


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)


        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb) # 10 hz
        rospy.Subscriber('/obstacle_waypoints', PoseStamped, self.obstacle_cb)
        rospy.Subscriber('/vehicle/dbw_enabled', Bool, self.dbw_enabled_cb, queue_size=1)
        #rospy.Subscriber("/current_velocity", TwistStamped, self.velocity_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.waypoints = None

        self.car_pose = None

	self.redlight_wp_index = None

        self.next_waypoint_index = None

        self.stop_waypoints = None

        self.dbw_enabled = True

        self.MAX_VEL = 8 #10#12
        self.stop_distance_tl_margin = 5.0 #7.0
        self.distance_to_stop = self.MAX_VEL * 5  #8 #10
        self.red_light_distance = 0.0


        self.loop()

        rospy.spin()

    def pose_cb(self, msg):
        # TODO: Implement
        self.car_pose = msg.pose

    def waypoints_cb(self, waypoints):
        # TODO: Implement
        #if (self.redlight_wp_index is None) or (self.redlight_wp_index == -1):
        self.waypoints = waypoints


    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        self.redlight_wp_index = msg.data


    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def dbw_enabled_cb(self, msg):
        self.dbw_enabled = msg.data

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoint, velocity):
        waypoint.twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    def is_ahead(self, current_car_pose, next_wp_pose):
        """
        Checks if the next wp position is ahead of current car position if test position is ahead of origin_pose
        Args:
            current_car_pose - geometry_msgs.msg.Pose instance
            next_wp_pose - geometry_msgs.msg.Point instance, or tuple (x,y)

        Returns:
            True if next_wp_pose is ahead of current_car_pose
        """
        current_x = current_car_pose.position.x
        current_y = current_car_pose.position.y

        next_x = next_wp_pose.pose.position.x
        next_y = next_wp_pose.pose.position.y

        _, _, current_car_yaw =  tf.transformations.euler_from_quaternion([current_car_pose.orientation.x, current_car_pose.orientation.y, current_car_pose.orientation.z, current_car_pose.orientation.w])

        isAhead = (next_x - current_x) * math.cos(current_car_yaw) + (next_y - current_y) * math.sin(current_car_yaw)

        return isAhead > 0

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        #TODO implement
        distances = []
        for wp in self.waypoints.waypoints:
            cur_x = pose.position.x
            cur_y = pose.position.y
            cur_z = pose.position.z

            wp_x = wp.pose.pose.position.x
            wp_y = wp.pose.pose.position.y
            wp_z = wp.pose.pose.position.z

            diff_x = cur_x - wp_x
            diff_y = cur_y - wp_y
            diff_z = cur_z - wp_z
            d = math.sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z)
            distances.append(d)

        min_index = distances.index(min(distances))

        return min_index

    def is_redlight_ahead(self):

        if self.redlight_wp_index is None or self.waypoints is None or self.redlight_wp_index <= 0:
            return False
	#12oct
	elif self.redlight_wp_index >= len(self.waypoints.waypoints):#3
            return True
        else:
            redlight_wp = self.waypoints.waypoints[self.redlight_wp_index]#4

            red_light_x = redlight_wp.pose.pose.position.x
            red_light_y = redlight_wp.pose.pose.position.y
            red_light_z = redlight_wp.pose.pose.position.z

            car_x = self.car_pose.position.x
            car_y = self.car_pose.position.y
            car_z = self.car_pose.position.z

            dx = red_light_x - car_x
            dy = red_light_y - car_y
            dz = red_light_z - car_z

            self.red_light_distance = math.sqrt(dx * dx + dy * dy + dz * dz)

            if self.is_ahead(self.car_pose, redlight_wp.pose) and self.red_light_distance <= (self.distance_to_stop + self.stop_distance_tl_margin):
                print("red light ahead")
                return True
            else:
                print("no red light ahead")
                return False


    def loop(self):
        """Publish to /final_waypoints with waypoints ahead of car
        """
        rate = rospy.Rate(2)#10
        while not rospy.is_shutdown():

            if (self.car_pose is not None) and (self.waypoints is not None):
                if self.is_redlight_ahead():
                    target_speed = 0
                else:
                    target_speed = self.MAX_VEL#10
                #print(target_speed)


                frame_id = self.waypoints.header.frame_id

                lane_start = self.get_closest_waypoint(self.car_pose)

                waypoints = self.waypoints.waypoints[lane_start:lane_start + LOOKAHEAD_WPS]

                for i, waypoint in enumerate(waypoints):
                    self.set_waypoint_velocity(waypoint, target_speed)

                lane = Lane()
                lane.header.frame_id = frame_id
                lane.header.stamp = rospy.Time.now()
                lane.waypoints = waypoints

                ## publish final_waypoints
                self.final_waypoints_pub.publish(lane)

                rate.sleep()

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
