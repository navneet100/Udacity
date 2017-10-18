#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import math

from scipy.misc import imresize

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        self.sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, lane):
        self.waypoints = lane.waypoints
        self.sub2.unregister()

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD and self.state_count <= 30:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            #if state == TrafficLight.GREEN and light_wp is not None:
            #    light_wp = -light_wp
            #elif state == TrafficLight.UNKNOWN:
            #    light_wp = -1

            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1
	#print(self.state,self.state_count )
	#self.has_image = False


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

        next_x = next_wp_pose.position.x
        next_y = next_wp_pose.position.y

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
        for wp in self.waypoints:
            cur_x = pose.position.x
            cur_y = pose.position.y
            cur_z = pose.position.z

            wp_x = wp.pose.pose.position.x
            wp_y = wp.pose.pose.position.y
            wp_z = wp.pose.pose.position.z

            diff_x = cur_x - wp_x
            diff_y = cur_y - wp_y
            diff_z = cur_z - wp_z
            distance = math.sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z)
            distances.append(distance)

        min_index = distances.index(min(distances))

        return min_index

    def project_to_image_plane(self, point_in_world):
        """Project point from 3D world coordinates to 2D camera image location

        Args:
            point_in_world (Point): 3D location of a point in the world

        Returns:
            x (int): x coordinate of target point in image
            y (int): y coordinate of target point in image

        """

        #fx = self.config['camera_info']['focal_length_x']
        #fy = self.config['camera_info']['focal_length_y']
        #image_width = self.config['camera_info']['image_width']
        #image_height = self.config['camera_info']['image_height']

        # get transform between pose of camera and world frame
        trans = None

	try:
            now = rospy.Time.now()
            self.listener.waitForTransform("/base_link",
                  "/world", now, rospy.Duration(1.0))
            (trans, rot) = self.listener.lookupTransform("/base_link",
                  "/world", now)

        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            rospy.logerr("Failed to find camera to map transform")

        #TODO Use tranform and rotation to calculate 2D position of light in image

        x = 0
        y = 0

        return (x, y)


    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
	#print("has_image")
	#print(self.has_image)
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        x, y = self.project_to_image_plane(light.pose.pose.position)

        #TODO use light location to zoom in on traffic light in image

        #Get classification
	cv_image = imresize(cv_image, (224, 224, 3))
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose and self.pose.pose):
            car_position = self.get_closest_waypoint(self.pose.pose)
	else:
            return -1, TrafficLight.UNKNOWN #10Oct

        #TODO find the closest visible traffic light (if one exists)

        light_in_front = None
        light_in_front_dist = float('inf')
	light_closest = None

        if self.pose.pose:
            car_x =  self.pose.pose.position.x
            car_y =  self.pose.pose.position.y
            car_z =  self.pose.pose.position.z

            for i, light in enumerate(self.lights):

                light_x = light.pose.pose.position.x
                light_y = light.pose.pose.position.y
                light_z = light.pose.pose.position.z

                dx = light_x - car_x
                dy = light_y - car_y
                dz = light_z - car_z

                distance = math.sqrt(dx * dx + dy * dy + dz * dz)
		#print("i=",i, " distance = ", distance)

                if distance <= 50: #60
                    light_state = self.get_light_state(light)
                    #if light_state == TrafficLight.RED:
                    if light_state != 99:
                        light_waypoint_index = self.get_closest_waypoint(light.pose.pose)
                        light_waypoint = self.waypoints[light_waypoint_index]
                        if self.is_ahead(self.pose.pose, light_waypoint.pose.pose):
                            light_waypoint_x = light_waypoint.pose.pose.position.x
                            light_waypoint_y = light_waypoint.pose.pose.position.y
                            light_waypoint_z = light_waypoint.pose.pose.position.z

                            dx = light_waypoint_x - car_x
                            dy = light_waypoint_y - car_y
                            dz = light_waypoint_z - car_z

                            distance = math.sqrt(dx * dx + dy * dy + dz * dz)
                            #print("light_in_front_dist=",light_in_front_dist)
                            if distance < light_in_front_dist:
                                light_in_front = light_waypoint_index
                                light_in_front_dist = distance
				light_closest = light
                        else:
				#print("light is behind");
				light_in_front = None
				light_in_front_dist = float('inf')
				light_closest = None
	#print("light_in_front = ",light_in_front)
	#print("light_in_front_dist = ",light_in_front_dist)
	#print("light_in_front is not None = ",light_in_front is not None)

        if light_in_front is not None:
            #print("light_closest = ", light_closest)
            if not math.isinf(light_in_front_dist):
		    #print("calling get_light_state")
		    state = self.get_light_state(light_closest)
		    #print("light_in_front =",light_in_front,"state=", state)
		    #return light_in_front, TrafficLight.RED
		    return light_in_front, state
            else:
		    return -1, TrafficLight.UNKNOWN
        else:
            return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
