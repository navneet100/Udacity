from styx_msgs.msg import Lane, Waypoint
from geometry_msgs.msg import PoseStamped, Pose


from lowpass import LowPassFilter
from pid import PID
from yaw_controller import YawController
import math
import rospy


GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, *args, **kwargs):
        # TODO: Implement

        self.previous_time = None

        self.brake_deadband  = kwargs['brake_deadband']

        self.decel_limit 	= kwargs['decel_limit']
        accel_limit     	= kwargs['accel_limit']

        wheel_base 		= kwargs['wheel_base']
        steer_ratio 	= kwargs['steer_ratio']
        max_lat_accel 	= kwargs['max_lat_accel']
        max_steer_angle = kwargs['max_steer_angle']

        vehicle_mass    = kwargs['vehicle_mass']
        fuel_capacity   = kwargs['fuel_capacity']
        wheel_radius 	= kwargs['wheel_radius']

        self.min_speed = kwargs['min_speed']

        self.brake_tourque_const = (vehicle_mass + fuel_capacity * GAS_DENSITY) * wheel_radius

        yaw_params = [wheel_base, steer_ratio, self.min_speed, max_lat_accel, max_steer_angle]

        self.yaw_controller = YawController(*yaw_params)

        self.throttle_pid = PID(kp=0.5, ki=0.05, kd=0.07, mn=self.decel_limit, mx= accel_limit )
        self.brake_pid = PID(kp=150.0, ki=0.01, kd=5.0, mn=self.brake_deadband, mx=500)
        self.steering_pid = PID(kp=1, ki=0.05, kd=0.1, mn=-0.2,mx=0.2)


        self.tau_correction = 0.2
        self.ts_correction = 0.1
        self.low_pass_filter_correction = LowPassFilter(self.tau_correction, self.ts_correction)


    def control(self, current_pose, target_velocity, current_velocity, final_waypoints, dbw_enabled , sample_step):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer

        ### throttle

        linear_velocity_error = target_velocity.linear.x - current_velocity.linear.x
        throttle = self.throttle_pid.step(linear_velocity_error, sample_step)
        throttle = self.low_pass_filter_correction.filt(throttle)
        if abs(target_velocity.linear.x) < 0.01 and abs(current_velocity.linear.x) < 0.3:
            throttle = self.decel_limit

        ### steering

	target_linear_velocity = target_velocity.linear.x
        target_angular_velocity = target_velocity.angular.z
        current_linear_velocity = current_velocity.linear.x


        next_steer = self.yaw_controller.get_steering(target_linear_velocity,target_angular_velocity,current_linear_velocity)

        angular_velocity_cte = target_angular_velocity - current_velocity.angular.z
        fine_tune_steer = self.steering_pid.step(angular_velocity_cte, sample_step)

        steering = next_steer + fine_tune_steer

        ### brake
        brake = 0.0

        if throttle > 0.0:
            brake = 0.0
            self.brake_pid.reset()
        else:
            if math.fabs(throttle) < self.brake_deadband:
                brake = 0.0
            else:
                decel = abs(throttle)
                brake = self.brake_tourque_const * decel

            throttle = 0.0
            self.throttle_pid.reset()

        return throttle, brake, steering
