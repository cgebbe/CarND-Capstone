from math import atan
import numpy as np

class YawController(object):
    def __init__(self,
                 wheel_base,
                 steer_ratio,
                 min_speed,
                 max_lat_accel,
                 max_steer_angle,
                 ):
        self.wheel_base = wheel_base
        self.steer_ratio = steer_ratio
        self.min_speed = min_speed
        self.max_lat_accel = max_lat_accel
        self.min_angle = -max_steer_angle
        self.max_angle = max_steer_angle

    def get_angle(self, radius):
        angle = atan(self.wheel_base / radius) * self.steer_ratio
        return max(self.min_angle, min(self.max_angle, angle))

    def get_steering(self, linear_velocity, angular_velocity, current_velocity):
        # "normalize" goal angular velocity by current velocity
        if abs(linear_velocity) > 0.:
            angular_velocity = angular_velocity  * current_velocity / linear_velocity
        else:
            angular_velocity = 0.

        # prevent too high values
        if abs(current_velocity) > 0.1:
            max_yaw_rate = abs(self.max_lat_accel / current_velocity);
            angular_velocity = np.clip(angular_velocity, -max_yaw_rate, +max_yaw_rate)

        # get angle
        if abs(angular_velocity) > 0.:
            angle = self.get_angle(max(current_velocity, self.min_speed) / angular_velocity)
        else:
            angle = 0
        return angle
