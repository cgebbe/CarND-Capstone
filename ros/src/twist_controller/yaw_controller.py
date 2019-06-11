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

    def get_steering(self, vel_linear_goal, vel_angular_goal, vel_linear_curr):
        # "normalize" goal angular velocity by current velocity
        max_yaw_rate = abs(self.max_lat_accel / vel_linear_curr);
        if abs(vel_linear_goal) < 0.01:
            return 0

        #vel_angular_goal = vel_angular_goal * vel_linear_curr / vel_linear_goal
        vel_angular_goal = np.clip(vel_angular_goal, -max_yaw_rate, +max_yaw_rate)
        angle = self.get_angle(max(vel_linear_curr, self.min_speed) / vel_angular_goal)
        return angle
