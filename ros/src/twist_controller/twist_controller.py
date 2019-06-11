import yaw_controller
import pid
import numpy as np

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, params):
        self.yaw_controller = yaw_controller.YawController(params['wheel_base'],
                                                           params['steer_ratio'],
                                                           params['min_speed'],
                                                           params['max_lat_accel'],
                                                           params['max_steer_angle'],
                                                           )

        self.pid = pid.PID(params['kp'], params['ki'], params['kd'])
        self.params = params
        self.vel_linear_goal = None
        self.vel_angular_goal = None
        self.vel_linear_curr = None

    def set_vel_goal(self, vel_linear_goal, vel_angular_goal):
        self.vel_angular_goal = vel_angular_goal
        self.vel_linear_goal = vel_linear_goal

    def set_vel_curr(self, vel_linear_curr):
        self.vel_linear_curr = vel_linear_curr

    def is_initialized(self):
        if self.vel_linear_goal is None: return False
        if self.vel_angular_goal is None: return False
        if self.vel_linear_curr is None: return False
        return True

    def control(self):
        # get steering angle
        angle = self.yaw_controller.get_steering(self.vel_linear_goal,
                                                 self.vel_angular_goal,
                                                 self.vel_linear_curr,
                                                 )

        # set acceleration and break values
        error = self.vel_linear_goal - self.vel_linear_curr
        acceleration = self.pid.step(error)
        acceleration = np.clip(acceleration, self.params['decel_limit'], self.params['accel_limit'])
        print("=== acceleration={}".format(acceleration))
        if acceleration > 0:
            throttle = acceleration
            brake = 0.0
        else:
            throttle = 0.
            brake = acceleration * self.params['vehicle_mass'] / self.params['wheel_base']

        return throttle, brake, angle
