#!/usr/bin/env python

import rospy
import styx_msgs.msg
import std_msgs.msg
from dbw_mkz_msgs.msg import ThrottleCmd, SteeringCmd, BrakeCmd, SteeringReport
import geometry_msgs.msg
import math

import twist_controller

'''
You can build this node only after you have built (or partially built) the `waypoint_updater` node.

You will subscribe to `/twist_cmd` message which provides the proposed linear and angular velocities.
You can subscribe to any other message that you find important or refer to the document for list
of messages subscribed to by the reference implementation of this node.

One thing to keep in mind while building this node and the `twist_controller` class is the status
of `dbw_enabled`. While in the simulator, its enabled all the time, in the real car, that will
not be the case. This may cause your PID controller to accumulate error because the car could
temporarily be driven by a human instead of your controller.

We have provided two launch files with this node. Vehicle specific values (like vehicle_mass,
wheel_base) etc should not be altered in these files.

We have also provided some reference implementations for PID controller and other utility classes.
You are free to use them or build your own.

Once you have the proposed throttle, brake, and steer values, publish it on the various publishers
that we have created in the `__init__` function.

'''

class DBWNode(object):
    def __init__(self):
        rospy.init_node('dbw_node')

        # get ROS parameters (default ones are for simulator)
        self.params = {}
        self.params['vehicle_mass'] = rospy.get_param('~vehicle_mass', 1080.)
        self.params['fuel_capacity'] = rospy.get_param('~fuel_capacity', 0.)
        self.params['brake_deadband'] = rospy.get_param('~brake_deadband', 0.2)
        self.params['decel_limit'] = rospy.get_param('~decel_limit', -5.)
        self.params['accel_limit'] = rospy.get_param('~accel_limit', 1.)
        self.params['wheel_radius'] = rospy.get_param('~wheel_radius', 0.335)
        self.params['wheel_base'] = rospy.get_param('~wheel_base', 3.)
        self.params['steer_ratio'] = rospy.get_param('~steer_ratio', 14.8)
        self.params['max_lat_accel'] = rospy.get_param('~max_lat_accel', 3.)
        self.params['max_steer_angle'] = rospy.get_param('~max_steer_angle', 8.)
        self.params['min_speed'] = 0 # required for yaw controller
        self.params['kp'] = 0.3 # required for pid controller
        self.params['kd'] = 0#4.0 # required for pid controller - 0
        self.params['ki'] = 0#0.003 # required for pid controller - 0.1 (maybe max accel=0.2)

        # internal
        self.controller = twist_controller.Controller(self.params)
        self.dbw_enabled = True
        self.flag_manual_break = False

        # setup publishers and subscribers
        rospy.Subscriber('/vehicle/dbw_enabled', std_msgs.msg.Bool, self.cb_dbw_enabled)
        rospy.Subscriber('/manual_break', std_msgs.msg.Bool, self.cb_manual_break)
        rospy.Subscriber('/twist_cmd', geometry_msgs.msg.TwistStamped, self.cb_twist_goal)
        rospy.Subscriber('/current_velocity', geometry_msgs.msg.TwistStamped, self.cb_velocity_curr)
        self.pub_steer = rospy.Publisher('/vehicle/steering_cmd', SteeringCmd, queue_size=1)
        self.pub_throttle = rospy.Publisher('/vehicle/throttle_cmd', ThrottleCmd, queue_size=1)
        self.pub_brake = rospy.Publisher('/vehicle/brake_cmd', BrakeCmd, queue_size=1)

        # run node
        self.loop()

    def loop(self):
        rate = rospy.Rate(50)  # 50Hz
        while not rospy.is_shutdown():
            if self.dbw_enabled:
                if self.controller.is_initialized():
                    throttle, brake, steering = self.controller.control()
                    self.publish(throttle, brake, steering)
            rate.sleep()

    def cb_manual_break(self, msg):
        # activate via  $rostopic pub /manual_break std_msgs/Bool True
        self.flag_manual_break = msg.data

    def cb_twist_goal(self, msg):
        vel_linear_goal = msg.twist.linear.x
        vel_angular_goal = msg.twist.angular.z
        if self.flag_manual_break:
            vel_linear_goal = -1.0
        self.controller.set_vel_goal(vel_linear_goal, vel_angular_goal)

    def cb_velocity_curr(self, msg):
        vel_linear_curr = msg.twist.linear.x
        self.controller.set_vel_curr(vel_linear_curr)

    def cb_dbw_enabled(self, msg):
        self.dbw_enabled = msg.data

    def publish(self, throttle, brake, steer):
        tcmd = ThrottleCmd()
        tcmd.enable = True
        tcmd.pedal_cmd_type = ThrottleCmd.CMD_PERCENT
        tcmd.pedal_cmd = throttle
        self.pub_throttle.publish(tcmd)

        scmd = SteeringCmd()
        scmd.enable = True
        scmd.steering_wheel_angle_cmd = steer
        self.pub_steer.publish(scmd)

        bcmd = BrakeCmd()
        bcmd.enable = True
        bcmd.pedal_cmd_type = BrakeCmd.CMD_TORQUE
        bcmd.pedal_cmd = brake
        self.pub_brake.publish(bcmd)


if __name__ == '__main__':
    DBWNode()
