#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
import geometry_msgs.msg
from styx_msgs.msg import Lane, Waypoint
import std_msgs.msg

import math
import scipy.spatial
import numpy as np

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

class WaypointUpdater(object):
    def __init__(self):

        # ros specific stuff
        rospy.init_node('waypoint_updater')
        rospy.Subscriber('/current_pose', PoseStamped, self.cb_pose)
        rospy.Subscriber('/base_waypoints', Lane, self.cb_waypoints)
        rospy.Subscriber('/traffic_waypoint', std_msgs.msg.Int32, self.cb_traffic)
        rospy.Subscriber('/current_velocity', geometry_msgs.msg.TwistStamped, self.cb_velocity_curr)
        self.pub_final_waypoints = rospy.Publisher('/final_waypoints', Lane, queue_size=1)
        self.pub_idx_closest_waypoint = rospy.Publisher('/idx_closest_waypoint', std_msgs.msg.Int32, queue_size=1)

        # other member variables
        self.pose = None
        self.msg_waypoints = None
        self.waypoints_2d = None
        self.waypoints_tree = None
        self.velocity_current = None
        self.velocity_in_meter_per_s = 20.0
        self.idx_wp_to_stop = -1

        # start looping
        self.loop()

    def loop(self):
        rate = rospy.Rate(10) # 10 Hz should suffice ?!
        while not rospy.is_shutdown():
            if self.pose and self.waypoints_tree:
                self.publish_waypoints()
            rate.sleep()

    def cb_velocity_curr(self, msg):
        self.velocity_current = msg.twist.linear.x

    def cb_pose(self, msg_pose):
        self.pose = msg_pose # gets called every ~20ms

    def cb_waypoints(self, msg_waypoints):
        if self.msg_waypoints is None: # only do it once!
            self.msg_waypoints = msg_waypoints
            self.waypoints_2d = [[wp.pose.pose.position.x, wp.pose.pose.position.y] for wp in msg_waypoints.waypoints]
            self.waypoints_tree = scipy.spatial.KDTree(self.waypoints_2d)

    def cb_traffic(self, msg):
        self.idx_wp_to_stop = msg.data - 5 # already stop a bit earlier...

    def cb_obstacle(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass


    def publish_waypoints(self):
        # create message with index of closest waypoint
        msg_idx = std_msgs.msg.Int32()
        idx_wp_closest = self.get_closest_waypoint_idx()
        msg_idx.data = idx_wp_closest
        self.pub_idx_closest_waypoint.publish(msg_idx)

        # create message with waypoints
        msg_lane = Lane()
        msg_lane.header = self.msg_waypoints.header
        msg_lane.waypoints = self.msg_waypoints.waypoints[idx_wp_closest: idx_wp_closest + LOOKAHEAD_WPS]
        msg_lane = self.set_velocities(msg_lane, idx_wp_start=idx_wp_closest)
        self.pub_final_waypoints.publish(msg_lane)

    def set_velocities(self, msg_lane, idx_wp_start):
        rospy.loginfo("idx_wp_to_stop={}, idx_wp_next={}".format(self.idx_wp_to_stop, idx_wp_start))
        #is_break_impossible = (self.calc_distance(idx_wp_start, idx_wp_start + idx_list_to_stop) < 5
        #                       and self.velocity_current > 0.9 * self.velocity_in_meter_per_s)
        if self.idx_wp_to_stop < 0: # negative value = no need to stop anywhere
            for wp in msg_lane.waypoints:
                wp.twist.twist.linear.x = self.velocity_in_meter_per_s
        else:
            idx_list_to_stop = self.idx_wp_to_stop - idx_wp_start
            dist = 0
            ACC_MIN = -3
            for idx in np.arange(LOOKAHEAD_WPS-1, -1, -1):
                if idx >= idx_list_to_stop:
                    msg_lane.waypoints[idx].twist.twist.linear.x = 0
                else:
                    dist += self.calc_distance(idx_wp_start + idx, idx_wp_start + idx+1)
                    vel = np.sqrt(2 * abs(ACC_MIN) * dist)
                    vel = np.clip(vel, 0, self.velocity_in_meter_per_s)
                    msg_lane.waypoints[idx].twist.twist.linear.x = vel
        return msg_lane

    def calc_distance(self, idx_wp1, idx_wp2):
        wp1 = np.asarray(self.waypoints_2d[idx_wp1])
        wp2 = np.asarray(self.waypoints_2d[idx_wp2])
        dist = wp1 - wp2
        dist_norm = np.linalg.norm(dist)
        return dist_norm


    def get_closest_waypoint_idx(self):
        car_x = self.pose.pose.position.x
        car_y = self.pose.pose.position.y
        distance, idx_closest = self.waypoints_tree.query([car_x, car_y], 1)

        # check if closest is ahead or behind vehicle
        xy_closest = self.waypoints_2d[idx_closest]
        xy_closest_prev = self.waypoints_2d[idx_closest-1]
        direction_front = np.array(xy_closest) - np.array(xy_closest_prev)
        direction_waypoint = np.array(xy_closest) - np.array([car_x,car_y])
        if np.dot(direction_front, direction_waypoint) > 0:
            return idx_closest
        else:
            # if closest waypoint is behind car, next one will certainly be in front (due to ordering of wp's in list)!
            idx_closest = (idx_closest+1) % len(self.waypoints_2d)
            return idx_closest





if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
