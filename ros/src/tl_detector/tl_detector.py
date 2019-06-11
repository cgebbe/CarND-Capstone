#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
import std_msgs.msg
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import scipy.spatial
import numpy as np

STATE_COUNT_THRESHOLD = 3

'''
/vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
helps you acquire an accurate ground truth data source for the traffic light
classifier by sending the current color state of all traffic lights in the
simulator. When testing on the vehicle, the color state will not be available. You'll need to
rely on the position of the light and the camera image to predict it.
'''

class Light:
    def __init__(self):
        self.idx = None
        self.x = None
        self.y = None
        self.idx_wp = None # index of corresponding waypoint

class TLDetector(object):
    def __init__(self):

        # ============== ROS specific stuff
        rospy.init_node('tl_detector')
        rospy.Subscriber('/current_pose', PoseStamped, self.cb_pose)
        rospy.Subscriber('/base_waypoints', Lane, self.cb_waypoints)
        rospy.Subscriber('/idx_closest_waypoint', std_msgs.msg.Int32, self.cb_waypoint_next)
        rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.cb_traffic_lights)
        rospy.Subscriber('/image_color', Image, self.cb_image)
        self.pub_idx_wp_to_stop = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        # ========== own stuff
        # waypoints
        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)
        self.msg_waypoints = None
        self.waypoints_2d = None
        self.waypoints_tree = None
        self.lights = None
        self.idx_light_next = 0

        # traffic lights
        self.pose = None
        self.camera_image = None
        self.msg_lights = []
        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        # start looping
        rospy.spin()

    def cb_pose(self, msg):
        self.pose = msg

    def cb_waypoints(self, msg_waypoints):
        if self.msg_waypoints is None:  # only do it once!
            # collect waypoints
            self.msg_waypoints = msg_waypoints
            self.waypoints_2d = [[wp.pose.pose.position.x, wp.pose.pose.position.y] for wp in msg_waypoints.waypoints]
            self.waypoints_tree = scipy.spatial.KDTree(self.waypoints_2d)

            # collect information about all traffic lights
            self.lights = []
            lights_pos_xy = self.config['stop_line_positions']
            for idx_light, pos_xy in enumerate(lights_pos_xy):
                light = Light()
                light.x = pos_xy[0]
                light.y = pos_xy[1]
                light.idx_wp = self.waypoints_tree.query(pos_xy, 1)[1]
                self.lights.append(light)

    def cb_waypoint_next(self, msg):
        idx_wp_next = msg.data
        # check whether next light is still ahead of current waypoint. Otherwise, get next light
        if self.lights:
            light_next = self.lights[self.idx_light_next]
            if idx_wp_next > light_next.idx_wp:
                self.idx_light_next = (self.idx_light_next + 1) % len(self.lights)

    def cb_traffic_lights(self, msg):
        self.msg_lights = msg.lights
        self.check_lights()

    def cb_image(self, msg):
        pass

    def check_lights(self):
        msg_out = Int32()
        msg_out.data = -1  # default value, meaning no need to stop at any waypoint
        if self.is_next_light_close_enough():
            light_gt = self.msg_lights[self.idx_light_next]
            if light_gt.state == light_gt.RED or light_gt.state == light_gt.YELLOW:
                rospy.loginfo("Light idx={} is red or yellow".format(self.idx_light_next))
                msg_out.data = self.lights[self.idx_light_next].idx_wp
            elif light_gt.state == light_gt.GREEN or light_gt.state == light_gt.UNKNOWN:
                rospy.loginfo("Light idx={} is green or unknown".format(self.idx_light_next))
            else:
                raise ValueError("unkown light state")
        #msg_out.data = 500 # DEBUG MODE
        self.pub_idx_wp_to_stop.publish(msg_out)


    def is_next_light_close_enough(self, d_min = 0, d_max=200):
        if self.pose and self.lights and self.idx_light_next is not None:
            car_xy = np.asarray([self.pose.pose.position.x, self.pose.pose.position.y])
            light_next = self.lights[self.idx_light_next]
            light_xy = np.asarray([light_next.x, light_next.y])
            distance = self.calc_distance(car_xy, light_xy)
            if d_min < distance and distance < d_max:
                return True
        return False

    @classmethod
    def argmax_with_min_val(cls, array, val_min):
        idx_sorted = np.argsort(array)
        for idx in idx_sorted:
            val = array[idx]
            if val >= val_min:
                return idx
        return -1 #if no index can be found!

    @classmethod
    def calc_distance(self, pt1, pt2):
        dist = pt1 - pt2
        dist_norm = np.linalg.norm(dist)
        return dist_norm

    ####################################################################

    def cb_image_org(self, msg):
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
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.pub_idx_wp_to_stop.publish(Int32(light_wp))
        else:
            self.pub_idx_wp_to_stop.publish(Int32(self.last_wp))
        self.state_count += 1

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
        if (self.pose):
            car_position = self.get_closest_waypoint(self.pose.pose)

        # TODO find the closest visible traffic light (if one exists)

        if light:
            state = self.get_light_state(light)
            light_wp = 999
            return light_wp, state
        self.msg_waypoints = None
        return -1, TrafficLight.UNKNOWN

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if (not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        # Get classification
        return self.light_classifier.get_classification(cv_image)


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
