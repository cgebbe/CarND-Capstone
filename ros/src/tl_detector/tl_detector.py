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
import cv2
import yaml
import scipy.spatial
import numpy as np
import time
import os
import matplotlib.pyplot as plt

'''
/vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
helps you acquire an accurate ground truth data source for the traffic light
classifier by sending the current color state of all traffic lights in the
simulator. When testing on the vehicle, the color state will not be available. You'll need to
rely on the position of the light and the camera image to predict it.
'''

# Enum manually copied from /vehicle/traffic_lights message
COLOR_PER_INT = {0: "red",
                 1: "yellow",
                 2: "green",
                 3: "unknown",
                 }


class Light:
    def __init__(self):
        self.idx = None
        self.x = None
        self.y = None
        self.idx_wp = None  # index of corresponding waypoint
        self.state_true = None
        self.state_pred = None


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

        # ========== other stuff
        # waypoints and egopose
        self.pose = None
        self.waypoints_2d = None
        self.waypoints_tree = None

        # traffic lights / stop lines
        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)
        self.lights = None
        self.idx_light_next = 0

        # image classification
        self.cv_bridge = CvBridge()
        self.img_cnt = 0
        self.img_t_last = time.time()
        self.classifier = TLClassifier(filename_pb="model.pb")
        self.img_queue = []

        # start looping
        rospy.spin()

    def cb_pose(self, msg):
        self.pose = msg

    def cb_waypoints(self, msg_waypoints):
        if self.waypoints_2d is None:  # only do it once!
            # collect waypoints
            self.waypoints_2d = [[wp.pose.pose.position.x, wp.pose.pose.position.y] for wp in msg_waypoints.waypoints]
            self.waypoints_tree = scipy.spatial.KDTree(self.waypoints_2d)

            # collect information about all traffic lights from config and waypoints
            self.lights = []
            lights_pos_xy = self.config['stop_line_positions']
            for idx_light, pos_xy in enumerate(lights_pos_xy):
                light = Light()
                light.idx = idx_light
                light.x = pos_xy[0]
                light.y = pos_xy[1]
                light.idx_wp = self.waypoints_tree.query(pos_xy, 1)[1]
                self.lights.append(light)

    def cb_waypoint_next(self, msg):
        idx_wp_next = msg.data
        # check whether next light is still ahead of current waypoint. Otherwise, get next light
        if self.lights is not None:
            light_next = self.lights[self.idx_light_next]
            if idx_wp_next > light_next.idx_wp:
                self.idx_light_next = (self.idx_light_next + 1) % len(self.lights)

    def cb_traffic_lights(self, msg):
        if self.lights is not None:
            # copy information from msg into class object
            for idx, light_gt in enumerate(msg.lights):
                self.lights[idx].state_true = light_gt.state

    def cb_image(self,
                 msg_img,
                 num_images_skip=3,
                 dt_min_between_images = 0.050,
                 flag_export=False,
                 ):
        # only do stuff with image if next traffic light is close. Also don't process every image
        distance = self.get_distance_to_next_light()
        if distance is not None and distance < 100:
            self.img_cnt += 1
            time_now = time.time()
            if self.img_cnt % num_images_skip == 0 and time_now - self.img_t_last > dt_min_between_images:
                self.img_t_last = time_now

                # start of actual processing
                img_numpy = self.cv_bridge.imgmsg_to_cv2(msg_img, "bgr8")
                if flag_export:
                    self.export_image(img_numpy, distance)
                else:
                    state = self.predict_state(img_numpy)
                    if state is not None:
                        next_light = self.lights[self.idx_light_next]
                        next_light.state_pred = state
                        self.check_lights()


    def predict_state(self,
                      img,
                      batch_size=3,
                      target_size=(320, 240),
                      ratio_min_detection=0.6,
                      ):
        # resize image, scale values, convert to RGB and  and append to queue
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        img = img / 255.
        img = img[:,:,::-1] # converts BGR to RGB
        self.img_queue.append(img)

        # only if queue has batch_size items in it, run prediction
        if len(self.img_queue) == batch_size:
            input_tensor = np.asarray(self.img_queue)
            assert(input_tensor.ndim == 4)
            self.img_queue = []

            prob_all = self.classifier.predict(input_tensor)
            prob_red = prob_all[:,0]
            if np.mean(prob_red > 0.5) > ratio_min_detection:
                return 0 # meaning traffic light = red
            elif np.mean(prob_red < 0.5) > ratio_min_detection:
                return 2 # meaning traffic light = green
            else:
                return None
        else:
            return None


    def export_image(self, img_numpy, distance):
        light_state = self.lights[self.idx_light_next].state_true  # state as int 0,1,2,3, see above
        time_in_ms = np.round(time.time() * 1000).astype(np.int)
        folder = os.path.join('/mnt/share/export', str(light_state))
        filename = ('img_' + str(time_in_ms)
                    # + '_state_' + str(light_state)
                    + '_lightidx_' + str(self.idx_light_next)
                    + '_dist_' + str(np.round(distance).astype(np.int))
                    + '.png'
                    )
        filepath = os.path.join(folder, filename)
        cv2.imwrite(filepath, img_numpy)
        rospy.loginfo("Written image {}".format(filename))

    def check_lights(self):
        msg_out = Int32()
        msg_out.data = -1  # default value, meaning no need to stop at any waypoint
        distance = self.get_distance_to_next_light()
        if distance is not None:
            light_next = self.lights[self.idx_light_next]
            if light_next.state_pred == 0:  # 0=red, see above
                msg_out.data = light_next.idx_wp
            if True:
                rospy.loginfo("Light idx={}: state_pred={}, state_true={}, distance={}".format(
                    self.idx_light_next,
                    COLOR_PER_INT[light_next.state_pred],
                    COLOR_PER_INT[light_next.state_true],
                    distance,
                ))
        self.pub_idx_wp_to_stop.publish(msg_out)

    def get_distance_to_next_light(self):
        if self.pose and self.lights and self.idx_light_next is not None:
            car_xy = np.asarray([self.pose.pose.position.x, self.pose.pose.position.y])
            light_next = self.lights[self.idx_light_next]
            light_next_xy = np.asarray([light_next.x, light_next.y])
            distance = self.calc_distance(car_xy, light_next_xy)
            return distance
        else:
            return None

    @classmethod
    def calc_distance(self, pt1, pt2):
        dist = pt1 - pt2
        dist_norm = np.linalg.norm(dist)
        return dist_norm



if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
