
# Notes for capstone project

* See git repo describing task and setup: https://github.com/udacity/CarND-Capstone
* Individual submission from me (, i.e. not team submission)


## Questions on project
- do I need to navigate both tracks in simulator?
	-> no, only the first one (cause second one has nothing on it!)
- do I need to work as a team ?
	-> no, only simulation is enough (from video "have fun")


# Setting up ROS and VM

## Useful ROS commands

- $rosnode list -- lists current nodes
- $rostopic list -- lists current topics
- $rostopic info /turtle1/cmd_vel --  information about topic (msg type,pubs,subs,)
- $rostopic echo /turtle1/cmd_vel -- prints all messages sent via topic
- $rosmsg info /geometry_msgs/Twist -- information about message

- run turtlesim
- turtelsim commands: list nodes, topics, echo topic, ...
- catkin
- roslaunch
- rosdep
- gazebo
- writing ROS nodes - publisher and subscriber (in python !)
- rospy.logwarn / loginfo !!!


## Which ROS version do I need?
Project requires: 
- rosdistro: kinetic
- rosversion: 1.12.6

I have: Linux mint 19 Tara
= Ubuntu Bionic 18.04
= ROS melodic

"ROS kinetic only supports Ubuntu 16.04", see http://answers.ros.org/question/297008/how-could-i-install-ros-on-ubuntu-18/
Alternatively, ROS melodic for Ubuntu 18.04

-> Better use ROS in VM t avoid any version mismatches


## Problem of using GPU in virtualbox

- GPU already required in host for simulator
- sharing can lead to memory problems, see https://superuser.com/questions/779070/use-nvidia-gpu-from-virtualbox
- vmware better?
- virtualbox 3d acceleration = "by intercepting OpenGL requests", see https://blogs.oracle.com/scoter/3d-acceleration-for-ubuntu-guests-v2
- virtualbox 2d acceleration = for video applications, see https://askubuntu.com/questions/187753/how-much-difference-can-2d-acceleration-in-virtualbox-bring
https://superuser.com/questions/1094936/what-are-2d-video-acceleration-and-3d-acceleration

-> I cannot use GPU in virtualbox. Thus, create classifier "offline" and run it using CPU only in VM.


## Setting up VM

1. activate colors in bash
	-> gedit ~/.bashrc and uncomment force-color line
	
2. install editors (such as pycharm / geany)
	-> sudo apt update -> results in errors (key not found or something)
	-> fix it with by searching on stackoverflow
	
3. Fork and clone github repo (CarNd-Capstone)

4. Compile sources
	- go to .../CarNd-Capstone/ros
	- catkin_make
	
5. Try to launch stuff
	- roslaunch .../CarNd-Capstone/ros/launch/styx.launch
	-> will yield errors?!

6. Open ports (4567!), see udacity virtualbox pdf
	
7. Disable simuatlor in launch file (not sure if necessary)
	- open .../CarNd-Capstone/ros/src/styx/launch/server.launch
	- comment out node after "launch simulator"

8. Try to launch ONLY styx server
	- roslaunch .../CarNd-Capstone/ros/src/stys/launch/server.launch
	-> yielded an "ImportError: no module named eventlet" error
	-> go to root folder of clone git repository, where you will find a requirements.txt file
	-> pip install -r requirements.txt (or similar command)
	
9. Debug with pycharm
	. Install pycharm (see website)
	. Start pycharm from terminal, so that environment variables are imported!
		$ source /opt/ros/kinetic/setup.bash (if not already in ~/.bashrc)
		$ source ~/git/CarND-Capstone/ros/devel/setup.bash
		$ bash /snap/pycharm-community/current/bin/pycharm.sh
	. Select as project interpreter (settings) /usr/bin/python (should be v2.7!)
	. Launch some nodes via terminal in roslaunch
	. Simply run/debug the *.py file of the new node from pycharm !
	
	. helpful links: http://answers.ros.org/question/105711/rospy-custom-message-importerror-no-module-named-msg/

# My notes during creating the solution

## Solutions from other students

- https://github.com/d2macster/self-driving-car-capstone
- https://github.com/swap1712/carnd-capstone
- https://dmavridis.github.io/CarND_System_Integration/
- http://jeremyshannon.com/2017/11/08/udacity-sdcnd-capstone-pt1.html
- https://medium.com/udacity/the-end-of-a-self-driving-car-journey-finale-running-code-on-a-real-car-graduation-a10607ed0180

## Tips for solution from udacity pages

1. waypoint updater node
	inputs: /base_waypoints = styx_msgs::Lane
			/current_pose = geometry_msgs::PoseStamped (x,y,z,w=yaw in quaternion)
	outputs: /final_waypoints = styx_msgs::Lane = list of waypoints,
		/final_waypoints = styx_msgs::Lane = list of waypoints, which contain
				geometry_msgs/PoseStamped pose = pose (x,y,z) and orientation (x,y,z,w=yaw (what for?!))
				geometry_msgs/TwistStamped twist = x,y,z, angle_x,y,z
	"purpose of this node is to update target velocity property of each waypoint"
	??? - How should I integrate velocity to output message?

1.1. Waypoint follower node (already existing!)
	inputs: /final_waypoints = styx_msgs::Lane
	outputs: /twist_cmd = geometry_msgs::TwistStamped
	
2. DBW node
	inputs: /twist_cmd, /current_velocity
	outputs: /vehicle/steering, throttle, brake
	--> milestone: car should drive in the simulator, ignoring the traffic lights
	
3. Traffic light detection node
	inputs: /image_color (/vehicle/traffic_lights for generating ground truth, see below)
	outputs: /traffic_waypoints
	
4. Waypoint updater node (Full)
	inputs: /traffic_waypoint in addition to previous inputs (see above)
	outputs: /final_waypoints


## Waypoint updater

- publish the _next_ ~200 waypoints ahead of vehicle !
- velocity is stored in twist.linear.x ?!
- see detailed walkthrough in video !

## DBW

- output values
	- throttle should be in range [0,1]
	- break should be in N*m -> compute using desired acceleration, weight and wheel radius
		(700 Nm of torque required for standstill)
		force = mass_ofcar * acceleration
		torque = force * wheel_radius
		torque limited by decel_limit parameter
- usage of twist controller
	input: twist data (goal?!)
	output: throttle, brake, steering
	WITHIN this method use...
		for acceleration (throttle, break) -> pid.py and lowpass.py
			PID controller values: Kp=0.3, Kd=4.0, Ki=0.003
		for steering -> yaw_controller.py
		

Idea how to implement (before realising there's also a walkthrough for that...):
	input:	
		twist_goal (=twist_cmd = proposed linear and angular velocities)
		twist_current (=current_velocity, pose?!)
	output:
		steering = yaw_controller.get_steering(vel_goal_z, vel_goal_angle_z, vel_curr)
		
- see detailed walkthrough in video (only found that later...) !

## Consider traffic lights using ground truth data

### Tips from video for traffic light detection
- for generating ground truth use (vehicle/traffic_lights), which contains
	- traffic light position in 3d 
	- current color state
- there are already pretrained SSD models available (tf zoo)
- potentially use dataset from others?
- Carla uses tensorflow 1.3.0
- for traffic light
	- if no stop line upfront -> return index (-1)

### Tips from video for final version of waypoint updater
- stop 2 waypoints before stop line 
- setup deceleration waypoints 
	acc = dv/dt = dv/(s/v) = dv*v/s <-> dv <= acc_max*s/v !
	OR v = sqrt(2*acc_max*dist)
- make sure that deceleration is not larger than speed limit at the end...


## Consider traffic lights using CNN classifier

### Export data
- not every image, but rather every third or fifth (little variation in between)
- a few manual runs, a few automatic runs using ground truth information
- make sure from each traffic light at least a few images red and green!
-> Got in total ~1000 images, though only 100 yellow ones...

### Image preprocessing
- cropping?
- Resizing? 
	thog
	
### Image augmentation
- hor flip
- slight zoom (really only slightly so that lights are still in image!)
- slight moving (+/-5 pixels in both directions?)
- color shift (don't think necessary, even far away exact same color in simulator)

### Manual feature engineering
- Only R-channel
	-> problem: sky is nearly white=100%R, so not only traffic lights...
	-> use also G&B to distinguish!
- HSV -> H=red and saturation = full (white is not 100% saturated!)
	-> problem with HS = nearly black can have same values as red

### Selection of model architecture
- idea to implement "if there is _any_ red light in the image"
	- a global max pooling layer or threshold + global summing layer?
	- before: some conv layer to identify red traffic light in various sizes
- searching for "fast image classification" yielded mobilenets
	https://hackernoon.com/creating-insanely-fast-image-classifiers-with-mobilenet-in-tensorflow-f030ce0a2991
	
#### How to train model?
- Main insight:
	- VM already has protobuf v3.8.0, which is current version.
	- So we can definitely use protobuf file as "exchange" format, i.e- train with different versions and simply run tensorflow pb file
- Option 1: Start from scratch using tensorflow 1.3.0
	--> Assures that runs with tf 1.3.0, but takes a looot of time...
- Option 2: Use some existing scripts
	- https://www.tensorflow.org/tutorials/images/hub_with_keras
	- https://www.tensorflow.org/hub/tutorials/image_retraining
	- https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md
	- https://hackernoon.com/creating-insanely-fast-image-classifiers-with-mobilenet-in-tensorflow-f030ce0a2991
	- At least give it a try, might work 
		- mobilenet_v2_100_224 does not work, because it uses dilations. Error when trying to run it:
			InvalidArgumentError (see above for traceback): NodeDef mentions attr 'dilations' not in Op
			(It seems that arbitrary dilations were only added in tf1.6, see https://github.com/tensorflow/tensorflow/releases?after=v1.6.0
		- mobilenet_v1 has same error :/
	- just create a simple model with keras yourself
		- Same error as above!!! (dilations not in Op) Argh...
	- Save checkpoint file and open that one in tf 1.3.0
		- error "ValueError: No op named DivNoNan in defined operations"
		- -> thus, go with option 1: train from scratch
- Option 1:
	- https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network.py
	- https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network_raw.py
	- Create simple architecture
		- input size 320x240
		- Options:
			4x blocks to scale down to 20x15
			(5x blocks to scale down to 10x8)
		- Then, another convolution and a maxpool over the whole image
			-> yields a (?,1,1,num_filters) size
		- flatten
		- some dense connection to connect to (?, num_classes)
		- > Works and is really small (~500kB protobuf file !!!)

			
		
	

	

	
