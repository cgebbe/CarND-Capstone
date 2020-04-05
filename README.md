# About

<img src="report/output.gif" width="600">

### Problem

Drive along a set of waypoints and respect the traffic lights, i.e. stop in front of them if they are red.

### Solution approach

![](README.assets/capstone_ros_graph_my.png)

We were already given a ROS system structure (shown above) and had to write three ROS nodes, which are explained in more detail below

1. Waypoint Updater node
2. Drive By Wire (DBW) node
3. Traffic Light Detection node

#### Waypoint Updater node

- Inputs:
  - From waypoint loader node: All waypoints (x,y)
  - From simulator: current pose (x,y,yaw)
  - From traffic light detection node: One waypoint (x,y) in case of a red light, where the car shall stop at
- Outputs:
  - To waypoint follower: Only the next ~200 waypoints. Additionally, each waypoint is assigned a target velocity, which usually equals the speed limit, but can drop to zero in case of a red light

#### Drive By Wire (DBW) node

- Inputs:
  - From waypoint follower node: twist command representing the target linear and angular velocities
  - From simulator: current velocity
- Outputs
  - To simulator: Steering angle, based on mainly the target angular velocity while taking account specific vehicle parameters such as the wheel base, steer ratio and maximum steering angle
  - To simulator: Acceleration & Break, calculated via a PID-controller which minimizes the difference between the current and target velocity

#### Traffic Light Detection node

- Inputs
  - From waypoint loader node: All waypoints including the information of the traffic light position
  - From simulator: current pose
  - From simulator: RGB camera image
- Outputs
  - To waypoint updater: If a red light was detected in the image, this node sends one waypoint at which the car shall stop. If there is no red light, this node sends a "free" signal. Whether an image contains a red light, is determined via a very simple detector written in tensorflow, for which the training data was manually collected from the simulator through manual steering.

### Result

- My solution passed the test successfully. If interested, see my [notes](report/notes.md)

# FAQ

See detailed instructions in original [original README.org](README_org.md)

### How to install

- Download the virtual machine (VM) image from https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/Udacity_VM_Base_V1.0.0.zip
  - Login with the user `student` and the password ` udacity-nd`
  - In the virtual machine
    - clone this repository
    - install the required python libraries `pip install -r requirements.txt` 
- Outside the VM: Download the capstone simulator from https://github.com/udacity/CarND-Capstone/releases

### How to run

- In the virtual machine

  - Make and run styx

    ```
    cd ros
    catkin_make
    source devel/setup.sh
    roslaunch launch/styx.launch
    ```

- Outside the VM: Start the capstone simulator
  - Select Highway Scenario
  - Turn off Manual Mode (only used for collecting training images)
  - Turn on Camera (otherwise traffic lights are ignored)