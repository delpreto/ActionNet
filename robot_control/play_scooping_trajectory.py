"""
Given an input HDF5 file containing scooping task trajectories
Plays the trajectory on baxter using Inverse Kinematics
"""
import os
import numpy as np
from collections import OrderedDict

from BaxterController import BaxterController
from BaxterHeadController import BaxterHeadController


# - Constants - #

# Specify resting joint angles for each arm.
resting_joint_angles_rad = {
  'left': OrderedDict([('left_s0', -0.2849369313497156), ('left_s1', -0.023009711818281205), ('left_e0', -0.388480634531981), ('left_e1', 1.2620826932327243), ('left_w0', 0.7689078699275637), ('left_w1', -1.1213399559442374), ('left_w2', 1.5558400141127808)]),
  'right': OrderedDict([('right_s0', 0.7025632008515195), ('right_s1', -0.146495165243057), ('right_e0', 0.15109710760671324), ('right_e1', 1.525543893552044), ('right_w0', 0.4421699621079705), ('right_w1', -1.565810889234036), ('right_w2', -1.7495050885833143)]),
}


# - Main - #

def play_scooping_trajectory(
    input_trajectory_directory,
    speed_factor,
):
    # Load trajectory
    time = np.load(input_trajectory_directory + 'time.npy')
    angles = np.load(input_trajectory_directory + 'angles.npy')

    # Inflate time series
    time *= 1 / speed_factor

    # Initialize controllers
    controller_right = BaxterController(limb_name='right', print_debug=True)
    controller_right.set_resting_joint_angles_rad(resting_joint_angles_rad['right'])
    controller_right.open_gripper()
    head_controller = BaxterHeadController()
    head_controller.showColor('black')
    head_controller.setHaloLED(green_percent=0, red_percent=0)

    joint_names = ['right_s0', 'right_s1', 'right_e0', 'right_e1', 'right_w0', 'right_w1', 'right_w2']
    def wrap_angle(joint_names, angle):
        return {k: v for k,v in zip(joint_names, angle)}
    
    joint_angles = [wrap_angle(joint_names, angle) for angle in angles]
    
    controller_right.build_trajectory_from_joint_angles(
        time.tolist(),
        joint_angles,
        should_print=False
    )

    # Move to start
    controller_right.move_to_trajectory_start(wait_for_completion=True)
    head_controller.setHaloLED(green_percent=100, red_percent=100)

    # Run trajectory
    controller_right.run_trajectory(wait_for_completion=True)
    head_controller.setHaloLED(green_percent=100, red_percent=0)
    head_controller.nod(times=2)
    
    # Shutdown
    head_controller.showColor('black')
    head_controller.setHaloLED(green_percent=0, red_percent=0)
    head_controller.quit()
    controller_right.quit()


if __name__ == '__main__':
    # Script inputs
    input_trajectory_directory = os.path.expanduser('~/data/scooping/LinOSS_train_scooping_5678_straight_angles/trajectory_001/')
    speed_factor = 0.66

    # Main
    play_scooping_trajectory(
        input_trajectory_directory,
        speed_factor,
    )