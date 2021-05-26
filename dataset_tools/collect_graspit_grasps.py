"""
runs SA grasp planner in GraspIt! and collects a list of grasps along with their
rankings (energy values)
"""
import rospy
from graspit_commander import GraspitCommander as gc
from graspit_interface.msg import Planner
from itertools import product
import numpy as np
from collect_surface_grasps import rospose2mat

def apply_grasp(grasp):
  """
  This function applies a grasp to the robot hand
  :param grasp:
  :return:
  """
  gc.setRobotPose(grasp.pose)
  gc.forceRobotDof(grasp.dofs)

if __name__ == '__main__':
  n_target_grasps = 5000

  proposed_grasps = []

  csv_header = []
  for y, x in product(range(3), range(4)):
    csv_header.append('T{:d}{:d}'.format(y, x))
  for i in range(20):
    csv_header.append('dof{:d}'.format(i))
  csv_header.append('POTENTIAL_QUALITY_ENERGY')
  csv_header.append('GUIDED_POTENTIAL_QUALITY_ENERGY')
  csv_header.append('CONTACT_ENERGY')
  csv_header.append('quality_volume')
  csv_header.append('quality_epsilon')

  while len(proposed_grasps) < n_target_grasps:
    # load the world
    gc.clearWorld()
    gc.loadWorld("hand_camera")

    # run SA planner
    results = gc.planGrasps(planner=Planner(Planner.SIM_ANN),
      search_energy="CONTACT_ENERGY")
    rospy.loginfo('Found {:d} grasps'.format(len(results.grasps)))

    # refine the top grasps and compute energies and quality measure
    for grasp in results.grasps:
      apply_grasp(grasp)
      # autograsp to close the fingers
      gc.autoGrasp()
      proposal = []

      # get refined pose
      robot = gc.getRobot().robot
      palm_pose = rospose2mat(robot.pose)
      proposal.extend(palm_pose[:3].flatten())
      proposal.extend(robot.dofs)

      # energies
      energy_type = csv_header[len(proposal)]
      try:
        e = gc.computeEnergy(energy_type)
        e = e.energy
      except:
        e = np.finfo(float).max
      proposal.append(e)

      energy_type = csv_header[len(proposal)]
      try:
        e = gc.computeEnergy(energy_type)
        e = e.energy
      except:
        e = np.finfo(float).max
      proposal.append(e)

      energy_type = csv_header[len(proposal)]
      try:
        e = gc.computeEnergy(energy_type)
        e = e.energy
      except:
        e = np.finfo(float).max
      proposal.append(e)

      # quality
      try:
        q = gc.computeQuality()
        qw, qe = q.volume, q.epsilon
      except:
        qw = qe = -np.finfo(float).max
      proposal.append(qw)
      proposal.append(qe)

      if len(proposal) == len(csv_header):
        proposed_grasps.append(proposal)

    header = ','.join(csv_header)
    X = np.asarray(proposed_grasps)
    np.savetxt('../data/grasps2.csv', X, delimiter=',', header=header)

    print('Saved {:d} grasps'.format(len(proposed_grasps)))