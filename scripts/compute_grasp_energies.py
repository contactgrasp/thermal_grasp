"""
takes a file containing grasps and computes their energy using graspit
"""
import init_paths
import numpy as np
from graspit_commander.graspit_commander import GraspitCommander
from graspit_interface.msg import Planner, SearchSpace, SearchContact
from geometry_msgs.msg import Pose, Point, Quaternion
import tf.transformations as tx
from utils import tf_utils
import os
import argparse
import roslaunch
from IPython.core.debugger import set_trace
osp = os.path

def mat2rospose(T):
  q = tx.quaternion_from_matrix(T)
  orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
  position = Point(x=T[0, 3], y=T[1, 3], z=T[2, 3])
  pose = Pose(position=position, orientation=orientation)
  return pose

def rospose2mat(pose):
  q = [pose.orientation.x, pose.orientation.y, pose.orientation.z,
    pose.orientation.w]
  T = tx.quaternion_matrix(q)
  T[0, 3] = pose.position.x
  T[1, 3] = pose.position.y
  T[2, 3] = pose.position.z
  return T


class EnergyComputer(object):
  def __init__(self,
      object_names=None,
      energy_types=['CONTACT_ENERGY'],
      data_dir='../data/grasps/',
      hand_name='human'
  ):
    self.hand_name = hand_name
    self.data_dir = osp.expanduser(data_dir)
    
    # candidate grasp filenames
    if object_names is not None:
      grasp_filenames = [osp.join(self.data_dir, '{:s}_grasps').\
          format(object_name) for object_name in object_names]
      if self.hand_name != 'human':
        grasp_filenames = ['{:s}_{:s}'.format(gn, self.hand_name) \
            for gn in self.grasp_filenames]
      grasp_filenames = ['{:s}.csv'.format(gn) for gn in grasp_filenames]
    else:
      grasp_filenames = []
      object_names = []
      pattern = '_grasps'
      if self.hand_name != 'human':
        pattern = '{:s}_{:s}'.format(pattern, self.hand_name)
      pattern = '{:s}.csv'.format(pattern)
      for filename in next(os.walk(self.data_dir))[-1]:
        if pattern not in filename:
          continue
        grasp_filenames.append(osp.join(self.data_dir, filename))
        object_name = filename.replace(pattern, '')
        object_names.append(object_name)

    # check which energies need to be computed for each grasp file
    energies_to_compute = []
    for grasp_filename in grasp_filenames:
      energies = []
      with open(grasp_filename, 'r') as f:
        header = f.readline().strip()
        header = header.split(',')
      for energy in energy_types:
        if (energy.upper() not in header) and (energy.lower() not in header):
          energies.append(energy)
      energies_to_compute.append(energies)

    # graspit world filenames
    world_names = []
    for object_name in object_names:
      prefix = 'hand' if self.hand_name == 'human' else self.hand_name
      world_name = '{:s}_{:s}'.format(prefix, object_name)
      world_names.append(world_name)
    

    self.gc = GraspitCommander()

    self.run(grasp_filenames, energies_to_compute, world_names)


  def _apply_grasp(self, pose, dofs):
    self.gc.setRobotPose(mat2rospose(pose))
    self.gc.forceRobotDof(dofs)

  
  def run(self, grasp_filenames, energies_to_compute, world_names):

    for grasp_filename, energies, world_name in \
        zip(grasp_filenames, energies_to_compute, world_names):

      if len(energies) == 0:
        continue
      
      # load world
      self.gc.clearWorld()
      self.gc.loadWorld(world_name)
      self.gc.toggleAllCollisions(False)

      # read file
      grasps = np.loadtxt(grasp_filename, delimiter=',')
      with open(grasp_filename, 'r') as f:
        header = f.readline().strip()[2:]
      header = header.split(',') + energies
      header = ','.join(header)

      # compute required energies
      output_grasps = []
      n_dofs = len(self.gc.getRobot().robot.dofs)
      for idx,grasp in enumerate(grasps):
        if idx % 100 == 0:
          print('{:s}: Grasp {:d}/{:d}'.format(world_name, idx, len(grasps)))
        T = np.eye(4)
        T[:3] = grasp[:12].reshape((3, 4))
        dofs = grasp[12 : 12+n_dofs]
        self._apply_grasp(T, dofs)

        es = []
        for energy in energies:
          try:
            e = self.gc.computeEnergy(energy)
            e = e.energy
          except:
            e = np.finfo(float).max
          es.append(e)
        output_grasps.append(np.hstack((grasp, es)))

      # save the csv file with energies
      X = np.asarray(output_grasps)
      output_filename = grasp_filename[:-4] + '_energies.csv'
      np.savetxt(output_filename, X, delimiter=',', header=header)
      print('{:s} written'.format(output_filename))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--object_names', default=None)
  parser.add_argument('--energy_types', default='CONTACT_ENERGY')
  parser.add_argument('--hand_name', required=True)
  args = parser.parse_args()

  # launch GraspIt!
  # rospy.init_node('graspit_node', anonymous=True)
  uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
  roslaunch.configure_logging(uuid)
  os.environ['GRASPIT'] = osp.abspath(osp.join('..', 'data', 'graspit'))
  catkin_dir = os.environ['ROS_PACKAGE_PATH'].split(':')[0]
  launch_filename = osp.expanduser(osp.join(catkin_dir, 'graspit_interface',
    'launch', 'graspit_interface.launch'))
  launch = roslaunch.parent.ROSLaunchParent(uuid, [launch_filename])
  launch.start()

  # collect grasps
  object_names = args.object_names
  if object_names is not None:
    object_names = object_names.split(',')
  energy_types = args.energy_types.split(',')
  ec = EnergyComputer(object_names, energy_types, hand_name=args.hand_name)

  # shutdown GraspIt!
  launch.shutdown()
