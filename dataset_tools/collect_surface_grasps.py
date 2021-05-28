"""
collects grasps by orienting the palm to the surface normal for each
point on object surface
"""
import init_paths
import open3d
import numpy as np
from graspit_commander.graspit_commander import GraspitCommander
from graspit_interface.msg import Planner, SearchSpace, SearchContact
from geometry_msgs.msg import Pose, Point, Quaternion
import tf.transformations as tx
from utils import tf_utils
from itertools import product
from sklearn.decomposition import PCA
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


class GraspCollector(object):
  def __init__(self,
      object_name,
      models_dir,
      output_dir='../data/grasps/',
      N_points=100,
      N_angles=4,
      N_dists=4,
      dist_step=0.01,
      N_graspit_grasps=2,
      height_thresh=-5000,
      scale_factor=1.15,
      hand_name='human'
  ):
    """
    :param object_name: assumes object has Z axis pointing up!
    :param: models_dir: directory containing all the object models
    :param output_dir: output will be saved to output_dir/object_name.csv
    :param N_points: number of points to sample of object surface
    :param N_angles: number of roll angles around approach vector
    :param N_dists: number of translation steps along the approach vector
    :param dist_step: length of each translation step along approach vector
    :param N_graspit_grasps: number of top GraspIt! grasps to consider for each
    seed
    :param height_thresh: object points below this height (Z axis) will be
    ignored
    :param scale_factor: scale factor by which the graspit world is scaled
    because of graspit's hand model being large
    :param hand_name: human, allegro, barrett
    """
    self.hand_name = hand_name
    self.object_name = object_name
    self.energy_type = 'GUIDED_POTENTIAL_QUALITY_ENERGY' if \
        self.hand_name=='barrett' else 'CONTACT_ENERGY'

    self.world_name = '{:s}_{:s}'.format(self.hand_name, self.object_name)
    self.output_filename = osp.join(output_dir,
        '{:s}_grasps_{:s}.csv'.format(self.object_name, self.hand_name))
    print('Will save results to {:s}'.format(self.output_filename))
    obj = open3d.read_triangle_mesh(osp.expanduser(osp.join(models_dir,
      '{:s}.ply'.format(object_name))))
    obj.transform(scale_factor * np.eye(4))
    obj.compute_vertex_normals(normalized=True)

    self.N_points = N_points
    self.N_angles = N_angles
    self.N_dists  = N_dists
    self.dist_step = dist_step
    self.N_graspit_grasps = N_graspit_grasps
    self.height_thresh = height_thresh
    self.back_off_dist = 0.02  # need to back off by this after initial contact,
    # so that valid grasps can be planned

    # read object points and normals
    obj_pts = np.asarray(obj.vertices)
    obj_ns = np.asarray(obj.vertex_normals)

    # find principal axes of the object (in world coordinate frame)
    pca = PCA(n_components=1)
    pca.fit(obj_pts.copy())
    p_axis = pca.components_[0]
    self.w_p_axis = p_axis / np.linalg.norm(p_axis)
    self.w_p_axis = np.hstack((self.w_p_axis, 1))

    # for debug
    # # idx = 5406  # camera
    # idx = 1087    # binoculars  8844
    # p = obj_pts[idx]
    # idx = np.logical_and(np.all(obj_pts > p-0.02, axis=1), np.all(obj_pts < p+0.02, axis=1))
    # obj_pts = obj_pts[idx]
    # obj_ns = obj_ns[idx]

    # ignore points below height_thresh
    idx = obj_pts[:, 2] >= self.height_thresh
    obj_pts = obj_pts[idx]
    obj_ns = obj_ns[idx]

    # shuffle object points
    idx = np.asarray(range(len(obj_pts)))
    np.random.shuffle(idx)
    self.obj_pts = obj_pts[idx]
    self.obj_ns = obj_ns[idx]

    # pose of palm w.r.t. approach vector
    # from HumanHand20DOF.xml, element approachDirection
    if self.hand_name == 'human':
      pTa = tf_utils.rotmat_from_vecs([-np.cos(np.pi / 4), -np.sin(np.pi / 4), 0],
        [0, 0, 1])
      pTa[:3, 3] = [-0.13, -0.015, 0]
    elif self.hand_name == 'allegro':
      pTa = tf_utils.rotmat_from_vecs([np.cos(np.pi / 4), 0, np.sin(np.pi / 4)],
        [0, 0, 1])
    elif self.hand_name == 'barrett':
      pTa = np.eye(4)
    self.aTp = tx.inverse_matrix(pTa)

    self.gc = GraspitCommander()

    # find number of dofs
    self.gc.loadWorld(self.world_name)
    n_dofs = len(self.gc.getRobot().robot.dofs)
    self.gc.clearWorld()

    csv_header = []
    for y, x in product(range(3), range(4)):
      csv_header.append('T{:d}{:d}'.format(y, x))
    for i in range(n_dofs):
      csv_header.append('dof{:d}'.format(i))
    # surface point that generated this grasp
    csv_header.append('surface_pt_x')
    csv_header.append('surface_pt_y')
    csv_header.append('surface_pt_z')
    # surface normal at the point
    csv_header.append('surface_normal_x')
    csv_header.append('surface_normal_y')
    csv_header.append('surface_normal_z')
    csv_header.append('roll')  # roll of the hand around approach axis
    csv_header.append('dist')  # distance of hand from the surface point,
    csv_header.append(self.energy_type)  # GraspIt's measure of energy
    # along approach axis
    self.csv_header = csv_header

  def _apply_grasp(self, grasp):
    """
    This function applies a grasp to the robot hand
    :param grasp:
    :return:
    """
    self.gc.setRobotPose(grasp.pose)
    self.gc.forceRobotDof(grasp.dofs)

  def _is_valid_pose(self):
    """
    Checks if the current hand pose is valid
    Checks if there are 2 contacts between hand and object
    :return:
    """
    contacts = self.gc.getRobot().robot.contacts
    n_object_contacts = 0
    is_thumb_in_contact = False
    for contact in contacts:
      if contact.body1 == self.object_name:
        n_object_contacts += 1
        if contact.body2 == '_chain4_link2':
          is_thumb_in_contact = True
      elif contact.body2 == self.object_name:
        n_object_contacts += 1
        if contact.body1 == '_chain4_link2':
          is_thumb_in_contact = True

    is_valid = n_object_contacts >= 2
    return is_valid

  def _back_off_along_approach_vector(self, back_off_dist):
    wTp = rospose2mat(self.gc.getRobot(0).robot.pose)
    wTa = np.dot(wTp, tx.inverse_matrix(self.aTp))
    T = np.eye(4)
    T[2, 3] = -back_off_dist
    wTa = np.dot(wTa, T)
    wTp = np.dot(wTa, self.aTp)
    self.gc.setRobotPose(mat2rospose(wTp))

  def run(self):
    proposed_grasps = []

    for pt_idx, (obj_pt, obj_n) in enumerate(zip(self.obj_pts[:self.N_points],
        self.obj_ns[:self.N_points])):

      if pt_idx % 1 == 0:
        print('{:d}/{:d}'.format(pt_idx, self.N_points))

      t = obj_pt + obj_n*0.01
      if t[2] < self.height_thresh:
        continue
      wTa_this_pt = tf_utils.rotmat_from_vecs(-obj_n, [0, 0, 1])
      wTa_this_pt[:3, 3] = t
      # align roll 0 degrees to object's principal axis
      # project object's principal axis to approach coordinate system
      a_p_axis = np.dot(tx.inverse_matrix(wTa_this_pt), self.w_p_axis)[:3]
      # project to unit vector in XY plane
      a_p_axis[2] = 0
      if np.linalg.norm(a_p_axis) >= 0.1:
        a_p_axis = a_p_axis / np.linalg.norm(a_p_axis)
        # rotation matrix to align X axis with the projected vector
        R = tf_utils.rotmat_from_vecs(a_p_axis, [1, 0, 0])
        wTa_this_pt = np.dot(wTa_this_pt, R)

      n_grasps = 0
      for angle in np.linspace(0, 360-(360.0/self.N_angles), self.N_angles):
        R = tx.euler_matrix(0, 0, np.deg2rad(angle))
        wTa = np.dot(wTa_this_pt, R)
        wTp = np.dot(wTa, self.aTp)
        if np.isnan(wTp).any():
          print('@@@ NaN value in pose, skipping @@@')
          continue

        print('Angle = {:f}'.format(angle))
        self.gc.clearWorld()
        self.gc.loadWorld(self.world_name)
        self.gc.toggleAllCollisions(True)
        self.gc.setRobotPose(mat2rospose(wTp))
        self.gc.findInitialContact(0.5)

        # back off a bit to be able to generate valid neighbor grasps
        self._back_off_along_approach_vector(self.back_off_dist)

        # plan grasps!
        grasps = self.gc.planGrasps(planner=Planner(Planner.SIM_ANN),
          search_energy=self.energy_type,
          search_space=SearchSpace(SearchSpace.SPACE_APPROACH),
          search_contact=SearchContact(SearchContact.CONTACT_PRESET),
          max_steps=45000).grasps
        n_grasps += min(len(grasps), self.N_graspit_grasps)
        self.gc.toggleAllCollisions(False)

        # refine the top grasps and compute energies and quality measure
        for grasp in grasps[:self.N_graspit_grasps]:
          for dist in range(self.N_dists):
            self._apply_grasp(grasp)

            # back-off
            self._back_off_along_approach_vector(dist*self.dist_step)

            dofs = np.asarray(grasp.dofs)
            self.gc.forceRobotDof(dofs)

            proposal = []
            robot = self.gc.getRobot().robot
            palm_pose = rospose2mat(robot.pose)
            proposal.extend(palm_pose[:3].flatten())
            proposal.extend(robot.dofs)
            proposal.extend([obj_pt[0], obj_pt[1], obj_pt[2]])
            proposal.extend([obj_n[0], obj_n[1], obj_n[2]])
            proposal.append(angle)
            proposal.append(dist*self.dist_step)
            try:
              e = self.gc.computeEnergy(self.energy_type)
              e = e.energy
            except:
              e = np.finfo(float).max
            proposal.append(e)

            if len(proposal) == len(self.csv_header):
              proposed_grasps.append(proposal)
            else:
              print('len(proposal) = {:d}, len(csv header) = {:d}'.format(
                len(proposal), len(self.csv_header)))


        header = ','.join(self.csv_header)
        X = np.asarray(proposed_grasps)
        np.savetxt(self.output_filename, X, delimiter=',', header=header)
      print('Found {:d} grasps'.format(n_grasps))

    print('Saved {:d} grasps to {:s}'.format(len(proposed_grasps),
      self.output_filename))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--object_name', required=True)
  parser.add_argument('--models_dir', default=osp.join('data', 'object_models'))
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
  gc = GraspCollector(args.object_name, args.models_dir, hand_name=args.hand_name)
  gc.run()

  # shutdown GraspIt!
  launch.shutdown()
