"""
Input: Values of the DoFs of the hand
Output: Rendering of the hand
"""
import numpy as np
from utils import tf_utils as tx
import open3d
import os.path as osp
import xml.etree.ElementTree as ET
import sympy
from copy import deepcopy

def interpret_dh(alpha, a, theta, d):
  """
  Converts the modified DH parameters to a 4x4 transformation matrix
  :param alpha: degrees
  :param a:
  :param theta: degrees
  :param d:
  :return: 4 x 4 np.ndarray
  """

  alpha = np.deg2rad(alpha)
  theta = np.deg2rad(theta)

  T_alpha = tx.euler_matrix(alpha, 0, 0)
  T_a = np.eye(4); T_a[0, 3] = a
  T_theta = tx.euler_matrix(0, 0, theta)
  T_d = np.eye(4); T_d[2, 3] = d

  T = T_theta
  T = np.dot(T, T_d)
  T = np.dot(T, T_a)
  T = np.dot(T, T_alpha)

  return T


class RobotRenderer(object):
  """
  Manages the rendering of the robot, needs graspit XML file as input
  """
  def __init__(self, xml_filename, palm_pose=np.eye(4),
      xml_dir=osp.join('~', '.graspit', 'models', 'robots', 'HumanHand'),
      geometry_dir=osp.join('..', 'data', 'human_hand')):
    """
    :param xml_filename:
    :param xml_dir:
    :param geometry_dir: directory containing the pointclouds of the hand-part
    meshes
    """
    self.palm_pose = palm_pose
    self.xml_filename = osp.join(osp.expanduser(xml_dir), xml_filename)
    self.geometry_dir = osp.expanduser(geometry_dir)
    self.links = {}  # mapping from link XML filenames to poses w.r.t. palm
    self.dof_mins = []
    self.dof_maxs = []
    self.link_pcs = {}

    # find number of DoFs
    self.n_dofs = 77  # random number
    self.dof_vals = np.zeros(self.n_dofs)
    self.parse_hand_file()
    self.dof_vals = np.zeros(self.n_dofs)

  def set_dof_vals(self, dof_vals):
    """
    :param dof_vals: iterable, in radians
    :return:
    """
    assert len(dof_vals) == self.n_dofs
    self.dof_vals = np.rad2deg(dof_vals)
    self.parse_hand_file()

  def eval_exp(self, s, min_val=np.float('inf'), max_val=-np.float('inf')):
    """
    evaluates symbolic expression by replacing the variable with the proper DoF
    value
    :param s: string representation of the symbolic expression
    :param min_val: min value of the evaluated expression
    :param max_val: max value of the evaluated expression
    :return:
    """
    expr = sympy.sympify(s)
    dof_var = list(expr.free_symbols)[0]
    dof_idx = int(str(dof_var)[1:])
    dof_val = self.dof_vals[dof_idx]
    # if dof_val > max_val:
    #   dof_val = max_val
    #   print 'DoF {:d} upper clamped to {:f}'.format(dof_idx, max_val)
    # elif dof_val < min_val:
    #   dof_val = min_val
    #   print 'DoF {:d} lower clamped to {:f}'.format(dof_idx, min_val)

    # evaluate expression
    out = expr.subs(dof_var, dof_val)
    out = float(out)

    return out

  def parse_hand_file(self):
    tree = ET.parse(self.xml_filename)
    robot = tree.getroot()

    # read palm information
    palm_filename = robot.find('palm').text
    self.links[palm_filename] =  np.eye(4)

    # get poses of all finger segments w.r.t. palm
    self.n_dofs = 0
    for chain in robot.findall('chain'):
      # transform from chain base to palm
      transform = chain.find('transform')
      translation = np.fromstring(transform.find('translation').text, sep=' ')
      T = np.eye(4)
      T[:3, -1] = translation
      try:
        angle, axis = transform.find('rotation').text.split(' ')
        angle = np.deg2rad(float(angle))
        if axis.upper() == 'X':
          R = tx.euler_matrix(angle, 0, 0)
        elif axis.upper() == 'Y':
          R = tx.euler_matrix(0, angle, 0)
        elif axis.upper() == 'Z':
          R = tx.euler_matrix(0, 0, angle)
        else:
          raise IOError('wrong axis of rotation {:s}'.format(axis))
      except AttributeError:
        rotmat = np.fromstring(transform.find('rotationMatrix').text, sep=' ')
        R = np.eye(4)
        R[:3, :3] = np.reshape(rotmat, (3, 3)).T

      T = np.dot(T, R)

      # chain DH parameters
      joint_Ts = []
      for joint in chain.findall('joint'):
        joint_min = float(joint.find('minValue').text)
        joint_max = float(joint.find('maxValue').text)

        theta = joint.find('theta').text
        if theta.startswith('d'):
          theta = self.eval_exp(theta, joint_min, joint_max)
          self.n_dofs += 1
        else:
          theta = float(theta)
        d = joint.find('d').text
        if d.startswith('d'):
          d = self.eval_exp(d, joint_min, joint_max)
          self.n_dofs += 1
        else:
          d = float(d)
        a = joint.find('a').text
        if a.startswith('d'):
          a = self.eval_exp(a, joint_min, joint_max)
          self.n_dofs += 1
        else:
          a = float(a)
        alpha = joint.find('alpha').text
        if alpha.startswith('d'):
          alpha = self.eval_exp(alpha, joint_min, joint_max)
          self.n_dofs += 1
        else:
          alpha = float(alpha)

        T_joint = interpret_dh(alpha, a, theta, d)
        joint_Ts.append(T_joint)

      # link information
      joint_idx = 0
      link_T = T
      for link in chain.findall('link'):
        if link.attrib['dynamicJointType'] == 'Universal':
          link_T = np.dot(link_T, joint_Ts[joint_idx])
          link_T = np.dot(link_T, joint_Ts[joint_idx+1])
          self.links[link.text] = link_T
          joint_idx += 2
        elif (link.attrib['dynamicJointType'] == 'Revolute') or\
            (link.attrib['dynamicJointType'] == 'Prismatic'):
          link_T = np.dot(link_T, joint_Ts[joint_idx])
          self.links[link.text] = link_T
          joint_idx += 1
        else:
          raise ValueError('Joint type {:s} not supported'.
            format(link.attrib['type']))

    print('Model has {:d} links and {:d} DoFs'.
          format(len(self.links), self.n_dofs))

  def get_hand_pointcloud(self):
    hand = open3d.PointCloud()
    for filename, T in self.links.items():
      # load link pointcloud
      link_name = filename.split('.')[0]
      filename = osp.join(self.geometry_dir, '{:s}.ply'.format(link_name))
      if filename not in self.link_pcs:
        self.link_pcs[filename] = open3d.read_point_cloud(filename)

      # transform the link pointcloud
      link_pc = deepcopy(self.link_pcs[filename])
      link_pc.transform(T)

      # append to hand
      hand += link_pc

    hand.transform(self.palm_pose)
    return hand

  def show(self):
    open3d.draw_geometries([self.get_hand_pointcloud()])

if __name__ == '__main__':
  rr = RobotRenderer('HumanHand20DOF.xml')
  rr.show()