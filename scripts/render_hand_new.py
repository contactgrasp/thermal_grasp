"""
Input: Values of the DoFs of the hand
Output: Rendering of the hand
"""
import numpy as np
from utils import pc_utils, tf_utils as tx
import open3d
import os.path as osp
import xml.etree.ElementTree as ET
import sympy
from utils.vis_utils import Viewer

def interpret_dh(alpha=0., a=0., theta=0., d=0.):
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


class Link(object):
  def __init__(self, filename='', min_limit=None, max_limit=None,
      d=None, a=None, alpha=None, theta=None, T=np.eye(4), children=None,
      parent_id=-1, T_parent=None):
    self.filename = filename
    self.min_limit = [-float('inf')] if min_limit is None else min_limit
    self.max_limit = [+float('inf')] if max_limit is None else max_limit
    self.d = [0.] if d is None else d
    self.a = [0.] if a is None else a
    self.alpha = [0.] if alpha is None else alpha
    self.theta = [0.] if theta is None else theta
    self.T = T
    self.T_parent = [] if T_parent is None else T_parent
    self.children = [] if children is None else children
    self.parent_id = parent_id


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
    xml_filename = osp.join(osp.expanduser(xml_dir), xml_filename)
    geometry_dir = osp.expanduser(geometry_dir)

    # contains all information about the links, populated by self.parse_hand_file()
    self.links = []

    # name of the DH param corresponding to each dof
    self.dof_types = []
    self.dof_info = []

    self.n_dofs = 77  # random number
    self.dof_vals = np.zeros(self.n_dofs)
    self.parse_hand_file(xml_filename)

    # contains the hand segment point clouds
    self.link_pcs = []
    assert len(self.links) > 0
    for l in self.links:
      link_name = l.filename.split('.')[0]
      filename = osp.join(geometry_dir, '{:s}.ply'.format(link_name))
      pc = open3d.read_point_cloud(filename)
      self.link_pcs.append(np.asarray(pc.points))
    self.link_Ts = [np.eye(4)] * len(self.links)

    # init hand
    self.set_dof_vals(np.zeros(self.n_dofs))
    self.set_palm_pose(palm_pose)

  def calc_dTs(self, ddof):
    """
    Calculates the pose changes to be applied to all segments affected by the
    ddofs
    :param ddof: delta dofs
    :return: list of pose changes. An element is None if its corresponding
    hand segment is not affected
    """
    assert len(ddof) == self.n_dofs
    dTs = [np.eye(4)] * len(self.links)

    for dof_idx, dd in enumerate(np.rad2deg(ddof)):
      link_idx = self.dof_info[dof_idx]['link_idx']
      dT_local = interpret_dh(**{self.dof_types[dof_idx]: dd})
      T_parent = self.link_Ts[self.links[link_idx].parent_id]
      T_parent = np.dot(T_parent, self.links[link_idx].T)
      joint_idx = self.dof_info[dof_idx]['joint_idx']
      for T in self.links[link_idx].T_parent[:joint_idx]:
        T_parent = np.dot(T_parent, T)
      dT_global = np.dot(np.dot(T_parent, dT_local), tx.inverse_matrix(T_parent))

      # collect children
      child_link_idx = []
      lq = [link_idx]
      while len(lq):
        li = lq.pop(0)
        child_link_idx.append(li)
        lq.extend(self.links[li].children)

      for ci in child_link_idx:
        dTs[ci] = np.dot(dTs[ci], dT_global)
    return dTs

  def _update_hand(self):
    old_link_Ts = [T.copy() for T in self.link_Ts]
    self._eval_links()
    for idx in range(len(self.link_pcs)):
      T = np.dot(self.link_Ts[idx], tx.inverse_matrix(old_link_Ts[idx]))
      self.link_pcs[idx] = pc_utils.transform_pc(self.link_pcs[idx], T)

  def set_dof_vals(self, dof_vals_new):
    """
    :param dof_vals_new:
    :return:
    """
    self.dof_vals = np.rad2deg(dof_vals_new)
    self._update_hand()

  def set_palm_pose(self, T):
    """
    sets the palm pose
    :param T: (4, 4)
    :return:
    """
    self.links[0].T = T.copy()
    self._update_hand()

  def set_hand_config(self, dof_vals=None, palm_pose=np.eye(4)):
    if dof_vals is None:
      dof_vals = np.zeros(self.n_dofs)
    self.dof_vals = np.rad2deg(dof_vals)
    self.links[0].T = palm_pose.copy()
    self._update_hand()

  def _eval_links(self):
    """
    converts the symbolic DH parameters of links to 4x4 transforms according to
    current self.dof_vals, and populates self.link_Ts
    :return:
    """
    assert len(self.dof_vals) == self.n_dofs

    self.link_Ts = []
    self.dof_info = [dict() for _ in range(self.n_dofs)]
    link_T = self.links[0].T
    for link_idx, l in enumerate(self.links):
      if l.parent_id == 0:  # base link of a chain
        link_T = np.dot(self.link_Ts[0], l.T)

      joint_idx = 0
      l.T_parent = []
      for min_limit, max_limit, d, a, alpha, theta in \
          zip(l.min_limit, l.max_limit, l.d, l.a, l.alpha, l.theta):
        n_parameters = 0
        if type(d) == str:
          d, dof_idx = self.eval_exp(d, min_limit, max_limit)
          self.dof_info[dof_idx]['link_idx'] = link_idx
          self.dof_info[dof_idx]['joint_idx'] = joint_idx
          n_parameters += 1
        if type(a) == str:
          a, dof_idx = self.eval_exp(a, min_limit, max_limit)
          self.dof_info[dof_idx]['link_idx'] = link_idx
          self.dof_info[dof_idx]['joint_idx'] = joint_idx
          n_parameters += 1
        if type(alpha) == str:
          alpha, dof_idx = self.eval_exp(alpha, min_limit, max_limit)
          self.dof_info[dof_idx]['link_idx'] = link_idx
          self.dof_info[dof_idx]['joint_idx'] = joint_idx
          n_parameters += 1
        if type(theta) == str:
          theta, dof_idx = self.eval_exp(theta, min_limit, max_limit)
          self.dof_info[dof_idx]['link_idx'] = link_idx
          self.dof_info[dof_idx]['joint_idx'] = joint_idx
          n_parameters += 1
        assert n_parameters <= 1
        joint_idx += 1

        l.T_parent.append(interpret_dh(alpha, a, theta, d))
        link_T = np.dot(link_T, l.T_parent[-1])
      self.link_Ts.append(link_T)

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
    #   print('DoF {:d} upper clamped to {:f}'.format(dof_idx, max_val))
    # elif dof_val < min_val:
    #   dof_val = min_val
    #   print('DoF {:d} lower clamped to {:f}'.format(dof_idx, min_val))

    # evaluate expression
    out = expr.subs(dof_var, dof_val)
    out = float(out)

    return out, dof_idx

  def parse_hand_file(self, filename):
    tree = ET.parse(filename)
    robot = tree.getroot()

    # read palm information
    palm_filename = robot.find('palm').text
    self.links = [Link(filename=palm_filename)]

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
          raise IOError('Wrong axis of rotation {:s}'.format(axis))
      except AttributeError:
        rotmat = np.fromstring(transform.find('rotationMatrix').text, sep=' ')
        R = np.eye(4)
        R[:3, :3] = np.reshape(rotmat, (3, 3)).T

      T = np.dot(T, R)

      # chain DH parameters
      joint_mins = []
      joint_maxs = []
      joint_as = []
      joint_ds = []
      joint_alphas = []
      joint_thetas = []
      for joint in chain.findall('joint'):
        joint_mins.append(float(joint.find('minValue').text))
        joint_maxs.append(float(joint.find('maxValue').text))

        theta = joint.find('theta').text
        if theta.startswith('d'):
          joint_thetas.append(theta)
          self.n_dofs += 1
          self.dof_types.append('theta')
        else:
          joint_thetas.append(float(theta))

        d = joint.find('d').text
        if d.startswith('d'):
          joint_ds.append(d)
          self.n_dofs += 1
          self.dof_types.append('d')
        else:
          joint_ds.append(float(d))

        a = joint.find('a').text
        if a.startswith('d'):
          joint_as.append(a)
          self.n_dofs += 1
          self.dof_types.append('a')
        else:
          joint_as.append(float(a))

        alpha = joint.find('alpha').text
        if alpha.startswith('d'):
          joint_alphas.append(alpha)
          self.n_dofs += 1
          self.dof_types.append('alpha')
        else:
          joint_alphas.append(float(alpha))

      # link information
      joint_idx = 0
      parent_id = 0
      for link in chain.findall('link'):

        link_init_T = T if joint_idx == 0 else np.eye(4)

        if link.attrib['dynamicJointType'] == 'Universal':
          l = Link(
            filename=link.text,
            min_limit=joint_mins[joint_idx:joint_idx+2],
            max_limit=joint_maxs[joint_idx:joint_idx+2],
            d=joint_ds[joint_idx:joint_idx+2],
            a=joint_as[joint_idx:joint_idx+2],
            alpha=joint_alphas[joint_idx:joint_idx+2],
            theta=joint_thetas[joint_idx:joint_idx+2],
            T=link_init_T,
            parent_id=parent_id)
          self.links.append(l)
          joint_idx += 2

        elif (link.attrib['dynamicJointType'] == 'Revolute') or\
            (link.attrib['dynamicJointType'] == 'Prismatic'):
          l = Link(
            filename=link.text,
            min_limit=joint_mins[joint_idx:joint_idx+1],
            max_limit=joint_maxs[joint_idx:joint_idx+1],
            d=joint_ds[joint_idx:joint_idx+1],
            a=joint_as[joint_idx:joint_idx+1],
            alpha=joint_alphas[joint_idx:joint_idx+1],
            theta=joint_thetas[joint_idx:joint_idx+1],
            T=link_init_T,
            parent_id=parent_id)
          self.links.append(l)
          joint_idx += 1

        else:
          raise ValueError('Joint type {:s} not supported'.
            format(link.attrib['type']))

        self.links[parent_id].children.append(len(self.links)-1)
        parent_id = len(self.links) - 1

    print('Model has {:d} links and {:d} DoFs'.
          format(len(self.links), self.n_dofs))

  def show(self):
    palm_pose = np.eye(4)
    palm_pose[0, 3] = 200
    self.set_hand_config(palm_pose=palm_pose)
    ddof_vals = \
      [ 20,74,0,0,
        0,68,0,10,
        0,71,0,0,
        0,0,0,0,
        -40,-30, 20,0]
    ddof_vals = np.deg2rad(ddof_vals)
    dTs = self.calc_dTs(ddof_vals)

    vis = Viewer()
    for link_idx, pc in enumerate(self.link_pcs):
      pc_T = pc_utils.transform_pc(self.link_pcs[link_idx], dTs[link_idx])
      pc_T = pc_utils.ndarray2pc(pc_T)
      vis.add_geometry(pc_T, 'hand')
    vis.show(block=False)


if __name__ == '__main__':
  rr = RobotRenderer('HumanHand20DOF.xml')
  rr.show()