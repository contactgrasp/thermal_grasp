"""
Converts a graspit model to DART
"""
from lxml import etree as ET
import os.path as osp
import numpy as np
import tf_utils
import sympy
import transforms3d.euler as txe
from IPython.core.debugger import set_trace

def eval_exp(s):
  expr = sympy.sympify(s)
  var = list(expr.free_symbols)[0]
  coeff = float(expr.coeff(var))
  const = float(expr.subs(var, 0))
  return const, coeff

def convert(input_filename, output_dir, data_dir='meshes', offsets_dir=None,
    mesh_ext='ply'):
  output_dir = osp.expanduser(output_dir)
  data_dir = osp.relpath(data_dir, start=output_dir)
  g_attrib = {
    'sx': '1e-3', 'sy': '1e-3', 'sz': '1e-3',
    'tx': '0', 'ty': '0', 'tz': '0',
    'rx': '0', 'ry': '0', 'rz': '0',
    'meshFile': '{:s}', 'type': 'mesh'}
  f_attrib = {'jointName': '', 'jointType': 'rotational', 'jointMin': '0',
    'jointMax': str(np.pi)}

  link_colors = {
    'palm':   {'red': '83', 'green': '147', 'blue': '186'},
    'index1': {'red': '22', 'green': '133', 'blue': '97'},
    'index2': {'red': '32', 'green': '191', 'blue': '139'},
    'index3': {'red': '111', 'green': '231', 'blue': '192'},
    'mid1':   {'red': '80', 'green': '63', 'blue': '113'},
    'mid2':   {'red': '110', 'green': '87', 'blue': '157'},
    'mid3':   {'red': '147', 'green': '128', 'blue': '184'},
    'ring1':  {'red': '149', 'green': '98', 'blue': '44'},
    'ring2':  {'red': '198', 'green': '132', 'blue': '63'},
    'ring3':  {'red': '210', 'green': '157', 'blue': '102'},
    'pinky1': {'red': '233', 'green': '217', 'blue': '61'},
    'pinky2': {'red': '241', 'green': '232', 'blue': '137'},
    'pinky3': {'red': '247', 'green': '241', 'blue': '183'},
    'thumb1': {'red': '114', 'green': '60', 'blue': '80'},
    'thumb2': {'red': '158', 'green': '84', 'blue': '112'},
    'thumb3': {'red': '217', 'green': '113', 'blue': '138'},
    'link1' : {'red': '22', 'green': '133', 'blue': '97'},
    'link2' : {'red': '80', 'green': '63', 'blue': '113'},
    'link3' : {'red': '149', 'green': '98', 'blue': '44'},
  }

  hand = ET.Element('model', attrib={'version': '1'})
  g_hand = ET.parse(input_filename).getroot()

  # palm
  link_name = 'palm'
  mesh_filename = osp.join(data_dir, '{:s}.{:s}'.format(link_name, mesh_ext))
  attrib = dict(g_attrib, meshFile=mesh_filename, **link_colors[link_name])
  if offsets_dir is not None:
    offset_filename = osp.join(offsets_dir, link_name, 'offset.txt')
    ox, oy, oz = np.loadtxt(offset_filename)
    attrib = dict(attrib, tx=str(ox/1000.0), ty=str(oy/1000.0), tz=str(oz/1000.0))
  ET.SubElement(hand, 'geom', attrib=attrib)

  chain_idx = 0
  for chain in g_hand.findall('chain'):

    joint_idx = 0
    f_handles = []
    parent = hand
    for joint in chain.findall('joint'):
      joint_name = '{:d}_{:d}'.format(chain_idx, joint_idx)
      min_offset = 0
      max_offset = 0
      # if (joint_idx == 0) and (chain_idx == 4):  # first joint in thumb
      #   min_offset = -70
      joint_min = str(np.deg2rad(float(joint.find('minValue').text)+min_offset))
      joint_max = str(np.deg2rad(float(joint.find('maxValue').text)+max_offset))
      tx = ty = tz = rx = ry = rz = ax = ay = az = 0
      if joint_idx == 0:
        init_T = chain.find('transform')
        for trans in init_T.findall('translation'):
          x, y, z = np.fromstring(trans.text, sep=' ')
          tx += x
          ty += y
          tz += z
        R = np.eye(3)
        for rot in init_T.findall('rotation'):
          angle, axis = rot.text.split(' ')
          angle = np.deg2rad(float(angle))
          x = y = z = 0
          if axis.upper() == 'X':
            x = angle
          elif axis.upper() == 'Y':
            y = angle
          elif axis.upper() == 'Z':
            z = angle
          else:
            raise IOError('Wrong axis of rotation {:s}'.format(axis))
          R = np.dot(R, txe.euler2mat(x, y, z))
        rx, ry, rz = txe.mat2euler(R)
        if init_T.get('rotationMatrix') is not None:
          rotmat = np.fromstring(init_T.find('rotationMatrix').text, sep=' ')
          R = np.eye(4)
          R[:3, :3] = np.reshape(rotmat, (3, 3)).T
          _, _, _, rx, ry, rz = tf_utils.T2xyzrpy(R)

      dh_theta = joint.find('theta').text
      try:
        dh_theta = float(dh_theta)
      except ValueError:
        joint_type = 'rotational'
        dh_theta, az = eval_exp(dh_theta)
      dh_theta = np.deg2rad(dh_theta)

      dh_alpha = joint.find('alpha').text
      try:
        dh_alpha = float(dh_alpha)
      except ValueError:
        joint_type = 'rotational'
        dh_alpha, ax = eval_exp(dh_alpha)
      dh_alpha = np.deg2rad(dh_alpha)

      dh_d = joint.find('d').text
      try:
        dh_d = float(dh_d)
      except ValueError:
        joint_type = 'prismatic'
        dh_d, az = eval_exp(dh_d)

      dh_a = joint.find('a').text
      try:
        dh_a = float(dh_a)
      except ValueError:
        joint_type = 'prismatic'
        dh_a, ax = eval_exp(dh_a)

      attrib = dict(f_attrib, jointName=joint_name, jointMin=joint_min,
        jointMax=joint_max, jointType=joint_type)
      parent = (ET.SubElement(parent, 'frame', attrib=attrib))
      f_handles.append(parent)
      ET.SubElement(parent, 'position',
        attrib={'x': str(tx/1000.0), 'y': str(ty/1000.0), 'z': str(tz/1000.0)})
      ET.SubElement(parent, 'orientation',
        attrib={'x': str(rx), 'y': str(ry), 'z': str(rz)})
      ET.SubElement(parent, 'axis',
        attrib={'x': str(ax), 'y': str(ay), 'z': str(az)})
      ET.SubElement(parent, 'dh_offset',
        attrib={'a': str(dh_a/1000.0), 'alpha': str(dh_alpha), 'd': str(dh_d/1000.0),
          'theta': str(dh_theta)})

      joint_idx += 1

    joint_idx = -1
    for link in chain.findall('link'):
      link_name = link.text.split('.')[0]

      if link.attrib['dynamicJointType'] == 'Universal':
        joint_idx += 2
      elif (link.attrib['dynamicJointType'] == 'Revolute') or \
           (link.attrib['dynamicJointType'] == 'Prismatic'):
        joint_idx += 1
      parent = f_handles[joint_idx]

      mesh_filename = osp.join(data_dir, '{:s}.{:s}'.format(link_name, mesh_ext))
      attrib = dict(g_attrib, meshFile=mesh_filename, **link_colors[link_name])
      if offsets_dir is not None:
        offset_filename = osp.join(offsets_dir, link_name, 'offset.txt')
        ox, oy, oz = np.loadtxt(offset_filename)
        attrib = dict(attrib, tx=str(ox/1000.0), ty=str(oy/1000.0),
          tz=str(oz/1000.0))
      ET.SubElement(parent, 'geom', attrib=attrib)
    chain_idx += 1

  tree = ET.ElementTree(hand)
  _, output_filename = osp.split(input_filename)
  output_filename = osp.join(output_dir, output_filename)
  tree.write(output_filename, pretty_print=True, xml_declaration=True)


if __name__ == '__main__':
  graspit_dir = osp.join('..', 'data', 'graspit')
  input_filename = osp.join(graspit_dir, 'models', 'robots', 'Barrett', 'Barrett.xml')
  data_dir = osp.join('..', 'data', 'barrett')
  output_dir = osp.join('~', 'research', 'dart', 'models', 'Barrett')
  # offsets_dir = osp.join('..', 'data', 'human_hand_meshes')
  mesh_ext = 'ply'
  convert(input_filename, output_dir, data_dir=data_dir, mesh_ext=mesh_ext)
  print('Written to {:s}'.format(output_dir))
