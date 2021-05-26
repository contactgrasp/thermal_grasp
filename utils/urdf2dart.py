"""
Converts a graspit model to DART
"""
from lxml import etree as ET
import os.path as osp
import numpy as np
import argparse
from collections import deque
import rospkg
from IPython.core.debugger import set_trace

class Link:
  def __init__(self, name, visual, link_color, output_dir):
    self.name = name
    t = '0 0 0'
    r = '0 0 0'
    if visual is not None:
      origin = visual.find('origin')
      if origin is not None:
        t = origin.get('xyz', '0 0 0').strip()
        r = origin.get('rpy', '0 0 0').strip()
    self.tx, self.ty, self.tz = t.split(' ')
    self.rx, self.ry, self.rz = r.split(' ')

    geom = visual.find('geometry')
    mesh = geom.find('mesh')
    if mesh is None:
      print('Geometry {:s} does not have a mesh tag'.format(self.name))
      raise ValueError
    rospack = rospkg.RosPack()
    mesh_filename = mesh.get('filename')
    if 'package://' in mesh_filename:
      mesh_filename = mesh_filename.replace('package://', '')
      base_path = rospack.get_path(mesh_filename.split('/')[0])
      mesh_filename = osp.join(base_path,
          osp.join(*mesh_filename.split('/')[1:]))
    self.mesh_filename = osp.relpath(osp.expanduser(mesh_filename),
        start=output_dir)
    self.scale = mesh.get('scale', '1')
    self.red, self.green, self.blue = link_color

  def get_attributes(self):
    return dict(
        sx=self.scale, sy=self.scale, sz=self.scale,
        tx=self.tx, ty=self.ty, tz=self.tz,
        rx=self.rx, ry=self.ry, rz=self.rz,
        red=self.red, green=self.green, blue=self.blue,
        meshFile=self.mesh_filename, type='mesh')

  def add_transform(self, tx, ty, tz, rx, ry, rz):
    self.tx = '{:f}'.format(float(self.tx) + tx)
    self.ty = '{:f}'.format(float(self.ty) + ty)
    self.tz = '{:f}'.format(float(self.tz) + tz)
    self.rx = '{:f}'.format(float(self.rx) + rx)
    self.ry = '{:f}'.format(float(self.ry) + ry)
    self.rz = '{:f}'.format(float(self.rz) + rz)


class Joint:
  def __init__(self, joint_type, parent, child, limit, origin=None, axis=None):
    self.type = joint_type
    if joint_type == 'continuous':
      self.type = 'rotational'
    elif joint_type == 'floating' or joint_type == 'planar':
      print('Joint type {:s} not implemented'.format(joint_type))
      raise ValueError

    self.parent = parent.get('link')
    self.child = child.get('link')
    if origin is not None:
      t = origin.get('xyz', '0 0 0').strip()
      r = origin.get('rpy', '0 0 0').strip()
    else:
      t = '0 0 0'
      r = '0 0 0'
    self.tx, self.ty, self.tz = t.split(' ')
    self.rx, self.ry, self.rz = r.split(' ')
    if axis is not None:
      a = axis.get('xyz', '1, 0, 0').strip()
    else:
      a = '1 0 0'
    self.ax, self.ay, self.az = a.split(' ')
    self.name = '{:s}_{:s}'.format(self.parent, self.child)
    
    self.min = self.max = '0'
    if self.type != 'fixed': 
      self.min = limit.get('lower', '0')
      self.max = limit.get('upper', '0')

  def get_attributes(self):
    return dict(
        jointName=self.name,
        jointType='rotational',
        jointMin=self.min, jointMax=self.max)

  def get_child_tags(self):
    pos  = ET.Element('position', attrib={'x': self.tx, 'y': self.ty,
      'z': self.tz})
    ori  = ET.Element('orientation', attrib={'x': self.rx, 'y': self.ry,
      'z': self.rz})
    axis = ET.Element('axis', attrib={'x': self.ax, 'y': self.ay,
      'z': self.az})
    return pos, ori, axis
    

def convert(input_filename, output_filename):
  """
  :param input_filename:
  :param output_filename: output XML filename for DART format file
  :return:
  """
  output_dir = osp.split(osp.abspath(output_filename))[0]
  g_attrib = {
    'sx': '1e-3', 'sy': '1e-3', 'sz': '1e-3',
    'tx': '0', 'ty': '0', 'tz': '0',
    'rx': '0', 'ry': '0', 'rz': '0',
    'meshFile': '{:s}', 'type': 'mesh'}
  f_attrib = {'jointName': '', 'jointType': 'rotational', 'jointMin': '0',
    'jointMax': str(np.pi)}

  link_colors = {}
  link_colors['palm'] = deque([['83', '147', '186']])
  link_colors['index'] = deque([
    ['22', '133', '97'],
    ['32', '191', '139'],
    ['111', '231', '192']])
  link_colors['mid'] = deque([
    ['80', '63', '113'],
    ['110', '87', '157'],
    ['147', '128', '184']])
  link_colors['ring'] = deque([
    ['149', '98', '44'],
    ['198', '132', '63'],
    ['210', '157', '102']])
  link_colors['pinky'] = deque([
    ['233', '217', '61'],
    ['241', '232', '137'],
    ['247', '241', '183']])
  link_colors['thumb'] = deque([
    ['114', '60', '80'],
    ['158', '84', '112'],
    ['217', '113', '138']])

  d_hand = ET.Element('model', attrib={'version': '1'})
  u_hand = ET.parse(input_filename).getroot()

  # stores all link info
  links = {}
  for link in u_hand.findall('link'):
    link_name = link.get('name')
    link_color = ['{:d}'.format(c) for c in np.random.randint(256, size=3)]
    for colors_name in link_colors.keys():
      if colors_name in link_name.lower():
        # cycle through the colors for this finger
        link_color = link_colors[colors_name].popleft()
        link_colors[colors_name].append(link_color)
        break
    l = Link(link_name, link.find('visual'), link_color, output_dir)
    links[l.name] = l

  # stores all joint info
  joints = {}
  parents = {}
  children = {}
  for joint in u_hand.findall('joint'):
    j = Joint(joint.get('type'), joint.find('parent'), joint.find('child'),
        joint.find('limit'), joint.find('origin'), joint.find('axis'))
    joints[j.name] = j
    parents[j.child] = j.parent
    if j.parent not in children:
      children[j.parent] = [j.child]
    else:
      children[j.parent].append(j.child)

  # determine root
  for link_name in links.keys():
    if link_name not in parents:
      root_link = link_name
      break

  # traverse the tree - BFS
  q = deque()
  q.append((root_link, d_hand))
  while len(q):
    link_name, parent_tag = q.popleft()

    ET.SubElement(parent_tag, 'geom', attrib=links[link_name].get_attributes())
    
    # create frames for which this link is parent
    if link_name not in children:
      continue
    for child_link_name in children[link_name]:
      joint_name = '_'.join([link_name, child_link_name])
      joint = joints[joint_name]
      if joint.type == 'fixed':  # don't create a new frame tag
        joint_tag = parent_tag
        # add the joint's and parent link's transform to the child link
        tx = float(joint.tx) + float(links[joint.parent].tx)
        ty = float(joint.ty) + float(links[joint.parent].ty)
        tz = float(joint.tz) + float(links[joint.parent].tz)
        rx = float(joint.rx) + float(links[joint.parent].rx)
        ry = float(joint.ry) + float(links[joint.parent].ry)
        rz = float(joint.rz) + float(links[joint.parent].rz)
        links[joint.child].add_transform(tx, ty, tz, rx, ry, rz)
      else:
        joint_tag = ET.SubElement(parent_tag, 'frame', attrib=joint.get_attributes())
        for t in joint.get_child_tags():
          joint_tag.append(t)
      q.append((joint.child, joint_tag))

  # write out XML
  tree = ET.ElementTree(d_hand)
  tree.write(output_filename, pretty_print=True, xml_declaration=True)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_filename', required=True)
  parser.add_argument('--output_filename', required=True)
  args = parser.parse_args()

  convert(osp.expanduser(args.input_filename),
      osp.expanduser(args.output_filename))
  print('Written to {:s}'.format(args.output_filename))
