"""
Generates .wrl object models and makes the .xml graspit world files
for all the objects in the dataset
"""
import init_paths
import os
from contact_heatmaps_ml.utils import object_names
import subprocess
from lxml import etree as ET
from IPython.core.debugger import set_trace
osp = os.path


def make_worlds(hand_name, meshlab_template='wrl_convert.mlx',
    models_dir=osp.join('~', 'deepgrasp_data', 'models'),
    graspit_dir=osp.join('..', 'data', 'graspit'), create_objects=False):
  meshlab_cmd = 'meshlab.meshlabserver -i {:s} -o {:s} -s {:s}'.format('{:s}',
      '{:s}', meshlab_template)
  models_dir = osp.expanduser(models_dir)
  for object_name in object_names:
    if create_objects:
      input_filename = osp.join(models_dir, '{:s}.ply'.format(object_name))
      output_filename = osp.join(graspit_dir, 'models', 'objects',
          '{:s}_mm.wrl'.format(object_name))
      cmd = meshlab_cmd.format(input_filename, output_filename)
      proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
          shell=True)
      stdoutdata, stderrdata = proc.communicate()
      print(stdoutdata, stderrdata)

      # generate object.xml
      root = ET.Element('root')
      ET.SubElement(root, 'material').text = 'rubber'
      ET.SubElement(root, 'mass').text = '500'
      ET.SubElement(root, 'geometryFile', attrib={'type': 'Inventor'}).text = \
          '{:s}_mm.wrl'.format(object_name)
      tree  = ET.ElementTree(root)
      obj_xml_filename = osp.join(graspit_dir, 'models', 'objects',
          '{:s}.xml'.format(object_name))
      tree.write(obj_xml_filename, pretty_print=True, xml_declaration=True)
      print('Written object file {:s}'.format(obj_xml_filename))

    # generate world file
    world = ET.Element('world')
    gbody = ET.SubElement(world, 'graspableBody')
    ET.SubElement(gbody, 'filename').text = \
        osp.join('models', 'objects', '{:s}.xml'.format(object_name))
    transform = ET.SubElement(gbody, 'transform')
    ET.SubElement(transform, 'fullTransform').text = '(+1 +0 +0 +0)[+0 +0 +0]'
    robot = ET.SubElement(world, 'robot')
    if hand_name == 'barrett':
      ET.SubElement(robot, 'filename').text = \
          'models/robots/Barrett/Barrett.xml'
    elif hand_name == 'allegro':
      ET.SubElement(robot, 'filename').text = \
          'models/robots/allegro/allegro.xml'
    elif hand_name == 'human':
      ET.SubElement(robot, 'filename').text = \
          'models/robots/HumanHand/HumanHand20DOF.xml'
    else:
      raise NotImplementedError
    ET.SubElement(robot, 'dofValues').text = \
        '+0 +0 +0 +0 +0 +0 +0 +0 +0 +0 +0 +0 +0 +0 +0 +0 +0 +0 +0 +0'
    transform = ET.SubElement(robot, 'transform')
    ET.SubElement(transform, 'fullTransform').text = \
        '(+0.698782 +0 -0.715335 +0)[+127.89 +5.36384 +362.1]'
    camera = ET.SubElement(world, 'camera')
    ET.SubElement(camera, 'position').text = '+14.5143 -1228.54 +256.164'
    ET.SubElement(camera, 'orientation').text = \
        '+0.644056 -0.0562009 +0.0552705 +0.760907'
    ET.SubElement(camera, 'focalDistance').text = '+1256.04'
    tree = ET.ElementTree(world)
    if hand_name == 'barrett':
      world_filename = 'barrett_{:s}.xml'.format(object_name)
    elif hand_name == 'allegro':
      world_filename = 'allegro_{:s}.xml'.format(object_name)
    elif hand_name == 'human':
      world_filename = 'hand_{:s}.xml'.format(object_name)
    else:
      raise NotImplementedError
    world_filename = osp.join(graspit_dir, 'worlds', world_filename)
    tree.write(world_filename, pretty_print=True, xml_declaration=True)
    print('Written world file {:s}'.format(world_filename))


if __name__ == '__main__':
  # barrett, human, allegro
  hand_name = 'allegro'
  make_worlds(hand_name, create_objects=False)
