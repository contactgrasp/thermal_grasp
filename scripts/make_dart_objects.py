"""
Generates .xml object files for DART
"""
import init_paths
import os
from contact_heatmaps_ml.utils import object_names
import subprocess
from lxml import etree as ET
from IPython.core.debugger import set_trace
osp = os.path


def make_objects(models_dir=osp.join('~', 'deepgrasp_data', 'models'),
    output_dir=osp.join('~', 'research', 'dart', 'models', 'object_models'),
    scale_factor=1.15):
  """
  :param scale_factor: the scale factor for the object, used while sampling grasps
  (see collect_surface_grasps.py
  """
  models_dir = osp.expanduser(models_dir)
  output_dir = osp.expanduser(output_dir)
  scale_factor = '{:.4f}'.format(scale_factor)
  for object_name in object_names:
    output_filename = osp.join(output_dir, '{:s}.xml'.format(object_name))

    # generate object.xml
    model = ET.Element('model')
    attrib = {
        'type': 'mesh',
        'sx': scale_factor, 'sy': scale_factor, 'sz': scale_factor,
        'tx': '0', 'ty': '0', 'tz': '0',
        'rx': '0', 'ry': '0', 'rz': '0',
        'red': '240', 'green': '240', 'blue': '240',
        'meshFile': '{:s}.ply'.format(object_name)
        }
    ET.SubElement(model, 'geom', attrib=attrib)
    tree  = ET.ElementTree(model)
    tree.write(output_filename, pretty_print=True, xml_declaration=True)
    print('Written object file {:s}'.format(output_filename))



if __name__ == '__main__':
  make_objects()
