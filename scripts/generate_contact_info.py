"""
saves a list of contact or no-contact points
"""
import init_paths
import open3d
import numpy as np
import argparse
import os
from utils import dataset_utils
from IPython.core.debugger import set_trace
osp = os.path


def save_contact_info(object_name, session_num, instruction, output_dir,
    sigmoid_k=10.0, thresh=0.4, scale_factor=1.15):
  data_dir = getattr(dataset_utils, '{:s}_data_dirs'.format(instruction))\
      [int(session_num)-1]
  session_name = 'full{:s}_{:s}'.format(session_num, instruction)
  object_name = object_name.split('-')[0]
  try:
    mesh_filename = dataset_utils.get_session_mesh_filenames(session_name,
        data_dir)[object_name]
  except KeyError:
    print('Session {:s} does not have {:s}'.format(session_name, object_name))
    return

  output_filename = '{:s}_{:s}_{:s}_contact_info.txt'.format(session_num,
      instruction, object_name)
  output_filename = osp.join(output_dir, output_filename)

  obj = open3d.read_triangle_mesh(mesh_filename)
  obj.transform(scale_factor * np.eye(4))
  assert obj.has_vertex_colors()
  obj.compute_vertex_normals(normalized=True)

  ops = np.asarray(obj.vertices)
  ons = np.asarray(obj.vertex_normals)
  ocs = np.asarray(obj.vertex_colors)

  contact = dataset_utils.texture_proc(ocs[:, 0], k=sigmoid_k)
  contact = dataset_utils.discretize_texture(contact, thresh=thresh)
  contact = contact.astype(int)
  contact[contact == 2] = 0  # unseen points don't have any contact
  # contact = np.logical_and(contact, ops[:, 0] < 0)
  
  fmt = '%d ' + '%.18e '*6
  data = np.hstack((contact[:, np.newaxis], ops, ons))
  np.savetxt(output_filename, data, fmt=fmt)
  print('{:s} written with {:d} contact points and {:d} non-contact points'.
    format(output_filename, np.sum(contact), np.sum(np.logical_not(contact))))


def save_all_contact_info(object_names, session_nums, instruction, output_dir,
    sigmoid_k, thresh):
  for object_name in object_names:
    for session_num in session_nums:
      save_contact_info(object_name, session_num, instruction, output_dir,
          sigmoid_k, thresh)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--object_names', help='comma separated', default=None)
  parser.add_argument('--session_nums', help='comma separated', default=None)
  parser.add_argument('--instruction', required=True)
  parser.add_argument('--output_dir', default=osp.join('..', 'data', 'grasps'))
  parser.add_argument('--sigmoid_k', default=10.0)
  parser.add_argument('--thresh', default=0.3)
  args = parser.parse_args()

  object_names = args.object_names
  if object_names is not None:
    object_names = object_names.split(',')
  else:
    object_names = getattr(dataset_utils, '{:s}_objects'.format(args.instruction))

  session_nums = args.session_nums
  if session_nums is not None:
    session_nums = session_nums.split(',')
  else:
    session_nums = ['{:d}'.format(idx) for idx in range(1, 51)]

  save_all_contact_info(object_names, session_nums, args.instruction,
      args.output_dir, sigmoid_k=float(args.sigmoid_k),
      thresh=float(args.thresh))

