import scipy.ndimage.morphology as scm
import os
import subprocess
import argparse
import sys
from thirdparty import binvox_rw
import numpy as np
osp = os.path

def voxelize_models(data_dir, fmt, grid_size):
  for mesh_filename in os.listdir(data_dir):
    if fmt not in mesh_filename:
      continue
    print('Converting ', mesh_filename)

    mesh_filename = osp.join(data_dir, mesh_filename)
    args = osp.join('..', 'thirdparty', 'binvox')
    args += ' -cb -d {:d} '.format(grid_size)
    args += mesh_filename

    try:
      subprocess.check_call(args, shell=True)
    except subprocess.CalledProcessError as e:
      print(e)
    print('Done')

    vox_filename = mesh_filename.split('.')[0] + '.binvox'
    with open(vox_filename, 'rb') as f:
      model = binvox_rw.read_as_3d_array(f)

    # fill holes
    scm.binary_fill_holes(model.data.copy(), output=model.data)
    with open(vox_filename, 'wb') as f:
      model.write(f)


  # TODO: write code for saving object min and max XYZ bounds


if __name__ == '__main__':
  parser = argparse.ArgumentParser(sys.argv[0])
  parser.add_argument('--data_dir', required=True,
    help='Directory containing mesh files', type=str)
  parser.add_argument('--format', default='ply',
    help='Extension of the mesh filenames')
  parser.add_argument('--grid_size', type=int, default=64,
    help='Voxel grid size')
  args = parser.parse_args()

  # explanation for grid size:
  # the voxel grid will be rotated during training
  # maximum rotated side length = orig side length * sqrt(2)

  model_grid_size = int(args.grid_size / np.sqrt(2))
  print('Using grid size {:d}'.format(model_grid_size))

  voxelize_models(osp.expanduser(args.data_dir), args.format, model_grid_size)