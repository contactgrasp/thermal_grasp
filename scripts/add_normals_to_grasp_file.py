"""
Temp file to add surface normals at grasp points to the existing CSV grasp file
"""
import numpy as np
import open3d
import os.path as osp

scale_factor = 1.15
object_name = 'binoculars'
object_filename = osp.join('~', 'deepgrasp_data', 'models',
  '{:s}.ply'.format(object_name))
object_filename = osp.expanduser(object_filename)
obj = open3d.read_triangle_mesh(object_filename)
obj.transform(scale_factor * np.eye(4))
obj.compute_vertex_normals(normalized=True)
object_normals = np.asarray(obj.vertex_normals)
tree = open3d.KDTreeFlann(obj)

data_filename = osp.join('~', 'deepgrasp_data', 'grasps',
  '{:s}.csv'.format(object_name))
data_filename = osp.expanduser(data_filename)
output_filename = 'test.csv'
with open(data_filename, 'r') as f:
  csv_header = f.readline()
csv_header = csv_header[2:-1]
data = np.loadtxt(data_filename, delimiter=',')

prev_pt = np.zeros(3)
normals = np.zeros((len(data), 3))
for idx, pt in enumerate(data[:, -5:-2]):

  if idx % 100 == 0:
    print('{:d} / {:d}'.format(idx, len(data)))

  if np.allclose(pt, prev_pt):
    normals[idx] = normals[idx-1]
    continue

  [k, nbr_idx, dist] = tree.search_knn_vector_3d(pt, 1)
  normals[idx] = object_normals[nbr_idx]

  prev_pt = pt

data = np.hstack((data, normals))
csv_header += ',normal_x,normal_y,normal_z'
np.savetxt(output_filename, data, delimiter=',', header=csv_header)
print('{:s} written'.format(output_filename))