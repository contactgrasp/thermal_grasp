from torch.utils.data import Dataset
import os
from thirdparty import binvox_rw
from dataset_tools.create_coarse_dataset import CoarseGraspCollector
import numpy as np
from utils import tf_utils
osp = os.path

class CoarseGraspDataset(Dataset):
  def __init__(self, objects, actions, grid_size=64, *args, **kwargs):
    """
    PyTorch dataset loader for coarse grasps.
    Loads (voxel grid, rotated voxel grid, grasp) tuples
    :param objects: List of objects to be included in this dataset
    :param action: List of actions. Each must be one of ['handoff', 'use']
    """
    super(CoarseGraspDataset, self).__init__()
    self.grid_size = grid_size
    cg_collector = CoarseGraspCollector(*args, **kwargs)
    self.base_dir = osp.expanduser(cg_collector.base_dir)
    self.grasps = [cg_collector.get_coarse_grasps(object_name, action)
      for object_name,action in zip(objects, actions)]
    for g in self.grasps:  # divide angles by 90 to get integer angle labels
      g[:, 3] /= 90
    self.objects = [self._load_object(object_name)
      for object_name in objects]
    # dataset is N_objects x N_grasps_per_object
    self.n_objects = len(objects)
    self.n_grasps_per_object = cg_collector.n_preds

  def _load_object(self, object_name):
    voxel_filename = osp.join(self.base_dir, 'models',
      '{:s}.binvox'.format(object_name))
    with open(voxel_filename, 'rb') as f:
      model = binvox_rw.read_as_3d_array(f)
    x, y, z = np.nonzero(model.data)
    obj = np.zeros((self.grid_size, self.grid_size, self.grid_size),
      dtype=np.bool)
    dim = model.data.shape[0]
    offset = int((self.grid_size - dim) / 2)
    obj[z+offset, y+offset, x+offset] = True
    return obj

  @staticmethod
  def _rotate_object(obj, v):
    """
    rotates the object voxel grid so that +Z axis is aligned with vector v
    :param obj: voxel grid
    :param v: (3, )
    :return: voxel grid of same size
    """
    v = np.asarray(v) / np.linalg.norm(v)
    R = tf_utils.rotmat_from_vecs(v, [0, 0, 1])[:3, :3].T
    z, y, x = np.nonzero(obj)
    p = np.vstack((x, y, z))
    o = np.asarray(obj.shape, dtype=int)[:, np.newaxis] // 2
    p -= o
    p = (R @ p).astype(int)
    p += o
    x, y, z = p
    robj = np.zeros(obj.shape, dtype=np.bool)
    robj[z, y, x] = True

    return robj

  def __len__(self):
    return self.n_objects * self.n_grasps_per_object

  def __getitem__(self, idx):
    """
    :param idx:
    :return: (object, rotated object, grasp)
    rotated object = object rotated so that Z axis is aligned with surface
    normal at the ground truth grasp point
    grasp format = [pt_x, pt_y, pt_z, angle, distance]
    """
    object_idx, grasp_idx = divmod(idx, self.n_grasps_per_object)

    grasp = self.grasps[object_idx][grasp_idx]
    grasp, n = grasp[:5], grasp[5:]

    obj  = self.objects[object_idx]
    robj = self._rotate_object(obj, n)

    obj = obj[np.newaxis, ...].astype(np.float32)
    robj = robj[np.newaxis, ...].astype(np.float32)

    return obj, robj, grasp


def test_object_rotation():
  import open3d

  d = CoarseGraspDataset(['camera'], ['use'])
  obj, _, _ = d[0]

  obj = d._rotate_object(obj, [1, 0, 0])

  z, y, x = np.nonzero(obj)
  pc = open3d.PointCloud()
  pc.points = open3d.Vector3dVector(np.vstack((x, y, z)).T)
  frame = open3d.create_mesh_coordinate_frame(size=80, origin=[0,0,0])
  open3d.draw_geometries([pc, frame])


def test_binvox():
  import open3d

  binvox_filename = osp.join('~', 'deepgrasp_data', 'models', 'camera.binvox')
  binvox_filename = osp.expanduser(binvox_filename)
  with open(binvox_filename, 'rb') as f:
    model = binvox_rw.read_as_3d_array(f)
    x, y, z = np.nonzero(model.data)

  pc = open3d.PointCloud()
  pc.points = open3d.Vector3dVector(np.vstack((x, y, z)).T)
  frame = open3d.create_mesh_coordinate_frame(size=80, origin=[0,0,0])
  open3d.draw_geometries([frame, pc])


if __name__ == '__main__':
  test_object_rotation()
  # test_binvox()