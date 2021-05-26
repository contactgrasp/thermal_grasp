"""
Generates artificial contact heatmap, given object model and hand pose
"""

import numpy as np
import os
import open3d
from scripts.render_hand_new import RobotRenderer
import pickle
from utils import pc_utils, tf_utils as tx
from utils.vis_utils import Viewer
from collections import OrderedDict
import multiprocessing as mp

osp = os.path


def heatmap_iou(h0, h1):
  assert len(h0) == len(h1)
  idx = np.logical_or(h0 > 0, h1 > 0)
  iou = np.mean(np.minimum(h0[idx], h1[idx]) / np.maximum(h0[idx], h1[idx]))
  return iou


class HeatmapGenerator(object):
  def __init__(self, object_filename, tsdf_filename, hand_xml_filename,
      heatmap_grid_size=64, heatmap_grid_size_mm=150, palm_pose=np.eye(4),
      t_p=10, t_n=10, render_only=False):
    """
    :param object_filename: object pointcloud filename
    :param tsdf_filename: .vti file storing the TSDF
    :param heatmap_grid_size: number of cells along an edge of the heatmap volume
    :param heatmap_grid_size_mm: size of the heatmap volume in mm
    :param hand_xml_filename: hand XML filename
    :param palm_pose: pose of the hand palm in the world
    :param t_n: -ve distance value which is mapped to intensity = 1 (mm)
    :param t_p: +ve distance value which is mapped to intensity = 0 (mm)
    :param render_only: set to True if you want to use this class only for
    rendering grasps
    """
    self.hr = RobotRenderer(hand_xml_filename)
    self.hand_tidx = []
    self.render_only = render_only

    object = open3d.read_point_cloud(object_filename)
    object.transform(1000*np.eye(4))  # convert object to mm

    if self.render_only:
      return

    open3d.estimate_normals(object)
    open3d.orient_normals_towards_camera_location(object, np.zeros(3))
    assert object.has_colors()

    # load TSDF
    print('Loading object TSDF...')
    self.tsdf, self.tsdf_cell_sizes = pc_utils.ndarray_from_vti(tsdf_filename)
    print('Done.')
    # convert TSDF to mm (it is always centered at the origin)
    self.tsdf *= 1000
    self.tsdf_cell_sizes = np.asarray(self.tsdf_cell_sizes) * 1000
    self.tsdf_min_point = -np.asarray(self.tsdf.shape)/2.0 *\
                             self.tsdf_cell_sizes
    self.tsdf_max_point = -self.tsdf_min_point
    # scale and shift the TSDF values into a [0, 1] range
    self.tsdf[np.isnan(self.tsdf)] = float('inf')
    self.tsdf = -(self.tsdf + t_n) / (t_p + t_n) + 1.0
    self.tsdf = np.clip(self.tsdf, a_min=0, a_max=1)

    print('Initializing data structures...')
    self.heatmap_cell_size = float(heatmap_grid_size_mm) / heatmap_grid_size
    ops = np.asarray(object.points)
    ons = np.asarray(object.normals)
    ocs = np.asarray(object.colors)[:, 0]
    ocs -= np.nanmin(ocs)  # remap ocs to [0, 1]
    ocs /= np.nanmax(ocs)
    origin = np.mean(ops, axis=0)
    self.heatmap_min_point = origin - heatmap_grid_size_mm/2.0*np.ones(3)
    self.heatmap_max_point = origin + heatmap_grid_size_mm/2.0*np.ones(3)
    valid_idx = np.all(np.logical_and(ops >= self.heatmap_min_point,
      ops <= self.heatmap_max_point), axis=1)
    assert np.all(valid_idx), 'Some object points are outside heatmap boundary'

    # object voxels
    ovs = [self.xyz2vidx(op) for op in ops]
    self.ovs, uidx = np.unique(ovs, return_inverse=True, axis=0)

    # map from object voxels to indices in the object pointcloud
    v2oi = {tuple(self.ovs[i]): np.where(uidx==i)[0] for i in range(len(self.ovs))}

    # colors at object voxels
    self.ocs = np.asarray([np.mean(ocs[v2oi[tuple(ov)]]) for ov in self.ovs])
    self.eps = 1e-5
    self.ocs[self.ocs < 0.5] = 0
    self.t_vol = np.asarray([[ov[0], ov[1], ov[2], oc] for ov, oc in
      zip(self.ovs, self.ocs)])

    # map from object voxel to list of normal voxels in the TSDF space
    n_normal_points = 21
    assert n_normal_points % 2

    v2onvs = {}
    onv_set = []  # set of all onvs
    for ov in self.ovs:
      ov = tuple(ov)
      opss = ops[v2oi[ov]]
      onss = ons[v2oi[ov]]

      ov_onv_set = [ov]
      for op, on in zip(opss, onss):
        onps = op + np.linspace(-t_p, t_n, n_normal_points)[:, np.newaxis]*on
        keep_idx = np.all(np.logical_and(onps >= self.heatmap_min_point,
          onps <= self.heatmap_max_point), axis=1)
        onps = onps[keep_idx]
        onvs = [self.xyz2vidx(onp) for onp in onps]
        ov_onv_set.extend(onvs)
        onv_set.extend(onvs)
      ov_onv_set = set(ov_onv_set)
      if len(ov_onv_set):
        v2onvs[ov] = np.asarray(list(ov_onv_set))
      else:
        raise ValueError('No TSDF voxels found along normal for object voxel')
    onv_set = set(onv_set)

    onv2tidx = {onv: self.vidx2tidx(np.asarray(onv)) for onv in list(onv_set)}
    self.v2tvs = OrderedDict()
    for ov in self.ovs:
      tidx = np.vstack([onv2tidx[tuple(onv)] for onv in v2onvs[tuple(ov)]])
      self.v2tvs[tuple(ov)] = np.ravel_multi_index(np.fliplr(tidx).T,
        self.tsdf.shape)

    # initialize hand
    self.set_hand_config(palm_pose=palm_pose)
    print('Done.')


  def set_hand_config(self, dof_vals=None, palm_pose=np.eye(4)):
    self.hr.set_hand_config(dof_vals=dof_vals, palm_pose=palm_pose)
    if self.render_only:
      return
    # indices into TSDF for hand segments
    self.hand_tidx = [self.hp2tidx(hps) for hps in self.hr.link_pcs]

  def vidx2tidx(self, v):
    """
    returns a list of TSDF volume indices which are enclosed by the given
    heatmap voxel
    :param v: heatmap voxel (vi, vj, vk)
    :return: N x 3
    """
    # get voxel boundary points in the TSDF coordinate frame
    min_pt = (v - 0.5) * self.heatmap_cell_size + self.heatmap_min_point
    max_pt = (v + 0.5) * self.heatmap_cell_size + self.heatmap_min_point
    min_pt = self.xyz2tidx(min_pt)
    max_pt = self.xyz2tidx(max_pt)
    ti, tj, tk = np.meshgrid(range(min_pt[0], max_pt[0]),
      range(min_pt[1], max_pt[1]), range(min_pt[2], max_pt[2]))
    tidx = np.stack((ti, tj, tk), axis=3).reshape((-1, 3))
    tidx = np.maximum(np.minimum(tidx, self.tsdf.shape[0]), 0)
    return tidx

  def xyz2tidx(self, p):
    """
    converts 3D point to voxel index in the TSDF
    IMPORTANT: for efficiency, assumes that p is within the TSDF volume!
    :param p: (x, y, z)
    :return: (i, j, k)
    """
    tidx = (p - self.tsdf_min_point) / self.tsdf_cell_sizes
    tidx = tuple(tidx.astype(np.int))
    return tidx

  def xyz2vidx(self, p):
    """
    converts 3D point to voxel index in the heatmap
    IMPORTANT: for efficiency, assumes that p is within the heatmap volume!
    :param p: (x, y, z)
    :return: (i, j, k)
    """
    vidx = (p - self.heatmap_min_point) / self.heatmap_cell_size
    vidx = tuple(vidx.astype(np.int))
    return vidx

  def hp2tidx(self, hps):
    """
    generates linear indices into TSDF from hand points
    :param hps:
    :return:
    """
    keep_idx = np.all(np.logical_and(hps >= self.tsdf_min_point,
      hps <= self.tsdf_max_point), axis=1)
    hps = hps[keep_idx]
    if len(hps) == 0:
      return np.empty(0, dtype=np.int)
    hv = [self.xyz2tidx(hp) for hp in hps]
    hv = np.unique(hv, axis=0)
    tidx = np.ravel_multi_index(np.fliplr(hv).T, self.tsdf.shape)
    return tidx

  def generate_heatmap(self, deltas):
    """
    Generates a voxelized heatmap by intersecting the object with hand
    :param deltas: tuple (delta_dofs, delta_palm_pose)
    :return: (grid_size, grid_size, grid_size) np.ndarray
    """
    delta_dofs, delta_palm_pose = deltas

    # determine which hand segments' tidx are affected
    dTs = [np.eye(4)] * self.hr.n_dofs
    if delta_dofs is not None:
      dT_dofs = self.hr.calc_dTs(delta_dofs)
      dTs = [np.dot(dT, dT_dof) for dT, dT_dof in zip(dTs, dT_dofs)]
    if not tx.is_identity(delta_palm_pose):
      dTs = [np.dot(delta_palm_pose, dT) for dT in dTs]

    updated = np.logical_not([tx.is_identity(dT) for dT in dTs])

    # collect tidx
    tidx = []
    for idx, (u, h_tidx) in enumerate(zip(updated, self.hand_tidx)):
      if not u:
        tidx.append(h_tidx)
      else:
        tidx.append(self.hp2tidx(pc_utils.transform_pc(self.hr.link_pcs[idx],
          dTs[idx])))
    tidx = np.hstack(tidx)

    # create masked TSDF
    # (all the voxels that don't have a hand point are set to 0)
    tsdf = np.zeros(self.tsdf.size)
    tsdf[tidx] = self.tsdf.ravel()[tidx]

    heatmap = np.asarray([np.max(tsdf[tvs]) for _,tvs in self.v2tvs.items()])
    return heatmap

  def _get_object_pointcloud(self, colors=None):
    pc = []
    for ov in self.v2tvs.keys():
      pc.append([ov[0], ov[1], ov[2]])
    pc = np.vstack(pc)
    pc = pc_utils.ndarray2pc(pc, colors=colors)
    return pc

  def generate_voxelized_heatmap(self, delta_dofs=None, delta_palm_pose=np.eye(4)):
    heatmap = self.generate_heatmap((delta_dofs, delta_palm_pose))
    pc = self._get_object_pointcloud(colors=heatmap)
    return pc, heatmap

  def generate_voxelized_heatmap_abs(self, dofs=None, palm_pose=np.eye(4)):
    # zero out the hand
    self.set_hand_config()
    pc, heatmap = self.generate_voxelized_heatmap(dofs, palm_pose)
    return pc, heatmap

  def calc_gradient(self, process_pool):
    """
    returns d(heatmap) / d(hand_params)
    :return:
    """
    delta_dof = np.deg2rad(5)
    delta_t = 5  # mm
    delta_r = np.deg2rad(10) # degrees
    deltas = np.asarray([delta_dof]*self.hr.n_dofs + [delta_t]*3 + [delta_r]*3)

    dp_pos = +deltas * np.eye(len(deltas))
    dp_neg = -deltas * np.eye(len(deltas))

    dp_pos = [(p[:self.hr.n_dofs], tx.xyzrpy2T(p[self.hr.n_dofs:])) for p in dp_pos]
    dp_neg = [(n[:self.hr.n_dofs], tx.xyzrpy2T(n[self.hr.n_dofs:])) for n in dp_neg]

    heatmaps = map(self.generate_heatmap, dp_pos+dp_neg)

    heatmaps = np.vstack([h for h in heatmaps])
    heatmaps_pos, heatmaps_neg = heatmaps[:len(deltas)], heatmaps[len(deltas):]

    grad = (heatmaps_pos - heatmaps_neg) / (2 * deltas[:, np.newaxis])

    return grad

if __name__ == '__main__':
  object_data_dir = osp.join('~', 'deepgrasp_data', 'full19_use', 'camera-0')
  object_data_dir = osp.expanduser(object_data_dir)

  object_filename = osp.join(object_data_dir, 'cams', 'textured.ply')
  tsdf_filename = osp.join(object_data_dir, 'tsdf.vti')

  heatmap_generator = HeatmapGenerator(object_filename, tsdf_filename,
    'HumanHand20DOF.xml')

  # TODO move from pkl to csv, see grasp_pkl2csv for code
  grasps_filename = '../data/ignore/grasps2.pkl'
  with open(grasps_filename, 'rb') as f:
    grasps = pickle.load(f)

  process_pool = mp.Pool()

  VIEW = 0
  ious = []
  # gidx = range(len(grasps))
  gidx = [3276, 4648, 1836, 3995, 2085, 2937, 2450, 3728, 3479, 2990]
  for idx, grasp in enumerate([grasps[i] for i in gidx]):
    print('Processing grasp {:d} / {:d}'.format(idx, len(grasps)))
    hand_config = grasp['hand_pose']
    q = hand_config.robot.pose.orientation
    t = hand_config.robot.pose.position
    palm_pose = tx.quaternion_matrix([q.w, q.x, q.y, q.z])
    palm_pose[0, 3] = t.x * 1000
    palm_pose[1, 3] = t.y * 1000
    palm_pose[2, 3] = t.z * 1000
    heatmap_generator.set_hand_config(hand_config.robot.dofs, palm_pose)
    dH_dp = heatmap_generator.calc_gradient(process_pool)
    g_pc = heatmap_generator._get_object_pointcloud(colors=dH_dp[0])
    # i_pc, i_heatmap = heatmap_generator.generate_voxelized_heatmap()
    # iou = heatmap_iou(i_heatmap, heatmap_generator.ocs)
    # ious.append(iou)
    # print('IoU = {:f}'.format(iou))

    if VIEW:
      v = Viewer()
      object = open3d.read_triangle_mesh(object_filename)
      object.transform(1000*np.eye(4))
      v.add_geometry(object, 'grasp')
      for h_pc in heatmap_generator.hr.link_pcs:
        v.add_geometry(pc_utils.ndarray2pc(h_pc), 'grasp')
      v.add_geometry(i_pc, 'contact heatmap')
      v.add_geometry(g_pc, 'gradient')
      t_pc = pc_utils.ndarray2pc(heatmap_generator.t_vol[:, :3],
        colors=heatmap_generator.t_vol[:, 3])
      v.add_geometry(t_pc, 'GT heatmap')
      v.show()
  if not VIEW:
    # np.savetxt('../data/ignore/human_scores2.txt', ious)
    pass