import numpy as np
from scripts.generate_heatmap import HeatmapGenerator
from utils.vis_utils import Viewer
import os.path as osp
from utils import tf_utils as tx
import open3d
from utils.pc_utils import ndarray2pc

if __name__ == '__main__':
  object_data_dir = osp.join('~', 'deepgrasp_data', 'full19_use', 'camera-0')
  object_data_dir = osp.expanduser(object_data_dir)

  object_filename = osp.join(object_data_dir, 'cams', 'textured.ply')
  tsdf_filename = osp.join(object_data_dir, 'tsdf.vti')

  RENDER_ONLY = True

  hg = HeatmapGenerator(object_filename, tsdf_filename, 'HumanHand20DOF.xml',
    render_only=RENDER_ONLY)

  palm_pose = tx.quaternion_matrix([+0.589375, +0.0883405, +0.0994665, +0.796831])
  palm_pose[0, 3] = -155
  palm_pose[1, 3] = +112.526
  palm_pose[2, 3] = +55.7786

  dof_vals = \
    [ 20,74,0,0,
      0,68,0,10,
      0,71,0,0,
      0,0,0,0,
      -40,-30, 20,0]
  dof_vals = np.deg2rad(dof_vals)

  hg.set_hand_config(palm_pose=palm_pose, dof_vals=dof_vals)
  if not RENDER_ONLY:
    i_vol, iou = hg.generate_voxel_heatmap()
    print('IoU = {:f}'.format(iou))

  v = Viewer()
  object = open3d.read_triangle_mesh(object_filename)
  object.transform(1000*np.eye(4))
  v.add_geometry(object, 'grasp')
  for h_pc in hg.hr.link_pcs:
    v.add_geometry(ndarray2pc(h_pc), 'grasp')
  if not RENDER_ONLY:
    i_pc = ndarray2pc(i_vol[:, :3], colors=i_vol[:, 3:])
    v.add_geometry(i_pc, 'contact heatmap')
    t_pc = ndarray2pc(hg.t_vol[:, :3], colors=hg.t_vol[:, 3:])
    v.add_geometry(t_pc, 'GT heatmap')
  v.show()