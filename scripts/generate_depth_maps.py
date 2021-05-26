"""
generates depth maps by loading a mesh model and rendering it from a list of
exhaustive viewpoints
"""
import open3d
import numpy as np
from utils import tf_utils as tx
import time
import os.path as osp

class RenderManager:
  def __init__(self, im_width, im_height, focal_length, object_name, max_Z = 3.0):
    self.im_width = im_width
    self.im_height = im_height
    self.focal_length = focal_length
    self.max_Z = max_Z
    self.index = 0
    self.output_dir = osp.join('data', 'human_hand_meshes', object_name)

    np.savetxt(osp.join(self.output_dir, 'camera.txt'),
      [im_width, im_height, focal_length])

    # camera center for a perfect camera
    self.cx = self.im_width/2.0 - 0.5
    self.cy = self.im_height/2.0 - 0.5

    # camera intrinsics object
    self.intrinsics = open3d.PinholeCameraIntrinsic(width=im_width,
      height=im_height, cx=self.cx, cy=self.cy, fx=focal_length, fy=focal_length)

    # camera trajectory
    r = 0.3
    elevs = [0, 45, -45]
    self.traj = []
    T_upright = tx.euler_matrix(np.deg2rad(90), -np.deg2rad(90), 0, axes='rzxy')
    for elev in elevs:
      for azim in np.linspace(0, 360, 6, endpoint=False):
        T = tx.euler_matrix(-np.deg2rad(azim), -np.deg2rad(elev), 0, axes='ryxz')
        T = np.dot(T_upright, T)
        T[2, 3] = r * np.sin(np.deg2rad(elev))
        XY = r * np.cos(np.deg2rad(elev))
        T[0, 3] = XY * np.cos(np.deg2rad(azim))
        T[1, 3] = XY * np.sin(np.deg2rad(azim))
        self.traj.append(tx.inverse_matrix(T))

  def animation_cb(self, vis):
    if self.index < 0 or self.index >= len(self.traj):
      vis.register_animation_callback(None)
      return False

    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(self.intrinsics,
      self.traj[self.index])
    depth = vis.capture_depth_float_buffer(True)
    print("Captured image {:d}".format(self.index))
    # plt.imshow(np.asarray(depth))
    # plt.show(block=True)

    pc = open3d.create_point_cloud_from_depth_image(depth, self.intrinsics,
      np.eye(4))
    filename = osp.join(self.output_dir, '{:03d}.pcd'.format(self.index))
    if not open3d.write_point_cloud(filename, pc):
      print('could not save pointcloud to {:s}'.format(filename))
    filename = osp.join(self.output_dir, '{:03d}.txt'.format(self.index))
    np.savetxt(filename, self.traj[self.index])

    self.index = self.index + 1
    time.sleep(0.4)

if __name__ == '__main__':

  im_width = 640
  im_height = 480
  focal_length = 1400
  object_name = 'mid3'

  m = open3d.read_triangle_mesh('data/human_hand_meshes/{:s}/{:s}.ply'.
    format(object_name, object_name))
  m.transform(np.eye(4)/1000.)
  rm = RenderManager(im_width, im_height, focal_length, object_name)
  open3d.draw_geometries_with_animation_callback([m], rm.animation_cb,
      width=im_width, height=im_height)