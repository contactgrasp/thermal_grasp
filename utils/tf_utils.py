import numpy as np
import transforms3d.euler as txe
import transforms3d.quaternions as txq

def euler_matrix(ax, ay, az, axes='sxyz'):
  T = np.eye(4)
  T[:3, :3] = txe.euler2mat(ax, ay, az, axes=axes)
  return T

def quaternion_matrix(q):
  T = np.eye(4)
  T[:3, :3] = txq.quat2mat(q)
  return T

def quat2euler(q):
  return txe.quat2euler(q)

def inverse_matrix(T):
  return np.linalg.inv(T)

def xyzrpy2T(xyzrpy):
  x, y, z, ax, ay, az = xyzrpy
  T = euler_matrix(ax, ay, az)
  T[:3, 3] = [x, y, z]
  return T

def T2xyzrpy(T):
  rx, ry, rz = txe.mat2euler(T[:3, :3])
  tx, ty, tz = T[:3, 3]
  return tx, ty, tz, rx, ry, rz

def is_identity(T):
  return np.allclose(T, np.eye(len(T)))

def rotmat_from_vecs(v1, v2=np.asarray([0, 0, 1])):
  """
  Returns a rotation matrix R_1_2
  :param v1: vector in frame 1
  :param v2: vector in frame 2
  :return:
  """
  v1 = v1 / np.linalg.norm(v1)
  v2 = v2 / np.linalg.norm(v2)
  v = np.cross(v2, v1)
  vx = np.asarray([
    [0,    -v[2], +v[1], 0],
    [+v[2], 0,    -v[0], 0],
    [-v[1], +v[0], 0,    0],
    [0,     0,     0,    0]])
  dotp = np.dot(v1, v2)

  if abs(1 + dotp) < 1e-3:  # vectors are opposite
    # find axis normal to v1
    for v in np.asarray([[1,0,0], [0,1,0], [0,0,1]]):
      axis = np.cross(v, v1)
      if np.linalg.norm(axis) > 1e-3:
        break
    # output is 180 rotation about axis
    R = np.eye(4)
    R[:3, :3] = txe.axangle2mat(axis, np.pi) 
    return R

  return np.eye(4) + vx + np.dot(vx, vx)/(1+dotp)

def normal2tangent(n):
  if n.ndim == 1:
    n = n[np.newaxis, :]
  t1 = np.cross(n, [0, 0, 1])
  t2 = np.cross(n, [0, 1, 0])
  w = np.linalg.norm(t1, axis=1, keepdims=True) > 1e-3
  t = t1*w + t2*np.logical_not(w)
  t /= np.linalg.norm(t, axis=1, keepdims=True)
  return t
