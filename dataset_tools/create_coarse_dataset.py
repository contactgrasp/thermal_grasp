"""
This class creates the coarse grasp dataset from the ground truth hand poses
The coarse grasp consists of discretized approach axis, rotation around that
axis, and translation from the origin along the approach axis
"""
import os
import numpy as np
import argparse
osp = os.path


def anms(x, scores, N):
  """
  Adaptive non maximum suppression
  :param x: data to be sampled from
  :param scores: score associated with each datapoint
  :param N: number of output samples
  :return: N indices
  """
  if len(x) <= N:
    return list(range(len(x)))

  idx = np.argsort(-scores)

  # array holding distances to the nearest better point
  d = [float('inf')]
  for i in range(1, len(idx)):
    dd = np.sum((x[np.newaxis, idx[i]] - x[idx[:i]])**2, axis=1)
    d.append(min(dd))

  idxx = np.argsort(-np.asarray(d))[:N]
  return idx[idxx]

class CoarseGraspCollector(object):
  def __init__(self, top_k=7, n_preds=1, scale_factor=1.15,
      base_dir=osp.join('~', 'deepgrasp_data')):
    """
    :param top_k: top K grasps (ranked by similarity to GT hand pose) will be
    used from each session
    :param n_preds: number of grasps to produce, in total, for the object. This
    is equal to the number of prediction heads of the neural network
    :param scale_factor: scale factor applied during the graspit grasp collection
    :param base_dir: Root directory containing all data
    """
    self.base_dir = osp.expanduser(base_dir)
    self.top_k = top_k
    self.n_preds = n_preds
    self.scale_factor = scale_factor
    self.valid_actions = ['handoff', 'use']

  def get_coarse_grasps(self, object_name, action):
    """
    get the list of coarse grasps
    :param object_name: name of object
    :param action: 'handoff' or 'use'
    :return:
    """
    assert action in self.valid_actions

    # read the graspit grasps
    grasps_filename = osp.join(self.base_dir, 'grasps',
      '{:s}.csv'.format(object_name))
    grasps = np.loadtxt(grasps_filename, delimiter=',',
      usecols=np.arange(-8, 0), dtype=np.float32)

    # read the object bounds
    bounds_filename = osp.join(self.base_dir, 'models',
      '{:s}.bounds'.format(object_name))
    object_min_pt, object_max_pt = np.loadtxt(bounds_filename)
    object_min_pt *= self.scale_factor
    object_max_pt *= self.scale_factor

    # normalize grasp points by object bounds
    grasps[:, :3] -= ((object_min_pt + object_max_pt) / 2.0)
    grasps[:, :3] /= ((object_max_pt - object_min_pt) / 2.0)

    # read indices for top GT hand pose similarity scores from each session
    indices = []
    errors = []
    for session_dir in os.listdir(self.base_dir):
      if not osp.isdir(osp.join(self.base_dir, session_dir)):
        continue
      if 'full19' not in session_dir:
        continue
      if action not in session_dir:
        continue

      for object_dir in os.listdir(osp.join(self.base_dir, session_dir)):
        if object_name not in object_dir:
          continue
        if '-' in object_dir:
          if object_dir.split('-')[-1] != '0':
            continue

        errors_filename = osp.join(self.base_dir, session_dir, object_dir,
          'errors.csv')
        print(session_dir)
        errs = np.loadtxt(errors_filename, delimiter=',', usecols=-1)
        idx = np.argsort(errs)[:self.top_k]
        indices.extend(idx)
        errors.extend(errs[idx])

    # remove duplicate grasps by keeping the min error
    unique_indices = np.unique(indices)
    errors = np.asarray(errors)
    errors = np.asarray([np.min(errors[indices == i]) for i in unique_indices])
    indices = unique_indices

    # sample n_pred grasps by NMS
    # TODO: test ANMS function
    idx_anms = anms(grasps[indices, :3], -errors, self.n_preds)
    indices = indices[idx_anms]

    return grasps[indices]


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-o', '--object_name', help='Name of object',
    required=True)
  parser.add_argument('-a', '--action', choices=('use', 'handoff'),
    required=True, help='Type of action')
  parser.add_argument('--dataset_dir', default=osp.join('~', 'deepgrasp_data'),
    help='Directory containing all the data')

  args = parser.parse_args()

  cgc = CoarseGraspCollector(args.object_name, args.action,
    base_dir=args.dataset_dir)
  grasps = cgc.get_coarse_grasps('camera', 'use')
  print(grasps)
