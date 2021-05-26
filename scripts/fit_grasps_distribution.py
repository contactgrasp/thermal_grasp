"""
This script fits a Gaussaian to the ground truth grasp joint values
so that good random initializations can be sampled from it
"""
import numpy as np
import os
from sklearn.mixture import GaussianMixture
from IPython.core.debugger import set_trace
osp = os.path

def fit_and_sample(N_samples=100, data_dir=osp.join('..', 'data', 'grasps')):
  # read all the grond truth files
  X = []
  for filename in next(os.walk(osp.expanduser(data_dir)))[-1]:
    if 'gt_hand_pose' not in filename:
      continue
    filename = osp.join(data_dir, filename)
    try:
      x = np.loadtxt(filename, delimiter=',')[12:]
    except ValueError:
      set_trace()
    X.append(x)
  X = np.vstack(X)

  # fit Gaussian
  mog = GaussianMixture(n_components=3, covariance_type='diag')
  mog.fit(X)
  print('MoG fit, log-likelihood = {:f}'.format(mog.score(X)))

  # sample from distribution
  XX = mog.sample(n_samples=N_samples)

  return XX


if __name__ == '__main__':
  N_samples = 100
  output_filename = osp.join('..', 'data', 'grasps', 'dof_samples.txt')
  
  X_samples = fit_and_sample(N_samples)[0]
  
  np.savetxt(output_filename, X_samples, delimiter=',')
  print('{:d} samples saved at {:s}'.format(N_samples, output_filename))
