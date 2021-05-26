"""
converts a grasp pickle file to CSV
"""
from utils import tf_utils as tx
from itertools import product
import numpy as np
import pickle

if __name__ == '__main__':
  input_filename = '../data/ignore/grasps2.pkl'
  with open(input_filename, 'rb') as f:
    grasps = pickle.load(f)
  score_filename = '../data/ignore/human_scores2.txt'
  order = np.loadtxt(score_filename)
  order = np.argsort(-order)

  output_filename = input_filename.split('.')[:-1] + ['csv']
  output_filename = '.'.join(output_filename)
  proposed_grasps = []

  csv_header = []
  for y, x in product(range(3), range(4)):
    csv_header.append('T{:d}{:d}'.format(y, x))
  for i in range(20):
    csv_header.append('dof{:d}'.format(i))
  csv_header.append('POTENTIAL_QUALITY_ENERGY')
  csv_header.append('GUIDED_POTENTIAL_QUALITY_ENERGY')
  csv_header.append('CONTACT_ENERGY')
  csv_header.append('quality_volume')
  csv_header.append('quality_epsilon')

  for grasp in grasps:
    proposal = []

    # get refined pose
    q = grasp['hand_pose'].robot.pose.orientation
    t = grasp['hand_pose'].robot.pose.position
    palm_pose = tx.quaternion_matrix([q.w, q.x, q.y, q.z])
    palm_pose[0, 3] = t.x
    palm_pose[1, 3] = t.y
    palm_pose[2, 3] = t.z
    proposal.extend(palm_pose[:3].flatten())
    proposal.extend(grasp['hand_pose'].robot.dofs)

    # energies
    energy_type = csv_header[len(proposal)]
    try:
      e = grasp[energy_type]
      e = e.energy
    except:
      e = np.finfo(float).max
    proposal.append(e)

    energy_type = csv_header[len(proposal)]
    try:
      e = grasp[energy_type]
      e = e.energy
    except:
      e = np.finfo(float).max
    proposal.append(e)

    energy_type = csv_header[len(proposal)]
    try:
      e = grasp[energy_type]
      e = e.energy
    except:
      e = np.finfo(float).max
    proposal.append(e)

    # quality
    try:
      q = grasp['quality']
      qw, qe = q.volume, q.epsilon
    except:
      qw = qe = -np.finfo(float).max
    proposal.append(qw)
    proposal.append(qe)

    proposed_grasps.append(proposal)

  csv_header = ','.join(csv_header)
  proposed_grasps = np.asarray(proposed_grasps)
  proposed_grasps = proposed_grasps[order]
  np.savetxt(output_filename, proposed_grasps, delimiter=',',
    header=csv_header)

  print('Saved {:d} grasps to {:s}'.format(len(proposed_grasps), output_filename))