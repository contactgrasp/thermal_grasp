import numpy as np
import argparse
import rank_analysis_data
import os
import shutil
import matplotlib.pyplot as plt
from IPython.core.debugger import set_trace

osp = os.path


def analyze_hand_grasps(instruction, hand_name,
    data_dir=osp.join('..', 'data', 'grasps')):
  object_sessions = getattr(rank_analysis_data,
      '{:s}_sessions'.format(instruction))

  dart_ranks = {}
  graspit_ranks = {}
  for object_name, session_num in object_sessions.items():
    print('Processing {:s}'.format(object_name))

    # read the rank of the correct grasp after DART ranking
    dart_filename = '{:d}_{:s}_{:s}_gt_hand_pose.txt'.format(session_num,
        instruction, object_name)
    if hand_name != 'human':
      dart_filename = '{:s}_{:s}'.format(hand_name, dart_filename)
    dart_filename = osp.join(data_dir, dart_filename)
    try:
      dart_grasp_data = np.loadtxt(dart_filename, delimiter=',')
    except IOError:
      print('{:s} not found'.format(dart_filename))
      continue

    if len(dart_grasp_data) != rank_analysis_data.n_dofs[hand_name] + 1:
      print('{:s} does not have rank information'.format(dart_filename))
      continue
    dart_rank = int(dart_grasp_data[0])

    # rank of the correct grasp w.r.t. Graspit energies
    dart_energy_filename = '{:d}_{:s}_{:s}_grasp_errors.csv'.format(session_num,
        instruction, object_name)
    if hand_name != 'human':
      dart_filename = '{:s}_{:s}'.format(hand_name, dart_filename)
    dart_energies = np.loadtxt(osp.join(data_dir, dart_energy_filename))

    graspit_energy_filename = '{:s}_grasps.csv'.format(object_name)
    if hand_name != 'human':
      graspit_energy_filename = '{:s}_grasps_{:s}.csv'.format(object_name,
          hand_name)
    graspit_energy_filename = osp.join(data_dir, graspit_energy_filename)
    graspit_energies = np.loadtxt(graspit_energy_filename, delimiter=',')
    graspit_energies = graspit_energies[:, -1]

    # follow the same protocol as dart/src/grasp_analyzer.cpp:set_order()
    if len(dart_energies) > len(graspit_energies):
      dart_energies = dart_energies[:len(graspit_energies)]
    elif len(dart_energies) < len(graspit_energies):
      len_diff = len(graspit_energies) - len(dart_energies)
      dart_energies = np.hstack((dart_energies, np.inf*np.ones(len_diff)))
    assert len(dart_energies) == len(graspit_energies)

    grasp_idx = np.argsort(dart_energies)[dart_rank]
    graspit_rank = np.where(np.argsort(graspit_energies) == grasp_idx)[0][0]
    
    dart_ranks[object_name] = float(dart_rank) / len(dart_energies)
    graspit_ranks[object_name] = float(graspit_rank) / len(graspit_energies)
  return dart_ranks, graspit_ranks


def analyze_grasps(instruction, data_dir=osp.join('..', 'data', 'grasps')):
  dart_data = []
  graspit_data = []
  hand_names = ['human', 'allegro', 'barrett']
  for hand_name in hand_names:
    dart_ranks, graspit_ranks = analyze_hand_grasps(instruction, hand_name,
        data_dir)
    dart_rank = np.median(np.asarray(dart_ranks.values(), dtype=np.float))
    dart_data.append(dart_rank)
    print('{:s} dart rank = {:f}'.format(hand_name, dart_rank))
    graspit_rank = np.median(np.asarray(graspit_ranks.values(), dtype=np.float))
    graspit_data.append(graspit_rank)
    print('{:s} graspit rank = {:f}'.format(hand_name, graspit_rank))
  fig = plt.figure()
  ax = fig.add_subplot(111)
  width = 0.35
  ind = np.arange(len(dart_data))
  dart_plot = ax.bar(ind-width/2, dart_data, width, color='green', align='center')
  graspit_plot = ax.bar(ind+width/2, graspit_data, width, color='red',
      align='center')
  ax.set_ylabel('Rank')
  ax.set_xticks(ind)
  ax.set_xticklabels(hand_names)
  plt.show()


def merge_grasp_files(data_dir=osp.join('..', 'data', 'grasps')):
  for filename in next(os.walk(data_dir))[-1]:
    if '_grasps_energies.csv' not in filename:
      continue
    src_filename = osp.join(data_dir, filename)
    dst_filename = filename.replace('_energies', '')
    dst_filename = osp.join(data_dir, dst_filename)
    shutil.move(src_filename, dst_filename)
    print('Move {:s} to {:s}'.format(src_filename, dst_filename))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--instruction', required=True)
  args = parser.parse_args()

  analyze_grasps(args.instruction)
  # merge_grasp_files()
