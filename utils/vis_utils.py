"""
Multi-threaded open3D viewer
"""
import open3d
from multiprocessing import Process


class Viewer(object):
  def __init__(self):
    self.geom = {}

  def add_geometry(self, g, window_name):
    if window_name in self.geom:
      self.geom[window_name].append(g)
    else:
      self.geom[window_name] = [g]

  @staticmethod
  def worker(window_name, gs):
    vis = open3d.Visualizer()
    vis.create_window(window_name, width=640, height=480)
    for g in gs:
      vis.add_geometry(g)
    vis.run()
    vis.destroy_window()

  def show(self, block=True):
    jobs = []
    for window_name, g in self.geom.items():
      p = Process(target=self.worker, args=(window_name, g))
      jobs.append(p)
      p.start()

    if block:
      for j in jobs:
        j.join()

if __name__ == '__main__':
  pc1 = open3d.read_point_cloud('../data/camera/000.pcd')
  pc2 = open3d.read_point_cloud('../data/camera/001.pcd')

  v = Viewer()
  v.add_geometry(pc1, 'window1')
  v.add_geometry(pc2, 'window2')
  v.show()