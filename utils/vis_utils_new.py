"""
Multi-threaded open3D viewer
"""
import open3d
import multiprocessing as mp

class Consumer(mp.Process):
  def __init__(self, vis_q):
    super(Consumer, self).__init__()
    self.vis_q = vis_q

  def run(self):
    while True:
      next_vis = self.vis_q.get()
      if next_vis is None:
        break

      next_vis.run()
      next_vis.destroy_window()

    return


class Viewer(object):
  def __init__(self):
    self.vis = {}

  def add_geometry(self, g, window_name):
    if window_name in self.vis:
      self.vis[window_name].add_geometry(g)
    else:
      self.vis[window_name] = open3d.Visualizer()
      self.vis[window_name].create_window(window_name, width=640, height=480)
      self.vis[window_name].add_geometry(g)

  def show(self, block=True):

    q = mp.JoinableQueue()
    num_workers = mp.cpu_count() * 2
    workers = [Consumer(q) for _ in xrange(num_workers)]
    for w in workers:
      w.start()

    for vis in self.vis.values():
      q.put(vis)

    for _ in xrange(num_workers):
      q.put(None)

    if block:
      q.join()

if __name__ == '__main__':
  pc1 = open3d.read_point_cloud('../data/camera/000.pcd')
  pc2 = open3d.read_point_cloud('../data/camera/001.pcd')

  v = Viewer()
  v.add_geometry(pc1, 'window1')
  v.add_geometry(pc2, 'window2')
  v.show()