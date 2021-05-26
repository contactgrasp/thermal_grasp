import numpy as np
import open3d
import vtk
from vtk.util import numpy_support
import matplotlib.pyplot as plt

def ndarray_from_vti(filename):
  """
  Reads a .vti file into a np.ndarray
  :param filename:
  :return:
  """
  reader = vtk.vtkXMLImageDataReader()
  reader.SetFileName(filename)
  reader.Update()
  im = reader.GetOutput()
  shape = im.GetDimensions()
  vox_sizes = im.GetSpacing()
  arr = im.GetPointData().GetArray(0)
  nparr = numpy_support.vtk_to_numpy(arr)
  nparr = np.reshape(nparr, shape)
  return nparr, vox_sizes

def ndarray2vti(nparr, filename):
  """
  writes a np.ndarray to a .vti file
  :param x: np.ndarray
  :param filename:
  :return:
  """
  im = vtk.vtkImageData()
  im.SetDimensions(*nparr.shape)

  arr = vtk.vtkDoubleArray()
  arr.SetNumberOfTuples(nparr.size)
  arr.SetName('intensity')
  for i in range(nparr.size):
    idx = np.unravel_index(i, nparr.shape)
    arr.SetValue(i, nparr[idx])

  im.GetPointData().AddArray(arr)

  writer = vtk.vtkXMLImageDataWriter()
  writer.SetFileName(filename)
  writer.SetInputData(im)
  writer.Write()

def ndarray2pc(x, colors=None):
  """
  converts np.ndarray to open3d.PointCloud
  :param x: np.ndarray (N, 3)
  :param colors: np.ndarray
  :return: open3d.PointCloud instance
  """
  assert x.ndim == 2
  assert x.shape[1] == 3

  pc = open3d.PointCloud()
  pts = open3d.Vector3dVector(x)
  pc.points.extend(pts)

  if colors is not None:
    if colors.ndim == 1:
      norm = plt.Normalize()
      colors = plt.cm.jet(norm(colors))[:, :3]
    colors = open3d.Vector3dVector(colors)
    pc.colors.extend(colors)

  return pc

def transform_pc(pc, T):
  """
  :param pc: N x 3
  :param T: 4 x 4
  :return:
  """
  pc = np.dot(T[:3, :3], pc.T)
  pc += T[:3, 3:]
  return pc.T
