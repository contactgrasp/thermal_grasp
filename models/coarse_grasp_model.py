"""
Simple 3D convolutions on voxel grid reprsentation of objects with multiple
grasp prediction heads
"""
import torch
import torch.nn as tnn
import torch.nn.functional as tnnF

class CoarseGraspModel(tnn.Module):
  def __init__(self, n_heads=1, n_angles=4, droprate=0.5):
    super(CoarseGraspModel, self).__init__()
    self.n_heads = n_heads
    self.n_angles = n_angles
    self.droprate = droprate

    # input size: 64
    self.conv1 = tnn.Conv3d(in_channels=1, out_channels=16, kernel_size=3,
      stride=1, padding=1)
    self.pool1 = tnn.MaxPool3d(kernel_size=2, stride=2, padding=0)
    self.bn1   = tnn.BatchNorm3d(num_features=self.conv1.out_channels)

    # input size: 32
    self.conv2 = tnn.Conv3d(in_channels=self.conv1.out_channels, out_channels=32,
      kernel_size=3, stride=1, padding=1)
    self.pool2 = tnn.MaxPool3d(kernel_size=2, stride=2, padding=0)
    self.bn2   = tnn.BatchNorm3d(num_features=self.conv2.out_channels)

    # input size: 16
    self.conv3 = tnn.Conv3d(in_channels=self.conv2.out_channels, out_channels=64,
      kernel_size=3, stride=1, padding=1)
    self.pool3 = tnn.MaxPool3d(kernel_size=2, stride=2, padding=0)
    self.bn3   = tnn.BatchNorm3d(num_features=self.conv3.out_channels)

    # input size: 8
    self.conv4 = tnn.Conv3d(in_channels=self.conv3.out_channels, out_channels=128,
      kernel_size=3, stride=1, padding=1)
    self.pool4 = tnn.MaxPool3d(kernel_size=8, stride=8, padding=0)
    self.bn4   = tnn.BatchNorm3d(num_features=self.conv4.out_channels)

    # input size: 128
    self.fc_pt = tnn.ModuleList(
      [tnn.Linear(in_features=self.conv4.out_channels, out_features=3)] *\
      self.n_heads)
    self.fc_angle = tnn.ModuleList(
      [tnn.Linear(in_features=self.conv4.out_channels, out_features=self.n_angles)] \
      * self.n_heads)
    self.fc_dist = tnn.ModuleList(
      [tnn.Linear(in_features=self.conv4.out_channels, out_features=1)] \
      * self.n_heads)

    self._init_params()

  def _init_params(self):
    pass

  def forward(self, x):
    x = self.pool1(self.bn1(tnnF.relu(self.conv1(x))))
    x = self.pool2(self.bn2(tnnF.relu(self.conv2(x))))
    x = self.pool3(self.bn3(tnnF.relu(self.conv3(x))))
    x = self.pool4(self.bn4(tnnF.relu(self.conv4(x))))

    x = x.view(x.shape[0], -1)

    if self.droprate > 0:
      x = tnnF.dropout(x, p=self.droprate, training=self.training)

    x_pt    = torch.stack([tnnF.tanh(fc(x)) for fc in self.fc_pt], dim=1)
    x_angle = torch.stack([fc(x) for fc in self.fc_angle], dim=1)
    x_dist  = torch.stack([fc(x) for fc in self.fc_dist], dim=1)

    x = torch.cat([x_pt, x_angle, x_dist], dim=-1)

    return x