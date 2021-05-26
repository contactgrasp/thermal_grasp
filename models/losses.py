import torch
import torch.nn as tnn


class CoarseGraspLoss(object):
  def __init__(self, n_angles=4):
    self.position_loss = tnn.MSELoss(reduce=False)
    self.angle_loss = tnn.CrossEntropyLoss()
    self.dist_loss = tnn.MSELoss()
    self.n_angles = n_angles

  def __call__(self, pred, targ):
    """
    Calculates the loss the minibatch. Each ground truth pose is matched with
    the closest prediction from M heads of the network. Closeness is defined
    by distance of predicted surface point from the ground truth surface point
    :param pred: N x M x 8
    :param targ: N x 5
    :return:
    """
    # position
    pos_pred = pred[:, :, :3]
    pos_targ = targ[:, :3]
    pos_loss = self.position_loss(pos_pred,
      pos_targ.unsqueeze(1).expand_as(pos_pred))
    pos_loss = pos_loss.sum(-1)
    idx = pos_loss.argmax(1)
    pos_loss = sum([pos_loss[i, idx[i]] for i in range(len(pos_loss))])

    angle_pred = torch.stack([pred[i, idx[i], 3:3+self.n_angles]
      for i in range(len(pred))])
    angle_targ = targ[:, 3].long()
    angle_loss = self.angle_loss(angle_pred, angle_targ)

    dist_pred = torch.stack([pred[i, idx[i], -1] for i in range(len(pred))])
    dist_targ = targ[:, 4]
    dist_loss = self.dist_loss(dist_pred, dist_targ)

    loss = pos_loss + angle_loss + dist_loss
    return loss, pos_loss, angle_loss, dist_loss
