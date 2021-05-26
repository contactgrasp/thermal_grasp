from models.coarse_grasp_model import CoarseGraspModel
from models.losses import CoarseGraspLoss
from dataset_tools.coarse_grasp import CoarseGraspDataset

from ignite.engine import Engine, Events
from ignite.handlers import Timer, ModelCheckpoint
import visdom
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np
import os.path as osp
import torch

def create_plot_window(vis, xlabel, ylabel, title, win, env, trace_name):
  return vis.line(X=np.array([1]), Y=np.array([np.nan]), win=win, env=env,
    name=trace_name,
    opts=dict(xlabel=xlabel, ylabel=ylabel, title=title))

def train(objects, actions, batch_size=2, max_epochs=1000, log_interval=1,
    device='cuda'):
  model = CoarseGraspModel(droprate=0.44)
  model.to(device=device)
  optim = Adam(model.parameters())
  loss_fn = CoarseGraspLoss()

  kwargs = {'n_preds': 1}
  train_dset = CoarseGraspDataset(objects=objects, actions=actions, **kwargs)
  train_dloader = DataLoader(train_dset, batch_size=batch_size, shuffle=True,
    pin_memory=True)

  def train_loop(engine: Engine, batch):
     model.train()
     optim.zero_grad()
     obj, robj, targ = batch
     pred = model(obj.to(device=device))
     rpred = model(robj.to(device=device))
     pred = torch.cat((pred[..., :3], rpred[..., 3:]), dim=-1)
     loss, pos_loss, angle_loss, dist_loss = loss_fn(pred, targ.to(device=device))
     loss.backward()
     optim.step()
     engine.state.pos_loss = pos_loss.item()
     engine.state.angle_loss = angle_loss.item()
     engine.state.dist_loss = dist_loss.item()
     return loss.item()
  trainer = Engine(train_loop)

  # callbacks
  vis = visdom.Visdom()
  train_loss_win = 'train_loss'
  env_name = 'coarse_grasp'
  create_plot_window(vis, '#Iterations', 'Loss', 'Training Loss',
    win=train_loss_win, env=env_name, trace_name='loss')

  @trainer.on(Events.ITERATION_COMPLETED)
  def log_training_loss(engine):
    iter = (engine.state.iteration - 1) % len(train_dloader) + 1
    if iter % log_interval == 0:
      print("Epoch[{}] Iteration[{}/{}] Loss: {:.2f}"
            "".format(engine.state.epoch, iter, len(train_dloader),
        engine.state.output))
      vis.line(X=np.array([engine.state.iteration]),
        Y=np.array([engine.state.output]),
        update='append', win=train_loss_win, env=env_name, name='loss')

  def checkpoint_fn(engine: Engine):
    return -engine.state.output

  checkpoint_handler =\
    ModelCheckpoint(dirname=osp.join('..', 'data', 'ignore', 'checkpoints'),
      filename_prefix='coarse_grasp', score_function=checkpoint_fn,
      score_name='train_loss', create_dir=False, require_empty=False)
  trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler,
    {'model': model})

  trainer.run(train_dloader, max_epochs)

if __name__ == '__main__':
  objects = ['camera', 'binoculars']
  actions = ['use', 'handoff']

  train(objects, actions)
