# Copyright (c) 2022 Battelle Energy Alliance, LLC
# Licensed under MIT License, please see LICENSE for details
# https://github.com/IdahoLabResearch/BIhNNs/blob/main/LICENSE

# Training Hamiltonian Neural Networks (HNNs) for Bayesian inference problems
# Original authors of HNNs code: Sam Greydanus, Misko Dzamba, Jason Yosinski (2019)
# Available at https://github.com/greydanus/hamiltonian-nn under the Apache License 2.0
# Modified by Som Dhulipala at Idaho National Laboratory for Bayesian inference problems
# Modifications include:
# - Generalizing the code to any number of dimensions
# - Introduce latent parameters to HNNs to improve expressivity
# - Reliance on the leap frog integrator for improved dynamics stability
# - Obtain the training from probability distribution space
# - Use a deep HNN arichtecture to improve predictive performance

import torch, argparse
import numpy as np
import sys, os
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
sys.path.append(PARENT_DIR)
from paper_code.utils import choose_nonlinearity
from paper_code.get_args import get_args
args = get_args()

class MLP(torch.nn.Module):
  '''Just a salt-of-the-earth MLP'''
  def __init__(self, input_dim, hidden_dim, output_dim, nonlinearity='sine', num_layers=5):
    super(MLP, self).__init__()
    self.layers_list = []
    self.layers_list.append(torch.nn.Linear(input_dim, hidden_dim))
    for _ in range(num_layers - 2):
      self.layers_list.append(torch.nn.Linear(hidden_dim, hidden_dim))
    self.layers_list.append(torch.nn.Linear(hidden_dim, output_dim, bias=None))
    self.layers = torch.nn.ModuleList(self.layers_list)

    for l in self.layers:
      torch.nn.init.orthogonal_(l.weight) # use a principled initialization

    # self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
    # self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
    # self.linear3 = torch.nn.Linear(hidden_dim, hidden_dim)
    # self.linear4 = torch.nn.Linear(hidden_dim, hidden_dim)
    # self.linear5 = torch.nn.Linear(hidden_dim, output_dim, bias=None)

    # for l in [self.linear1, self.linear2, self.linear3, self.linear4, self.linear5]:
    #   torch.nn.init.orthogonal_(l.weight) # use a principled initialization

    self.nonlinearity = choose_nonlinearity(nonlinearity)

  def forward(self, x, separate_fields=False):
    for i, layer in enumerate(self.layers):
      if i == 0:
        h = self.nonlinearity(layer(x))
      elif i == len(self.layers)-1:
        h = layer(h)
      else:
        h = self.nonlinearity(layer(h))
    return h
    # h = self.nonlinearity( self.linear1(x) )
    # h = self.nonlinearity( self.linear2(h) )
    # h = self.nonlinearity( self.linear3(h) )
    # h = self.nonlinearity( self.linear4(h) )
    # return self.linear5(h)