import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import networkx as nx
import argparse
import pickle
from itertools import product

from utils import update_columns
from utils import select_columns

# Path of the train.csv and test.csv
DATA_DIR = './data-example/'

# Path of the rpc_dependency.csv
DAG_DIR = './data-example/'

# Dimension of the latency vectors
DIST_DIM = 2

# Dimension of the hidden variables
Z_DIM = 1

# Max latency (us) to clip
MAX_CLIP = 200000

# Whether to use the log scale for the latency
LOG_SCALE = False

# Whether to clip
CLIP = True

train_model_path = './models/'

csv_file_train = DATA_DIR + 'train.csv'
csv_file_test = DATA_DIR + 'test.csv'


class GraphicalCVAE(nn.Module):
  def __init__(
      self,
      rpc_deps_dag,
      services,
      rpcs,
      x_nodes,
      x_net_nodes,
      feature_cols,
      root_service,
      pn_sigma=0.01,
      latent_dim=1,
      alpha=0.9,
      beta=1,
      gamma=0.5,
      en_params=(10, 10, 5),
      de_params=(5, 10, 10),
      pn_params=(5, 5, 5),
  ):

    super(GraphicalCVAE, self).__init__()

    self.en_params = en_params
    self.de_params = de_params
    self.rpc_deps_dag = rpc_deps_dag
    self.services = services
    self.rpcs = rpcs
    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    self.latent_dim = latent_dim
    self.pn_params = pn_params
    self.feature_cols = feature_cols
    self.pn_sigma = pn_sigma
    self.x_nodes = x_nodes
    self.x_net_nodes = x_net_nodes
    self.root_service = root_service

    # Network layers for all services
    self.en_layers = nn.ModuleDict()
    self.de_layers = nn.ModuleDict()
    self.pn_layers = nn.ModuleDict()

    self.y_nodes = self.rpcs
    self.z_nodes = self.services

    # Add networking latency nodes
    client_y_nodes = [r.replace('_client', '') for r in self.y_nodes if r.endswith('client')]
    self.y_net_nodes = [r + '_net_req' for r in client_y_nodes] + [r + '_net_resp' for r in client_y_nodes]
    self.z_net_nodes = [s + '_net' for s in self.services]

    # Construct the encoder
    # Determine the inputs of the encoder
    self.en_input_y_nodes = {}
    self.en_input_x_nodes = {}
    self.en_input_dim = {}

    for z_node in self.z_nodes:
      # Child nodes
      y_child_nodes = [p for p in sorted(list(rpc_deps_dag.successors(z_node))) if p in self.y_nodes]
      # Patent of child nodes
      y_child_parent_nodes = []
      for y in y_child_nodes:
        y_child_parent_nodes += [p for p in sorted(list(rpc_deps_dag.predecessors(y))) if p in self.y_nodes]
      self.en_input_y_nodes[z_node] = y_child_nodes + y_child_parent_nodes
      # Add network nodes
      for y in y_child_nodes:
        if y.endswith('client'):
          self.en_input_y_nodes[z_node] += [y.replace('client', 'net_req'), y.replace('client', 'net_resp')]

      self.en_input_x_nodes[z_node] = [s + '_' + x for (s, x) in product([z_node], self.x_nodes)]
      self.en_input_dim[z_node] = DIST_DIM * len(self.en_input_y_nodes[z_node]) + len(self.en_input_x_nodes[z_node])


      # Construct the encoder layers
      self.en_layers[z_node] = nn.ModuleList()
      self.en_layers[z_node].append(nn.Linear(self.en_input_dim[z_node], self.en_params[0]))
      # self.en_layers[z_node].append(nn.BatchNorm1d(num_features=self.en_params[0]))
      self.en_layers[z_node].append(nn.ReLU())
      # self.en_layers[z_node].append(nn.Dropout(p=0.2))
      for i in range(0, len(self.en_params) - 1):
        self.en_layers[z_node].append(nn.Linear(self.en_params[i], self.en_params[i + 1]))
        # self.en_layers[z_node].append(nn.BatchNorm1d(num_features=self.en_params[i + 1]))
        self.en_layers[z_node].append(nn.ReLU())
        # self.en_layers[z_node].append(nn.Dropout(p=0.2))
      self.en_layers[z_node].append(nn.Linear(self.en_params[-1], self.latent_dim))
      self.en_layers[z_node].append(nn.Linear(self.en_params[-1], self.latent_dim))

    # Net nodes
    for z_net_node in self.z_net_nodes:
      z_node = z_net_node[:-4]
      client_y_nodes = [p for p in sorted(list(rpc_deps_dag.successors(z_node))) if p in self.y_nodes and p.endswith('client')]
      server_y_nodes = [p for p in sorted(list(rpc_deps_dag.successors(z_node))) if p in self.y_nodes and p.endswith('server')]

      if z_node == self.root_service:
        self.en_input_y_nodes[z_net_node] = [r.replace('client', 'net_req') for r in client_y_nodes]
      else:
        self.en_input_y_nodes[z_net_node] = [r.replace('client', 'net_req') for r in client_y_nodes] + [r.replace('server', 'net_resp') for r in server_y_nodes]
      self.en_input_x_nodes[z_net_node] = [s + '_' + x for (s, x) in product([z_node], self.x_net_nodes)]
      self.en_input_dim[z_net_node] = DIST_DIM * len(self.en_input_y_nodes[z_net_node]) + len(self.en_input_x_nodes[z_net_node])

      # Construct the encoder layers
      self.en_layers[z_net_node] = nn.ModuleList()
      self.en_layers[z_net_node].append(nn.Linear(self.en_input_dim[z_net_node], self.en_params[0]))
      # self.en_layers[z_net_node].append(nn.BatchNorm1d(num_features=self.en_params[0]))
      self.en_layers[z_net_node].append(nn.ReLU())
      # self.en_layers[z_net_node].append(nn.Dropout(p=0.2))
      for i in range(0, len(self.en_params) - 1):
        self.en_layers[z_net_node].append(nn.Linear(self.en_params[i], self.en_params[i + 1]))
        # self.en_layers[z_net_node].append(nn.BatchNorm1d(num_features=self.en_params[i + 1]))
        self.en_layers[z_net_node].append(nn.ReLU())
        # self.en_layers[z_net_node].append(nn.Dropout(p=0.2))
      self.en_layers[z_net_node].append(nn.Linear(self.en_params[-1], self.latent_dim))
      self.en_layers[z_net_node].append(nn.Linear(self.en_params[-1], self.latent_dim))


    self.de_input_y_nodes = {}
    self.de_input_z_nodes = {}
    self.de_input_x_nodes = {}
    self.de_input_dim = {}

    # Construct the decoder for each latency y_node
    for y_node in self.y_nodes:
      self.de_input_y_nodes[y_node] = [p for p in sorted(list(rpc_deps_dag.predecessors(y_node))) if p in self.y_nodes]
      # req and resp networking latency will only affact the client side latency
      if y_node.endswith('client'):
        self.de_input_y_nodes[y_node] += [y_node.replace('client', 'net_req'), y_node.replace('client', 'net_resp')]
      self.de_input_z_nodes[y_node] = [p for p in sorted(list(rpc_deps_dag.predecessors(y_node))) if p in self.z_nodes]
      self.de_input_x_nodes[y_node] = [s + '_' + x for (s, x) in product(self.de_input_z_nodes[y_node], self.x_nodes)]
      self.de_input_dim[y_node] = DIST_DIM * len(self.de_input_y_nodes[y_node]) + self.latent_dim * len(self.de_input_z_nodes[y_node]) + len(self.de_input_x_nodes[y_node])

      # Construct decoder layers
      self.de_layers[y_node] = nn.ModuleList()
      self.de_layers[y_node].append(nn.Linear(self.de_input_dim[y_node], self.de_params[0]))
      # self.de_layers[y_node].append(nn.BatchNorm1d(num_features=self.de_params[0]))
      self.de_layers[y_node].append(nn.ReLU())
      # self.de_layers[y_node].append(nn.Dropout(p=0.2))
      for i in range(0, len(self.de_params) - 1):
        self.de_layers[y_node].append(nn.Linear(self.de_params[i], self.de_params[i + 1]))
        # self.de_layers[y_node].append(nn.BatchNorm1d(num_features=self.de_params[i + 1]))
        self.de_layers[y_node].append(nn.ReLU())
        # self.de_layers[y_node].append(nn.Dropout(p=0.2))
      self.de_layers[y_node].append(nn.Linear(self.de_params[-1], DIST_DIM))

    # Construct the decoder for each latency y_net_node
    for y_net_node in self.y_net_nodes:
      self.de_input_y_nodes[y_net_node] = []
      if y_net_node.endswith('req'):
        y_node = y_net_node.replace('net_req', 'client')
      elif y_net_node.endswith('resp'):
        y_node = y_net_node.replace('net_resp', 'server')
      self.de_input_z_nodes[y_net_node] = [p + '_net' for p in sorted(list(rpc_deps_dag.predecessors(y_node))) if p in self.z_nodes]
      self.de_input_x_nodes[y_net_node] = [s.replace('_net', '') + '_' + x for (s, x) in product(self.de_input_z_nodes[y_net_node], self.x_net_nodes)]
      self.de_input_dim[y_net_node] = self.latent_dim * len(self.de_input_z_nodes[y_net_node]) + len(self.de_input_x_nodes[y_net_node])

      # Construct decoder layers
      self.de_layers[y_net_node] = nn.ModuleList()
      self.de_layers[y_net_node].append(nn.Linear(self.de_input_dim[y_net_node], self.de_params[0]))
      # self.de_layers[y_net_node].append(nn.BatchNorm1d(num_features=self.de_params[0]))
      self.de_layers[y_net_node].append(nn.ReLU())
      # self.de_layers[y_net_node].append(nn.Dropout(p=0.2))
      for i in range(0, len(self.de_params) - 1):
        self.de_layers[y_net_node].append(nn.Linear(self.de_params[i], self.de_params[i + 1]))
        # self.de_layers[y_net_node].append(nn.BatchNorm1d(num_features=self.de_params[i + 1]))
        self.de_layers[y_net_node].append(nn.ReLU())
        # self.de_layers[y_net_node].append(nn.Dropout(p=0.2))
      self.de_layers[y_net_node].append(nn.Linear(self.de_params[-1], DIST_DIM))


    self.pn_input_x_nodes = {}

    # Construct the prior network layers
    for z_node in self.z_nodes:
      self.pn_input_x_nodes[z_node] = [z_node + '_' + x for x in self.x_nodes]

      self.pn_layers[z_node] = nn.ModuleList()
      self.pn_layers[z_node].append(nn.Linear(len(self.pn_input_x_nodes[z_node]), self.pn_params[0]))
      # self.pn_layers[z_node].append(nn.BatchNorm1d(num_features=self.pn_params[0]))
      self.pn_layers[z_node].append(nn.ReLU())
      # self.pn_layers[z_node].append(nn.Dropout(p=0.2))
      for i in range(0, len(self.pn_params) - 1):
        self.pn_layers[z_node].append(nn.Linear(self.pn_params[i], self.pn_params[i + 1]))
        # self.pn_layers[z_node].append(nn.BatchNorm1d(num_features=self.pn_params[i + 1]))
        self.pn_layers[z_node].append(nn.ReLU())
        # self.pn_layers[z_node].append(nn.Dropout(p=0.2))

      self.pn_layers[z_node].append(nn.Linear(self.pn_params[-1], self.latent_dim))
      self.pn_layers[z_node].append(nn.Linear(self.pn_params[-1], self.latent_dim))


    # Construct the prior network layers for networking
    for z_net_node in self.z_net_nodes:
      z_node = z_net_node[:-4]
      self.pn_input_x_nodes[z_net_node] = [z_node + '_' + x for x in self.x_net_nodes]

      self.pn_layers[z_net_node] = nn.ModuleList()
      self.pn_layers[z_net_node].append(nn.Linear(len(self.pn_input_x_nodes[z_net_node]), self.pn_params[0]))
      # self.pn_layers[z_net_node].append(nn.BatchNorm1d(num_features=self.pn_params[0]))
      self.pn_layers[z_net_node].append(nn.ReLU())
      # self.pn_layers[z_net_node].append(nn.Dropout(p=0.2))
      for i in range(0, len(self.pn_params) - 1):
        self.pn_layers[z_net_node].append(nn.Linear(self.pn_params[i], self.pn_params[i + 1]))
        # self.pn_layers[z_net_node].append(nn.BatchNorm1d(num_features=self.pn_params[i + 1]))
        self.pn_layers[z_net_node].append(nn.ReLU())
        # self.pn_layers[z_net_node].append(nn.Dropout(p=0.2))

      self.pn_layers[z_net_node].append(nn.Linear(self.pn_params[-1], self.latent_dim))
      self.pn_layers[z_net_node].append(nn.Linear(self.pn_params[-1], self.latent_dim))


  def rpc_to_service(self, rpc):
    p_list = []
    for p in sorted(list(self.rpc_deps_dag.predecessors(rpc))):
      if self.rpc_deps_dag.in_degree(p) == 0:
        p_list.append(p)
    return p_list

  def encoder(self, z_node, data):
    en_input_y_tensor = select_columns(self.feature_cols, self.en_input_y_nodes[z_node], data)
    en_input_x_tensor = select_columns(self.feature_cols, self.en_input_x_nodes[z_node], data)
    en_input_tensor = torch.cat((en_input_y_tensor, en_input_x_tensor), axis=1)
    h = self.en_layers[z_node][0](en_input_tensor)
    for i in range(1, 2 * len(self.en_params)):
      h = self.en_layers[z_node][i](h)
    mu = self.en_layers[z_node][-2](h)
    logvar = self.en_layers[z_node][-1](h)
    return mu, logvar

  def decoder(self, y_node, de_input_x_tensor, de_input_z_tensor, de_input_y_tensors):
    if len(self.de_input_y_nodes[y_node]):
      de_input_y_tensor = torch.cat([de_input_y_tensors[r] for r in self.de_input_y_nodes[y_node]], axis=1)
      de_input_tensor = torch.cat([de_input_x_tensor, de_input_y_tensor, de_input_z_tensor], axis=1)
    else:
      de_input_tensor = torch.cat([de_input_x_tensor, de_input_z_tensor], axis=1)
    h = self.de_layers[y_node][0](de_input_tensor)
    for i in range(1, 2 * len(self.de_params)):
      h = self.de_layers[y_node][i](h)
    result = self.de_layers[y_node][-1](h)
    return result

  def prior_network(self, z_node, data):
    pn_input_tensor = select_columns(self.feature_cols, self.pn_input_x_nodes[z_node], data)
    h = self.pn_layers[z_node][0](pn_input_tensor)
    for i in range(1, 2 * len(self.pn_params)):
      h = self.pn_layers[z_node][i](h)
    pn_mu = self.pn_layers[z_node][-2](h)
    pn_logvar = self.pn_layers[z_node][-1](h)
    return pn_mu, pn_logvar

  def forward(self, data):
    mus = {}
    logvars = {}
    pn_mus = {}
    pn_logvars = {}

    de_en_de_output_y_tensors = {}
    de_pn_de_output_y_tensors = {}
    de_en_raw_output_y_tensors = {}
    de_pn_raw_output_y_tensors = {}
    de_raw_input_y_tensors = {}


    for y in self.y_net_nodes + self.y_nodes:
      de_raw_input_y_tensors[y] = select_columns(self.feature_cols, [y], data)

    for z in self.z_net_nodes + self.z_nodes:
      mus[z], logvars[z] = self.encoder(z, data)
      pn_mus[z], pn_logvars[z] = self.prior_network(z, data)
    
    for y in self.y_net_nodes + self.y_nodes:
      # Reconstruction with z from encoder and y from decoder
      de_en_de_input_x_tensor = select_columns(self.feature_cols, self.de_input_x_nodes[y], data)
      de_en_de_input_z_tensor = torch.cat([self.reparametrize(mus[s], logvars[s]) for s in self.de_input_z_nodes[y]], axis=1)
      de_en_de_output_y_tensors[y] = self.decoder(y, de_en_de_input_x_tensor, de_en_de_input_z_tensor, de_en_de_output_y_tensors)
      # Reconstruction with z from prior network and y from decoder
      de_pn_de_input_x_tensor = de_en_de_input_x_tensor
      de_pn_de_input_z_tensor = torch.cat([self.reparametrize(pn_mus[s], pn_logvars[s]) for s in self.de_input_z_nodes[y]], axis=1)
      de_pn_de_output_y_tensors[y] = self.decoder(y, de_pn_de_input_x_tensor, de_pn_de_input_z_tensor, de_pn_de_output_y_tensors)
      # Reconstruction with z from encoder and y from raw data
      de_en_raw_input_x_tensor = de_en_de_input_x_tensor
      de_en_raw_input_z_tensor = de_en_de_input_z_tensor
      de_en_raw_output_y_tensors[y] = self.decoder(y, de_en_raw_input_x_tensor, de_en_raw_input_z_tensor, de_raw_input_y_tensors)
      # Reconstruction with z from prior network and y from raw data
      de_pn_raw_input_x_tensor = de_en_de_input_x_tensor
      de_pn_raw_input_z_tensor = de_pn_de_input_z_tensor
      de_pn_raw_output_y_tensors[y] = self.decoder(y, de_pn_raw_input_x_tensor, de_pn_raw_input_z_tensor, de_raw_input_y_tensors)

    # reconstructed latency
    recon_en_de_y = [de_en_de_output_y_tensors[y] for y in self.y_net_nodes + self.y_nodes]
    recon_en_de_y = torch.cat(recon_en_de_y, axis=1)

    # reconstructed latency via prior network
    recon_pn_de_y = [de_pn_de_output_y_tensors[y] for y in self.y_net_nodes + self.y_nodes]
    recon_pn_de_y = torch.cat(recon_pn_de_y, axis=1)

    # reconstructed latency
    recon_en_raw_y = [de_en_raw_output_y_tensors[y] for y in self.y_net_nodes + self.y_nodes]
    recon_en_raw_y = torch.cat(recon_en_raw_y, axis=1)

    # reconstructed latency via prior network
    recon_pn_raw_y = [de_pn_raw_output_y_tensors[y] for y in self.y_net_nodes + self.y_nodes]
    recon_pn_raw_y = torch.cat(recon_pn_raw_y, axis=1)

    return recon_en_de_y, recon_pn_de_y, recon_en_raw_y, recon_pn_raw_y, mus, logvars, pn_mus, pn_logvars

  def reparametrize(self, mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)

  def kl_divergence(self, p_mu, p_logvar, q_mu, q_logvar):
    p_var = p_logvar.exp()
    q_var = q_logvar.exp()
    det_p = torch.prod(p_var, dim=1).view(-1, 1)
    det_q = torch.prod(q_var, dim=1).view(-1, 1)
    d = torch.tensor(p_mu.shape[1]).view(-1, 1).repeat(p_mu.shape[0], 1).float()
    tr_pq = torch.div(p_var, q_var).sum(dim=1).view(-1, 1)
    mu_squa = (p_mu - q_mu).pow(2).div(q_var).sum(dim=1).view(-1, 1)
    return 0.5 * (torch.log(det_q / det_p) - d + tr_pq + mu_squa).sum()

  def loss_function(self, recon_en_de_y, recon_pn_de_y, recon_en_raw_y, recon_pn_raw_y, data, mus, logvars, pn_mus, pn_logvars):
    target_y = select_columns(self.feature_cols, self.y_net_nodes + self.y_nodes, data)
    recon_en_de_error = F.mse_loss(recon_en_de_y, target_y, reduction='sum')
    recon_pn_de_error = F.mse_loss(recon_pn_de_y, target_y, reduction='sum')
    recon_en_raw_error = F.mse_loss(recon_en_raw_y, target_y, reduction='sum')
    recon_pn_raw_error = F.mse_loss(recon_pn_raw_y, target_y, reduction='sum')
    kld = 0
    for s in self.services:
      kld += self.kl_divergence(mus[s], logvars[s], pn_mus[s], pn_logvars[s])
    return self.alpha * (self.gamma * recon_en_de_error + \
        (1 - self.gamma) * recon_en_raw_error + self.beta * kld) + \
        (1 - self.alpha) * (self.gamma * recon_pn_de_error +
        (1 - self.gamma) * recon_pn_raw_error)
        