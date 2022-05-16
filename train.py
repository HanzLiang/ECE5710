import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import pandas as pd
import numpy as np
import networkx as nx
import argparse
import os
import pickle
from itertools import product
import sys
import logging

from gcvae import GraphicalCVAE
from jaeger_dataset import JaegerDataset
from utils import build_rpc_deps
from utils import make_sorted_nodes


parser = argparse.ArgumentParser()
parser.add_argument('-e', dest='n_epochs', type=int, default=100)
parser.add_argument('--incremental', dest='incremental', action='store_true')
parser.add_argument('-r', dest='rate', type=float, default=1e-2)
parser.add_argument('--interval', dest='interval', type=int, default=30)
args = parser.parse_args()

n_epochs = args.n_epochs
incremental = args.incremental
rate = args.rate
interval = args.interval

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


logger = logging.getLogger('gcvae_train')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def train_epoch(model, optimizer, epoch):
  model.train()
  train_loss = 0
  for batch_idx, batch_data in enumerate(train_loader):
    batch_data = batch_data.to(device)
    optimizer.zero_grad()
    recon_en_de_y, recon_pn_de_y, recon_en_raw_y, recon_pn_raw_y, mu, logvar, pn_mu, pn_logvar = model(batch_data)
    loss = model.loss_function(recon_en_de_y, recon_pn_de_y, recon_en_raw_y, recon_pn_raw_y, batch_data, mu, logvar, pn_mu, pn_logvar)
    loss.backward()
    train_loss += loss.item()
    optimizer.step()
  logger.info('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss 
      / len(train_loader.dataset)))


def test_epoch(model, epoch):
  model.eval()
  test_loss = 0
  with torch.no_grad():
    for batch_idx, batch_data in enumerate(test_loader):
      batch_data = batch_data.to(device)
      recon_en_de_y, recon_pn_de_y, recon_en_raw_y, recon_pn_raw_y, mu, logvar, pn_mu, pn_logvar = model(batch_data)
      loss = model.loss_function(recon_en_de_y, recon_pn_de_y, recon_en_raw_y, recon_pn_raw_y, batch_data, mu, logvar, pn_mu, pn_logvar)
      test_loss += loss.item()
  test_loss /= len(test_loader.dataset)
  logger.info('====> Test set loss: {:.4f}'.format(test_loss))


rpc_deps_dag = build_rpc_deps(DAG_DIR + 'rpc_dependency.csv')

dataset_train = JaegerDataset(csv_file_train, rpc_deps_dag,  clip=CLIP, log_scale=LOG_SCALE)
dataset_test = JaegerDataset(csv_file_test, rpc_deps_dag, clip=CLIP, log_scale=LOG_SCALE)

train_loader = DataLoader(dataset=dataset_train, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=dataset_test, batch_size=1024, shuffle=True)
device = torch.device('cpu')

params = {
    'en_params': (10, 10, 10, 10, 10),
    'de_params': (10, 10, 10, 10, 10),
    'pn_params': (10, 10, 10, 10, 10),
    'lr': rate,
    'n_epochs': n_epochs,
}


def train_model():

  x_nodes = ['core_util', 'cpu_util', 'num_cores', 'mem_util', 'blkio_rd', 'blkio_wr']
  x_net_nodes = ['ping', 'pkt_loss', 'netio_rd', 'netio_wr']

  services, rpcs = make_sorted_nodes(rpc_deps_dag)

  vae = GraphicalCVAE(rpc_deps_dag, services, rpcs, x_nodes, x_net_nodes,
      dataset_train.feature_cols, root_service='service_0', pn_sigma=0.1, 
      latent_dim=Z_DIM, en_params=params['en_params'], de_params=params['de_params'],
      pn_params=params['pn_params'], alpha=0.9, beta=1, gamma=0.5).to(device)

  optimizer = optim.Adam(vae.parameters(), lr=params['lr'])
  if incremental:
    vae.load_state_dict(torch.load(train_model_path + 'vae'))

  for epoch in range(1, params['n_epochs'] + 1):
    train_epoch(vae, optimizer, epoch,)
    test_epoch(vae, epoch)

  torch.save(vae.state_dict(), train_model_path + 'vae')

try:
  os.makedirs(train_model_path)
except:
  pass
  
train_model()