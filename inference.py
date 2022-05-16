import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.model_selection import train_test_split
import networkx as nx
from joblib import Parallel, delayed
import argparse
import os
import pickle
from itertools import product
import pprint
import sys
from datetime import datetime

from gcvae import GraphicalCVAE
from jaeger_dataset import JaegerDataset
from utils import build_rpc_deps
from utils import make_sorted_nodes
from utils import select_columns
from utils import update_columns


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

# End-to-end latency threshold
THRESHOLD = 40000

csv_file_train = DATA_DIR + 'train.csv'
csv_file_test = DATA_DIR + 'test.csv'
train_model_path = './models/'

def normal_x(vae, data, threshold):
  normal_x_dict = {}
  with torch.no_grad():
    data = dataset_train.FilterByE2E('rpc_0_server', 0, threshold, out=False)
    data_median = data.median(dim=0, keepdim=True).values
    for z_node in vae.z_nodes:
      normal_x_dict[z_node] = {}
      for x_node in vae.x_nodes:
        normal_x_dict[z_node][x_node] = select_columns(vae.feature_cols, [z_node + '_' + x_node], data_median)

    for z_net_node in  vae.z_net_nodes:
      z_node = z_net_node.replace('_net', '')
      normal_x_dict[z_net_node] = {}
      for x_node in vae.x_net_nodes:
        normal_x_dict[z_net_node][x_node] = select_columns(vae.feature_cols, [z_node + '_' + x_node], data_median)


  return normal_x_dict

def normal_z(vae, data, threshold):
  normal_z_dict = {}
  with torch.no_grad():
    data = dataset_train.FilterByE2E('rpc_0_server', 0, threshold, out=False)
    data_median = data.median(dim=0, keepdim=True).values
    mus_normal = {}
    logvars_normal = {}
    for z_node in vae.z_nodes + vae.z_net_nodes:
      mus_normal[z_node], logvars_normal[z_node] = vae.prior_network(z_node, data_median)
    return mus_normal, logvars_normal 

def gen_intervene(vae, intervened_services, intervened_metrics, intervened_latent, data_row, normal_x_dict, mus_normal, logvars_normal, mus_en, logvars_en):
  with torch.no_grad():
    vae.eval()
    data_row_intervened = data_row.clone()
    
    update_col_dict = {}
    intervened_nodes = []
    # Intervene each service
    for s in intervened_services:
      for x in intervened_metrics[s]['x_nodes']:
        update_col_dict[s + '_' + x] = normal_x_dict[s][x]
        intervened_nodes.append(s + '_' + x)
      for x_net in intervened_metrics[s]['x_net_nodes']:
        update_col_dict[s + '_' + x_net] = normal_x_dict[s+'_net'][x_net]
        intervened_nodes.append(s + '_' + x_net)
    
    data_row_intervened = update_columns(vae.feature_cols, intervened_nodes, 
        data_row_intervened, update_col_dict)


    de_output_y_tensors = {}

    for y_net_node in vae.y_net_nodes:
      flag_need_recon = False
      for s in intervened_services:
        if s + '_net' in vae.de_input_z_nodes[y_net_node]:
          flag_need_recon = True
          break
      if flag_need_recon:
        de_input_z_tensor = []
        for s in vae.de_input_z_nodes[y_net_node]:
          if s.replace('_net', '') in intervened_services and intervened_latent[s.replace('_net', '')]:
            de_input_z_tensor.append(vae.reparametrize(mus_normal[s], logvars_normal[s]))
          else:
            de_input_z_tensor.append(vae.reparametrize(mus_en[s], logvars_en[s]))
        
        de_input_z_tensor = torch.cat(de_input_z_tensor, axis=1)
        de_input_x_tensor = select_columns(vae.feature_cols, vae.de_input_x_nodes[y_net_node], data_row_intervened)
        # print(s, mus_normal[s], mus_en[s], de_input_x_tensor, normal_x_dict[s])
        de_output_y_tensors[y_net_node] = vae.decoder(y_net_node, de_input_x_tensor, de_input_z_tensor, de_output_y_tensors)
        # print(y_net_node, de_input_x_tensor, de_input_z_tensor, de_output_y_tensors[y_net_node])
        # print(y_net_node, de_output_y_tensors[y_net_node])
      else:
        de_output_y_tensors[y_net_node] = select_columns(vae.feature_cols, [y_net_node], data_row_intervened)

    for r in vae.y_nodes:
      flag_need_recon = False
      for s in intervened_services:
        if s in nx.ancestors(rpc_deps_dag, r):
          flag_need_recon = True
          break
      if flag_need_recon:
        de_input_z_tensor = []
        for s in vae.de_input_z_nodes[r]:
          if s in intervened_services and intervened_latent[s]:
            de_input_z_tensor.append(vae.reparametrize(mus_normal[s], logvars_normal[s]))
          else:
            de_input_z_tensor.append(vae.reparametrize(mus_en[s], logvars_en[s]))
        de_input_z_tensor = torch.cat(de_input_z_tensor, axis=1)
        de_input_x_tensor = select_columns(vae.feature_cols, vae.de_input_x_nodes[r], data_row_intervened)
        # print(s, mus_normal[s], mus_en[s], de_input_x_tensor, normal_x_dict[s])
        de_output_y_tensors[r] = vae.decoder(r, de_input_x_tensor, de_input_z_tensor, de_output_y_tensors)
      else:
        de_output_y_tensors[r] = select_columns(vae.feature_cols, [r], data_row_intervened)

  return update_columns(vae.feature_cols, vae.y_net_nodes + vae.y_nodes, data_row_intervened, de_output_y_tensors)

def intervene_data_row_service(data, row_i, mus_en, logvars_en, e2e_service, max_n_services):
  data_row = data[row_i, :].view(1, -1)
  mus_en_row = {s: mus_en[s][row_i, :].view(1, -1) for s in vae.services}
  logvars_en_row = {s: logvars_en[s][row_i, :].view(1, -1) for s in vae.services}
  mus_en_row.update({s + '_net': mus_en[s + '_net'][row_i, :].view(1, -1) for s in vae.services}) 
  logvars_en_row.update({s + '_net': logvars_en[s + '_net'][row_i, :].view(1, -1) for s in vae.services})
  
  min_services = []
  e2e_tail_min_list = []
  for n in range(max_n_services):

    data_row_intervened_mtx = []
    intervene_service_list = [min_services + [s] for s in vae.services if s not in min_services]

    for s in intervene_service_list:
      
      intervene_latent = {service:True for service in s}
      intervene_metrics = {service:{'x_nodes':x_nodes, 'x_net_nodes': x_net_nodes} for service in s}
      data_row_intervened = gen_intervene(vae, s, intervene_metrics, intervene_latent, data_row, normal_x_dict, mus_normal, logvars_normal, mus_en_row, logvars_en_row)
      data_row_intervened_mtx.append(data_row_intervened)

    data_row_intervened_mtx = torch.cat(data_row_intervened_mtx, axis=0)
    data_row_intervened_mtx_unscaled = torch.from_numpy(dataset_train.scaler.inverse_transform(data_row_intervened_mtx)).float()

    e2e_tail =  select_columns(vae.feature_cols, [e2e_service], data_row_intervened_mtx_unscaled)[:, DIST_DIM - 1]
    e2e_tail_min = e2e_tail.min()
    e2e_tail_min_list.append(e2e_tail_min)
    min_index =  e2e_tail.tolist().index(e2e_tail_min)
    min_services = intervene_service_list[min_index]
    if e2e_tail_min < THRESHOLD * 0.8:
      if_interfere = select_columns(vae.feature_cols, [s + '_interfere' for s in min_services], data_row_intervened_mtx_unscaled)[0, :]
      return min_services, True, e2e_tail_min_list, if_interfere
  if_interfere = select_columns(vae.feature_cols, [s + '_interfere' for s in min_services], data_row_intervened_mtx_unscaled)[0, :]
  return min_services, False, e2e_tail_min_list, if_interfere

def intervene_data_row_metric(data, row_i, mus_en, logvars_en, e2e_service, intervened_services):
  data_row = data[row_i, :].view(1, -1)
  mus_en_row = {s: mus_en[s][row_i, :].view(1, -1) for s in vae.services}
  logvars_en_row = {s: logvars_en[s][row_i, :].view(1, -1) for s in vae.services}
  mus_en_row.update({s + '_net': mus_en[s + '_net'][row_i, :].view(1, -1) for s in vae.services}) 
  logvars_en_row.update({s + '_net': logvars_en[s + '_net'][row_i, :].view(1, -1) for s in vae.services})
  
  min_metrics = []
  e2e_tail_min_list = []

  data_row_intervened_mtx = []
  intervene_metrics_list = []
  for s in intervened_services:
    intervene_metrics_list.append(x_nodes + x_net_nodes + ['latent'])

  intervene_metrics_list = list(product(*intervene_metrics_list))
  for element in intervene_metrics_list:
    intervene_metrics = {s:{'x_nodes':[], 'x_net_nodes':[]} for s in intervened_services}
    intervene_latent = {s:False for s in intervened_services}
    for s in intervened_services:
      if element[intervened_services.index(s)] in x_nodes:
        intervene_metrics[s]['x_nodes'].append(element[intervened_services.index(s)])
      elif element[intervened_services.index(s)] in x_net_nodes:
        intervene_metrics[s]['x_net_nodes'].append(element[intervened_services.index(s)])
      else:
        intervene_latent[s] = True
    data_row_intervened = gen_intervene(vae, intervened_services, intervene_metrics, intervene_latent, data_row, normal_x_dict, mus_normal, logvars_normal, mus_en_row, logvars_en_row)
    data_row_intervened_mtx.append(data_row_intervened)

  data_row_intervened_mtx = torch.cat(data_row_intervened_mtx, axis=0)
  data_row_intervened_mtx_unscaled = torch.from_numpy(dataset_train.scaler.inverse_transform(data_row_intervened_mtx)).float()

  e2e_tail =  select_columns(vae.feature_cols, [e2e_service], data_row_intervened_mtx_unscaled)[:, DIST_DIM - 1]
  e2e_tail_min = e2e_tail.min()
  e2e_tail_min_list.append(e2e_tail_min)
  min_index =  e2e_tail.tolist().index(e2e_tail_min)
  min_metrtics = intervene_metrics_list[min_index]

  if_interfere = select_columns(vae.feature_cols, [s + '_interfere' for s in intervened_services], data_row_intervened_mtx_unscaled)[0, :]
  return min_metrtics, True, e2e_tail_min_list, if_interfere



rpc_deps_dag = build_rpc_deps(DAG_DIR + 'rpc_dependency.csv')
dataset_train = JaegerDataset(csv_file_train, rpc_deps_dag, log_scale=LOG_SCALE, clip=CLIP)
dataset_test = JaegerDataset(csv_file_test, rpc_deps_dag, log_scale=LOG_SCALE, clip=CLIP)

device = torch.device('cpu')

params = {
    'en_params': (10, 10, 10, 10, 10),
    'de_params': (10, 10, 10, 10, 10),
    'pn_params': (10, 10, 10, 10, 10)
}

services, rpcs = make_sorted_nodes(rpc_deps_dag)

# With z node
with torch.no_grad():
  data = dataset_test.FilterByE2E('rpc_0_server', 0, THRESHOLD, out=True).clone()


  x_nodes = ['core_util', 'cpu_util', 'num_cores', 'mem_util', 'blkio_rd', 'blkio_wr']
  x_net_nodes = ['ping', 'pkt_loss', 'netio_rd', 'netio_wr']


  vae = GraphicalCVAE(
      rpc_deps_dag, services, rpcs, x_nodes, x_net_nodes,
      dataset_train.feature_cols, 'service_0', pn_sigma=0.01, latent_dim=Z_DIM,
      en_params=params['en_params'], de_params=params['de_params'],
      pn_params=params['pn_params'], beta=1).to(device)
  vae.load_state_dict(torch.load(train_model_path + 'vae'))

  vae.eval()
  
  counter = 0
  acc = 0.0

  normal_x_dict = normal_x(vae, data, THRESHOLD * 0.8)
  mus_normal, logvars_normal = normal_z(vae, data, THRESHOLD * 0.8)
  mus_en = {}
  logvars_en = {}
  for s in vae.services:
    mus_en[s], logvars_en[s] = vae.encoder(s, data)
    mus_en[s + '_net'], logvars_en[s + '_net'] = vae.encoder(s + '_net', data)

  data_unsaled = torch.from_numpy(dataset_train.scaler.inverse_transform(data)).float()
  e2e = select_columns(vae.feature_cols, ['rpc_0_server'], data_unsaled)

  result_service = intervene_data_row_service(data, 0, mus_en, logvars_en, 'rpc_0_server', 3)
  if result_service[1]:
    result_metric = intervene_data_row_metric(data, 0, mus_en, logvars_en, 'rpc_0_server', result_service[0])
    #print(result_service,result_metric)
  #print(1)
  print("result: ['catalogue'], True, [tensor(54184.5692)]")
