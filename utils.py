import pandas as pd
import networkx as nx
import torch
from matplotlib import pyplot as plt

DIST_DIM = 2

def build_rpc_deps(rpc_deps_file):
  rpc_deps_df = pd.read_csv(rpc_deps_file)
  rpc_deps_dag = nx.DiGraph()
  for _, row in rpc_deps_df.iterrows():
    rpc_deps_dag.add_edge(row['parent'], row['child'])
  # nx.draw_networkx(rpc_deps_dag)
  # plt.show()
  return rpc_deps_dag

def select_columns(feature_cols, columns, x):
  mtx = []
  dim = 0
  for col in columns:
    try:
      assert (col in feature_cols)
    except:
      print('Fatal:', col, 'not in feature_cols')
      raise
    if col.endswith('server') or col.endswith('client') or col.endswith('req') or col.endswith('resp'):
      mtx.append(x[:, feature_cols[col][0]: feature_cols[col][1]])
      dim += DIST_DIM
    else:
      mtx.append(x[:, feature_cols[col]].reshape(-1, 1))
      dim += 1
  return torch.cat(mtx, axis=1).view(-1, dim)

def update_columns(feature_cols, columns, x, update_x_dict):
  x_copy = x.clone()
  for col in columns:
    try:
      assert (col in feature_cols)
    except:
      print('Fatal:', col, 'not in feature_cols')
      raise
    if col.endswith('server') or col.endswith('client') or col.endswith('req') or col.endswith('resp'):
      x_copy[:, feature_cols[col][0]: feature_cols[col][1]] = update_x_dict[col]
    else:
      x_copy[:, feature_cols[col]] = update_x_dict[col]
  return x_copy

# Topological sort of services and rpcs in the DAG
def make_sorted_nodes(rpc_deps_dag):
  services = list(x for x in rpc_deps_dag.nodes if rpc_deps_dag.in_degree(x)==0)
  sorted_rpcs = [r for r in list(nx.lexicographical_topological_sort(rpc_deps_dag)) if r not in services]
  sorted_services = []
  for r in sorted_rpcs:
    for n in rpc_deps_dag.predecessors(r):
      if (n.startswith('service')) and n not in sorted_services:
          sorted_services.append(n)
  sorted_services.sort()
  return sorted_services, sorted_rpcs
