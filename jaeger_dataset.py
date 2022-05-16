import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.model_selection import train_test_split
import networkx as nx
import pickle

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

def makeArray(text):
  return np.fromstring(text[1:-1], sep=', ')


def MakeScaler(csv_file, clip, log_scale, rpc_deps_dag):
  try:
    with open('scaler.pkl', 'rb') as f:
      scaler = pickle.load(f)
      return scaler
  except:
    df = pd.read_csv(csv_file, engine='python', index_col=0,sep='/')
    if 'quantile' in df.columns:
      df.drop(columns=['quantile'], axis=1, inplace=True)
    feature_cols = {}
    label_cols = []
    idx = 0
    mtx = None
    for col in df.columns:
      if col.endswith('server') or col.endswith('client'):
        if col in rpc_deps_dag.nodes():
          feature_cols[col] = [idx, idx + DIST_DIM]
          latency_col = np.vstack(df[col].apply(makeArray).values)
          idx += DIST_DIM
          if mtx is None:
            mtx = latency_col
          else:
            mtx = np.concatenate((mtx, latency_col), axis=1)

      elif col.endswith('req') or col.endswith('resp'):
        feature_cols[col] = [idx, idx + DIST_DIM]
        latency_col = np.vstack(df[col].apply(makeArray).values)
        idx += DIST_DIM
        if mtx is None:
          mtx = latency_col
        else:
          mtx = np.concatenate((mtx, latency_col), axis=1)

      else:
        feature_cols[col] = idx
        idx += 1
        hint_col = df[col].values.reshape(-1, 1)
        if mtx is None:
          mtx = hint_col
        else:
          mtx = np.concatenate((mtx, hint_col,), axis=1)
    if clip:
      mtx = np.clip(mtx, a_min=0, a_max=MAX_CLIP)
    if log_scale:
      transformer = FunctionTransformer(np.log1p, validate=True)
      mtx = transformer.transform(mtx)
    scaler = StandardScaler()
    scaler.fit(mtx)
    with open('scaler.pkl', 'wb') as f:
      pickle.dump(scaler, f)
    return scaler


class JaegerDataset(Dataset):
  def __init__(self, csv_file, rpc_deps_dag, log_scale=False, clip=False):
    df = pd.read_csv(csv_file, engine='python', index_col=0,sep='/')
    if 'quantile' in df.columns:
      df.drop(columns=['quantile'], axis=1, inplace=True)
    self.feature_cols = {}
    self.latency_cols = []
    self.label_cols = []
    idx = 0
    mtx = None
    for col in df.columns:
      if col.endswith('server') or col.endswith('client'):
        if col in rpc_deps_dag.nodes():
          self.latency_cols.append(col)
          self.feature_cols[col] = [idx, idx + DIST_DIM]
          latency_col = np.vstack(df[col].apply(makeArray).values)
          idx += DIST_DIM
          if mtx is None:
            mtx = latency_col
          else:
            mtx = np.concatenate((mtx, latency_col), axis=1)

      elif col.endswith('req') or col.endswith('resp'):
        self.latency_cols.append(col)
        self.feature_cols[col] = [idx, idx + DIST_DIM]
        latency_col = np.vstack(df[col].apply(makeArray).values)
        idx += DIST_DIM
        if mtx is None:
          mtx = latency_col
        else:
          mtx = np.concatenate((mtx, latency_col), axis=1)

      else:
        self.feature_cols[col] = idx
        idx += 1
        hint_col = df[col].values.reshape(-1, 1)
        if mtx is None:
          mtx = hint_col
        else:
          mtx = np.concatenate((mtx, hint_col,), axis=1)


    if clip:
      mtx = np.clip(mtx, a_min=0, a_max=MAX_CLIP)
    if log_scale:
      transformer = FunctionTransformer(np.log1p, validate=True)
      mtx = transformer.transform(mtx)
    self.scaler = MakeScaler(csv_file_train, clip, log_scale, rpc_deps_dag)
    self.data = self.scaler.transform(mtx)

    self.data = torch.from_numpy(self.data).float()

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return self.data[idx]

  def FilterByE2E(self, e2e_col, e2e_min=0, e2e_max=0, out=False):
    assert(e2e_col in self.feature_cols)
    if not out:
      if e2e_min and e2e_max:
        data = self.data[self.scaler.inverse_transform(self.data)[:, self.feature_cols[e2e_col][1] - 1] >= e2e_min and self.scaler.inverse_transform(self.data)[:, self.feature_cols[e2e_col][1] - 1] <= e2e_max]
      elif e2e_min:
        data = self.data[self.scaler.inverse_transform(self.data)[:, self.feature_cols[e2e_col][1] - 1] >= e2e_min]
      elif e2e_max:
        data = self.data[self.scaler.inverse_transform(self.data)[:, self.feature_cols[e2e_col][1] - 1] <= e2e_max]
      else:
        data = self.data
      return data
    else:
      if e2e_min and e2e_max:
        data = self.data[self.scaler.inverse_transform(self.data)[:, self.feature_cols[e2e_col][1] - 1] <= e2e_min or self.scaler.inverse_transform(self.data)[:, self.feature_cols[e2e_col][1] - 1] >= e2e_max]
      elif e2e_min:
        data = self.data[self.scaler.inverse_transform(self.data)[:, self.feature_cols[e2e_col][1] - 1] <= e2e_min]
      elif e2e_max:
        data = self.data[self.scaler.inverse_transform(self.data)[:, self.feature_cols[e2e_col][1] - 1] >= e2e_max]
      else:
        data = self.data
      return data
