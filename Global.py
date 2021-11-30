from enum import Enum
import torch
from torch.nn import Module,\
                     ModuleList,\
                     Embedding,\
                     Linear,\
                     Dropout
from openbiolink.obl2021 import OBL2021Dataset, OBL2021Evaluator
import torch.nn.functional as F
from torch_geometric.nn.conv import RGCNConv
from torch_geometric.data import Data
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
import typing 
from typing import NoReturn


class Global(Enum):
  HEAD_INDEX = 0
  RELATION_INDEX = 1
  TAIL_INDEX = 2
  FEATURE_ENG = 'one-hot'
  NUM_RELATIONS = 28
  DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
  MINI_BATCH_SIZE = 32
  DISP_HITS = False
  MANUAL_SEED = 2021
  MARGIN = 35
  NUM_BASES = 3
  REDUCTION = 'sum'


# torch.manual_seed(Global.MANUAL_SEED.value)
dataset = OBL2021Dataset()
evaluator = OBL2021Evaluator()
kg = torch.cat((dataset.training, dataset.validation, dataset.testing), dim=0)
sorted_on_tails_indcs_kg = torch.sort(kg[:, Global.TAIL_INDEX.value])[1]
kg_sorted_tails = kg[sorted_on_tails_indcs_kg]
sorted_on_tails_indcs_kg = None
sorted_on_heads_indcs_kg = torch.sort(kg[:, Global.HEAD_INDEX.value])[1]
kg_sorted_heads = kg[sorted_on_heads_indcs_kg]
sorted_on_heads_indcs_kg = None
features = dataset.candidates.to(Global.DEVICE.value) 
train_set = dataset.training.to(Global.DEVICE.value) # torch.tensor of shape(num_train,3)
val_set = dataset.validation.to(Global.DEVICE.value) # torch.tensor of shape(num_val,3)
test_set = dataset.testing.to(Global.DEVICE.value)   # torch.tensor of shape(num_train,3)
dataset = None
sorted_train_set = train_set[torch.sort(train_set[:, Global.HEAD_INDEX.value])[1]]
