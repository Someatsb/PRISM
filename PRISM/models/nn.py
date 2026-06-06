
from torch_geometric.nn.conv import GCNConv, SAGEConv
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import torch.nn as nn
from torch_geometric.nn import LabelPropagation
from torch_geometric.nn.models import GAT
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCN2Conv, APPNP
from torch_geometric.nn import GATConv as PYGGATConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import models.rev.memgcn as memgcn
from models.rev.rev_layer import SharedDropout
import copy
import tqdm
from dgl import function as fn
from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair
import torch_geometric.utils as utils
import time
from torch.cuda.amp import autocast
from torch_sparse import SparseTensor
import numpy as np
import torch.optim as optim
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


def get_model(args):
    if args.model_name == 'GCN':
        return GCN(args.num_layers, args.input_dim, args.hidden_dimension, args.num_classes, args.dropout, args.norm)
    elif args.model_name == 'GraphSAGE':
        return SAGE(args.num_layers, args.input_dim, args.hidden_dimension, args.num_classes, args.dropout, args.norm)
    
class GCN(torch.nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dimension, num_classes, dropout, norm=None) -> None:
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.num_layers = num_layers
        self.dropout = dropout
        if num_layers == 1:
            self.convs.append(GCNConv(input_dim, num_classes, cached=False,
                             normalize=True))
        else:
            self.convs.append(GCNConv(input_dim, hidden_dimension, cached=False,
                             normalize=True))
            if norm:
                self.norms.append(torch.nn.BatchNorm1d(hidden_dimension))
            else:
                self.norms.append(torch.nn.Identity())

            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_dimension, hidden_dimension, cached=False,
                             normalize=True))
                if norm:
                    self.norms.append(torch.nn.BatchNorm1d(hidden_dimension))
                else:
                    self.norms.append(torch.nn.Identity())

            self.convs.append(GCNConv(hidden_dimension, num_classes, cached=False, normalize=True))

    def forward(self, data):
        x, edge_index, edge_weight= data.x, data.edge_index, data.edge_weight
        for i in range(self.num_layers-1):
            x = F.dropout(x, p=self.dropout, training=self.training)
            if edge_weight != None:
                x = self.convs[i](x, edge_index, edge_weight)
            else:
                x = self.convs[i](x, edge_index)
            if i != self.num_layers - 1:
                x = self.norms[i](x)
                x = F.relu(x)
        embedding = x.clone()
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 마지막 conv 레이어는 리스트의 맨 뒤([-1])에 있습니다.
        if edge_weight is not None:
            x = self.convs[-1](x, edge_index, edge_weight)
        else:
            x = self.convs[-1](x, edge_index)
            
        # 4. 두 개를 모두 반환
        return x, embedding
    

class SAGE(torch.nn.Module):
    def __init__(
        self,
        num_layers,
        input_dim,
        hidden_dimension,
        num_classes,
        dropout,
        norm=None
    ) -> None:
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.num_layers = num_layers
        self.dropout = dropout

        if num_layers == 1:
            self.convs.append(
                SAGEConv(input_dim, num_classes, normalize=True)
            )
        else:
            self.convs.append(
                SAGEConv(input_dim, hidden_dimension, normalize=True)
            )

            if norm:
                self.norms.append(torch.nn.BatchNorm1d(hidden_dimension))
            else:
                self.norms.append(torch.nn.Identity())

            for _ in range(num_layers - 2):
                self.convs.append(
                    SAGEConv(hidden_dimension, hidden_dimension, normalize=True)
                )
                if norm:
                    self.norms.append(torch.nn.BatchNorm1d(hidden_dimension))
                else:
                    self.norms.append(torch.nn.Identity())

            self.convs.append(
                SAGEConv(hidden_dimension, num_classes, normalize=True)
            )

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight

        if self.num_layers == 1:
            x = F.dropout(x, p=self.dropout, training=self.training)
            if edge_weight is not None:
                x = self.convs[0](x, edge_index, edge_weight)
            else:
                x = self.convs[0](x, edge_index)
            embedding = x.clone()
            return x, embedding

        for i in range(self.num_layers - 1):
            x = F.dropout(x, p=self.dropout, training=self.training)
            if edge_weight is not None:
                x = self.convs[i](x, edge_index, edge_weight)
            else:
                x = self.convs[i](x, edge_index)

            x = self.norms[i](x)
            x = F.relu(x)

        embedding = x.clone()

        x = F.dropout(x, p=self.dropout, training=self.training)
        if edge_weight is not None:
            x = self.convs[-1](x, edge_index, edge_weight)
        else:
            x = self.convs[-1](x, edge_index)

        return x, embedding

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout):
        super().__init__()

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, input_tensor=False):

        # numpy → tensor 변환
        if not input_tensor:
            x = torch.tensor(x, dtype=torch.float32)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)

        x = F.normalize(x, dim=1)

        return x