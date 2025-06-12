import warnings
warnings.filterwarnings('ignore')

# Standard library
import random
import argparse
import pathlib
import sys
import os
from abc import abstractmethod
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

# PyTorch
import torch
from torch import nn, FloatTensor
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning import seed_everything

# Deep Graph Library (DGL)
import dgl
from dgl.data.utils import load_graphs
from dgl.nn.pytorch.conv import NNConv
from dgl.nn.pytorch.glob import MaxPooling

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Progress bar
from tqdm import tqdm

# Plotly for visualization
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# OCC-based utilities
from occwl.graph import face_adjacency
from occwl.uvgrid import ugrid, uvgrid
from occwl.io import load_step


def linear_stiffness_extaraction_each(df, main_axes_inverse):
    slope_list = []
    slope_list2 = []
    slope_gap_list = []

    ## change (N/mm, N-mm/deg) to (kgf/mm, kgf-cm/deg)
    df_x = np.linspace(0, main_axes_inverse, len(df))
    for j in range(int(len(df))):
        if j == 0 or j == len(df) -1:
            slope_list.append(0)
            slope_list2.append(0)
            slope_gap_list.append(0)
        else :
            slope = df[j] / df_x[j] # axis 값과 Force_or_Moment 값의 비율 계산 (데이터 위치에 따라 달라질 수 있음)
            slope2 = (df[j+1]- df[j-1]) / (df_x[j+1]- df_x[j-1])
            slope_gap = slope2/slope
            slope_list.append(slope)
            slope_list2.append(slope2)
            slope_gap_list.append(slope_gap)

    # slope_gap_list에서 1에 가장 가까운 값의 index를 찾아 해당 index의 변환된 slope_list 값을 반환
    if slope_gap_list:
        min_gap_index = min(range(len(slope_gap_list)), key=lambda i: abs(slope_gap_list[i] - 1))
        linear_stiffness = slope_list[min_gap_index]
    else:
        linear_stiffness = None

    return linear_stiffness


## uvnet.util ###############################################################################
def bounding_box_uvgrid(inp: torch.Tensor):
    pts = inp[..., :3].reshape((-1, 3))
    return bounding_box_pointcloud(pts)

def bounding_box_pointcloud(pts: torch.Tensor):
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    box = [[x.min(), y.min(), z.min()], [x.max(), y.max(), z.max()]]
    return torch.tensor(box)


def center_and_scale_uvgrid(inp: torch.Tensor, return_center_scale=False):
    bbox = bounding_box_uvgrid(inp)
    diag = bbox[1] - bbox[0]
    scale = 2.0 / max(diag[0], diag[1], diag[2])
    center = 0.5 * (bbox[0] + bbox[1])
    inp[..., :3] -= center
    inp[..., :3] *= scale
    if return_center_scale:
        return inp, center, scale
    return inp


def get_random_rotation():
    """Get a random rotation in 90 degree increments along the canonical axes"""
    axes = [
        np.array([1, 0, 0]),
        np.array([0, 1, 0]),
        np.array([0, 0, 1]),
    ]
    # angles = [0.0, 90.0, 180.0, 270.0]  # default
    angles = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 240.0, 270.0, 300.0, 330.0]  # new
    axis = random.choice(axes)
    angle_radians = np.radians(random.choice(angles))
    return Rotation.from_rotvec(angle_radians * axis)

def rotate_bbox(bbox, rotation):
    """Rotate a bounding box by transforming its 8 corner points"""
    Rmat = torch.tensor(rotation.as_matrix()).float()

    # 1. bbox에서 min/max 분리
    bbox_min = bbox[..., :3]  # (x_min, y_min, z_min)
    bbox_max = bbox[..., 3:6]  # (x_max, y_max, z_max)

    # 2. 8개 코너 포인트 생성
    corners = torch.stack([
        bbox_min,
        torch.stack([bbox_max[..., 0], bbox_min[..., 1], bbox_min[..., 2]], dim=-1),
        torch.stack([bbox_min[..., 0], bbox_max[..., 1], bbox_min[..., 2]], dim=-1),
        torch.stack([bbox_min[..., 0], bbox_min[..., 1], bbox_max[..., 2]], dim=-1),
        torch.stack([bbox_max[..., 0], bbox_max[..., 1], bbox_min[..., 2]], dim=-1),
        torch.stack([bbox_max[..., 0], bbox_min[..., 1], bbox_max[..., 2]], dim=-1),
        torch.stack([bbox_min[..., 0], bbox_max[..., 1], bbox_max[..., 2]], dim=-1),
        bbox_max,
    ], dim=1)  # Shape: (N, 8, 3)

    # 3. 회전 적용
    rotated_corners = torch.matmul(corners, Rmat.T)

    # 4. 회전된 bbox의 min/max 재계산
    bbox_min_rot = rotated_corners.min(dim=1)[0]
    bbox_max_rot = rotated_corners.max(dim=1)[0]

    # 5. 다시 bbox 형태로 반환
    return torch.cat([bbox_min_rot, bbox_max_rot], dim=-1)

def rotate_uvgrid(inp, rotation):
    """Rotate the node features in the graph by a given rotation"""
    Rmat = torch.tensor(rotation.as_matrix()).float()
    orig_size = inp[..., :3].size()
    inp[..., :3] = torch.mm(inp[..., :3].view(-1, 3), Rmat).view(
        orig_size
    )  # Points
    inp[..., 3:6] = torch.mm(inp[..., 3:6].view(-1, 3), Rmat).view(
        orig_size
    )  # Normals/tangents
    return inp


#%%
## uvnet.base ###############################################################################
class BaseDataset(Dataset):
    @staticmethod
    @abstractmethod
    def num_classes():
        pass

    def load_graphs(self, file_paths, center_and_scale=True):
        self.data = []
        for fn in tqdm(file_paths, desc="Loading graphs", disable=True):
            if not fn.exists():
                continue
            sample = self.load_one_graph(fn)
            if sample is None:
                continue
            if sample["graph"].edata["x"].size(0) == 0:
                # Catch the case of graphs with no edges
                continue
            self.data.append(sample)
        if center_and_scale:
            self.center_and_scale()
        self.convert_to_float32()

    def center_and_scale_graph(self, graph):
        graph.ndata["x"], center, scale = center_and_scale_uvgrid(
            graph.ndata["x"], return_center_scale=True
        )
        graph.edata["x"][..., :3] -= center
        graph.edata["x"][..., :3] *= scale
        return graph

    def jitter_graph(self, graph):
        noise = random.uniform(-0.2, 0.2)
        graph.edata["x"][..., :3] += noise
        graph.edata["x"][..., :3] += noise
        return graph

    def center_and_scale(self):
        for i in range(len(self.data)):
            self.data[i]["graph"] = self.center_and_scale_graph(self.data[i]["graph"])

    def convert_to_float32(self):
        for i in range(len(self.data)):
            self.data[i]["graph"].ndata["x"] = self.data[i]["graph"].ndata["x"].type(FloatTensor)
            self.data[i]["graph"].edata["x"] = self.data[i]["graph"].edata["x"].type(FloatTensor)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.random_rotate:
            rotation = get_random_rotation()
            sample["graph"].ndata["x"] = rotate_uvgrid(sample["graph"].ndata["x"], rotation)
            sample["graph"].edata["x"] = rotate_uvgrid(sample["graph"].edata["x"], rotation)

            # added part for bounding box
            sample["graph"].ndata["bbox"] = rotate_bbox(sample["graph"].ndata["bbox"], rotation)
            sample["graph"].edata["bbox"] = rotate_bbox(sample["graph"].edata["bbox"], rotation)
        return sample

    def _collate(self, batch):
        batched_graph = dgl.batch([sample["graph"] for sample in batch])
        batched_filenames = [sample["filename"] for sample in batch]
        batched_inputs = [sample["input"] for sample in batch]

        return {"graph": batched_graph, "filename": batched_filenames, "input": batched_inputs}

    def get_dataloader(self, batch_size=128, shuffle=True, num_workers=0):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate,
            num_workers=num_workers,  # Can be set to non-zero on Linux
            drop_last=True,
        )

#%%
## uvnet.encoders ###############################################################################
def _conv1d(in_channels, out_channels, kernel_size=3, padding=0, bias=False):
    return nn.Sequential(
        nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias
        ),
        nn.BatchNorm1d(out_channels),
        nn.ELU(),
    )


def _conv2d(in_channels, out_channels, kernel_size, padding=0, bias=False):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ELU(),
    )


def _fc(in_features, out_features, bias=False):
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=bias),
        nn.BatchNorm1d(out_features),
        nn.ELU(),
    )


class _MLP(nn.Module):

    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):

        super(_MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("Number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            # TODO: this could move inside the above loop
            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = F.elu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)


class UVNetCurveEncoder(nn.Module):
    def __init__(self, in_channels=3, output_dims=64):

        super(UVNetCurveEncoder, self).__init__()
        self.in_channels = in_channels
        self.conv1 = _conv1d(in_channels, 64, kernel_size=3, padding=1, bias=False)
        self.conv2 = _conv1d(64, 128, kernel_size=3, padding=1, bias=False)
        self.conv3 = _conv1d(128, 256, kernel_size=3, padding=1, bias=False)
        self.final_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = _fc(256, output_dims, bias=False)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        assert x.size(1) == self.in_channels
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.final_pool(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x


class UVNetSurfaceEncoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        output_dims=64,
    ):
        super(UVNetSurfaceEncoder, self).__init__()
        self.in_channels = in_channels
        self.conv1 = _conv2d(in_channels, 64, 3, padding=1, bias=False)
        self.conv2 = _conv2d(64, 128, 3, padding=1, bias=False)
        self.conv3 = _conv2d(128, 256, 3, padding=1, bias=False)
        self.final_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = _fc(256, output_dims, bias=False)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        assert x.size(1) == self.in_channels
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.final_pool(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x



class _EdgeConv(nn.Module):
    def __init__(
        self,
        edge_feats,
        out_feats,
        node_feats,
        num_mlp_layers=2,
        hidden_mlp_dim=64,
    ):
        super(_EdgeConv, self).__init__()
        self.proj = _MLP(1, node_feats, hidden_mlp_dim, edge_feats)
        self.mlp = _MLP(num_mlp_layers, edge_feats, hidden_mlp_dim, out_feats)
        self.batchnorm = nn.BatchNorm1d(out_feats)
        self.eps = torch.nn.Parameter(torch.FloatTensor([0.0]))

    def forward(self, graph, nfeat, efeat):
        src, dst = graph.edges()
        proj1, proj2 = self.proj(nfeat[src]), self.proj(nfeat[dst])
        agg = proj1 + proj2
        h = self.mlp((1 + self.eps) * efeat + agg)
        h = F.elu(self.batchnorm(h))
        return h


class _NodeConv(nn.Module):
    def __init__(
        self,
        node_feats,
        out_feats,
        edge_feats,
        num_mlp_layers=2,
        hidden_mlp_dim=64,
    ):
        super(_NodeConv, self).__init__()
        self.gconv = NNConv(
            in_feats=node_feats,
            out_feats=out_feats,
            edge_func=nn.Linear(edge_feats, node_feats * out_feats),
            aggregator_type="sum",
            bias=False,
        )
        self.batchnorm = nn.BatchNorm1d(out_feats)
        self.mlp = _MLP(num_mlp_layers, node_feats, hidden_mlp_dim, out_feats)
        self.eps = torch.nn.Parameter(torch.FloatTensor([0.0]))

    def forward(self, graph, nfeat, efeat):
        h = (1 + self.eps) * nfeat
        h = self.gconv(graph, h, efeat)
        h = self.mlp(h)
        h = F.leaky_relu(self.batchnorm(h))
        return h


class UVNetGraphEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        input_edge_dim,
        output_dim,
        hidden_dim=64,
        learn_eps=True,
        num_layers=3,
        num_mlp_layers=2,
    ):
        """
        This is the graph neural network used for message-passing features in the
        face-adjacency graph.  (see Section 3.2, Message passing in paper)

        Args:
            input_dim ([type]): [description]
            input_edge_dim ([type]): [description]
            output_dim ([type]): [description]
            hidden_dim (int, optional): [description]. Defaults to 64.
            learn_eps (bool, optional): [description]. Defaults to True.
            num_layers (int, optional): [description]. Defaults to 3.
            num_mlp_layers (int, optional): [description]. Defaults to 2.
        """
        super(UVNetGraphEncoder, self).__init__()
        self.num_layers = num_layers
        self.learn_eps = learn_eps

        # List of layers for node and edge feature message passing
        self.node_conv_layers = torch.nn.ModuleList()
        self.edge_conv_layers = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            node_feats = input_dim if layer == 0 else hidden_dim
            edge_feats = input_edge_dim if layer == 0 else hidden_dim
            self.node_conv_layers.append(
                _NodeConv(
                    node_feats=node_feats,
                    out_feats=hidden_dim,
                    edge_feats=edge_feats,
                    num_mlp_layers=num_mlp_layers,
                    hidden_mlp_dim=hidden_dim,
                ),
            )
            self.edge_conv_layers.append(
                _EdgeConv(
                    edge_feats=edge_feats,
                    out_feats=hidden_dim,
                    node_feats=node_feats,
                    num_mlp_layers=num_mlp_layers,
                    hidden_mlp_dim=hidden_dim,
                )
            )

        # Linear function for graph poolings of output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()

        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(nn.Linear(input_dim, output_dim))
            else:
                self.linears_prediction.append(nn.Linear(hidden_dim, output_dim))

        self.drop1 = nn.Dropout(0.3)
        self.drop = nn.Dropout(0.5)
        self.pool = MaxPooling()

    def forward(self, g, h, efeat):
        hidden_rep = [h]
        he = efeat

        for i in range(self.num_layers - 1):
            # Update node features
            h = self.node_conv_layers[i](g, h, he)
            # Update edge features
            he = self.edge_conv_layers[i](g, h, he)
            hidden_rep.append(h)

        out = hidden_rep[-1]
        out = self.drop1(out)
        score_over_layer = 0

        # Perform pooling over all nodes in each graph in every layer
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(g, h)
            score_over_layer += self.drop(self.linears_prediction[i](pooled_h))

        return out, score_over_layer

#%%
## uvnet.step2graph ###############################################################################

def get_bbox(points):
    min_point = np.min(points, axis=0)
    max_point = np.max(points, axis=0)
    return np.concatenate([min_point, max_point])  # [xmin, ymin, zmin, xmax, ymax, zmax]

def l2_norm_bbox(bbox_arr):
    dims = bbox_arr[:, 3:] - bbox_arr[:, :3]
    return np.linalg.norm(dims, axis=1)

def build_graph(solid, curv_num_u_samples, surf_num_u_samples, surf_num_v_samples, bbox_l2_threshold=10.0):
    graph = face_adjacency(solid)

    graph_face_feat = []
    graph_face_bbox = []
    original_face_indices = []

    for face_idx in graph.nodes:
        # Get the B-rep face
        face = graph.nodes[face_idx]["face"]
        # Compute UV-grids
        points = uvgrid(
            face, method="point", num_u=surf_num_u_samples, num_v=surf_num_v_samples
        )
        face_feat = points
        graph_face_feat.append(face_feat)
        bbox = get_bbox(points.reshape(-1, 3))  # [xmin, ymin, zmin, xmax, ymax, zmax]
        graph_face_bbox.append(bbox)
        original_face_indices.append(face_idx)

    graph_face_feat = np.asarray(graph_face_feat)
    graph_face_bbox = np.asarray(graph_face_bbox)

    ## added part
    face_l2 = l2_norm_bbox(graph_face_bbox)
    keep_face_mask = face_l2 > bbox_l2_threshold
    keep_face_idx = np.nonzero(keep_face_mask)[0]
    kept_original_face_idx = [original_face_indices[i] for i in keep_face_idx]
    old_to_new = {old: new for new, old in enumerate(kept_original_face_idx)}
    
    graph_face_feat = graph_face_feat[keep_face_mask]
    graph_face_bbox = graph_face_bbox[keep_face_mask]
    
    # ➤ Edge filtering & connection rebuild
    edges = list(graph.edges)
    src_all = [e[0] for e in edges]
    dst_all = [e[1] for e in edges]

    graph_edge_feat = []
    graph_edge_bbox = []
    src_filtered = []
    dst_filtered = []

    for (src, dst) in zip(src_all, dst_all):
        edge = graph.edges[(src, dst)]["edge"]
        if not edge.has_curve():
            continue
        points = ugrid(edge, method="point", num_u=curv_num_u_samples)
        bbox = get_bbox(points.reshape(-1, 3))
        l2 = l2_norm_bbox(np.expand_dims(bbox, 0))[0]

        if src in old_to_new and dst in old_to_new: #and l2 > bbox_l2_threshold:
            src_filtered.append(old_to_new[src])
            dst_filtered.append(old_to_new[dst])
            graph_edge_feat.append(points)
            graph_edge_bbox.append(bbox)

    graph_edge_feat = np.array(graph_edge_feat)
    graph_edge_bbox = np.array(graph_edge_bbox)

    # ➤ DGL graph 생성
    dgl_graph = dgl.graph((src_filtered, dst_filtered), num_nodes=len(graph_face_feat))
    dgl_graph.ndata["x"] = torch.from_numpy(graph_face_feat)
    dgl_graph.ndata["bbox"] = torch.from_numpy(graph_face_bbox).float()
    dgl_graph.edata["x"] = torch.from_numpy(graph_edge_feat)
    dgl_graph.edata["bbox"] = torch.from_numpy(graph_edge_bbox).float()

    return dgl_graph

#%%
## bushing_FD_rev ###############################################################################
class MyStandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        # X는 (n_samples, n_features) 형태의 numpy 배열, float64로 계산
        self.mean_ = np.mean(X, axis=0, dtype=np.float32)
        self.std_ = np.std(X, axis=0, dtype=np.float32)
        # std가 0인 경우 1로 처리하여 0으로 나누는 문제 방지
        # self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def inverse_transform(self, X):
        return X * self.std_ + self.mean_


def load_bush_data(total_output_data):
    """
    부시별 input/output 딕셔너리를 
    (bush_names, inputs, outputs) 형태로 반환
    """
    bush_names = list(total_output_data.keys())  # 예: ["06_04_NX4", "06_05_NX4", ...]
    
    # 부시별로 input/output을 하나의 리스트에 쌓기
    all_inputs = []
    all_outputs = []
    for name in bush_names:
        sub_dict = total_output_data[name]
        bush_input = sub_dict["input"]   # shape 예: (feature_dim,)
        bush_output = sub_dict["output"] # shape 예: (6,16,16) or (16,16) 등
        all_inputs.append(bush_input)
        all_outputs.append(bush_output)

    all_inputs = np.array(all_inputs, dtype=np.float32)
    all_outputs = np.array(all_outputs, dtype=np.float32)

    return bush_names, all_inputs, all_outputs


class Bushings(BaseDataset):
    @staticmethod
    def num_classes():
        return (6, 16, 16)
        
    def __init__(
        self,

        center_and_scale=True,
        output_scalers=None,
        input_scalers=None,
        bbox_scaler = None,

        curv_u_samples=10,
        surf_u_samples=10,
        surf_v_samples=10,
        test_step_file: str = None,
        test_csv_file = None,
        ):

        self._test_step_file = test_step_file
        self._test_csv_file  = test_csv_file.input_data


        self.curv_u_samples = curv_u_samples
        self.surf_u_samples = surf_u_samples
        self.surf_v_samples = surf_v_samples

        self.random_rotate = False


        file_paths = [pathlib.Path(test_step_file)]
        self.load_graphs(file_paths, center_and_scale)


        self.output_scaler = output_scalers
        self.input_scaler = input_scalers
        self.bbox_scaler = bbox_scaler
    

    def _transform_input_scaler(self, dataset, scaler):
        for s in dataset:
            arr = s["input"].cpu().numpy().ravel().reshape(-1,1)
            scaled = scaler.transform(arr).reshape(s["input"].shape)
            s["input"] = torch.from_numpy(scaled).float()


    def _transform_bbox_scaler(self, dataset, scaler):
        for s in dataset:
            g = s["graph"]
            if "bbox" in g.ndata:
                arr = g.ndata["bbox"].cpu().numpy().reshape(-1,1)
                g.ndata["bbox"] = torch.from_numpy(scaler.transform(arr)
                                        .reshape(g.ndata["bbox"].shape)).float()
            if "bbox" in g.edata:
                arr = g.edata["bbox"].cpu().numpy().reshape(-1,1)
                g.edata["bbox"] = torch.from_numpy(scaler.transform(arr)
                                        .reshape(g.edata["bbox"].shape)).float()
                    
    def load_one_graph(self, file_path):
        solid = load_step(str(file_path))[0]
        graph = build_graph(solid,
                            self.curv_u_samples,
                            self.surf_u_samples,
                            self.surf_v_samples)
        sample = {"graph": graph, "filename": file_path.stem}

        linear_stiff = self._test_csv_file[0,8:].astype(np.float32)  
        log_linear_stiff = np.log1p(linear_stiff)
        sample["input"] = torch.tensor(log_linear_stiff, dtype=torch.float32)
        return sample


## models ###############################################################################
class MLP_Predictor(nn.Module):
    def __init__(self,
                 seg1_dim=128, seg2_dim=6,
                 embed_dim=128,   
                 hidden_dim=256,  
                 depth=4,         
                 out_H=16, out_W=16,
                 out_ch=6):
        super().__init__()

        self.embed1 = nn.Sequential(
            nn.Linear(seg2_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ELU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )
        self.embed2 = nn.Sequential(
            nn.Linear(embed_dim*2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(0.3)
        )
        
        dims = [hidden_dim] * depth  
        layers = []
        for d0, d1 in zip(dims, dims[1:]):
            layers += [
                nn.Linear(d0, d1),
                nn.BatchNorm1d(d1),
                nn.ELU(inplace=True),
            ]

        self.encoder = nn.Sequential(*layers)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, out_ch*out_H*out_W),)


    def forward(self, input, linear_stiff):

        seg1 = input
        seg2 = linear_stiff  

        e2 = self.embed1(seg2) 

        h = torch.cat([seg1, e2], dim=1)
        h = self.embed2(h)  
        h = self.encoder(h)     

        out =self.decoder(h).view([-1,6,16,16])
        return out
    
class UVNetPredictor(nn.Module):

    def __init__(
            self,
            crv_emb_dim=64,
            srf_emb_dim=64,
            graph_emb_dim=128, 
    ):
        
        super().__init__()
        self.curv_encoder = UVNetCurveEncoder(
            in_channels=3, output_dims=crv_emb_dim
        )
        self.surf_encoder = UVNetSurfaceEncoder(
            in_channels=3, output_dims=srf_emb_dim
        )
        self.graph_encoder = UVNetGraphEncoder(
            srf_emb_dim, crv_emb_dim, graph_emb_dim,
        )
        self.curv_embed = nn.Sequential(
            nn.Linear(6, srf_emb_dim),
            nn.BatchNorm1d(srf_emb_dim),
            nn.ELU(),
            nn.Linear(srf_emb_dim, srf_emb_dim),
        ) 
        self.surf_embed = nn.Sequential(
            nn.Linear(6, srf_emb_dim),
            nn.BatchNorm1d(srf_emb_dim),
            nn.ELU(),
            nn.Linear(srf_emb_dim, srf_emb_dim),
        ) 
        self.clf = MLP_Predictor()

    def forward(self, batched_graph, linear_stiff):
        
        input_crv_feat = batched_graph.edata["x"]
        input_srf_feat = batched_graph.ndata["x"]

        input_crv_bbox = batched_graph.edata["bbox"]
        input_srf_bbox = batched_graph.ndata["bbox"]

        hidden_crv_feat = self.curv_encoder(input_crv_feat)
        hidden_srf_feat = self.surf_encoder(input_srf_feat)
        
        bounding_crv_feat = self.curv_embed(input_crv_bbox)
        bounding_srf_feat = self.surf_embed(input_srf_bbox)  

        hidden_crv_feat = hidden_crv_feat + bounding_crv_feat
        hidden_srf_feat = hidden_srf_feat + bounding_srf_feat

        _, graph_emb = self.graph_encoder(batched_graph, hidden_srf_feat, hidden_crv_feat)
        
        out = self.clf(graph_emb, linear_stiff)
        
        return out


class FDPrediction(pl.LightningModule):

    def __init__(self, output_dim, output_scalers=None):

        super().__init__()
        self.save_hyperparameters()
        self.model = UVNetPredictor()
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()
        self.output_scalers = output_scalers  

    def forward(self, batched_graph, linear_stiff):
        logits = self.model(batched_graph, linear_stiff)
        return logits

    def inverse_scale_data(self, data, scaler):
        data_temp = data.detach().cpu().numpy().flatten()
        reshaped = data_temp.reshape(-1, 1)
        inversed = scaler.inverse_transform(reshaped)
        return torch.tensor(inversed.reshape(data.shape), dtype=torch.float32)
        
    

class InferenceVEPDataset():
    def __init__(self, csv_path: str, extrapolation_values=None):
        
        df = pd.read_csv(csv_path)

        # 고정된 열 순서 (전체 8개 shape feature 순서 고정)
        shape_cols_required = ['Inner_radius', 'Thickness', 'Outer_height', 'Gap',
                               'Control_point_x', 'Control_point_y',
                               'Inner_wavy_depth', 'Inner_wavy_height']

        # extrapolation일 경우 필수 열 5개만 확인
        required_if_extrap = ['Inner_radius', 'Thickness', 'Outer_height', 'Gap', 'Inner_wavy_depth']
        if extrapolation_values is None:
            missing_cols = [col for col in required_if_extrap if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Extrapolation requires the following shape parameters: {missing_cols}")
            if df[required_if_extrap].isnull().values.any():
                raise ValueError("Inputs are missing")

        # 전체 shape 열 순서를 유지하면서 없는 열은 0으로 채우기
        shape_data = []
        for col in shape_cols_required:
            if col in df.columns:
                shape_data.append(df[col].fillna(0).values.astype(np.float32))
            else:
                shape_data.append(np.zeros(len(df), dtype=np.float32))
        input_data_shape = np.column_stack(shape_data)


        # Extract linear stiffness input
        # First 3 columns: kgf/mm → N/mm (×9.806650)
        linear_1_3 = df[['linear_stiffness_1', 'linear_stiffness_2', 'linear_stiffness_3']].values.astype(np.float32) * 9.806650
        # Last 3 columns: kgf·cm/deg → N·mm/rad (×5622.786320)
        linear_4_6 = df[['linear_stiffness_4', 'linear_stiffness_5', 'linear_stiffness_6']].values.astype(np.float32) * 5622.786320
        input_data_linear = np.hstack((linear_1_3, linear_4_6))


        # Combine into a single numpy array
        input_data = np.hstack((input_data_shape, input_data_linear))
        inference_name = os.path.splitext(os.path.basename(csv_path))[0]
        
        self.input_data = input_data
        self.inference_name = inference_name


##### POLY REGRESSION #####
def predict_on_grid(model, poly, grid_points):
    """
    Predict using the polynomial regression model on the grid points.
    Args:
        model: Trained LinearRegression model.
        poly: PolynomialFeatures instance.
        grid_points: Grid points (2D numpy array).
    Returns:
        Predicted values on the grid.
    """
    grid_points_poly = poly.transform(grid_points) 
    return model.predict(grid_points_poly)

def polynomial_regression(X, Z, degree, prev_model=None, prev_poly=None):

    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)  

    model = LinearRegression(positive=True, fit_intercept=False)

    eps = 1.0
    
    y_pos = X[:, 1]
    weights = np.exp(-eps * y_pos / (np.max(y_pos) + 1e-6))

    try:
        model.fit(X_poly, Z, sample_weight=weights / (Z + eps))

    except RuntimeError as e:
        if prev_model is not None and prev_poly is not None:
            return prev_model, prev_poly
        else:
            raise RuntimeError("No previous model available to fallback.")
    return model, poly

def loocv_optimization(X, Z, max_degree=6):

    from sklearn.exceptions import ConvergenceWarning
    import warnings
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    loo = LeaveOneOut()
    errors = []
    valid_degrees = []

    for degree in range(2, max_degree + 1):
        try:
            mse_list = []
            for train_index, test_index in loo.split(X):
                X_train, X_test = X[train_index], X[test_index]
                Z_train, Z_test = Z[train_index], Z[test_index]

                # Train the model
                model, poly = polynomial_regression(X_train, Z_train, degree)
                # Predict on the test set
                Z_pred = predict_on_grid(model, poly, X_test)
                # Compute the mean squared error
                mse_list.append(mean_squared_error(Z_test, Z_pred))

            # Average MSE for this degree
            errors.append(np.mean(mse_list))
            valid_degrees.append(degree)

        except Exception as e:
            print(f"Error occurred for degree {degree}: {str(e)}")
            continue

    if not valid_degrees:
        raise ValueError("All degrees failed during LOOCV.")

    # Find the degree with the lowest error
    optimal_degree = valid_degrees[np.argmin(errors)]
    return optimal_degree
###################################################################


######## Result Extraction #######
def get_extrapolation_range(stiffness_value_to_train, df):

    if isinstance(stiffness_value_to_train, str) and stiffness_value_to_train.startswith("Stiffness_"):
        stiffness_value_to_train = stiffness_value_to_train.split("_")[-1]
    else:
        stiffness_value_to_train = str(stiffness_value_to_train)  # 문자열로 변환

    scale_factor = 1.0487
    D_O_RUBBER = 2 * (df[0] + df[1])
    D_I_RUBBER = 2 * df[0]
    L_O_RUBBER = 2 * df[2]
    L_I_RUBBER = 2 * (df[2] + df[3])

    x_disp = np.where(df[6]<= 0,
                  (D_O_RUBBER - D_I_RUBBER) / 2,
                  (D_O_RUBBER - D_I_RUBBER) / 2 - df[6])
    z_disp = (L_I_RUBBER*scale_factor - L_O_RUBBER) / 2
    theta_x = (np.arctan(D_O_RUBBER / L_O_RUBBER) - np.arcsin(D_I_RUBBER / np.sqrt(D_O_RUBBER**2 + L_O_RUBBER**2)))

    if stiffness_value_to_train in ["1", "2"]:
        subAxes = theta_x
        mainAxes= x_disp
    elif stiffness_value_to_train in ["3"]:
        subAxes = theta_x
        mainAxes= z_disp
    elif stiffness_value_to_train in ["4", "5"]:
        subAxes = z_disp
        mainAxes= theta_x
    elif stiffness_value_to_train in ["6"]:
        subAxes = z_disp
        # mainAxes= 1.5708 # 90 degree
        mainAxes = 0.2617993877991494 # 15 degree

    return subAxes, mainAxes


def inference_results_extraction(input_data_unscaled, prediction, stiffness_num, save_path:str, linear_scaling=False, extrapolation_values=None):
    
    prev_model, prev_poly = None, None
    
    if extrapolation_values is not None:
        if len(extrapolation_values) != 5:
            raise ValueError("Extrapolation values must be a list of 5 numbers: [d1, d2, d3, a1, a2]")
        d1, d2, d3, a1, a2 = map(float, extrapolation_values)

        a1 = math.radians(a1)
        a2 = math.radians(a2)
    
        if stiffness_num in [1]:
            grid_x1, grid_y1 = np.meshgrid(np.linspace(0, a2, 25),
                                           np.linspace(0, d1, 25))
            grid_x, grid_y = np.meshgrid(np.linspace(0, a2*0.7, 16),
                                         np.linspace(0, d1*0.7, 16))
            linear_scaling_max = d1

        elif stiffness_num in [2]:
            grid_x1, grid_y1 = np.meshgrid(np.linspace(0, a2, 25),
                                           np.linspace(0, d2, 25))
            grid_x, grid_y = np.meshgrid(np.linspace(0, a2*0.7, 16),
                                         np.linspace(0, d2*0.7, 16))
            linear_scaling_max = d2

        elif stiffness_num in [3]:
            grid_x1, grid_y1 = np.meshgrid(np.linspace(0, a2, 25),
                                        np.linspace(0, d3, 25))    
            grid_x, grid_y = np.meshgrid(np.linspace(0, a2*0.7, 16),
                                         np.linspace(0, d3*0.7, 16))
            linear_scaling_max = d3

        elif stiffness_num in [4]:
            grid_x1, grid_y1 = np.meshgrid(np.linspace(0, d3, 25),
                                           np.linspace(0, a1, 25))    
            grid_x, grid_y = np.meshgrid(np.linspace(0, d3*0.7, 16),
                                         np.linspace(0, a1*0.7, 16))
            linear_scaling_max = a1
           
        elif stiffness_num in [5]:
            grid_x1, grid_y1 = np.meshgrid(np.linspace(0, d3, 25),
                                           np.linspace(0, a2, 25)) 
            grid_x, grid_y = np.meshgrid(np.linspace(0, d3*0.7, 16),
                                         np.linspace(0, a2*0.7, 16))
            linear_scaling_max = a2

        else:
            deg_100 = abs(round(math.radians(100), 6))
            grid_x1, grid_y1 = np.meshgrid(np.linspace(0, d3, 25),
                                        np.linspace(0, deg_100, 25))
            grid_x, grid_y = np.meshgrid(np.linspace(0, d3*0.7, 16),
                                         np.linspace(0, 0.2617993877991494*0.7, 16))
            linear_scaling_max = deg_100

        train_X = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        train_X1 = np.column_stack([grid_x1.ravel(), grid_y1.ravel()]) 
            
    else:
        subAxes, mainAxes = get_extrapolation_range(stiffness_num, input_data_unscaled[0,:8])
        grid_x, grid_y = np.meshgrid(np.linspace(0, subAxes*0.7, 16),
                                    np.linspace(0, mainAxes*0.7, 16))
    
        if stiffness_num in [1, 2, 3, 4, 5]:
            grid_x1, grid_y1 = np.meshgrid(np.linspace(0, subAxes, 25),
                                        np.linspace(0, mainAxes, 25))
            
            linear_scaling_max = mainAxes
        else:
            deg_100 = abs(round(math.radians(100), 6))
            grid_x1, grid_y1 = np.meshgrid(np.linspace(0, subAxes, 25),
                                        np.linspace(0, deg_100, 25))
            linear_scaling_max = deg_100

        train_X = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        train_X1 = np.column_stack([grid_x1.ravel(), grid_y1.ravel()])
    
    optimal_degree = loocv_optimization(train_X, prediction[:,:].flatten())

    iterations =50
    for iteration in range(iterations):

        try:
            poly_model, poly = polynomial_regression(train_X, prediction[:,:].flatten(), optimal_degree, prev_model, prev_poly)
            prev_model, prev_poly = poly_model, poly  # Update the previous model
        except RuntimeError:
            continue

        z_pred = predict_on_grid(poly_model, poly, train_X)
        y_zero_indices = np.where(train_X[:, 1] == 0)
        z_pred[y_zero_indices] = 0

        prediction[:,:] = z_pred.reshape(16,16)

    Z_pred = predict_on_grid(poly_model, poly, train_X1)
    Z_pred = Z_pred.reshape(25, 25)
    y_zero_indices_final = np.where(grid_y1 == 0)
    Z_pred[y_zero_indices_final] = 0
    
    if linear_scaling:
            
        linear_stiff = linear_stiffness_extaraction_each(Z_pred[:, 0], linear_scaling_max)
        linear_stiff_target = input_data_unscaled[0, 7+ stiffness_num]

        scaling_factor = linear_stiff_target/linear_stiff
        Z_pred = scaling_factor * Z_pred
    

    # grid_x1, grid_y1, Z_pred를 1차원 배열로 펼침 (총 625개 점)
    grid_x1_flat = grid_x1.ravel()
    grid_y1_flat = grid_y1.ravel()
    Z_pred_flat = Z_pred.ravel()
    
    grid_x1_flat_sym = -grid_x1_flat
    grid_y1_flat_sym = grid_y1_flat
    Z_pred_flat_sym = Z_pred_flat
    # Concatenate original and symmetric points
    grid_x2 = np.concatenate((grid_x1_flat, grid_x1_flat_sym))
    grid_y2 = np.concatenate((grid_y1_flat, grid_y1_flat_sym))
    Z_pred2 = np.concatenate((Z_pred_flat, Z_pred_flat_sym))

    # Remove duplicate (x, y) points, keeping the first occurrence
    coords = np.column_stack((grid_x2, grid_y2))
    _, unique_indices = np.unique(coords, axis=0, return_index=True)
    unique_indices_sorted = np.sort(unique_indices)  # Keep order of first appearance

    grid_x2 = grid_x2[unique_indices_sorted]
    grid_y2 = grid_y2[unique_indices_sorted]
    Z_pred2 = Z_pred2[unique_indices_sorted]
    
    grid_x2_sym = grid_x2
    grid_y2_sym = -grid_y2
    Z_pred2_sym = -Z_pred2
    
    grid_x3 = np.concatenate((grid_x2, grid_x2_sym))
    grid_y3 = np.concatenate((grid_y2, grid_y2_sym))
    Z_pred3 = np.concatenate((Z_pred2, Z_pred2_sym))
    
    block_height = Z_pred3.max() * 100.0

    if stiffness_num in [1, 2, 3]:
        y_col = 1  # MainAxis = Y

        points = np.column_stack((Z_pred3, grid_y3, grid_x3))
        rounded_points = np.round(points, decimals=8)
        _, unique_indices = np.unique(rounded_points, axis=0, return_index=True)
        points_unique = points[sorted(unique_indices)]  # 원래 값에서 추출

        sorted_indices = np.lexsort((points_unique[:, 1], points_unique[:, 2]))  # (X 기준, Y 기준)
        points_sorted = points_unique[sorted_indices]

        # Plot the symmetric surface
        fig_sym = plt.figure()
        ax_sym = fig_sym.add_subplot(111, projection='3d')
        ax_sym.scatter(points_sorted[:, 2],  # X
                    points_sorted[:, 1],  # Y
                    points_sorted[:, 0],  # Z
                    c=points_sorted[:, 0], cmap='viridis', alpha=0.7)
        ax_sym.set_xlabel('SubAxis')
        ax_sym.set_ylabel('MainAxis')
        ax_sym.set_zlabel('RF')
        img_path_sym = os.path.join(save_path, f'Stiffness_Surface.png')
        plt.savefig(img_path_sym, dpi=300)
        plt.close(fig_sym)
        # print(f"Saved: {img_path_sym}")

        ## set block height
        unique_y = np.unique(points_sorted[:, y_col])
        max_y, min_y = unique_y.max(), unique_y.min()

        for i in range(len(points_sorted)):
            y_val = points_sorted[i, y_col]
            if np.isclose(y_val, max_y, atol=1e-8):
                points_sorted[i, 0] = block_height  # Y 최댓값 → +블록
            elif np.isclose(y_val, min_y, atol=1e-8):
                points_sorted[i, 0] = -block_height  # Y 최솟값 → -블록

        # Save as TXT
        output_txt_path_sym = os.path.join(save_path, f'Stiffness_Surface.txt')
        with open(output_txt_path_sym, 'w') as f:
            for row in points_sorted:
                f.write(f"{row[0]},{row[1]},{row[2]}\n")
        # print(f"Saved TXT: {output_txt_path_sym}")

        output_txt_path_sym_1D = os.path.join(save_path, f'Stiffness_Surface_1D.txt')
        with open(output_txt_path_sym_1D, 'w') as f:
            for row in points_sorted:
                if np.isclose(row[2], 0.0, atol=1e-8):
                    f.write(f"{row[0]},{row[1]}\n")
        # print(f"Saved TXT: {output_txt_path_sym_1D}")


    elif stiffness_num in [4, 5, 6]:
        y_col = 2  # MainAxis = Y (위치 바뀜)

        points = np.column_stack((Z_pred3, grid_x3, grid_y3))
        rounded_points = np.round(points, decimals=8)
        _, unique_indices = np.unique(rounded_points, axis=0, return_index=True)
        points_unique = points[sorted(unique_indices)]  # 원래 값에서 추출

        sorted_indices = np.lexsort((points_unique[:, 1], points_unique[:, 2]))  # (X 기준, Y 기준)
        points_sorted = points_unique[sorted_indices]


        # Plot the symmetric surface
        fig_sym = plt.figure()
        ax_sym = fig_sym.add_subplot(111, projection='3d')
        ax_sym.scatter(points_sorted[:, 1],  # X
                    points_sorted[:, 2],  # Y
                    points_sorted[:, 0],  # Z
                    c=points_sorted[:, 0], cmap='viridis', alpha=0.7)
        ax_sym.set_xlabel('SubAxis')
        ax_sym.set_ylabel('MainAxis')
        ax_sym.set_zlabel('RM')
        img_path_sym = os.path.join(save_path, f'Stiffness_Surface.png')
        plt.savefig(img_path_sym, dpi=300)
        plt.close(fig_sym)
        # print(f"Saved: {img_path_sym}")


        ## set block height
        unique_y = np.unique(points_sorted[:, y_col])
        max_y, min_y = unique_y.max(), unique_y.min()

        for i in range(len(points_sorted)):
            y_val = points_sorted[i, y_col]
            if np.isclose(y_val, max_y, atol=1e-8):
                points_sorted[i, 0] = block_height  # Y 최댓값 → +블록
            elif np.isclose(y_val, min_y, atol=1e-8):
                points_sorted[i, 0] = -block_height  # Y 최솟값 → -블록

        # Save as TXT
        output_txt_path_sym = os.path.join(save_path, f'Stiffness_Surface.txt')
        with open(output_txt_path_sym, 'w') as f:
            for row in points_sorted:
                f.write(f"{row[0]},{row[1]},{row[2]}\n")
        # print(f"Saved TXT: {output_txt_path_sym}")
        
        output_txt_path_sym_1D = os.path.join(save_path, f'Stiffness_Surface_1D.txt')
        with open(output_txt_path_sym_1D, 'w') as f:
            for row in points_sorted:
                if np.isclose(row[1], 0.0, atol=1e-8):
                    f.write(f"{row[0]},{row[2]}\n")
        # print(f"Saved TXT: {output_txt_path_sym_1D}")

        # print(f"Saved TXT: {save_path}")



def model_test(dataset, all_preds, result_path: str = None, linear_scaling: bool = False, extrapolation_values=None):

    test_key = [dataset.inference_name]
    test_inputs_unscaled_original = dataset.input_data.copy()

    bush_folder_name = f"{test_key[0]}"
    bush_save_path = os.path.join(result_path, bush_folder_name)
    os.makedirs(bush_save_path, exist_ok=True)


    for idx in range(all_preds.shape[0]):       
        stiffness_num = idx + 1                  
        stiffness_folder  = os.path.join(bush_save_path, f"Stiffness_{stiffness_num}")
        os.makedirs(stiffness_folder, exist_ok=True)

        # 그 안에 Final_FD 폴더 생성
        save_path = os.path.join(stiffness_folder, "Final_FD")
        os.makedirs(save_path, exist_ok=True)

        pred_2d = all_preds[idx]
        
        inference_results_extraction(
            test_inputs_unscaled_original,
            pred_2d,
            stiffness_num=stiffness_num,
            save_path=save_path,
            linear_scaling=linear_scaling,
            extrapolation_values=extrapolation_values
        )


def main(csv_path: str, step_path: str, linear_scaling: bool = False, extrapolation_values=None):

    if getattr(sys, 'frozen', False):
        BASE_DIR = sys._MEIPASS
    else:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    csv_path = os.path.abspath(csv_path)
    step_path = os.path.abspath(step_path)

    checkpoint_path = os.path.join(BASE_DIR, "model_CAD", "trained_model.ckpt")
    input_scaler_path = os.path.join(BASE_DIR, "model_CAD", "input_scaler.pth")
    bounding_scaler_path = os.path.join(BASE_DIR, "model_CAD", "bbox_scaler.pth")
    output_scaler_path = os.path.join(BASE_DIR, "model_CAD", "output_scaler.pth")

    #################################################################################
    num_workers = 0
    result_path = f"./Output_AI/CAD_AI_Model"
    dataset = InferenceVEPDataset(csv_path=csv_path, extrapolation_values=extrapolation_values)


    selected_step_file = step_path


    fitted_input_scaler = torch.load(input_scaler_path)
    fitted_bounding_scaler = torch.load(bounding_scaler_path)
    fitted_output_scaler = torch.load(output_scaler_path)

    test_data = Bushings(test_step_file = selected_step_file,
                         test_csv_file =  dataset,
                         input_scalers=fitted_input_scaler,
                         bbox_scaler=fitted_bounding_scaler,
                         output_scalers=fitted_output_scaler,
                         center_and_scale=True)
    
    test_data._transform_input_scaler(test_data.data, fitted_input_scaler)
    test_data._transform_bbox_scaler(test_data.data, fitted_bounding_scaler)

    test_loader = test_data.get_dataloader(
        batch_size=1, shuffle=False, num_workers=num_workers
    )

    model = FDPrediction.load_from_checkpoint(checkpoint_path, output_scalers=fitted_output_scaler)

    model.eval()
    device = "cpu"

    model.to(device)

    all_preds = []
    all_filenames = []
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["graph"].to(device)
            inputs.ndata["x"] = inputs.ndata["x"].permute(0, 3, 1, 2)
            inputs.edata["x"] = inputs.edata["x"].permute(0, 2, 1)
            linear_stiff = torch.stack(batch["input"], dim=0).to(model.device)
            
            preds = model(inputs, linear_stiff)                         
            preds = model.inverse_scale_data(preds, fitted_output_scaler)  
            preds = torch.expm1(preds).cpu().numpy()                    
            
            all_preds.append(preds)                                     
            all_filenames.extend(batch["filename"])

    all_preds = np.concatenate(all_preds, axis=0) 
    all_preds = all_preds.squeeze(0)

    # step_name = pathlib.Path(selected_step_file).stem
    # csv_dir = pathlib.Path("./Output_AI/CAD_AI_Model") / step_name
    # csv_dir.mkdir(parents=True, exist_ok=True)


    model_test(
        dataset=dataset,
        all_preds=all_preds,
        result_path=result_path,
        linear_scaling=linear_scaling,
        extrapolation_values=extrapolation_values
    ) 
    print(f"[CAD Input AI Model] 2D Stiffness Prediction Complete")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, required=True, help='Path to CSV file')
    parser.add_argument('--step', type=str, required=True, help='Path to STEP file')

    # parser.add_argument('--csv_path', type=str,default='./Input_AI/06_05_NX4_Bush/06_05_NX4_Bush.csv', help='Path to CSV file')
    # parser.add_argument('--step', type=str, default = './Input_AI/06_05_NX4_Bush/06_05_NX4_Bush.step', help='Path to STEP file')
    parser.add_argument('--linear_scaling', action='store_true')
    parser.add_argument('--extrapolation_values', type=float, nargs=5,
                        help='Extrapolation values: d1 d2 d3 a1 a2')
    # parser.add_argument('--extrapolation_values', type=float, nargs=5, default=[8,8,10,30,30],
    #                     help='Extrapolation values: d1 d2 d3 a1 a2')
    
    args = parser.parse_args()
    main(args.csv_path, args.step, linear_scaling=args.linear_scaling, extrapolation_values=args.extrapolation_values)