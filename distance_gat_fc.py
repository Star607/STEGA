from graph_model import GATLayerImp3
import torch
import torch.nn as nn
import os
os.environ['DGLBACKEND'] = 'pytorch'
from dgl.nn import GraphormerLayer
from graph_model import GAT, GCN, SAGE

class DistanceGatFC(nn.Module):
    def __init__(self, config, data_feature, device):
        super(DistanceGatFC, self).__init__()
        self.embed_dim = config['embed_dim']
        self.num_of_layers = config['num_of_layers']
        self.num_of_heads = config['num_of_heads']
        self.concat = config['concat']
        self.device = device
        self.gnn_model = config['gnn_model']
        self.gps_emb_dim = config['gps_emb_dim']
        self.distance_mode = config['distance_mode']
        self.no_gps_emb = config.get('no_gps_emb', False)
        self.adj_mx = data_feature.get('adj_mx')
        self.Apt = torch.LongTensor([self.adj_mx.row.tolist(), self.adj_mx.col.tolist()]).to(self.device)
        self.node_features = data_feature['node_features']
        if not self.no_gps_emb:
            self.feature_dim = self.node_features.shape[1] - 2 + self.gps_emb_dim * 2
        else:
            self.feature_dim = self.node_features.shape[1]
        if self.gnn_model == 'gat':
            self.gat_encoder = GATLayerImp3(num_in_features=self.feature_dim, num_out_features=self.embed_dim, num_of_heads=self.num_of_heads, concat=self.concat, device=self.device)
        elif self.gnn_model == 'graphormer':
            self.gat_encoder = GraphormerLayer(feat_size=self.feature_dim, hidden_size=self.embed_dim, num_heads=self.num_of_heads)
        elif self.gnn_model == 'gat_v1':
            self.gat_encoder = GAT(self.feature_dim, self.embed_dim, self.embed_dim, self.num_of_layers, self.num_of_heads)

        if not self.no_gps_emb:
            self.lat_embed = nn.Embedding(num_embeddings=data_feature['img_height'], embedding_dim=self.gps_emb_dim)
            self.lon_embed = nn.Embedding(num_embeddings=data_feature['img_width'], embedding_dim=self.gps_emb_dim)

    def compute_gnn(self):
        if self.Apt.device != self.device:
            self.Apt.to(self.device)
        if not self.no_gps_emb:
            # embedding for lon_grid and lat_grid
            lon_feature = self.node_features[:, -2].long()
            lat_feature = self.node_features[:, -1].long()
            lon_feature_emb = self.lon_embed(lon_feature)
            lat_feature_emb = self.lat_embed(lat_feature)
            in_feature = torch.cat([lon_feature_emb, lat_feature_emb, self.node_features[:, :-2]], dim=1)
        else:
            in_feature = self.node_features
        if self.gnn_model == 'gat':
            self.gat_encoder.to(self.device)
            self.gat_encoder.device = self.device
            encode_feature = self.gat_encoder([in_feature, self.Apt])[0]
        elif self.gnn_model == 'graphormer':
            encode_feature = self.gat_encoder(in_feature.unsqueeze(0))[0]
        return encode_feature  # [batch, embed_dim]
