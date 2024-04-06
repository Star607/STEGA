import copy
import json
import logging
import math
import os
import random
from datetime import datetime
from functools import partial
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from geopy import distance
from shapely.geometry import LineString
from sklearn.preprocessing import LabelEncoder
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

params_map = {
    "BJ_Taxi": {
        "na_value": {
            "lanes": "unknown",
            "bridge": "no",
            "access": "unknown",
            "maxspeed": 120,
            "tunnel": "no",
            "junction": "no",
            "width": 100,
        },
        "norm_dict": {"length": 2, "maxspeed": 6, "width": 9},
        "onehot_list": [
            "highway",
            "oneway",
            "lanes",
            "bridge",
            "access",
            "tunnel",
            "junction",
        ],
    },
    "Porto_Taxi": {
        "na_value": {
            "lanes": "unknown",
            "bridge": "no",
            "maxspeed": 120,
            "tunnel": "no",
        },
        "norm_dict": {"length": 2, "maxspeed": 6},
        "onehot_list": ["highway", "oneway", "lanes", "bridge", "tunnel"],
    },
    "Shanghai_Taxi": {
        "na_value": {
            "lanes": "unknown",
            "bridge": "no",
            "access": "unknown",
            "maxspeed": 120,
            "tunnel": "no",
            "junction": "no",
            "width": 100,
        },
        "norm_dict": {"length": 2, "maxspeed": 6, "width": 9},
        "onehot_list": [
            "highway",
            "oneway",
            "lanes",
            "bridge",
            "access",
            "tunnel",
            "junction",
        ],
    },
    "Chengdu_Taxi": {
        "na_value": {
            "lanes": "unknown",
            "bridge": "no",
            "access": "unknown",
            "maxspeed": 120,
            "tunnel": "no",
            "junction": "no",
            "width": 100,
        },
        "norm_dict": {"length": 2, "maxspeed": 6, "width": 9},
        "onehot_list": [
            "highway",
            "oneway",
            "lanes",
            "bridge",
            "access",
            "tunnel",
            "junction",
        ],
    },
}


def str2bool(s):
    if isinstance(s, bool):
        return s
    if s.lower() in ("yes", "true"):
        return True
    elif s.lower() in ("no", "false"):
        return False
    else:
        print("bool value expected.")


class MapManager(object):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        if self.dataset_name == "Xian":
            self.lon_0 = 108.8093988
            self.lon_1 = 109.0499449
            self.lat_0 = 34.17026046
            self.lat_1 = 34.29639324
            self.img_unit = 0.005  # grid size like 0.42 km * 0.55 km
            self.img_width = (
                math.ceil((self.lon_1 - self.lon_0) / self.img_unit) + 1
            )  # width of image
            self.img_height = (
                math.ceil((self.lat_1 - self.lat_0) / self.img_unit) + 1
            )  # height of image

        elif self.dataset_name == "BJ_Taxi":
            self.lon_0 = 116.25
            self.lat_0 = 39.79
            self.lon_range = 0.2507  # span of longitude
            self.lat_range = 0.21  # span of latitude
            self.img_unit = 0.005  # grid size like 0.42 km * 0.55 km 
            self.img_width = math.ceil(self.lon_range / self.img_unit) + 1  
            self.img_height = math.ceil(self.lat_range / self.img_unit) + 1 
            self.road_num = 37684
            self.block_size = 60
            self.min_len = 5

        elif self.dataset_name == "Porto_Taxi":
            self.lon_0 = -8.6887
            self.lat_0 = 41.1405
            self.lon_range = 0.133
            self.lat_range = 0.046
            self.img_unit = 0.005
            self.img_width = math.ceil(self.lon_range / self.img_unit) + 1  
            self.img_height = math.ceil(self.lat_range / self.img_unit) + 1 
            self.road_num = 10904
            self.block_size = 173  # 276
            self.min_len = 5

        elif self.dataset_name == "Shanghai_Taxi":
            self.lon_0 = 120.8579
            self.lat_0 = 30.6988
            self.lon_range = 1.062
            self.lat_range = 1.150
            self.img_unit = 0.005
            self.img_width = math.ceil(self.lon_range / self.img_unit) + 1  
            self.img_height = math.ceil(self.lat_range / self.img_unit) + 1 
            self.road_num = 39952
            self.block_size = 157
            self.min_len = 5

        elif self.dataset_name == "Chengdu_Taxi":
            self.lon_0 = 103.4784
            self.lat_0 = 30.2945
            self.lon_range = 1.062
            self.lat_range = 0.705
            self.img_unit = 0.005
            self.img_width = math.ceil(self.lon_range / self.img_unit) + 1  
            self.img_height = math.ceil(self.lat_range / self.img_unit) + 1 
            self.road_num = 14201
            self.block_size = 119
            self.min_len = 5
        else:
            raise NotImplementedError()

    def gps2grid(self, lon, lat):
        x = math.floor(abs(lon - self.lon_0) / self.img_unit)
        y = math.floor((lat - self.lat_0) / self.img_unit)
        assert 0 <= x <= self.img_width
        assert 0 <= y <= self.img_height
        return x, y


class ListDataset(Dataset):
    def __init__(self, data):
        """
        data: list
        """
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_logger(log_dir="./logs/", log_prefix=""):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    ts = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    fh = logging.FileHandler(f"{log_dir}/{log_prefix}-{ts}.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def encode_time(timestamp):
    if "T" in timestamp:
        time = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")
    else:
        time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")

    if time.weekday() == 5 or time.weekday() == 6:
        return time.hour * 60 + time.minute + 1440
    else:
        return time.hour * 60 + time.minute


def read_rid_gps(data):
    rid_gps_file = f"./data/{data}/rid_gps.json"

    if os.path.exists(rid_gps_file):
        with open(rid_gps_file, "r") as f:
            rid_gps = json.load(f)
    else:
        rid_gps = {}
        rid_info = pd.read_csv(f"./data/{data}/roadmap.geo")
        for index, row in tqdm(
            rid_info.iterrows(), total=rid_info.shape[0], desc="cal road gps dict"
        ):
            rid = row["geo_id"]
            coordinate = eval(row["coordinates"])
            road_line = LineString(coordinates=coordinate)
            center_coord = road_line.centroid
            center_lon, center_lat = center_coord.x, center_coord.y
            rid_gps[str(rid)] = (center_lon, center_lat)
        with open(rid_gps_file, "w") as f:
            json.dump(rid_gps, f)
    return rid_gps


def read_start_t_probs(data):
    split = "tra"
    prob_file = f"./data/{data}/start_t_probs.pt"
    if os.path.exists(prob_file):
        probs = torch.load(prob_file)
    else:
        df = pd.read_csv(f"./data/{data}/traj_{split}.csv")
        df["start_t"] = df["time_list"].apply(lambda x: encode_time(x.split(",")[0]))
        res = df["start_t"].value_counts()
        vals, cnts = res.index, res.values
        probs = np.zeros(2880)
        probs[vals] = cnts
        probs = torch.from_numpy(probs / probs.sum())
        torch.save(probs, prob_file)
    return probs


def read_od_pair_distribution(data):
    split = "tra"
    od_pair_file = f"./data/{data}/od_and_probs_float.pt"
    if os.path.exists(od_pair_file):
        od_and_probs_float = torch.load(od_pair_file)
    else:
        df = pd.read_csv(f"./data/{data}/traj_{split}.csv")
        df["origin"] = df["rid_list"].apply(lambda x: int(x.split(",")[0]))
        df["destination"] = df["rid_list"].apply(lambda x: int(x.split(",")[-1]))
        od_cnt_df = (
            df.groupby(["origin", "destination"])
            .count()
            .sort_values("mm_id", ascending=False)
            .reset_index()
        )
        od_and_probs_float = torch.tensor(
            od_cnt_df[["origin", "destination", "mm_id"]].values
        ).float()
        # od_and_probs_float[:, -1] = od_and_probs_float[:, -1]/od_and_probs_float[:, -1].sum()
        torch.save(od_and_probs_float, od_pair_file)
    return od_and_probs_float


def read_adjcent_file(data):
    adjacent_np_file = f"./data/{data}/adjacent_mx.npz"
    map_manager = MapManager(data)

    if os.path.exists(adjacent_np_file):
        adj_mx = sp.load_npz(adjacent_np_file)
    else:
        road_rel = pd.read_csv(f"./data/{data}/roadmap.rel")
        # construct adjcent matrix with sparse matrix
        adj_row = []
        adj_col = []
        adj_data = []
        adj_set = set()
        for index, row in tqdm(
            road_rel.iterrows(), total=road_rel.shape[0], desc="cal adj mx"
        ):
            f_id = row["origin_id"]
            t_id = row["destination_id"]
            if (f_id, t_id) not in adj_set:
                adj_set.add((f_id, t_id))
                adj_row.append(f_id)
                adj_col.append(t_id)
                adj_data.append(1.0)
        num = map_manager.road_num
        adj_mx = sp.coo_matrix((adj_data, (adj_row, adj_col)), shape=(num, num))
        sp.save_npz(adjacent_np_file, adj_mx)
    return adj_mx


def get_max_from_str(x):
    if type(x) is int:
        return x
    elif isinstance(eval(x), list):
        return max(list(map(int, eval(x))))
    elif isinstance(eval(x), int):
        return int(x)


def read_node_feature_file(data="", device=""):
    node_feature_file = f"./data/{data}/node_feature.pt"

    if os.path.exists(node_feature_file):
        node_features = torch.load(node_feature_file, map_location="cpu").to(device)
    else:
        road_info = pd.read_csv(f"./data/{data}/roadmap.geo")
        vocab_size = road_info["geo_id"].max()
        assert road_info["geo_id"].max() + 1 == len(road_info)
        na_value = params_map[data]["na_value"]
        encode_feature = ["highway", "oneway", "length"] + list(na_value.keys())
        node_features = road_info[encode_feature]
        node_features = node_features.fillna(na_value)

        if data in ["Shanghai_Taxi"]:
            node_features["maxspeed"] = node_features["maxspeed"].apply(
                lambda x: get_max_from_str(x)
            )
            node_features["width"] = node_features["maxspeed"].apply(
                lambda x: get_max_from_str(x)
            )

        # normalization for continuous attribution
        norm_dict = params_map[data]["norm_dict"]
        for k, v in norm_dict.items():
            d = node_features[k]
            min_ = d.min()
            max_ = d.max()
            dnew = (d - min_) / (max_ - min_)
            node_features = node_features.drop(labels=k, axis=1)
            node_features.insert(v, k, dnew)

        # one-hot encoding for discrete attribution
        onehot_list = params_map[data]["onehot_list"]
        label_encoder = LabelEncoder()
        for label in onehot_list:
            encoded_label = label_encoder.fit_transform(road_info[label])
            node_features["{}_encoded".format(label)] = encoded_label
        node_features = node_features.drop(columns=onehot_list)

        with open(f"./data/{data}/rid_gps.json", "r") as f:
            rid_gps = json.load(f)
        lon_grid = []  # x
        lat_grid = []  # y
        total_road = node_features.shape[0]
        map_manager = MapManager(dataset_name=data)
        for i in range(total_road):
            gps = rid_gps[str(i)]
            x, y = map_manager.gps2grid(lon=gps[0], lat=gps[1])
            lon_grid.append(x)
            lat_grid.append(y)
        node_features["lon_grid"] = lon_grid
        node_features["lat_grid"] = lat_grid
        node_features = node_features.values
        #  cache of node_features
        node_features = torch.FloatTensor(node_features)
        torch.save(node_features, node_feature_file)
        node_features = node_features.to(device)

    vocab_size = len(node_features)
    return node_features, vocab_size


def read_road2grid(data, map_manager):
    road_gps = read_rid_gps(data)
    road2grid_file = f"./data/{data}/road2grid.json"
    if not os.path.exists(road2grid_file):
        road2grid = {}
        for road in road_gps:
            gps = road_gps[road]
            x = math.ceil((gps[0] - map_manager.lon_0) / map_manager.img_unit)
            y = math.ceil((gps[1] - map_manager.lat_0) / map_manager.img_unit)
            road2grid[road] = (x, y)
        with open(road2grid_file, "w") as f:
            json.dump(road2grid, f)
    else:
        with open(road2grid_file, "r") as f:
            road2grid = json.load(f)
    return road2grid


def add_eos_and_pad_seq(seqs, EOS=None, mode="no-eos"):
    max_seq = 300
    valid_len = [len(seq) for seq in seqs]
    for i, seq in enumerate(seqs):
        if valid_len[i] < max_seq:
            if mode == "add-eos":
                seq.append(EOS)
                valid_len[i] += 1
                if valid_len[i] < max_seq:
                    seq.extend([0] * (max_seq - valid_len[i]))
            else:
                seq.extend([0] * (max_seq - valid_len[i]))
        assert len(seq) == max_seq
    return seqs, valid_len


def my_collate_fn(indices, adj, dist_geo, device):
    trace_loc = []
    trace_tim = []
    for i in indices:
        trace_loc.append(torch.tensor(i[0]))
        trace_tim.append(torch.tensor(i[1]))
    trace_loc = pad_sequence(trace_loc, batch_first=True, padding_value=0)
    trace_tim = pad_sequence(trace_tim, batch_first=True, padding_value=-1).float()

    x_seq = trace_loc[:, :-1].clone()
    y_seq = trace_loc[:, 1:].clone()
    mask = trace_tim[:, 1:] == -1
    y_seq[mask] = -1
    des_seq = y_seq[:, -1].unsqueeze(1).repeat(1, y_seq.shape[-1])

    return [
        x_seq.to(device),
        y_seq.to(device),
        trace_tim.to(device),
        torch.from_numpy(adj[x_seq]).to(device),
        torch.from_numpy(dist_geo[x_seq]).to(device),
        torch.from_numpy(dist_geo[des_seq]).to(device),
    ]


def generate_data_loader(
    city, split, batch_size=None, adj=None, dist=None, device=None
):
    if split == "tes":
        data = []
        df = pd.read_csv(f"./data/{city}/traj_{split}.csv")
        for index, row in tqdm(
            df.iterrows(), total=df.shape[0], desc="generate test data"
        ):
            traj_loc = list(map(int, row["rid_list"].split(",")))
            traj_tim = list(map(encode_time, row["time_list"].split(",")))
            data.append([traj_loc, traj_tim])
        return data
    elif split == "tra_and_val":
        data_tra = []
        df_tra = pd.read_csv(f"./data/{city}/traj_tra.csv")
        for index, row in tqdm(
            df_tra.iterrows(),
            total=df_tra.shape[0],
            desc="generate training data loader",
        ):
            traj_loc = list(map(int, row["rid_list"].split(",")))
            traj_tim = list(map(encode_time, row["time_list"].split(",")))
            data_tra.append([traj_loc, traj_tim])
        tra_dataset = ListDataset(data_tra)

        data_val = []
        df_val = pd.read_csv(f"./data/{city}/traj_val.csv")
        for index, row in tqdm(
            df_val.iterrows(), total=df_val.shape[0], desc="generate valid data loader"
        ):
            traj_loc = list(map(int, row["rid_list"].split(",")))
            traj_tim = list(map(encode_time, row["time_list"].split(",")))
            data_val.append([traj_loc, traj_tim])
        val_dataset = ListDataset(data_val)
        return DataLoader(
            tra_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda b: my_collate_fn(b, adj, dist, device),
        ), DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda b: my_collate_fn(b, adj, dist, device),
        )
    else:
        print("Unvalid split name!")


def read_data_from_file(fp):
    path = []
    with open(fp, 'r') as f:
        lines = f.readlines()
        for line in lines:
            pois = line.split(' ')
            path.append([int(poi) for poi in pois])
    return path