import os
import argparse
import scipy.stats
import numpy as np
from collections import Counter
from math import radians, cos, sin, asin, sqrt
from scipy.spatial.distance import euclidean, cosine, cityblock, chebyshev
import json
import hausdorff
from fastdtw import fastdtw
from collections import OrderedDict
import helpers 
import numba
from tqdm import tqdm


def edit_distance(trace1, trace2):
    trace1 = np.array(trace1).astype(np.int32)
    trace2 = np.array(trace2).astype(np.int32)
    return edit_distance_impl(trace1, trace2)
    
@numba.jit(numba.int32(numba.int32[:], numba.int32[:]), nopython=True, nogil=True, cache=True)
def edit_distance_impl(trace1, trace2):
    dp = np.zeros((2, len(trace2) + 1), dtype=np.int32)
    for i, s1 in enumerate(trace1):
        for j, s2 in enumerate(trace2):
            d = 1 if s1 != s2 else 0
            dp[(i+1)%2,j+1] = min(
                dp[i%2, j+1] + 1,
                dp[(i+1)%2, j] + 1,
                dp[i%2, j] + d,
            )
    return dp[len(trace1)%2, len(trace2)]


def hausdorff_metric(truth, pred, distance='haversine'):
        """hausdorff distance
        ref: https://github.com/mavillan/py-hausdorff
        Args:
            truth: longitude and latitude, (trace_len, 2)
            pred: longitude and latitude, (trace_len, 2)
            distance: computation method for distance, include haversine, manhattan, euclidean, chebyshev, cosine
        """
        return hausdorff.hausdorff_distance(truth, pred, distance=distance)

@numba.jit(numba.float64(numba.float64[:], numba.float64[:]), nopython=True, nogil=True, cache=True)
def haversine(array_x, array_y):
    radians = np.pi / 180.0
    lat_x = radians * array_x[0]
    lon_x = radians * array_x[1]
    lat_y = radians * array_y[0]
    lon_y = radians * array_y[1]
    dlon = lon_y - lon_x
    dlat = lat_y - lat_x
    a = sin(dlat/2)**2 + cos(lat_x) * cos(lat_y) * sin(dlon/2)**2
    return  2*asin(sqrt(a))*6371

def dtw_metric(truth, pred, distance='haversine'):
    """ dynamic time wrapping
    ref: https://github.com/slaypni/fastdtw
    Args:
        truth: longitude and latitude, (trace_len, 2)
        pred: longitude and latitude, (trace_len, 2)
        distance: computation method for distance, include haversine, manhattan, euclidean, chebyshev, cosine
    """
    if distance == 'haversine':
        distance, path = fastdtw(truth, pred, dist=haversine)
    elif distance == 'manhattan':
        distance, path = fastdtw(truth, pred, dist=cityblock)
    elif distance == 'euclidean':
        distance, path = fastdtw(truth, pred, dist=euclidean)
    elif distance == 'chebyshev':
        distance, path = fastdtw(truth, pred, dist=chebyshev)
    elif distance == 'cosine':
        distance, path = fastdtw(truth, pred, dist=cosine)
    else:
        distance, path = fastdtw(truth, pred, dist=euclidean)
    return distance