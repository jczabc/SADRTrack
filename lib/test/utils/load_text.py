import numpy as np
import pandas as pd

import numpy as np

# def load_text_numpy(path, delimiter=(' ', ',', '\t'), dtype=np.float64):
#     if isinstance(delimiter, str):
#         delimiter = [delimiter]
#
#     for d in delimiter:
#         try:
#             data = []
#             with open(path, 'r') as f:
#                 for line in f:
#                     # 分割每行
#                     parts = line.strip().split(d)
#                     # 提取能转成 float 的字段
#                     numeric = []
#                     for p in parts:
#                         try:
#                             numeric.append(dtype(p))
#                         except ValueError:
#                             continue
#                     if numeric:
#                         data.append(numeric)
#             if data:
#                 # 自动补齐为矩阵（不规则行也能处理）
#                 max_len = max(len(row) for row in data)
#                 padded = [row + [0.0] * (max_len - len(row)) for row in data]
#                 return np.array(padded, dtype=dtype)
#         except Exception:
#             continue
#
#     raise RuntimeError(f"Could not read file {path} with delimiters {delimiter}")
#
#
def load_text_numpy(path, delimiter, dtype):
    if isinstance(delimiter, (tuple, list)):
        for d in delimiter:
            try:
                ground_truth_rect = np.loadtxt(path, delimiter=d, dtype=dtype)
                return ground_truth_rect
            except:
                pass

        raise Exception('Could not read file {}'.format(path))
    else:
        ground_truth_rect = np.loadtxt(path, delimiter=delimiter, dtype=dtype)
        return ground_truth_rect



def load_text_pandas(path, delimiter, dtype):
    if isinstance(delimiter, (tuple, list)):
        for d in delimiter:
            try:
                ground_truth_rect = pd.read_csv(path, delimiter=d, header=None, dtype=dtype, na_filter=False,
                                                low_memory=False).values
                return ground_truth_rect
            except Exception as e:
                pass

        raise Exception('Could not read file {}'.format(path))
    else:
        ground_truth_rect = pd.read_csv(path, delimiter=delimiter, header=None, dtype=dtype, na_filter=False,
                                        low_memory=False).values
        return ground_truth_rect


def load_text(path, delimiter=' ', dtype=np.float32, backend='numpy'):
    if backend == 'numpy':
        return load_text_numpy(path, delimiter, dtype)
    elif backend == 'pandas':
        return load_text_pandas(path, delimiter, dtype)


def load_str(path):
    with open(path, "r") as f:
        text_str = f.readline().strip().lower()
    return text_str
