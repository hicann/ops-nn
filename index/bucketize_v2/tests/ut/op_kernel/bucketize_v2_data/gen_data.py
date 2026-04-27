#!/usr/bin/python3
import sys
import os
import numpy as np
import torch 

np.random.seed(42)

case0_params = {
    "x": np.random.randint(-2048, 4097, size=2048).astype(np.float32),
    "boundaries": np.arange(512).astype(np.float32),
    "out_int32":True,
    'right': False
}

case1_params = {
    "x": np.random.randint(-2048, 4097, size=2048).astype(np.float32),
    "boundaries": np.arange(512).astype(np.float32),
    "out_int32":True,
    'right': True
}

test_case = {
    'case0': case0_params,
    'case1': case1_params,
}


def gen_data_and_golden(case_num):
    case_params = test_case[str(case_num)]
    x = case_params['x']
    boundaries = case_params['boundaries']
    out_int32 = case_params.get('out_int32', False)
    right = case_params.get('right', False)
    x.tofile('./input_x.bin')
    boundaries.tofile('./input_boundaries.bin')
    res = torch.bucketize(torch.tensor(x), torch.tensor(boundaries), out_int32=out_int32, right=right).numpy()
    res.tofile('./res_golden.bin')
    print(case_num, res.shape, 'right:', right)

if __name__ == "__main__":
    gen_data_and_golden(sys.argv[1])