#!/usr/bin/python3
import sys
import numpy as np


def weight_quant_batch_matmul_v2(m, k, n, group):
    np.random.random([m, k]).astype(np.float16).tofile("x.bin")
    np.random.uniform(-5, 5, [k, n]).astype(np.int8).tofile("weight.bin")
    np.random.random([n, group]).astype(np.float16).tofile("antiquant_scale.bin")
    np.random.random([n, group]).astype(np.float16).tofile("antiquant_offset.bin")
    np.random.random([n, group]).astype(np.float32).tofile("quant_scale.bin")
    np.random.random([n, group]).astype(np.float32).tofile("quant_offset.bin")

if __name__ == '__main__':
    m, k, n, group = [int(p) for p in sys.argv[1:]]
    weight_quant_batch_matmul_v2(m, k, n, group)
