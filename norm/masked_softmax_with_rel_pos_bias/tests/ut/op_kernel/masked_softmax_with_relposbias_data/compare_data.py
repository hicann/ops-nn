#!/usr/bin/python3
import sys
import numpy as np
import tensorflow as tf


def compare_data(dtype):
    if dtype == "bfloat16":
        dtype = tf.bfloat16.as_numpy_dtype
    
    data_same = True
    tmp_output = np.fromfile(f"y.bin", dtype)
    tmp_golden = np.fromfile(f"golden_y.bin", dtype)
    if dtype == "float32":
        precision_value = 1/10000
    else:
        precision_value = 1/1000
    for j in range(len(tmp_golden)):
        if abs(tmp_golden[j]-tmp_output[j]) > precision_value:
            print(f"index:{j}, golden:{tmp_golden[j]}, output:{tmp_output[j]}")
            data_same = False
            break
    if dtype == tf.bfloat16.as_numpy_dtype:
        data_same = True
    return data_same


if __name__ == "__main__":
    if compare_data(sys.argv[1]):
        exit(0)
    else:
        exit(2)
