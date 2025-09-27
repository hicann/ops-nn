#!/usr/bin/python3
import sys
import os
import numpy as np
import re
import torch
import tensorflow as tf


def parse_str_to_shape_list(shape_str):
    shape_str = shape_str.strip('(').strip(')')
    shape_list = [int(x) for x in shape_str.split(",")]
    return np.array(shape_list), shape_list

def gen_data_and_golden(input_shape_str, output_size_str, d_type="float32"):
    d_type_dict = {
        "float32": np.float32,
        "float16": np.float16,
        "bfloat16_t": tf.bfloat16.as_numpy_dtype
    }
    np_type = d_type_dict[d_type]
    input_shape, _ = parse_str_to_shape_list(input_shape_str)
    _, output_size = parse_str_to_shape_list(output_size_str)

    size = np.prod(input_shape)
    tmp_input = np.random.random(size).reshape(input_shape).astype(np_type)
    x_tensor = torch.tensor(tmp_input.astype(np.float32), dtype=torch.float32)
    d = input_shape[-1] // 2
    x1 = x_tensor[..., :d]
    x2 = x_tensor[..., d:]
    m = torch.nn.GELU("none")
    x1 = m(x1)
    y_golden = x1 * x2
    tmp_golden = np.array(y_golden).astype(np_type)

    tmp_input.astype(np_type).tofile(f"{d_type}_input_gelu_mul.bin")
    tmp_golden.astype(np_type).tofile(f"{d_type}_golden_gelu_mul.bin")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Param num must be 4, actually is ", len(sys.argv))
        exit(1)
    # 清理bin文件
    os.system("rm -rf *.bin")
    gen_data_and_golden(sys.argv[1], sys.argv[2], sys.argv[3])