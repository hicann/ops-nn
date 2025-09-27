#!/usr/bin/python3
import sys
import os
import numpy as np
import re
import tensorflow as tf

def parse_str_to_shape_list(shape_str):
    shape_list = []
    shape_str_arr = re.findall(r"\{([0-9 ,]+)\}", shape_str)
    for shape_str in shape_str_arr:
        single_shape = [int(x) for x in shape_str.split(",")]
        shape_list.append(single_shape)
    return shape_list


def gen_data_and_golden(shape_str, scale_value=2.0, d_type="float32"):
    d_type_dict = {
        "float32": np.float32,
        "float16": np.float16,
        "int32": np.int32,
        "bfloat16_t": tf.bfloat16.as_numpy_dtype
    }
    np_type = d_type_dict[d_type]
    shape_list = parse_str_to_shape_list(shape_str)
    if d_type == "bfloat16_t":
        scalar = np.array(scale_value).astype(np.float32)
    else:
        scalar = np.array(scale_value).astype(np_type)
    for index, shape in enumerate(shape_list):
        tmp_input_1 = np.ones(shape) * 100
        tmp_input_2 = np.ones(shape) * 100
        tmp_golden = tmp_input_1.astype(np_type) + tmp_input_2.astype(np_type) * scalar

        tmp_input_1.astype(np_type).tofile(f"{d_type}_input_t_foreach_add{index}_1.bin")
        tmp_input_2.astype(np_type).tofile(f"{d_type}_input_t_foreach_add{index}_2.bin")
        tmp_golden.astype(np_type).tofile(f"{d_type}_golden_t_foreach_add{index}.bin")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Param num must be 4.")
        exit(1)
    # 清理bin文件
    os.system("rm -rf *.bin")
    gen_data_and_golden(sys.argv[1], sys.argv[2], sys.argv[3])
