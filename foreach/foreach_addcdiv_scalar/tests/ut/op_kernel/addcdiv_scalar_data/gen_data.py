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
        scalars = np.array([scale_value for i in range(len(shape_list))]).astype(np.float32)
    else:
        scalars = np.array([scale_value for i in range(len(shape_list))]).astype(np_type)
    for index, shape in enumerate(shape_list):
        tmp_input = np.ones(shape)*10
        tmp_input = tmp_input.astype(np_type)
        tmp_tensor1 = np.ones(shape)*10
        tmp_tensor1 = tmp_tensor1.astype(np_type)
        tmp_tensor2 = np.ones(shape)*10
        tmp_tensor2 = tmp_tensor2.astype(np_type)
        tmp_golden = np.ones(shape)*10
        tmp_golden = tmp_golden.astype(np_type)

        for i in range(len(tmp_golden)):
            tmp_golden[i] = tmp_input[i] + tmp_tensor1[i] / tmp_tensor2[i] * scalars[index]

        tmp_golden = tmp_golden.astype(np_type)

        tmp_input.astype(np_type).tofile(f"{d_type}_input_t_foreach_addcdiv{index}.bin")
        tmp_tensor1.astype(np_type).tofile(f"{d_type}_tensor1_t_foreach_addcdiv{index}.bin")
        tmp_tensor2.astype(np_type).tofile(f"{d_type}_tensor2_t_foreach_addcdiv{index}.bin")
        tmp_golden.astype(np_type).tofile(f"{d_type}_golden_t_foreach_addcdiv{index}.bin")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Param num must be 4.")
        exit(1)
    # 清理bin文件
    os.system("rm -rf *.bin")
    gen_data_and_golden(sys.argv[1], sys.argv[2], sys.argv[3])
