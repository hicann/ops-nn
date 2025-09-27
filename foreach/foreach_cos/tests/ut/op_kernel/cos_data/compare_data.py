import sys
import numpy as np
import glob
import os
import tensorflow as tf

curr_dir = os.path.dirname(os.path.realpath(__file__))

def compare_data(golden_file_lists, output_file_lists, d_type):
    np_dtype = np.float32
    if d_type == "float16":
        np_dtype = np.float16
        precision = 1/1000
    elif d_type == "float32":
        precision = 1/10000
    else:
        np_dtype = tf.bfloat16.as_numpy_dtype
        precision = 1/10
    
    data_same = True
    for gold, out in zip(golden_file_lists, output_file_lists):
        tmp_out = np.fromfile(out, np_dtype)
        tmp_gold = np.fromfile(gold, np_dtype)
        diff_res = np.isclose(tmp_out, tmp_gold, precision, 0, True)
        diff_idx = np.where(diff_res != True)[0]
        if len(diff_idx) == 0:
            print("PASSED!")
        else:
            print("FAILED!")
            for idx in diff_idx[:5]:
                print(f"index: {idx}, output: {tmp_out[idx]}, golden: {tmp_gold[idx]}")
            data_same = False
    return data_same

def get_file_lists(dtype):
    golden_file_lists = sorted(glob.glob(curr_dir + "/*golden*.bin"))
    output_file_lists = sorted(glob.glob(curr_dir + "/*output*.bin"))
    return golden_file_lists, output_file_lists

def process(d_type):
    golden_file_lists, output_file_lists = get_file_lists(d_type)
    result = compare_data(golden_file_lists, output_file_lists, d_type)

if __name__ == '__main__':
    process(sys.argv[1])