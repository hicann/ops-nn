#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import glob
import os

# 尝试导入 ml_dtypes
try:
    import ml_dtypes
except ImportError:
    pass # 后面用到时再报错

curr_dir = os.path.dirname(os.path.realpath(__file__))

def compare_data(golden_file_lists, output_file_lists, d_type):
    # 1. 确定 Numpy 数据类型
    if d_type == "float16":
        np_dtype = np.float16
    elif d_type == "float32":
        np_dtype = np.float32
    elif d_type in ["bf16", "bfloat16"]:
        try:
            np_dtype = ml_dtypes.bfloat16
        except NameError:
             print("Error: ml_dtypes not installed.")
             return False
    else:
        raise ValueError(f"Unsupported d_type: {d_type}")

    data_same = True
    
    # 2. 遍历比较
    for gold, out in zip(golden_file_lists, output_file_lists):
        print(f"Comparing:\n  Gold: {os.path.basename(gold)}\n  Out : {os.path.basename(out)}")
        
        # 加载数据 (强制使用指定的 np_dtype，忽略文件名可能带来的误导)
        tmp_gold = np.fromfile(gold, dtype=np_dtype)
        tmp_out = np.fromfile(out, dtype=np_dtype)

        # 检查元素数量是否一致
        if tmp_gold.size != tmp_out.size:
            print(f"FAILED! Size mismatch: Gold {tmp_gold.size} vs Out {tmp_out.size}")
            data_same = False
            continue

        # 执行比较
        # 注意：bf16 和 float16 不能混用 isclose，这里强制两者类型一致
        diff_res = np.isclose(tmp_out, tmp_gold, 0, 0, True)
        diff_idx = np.where(diff_res != True)[0]
        
        if len(diff_idx) == 0:
            print("PASSED!")
        else:
            print("FAILED!")
            # 打印前 5 个错误
            for idx in diff_idx[:5]:
                print(f"  index: {idx}, output: {tmp_out[idx]}, golden: {tmp_gold[idx]}")
            data_same = False

    return data_same

def get_file_lists(dtype):
    # ================== 关键修改 ==================
    # 增加 dtype 过滤，防止 glob 扫到其他类型的文件导致错配
    # 假设文件名格式包含 dtype 字符串 (例如 golden_float16_... 或 golden_bf16_...)
    
    # 归一化搜索关键字
    search_key = "bf16" if dtype in ["bf16", "bfloat16"] else dtype
    
    # 只搜索包含 search_key 的 bin 文件
    golden_pattern = f"{curr_dir}/*golden*{search_key}*.bin"
    output_pattern = f"{curr_dir}/*{search_key}*output*.bin"
    
    golden_file_lists = sorted(glob.glob(golden_pattern))
    output_file_lists = sorted(glob.glob(output_pattern))
    
    # 简单的数量检查
    if len(golden_file_lists) != len(output_file_lists):
        print(f"Warning: File count mismatch! Golden: {len(golden_file_lists)}, Output: {len(output_file_lists)}")
        print(f"Golden pattern: {golden_pattern}")
        print(f"Output pattern: {output_pattern}")

    return golden_file_lists, output_file_lists

def process(d_type):
    golden_file_lists, output_file_lists = get_file_lists(d_type)
    
    if not golden_file_lists:
        print(f"Error: No golden files found for dtype '{d_type}' in {curr_dir}")
        return False
        
    result = compare_data(golden_file_lists, output_file_lists, d_type)
    print("Compare result:", result)
    return result

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 compare_data.py <dtype>")
        exit(1)
    
    # 获取参数并执行
    dtype_arg = sys.argv[1]
    ret = process(dtype_arg)
    exit(0 if ret else 1)