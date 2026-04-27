#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import numpy as np

def parse_str_to_shape_list(shape_str):
    # 处理类似 "(128, 64)" 的字符串
    shape_str = shape_str.strip('(').strip(')')
    shape_list = [int(x) for x in shape_str.split(",") if x.strip()]
    return tuple(shape_list)

def sigmoid(x):
    # 记录原始数据类型
    original_dtype = x.dtype
    
    # 强制上转型为 float64 保证计算精度
    # 对于 bfloat16，numpy 无法直接计算 exp，必须转为 float32/64
    x_64 = np.asarray(x, dtype=np.float64)
    
    # 数值稳定的 Sigmoid 实现
    # 防止 x 过小导致 exp(-x) 溢出，或 x 过大导致 exp(x) 溢出
    result_64 = np.where(
        x_64 >= 0,
        1 / (1 + np.exp(-x_64)),
        np.exp(x_64) / (1 + np.exp(x_64))
    )
    
    # 计算完成后强转回原始类型 (模拟算子行为)
    return result_64.astype(original_dtype)

def gen_data_and_golden(shape_str, d_type="float16"):
    np_type = None

    # --- 类型映射处理 ---
    if d_type in ["bf16", "bfloat16"]:
        try:
            from ml_dtypes import bfloat16
            np_type = bfloat16
            d_type = "bf16" # 统一文件名命名为 bf16
        except ImportError:
            print("Error: ml_dtypes library is required for bfloat16 support.")
            print("Please install it using: pip install ml_dtypes")
            sys.exit(1)
    elif d_type == "float32":
        np_type = np.float32
    elif d_type == "float16":
        np_type = np.float16
    else:
        print(f"Error: Unsupported dtype '{d_type}'. Supported: float16, float32, bf16")
        sys.exit(1)

    shape = parse_str_to_shape_list(shape_str)
    size = np.prod(shape)
    
    print(f"Generating data: Shape={shape}, Dtype={d_type} (numpy type: {np_type})")

    # --- 生成随机输入数据 ---
    # 先生成 float64 的均匀分布 (-10, 10)，涵盖 Sigmoid 的线性区和饱和区
    tmp_input_raw = np.random.uniform(low=-10, high=10, size=size).reshape(shape)
    
    # 转换为目标类型 (这一步会发生精度截断，模拟真实输入)
    tmp_input = tmp_input_raw.astype(np_type)
    
    # --- 计算 Golden 数据 ---
    # 使用截断后的 tmp_input 进行计算，确保 Golden 与算子输入一致
    tmp_golden = sigmoid(tmp_input)

    # --- 保存文件 ---
    # 文件名格式: {dtype}_input_t_sigmoid.bin
    input_file_name = f"{d_type}_input_t_sigmoid.bin"
    
    # 算子输出的 Golden 通常用于对比
    golden_file_name = f"golden_{d_type}_sigmoid.bin"

    # 注意：.tofile() 会直接转储内存中的二进制位
    tmp_input.tofile(input_file_name)
    tmp_golden.tofile(golden_file_name)
    
    print(f"Success: Generated files:")
    print(f"  Input : {input_file_name} ({os.path.getsize(input_file_name)} bytes)")
    print(f"  Golden: {golden_file_name} ({os.path.getsize(golden_file_name)} bytes)")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 gen_data.py '(shape)' 'dtype'")
        print("Example: python3 gen_data.py '(128,64)' 'float16'")
        print("Example: python3 gen_data.py '(64,64)' 'bf16'")
        sys.exit(1)
    
    # 1. 获取参数
    shape_arg = sys.argv[1]
    dtype_arg = sys.argv[2]

    # 2. 清理当前目录下旧的 .bin 文件，防止混淆
    # 注意：只清理脚本生成的特定命名的文件，避免误删
    files_to_clean = []
    for f in os.listdir('.'):
        if f.endswith(".bin") and ("sigmoid" in f):
            os.remove(f)

    # 3. 执行生成
    gen_data_and_golden(shape_arg, dtype_arg)