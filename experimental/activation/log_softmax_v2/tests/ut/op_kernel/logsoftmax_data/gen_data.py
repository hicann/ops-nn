import sys
import numpy as np
import os

def log_softmax(x, axis):
    """
    使用 NumPy 实现数值稳定的 log_softmax
    Formula: x - log(sum(exp(x)))
    为了防止溢出，通常使用: x - max(x) - log(sum(exp(x - max(x))))
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    # 计算 exp(x - x_max)
    exp_x = np.exp(x - x_max)
    # 计算 sum
    sum_exp_x = np.sum(exp_x, axis=axis, keepdims=True)
    # log_softmax
    return (x - x_max) - np.log(sum_exp_x)

def main():
    # 参数解析: python3 gen_data.py '(32, 64)' 'float32'
    if len(sys.argv) < 3:
        print("Usage: python3 gen_data.py <shape_str> <dtype_str>")
        sys.exit(1)

    shape_str = sys.argv[1]
    dtype_str = sys.argv[2]

    # 将字符串 shape 转换为 tuple，例如 '(32, 64)' -> (32, 64)
    # 注意：在生产环境中建议使用更安全的解析方式，但在测试脚本中 eval 很常见
    shape = eval(shape_str)
    
    # 映射 dtype 字符串到 numpy dtype
    if dtype_str == 'float32':
        np_dtype = np.float32
    elif dtype_str == 'float16':
        np_dtype = np.float16
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")

    print(f"[gen_data] Generating data for shape: {shape}, dtype: {dtype_str}")

    # 1. 生成随机输入数据 (-1 到 1 之间)
    input_data = np.random.uniform(-1, 1, size=shape).astype(np_dtype)

    # 2. 计算真值 (Golden)
    # C++代码中 tilingDatafromBin->axis = 1，所以这里 axis=1
    golden_data = log_softmax(input_data, axis=1).astype(np_dtype)

    # 3. 保存文件
    # C++ 代码期望的输入文件名: float32_input_logsoftmax.bin
    input_filename = f"{dtype_str}_input_logsoftmax.bin"
    golden_filename = f"{dtype_str}_golden_logsoftmax.bin"

    input_data.tofile(input_filename)
    golden_data.tofile(golden_filename)

    print(f"[gen_data] Generated input: {input_filename}")
    print(f"[gen_data] Generated golden: {golden_filename}")

if __name__ == "__main__":
    main()