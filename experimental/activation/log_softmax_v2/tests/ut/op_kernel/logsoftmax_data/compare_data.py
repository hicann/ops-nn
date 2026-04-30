import sys
import numpy as np
import os

def main():
    # 参数解析: python3 compare_data.py 'float32'
    if len(sys.argv) < 2:
        print("Usage: python3 compare_data.py <dtype_str>")
        sys.exit(1)

    dtype_str = sys.argv[1]

    if dtype_str == 'float32':
        np_dtype = np.float32
        # float32 精度要求通常在 1e-4 或 1e-3 左右
        rtol = 1e-4
        atol = 1e-4
    elif dtype_str == 'float16':
        np_dtype = np.float16
        # float16 精度要求较低，通常在 1e-3 左右
        rtol = 1e-3
        atol = 1e-3
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")

    # 定义文件名
    # C++ 算子输出的文件
    output_filename = f"{dtype_str}_output_logsoftmax.bin"
    # gen_data.py 生成的真值文件
    golden_filename = f"{dtype_str}_golden_logsoftmax.bin"

    # 检查文件是否存在
    if not os.path.exists(output_filename):
        print(f"[compare_data] Error: Output file {output_filename} not found.")
        sys.exit(1)
    if not os.path.exists(golden_filename):
        print(f"[compare_data] Error: Golden file {golden_filename} not found.")
        sys.exit(1)

    # 读取数据
    actual_data = np.fromfile(output_filename, dtype=np_dtype)
    golden_data = np.fromfile(golden_filename, dtype=np_dtype)

    print(f"[compare_data] Comparing {dtype_str} data...")
    print(f"  Shape (flattened): {actual_data.shape}")

    # 检查数据大小是否一致
    if actual_data.size != golden_data.size:
        print(f"[compare_data] Error: Size mismatch. Actual: {actual_data.size}, Golden: {golden_data.size}")
        sys.exit(1)

    # 进行比较 (使用 np.allclose)
    is_close = np.allclose(actual_data, golden_data, rtol=rtol, atol=atol)

    if is_close:
        print("[compare_data] Compare Success!")
        sys.exit(0)
    else:
        # 如果失败，打印最大误差信息以便调试
        diff = np.abs(actual_data - golden_data)
        max_diff = np.max(diff)
        max_diff_idx = np.argmax(diff)
        
        print("[compare_data] Compare Failed!")
        print(f"  Max Diff: {max_diff}")
        print(f"  At Index: {max_diff_idx}")
        print(f"  Actual: {actual_data[max_diff_idx]}")
        print(f"  Golden: {golden_data[max_diff_idx]}")
        sys.exit(1)

if __name__ == "__main__":
    main()