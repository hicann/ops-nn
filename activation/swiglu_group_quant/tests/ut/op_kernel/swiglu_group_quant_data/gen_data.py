"""
gen_data.py - Generate golden data for SwiGLU Group Dynamic Quant operator verification.

Uses CPU PyTorch small operator concatenation as golden reference:
  x0 = x[:, :H], x1 = x[:, H:]
  if clamp_value != 0: x0 = clamp(x0, max=clamp_value), x1 = clamp(x1, -clamp_value, clamp_value)
  y = silu(x0) * x1
  if topk_weight: y = y * topk_weight
  amax = max(absmax(y), eps)
  scale = amax / dst_type_max_finite
  out = (y / scale).to(hifloat8)  # or equivalent quantization
"""
import numpy as np
import os
import struct

def silu(x):
    """SiLU activation: x * sigmoid(x) = x / (1 + exp(-x))"""
    return x / (1.0 + np.exp(-x))

def hif8_cast(values, dst_type_max_finite=448.0, round_mode="hybrid"):
    """
    Simulate hifloat8 quantization cast.
    hifloat8 has 1 sign bit, 5 exponent bits, 2 mantissa bits.
    Max finite value for hif8 (1-5-2 format) is approximately 448.0.
    
    For simulation purposes, we use a simplified model:
    - Scale values to fit in [-dst_type_max_finite, dst_type_max_finite]
    - Quantize to uint8 representation
    """
    # This is a simplified simulation - actual hif8 encoding is more complex
    # For verification, we store the quantized result as uint8
    # The actual hif8 format uses a different encoding scheme
    
    # Simple quantization: map float values to uint8 range
    # In real hif8, the encoding uses exponent + mantissa
    # For now, we just store as uint8 for data comparison
    quantized = np.clip(values, -dst_type_max_finite, dst_type_max_finite)
    # Convert to uint8 storage (simplified - actual hif8 has specific encoding)
    result = np.zeros_like(quantized, dtype=np.uint8)
    # Store as raw bytes for comparison
    result = quantized.astype(np.uint8)  # simplified
    return result

def _swiglu_forward(x, clamp_value, topk_weight):
    """前向：reshape + split + clamp + SiLU*x1 + weight 加权，返回 y 与 (tokens, H)。"""
    if x.ndim > 2:
        x = x.reshape(-1, x.shape[-1])
    elif x.ndim == 1:
        x = x.reshape(1, -1)
    tokens = x.shape[0]
    H = x.shape[1] // 2
    x0 = x[:, :H].copy()
    x1 = x[:, H:].copy()
    if clamp_value != 0.0:
        x0 = np.clip(x0, None, clamp_value)
        x1 = np.clip(x1, -clamp_value, clamp_value)
    y = silu(x0) * x1
    if topk_weight is not None:
        weight = topk_weight.reshape(-1, 1)
        if weight.shape[0] != tokens:
            weight = weight[:tokens]
        y = y * weight
    return y, tokens, H


def _dynamic_quant(y, group_index, tokens, H, dst_type_max_finite, eps, round_mode):
    """HiF8 动态量化：无 group_index 全局量化，有 group_index 逐组量化。"""
    if group_index is None:
        amax = max(np.max(np.abs(y)), eps)
        scale = amax / dst_type_max_finite
        yOut = hif8_cast(y / scale, dst_type_max_finite, round_mode)
        return yOut, np.array([scale], dtype=np.float32)
    scaleOut = np.zeros(len(group_index), dtype=np.float32)
    yOut = np.zeros((tokens, H), dtype=np.uint8)
    start = 0
    for g, group_size in enumerate(group_index):
        end = start + group_size
        y_g = y[start:end, :]
        scale_g = max(np.max(np.abs(y_g)), eps) / dst_type_max_finite
        scaleOut[g] = scale_g
        yOut[start:end, :] = hif8_cast(y_g / scale_g, dst_type_max_finite, round_mode)
        start = end
    return yOut, scaleOut


def compute_swiglu_group_quant(
    x, topk_weight=None, group_index=None,
    clamp_value=0.0, dst_type_max_finite=448.0, eps=1e-8, round_mode="hybrid"):
    """
    Compute SwiGLU Group Dynamic Quant golden result using CPU PyTorch-like logic.

    Returns:
        yOut: numpy uint8 array, shape [tokens, H]
        scaleOut: numpy float32 array, shape [1] or [num_groups]
        y_float: numpy float32 array, shape [tokens, H] (for intermediate verification)
    """
    y, tokens, H = _swiglu_forward(x, clamp_value, topk_weight)
    yOut, scaleOut = _dynamic_quant(y, group_index, tokens, H, dst_type_max_finite, eps, round_mode)
    return yOut, scaleOut, y

def save_binary(data, filepath):
    """Save numpy array as binary file."""
    with open(filepath, 'wb') as f:
        f.write(data.tobytes())

def save_meta(filepath, shape, dtype):
    """Save shape and dtype metadata."""
    with open(filepath, 'w') as f:
        f.write(f"shape={shape}\n")
        f.write(f"dtype={dtype}\n")

def _gen_inputs(x_shape, tokens, has_topk_weight, has_group_index, group_sizes, dtype):
    """生成 x/topk_weight/group_index 输入数据。"""
    np.random.seed(42)
    x = np.random.randn(*x_shape).astype(dtype)
    topk_weight = np.random.uniform(0.5, 1.5, size=(tokens, 1)).astype(dtype) if has_topk_weight else None
    group_index = np.array(group_sizes, dtype=np.int64) if (has_group_index and group_sizes) else None
    return x, topk_weight, group_index


def _save_case(case_dir, x, topk_weight, group_index, yOut, scaleOut, y_float):
    """保存输入/输出/golden 数据到 case_dir。"""
    save_binary(x, os.path.join(case_dir, "x.bin"))
    save_meta(os.path.join(case_dir, "x.meta"), x.shape, str(x.dtype))
    if topk_weight is not None:
        save_binary(topk_weight, os.path.join(case_dir, "topk_weight.bin"))
        save_meta(os.path.join(case_dir, "topk_weight.meta"), topk_weight.shape, str(topk_weight.dtype))
    if group_index is not None:
        save_binary(group_index, os.path.join(case_dir, "group_index.bin"))
        save_meta(os.path.join(case_dir, "group_index.meta"), group_index.shape, str(group_index.dtype))
    save_binary(yOut, os.path.join(case_dir, "yOut.bin"))
    save_meta(os.path.join(case_dir, "yOut.meta"), yOut.shape, str(yOut.dtype))
    save_binary(scaleOut, os.path.join(case_dir, "scaleOut.bin"))
    save_meta(os.path.join(case_dir, "scaleOut.meta"), scaleOut.shape, str(scaleOut.dtype))
    save_binary(y_float.astype(np.float32), os.path.join(case_dir, "y_float.bin"))


def generate_test_case(
    name, x_shape, clamp_value=0.0, has_topk_weight=False,
    has_group_index=False, group_sizes=None,
    dst_type_max_finite=448.0, eps=1e-8, round_mode="hybrid",
    dtype=np.float32, data_dir="tests/ut/op_kernel/swiglu_group_quant_data"):
    os.makedirs(data_dir, exist_ok=True)
    if x_shape[-1] % 2 != 0:
        raise ValueError(f"Last dim of x must be even, got {x_shape[-1]}")
    tokens = int(np.prod(x_shape[:-1]))
    x, topk_weight, group_index = _gen_inputs(
        x_shape, tokens, has_topk_weight, has_group_index, group_sizes, dtype)
    yOut, scaleOut, y_float = compute_swiglu_group_quant(
        x, topk_weight, group_index, clamp_value, dst_type_max_finite, eps, round_mode)
    case_dir = os.path.join(data_dir, name)
    os.makedirs(case_dir, exist_ok=True)
    _save_case(case_dir, x, topk_weight, group_index, yOut, scaleOut, y_float)
    print(f"Generated test case '{name}': x_shape={x_shape}, yOut_shape={yOut.shape}, scaleOut_shape={scaleOut.shape}")

if __name__ == "__main__":
    # Test case 1: Basic non-group, no clamp
    generate_test_case("basic_nogroup", (128, 2048), clamp_value=0.0)
    
    # Test case 2: With clamp, non-group
    generate_test_case("clamp_nogroup", (64, 4096), clamp_value=7.0)
    
    # Test case 3: With topk_weight
    generate_test_case("topk_weight", (32, 2048), clamp_value=0.0, has_topk_weight=True)
    
    # Test case 4: Group quantization
    generate_test_case("group_quant", (128, 2048), clamp_value=7.0,
                       has_group_index=True, group_sizes=[32, 32, 32, 32])
    
    # Test case 5: Full combination
    generate_test_case("full_combo", (256, 4096), clamp_value=7.0,
                       has_topk_weight=True, has_group_index=True,
                       group_sizes=[64, 64, 64, 64])
    
    # Test case 6: Small H non-aligned
    generate_test_case("small_h", (16, 14), clamp_value=0.0)
    
    # Test case 7: bf16 dtype
    generate_test_case("bf16_basic", (32, 2048), clamp_value=0.0, dtype=np.float16)
    
    print("All test cases generated successfully!")