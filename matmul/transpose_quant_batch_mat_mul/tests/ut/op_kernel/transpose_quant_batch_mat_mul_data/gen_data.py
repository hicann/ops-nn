# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import sys
import os
import numpy as np
import struct

def float_to_bf16(f):
    """Convert float to bf16 (uint16 representation)"""
    bf16 = np.float32(f).view(np.uint32)
    bf16 = (bf16 >> 16) & 0xFFFF
    return bf16

def gen_fp8_data(M, Batch, K, N, perm_x1, perm_x2, perm_y):
    """Generate FP8 (E4M3FN) test data"""
    # FP8 uses int8_t type (E4M3FN format)
    # Generate random FP8 data (scaled to avoid overflow)
    x1 = np.random.randint(-127, 127, [M, Batch, K]).astype(np.int8)
    
    if perm_x2 == [0, 2, 1]:  # [Batch, K, N] -> [Batch, N, K]
        x2 = np.random.randint(-127, 127, [Batch, N, K]).astype(np.int8)
    else:  # [Batch, K, N]
        x2 = np.random.randint(-127, 127, [Batch, K, N]).astype(np.int8)
    
    # Scales (per-token or per-channel)
    x1_scale = np.random.uniform(2, 10, [M]).astype(np.float32)
    x2_scale = np.random.uniform(2, 10, [N]).astype(np.float32)
    
    # Output (BF16 format, stored as uint16)
    output_shape = [M, Batch, N]
    output = np.zeros(output_shape, dtype=np.float16)
    # Save to binary files
    x1.tofile("shape_x1.bin")
    x2.tofile("shape_x2.bin")
    x1_scale.tofile("shape_x1_scale.bin")
    x2_scale.tofile("shape_x2_scale.bin")
    output.tofile("shape_output.bin")
    print(f"FP8 data generated: x1={x1.shape}, x2={x2.shape}, output={output.shape}")

def gen_mxfp8_data(M, Batch, K, N, perm_x1, perm_x2, perm_y, group_size):
    """Generate MXFP8 test data with grouped quantization"""
    num_groups_x1 = K // group_size // 2
    num_groups_x2 = K // group_size // 2
    x1_scale = np.random.uniform(2, 10, [M, Batch, num_groups_x1, 2]).astype(np.uint8)
    x1 = np.random.randint(-127, 127, [M, Batch, K]).astype(np.int8)
    
    if perm_x2 == [0, 2, 1]:
        x2 = np.random.randint(-127, 127, [Batch, N, K]).astype(np.int8)
        x2_scale = np.random.uniform(2, 10, [Batch, N, num_groups_x2, 2]).astype(np.uint8)
    else:
        x2 = np.random.randint(-127, 127, [Batch, K, N]).astype(np.int8)
        x2_scale = np.random.uniform(2, 10, [Batch, num_groups_x2, N, 2]).astype(np.uint8)
    output_shape = [M, Batch, N]
    output = np.zeros(output_shape, dtype=np.uint16)
    
    x1.tofile("shape_x1.bin")
    x2.tofile("shape_x2.bin")
    x1_scale.tofile("shape_x1_scale.bin")
    x2_scale.tofile("shape_x2_scale.bin")
    output.tofile("shape_output.bin")
    
    print(f"MXFP8 data generated: x1={x1.shape}, x2={x2.shape}, output={output.shape}, group_size={group_size}")

if __name__ == "__main__":
    os.system("rm -rf *.bin")
    params = sys.argv[1:]
    
    if len(params) == 8:  # FP8 mode: M Batch K N perm_x1 perm_x2 perm_y mode
        M, Batch, K, N = [int(p) for p in params[0:4]]
        perm_x1 = [int(p) for p in params[4].split(',')]
        perm_x2 = [int(p) for p in params[5].split(',')]
        perm_y = [int(p) for p in params[6].split(',')]
        mode = int(params[7])
        
        if mode == 0:  # FP8
            gen_fp8_data(M, Batch, K, N, perm_x1, perm_x2, perm_y)
        elif mode == 1:  # MXFP8 with group_size=32
            group_size = 32
            gen_mxfp8_data(M, Batch, K, N, perm_x1, perm_x2, perm_y, group_size)