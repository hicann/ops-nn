"""
gen_tiling.py - Generate tiling data for SwiGLU Group Dynamic Quant operator unit tests.

Tiling struct layout must match op_host/swiglu_group_quant_tiling.h:
  14 scalar uint32/float fields + coreGroupStartArr[64] + coreGroupCountArr[64]
"""
import numpy as np
import os
import struct

MAX_CORE_COUNT = 64


def ceil_div(a, b):
    return 0 if b == 0 else (a + b - 1) // b


def continuous_group_split(total_core, group_num, group_tokens):
    """Continuous segment split: each core handles a contiguous group range [start, start+count)."""
    used_core_num = min(total_core, group_num, MAX_CORE_COUNT)
    start_arr = [0] * MAX_CORE_COUNT
    count_arr = [0] * MAX_CORE_COUNT
    if used_core_num == 0 or group_num == 0:
        return used_core_num, start_arr, count_arr

    total_group_tokens = sum(group_tokens)
    target_load = ceil_div(total_group_tokens, used_core_num)

    g = 0
    for core in range(used_core_num):
        start = g
        remaining_cores = used_core_num - core - 1
        if core == used_core_num - 1:
            g = group_num
        else:
            core_load = 0
            while g < group_num:
                remaining_groups = group_num - g
                if remaining_groups <= remaining_cores:
                    break
                core_load += group_tokens[g]
                g += 1
                if core_load >= target_load:
                    break
        start_arr[core] = start
        count_arr[core] = g - start
    return used_core_num, start_arr, count_arr


def _calc_core_distribution(isGroup, groupTokens, groupNum, coreNumAll, totalTokens):
    """计算核间分配：分组模式用 continuous_group_split，非分组模式均分 tokens。"""
    if isGroup == 1 and groupTokens is not None and groupNum > 0:
        usedCoreNum, startArr, countArr = continuous_group_split(coreNumAll, groupNum, groupTokens)
        return usedCoreNum, startArr, countArr, 0
    usedCoreNum = min(coreNumAll, totalTokens)
    tokensPerCore = ceil_div(totalTokens, usedCoreNum) if usedCoreNum > 0 else 0
    return usedCoreNum, [0] * MAX_CORE_COUNT, [0] * MAX_CORE_COUNT, tokensPerCore


def _pack_tiling_data(totalTokens, dim2H, dimH, isGroup, hasWeight,
                      hasClamp, outputOrigin, clampLimit, dstTypeMaxFinite,
                      tileTokens, usedCoreNum, tokensPerCore,
                      groupTokensSum, minLoadCoreIdx,
                      coreGroupStartArr, coreGroupCountArr):
    """按 op_host tiling 结构打包为二进制（14 标量 + 2×64 数组）。"""
    data = struct.pack(
        'IIIIIIIffIIIII',
        totalTokens, dim2H, dimH, isGroup, hasWeight, hasClamp,
        outputOrigin, clampLimit, dstTypeMaxFinite,
        tileTokens, usedCoreNum, tokensPerCore, groupTokensSum, minLoadCoreIdx)
    data += struct.pack('64I', *coreGroupStartArr)
    data += struct.pack('64I', *coreGroupCountArr)
    return data


def generate_tiling_data(
    totalTokens, dim2H, isGroup=0, hasWeight=0, hasClamp=0,
    clampLimit=0.0, dstTypeMaxFinite=448.0,
    groupNum=0, coreNumAll=1,
    tileTokens=None, outputOrigin=0,
    groupTokens=None,
    output_path="tests/ut/op_kernel/swiglu_group_quant_data"):
    dimH = dim2H // 2
    if tileTokens is None:
        tileTokens = totalTokens

    usedCoreNum, coreGroupStartArr, coreGroupCountArr, tokensPerCore = _calc_core_distribution(
        isGroup, groupTokens, groupNum, coreNumAll, totalTokens)

    if isGroup == 1 and groupTokens is not None and groupNum > 0:
        groupTokensSum = sum(groupTokens)
        minLoadCoreIdx = 0
        minLoad = float('inf')
        for core in range(usedCoreNum):
            s = coreGroupStartArr[core]
            c = coreGroupCountArr[core]
            coreLoad = sum(groupTokens[s:s + c]) if c > 0 else 0
            if coreLoad < minLoad:
                minLoad = coreLoad
                minLoadCoreIdx = core
    else:
        groupTokensSum = totalTokens
        minLoadCoreIdx = 0

    data = _pack_tiling_data(
        totalTokens, dim2H, dimH, isGroup, hasWeight, hasClamp,
        outputOrigin, clampLimit, dstTypeMaxFinite,
        tileTokens, usedCoreNum, tokensPerCore,
        groupTokensSum, minLoadCoreIdx,
        coreGroupStartArr, coreGroupCountArr)

    os.makedirs(output_path, exist_ok=True)
    filepath = os.path.join(output_path, "tiling.bin")
    with open(filepath, 'wb') as f:
        f.write(data)

    print(f"Generated tiling data: totalTokens={totalTokens}, dim2H={dim2H}, dimH={dimH}, "
          f"isGroup={isGroup}, tileTokens={tileTokens}, usedCoreNum={usedCoreNum}")
    return filepath

if __name__ == "__main__":
    # Test case 1: Basic non-group
    generate_tiling_data(128, 2048, isGroup=0, hasClamp=0, clampLimit=0.0)

    # Test case 2: With clamp, non-group
    generate_tiling_data(64, 4096, isGroup=0, hasClamp=1, clampLimit=7.0)

    # Test case 3: With weight
    generate_tiling_data(32, 2048, isGroup=0, hasWeight=1, hasClamp=0)

    # Test case 4: Group quantization (4 groups)
    generate_tiling_data(128, 2048, isGroup=1, groupNum=4, hasClamp=1, clampLimit=7.0,
                         groupTokens=[32, 32, 32, 32], coreNumAll=2)

    # Test case 5: Small H
    generate_tiling_data(16, 14, isGroup=0, hasClamp=0)

    print("All tiling data generated successfully!")
