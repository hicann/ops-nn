/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// QuantMaxV2 公共定义及类型转换配置。

#ifndef QUANT_MAX_V2_COMMON_H
#define QUANT_MAX_V2_COMMON_H

#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "op_kernel/platform_util.h"
#include "quant_max_v2_struct.h"

namespace QuantMaxV2 {
using namespace AscendC;

// 动态第二遍启用双链的最大每核 tile 数：更大规模（HBM 级）shape 配对加载会打断 DMA 流水。
constexpr int64_t DUAL_CHAIN_MAX_TILES_PER_CORE = 64;
// 动态量化系数 q = dstTypeMax/amax 的溢出防护：amax 极小时（< dstTypeMax/FLT_MAX，如 E5M2 的
// 57344 对应 1.7e-34）q 溢出为 Inf，x*q 产生 Inf/NaN。把分母钳到该安全下限后 q < FLT_MAX 恒有限；
// 量化峰值 ≥ dstTypeMax 的元素由 SAT Cast 饱和到目标类型最大值——数学上正是正确输出（x=amax 处
// 本应量化到 dstTypeMax，钳制分母只会让结果 ≥ 正确值，饱和截断后一致）。对 HiF8 的 dstTypeMax=15
// 该下限留有 ~2e33 倍裕量，正常输入永不触发。
constexpr float DYNAMIC_AMAX_FLOOR = 1.8e-34f; // > 57344/FLT_MAX ≈ 1.69e-34，覆盖最大 dstTypeMax
// 第二遍首遍尾部保留的 tile 数（tensor 路反向复用 L2 热数据）。
constexpr int64_t RETAINED_TAIL_NUM = 3;
using namespace Ops::Base;

// 向量/标量除法策略：true 时用 Host 预计算的倒数做乘法高位求商并以余数修正（结果与整数除法逐位等价）；
// false 时直接用整数除法。当前取 false——分母是每核固定的 ubFactor，除法只发生在每组/每核入口的
// 标量段（每组一次、每核一次），不在 tile 热环内，倒数版在实测中未带来可分辨的收益。
constexpr bool QUANT_MAX_V2_USE_RECIPROCAL_DIVISION = false;

// Host 预计算 floor(2^32 / divisor)。对非负 32 位分子，乘法高位得到的商最多少 1；通过余数修正，保持 tile 和 offset
// 精确。
__aicore__ inline int64_t QuantMaxExactDivide(int64_t numerator, int64_t divisor, uint32_t reciprocal)
{
    if constexpr (!QUANT_MAX_V2_USE_RECIPROCAL_DIVISION) {
        return numerator / divisor;
    }
    if (divisor == 1) {
        return numerator;
    }
    if (static_cast<uint64_t>(numerator) > 0xffffffffULL || reciprocal == 0) {
        return numerator / divisor;
    }
    const uint64_t quotient = (static_cast<uint64_t>(numerator) * reciprocal) >> 32;
    const uint64_t remainder = static_cast<uint64_t>(numerator) - quotient * divisor;
    return static_cast<int64_t>(quotient + (remainder >= static_cast<uint64_t>(divisor)));
}

template <typename T, typename U, uint64_t RoundMode>
class QuantMaxV2Base {
public:
    __aicore__ inline QuantMaxV2Base(){};

protected:
    // 独立 VF 函数需要访问类型转换配置，因此这些常量保持公开。
public:
    constexpr static AscendC::Reg::CastTrait CAST_TRAIT_HALF_TO_FP32 = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
        RoundMode::UNKNOWN};

    // bf16→fp32 加宽为无损转换（位模式左移 16 位即得），用于双链/归并前的数值域抬升，
    // 不涉及舍入与饱和，SatMode/RoundMode 均 UNKNOWN。
    constexpr static AscendC::Reg::CastTrait CAST_TRAIT_BF16_TO_FP32 = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
        RoundMode::UNKNOWN};

    static constexpr AscendC::Reg::CastTrait CAST_TRAIT_FP32_TO_HIFP8 = []() {
        if constexpr (RoundMode == TPL_ROUND_MODE_HYBRID) {
            return AscendC::Reg::CastTrait{AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT,
                                           AscendC::Reg::MaskMergeMode::ZEROING, RoundMode::CAST_HYBRID};
        } else {
            return AscendC::Reg::CastTrait{AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT,
                                           AscendC::Reg::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};
        }
    }();

    static constexpr AscendC::Reg::CastTrait CAST_TRAIT_FP32_TO_X_DETYPE = []() {
        return AscendC::Reg::CastTrait{AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT,
                                       AscendC::Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    }();

    static constexpr AscendC::Reg::CastTrait CAST_TRAIT_FP32_TO_FP8E5M2 = []() {
        return AscendC::Reg::CastTrait{AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT,
                                       AscendC::Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    }();

    static constexpr AscendC::Reg::CastTrait CAST_TRAIT_FP32_TO_FP8E4M3 = []() {
        return AscendC::Reg::CastTrait{AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT,
                                       AscendC::Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    }();
};

} // namespace QuantMaxV2
#endif
