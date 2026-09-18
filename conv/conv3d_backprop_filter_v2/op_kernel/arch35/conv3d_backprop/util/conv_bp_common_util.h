/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file conv_bp_common_util.h
 * \brief 卷积反向公共工具层：winograd/dload 等模板共用的输入张量枚举、基本块范围描述与
 *        核号解算。★蛇形分核走位已拆分至 conv_bp_common_data_blocks.h（第二十一轮改名轮：
 *        不全挤在 util 里——依赖方向 data_blocks → util 单向）。
 *        ★文件名/守卫须保持全局唯一（第十二轮续3）：旧名 conv_bp_util.h 与引擎
 *        conv_bp_util_arch35.h / arch22 conv_bp_util.h 三者守卫 CONV_BP_UTIL_H 撞名，
 *        引擎 TU 先含引擎头时本文件内容被守卫整体跳过 → BpUtils 未声明（板测实证）。
 *        显式包含 kernel_basic_intf.h（GetBlockIdx/GetBlockNum/ASCEND_IS_AIC/DEFAULT_C0_SIZE 来源）
 */

#ifndef CONV_BP_COMMON_UTIL_H
#define CONV_BP_COMMON_UTIL_H

#include "basic_api/kernel_basic_intf.h"
#include "op_kernel/math_util.h"
#include "utils/std/algorithm.h"

namespace BpUtils {
// L1 NZ 布局 C0 datablock 元素数（32B / sizeof(T)），自 winograd conv_bp_wino_util 迁入
template <typename T>
static constexpr __aicore__ inline uint32_t C0()
{
    return AscendC::DEFAULT_C0_SIZE / sizeof(T);
}

// 卷积反向双输入标识：FMAP = 正向输入特征图（cin 侧），DY = 反向传播梯度（cout 侧）。
// 保持非 scoped enum，兼容既有代码对 FMAP/DY 枚举值的裸用法
enum InputTensor {
    FMAP,
    DY,
};

// 基本块在 cout/cin 两个维度上的覆盖范围，左闭右开区间 [idx, idx + length)
struct CoutCinRange {
    uint32_t coutIdx = 0;
    uint32_t cinIdx = 0;
    uint32_t coutLength = 0;
    uint32_t cinLength = 0;

    template <InputTensor t>
    __aicore__ inline uint32_t GetIdx() const
    {
        if constexpr (t == InputTensor::FMAP) {
            return cinIdx;
        } else if constexpr (t == InputTensor::DY) {
            return coutIdx;
        }
    }

    template <InputTensor t>
    __aicore__ inline uint32_t GetLen() const
    {
        if constexpr (t == InputTensor::FMAP) {
            return cinLength;
        } else if constexpr (t == InputTensor::DY) {
            return coutLength;
        }
    }

    __aicore__ inline bool NotEmpty() const { return coutLength != 0 && cinLength != 0; }
};

// AI Core 组内 AIV 数（3510 为 1 AIC + 2 AIV，其余架构 1），自 winograd conv_bp_wino_util 迁入
inline constexpr uint32_t __aicore__ AivNumInBlock()
{
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    return 2;
#else
    return 1;
#endif
}

// 当前 AIC 逻辑核号：AIC 侧即块号，AIV 侧按组内 AIV 数折算，自 winograd conv_bp_wino_util 迁入
inline uint32_t __aicore__ AicCoreId()
{
    if ASCEND_IS_AIC {
        return AscendC::GetBlockIdx();
    }
    if ASCEND_IS_AIV {
        return AscendC::GetBlockIdx() / AivNumInBlock();
    }
    return 0;
}

static inline uint32_t __aicore__ AivCoreId()
{
    // use it in aiv only
    return AscendC::GetBlockIdx();
}

static inline uint32_t __aicore__ AivNums() { return AscendC::GetBlockNum() * AivNumInBlock(); }

} // namespace BpUtils

#endif // CONV_BP_COMMON_UTIL_H
