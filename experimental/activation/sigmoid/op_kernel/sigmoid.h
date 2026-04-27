/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file sigmoid.h
 * \brief
 */
#ifndef SIGMOID_H
#define SIGMOID_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "sigmoid_tiling_data.h"
#include "sigmoid_tiling_key.h"

namespace MySigmoid {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
// FP32类型下，exp函数的最小输入值，用于裁剪
constexpr float NEG_LN_FP32_MAX = -89.0f;
// FP16类型下，exp函数的最小输入值，用于裁剪
constexpr float  NEG_LN_FP16_MAX = -12.0; // half不是标准数据类型，不支持constexpr, 这里使用float类型的最小值

#if defined(HIGH_PERFORMANCE) && HIGH_PERFORMANCE == 1
    constexpr bool is_high_perf = true;
#else
    constexpr bool is_high_perf = false;
#endif

template <uint32_t schMode>
class KernelSigmoid {
public:
    __aicore__ inline KernelSigmoid() {};

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, uint64_t smallCoreDataNum, uint64_t bigCoreDataNum, uint64_t finalBigTileNum,
        uint64_t finalSmallTileNum, uint64_t tileDataNum, uint64_t smallTailDataNum, uint64_t bigTailDataNum,
        uint64_t tailBlockNum);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void ComputeStandard();
    __aicore__ inline void ComputePoly();

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpCommon;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpForPloy1; 
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpForPloy2;
    // 专门用于类型转换的 buffer (bf16/fp16 -> float)
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpForCast;
    // 用于 需要类型转换 计算路径中的 1.0 (float)
    AscendC::LocalTensor<float> ones_float;
    // 用于 保持原类型 计算路径中的 1.0
    AscendC::LocalTensor<DTYPE_X> ones;

    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    uint64_t coreDataNum;
    uint64_t tileNum;
    uint64_t tileDataNum;
    uint64_t tailDataNum;
    uint64_t processDataNum;
};

template <uint32_t schMode>
__aicore__ inline void KernelSigmoid<schMode>::Init(
    GM_ADDR x, GM_ADDR y, uint64_t smallCoreDataNum, uint64_t bigCoreDataNum, uint64_t finalBigTileNum,
    uint64_t finalSmallTileNum, uint64_t tileDataNum, uint64_t smallTailDataNum, uint64_t bigTailDataNum,
    uint64_t tailBlockNum)
{
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");

    uint64_t coreId = AscendC::GetBlockIdx();
    uint64_t globalBufferIndex = bigCoreDataNum * coreId;
    this->tileDataNum = tileDataNum;

    if (coreId < tailBlockNum) {
        this->coreDataNum = bigCoreDataNum;
        this->tileNum = finalBigTileNum;
        this->tailDataNum = bigTailDataNum;
    } else {
        this->coreDataNum = smallCoreDataNum;
        this->tileNum = finalSmallTileNum;
        this->tailDataNum = smallTailDataNum;
        globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (coreId - tailBlockNum);
    }

    xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x + globalBufferIndex, this->coreDataNum);
    yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y + globalBufferIndex, this->coreDataNum);

    pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(DTYPE_X));
    pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(DTYPE_Y));

    // 分配临时Buffer逻辑
    #if __NPU_ARCH__ == 2002
    if constexpr (is_high_perf) { 
        // 310P 高性能模式 (Poly拟合)
        if constexpr (std::is_same_v<DTYPE_X, half>) { // half 类型不是基础数据类型，在Ascend C代码实现中应写为half
            pipe.InitBuffer(tmpForPloy1, tileDataNum * sizeof(half));
            pipe.InitBuffer(tmpForPloy2, tileDataNum * sizeof(half));
        } else {
            pipe.InitBuffer(tmpForPloy1, tileDataNum * sizeof(float)); 
            pipe.InitBuffer(tmpForPloy2, tileDataNum * sizeof(float));
        }
        // bf16需要额外分配用于cast的buffer
        if constexpr (std::is_same_v<DTYPE_X, bfloat16_t>) {
            pipe.InitBuffer(tmpForCast, tileDataNum * sizeof(float)); 
        }
    } else 
    #endif
    if constexpr (std::is_same_v<DTYPE_X, bfloat16_t> || 
        (!is_high_perf && std::is_same_v<DTYPE_X, half>)) { 
        // 这里的条件包含了 bf16，或者特定情况下的 fp16，都需要 cast 为 float 计算
        pipe.InitBuffer(tmpForCast, tileDataNum * sizeof(float));
        pipe.InitBuffer(tmpCommon, tileDataNum * sizeof(float));
        ones_float = tmpCommon.Get<float>();
        
        constexpr float one_val = 1.0f;
        AscendC::Duplicate(ones_float, one_val, tileDataNum); // 初始化浮点数 1.0, 用于最后取倒数
    } else { 
        // 直接使用原类型计算 (fp16 / fp32)
        pipe.InitBuffer(tmpCommon, tileDataNum * sizeof(DTYPE_X));
        ones = tmpCommon.Get<DTYPE_X>();
        
        DTYPE_X one_val = static_cast<DTYPE_X>(1.0f); // DTYPE_X 可能为half， half不是基础数据类型，不支持constexpr
        AscendC::Duplicate(ones, one_val, tileDataNum);
    }
}

template <uint32_t schMode>
__aicore__ inline void KernelSigmoid<schMode>::CopyIn(int32_t progress)
{
    AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
    AscendC::DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processDataNum);
    inQueueX.EnQue(xLocal);
}

template <uint32_t schMode>
__aicore__ inline void KernelSigmoid<schMode>::CopyOut(int32_t progress)
{
    AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
    AscendC::DataCopy(yGm[progress * this->tileDataNum], yLocal, this->processDataNum);
    outQueueY.FreeTensor(yLocal);
}

// 多项式拟合计算
template <uint32_t schMode>
__aicore__ inline void KernelSigmoid<schMode>::ComputePoly()
{
    AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
    AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();

    // 输入为 bfloat16_t，内部强制转为 float 计算
    if constexpr (std::is_same_v<DTYPE_X, bfloat16_t>) {
        // 获取 Cast 用的 buffer
        AscendC::LocalTensor<float> castTensor = tmpForCast.Get<float>();
        // 获取计算用的临时 Tensor (float)
        AscendC::LocalTensor<float> tmpTensor1 = tmpForPloy1.Get<float>();
        AscendC::LocalTensor<float> tmpTensor2 = tmpForPloy2.Get<float>();

        // 多项式系数 (直接定义为 float)
        constexpr float POLY_A = 0.229270815f;
        constexpr float POLY_B = -0.0102459298f;
        constexpr float POLY_C = 0.000207697530f;
        constexpr float POLY_D = 0.5f;
        constexpr float POLY_0 = 0.0f;
        constexpr float POLY_1 = 1.0f;

        // 1. Cast: bf16 -> float
        AscendC::Cast(castTensor, xLocal, AscendC::RoundMode::CAST_NONE, processDataNum);
        
        // 2. 计算流程 (全部使用 float 类型的 castTensor 和 tmpTensor)
        // x2 = x*x
        AscendC::Mul(tmpTensor1, castTensor, castTensor, processDataNum);
        // tmpTensor2 = x2
        AscendC::Adds(tmpTensor2, tmpTensor1, POLY_0, processDataNum);
        // cx2 = c*x2
        AscendC::Muls(tmpTensor1, tmpTensor1, POLY_C, processDataNum);
        // b + cx2
        AscendC::Adds(tmpTensor1, tmpTensor1, POLY_B, processDataNum);
        // x2*(b+cx2)
        AscendC::Mul(tmpTensor2, tmpTensor1, tmpTensor2, processDataNum);
        // a + x2*(b+cx2)
        AscendC::Adds(tmpTensor2, tmpTensor2, POLY_A, processDataNum);
        // x*(a+...)
        AscendC::Mul(tmpTensor2, castTensor, tmpTensor2, processDataNum);
        // +d
        AscendC::Adds(tmpTensor2, tmpTensor2, POLY_D, processDataNum);
        // clamp [0,1]
        AscendC::Mins(tmpTensor2, tmpTensor2, POLY_1, processDataNum);
        AscendC::Maxs(tmpTensor2, tmpTensor2, POLY_0, processDataNum);

        // 3. 结果 Cast 回 bf16
        AscendC::Cast(yLocal, tmpTensor2, AscendC::RoundMode::CAST_ROUND, processDataNum);
    } else { // 输入为 float 或 half，直接使用原始类型 DTYPE_X 计算
        AscendC::LocalTensor<DTYPE_X> tmpTensor1 = tmpForPloy1.Get<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> tmpTensor2 = tmpForPloy2.Get<DTYPE_X>();

        // 多项式系数 (转为 DTYPE_X)
        DTYPE_X POLY_A = static_cast<DTYPE_X>(0.229270815f); // DTYPE_X可能为half，half不支持constexpr
        DTYPE_X POLY_B = static_cast<DTYPE_X>(-0.0102459298f);
        DTYPE_X POLY_C = static_cast<DTYPE_X>(0.000207697530f);
        DTYPE_X POLY_D = static_cast<DTYPE_X>(0.5f);
        DTYPE_X POLY_0 = static_cast<DTYPE_X>(0.0f);
        DTYPE_X POLY_1 = static_cast<DTYPE_X>(1.0f);

        // 计算流程 (直接使用 xLocal)
        // x2 = x*x
        AscendC::Mul(tmpTensor1, xLocal, xLocal, processDataNum);
        // tmpTensor2 = x2
        AscendC::Adds(tmpTensor2, tmpTensor1, POLY_0, processDataNum);
        // cx2 = c*x2
        AscendC::Muls(tmpTensor1, tmpTensor1, POLY_C, processDataNum);
        // b + cx2
        AscendC::Adds(tmpTensor1, tmpTensor1, POLY_B, processDataNum);
        // x2*(b+cx2)
        AscendC::Mul(tmpTensor2, tmpTensor1, tmpTensor2, processDataNum);
        // a + x2*(b+cx2)
        AscendC::Adds(tmpTensor2, tmpTensor2, POLY_A, processDataNum);
        // x*(a+...)
        AscendC::Mul(tmpTensor2, xLocal, tmpTensor2, processDataNum);
        // +d
        AscendC::Adds(tmpTensor2, tmpTensor2, POLY_D, processDataNum);
        // clamp [0,1]
        AscendC::Mins(tmpTensor2, tmpTensor2, POLY_1, processDataNum);
        AscendC::Maxs(yLocal, tmpTensor2, POLY_0, processDataNum);
    }

    outQueueY.EnQue<DTYPE_Y>(yLocal);
    inQueueX.FreeTensor(xLocal);
}

// 标准 Sigmoid 计算 (1 / (1 + exp(-x)))
template <uint32_t schMode>
__aicore__ inline void KernelSigmoid<schMode>::ComputeStandard()
{
    AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
    AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();

    // 处理 需要类型转换 情况
    if constexpr (std::is_same_v<DTYPE_X, bfloat16_t> || 
        (!is_high_perf && std::is_same_v<DTYPE_X, half>)) {
        // 获取 float 类型的临时 buffer
        AscendC::LocalTensor<float> computeTensor = tmpForCast.Get<float>();
        
        AscendC::Cast(computeTensor, xLocal, AscendC::RoundMode::CAST_NONE, processDataNum);

        // 910 平台的 Clip 逻辑 (在 float 下做)
        #if __NPU_ARCH__ == 1001
        AscendC::Maxs(computeTensor, computeTensor, NEG_LN_FP32_MAX, processDataNum);
        #endif

        // 计算 Sigmoid (float)
        // x = -x
        AscendC::Muls(computeTensor, computeTensor, -1.0f, processDataNum);
        // x = exp(x)
        AscendC::Exp(computeTensor, computeTensor, processDataNum);
        // x = x + 1
        AscendC::Adds(computeTensor, computeTensor, 1.0f, processDataNum);
        // y = 1 / x (使用 ones_float)
        AscendC::Div(computeTensor, ones_float, computeTensor, processDataNum);

        AscendC::Cast(yLocal, computeTensor, AscendC::RoundMode::CAST_ROUND, processDataNum);

    } else {
        // 类型不变
        DTYPE_X n_one_val = static_cast<DTYPE_X>(-1.0f);
        DTYPE_X one_val = static_cast<DTYPE_X>(1.0f);
        #if __NPU_ARCH__ == 1001
        DTYPE_X ln_res_val = static_cast<DTYPE_X>(NEG_LN_FP16_MAX);
        if constexpr (std::is_same_v<DTYPE_X, float>) {
            ln_res_val = static_cast<DTYPE_X>(NEG_LN_FP32_MAX);
        }
        AscendC::Maxs(xLocal, xLocal, ln_res_val, processDataNum);
        #endif

        AscendC::Muls(xLocal, xLocal, n_one_val, processDataNum);
        AscendC::Exp(xLocal, xLocal, processDataNum);
        AscendC::Adds(xLocal, xLocal, one_val, processDataNum);
        AscendC::Div(yLocal, ones, xLocal, processDataNum);
    }

    outQueueY.EnQue<DTYPE_Y>(yLocal);
    inQueueX.FreeTensor(xLocal);
}

template <uint32_t schMode>
__aicore__ inline void KernelSigmoid<schMode>::Process()
{
    int32_t loopCount = this->tileNum;
    this->processDataNum = this->tileDataNum;
    
    for (int32_t i = 0; i < loopCount - 1; i++) {
        CopyIn(i);
        #if __NPU_ARCH__ == 2002
        if constexpr (is_high_perf) { 
            ComputePoly();
        } else 
        #endif
        {
            ComputeStandard();
        }
        CopyOut(i);
    }
    
    this->processDataNum = this->tailDataNum;
    CopyIn(loopCount - 1);
    #if __NPU_ARCH__ == 2002
    if constexpr (is_high_perf) { 
        ComputePoly();
    } else 
    #endif
    {
        ComputeStandard();
    }
    CopyOut(loopCount - 1);
}

} // namespace MySigmoid
#endif // SIGMOID_H