/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file weight_quant_batch_matmul_v2_adaptive_sliding_window.h
 * \brief
 */

#ifndef WEIGHT_QUANT_BATCH_MATMUL_V2_ADAPTIVE_SLIDING_WINDOW_H
#define WEIGHT_QUANT_BATCH_MATMUL_V2_ADAPTIVE_SLIDING_WINDOW_H

#include "weight_quant_batch_matmul_v2_asw_block.h"
#include "../weight_quant_batch_matmul_v2_constant.h"
#include "weight_quant_batch_matmul_v2_arch35_tiling_data.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "lib/matmul_intf.h"
#include "../tool.h"

#define LOCAL_TEMPLATE_CLASS_PARAMS                                                                        \
    template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans, \
              QuantType antiQuantType, bool hasAntiQuantOffset, QuantType quantType>
#define LOCAL_TEMPLATE_FUNC_PARAMS \
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType

using AscendC::AIC;
using AscendC::AIV;
using AscendC::BinaryRepeatParams;
using AscendC::DataCopyPadParams;
using AscendC::DataCopyParams;
using AscendC::GetBlockIdx;
using AscendC::GlobalTensor;
using AscendC::int4b_t;
using AscendC::LocalTensor;
using AscendC::ONE_BLK_SIZE;
using AscendC::QuePosition;
using AscendC::RoundMode;
using AscendC::SyncAll;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::TPosition;
using AscendC::TQue;
using matmul::MatmulImpl;
using matmul::MatmulType;
namespace WeightQuantBatchMatmulV2::Arch35 {

constexpr MatmulConfig MM_CFG_VEC_ND2NZ = GetMDLConfig(false, false, 0, true);
constexpr MatmulConfig MM_CFG_NO_PRELOAD = GetMDLConfig(false, false, 0, false, false, false, false);
constexpr MatmulConfig MM_CFG_NO_PRELOAD_OPEN_UNIT_FLAG = GetMDLConfig(false, false, 0, false, false, false, true);
constexpr MatmulConfig MM_CFG_PRELOAD = GetMDLConfig(false, false, 1, false, false, false, false);

LOCAL_TEMPLATE_CLASS_PARAMS
class WeightQuantBatchMatmulV2ASWKernel {
public:
    __aicore__ inline WeightQuantBatchMatmulV2ASWKernel()
    {
    }
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset,
                                GM_ADDR quantScale, GM_ADDR quantOffset, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace,
                                const void *tilingData, TPipe *tPipe);
    __aicore__ inline void UpdateGlobalAddr(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset,
                                            GM_ADDR quantScale, GM_ADDR quantOffset, GM_ADDR bias, GM_ADDR y,
                                            GM_ADDR workspace);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void ComputeBasicOptiLoop();
    __aicore__ inline void ComputeBmmOptiLoop();
    __aicore__ inline void SetMMParaAndCompute();

    uint32_t blockIdx_;
    const wqbmmv2_tiling::WeightQuantBatchMatmulV2ASWTilingDataParams *tiling_;

    GlobalTensor<xType> aGlobal_;
    GlobalTensor<wType> bGlobal_;
    GlobalTensor<yType> cGlobal_;
    GlobalTensor<biasType> biasGlobal_;
    WeightQuantBmmAswBlock block_;
    GlobalTensor<uint64_t> scaleGlobal_;
    using aType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, xType, aTrans>;
    using bType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, wType, bTrans>;
    using cType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, yType>;
    using biasMatmulType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, biasType>;
    matmul::MatmulImpl<aType, bType, cType, biasMatmulType, MM_CFG_NO_PRELOAD_OPEN_UNIT_FLAG> mm_;
};

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantBatchMatmulV2ASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::Init(
    GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset, GM_ADDR quantScale, GM_ADDR quantOffset,
    GM_ADDR bias, GM_ADDR y, GM_ADDR workspace, const void *tilingData, TPipe *tPipe)
{
    if ASCEND_IS_AIV {
        return;
    }
    tiling_ = static_cast<const wqbmmv2_tiling::WeightQuantBatchMatmulV2ASWTilingDataParams*>(tilingData);
    blockIdx_ = GetBlockIdx();
    UpdateGlobalAddr(x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, workspace);
    mm_.SetSubBlockIdx(0);
    mm_.Init(&tiling_->matmulTiling, tPipe);
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantBatchMatmulV2ASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::UpdateGlobalAddr(
    GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset, GM_ADDR quantScale, GM_ADDR quantOffset,
    GM_ADDR bias, GM_ADDR y, GM_ADDR workspace)
{
    block_.Init(tiling_, blockIdx_);

    if constexpr (antiQuantType == QuantType::PER_TENSOR) {  // pertensor
        block_.offset_.scaleScalar = *((__gm__ uint64_t *)antiquantScale);
    } else {
        scaleGlobal_.SetGlobalBuffer((__gm__ uint64_t *)antiquantScale);
    }

    // update global buffer
    aGlobal_.SetGlobalBuffer((__gm__ xType *)x);
    bGlobal_.SetGlobalBuffer((__gm__ wType *)weight);
    cGlobal_.SetGlobalBuffer((__gm__ yType *)y);
    if (static_cast<bool>(tiling_->matmulTiling.isBias)) {
        biasGlobal_.SetGlobalBuffer((__gm__ biasType*)bias);
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantBatchMatmulV2ASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::Process()
{
    if ASCEND_IS_AIV {
        return;
    }
    if (tiling_->params.batchC == 1UL) {
        block_.offset_.batchCOffset = 0;
        block_.offset_.batchAOffset = 0;
        block_.offset_.batchBOffset = 0;
        ComputeBasicOptiLoop();
    } else {
        ComputeBmmOptiLoop();
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantBatchMatmulV2ASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::ComputeBasicOptiLoop()
{
    for (uint64_t roundIndex = 0; roundIndex < block_.params_.round; ++roundIndex) {
        block_.UpdateBasicIndex(roundIndex);
        // 1. Set single core param
        block_.UpdateBlockParams(roundIndex);
        if (block_.params_.singleCoreM <= 0 || block_.params_.singleCoreN <= 0) {
            continue;
        }
        mm_.SetSingleShape(block_.params_.singleCoreM, block_.params_.singleCoreN, tiling_->matmulTiling.Ka);
        // 2. compute offset
        block_.CalcGMOffset<aTrans, bTrans, CubeFormat::ND>();
        // 3. set offset and compute
        SetMMParaAndCompute();
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantBatchMatmulV2ASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::ComputeBmmOptiLoop()
{
    uint64_t batchC3C4 = static_cast<uint64_t>(tiling_->params.batchC3) * tiling_->params.batchC4;
    uint64_t batchC2C3C4 = tiling_->params.batchC2 * batchC3C4;
    uint64_t batchB3B4 = static_cast<uint64_t>(tiling_->params.batchB3) * tiling_->params.batchB4;
    uint64_t batchB2B3B4 = tiling_->params.batchB2 * batchB3B4;
    uint64_t batchA3A4 = static_cast<uint64_t>(tiling_->params.batchA3) * tiling_->params.batchA4;
    uint64_t batchA2A3A4 = tiling_->params.batchA2 * batchA3A4;
    uint32_t multiA1C1 = tiling_->params.batchA1 / tiling_->params.batchC1;
    uint32_t multiA2C2 = tiling_->params.batchA2 / tiling_->params.batchC2;
    uint32_t multiA3C3 = tiling_->params.batchA3 / tiling_->params.batchC3;
    uint32_t multiA4C4 = tiling_->params.batchA4 / tiling_->params.batchC4;
    uint32_t multiB1C1 = tiling_->params.batchB1 / tiling_->params.batchC1;
    uint32_t multiB2C2 = tiling_->params.batchB2 / tiling_->params.batchC2;
    uint32_t multiB3C3 = tiling_->params.batchB3 / tiling_->params.batchC3;
    uint32_t multiB4C4 = tiling_->params.batchB4 / tiling_->params.batchC4;
    for (uint64_t b1Index = 0; b1Index < tiling_->params.batchC1; ++b1Index) {
        uint64_t batchC1Offset = b1Index * batchC2C3C4;
        uint64_t batchA1Offset = b1Index * batchA2A3A4 * multiA1C1;
        uint64_t batchB1Offset = b1Index * batchB2B3B4 * multiB1C1;
        for (uint64_t b2Index = 0; b2Index < tiling_->params.batchC2; ++b2Index) {
            uint64_t batchC2Offset = b2Index * batchC3C4 + batchC1Offset;
            uint64_t batchA2Offset = b2Index * batchA3A4 * multiA2C2 + batchA1Offset;
            uint64_t batchB2Offset = b2Index * batchB3B4 * multiB2C2 + batchB1Offset;
            for (uint64_t b3Index = 0; b3Index < tiling_->params.batchC3; ++b3Index) {
                uint64_t batchC3Offset = b3Index * tiling_->params.batchC4 + batchC2Offset;
                uint64_t batchA3Offset = b3Index * tiling_->params.batchA4 * multiA3C3 + batchA2Offset;
                uint64_t batchB3Offset = b3Index * tiling_->params.batchB4 * multiB3C3 + batchB2Offset;
                for (uint64_t b4Index = 0; b4Index < tiling_->params.batchC4; ++b4Index) {
                    block_.offset_.batchCOffset = batchC3Offset + b4Index;
                    block_.offset_.batchAOffset = batchA3Offset + b4Index * multiA4C4;
                    block_.offset_.batchBOffset = batchB3Offset + b4Index * multiB4C4;
                    block_.ResetAddressOffsets();
                    ComputeBasicOptiLoop();
                }
            }
        }
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantBatchMatmulV2ASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::SetMMParaAndCompute()
{
    if constexpr (antiQuantType == QuantType::PER_TENSOR) {  // pertensor
        mm_.SetQuantScalar(block_.offset_.scaleScalar);
    } else {
        mm_.SetQuantVector(scaleGlobal_[block_.offset_.offsetScale]);
    }

    if (tiling_->matmulTiling.isBias) {
        mm_.SetBias(biasGlobal_[block_.offset_.offsetBias]);
    }
    mm_.SetTensorA(aGlobal_[block_.offset_.offsetA], aTrans);
    mm_.SetTensorB(bGlobal_[block_.offset_.offsetB], bTrans);
    mm_.Iterate();
    mm_.GetTensorC(cGlobal_[block_.offset_.offsetC]);
}
}  // namespace WeightQuantBatchMatmulV2::Arch35
#endif  // WEIGHT_QUANT_BATCH_MATMUL_V2_ADAPTIVE_SLIDING_WINDOW_H