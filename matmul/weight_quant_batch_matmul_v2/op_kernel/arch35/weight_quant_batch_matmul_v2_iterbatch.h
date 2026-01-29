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
 * \file weight_quant_batch_matmul_v2_iterbatch.h
 * \brief
 */

#ifndef WEIGHT_QUANT_BATCH_MATMUL_V2_ITERBATCH_H
#define WEIGHT_QUANT_BATCH_MATMUL_V2_ITERBATCH_H

#include "kernel_basic_intf.h"
#include "lib/matmul_intf.h"
#include "../tool.h"
#include "weight_quant_batch_matmul_v2_iterbatch_block.h"
#include "../weight_quant_batch_matmul_v2_constant.h"
#include "weight_quant_batch_matmul_v2_arch35_tiling_data.h"

#define LOCAL_TEMPLATE_CLASS_PARAMS                                                                        \
    template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans, \
              QuantType antiQuantType, bool hasAntiQuantOffset, QuantType quantType>
#define LOCAL_TEMPLATE_FUNC_PARAMS \
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType

using AscendC::GetBlockIdx;
using AscendC::GlobalTensor;
using AscendC::TPipe;
using AscendC::TPosition;
using matmul::MatmulImpl;
using matmul::MatmulType;
namespace WeightQuantBatchMatmulV2::Arch35 {
constexpr MatmulConfigMode configMode = MatmulConfigMode::CONFIG_NORM;
constexpr MatmulBatchParams batchParams{false, BatchMode::BATCH_LESS_THAN_L1, false, BatchOutMode::MULTI_BATCH};
constexpr MatmulConfig MM_CFG_MULTI_BATCH = GetMMConfig<configMode>(batchParams);
constexpr MatmulBatchParams batchParamsNoBatchOut{false, BatchMode::BATCH_LESS_THAN_L1, false, BatchOutMode::SINGLE_BATCH};
constexpr MatmulConfig MM_CFG_MULTI_BATCH_NO_BATCH_OUT = GetMMConfig<configMode>(batchParamsNoBatchOut);

LOCAL_TEMPLATE_CLASS_PARAMS
class WeightQuantBatchMatmulV2IterBatchKernel {
public:
    __aicore__ inline WeightQuantBatchMatmulV2IterBatchKernel()
    {
    }
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset,
                                GM_ADDR quantScale, GM_ADDR quantOffset, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace,
                                const void* tilingData, TPipe* tPipe);
    __aicore__ inline void UpdateGlobalAddr(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset,
                                            GM_ADDR quantScale, GM_ADDR quantOffset, GM_ADDR bias, GM_ADDR y,
                                            GM_ADDR workspace);
    __aicore__ inline void Process();
protected:
    __aicore__ inline void CalcMmWithBatch();
    uint32_t blockIdx_;
    const wqbmmv2_tiling::WeightQuantBatchMatmulV2ASWTilingDataParams* tiling_;

    GlobalTensor<xType> aGlobal_;
    GlobalTensor<wType> bGlobal_;
    GlobalTensor<yType> cGlobal_;
    GlobalTensor<biasType> biasGlobal_;
    WeightQuantBmmIterBatchBlock block_;
    GlobalTensor<uint64_t> scaleGlobal_;
    using aType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, xType, aTrans, LayoutMode::NORMAL>;
    using bType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, wType, bTrans, LayoutMode::NORMAL>;
    using cType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, yType, false, LayoutMode::NORMAL>;
    using biasMatmulType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, biasType>;
    matmul::MatmulImpl<aType, bType, cType, biasMatmulType, MM_CFG_MULTI_BATCH_NO_BATCH_OUT> mm_;
};

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantBatchMatmulV2IterBatchKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::Init(
    GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset, GM_ADDR quantScale, GM_ADDR quantOffset,
    GM_ADDR bias, GM_ADDR y, GM_ADDR workspace, const void* tilingData, TPipe* tPipe)
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
__aicore__ inline void WeightQuantBatchMatmulV2IterBatchKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::UpdateGlobalAddr(
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
__aicore__ inline void WeightQuantBatchMatmulV2IterBatchKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::Process()
{
    if ASCEND_IS_AIV {
        return;
    }
    if (blockIdx_ >= block_.tilingData_->matmulTiling.usedCoreNum) {
        return;
    }
    if (block_.params_.innerBatchNum == 0) {
        block_.offset_.batchCOffset = 0;
        CalcMmWithBatch();
    } else {
        for (uint64_t b1Index = 0; b1Index < block_.params_.iterBatch1; ++b1Index) {
            for (uint64_t b2Index = 0; b2Index < block_.params_.iterBatch2; ++b2Index) {
                for (uint64_t b3Index = 0; b3Index < block_.params_.iterBatch3; ++b3Index) {
                    block_.GetBroadcastBatchOffset(b1Index, b2Index, b3Index);
                    CalcMmWithBatch();
                }
            }
        }
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantBatchMatmulV2IterBatchKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::CalcMmWithBatch()
{
    for (uint64_t loopIndex = 0; loopIndex < block_.params_.loopTimes; loopIndex++) {
        if constexpr(antiQuantType == QuantType::PER_TENSOR) {
            mm_.SetQuantScalar(block_.offset_.scaleScalar);
        } else {
            mm_.SetQuantVector(scaleGlobal_[block_.offset_.offsetScale]);
        }
        block_.GetMultiBatchInfo(loopIndex);
        block_.CalcGMOffset();
        if (block_.tilingData_->matmulTiling.isBias) {
            mm_.SetBias(biasGlobal_[block_.offset_.offsetBias]);
        }
        mm_.SetTensorA(aGlobal_[block_.offset_.offsetA], aTrans);
        mm_.SetTensorB(bGlobal_[block_.offset_.offsetB], bTrans);
        mm_.SetBatchNum(block_.params_.batchANum, block_.params_.batchBNum);
        mm_.IterateBatch(cGlobal_[block_.offset_.offsetC], false, 0, false, block_.params_.singleASize,
                            block_.params_.singleBSize);
    }
}
}  // namespace WeightQuantBatchMatmulV2::Arch35
#endif  // WEIGHT_QUANT_BATCH_MATMUL_V2_ITERBATCH_H