/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file qbmm_cube_on_the_fly_iterbatch.h
 * \brief
 */

#ifndef QBMM_CUBE_ON_THE_FLY_ITERBATCH_H
#define QBMM_CUBE_ON_THE_FLY_ITERBATCH_H

#include "../quant_batch_matmul_v3_base.h"
#include "qbmm_iterbatch_block.h"

#define LOCAL_TEMPLATE_CLASS_PARAMS_V2                                                                            \
    template <class x1Type, class x2Type, class inputScaleType, class biasType, class yType, CubeFormat formatX1, \
              CubeFormat formatX2, CubeFormat formatY, bool aTrans, bool bTrans, const MatmulConfig &mmCfg,       \
              FusedOpType fusedOpType>
#define LOCAL_TEMPLATE_FUNC_PARAMS_V2 \
    x1Type, x2Type, inputScaleType, biasType, yType, formatX1, formatX2, formatY, aTrans, bTrans, mmCfg, fusedOpType
namespace QuantBatchMatmulV3 {

using namespace AscendC;
using namespace matmul;

namespace IterBatch {
template <const MatmulConfig &mmCfg, FusedOpType fusedOpType>
constexpr auto cfg_v = [] {
    auto cfg = mmCfg;
    if constexpr (fusedOpType == FusedOpType::RELU) {
        cfg.enableRelu = true;
    }
    return cfg;
}();
}

template <class x1Type, class x2Type, class inputScaleType, class biasType, class yType, CubeFormat formatX1,
          CubeFormat formatX2, CubeFormat formatY, bool aTrans, bool bTrans, const MatmulConfig &mmCfg,
          FusedOpType fusedOpType = FusedOpType::NONE>
class QbmmIterBatchKernel {
public:
    __aicore__ inline QbmmIterBatchKernel()
    {
    }
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR bias, GM_ADDR scale, GM_ADDR perTokenScale,
                                GM_ADDR cGM, GM_ADDR workSpace, const void *tilingData, TPipe *que);
    __aicore__ inline void UpdateGlobalAddr(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR bias, GM_ADDR scale,
                                            GM_ADDR perTokenScale, GM_ADDR cGM, GM_ADDR workSpace);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void CalcMmWithBatch();

    uint32_t blockIdx_;
    const DequantBmm::QuantBatchMatmulV3TilingDataParams *quantBmmTilingData_;

    GlobalTensor<x1Type> aGlobal_;
    GlobalTensor<x2Type> bGlobal_;
    GlobalTensor<yType> cGlobal_;
    GlobalTensor<biasType> biasGlobal_;
    QuantBatchMatmulV3::QbmmMultiBatchBaseBlock block_;

    GlobalTensor<uint64_t> scaleGlobal_;
    using scaleType =
        typename AscendC::Conditional<IsSameType<inputScaleType, int64_t>::value, uint64_t, inputScaleType>::type;
    using aType = matmul::MatmulType<AscendC::TPosition::GM, formatX1, x1Type, aTrans, LayoutMode::NORMAL>;
    using bType = matmul::MatmulType<AscendC::TPosition::GM, formatX2, x2Type, bTrans, LayoutMode::NORMAL>;
    using cType = matmul::MatmulType<AscendC::TPosition::GM, formatY, yType, false, LayoutMode::NORMAL>;
    using biasMatmulType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, biasType>;
    matmul::MatmulImpl<aType, bType, cType, biasMatmulType, IterBatch::cfg_v<mmCfg, fusedOpType>> mm_;
};

LOCAL_TEMPLATE_CLASS_PARAMS_V2
__aicore__ inline void QbmmIterBatchKernel<LOCAL_TEMPLATE_FUNC_PARAMS_V2>::Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR bias,
                                                                                GM_ADDR scale, GM_ADDR perTokenScale,
                                                                                GM_ADDR cGM, GM_ADDR workSpace,
                                                                                const void *tilingData, TPipe *que)
{
    if ASCEND_IS_AIV {
        return;
    }
    quantBmmTilingData_ = static_cast<const DequantBmm::QuantBatchMatmulV3TilingDataParams *>(tilingData);
    bool isWeightNz = formatX2 == CubeFormat::NZ;
    block_.Init(quantBmmTilingData_, bTrans, isWeightNz);
    blockIdx_ = GetBlockIdx();
    UpdateGlobalAddr(aGM, bGM, bias, scale, perTokenScale, cGM, workSpace);
    mm_.SetSubBlockIdx(0);
    mm_.Init(&block_.tilingData_->matmulTiling, que);
}

LOCAL_TEMPLATE_CLASS_PARAMS_V2
__aicore__ inline void QbmmIterBatchKernel<LOCAL_TEMPLATE_FUNC_PARAMS_V2>::UpdateGlobalAddr(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR bias, GM_ADDR scale, GM_ADDR perTokenScale, GM_ADDR cGM, GM_ADDR workSpace)
{
    if (static_cast<bool>(block_.tilingData_->params.isPerTensor)) {  // pertensor
        if constexpr (!IsSameType<scaleType, uint64_t>::value) {
            uint32_t uint32Scale = 0;
            if constexpr (IsSameType<scaleType, bfloat16_t>::value) {
                uint16_t uint16Scale = *((__gm__ uint16_t *)scale);
                uint32Scale = uint16Scale << 16;
            } else {
                uint32Scale = *((__gm__ uint32_t *)scale);
            }
            block_.offset_.scaleScalar = uint32Scale & DEQ_SCALE_MUL;
        } else {
            block_.offset_.scaleScalar = *((__gm__ uint64_t *)scale);
        }
    } else {
        scaleGlobal_.SetGlobalBuffer((__gm__ uint64_t *)scale);
    }
    // update global buffer
    aGlobal_.SetGlobalBuffer((__gm__ x1Type *)aGM);
    bGlobal_.SetGlobalBuffer((__gm__ x2Type *)bGM);
    if (static_cast<bool>(block_.tilingData_->matmulTiling.isBias)) {
        biasGlobal_.SetGlobalBuffer((__gm__ biasType *)bias);
    }
    cGlobal_.SetGlobalBuffer((__gm__ yType *)cGM);
}

LOCAL_TEMPLATE_CLASS_PARAMS_V2
__aicore__ inline void QbmmIterBatchKernel<LOCAL_TEMPLATE_FUNC_PARAMS_V2>::CalcMmWithBatch()
{
    for (uint64_t loopIndex = 0; loopIndex < block_.params_.loopTimes; loopIndex++) {
        block_.GetMultiBatchInfo(loopIndex, blockIdx_);
        if (block_.params_.batchANum > 0) {
            block_.CalcGMOffset();
            if (static_cast<bool>(block_.tilingData_->params.isPerTensor)) {
                mm_.SetQuantScalar(block_.offset_.scaleScalar);
            } else {
                mm_.SetQuantVector(scaleGlobal_[0]);
            }
            if (block_.tilingData_->matmulTiling.isBias) {
                mm_.SetBias(biasGlobal_[block_.offset_.offsetBias]);
            }
            mm_.SetTensorA(aGlobal_[block_.offset_.offsetA], aTrans);
            mm_.SetTensorB(bGlobal_[block_.offset_.offsetB], bTrans);
            mm_.SetNBatchOutNum(block_.params_.nBatchOutNum);
            mm_.SetBatchNum(block_.params_.batchANum, block_.params_.batchBNum);
            mm_.IterateBatch(cGlobal_[block_.offset_.offsetC], false, 0, false, block_.params_.singleASize,
                             block_.params_.singleBSize);
        }
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS_V2
__aicore__ inline void QbmmIterBatchKernel<LOCAL_TEMPLATE_FUNC_PARAMS_V2>::Process()
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
                    block_.GetCalcBatchOffset(b1Index, b2Index, b3Index);
                    CalcMmWithBatch();
                }
            }
        }
    }
}
}  // namespace QuantBatchMatmulV3
#endif  // QBMM_CUBE_ON_THE_FLY_ITERBATCH_H