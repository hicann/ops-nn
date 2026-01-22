/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file qbmm_cube_on_the_fly_bl1_full_load.h
 * \brief
 */
#ifndef QBMM_CUBE_ON_THE_FLY_BL1_FULL_LOAD_H
#define QBMM_CUBE_ON_THE_FLY_BL1_FULL_LOAD_H

#include "../quant_batch_matmul_v3_base.h"
#include "qbmm_api_utils.h"
#include "qbmm_asw_block.h"
#include "qbmm_cube_on_the_fly.h"


namespace QuantBatchMatmulV3 {

using namespace AscendC;
using namespace matmul;

namespace BL1FullLoad {
template <FusedOpType fusedOpType>
constexpr auto cfg_v = [] {
    auto cfg = MM_CFG_NO_PRELOAD_OPEN_UNIT_FLAG;
    if constexpr (fusedOpType == FusedOpType::RELU) {
        cfg.enableRelu = true;
    }
    return cfg;
}();
}

template <class x1Type, class x2Type, class inputScaleType, class biasType, class yType, CubeFormat formatX1,
          CubeFormat formatX2, CubeFormat formatY, bool aTrans, bool bTrans, bool isLut = false,
          FusedOpType fusedOpType = FusedOpType::NONE>
class MatmulAswKernelBL1FullLoad : public MatMulASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS> {
public:
    __aicore__ inline MatmulAswKernelBL1FullLoad()
    {
    }
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR bias, GM_ADDR scale, GM_ADDR perTokenScale,
                                GM_ADDR cGM, GM_ADDR workspace, const void *tilingData, TPipe *pipe);
    __aicore__ inline void SetScaleTensor();
    __aicore__ inline void Process();
    __aicore__ inline void ProcessWithoutBatch();

protected:
    using scaleType =
        typename AscendC::Conditional<IsSameType<inputScaleType, int64_t>::value, uint64_t, inputScaleType>::type;
    using aType = matmul::MatmulType<AscendC::TPosition::GM, formatX1, x1Type, aTrans>;
    using bType = matmul::MatmulType<AscendC::TPosition::TSCM, formatX2, x2Type, bTrans>;
    using cType = matmul::MatmulType<AscendC::TPosition::GM, formatY, yType>;
    using biasMatmulType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, biasType>;
    matmul::MatmulImpl<aType, bType, cType, biasMatmulType, BL1FullLoad::cfg_v<fusedOpType>> mm_;
    TQue<QuePosition::B1, 1> InQueueBL1_;
    TPipe *pipe_;
    LocalTensor<x2Type> bl1Local_;
    bool isNMultiCore_;
};

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void MatmulAswKernelBL1FullLoad<LOCAL_TEMPLATE_FUNC_PARAMS>::Init(GM_ADDR aGM, GM_ADDR bGM,
                                                                                    GM_ADDR bias, GM_ADDR scale,
                                                                                    GM_ADDR perTokenScale, GM_ADDR cGM,
                                                                                    GM_ADDR workSpace,
                                                                                    const void *tilingData, TPipe *pipe)
{
    if ASCEND_IS_AIV {
        return;
    }
    pipe_ = pipe;
    this->blockIdx_ = GetBlockIdx();
    // 在kernel中定义tilingData结构体
    this->quantBmmTilingData_ = static_cast<const DequantBmm::QuantBatchMatmulV3TilingDataParams *>(tilingData);
    this->UpdateGlobalAddr(aGM, bGM, bias, scale, perTokenScale, nullptr, cGM, workSpace);
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void MatmulAswKernelBL1FullLoad<LOCAL_TEMPLATE_FUNC_PARAMS>::Process()
{
    if ASCEND_IS_AIV {
        return;
    }
    isNMultiCore_ = this->block_.tilingData_->matmulTiling.singleCoreN < this->block_.tilingData_->matmulTiling.N;
    // 临时代码，bl1支持s8s8
    uint64_t innerX2AlignedBlock = DequantBmm::GetC0Size<x2Type>();
    uint64_t kbAligned = 0;
    uint64_t nAligned = 0;
    if constexpr (bTrans) {
        kbAligned = DequantBmm::Align(this->block_.tilingData_->matmulTiling.Ka, innerX2AlignedBlock);
        nAligned = DequantBmm::Align(this->block_.tilingData_->matmulTiling.singleCoreN, BMM_BLOCK_NUM);
    } else {
        kbAligned = DequantBmm::Align(this->block_.tilingData_->matmulTiling.Ka, BMM_BLOCK_NUM);
        nAligned =
            DequantBmm::Align(this->block_.tilingData_->matmulTiling.singleCoreN, innerX2AlignedBlock);
    }
    pipe_->InitBuffer(InQueueBL1_, 1, DequantBmm::GetSizeWithDataType<x2Type>(kbAligned * nAligned));

    // 尽可能将mm api init提前做 让BL1的MTE2和MM的初始化并行起来
    mm_.SetSubBlockIdx(0);
    mm_.Init(&this->block_.tilingData_->matmulTiling, pipe_);

    bl1Local_ = InQueueBL1_.AllocTensor<x2Type>();
    this->block_.UpdateBasicIndex4BL1FullLoad(0);
    if constexpr (formatX2 == CubeFormat::ND) {
        CopyInNDB1<x2Type, bTrans>(this->block_, this->block_.params_.nIndex, isNMultiCore_, bl1Local_, this->bGlobal_);
    } else {
        CopyInNZB1<x2Type, bTrans>(this->block_, this->block_.params_.nIndex, isNMultiCore_, bl1Local_, this->bGlobal_,
                                   innerX2AlignedBlock);
    }

    this->block_.offset_.batchCOffset = 0;
    this->block_.offset_.batchAOffset = 0;
    this->block_.offset_.batchBOffset = 0;
    ProcessWithoutBatch();
    InQueueBL1_.FreeTensor(bl1Local_);
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void MatmulAswKernelBL1FullLoad<LOCAL_TEMPLATE_FUNC_PARAMS>::SetScaleTensor()
{
    if (static_cast<bool>(this->block_.tilingData_->params.isPerTensor) ||
        static_cast<bool>(this->block_.tilingData_->params.isDoubleScale)) {
        mm_.SetQuantScalar(this->block_.offset_.scaleScalar);
    } else {
        mm_.SetQuantVector(this->scaleGlobal_[this->block_.offset_.offsetScale]);
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void MatmulAswKernelBL1FullLoad<LOCAL_TEMPLATE_FUNC_PARAMS>::ProcessWithoutBatch()
{
    for (uint64_t roundIdx = 0; roundIdx < this->block_.params_.round; roundIdx++) {
        if (roundIdx != 0) {
            this->block_.UpdateBasicIndex4BL1FullLoad(roundIdx);
        }
        if (this->block_.params_.index < this->block_.params_.totalCnt) {
            this->block_.template UpdateBlockParams4BL1FullLoad<bTrans, formatX2>(roundIdx);
            if (this->block_.params_.singleCoreM > 0 && this->block_.params_.singleCoreN > 0) {
                this->block_.template CalcGMOffset<aTrans, bTrans, scaleType, formatX2>();
                SetScaleTensor();
                if (this->block_.tilingData_->matmulTiling.isBias) {
                    mm_.SetBias(this->biasGlobal_[this->block_.offset_.offsetBias]);
                }
                mm_.SetTensorA(this->aGlobal_[this->block_.offset_.offsetA], aTrans);
                mm_.SetTensorB(bl1Local_, bTrans);
                mm_.SetSingleShape(this->block_.params_.singleCoreM, this->block_.params_.singleCoreN,
                                   this->block_.tilingData_->matmulTiling.singleCoreK);
                mm_.SetOrgShape(this->block_.tilingData_->matmulTiling.M, this->block_.params_.singleCoreN,
                                this->block_.tilingData_->matmulTiling.Ka, this->block_.tilingData_->matmulTiling.Kb,
                                this->block_.tilingData_->matmulTiling.N);
                mm_.Iterate();
                mm_.GetTensorC(this->cGlobal_[this->block_.offset_.offsetC]);
            }
        }
    }
}
}  // namespace QuantBatchMatmulV3

#endif  // QBMM_CUBE_ON_THE_FLY_BL1_FULL_LOAD_H