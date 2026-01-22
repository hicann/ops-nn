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
 * \file qbmm_cube_on_the_fly_abl1_full_load.h
 * \brief
 */
#ifndef QBMM_CUBE_ON_THE_FLY_ABL1_FULL_LOAD_H
#define QBMM_CUBE_ON_THE_FLY_ABL1_FULL_LOAD_H

#include "../quant_batch_matmul_v3_base.h"
#include "qbmm_api_utils.h"
#include "qbmm_asw_block.h"
#include "qbmm_cube_on_the_fly.h"

namespace QuantBatchMatmulV3 {

using AscendC::GetBlockIdx;
using AscendC::InitConstValue;
using AscendC::LocalTensor;
using AscendC::ONE_BLK_SIZE;
using AscendC::PipeBarrier;
using AscendC::QuePosition;
using AscendC::TPipe;
using AscendC::TQue;

namespace ABL1FullLoad {
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
class MatmulAswKernelABL1FullLoad : public MatMulASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS> {
public:
    __aicore__ inline MatmulAswKernelABL1FullLoad()
    {
    }
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR bias, GM_ADDR scale, GM_ADDR perTokenScale,
                                GM_ADDR cGM, GM_ADDR workspace, const void *tilingData, TPipe *pipe);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessWithoutBatch();

protected:
    using scaleType =
        typename AscendC::Conditional<IsSameType<inputScaleType, int64_t>::value, uint64_t, inputScaleType>::type;
    // Temp Swap Cache Memory, 用于临时把数据交换到额外空间，进行Matmul计算
    using aType = matmul::MatmulType<AscendC::TPosition::TSCM, formatX1, x1Type, aTrans>;
    using bType = matmul::MatmulType<AscendC::TPosition::TSCM, formatX2, x2Type, bTrans>;
    using cType = matmul::MatmulType<AscendC::TPosition::GM, formatY, yType>;
    using biasMatmulType = matmul::MatmulType<AscendC::TPosition::TSCM, CubeFormat::ND, biasType>;
    matmul::MatmulImpl<aType, bType, cType, biasMatmulType, ABL1FullLoad::cfg_v<fusedOpType>> mm_;
    TQue<QuePosition::A1, 1> InQueueABL1_;
    TPipe *pipe_;
    LocalTensor<x1Type> al1Local_;
    LocalTensor<x2Type> bl1Local_;
    LocalTensor<biasType> biasl1Local_;
    LocalTensor<int8_t> abl1Local_;
    bool isMMultiCore_;
    bool isNMultiCore_;
};

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void MatmulAswKernelABL1FullLoad<LOCAL_TEMPLATE_FUNC_PARAMS>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR bias, GM_ADDR scale, GM_ADDR perTokenScale, GM_ADDR cGM, GM_ADDR workSpace,
    const void *tilingData, TPipe *pipe)
{
    if ASCEND_IS_AIV {
        return;
    }
    pipe_ = pipe;
    this->blockIdx_ = GetBlockIdx();
    this->quantBmmTilingData_ = static_cast<const DequantBmm::QuantBatchMatmulV3TilingDataParams *>(tilingData);
    this->UpdateGlobalAddr(aGM, bGM, bias, scale, perTokenScale, nullptr, cGM, workSpace);
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void MatmulAswKernelABL1FullLoad<LOCAL_TEMPLATE_FUNC_PARAMS>::Process()
{
    if ASCEND_IS_AIV {
        return;
    }
    isMMultiCore_ = this->block_.tilingData_->matmulTiling.singleCoreM < this->block_.tilingData_->matmulTiling.M;
    isNMultiCore_ = this->block_.tilingData_->matmulTiling.singleCoreN < this->block_.tilingData_->matmulTiling.N;
    // 临时代码，abl1支持s8s8 s8s4,
    uint64_t innerX1AlignedBlock = DequantBmm::GetC0Size<x1Type>();
    uint64_t innerX2AlignedBlock = DequantBmm::GetC0Size<x2Type>();
    uint64_t innerBiasAlignedBlock = ONE_BLK_SIZE / sizeof(biasType);

    uint64_t mAligned = 0;
    uint64_t kaAligned = 0;
    if constexpr (aTrans) {
        mAligned =
            DequantBmm::Align(this->block_.tilingData_->matmulTiling.singleCoreM, innerX1AlignedBlock);
        kaAligned = DequantBmm::Align(this->block_.tilingData_->matmulTiling.Ka, BMM_BLOCK_NUM);
    } else {
        mAligned = DequantBmm::Align(this->block_.tilingData_->matmulTiling.singleCoreM, BMM_BLOCK_NUM);
        kaAligned = DequantBmm::Align(this->block_.tilingData_->matmulTiling.Ka, innerX1AlignedBlock);
    }

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

    this->block_.UpdateBasicIndex(0);

    uint64_t aBufferSize = DequantBmm::GetSizeWithDataType<x1Type>(mAligned * kaAligned);
    uint64_t bBufferSize = DequantBmm::GetSizeWithDataType<x2Type>(kbAligned * nAligned);
    uint64_t biasBufferSize = 0UL;
    if (this->block_.tilingData_->matmulTiling.isBias) {
        biasBufferSize = nAligned * sizeof(biasType);
    }
    uint64_t bufferSize = aBufferSize + bBufferSize + biasBufferSize;

    // 统一申请buff
    pipe_->InitBuffer(InQueueABL1_, 1, bufferSize);

    // 尽可能将mm api init提前做 让ABL1的MTE2和MM的初始化并行起来
    mm_.SetSubBlockIdx(0);
    mm_.Init(&this->block_.tilingData_->matmulTiling, pipe_);

    abl1Local_ = InQueueABL1_.AllocTensor<int8_t>();
    al1Local_ = abl1Local_[0].template ReinterpretCast<x1Type>();
    CopyInA1<x1Type, aTrans>(this->block_, this->block_.params_.mIndex, isMMultiCore_, al1Local_, this->aGlobal_);
    bl1Local_ = abl1Local_[aBufferSize].template ReinterpretCast<x2Type>();
    if constexpr (formatX2 == CubeFormat::ND) {
        CopyInNDB1<x2Type, bTrans>(this->block_, this->block_.params_.nIndex, isNMultiCore_, bl1Local_, this->bGlobal_);
    } else {
        CopyInNZB1<x2Type, bTrans>(this->block_, this->block_.params_.nIndex, isNMultiCore_, bl1Local_, this->bGlobal_,
                                   innerX2AlignedBlock);
    }
    biasl1Local_ = abl1Local_[aBufferSize + bBufferSize].template ReinterpretCast<biasType>();
    if (this->block_.tilingData_->matmulTiling.isBias) {
        if constexpr (!IsSameType<biasType, int32_t>::value) {
            // L1清空，防止脏数据; 当biasType为int32时，可以不清除脏数据，不会出现inf -inf结果
            LocalTensor<uint16_t> l1BiasTmp = biasl1Local_.template ReinterpretCast<uint16_t>();
            InitConstValue(
                l1BiasTmp,
                {1, static_cast<uint16_t>(nAligned * sizeof(biasType) / static_cast<uint64_t>(DATA_BLOCK)), 0, 0U});
            PipeBarrier<PIPE_MTE2>();
        }
        // bias矩阵 biasTrans false, format ND
        CopyInB1Bias<biasType>(this->block_, this->block_.params_.nIndex, isNMultiCore_, biasl1Local_,
                               this->biasGlobal_, innerBiasAlignedBlock);
    }
    InQueueABL1_.EnQue(abl1Local_);

    this->block_.offset_.batchCOffset = 0;
    this->block_.offset_.batchAOffset = 0;
    this->block_.offset_.batchBOffset = 0;
    ProcessWithoutBatch();

    InQueueABL1_.FreeTensor(abl1Local_);
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void MatmulAswKernelABL1FullLoad<LOCAL_TEMPLATE_FUNC_PARAMS>::ProcessWithoutBatch()
{
    this->block_.template UpdateBlockParams<bTrans, formatX2>(0);
    if (this->block_.params_.singleCoreM > 0 && this->block_.params_.singleCoreN > 0) {
        this->block_.template CalcGMOffset<aTrans, bTrans, scaleType, formatX2>();

        if (static_cast<bool>(this->block_.tilingData_->params.isPerTensor) ||
            static_cast<bool>(this->block_.tilingData_->params.isDoubleScale)) {
            mm_.SetQuantScalar(this->block_.offset_.scaleScalar);
        } else {
            mm_.SetQuantVector(this->scaleGlobal_[this->block_.offset_.offsetScale]);
        }
        abl1Local_ = InQueueABL1_.DeQue<int8_t>();
        if (this->block_.tilingData_->matmulTiling.isBias) {
            mm_.SetBias(biasl1Local_);
        }
        mm_.SetTensorA(al1Local_, aTrans);
        mm_.SetTensorB(bl1Local_, bTrans);

        mm_.SetSingleShape(this->block_.params_.singleCoreM, this->block_.params_.singleCoreN,
                           this->block_.tilingData_->matmulTiling.singleCoreK);
        // MDL模板，L1输入场景默认bl1N=N，分核全载需要通过设置SetOrgShape指定al1N=singleCoreN
        mm_.SetOrgShape(this->block_.params_.singleCoreM, this->block_.params_.singleCoreN,
                        this->block_.tilingData_->matmulTiling.Ka);
        mm_.Iterate();
        // MDL模板，N为内轴，L0C->GM需重置shape
        mm_.SetOrgShape(this->block_.tilingData_->matmulTiling.M, this->block_.tilingData_->matmulTiling.N,
                        this->block_.tilingData_->matmulTiling.Ka);
        mm_.GetTensorC(this->cGlobal_[this->block_.offset_.offsetC]);
    }
}

}  // namespace QuantBatchMatmulV3

#endif  // QBMM_CUBE_ON_THE_FLY_ABL1_FULL_LOAD_H