/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file qbmm_cube_on_the_fly_al1_full_load.h
 * \brief
 */
#ifndef QBMM_CUBE_ON_THE_FLY_AL1_FULL_LOAD_H
#define QBMM_CUBE_ON_THE_FLY_AL1_FULL_LOAD_H

#include "qbmm_cube_on_the_fly.h"
#include "qbmm_asw_block.h"
#include "../quant_batch_matmul_v3_base.h"
#include "qbmm_api_utils.h"

namespace QuantBatchMatmulV3 {

using namespace AscendC;
using namespace matmul;

namespace AL1FullLoad {
template <bool isLut, class x2Type, FusedOpType fusedOpType>
constexpr auto cfg_v = [] {
    auto cfg = MM_CFG_NO_PRELOAD_OPEN_UNIT_FLAG;
    if constexpr (isLut) {
        cfg = CFG_NORM;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102)
        if constexpr (AscendC::IsSameType<x2Type, AscendC::int2b_t>::value) {
            cfg.decompMode = DecompressionMode::DECOMP_2bitTo4bit;
        } else if constexpr (AscendC::IsSameType<x2Type, AscendC::uint1b_t>::value) {
            cfg.decompMode = DecompressionMode::DECOMP_1bitTo4bit;
        } else {
            cfg.decompMode = DecompressionMode::DECOMP_4bitTo8bit;
        }
#endif
    }
    if constexpr (fusedOpType == FusedOpType::RELU) {
        cfg.enableRelu = true;
    }
    return cfg;
}();
}

template <class x1Type, class x2Type, class inputScaleType, class biasType, class yType, CubeFormat formatX1,
          CubeFormat formatX2, CubeFormat formatY, bool aTrans, bool bTrans, bool isLut = false,
          FusedOpType fusedOpType = FusedOpType::NONE>
class MatmulAswKernelAL1FullLoad : public MatMulASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS> {
public:
    __aicore__ inline MatmulAswKernelAL1FullLoad() {}
    // 分成两个Init函数，包含x2Table和不包含x2Table，分别用于qbmmv4,qbmmv3
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR bias, GM_ADDR scale, GM_ADDR perTokenScale,
                                GM_ADDR x2Table, GM_ADDR cGM, GM_ADDR workspace, const void *tilingData, TPipe *pipe);
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR bias, GM_ADDR scale, GM_ADDR perTokenScale,
                                GM_ADDR cGM, GM_ADDR workspace, const void *tilingData, TPipe *pipe);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessWithoutBatch();
    __aicore__ inline void SetScaleTensor();

protected:
    using scaleType =
        typename AscendC::Conditional<IsSameType<inputScaleType, int64_t>::value, uint64_t, inputScaleType>::type;
    using aType = typename AscendC::Conditional<
        DequantBmm::IsMxType<scaleType>(),
        matmul::MatmulTypeWithScale<AscendC::TPosition::TSCM, AscendC::TPosition::TSCM, formatX1, x1Type, aTrans>,
        matmul::MatmulType<AscendC::TPosition::TSCM, formatX1, x1Type, aTrans>>::type;
    using bType = typename AscendC::Conditional<
        DequantBmm::IsMxType<scaleType>(),
        matmul::MatmulTypeWithScale<AscendC::TPosition::GM, AscendC::TPosition::GM, formatX2, x2Type, bTrans>,
        matmul::MatmulType<AscendC::TPosition::GM, formatX2, x2Type, bTrans>>::type;

    using cType = matmul::MatmulType<AscendC::TPosition::GM, formatY, yType>;
    using biasMatmulType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, biasType>;
    using MmType = typename AscendC::Conditional<
        DequantBmm::IsMxType<scaleType>(),
        matmul::MatmulImpl<aType, bType, cType, biasMatmulType, AL1FullLoad::cfg_v<isLut, x2Type, fusedOpType>,
                           MatmulCallBackFunc<nullptr, nullptr, nullptr>, AscendC::Impl::Detail::MatmulWithScalePolicy>,
        matmul::MatmulImpl<aType, bType, cType, biasMatmulType, AL1FullLoad::cfg_v<isLut, x2Type, fusedOpType>>>::type;
    MmType mm_;
    TQue<QuePosition::A1, 1> InQueueAL1_;
    TPipe *pipe_;
    LocalTensor<x1Type> al1Local_;
    TQue<QuePosition::A1, 1> InQueueScaleA_;
    LocalTensor<fp8_e8m0_t> scaleALocal_;

    bool isMultiCore_;
};

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void MatmulAswKernelAL1FullLoad<LOCAL_TEMPLATE_FUNC_PARAMS>::Init(GM_ADDR aGM, GM_ADDR bGM,
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
    this->quantBmmTilingData_ = static_cast<const DequantBmm::QuantBatchMatmulV3TilingDataParams *>(tilingData);
    this->UpdateGlobalAddr(aGM, bGM, bias, scale, perTokenScale, nullptr, cGM, workSpace);
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void MatmulAswKernelAL1FullLoad<LOCAL_TEMPLATE_FUNC_PARAMS>::Init(GM_ADDR aGM, GM_ADDR bGM,
                                                                                    GM_ADDR bias, GM_ADDR scale,
                                                                                    GM_ADDR perTokenScale,
                                                                                    GM_ADDR x2Table, GM_ADDR cGM,
                                                                                    GM_ADDR workSpace,
                                                                                    const void *tilingData, TPipe *pipe)
{
    if ASCEND_IS_AIV {
        return;
    }
    pipe_ = pipe;
    this->blockIdx_ = GetBlockIdx();
    this->quantBmmTilingData_ = static_cast<const DequantBmm::QuantBatchMatmulV3TilingDataParams *>(tilingData);
    this->UpdateGlobalAddr(aGM, bGM, bias, scale, perTokenScale, x2Table, cGM, workSpace);
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void MatmulAswKernelAL1FullLoad<LOCAL_TEMPLATE_FUNC_PARAMS>::Process()
{
    if ASCEND_IS_AIV {
        return;
    }
    isMultiCore_ = this->block_.tilingData_->matmulTiling.singleCoreM < this->block_.tilingData_->matmulTiling.M;
    uint64_t innerAlignedBlock = 0;
    if constexpr (DequantBmm::IsMxType<scaleType>()) {
        innerAlignedBlock = static_cast<uint64_t>(MXFP_DIVISOR_SIZE);
    } else {
        // supportMmadS8S4平台的al1全载支持s8s8, s8s4, s4s4; david的al1全载支持s8s8, f8s8, f4f4
        innerAlignedBlock = DequantBmm::GetC0Size<x1Type>();
    }
    uint64_t mAligned = 0;
    uint64_t kAligned = 0;
    if constexpr (aTrans) {
        mAligned = DequantBmm::Align(this->block_.tilingData_->matmulTiling.singleCoreM, innerAlignedBlock);
        kAligned = DequantBmm::Align(this->block_.tilingData_->matmulTiling.Ka, BMM_BLOCK_NUM);
    } else {
        mAligned = DequantBmm::Align(this->block_.tilingData_->matmulTiling.singleCoreM, BMM_BLOCK_NUM);
        kAligned = DequantBmm::Align(this->block_.tilingData_->matmulTiling.Ka, innerAlignedBlock);
    }
    if (!isMultiCore_) {
        this->block_.params_.mCnt = 1;
        this->block_.params_.mainRow = 0;
        this->block_.params_.totalCnt = this->block_.params_.nCnt;
    }
    // 之前是基于sizeof(fp4) = 1的在做运算，现在统一用GetSizeWithDataType
    pipe_->InitBuffer(InQueueAL1_, 1, DequantBmm::GetSizeWithDataType<x1Type>(mAligned * kAligned));  // m k的计算

    al1Local_ = InQueueAL1_.AllocTensor<x1Type>();
    if constexpr (DequantBmm::IsMxType<scaleType>()) {
        if (kAligned - this->block_.tilingData_->matmulTiling.Ka >= MXFP_GROUP_SIZE) {
            auto padTensor = al1Local_.template ReinterpretCast<uint16_t>();
            InitConstValueParams<uint16_t> initConstValueParams;
            initConstValueParams.repeatTimes = 1;
            initConstValueParams.blockNum =
                DequantBmm::GetSizeWithDataType<x1Type>(mAligned * MXFP_GROUP_SIZE) / DATA_BLOCK;
            initConstValueParams.dstGap = 0;
            initConstValueParams.initValue = 0;
            uint64_t offset = mAligned * (kAligned - MXFP_GROUP_SIZE) / 2UL; // 2 means 2 8-bit elements
            InitConstValue(padTensor[offset], initConstValueParams);
            PipeBarrier<PIPE_MTE2>();
        }
    }
    CopyInA1<x1Type, aTrans>(this->block_, static_cast<uint64_t>(this->blockIdx_) % this->block_.params_.mCnt,
                             isMultiCore_, al1Local_, this->aGlobal_);

    if constexpr (DequantBmm::IsMxType<scaleType>()) {
        pipe_->InitBuffer(
            InQueueScaleA_, 1,
            mAligned * DequantBmm::CeilDiv(kAligned, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE * sizeof(fp8_e8m0_t));
        scaleALocal_ = InQueueScaleA_.AllocTensor<fp8_e8m0_t>();
        CopyInScaleA<fp8_e8m0_t, aTrans>(this->block_, this->blockIdx_, isMultiCore_, scaleALocal_,
                                            this->scaleAGlobal_);
    }
    mm_.SetSubBlockIdx(0);
    mm_.Init(&this->block_.tilingData_->matmulTiling, pipe_);

    if (likely(this->quantBmmTilingData_->params.batchC == 1)) {
        this->block_.offset_.batchCOffset = 0;
        this->block_.offset_.batchAOffset = 0;
        this->block_.offset_.batchBOffset = 0;
        ProcessWithoutBatch();
    } else {
        ProcessWithBatch<MatmulAswKernelAL1FullLoad>(this->block_, *this);
    }

    InQueueAL1_.FreeTensor(al1Local_);
    if constexpr (DequantBmm::IsMxType<scaleType>()) {
        InQueueScaleA_.FreeTensor(scaleALocal_);
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void MatmulAswKernelAL1FullLoad<LOCAL_TEMPLATE_FUNC_PARAMS>::SetScaleTensor()
{
    if constexpr (DequantBmm::IsMxType<scaleType>()) {
        mm_.SetTensorScaleA(scaleALocal_);
        mm_.SetTensorScaleB(this->scaleBGlobal_[this->block_.offset_.offsetScale]);
    } else {
        if (static_cast<bool>(this->block_.tilingData_->params.isPerTensor) ||
            static_cast<bool>(this->block_.tilingData_->params.isDoubleScale)) {
            mm_.SetQuantScalar(this->block_.offset_.scaleScalar);
        } else {
            mm_.SetQuantVector(this->scaleGlobal_[this->block_.offset_.offsetScale]);
        }
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void MatmulAswKernelAL1FullLoad<LOCAL_TEMPLATE_FUNC_PARAMS>::ProcessWithoutBatch()
{
    for (uint64_t j = 0; j < this->block_.params_.round; j++) {
        // 主块index和尾块index计算合并在一个函数
        this->block_.UpdateBasicIndex4AL1FullLoad(j);
        if (this->block_.params_.index < this->block_.params_.totalCnt) {
            // 主块Params和尾块Params计算合并在一个函数
            this->block_.template UpdateBlockParams4AL1FullLoad<bTrans, formatX2>(j);
            if (this->block_.params_.singleCoreM > 0 && this->block_.params_.singleCoreN > 0) {
                this->block_.template CalcGMOffset<aTrans, bTrans, scaleType, formatX2, isLut>();
                SetScaleTensor();
                if (this->block_.tilingData_->matmulTiling.isBias) {
                    mm_.SetBias(this->biasGlobal_[this->block_.offset_.offsetBias]);
                }
                mm_.SetTensorA(al1Local_, aTrans);
                uint64_t newOffsetB = this->block_.offset_.offsetB;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102)
                if constexpr (isLut && AscendC::IsSameType<x2Type, AscendC::uint1b_t>::value) {
                    // 临时代码，待api支持uint1后修正
                    newOffsetB = newOffsetB / INT1_X2_OFFSET_FACTOR_8;
                }
#endif
                mm_.SetTensorB(this->bGlobal_[newOffsetB], bTrans);
                if constexpr (isLut) {
                    mm_.SetLookupTable(this->x2TableGlobal_[this->block_.offset_.offsetX2Table]);
                }
                mm_.SetSingleShape(this->block_.params_.singleCoreM, this->block_.params_.singleCoreN,
                                   this->block_.tilingData_->matmulTiling.singleCoreK);
                if (isMultiCore_) {  // MDL模板，L1输入场景默认al1M=M，分核全载需要通过设置SetOrgShape指定al1M=singleCoreM
                    mm_.SetOrgShape(this->block_.params_.singleCoreM, this->block_.tilingData_->matmulTiling.N,
                                    this->block_.tilingData_->matmulTiling.Ka);
                }
                mm_.Iterate();
                mm_.GetTensorC(this->cGlobal_[this->block_.offset_.offsetC]);
            }
        }
    }
}

}  // namespace QuantBatchMatmulV3

#endif  // QBMM_CUBE_ON_THE_FLY_AL1_FULL_LOAD_H