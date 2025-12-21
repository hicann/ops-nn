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
 * \file matmul_custom_impl.h
 * \brief
 */
#ifndef MATMUL_CUSTOM_IMPL_H
#define MATMUL_CUSTOM_IMPL_H

#include "tool.h"

using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::SetFlag;
using AscendC::TEventID;
using AscendC::WaitFlag;

namespace QuantBatchMatmulV4
{
template <typename xType, typename wType, typename biasType, typename yType, typename scaleType, bool aTrans,
          bool bTrans, typename inputWType>
class MatmulCustomImpl
{
    using bL1DataType = typename inputWType::T;
    using aL0DataType = typename AscendC::GetL0DataType<xType, true>::Type;
    using bL0DataType = typename AscendC::GetL0DataType<typename inputWType::T, true>::Type;

public:
    __aicore__ inline MatmulCustomImpl(){};

    __aicore__ inline void Init(const TCubeTiling* __restrict matmulTiling, AscendC::TPipe* tpipe);
    __aicore__ inline void SetTensorA(const LocalTensor<xType> &aL1Tensor);
    __aicore__ inline void SetTensorB(const LocalTensor<bL1DataType> &bL1Tensor);
    __aicore__ inline void SetTensorScaleA(const LocalTensor<scaleType> &scaleAL1Tensor);
    __aicore__ inline void SetTensorScaleB(const LocalTensor<scaleType> &scaleBL1Tensor);
    __aicore__ inline void SetBias(const LocalTensor<biasType> &biasL1Tensor);
    __aicore__ inline void Iterate(bool enPartialSum);
    __aicore__ inline void GetTensorC(const GlobalTensor<yType> &outTensor);
    __aicore__ inline void SetTail(int64_t singleM, int64_t singleN, int64_t singleK);
    __aicore__ inline void SetOrgShape(int64_t mSize, int64_t nSize, int64_t kASize, int64_t kBSize,
                                       int64_t originNSize);
    __aicore__ inline void End();

private:
    __aicore__ inline void CopyAMatrixL1ToL0A(LocalTensor<aL0DataType> aL0Tensor, int32_t kL0Idx, int32_t baseK);
    __aicore__ inline void CopyBMatrixL1ToL0B(LocalTensor<bL0DataType> bL0Tensor, int32_t kL0Idx, int32_t baseK);
    __aicore__ inline void CopyBiasL1ToBT(LocalTensor<float> biasBtTensor);
    __aicore__ inline void PostUpdateParams();
    TBuf<TPosition::A2> a2TBuf_;
    TBuf<TPosition::B2> b2TBuf_;
    TBuf<TPosition::CO1> co1TBuf_;
    TBuf<TPosition::C2> biasTableTBuf_;

    LocalTensor<float> cL0Tensor_;
    LocalTensor<aL0DataType> aL0Tensor_;
    LocalTensor<bL0DataType> bL0Tensor_;
    LocalTensor<float> biasBtTensor_;

    LocalTensor<xType> aL1Tensor_;
    LocalTensor<bL1DataType> bL1Tensor_;
    LocalTensor<scaleType> scaleAL1Tensor_;
    LocalTensor<scaleType> scaleBL1Tensor_;
    LocalTensor<biasType> biasL1Tensor_;

    const TCubeTiling* matmulTiling_;

    TEventID eventIdsMToMte1_[2];
    TEventID eventIdsMte1ToM_[2];
    TEventID eventIdsMToFix_[2];
    TEventID eventIdsFixToM_[2];

    static constexpr int32_t L0C_BUFFER_SIZE = 262144;
    static constexpr int32_t L0A_BUFFER_SIZE = 65536;
    static constexpr int32_t L0B_BUFFER_SIZE = 65536;
    static constexpr int32_t BIAS_TABLE_SIZE = 4096;
    static constexpr int32_t BUFFER_HALF_SHL = 15;

    int64_t madLoopIdx_ = 0;
    int64_t totalKLoopIdx_ = 0;  // 第几次K的循环，由外部控制, 用于获取当前iterate实际的K方向计算量
    int64_t kIterNum_ = 0;       // k方向L1外的总计循环次数
    int64_t mAL1Size_ = 0;       // 根据当前tiling策略，等于baseM
    int64_t nBL1Size_ = 0;       // 根据当前tiling策略，等于baseN
    int64_t kAL1Size_ = 0;
    int64_t kBL1Size_ = 0;
    int64_t originNSize_ = 0;
    int64_t scaleAFactor_ = 1;   // x1Scale相对与AL1在K方向的载入倍数
    int64_t scaleBFactor_ = 1;
    int64_t singleM_;
    int64_t singleN_;
    int64_t singleK_;
};

template <typename xType, typename wType, typename biasType, typename yType, typename scaleType, bool aTrans,
          bool bTrans, typename inputWType>
__aicore__ inline void MatmulCustomImpl<xType, wType, biasType, yType, scaleType, aTrans, bTrans,
                                        inputWType>::SetOrgShape(int64_t mSize, int64_t nSize, int64_t kASize,
                                                                 int64_t kBSize, int64_t originNSize)
{
    mAL1Size_ = mSize;
    nBL1Size_ = nSize;
    kAL1Size_ = kASize;
    kBL1Size_ = kBSize;
    originNSize_ = originNSize;
    kIterNum_ = CeilDiv(static_cast<int64_t>(matmulTiling_->Kb), kBL1Size_);
    scaleAFactor_ = matmulTiling_->mxTypePara & 0xff;
    scaleBFactor_ = (matmulTiling_->mxTypePara >> 8) & 0xff;
}

template <typename xType, typename wType, typename biasType, typename yType, typename scaleType, bool aTrans,
          bool bTrans, typename inputWType>
__aicore__ inline void MatmulCustomImpl<xType, wType, biasType, yType, scaleType, aTrans, bTrans, inputWType>::SetTail(
    int64_t singleM, int64_t singleN, int64_t singleK)
{
    singleM_ = singleM;
    singleN_ = singleN;
    singleK_ = singleK;
}

template <typename xType, typename wType, typename biasType, typename yType, typename scaleType, bool aTrans,
          bool bTrans, typename inputWType>
__aicore__ inline void MatmulCustomImpl<xType, wType, biasType, yType, scaleType, aTrans, bTrans, inputWType>::Init(
    const TCubeTiling* __restrict matmulTiling, AscendC::TPipe* tpipe)
{
    matmulTiling_ = matmulTiling;

    tpipe->InitBuffer(co1TBuf_, L0C_BUFFER_SIZE);
    tpipe->InitBuffer(a2TBuf_, L0A_BUFFER_SIZE);
    tpipe->InitBuffer(b2TBuf_, L0B_BUFFER_SIZE);
    tpipe->InitBuffer(biasTableTBuf_, BIAS_TABLE_SIZE);

    cL0Tensor_ = co1TBuf_.template Get<float>();
    aL0Tensor_ = a2TBuf_.template Get<aL0DataType>();
    bL0Tensor_ = b2TBuf_.template Get<bL0DataType>();
    biasBtTensor_ = biasTableTBuf_.template Get<float>();

    eventIdsMToMte1_[0] = GetTPipePtr()->AllocEventID<HardEvent::M_MTE1>();
    eventIdsMToMte1_[1] = GetTPipePtr()->AllocEventID<HardEvent::M_MTE1>();
    eventIdsMte1ToM_[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE1_M>();
    eventIdsMte1ToM_[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE1_M>();
    eventIdsMToFix_[0] = GetTPipePtr()->AllocEventID<HardEvent::M_FIX>();
    eventIdsMToFix_[1] = GetTPipePtr()->AllocEventID<HardEvent::M_FIX>();
    eventIdsFixToM_[0] = GetTPipePtr()->AllocEventID<HardEvent::FIX_M>();
    eventIdsFixToM_[1] = GetTPipePtr()->AllocEventID<HardEvent::FIX_M>();

    SetFlag<HardEvent::M_MTE1>(eventIdsMToMte1_[0]);
    SetFlag<HardEvent::M_MTE1>(eventIdsMToMte1_[1]);
    // L0C 一定要开DB
    SetFlag<HardEvent::FIX_M>(eventIdsFixToM_[0]);
    SetFlag<HardEvent::FIX_M>(eventIdsFixToM_[1]);
}

//  每次GetTensorC之后调用
template <typename xType, typename wType, typename biasType, typename yType, typename scaleType, bool aTrans,
          bool bTrans, typename inputWType>
__aicore__ inline void
MatmulCustomImpl<xType, wType, biasType, yType, scaleType, aTrans, bTrans, inputWType>::PostUpdateParams()
{
    madLoopIdx_ += 1;
    totalKLoopIdx_ = 0;
}

template <typename xType, typename wType, typename biasType, typename yType, typename scaleType, bool aTrans,
          bool bTrans, typename inputWType>
__aicore__ inline void MatmulCustomImpl<xType, wType, biasType, yType, scaleType, aTrans, bTrans,
                                        inputWType>::CopyAMatrixL1ToL0A(LocalTensor<aL0DataType> aL0Tensor,
                                                                        int32_t kL0Idx, int32_t baseK)
{
    AscendC::LoadData2DParamsV2 aL0Load2dParams;
    aL0Load2dParams.mStartPosition = 0;
    aL0Load2dParams.kStartPosition =
        CeilDiv(static_cast<int64_t>(kL0Idx * matmulTiling_->baseK * sizeof(xType)), 32L);  // 单位32B
    aL0Load2dParams.mStep = CeilDiv(singleM_, static_cast<int64_t>(AscendC::BLOCK_CUBE));   // 单位16元素
    aL0Load2dParams.kStep = CeilDiv(baseK, 32);                                             // 单位32B
    aL0Load2dParams.srcStride = CeilDiv(mAL1Size_ * 32, 512L);  // 32为K0，stride单位为512B
    aL0Load2dParams.dstStride = aL0Load2dParams.srcStride;      // L1和L0 M大小一样，k方向stride也一样
    AscendC::LoadData2DMxParams aL0Load2dMx;
    aL0Load2dMx.xStartPosition = 0;
    aL0Load2dMx.yStartPosition = CeilDiv(kL0Idx * matmulTiling_->baseK, 64);  // 单位是32B
    aL0Load2dMx.xStep = CeilDiv(singleM_, static_cast<int64_t>(AscendC::BLOCK_CUBE));  // 单位是1个32B分形，
    aL0Load2dMx.yStep = CeilDiv(baseK, 64);                                   // baseK / 32 / 2
    aL0Load2dMx.srcStride = CeilDiv(matmulTiling_->baseK * matmulTiling_->stepKa * scaleAFactor_, 64L);
    aL0Load2dMx.dstStride = CeilDiv(baseK, 64);
    AscendC::LoadData(aL0Tensor, aL1Tensor_, scaleAL1Tensor_, aL0Load2dParams, aL0Load2dMx);
}

template <typename xType, typename wType, typename biasType, typename yType, typename scaleType, bool aTrans,
          bool bTrans, typename inputWType>
__aicore__ inline void MatmulCustomImpl<xType, wType, biasType, yType, scaleType, aTrans, bTrans,
                                        inputWType>::CopyBMatrixL1ToL0B(LocalTensor<bL0DataType> bL0Tensor,
                                                                        int32_t kL0Idx, int32_t baseK)
{
    AscendC::LoadData2DParamsV2 bL0Load2dParams;
    bL0Load2dParams.mStartPosition = 0;
    bL0Load2dParams.kStartPosition =
        CeilDiv(static_cast<int64_t>(kL0Idx * matmulTiling_->baseK * sizeof(xType)), 32L);  // 单位32B
    bL0Load2dParams.mStep =  CeilDiv(singleN_, static_cast<int64_t>(AscendC::BLOCK_CUBE));  // 单位16元素
    bL0Load2dParams.kStep = CeilDiv(baseK, 32);                                             // 单位32B
    bL0Load2dParams.srcStride = CeilDiv(nBL1Size_ * 32, 512L);  // 32为K0，stride单位为512B
    bL0Load2dParams.dstStride = bL0Load2dParams.srcStride;      // L1和L0 N方向大小一样，k方向stride也一样
    AscendC::LoadData2DMxParams bL0Load2dMx;
    bL0Load2dMx.xStartPosition = 0;
    bL0Load2dMx.yStartPosition = CeilDiv(kL0Idx * matmulTiling_->baseK, 64);  // 单位是32B
    bL0Load2dMx.xStep =  CeilDiv(singleN_, static_cast<int64_t>(AscendC::BLOCK_CUBE));   // 单位是1个32B分形，
    bL0Load2dMx.yStep = CeilDiv(baseK, 64);                                   // baseK / 32 / 2
    bL0Load2dMx.srcStride = CeilDiv(matmulTiling_->baseK * matmulTiling_->stepKb * scaleBFactor_, 64L);
    bL0Load2dMx.dstStride = CeilDiv(baseK, 64);
    AscendC::LoadData(bL0Tensor, bL1Tensor_, scaleBL1Tensor_, bL0Load2dParams, bL0Load2dMx);
}

template <typename xType, typename wType, typename biasType, typename yType, typename scaleType, bool aTrans,
          bool bTrans, typename inputWType>
__aicore__ inline void MatmulCustomImpl<xType, wType, biasType, yType, scaleType, aTrans, bTrans,
                                        inputWType>::CopyBiasL1ToBT(LocalTensor<float> biasBtTensor)
{
    constexpr auto biasDataType = IsSameType<float, biasType>::value ? 2 : 1;  // 2::fp32, 1:fp16/bf16
    uint16_t lenBurst = CeilDiv(singleN_* biasDataType * 2, 32L);
    if constexpr (IsSameType<float, biasType>::value) {
        lenBurst = CeilAlign(lenBurst, static_cast<uint16_t>(2));
    }
    AscendC::DataCopy(biasBtTensor, biasL1Tensor_, {1, lenBurst, 0, 0});
}

// 要求KAL1 = KBL1
template <typename xType, typename wType, typename biasType, typename yType, typename scaleType, bool aTrans,
          bool bTrans, typename inputWType>
__aicore__ inline void MatmulCustomImpl<xType, wType, biasType, yType, scaleType, aTrans, bTrans, inputWType>::Iterate(
    bool enPartialSum)
{
    LocalTensor<float> cL0Tensor = cL0Tensor_[(madLoopIdx_ & 1) << BUFFER_HALF_SHL];
    LocalTensor<aL0DataType> aL0Tensor;
    LocalTensor<bL0DataType> bL0Tensor;
    LocalTensor<float> biasBtTensor;

    AscendC::MmadParams mmadParams;
    mmadParams.m = CeilAlign(singleM_, static_cast<int64_t>(AscendC::BLOCK_CUBE));
    mmadParams.n = CeilAlign(singleN_, static_cast<int64_t>(AscendC::BLOCK_CUBE));
    mmadParams.cmatrixInitVal = !enPartialSum;
    mmadParams.cmatrixSource = false;
    int32_t kFractalIdx = totalKLoopIdx_ * matmulTiling_->stepKa;  // 正在计算第几个基本块，用于选择pingpong
    int32_t stepK = CeilDiv(kAL1Size_, static_cast<int64_t>(matmulTiling_->baseK));
    bool needPipeM = mmadParams.m * mmadParams.n < 2560; // baseM/16 * baseN/16 < 10时，需要插入同步保证精度
    for (int32_t kL0Idx = 0; kL0Idx < stepK; kL0Idx++) {
        int32_t baseK = (kL0Idx == stepK - 1) ? kAL1Size_ - kL0Idx * matmulTiling_->baseK : matmulTiling_->baseK;
        kFractalIdx += kL0Idx;
        aL0Tensor = aL0Tensor_[(kFractalIdx & 1) << BUFFER_HALF_SHL];
        bL0Tensor = bL0Tensor_[(kFractalIdx & 1) << BUFFER_HALF_SHL];
        WaitFlag<HardEvent::M_MTE1>(eventIdsMToMte1_[kFractalIdx & 1]);
        CopyAMatrixL1ToL0A(aL0Tensor, kL0Idx, baseK);
        CopyBMatrixL1ToL0B(bL0Tensor, kL0Idx, baseK);

        if (kFractalIdx == 0) {
            WaitFlag<HardEvent::FIX_M>(eventIdsFixToM_[madLoopIdx_ & 1]);
        }
        mmadParams.k = baseK;
        if (kL0Idx > 0) {
            mmadParams.cmatrixInitVal = false;
        }
        if (matmulTiling_->isBias) {
            biasBtTensor = biasBtTensor_[(kFractalIdx & 1) * 512]; // 512: biasTable的一半
            CopyBiasL1ToBT(biasBtTensor);
        }
        SetFlag<HardEvent::MTE1_M>(eventIdsMte1ToM_[kFractalIdx & 1]);
        WaitFlag<HardEvent::MTE1_M>(eventIdsMte1ToM_[kFractalIdx & 1]);
        if (matmulTiling_->isBias && kFractalIdx == 0) {
            mmadParams.cmatrixInitVal = false;
            AscendC::Mmad(cL0Tensor, aL0Tensor, bL0Tensor, biasBtTensor, mmadParams);
        } else {
            AscendC::Mmad(cL0Tensor, aL0Tensor, bL0Tensor, mmadParams);
        }
        if (needPipeM) {
            PipeBarrier<PIPE_M>();
        }
        SetFlag<HardEvent::M_MTE1>(eventIdsMToMte1_[kFractalIdx & 1]);
    }
    totalKLoopIdx_ += 1;
}

template <typename xType, typename wType, typename biasType, typename yType, typename scaleType, bool aTrans,
          bool bTrans, typename inputWType>
__aicore__ inline void MatmulCustomImpl<xType, wType, biasType, yType, scaleType, aTrans, bTrans,
                                        inputWType>::GetTensorC(const GlobalTensor<yType> &outTensor)
{
    LocalTensor<float> cL0Tensor = cL0Tensor_[(madLoopIdx_ & 1) << BUFFER_HALF_SHL];
    AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> fixParams;
    fixParams.nSize = singleN_;
    fixParams.mSize = singleM_;
    fixParams.srcStride = CeilAlign(singleM_, static_cast<int64_t>(AscendC::BLOCK_CUBE));
    fixParams.dstStride = originNSize_;
    if constexpr (IsSameType<yType, bfloat16_t>::value) {
        fixParams.quantPre = QuantMode_t::F322BF16;
    } else {
        fixParams.quantPre = QuantMode_t::F322F16;
    }
    fixParams.params.ndNum = 1;
    SetFlag<HardEvent::M_FIX>(eventIdsMToFix_[madLoopIdx_ & 1]);
    WaitFlag<HardEvent::M_FIX>(eventIdsMToFix_[madLoopIdx_ & 1]);
    AscendC::Fixpipe<yType, float, AscendC::CFG_ROW_MAJOR>(outTensor, cL0Tensor, fixParams);
    SetFlag<HardEvent::FIX_M>(eventIdsFixToM_[madLoopIdx_ & 1]);
    PostUpdateParams();
}

template <typename xType, typename wType, typename biasType, typename yType, typename scaleType, bool aTrans,
          bool bTrans, typename inputWType>
__aicore__ inline void MatmulCustomImpl<xType, wType, biasType, yType, scaleType, aTrans, bTrans,
                                        inputWType>::SetTensorA(const LocalTensor<xType> &aL1Tensor)
{
    aL1Tensor_ = aL1Tensor;
}

template <typename xType, typename wType, typename biasType, typename yType, typename scaleType, bool aTrans,
          bool bTrans, typename inputWType>
__aicore__ inline void MatmulCustomImpl<xType, wType, biasType, yType, scaleType, aTrans, bTrans,
                                        inputWType>::SetTensorB(const LocalTensor<bL1DataType> &bL1Tensor)
{
    bL1Tensor_ = bL1Tensor;
}

template <typename xType, typename wType, typename biasType, typename yType, typename scaleType, bool aTrans,
          bool bTrans, typename inputWType>
__aicore__ inline void MatmulCustomImpl<xType, wType, biasType, yType, scaleType, aTrans, bTrans,
                                        inputWType>::SetTensorScaleA(const LocalTensor<scaleType> &scaleAL1Tensor)
{
    scaleAL1Tensor_ = scaleAL1Tensor;
}

template <typename xType, typename wType, typename biasType, typename yType, typename scaleType, bool aTrans,
          bool bTrans, typename inputWType>
__aicore__ inline void MatmulCustomImpl<xType, wType, biasType, yType, scaleType, aTrans, bTrans,
                                        inputWType>::SetTensorScaleB(const LocalTensor<scaleType> &scaleBL1Tensor)
{
    scaleBL1Tensor_ = scaleBL1Tensor;
}

template <typename xType, typename wType, typename biasType, typename yType, typename scaleType, bool aTrans,
          bool bTrans, typename inputWType>
__aicore__ inline void MatmulCustomImpl<xType, wType, biasType, yType, scaleType, aTrans, bTrans, inputWType>::SetBias(
    const LocalTensor<biasType> &biasTensor)
{
    biasL1Tensor_ = biasTensor;
}

template <typename xType, typename wType, typename biasType, typename yType, typename scaleType, bool aTrans,
          bool bTrans, typename inputWType>
__aicore__ inline void MatmulCustomImpl<xType, wType, biasType, yType, scaleType, aTrans, bTrans, inputWType>::End()
{
    WaitFlag<HardEvent::M_MTE1>(eventIdsMToMte1_[0]);
    WaitFlag<HardEvent::M_MTE1>(eventIdsMToMte1_[1]);
    WaitFlag<HardEvent::FIX_M>(eventIdsFixToM_[0]);
    WaitFlag<HardEvent::FIX_M>(eventIdsFixToM_[1]);

    GetTPipePtr()->ReleaseEventID<HardEvent::M_MTE1>(eventIdsMToMte1_[0]);
    GetTPipePtr()->ReleaseEventID<HardEvent::M_MTE1>(eventIdsMToMte1_[1]);
    GetTPipePtr()->ReleaseEventID<HardEvent::FIX_M>(eventIdsFixToM_[0]);
    GetTPipePtr()->ReleaseEventID<HardEvent::FIX_M>(eventIdsFixToM_[1]);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_M>(eventIdsMte1ToM_[0]);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_M>(eventIdsMte1ToM_[1]);
    GetTPipePtr()->ReleaseEventID<HardEvent::M_FIX>(eventIdsMToFix_[0]);
    GetTPipePtr()->ReleaseEventID<HardEvent::M_FIX>(eventIdsMToFix_[1]);
}

}  // namespace QuantBatchMatmulV4
#endif  // MATMUL_CUSTOM_IMPL_H