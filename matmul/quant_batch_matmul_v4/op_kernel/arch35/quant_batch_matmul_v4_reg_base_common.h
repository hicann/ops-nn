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
 * \file quant_batch_matmul_v4_reg_base_common.h
 * \brief
 */

#ifndef QUANT_BATCH_MATMUL_V4_REG_BASE_COMMON_H
#define QUANT_BATCH_MATMUL_V4_REG_BASE_COMMON_H

#include "lib/matmul_intf.h"
#include "quant_batch_matmul_v4_constant.h"
#include "quant_batch_matmul_v4_tiling_data.h"
#include "quant_batch_matmul_v4_vf.h"

using AscendC::BLOCK_CUBE;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;
using AscendC::DataCopyExtParams;
using AscendC::DataCopyPadExtParams;
using AscendC::DataCopyPadParams;
using AscendC::DataCopyParams;
using AscendC::GetBlockIdx;
using AscendC::GetTaskRation;
using AscendC::GlobalTensor;
using AscendC::HardEvent;
using AscendC::int4b_t;
using AscendC::LocalTensor;
using AscendC::ONE_BLK_SIZE;
using AscendC::QuePosition;
using AscendC::RoundMode;
using AscendC::SetFlag;
using AscendC::SupportType;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::TPosition;
using AscendC::TQue;
using AscendC::VECTOR_REG_WIDTH;
using AscendC::WaitFlag;
namespace MicroAPI = AscendC::MicroAPI;
using AscendC::Dn2NzParams;
using AscendC::MicroAPI::GetRound;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::TypeGet;
using matmul::MatmulTypeWithScale;
using matmul::MatmulType;

namespace QuantBatchMatmulV4
{
using AscendC::IsSameType;

constexpr uint8_t MAX_AL1_BUF_NUM = 4;
template <TPosition POSITION, CubeFormat FORMAT, typename TYPE, bool ISTRANS = false,
          LayoutMode LAYOUT = LayoutMode::NONE, bool IBSHARE = false>
struct MatmulL1GmType : MatmulType<POSITION, FORMAT, TYPE, ISTRANS, LAYOUT, IBSHARE> {
    constexpr static TPosition srcPos = TPosition::GM;
};

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, QuantType antiQuantType, typename scaleType, bool weightNz = false>
class QuantBatchMatmulV4RegBaseCommonKernel
{
    using VregType = typename TypeGet<xType>::T;

public:
    __aicore__ inline QuantBatchMatmulV4RegBaseCommonKernel(){};

    __aicore__ inline void InitInputs(GM_ADDR x1, GM_ADDR x2, GM_ADDR x1_scale, GM_ADDR x2_scale, GM_ADDR x2_offset,
                                      GM_ADDR y_scale, GM_ADDR y_offset, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace);
    __aicore__ inline void UpdateIdxAndStep();
    __aicore__ inline void InitTilingData();
    __aicore__ inline void InitL1Params();
    __aicore__ inline bool IterMatmulOut();
    __aicore__ inline uint64_t GetVecScaleLen(uint64_t vecPingpong);
    __aicore__ inline void TwoVectorCoreSplit();
    __aicore__ inline void CopyInTensorWeight(int64_t bubKOffset, int64_t bubNOffset, int32_t bubKLen, int32_t bubNLen);
    __aicore__ inline void CopyInScaleOffset(int64_t bubNOffset, int32_t bubNLen, int32_t bubNLoopIdx,
                                             int32_t bubKLoopIdx, int64_t bubKOffset, int64_t bubKFactor);
    __aicore__ inline void AntiQuantCompute(int32_t bubKOffset, int32_t bubNLen, int32_t bubKLen);
    __aicore__ inline void AntiQuantComputeNormal(int32_t bubKOffset, int32_t bubNLen, int32_t bubKLen);
    __aicore__ inline void AntiQuantComputeKNGroupNz(int32_t bubKOffset, int32_t bubNLen, int32_t bubKLen);
    __aicore__ inline void AntiQuantComputeNKGroup(int32_t bubKOffset, int32_t bubNLen, int32_t bubKLen);
    __aicore__ inline void AntiQuantComputeNKMxNz(int32_t bubKOffset, int32_t bubNLen, int32_t bubKLen);
    __aicore__ inline int64_t GetVecGn(int32_t bubKOffset, int32_t bubKLen);
    __aicore__ inline void CopyVecOut2L1(int64_t l1Offset, LocalTensor<xType> ubLocal, int32_t bubKLen,
                                         int32_t bubNLen);
     __aicore__ inline void CopyVecOut2L1WeightNz(int64_t l1Offset, LocalTensor<xType> ubLocal, int32_t bubKLen,
                                                int32_t bubNLen);
    __aicore__ inline void MovWeightAndCompute(int64_t bubKOffset, int64_t bubNOffset, int32_t bubKLen,
                                               int32_t bubNLen);
    __aicore__ inline void BL1Process(uint64_t curBL1BufIdx, int64_t nBL1Offset, int64_t kBL1Offset, int32_t kL1Len,
                                      int32_t nL0Len);
    __aicore__ inline void BL1ProcessNK(uint64_t curBL1BufIdx, int64_t nBL1Offset, int64_t kBL1Offset, int32_t kL1Len,
                                        int32_t nL0Len);
    __aicore__ inline void BL1ProcessNK4Buffer(uint64_t curBL1BufIdx, int64_t nBL1Offset, int64_t kBL1Offset,
                                               int32_t kL1Len, int32_t nL0Len);
    __aicore__ inline void BL1ProcessKN(uint64_t curBL1BufIdx, int64_t nBL1Offset, int64_t kBL1Offset, int32_t kL1Len,
                                        int32_t nL0Len);
    __aicore__ inline void ComputeAntiquantParam(int32_t innerExtend, int32_t bubKOffset, int32_t outerExtend);
    __aicore__ inline int64_t CeilDiv(int64_t x, int64_t y);
    __aicore__ inline int64_t CeilAlign(int64_t x, int64_t y);
    __aicore__ inline int64_t FloorAlign(int64_t x, int64_t y);
    __aicore__ inline int32_t Max(int32_t x, int32_t y);
    __aicore__ inline int32_t Min(int32_t x, int32_t y);
    __aicore__ inline void CopyUb2L1();
    __aicore__ inline void VectorProcess();
    __aicore__ inline void InitSync(AscendC::TEventID eventIdsMte1ToMte2[MAX_AL1_BUF_NUM]);
    __aicore__ inline void EndSync(AscendC::TEventID eventIdsMte1ToMte2[MAX_AL1_BUF_NUM]);
    __aicore__ inline void PostProcess(int32_t kFactorIdx, AscendC::TEventID eventIdsMte1ToMte2[MAX_AL1_BUF_NUM]);

    __aicore__ inline void NotifyCube()
    {
#ifndef __CCE_KT_TEST__
        CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE3>(1);
#endif
    }

    __aicore__ inline void WaitForVector(uint64_t bL1BufIdx)
    {
        if (likely(tiling_->BL1Pingpong == QUADRUPLE_BUFFER)) {
#ifndef __CCE_KT_TEST__
            CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE1>(1 + FLAG_ID_MAX);
            CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE1>(1);
#endif
            return;
        }

        if (unlikely((tiling_->BL1Pingpong == 1) && (tiling_->vecCoreParallel == 0))) {
#ifndef __CCE_KT_TEST__
            CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE1>(1);
#endif
            return;
        }
        if (unlikely((tiling_->BL1Pingpong == 1) && (tiling_->vecCoreParallel == 1))) {
#ifndef __CCE_KT_TEST__
            CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE1>(1 + FLAG_ID_MAX);
            CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE1>(1);
#endif
            return;
        }

        if (likely(tiling_->BL1Pingpong == DOUBLE_BUFFER)) {
            if (bL1BufIdx == 1) {
#ifndef __CCE_KT_TEST__
                CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE1>(1 + FLAG_ID_MAX);
#endif
            } else {
#ifndef __CCE_KT_TEST__
                CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE1>(1);
#endif
            }
        }
    }

    __aicore__ inline void NotifyVector(uint64_t bL1BufIdx)
    {
        if (likely(tiling_->BL1Pingpong == QUADRUPLE_BUFFER)) {
#ifndef __CCE_KT_TEST__
            CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIV_AIC_FLAG + FLAG_ID_MAX);
            CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIV_AIC_FLAG);
#endif
            return;
        }

        if (unlikely((tiling_->BL1Pingpong == 1) && (tiling_->vecCoreParallel == 0))) {
#ifndef __CCE_KT_TEST__
            CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIV_AIC_FLAG);
#endif
            return;
        }

        if (unlikely((tiling_->BL1Pingpong == 1) && (tiling_->vecCoreParallel == 1))) {
#ifndef __CCE_KT_TEST__
            CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIV_AIC_FLAG + FLAG_ID_MAX);
            CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIV_AIC_FLAG);
#endif
            return;
        }

        if (likely(tiling_->BL1Pingpong == DOUBLE_BUFFER)) {
            if (bL1BufIdx == IDX_1) {
#ifndef __CCE_KT_TEST__
                CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIV_AIC_FLAG + FLAG_ID_MAX);
#endif
            } else {
#ifndef __CCE_KT_TEST__
                CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIV_AIC_FLAG);
#endif
            }
        }
    }

    __aicore__ inline void WaitForCube()
    {
#ifndef __CCE_KT_TEST__
        CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE3>(SYNC_AIV_AIC_FLAG);
#endif
    }

    TPipe* pipe_;
    const qbmmv4_tiling::QuantBatchMatmulV4TilingDataParams* tiling_;

    scaleType scaleValue_;
    xType offsetValue_;

    int32_t curBlockIdx_;
    int32_t nDimIdx_;
    int32_t mDimIdx_;

    int64_t mAL1Size_;
    int64_t kAL1Size_;
    int64_t nBL1Size_;
    int64_t kBL1Size_;
    int64_t aL1DataSize_;
    int64_t bL1DataSize_;

    int64_t kSingleCoreIterNum_;

    int64_t mSingleCoreSize_;
    int64_t kSingleCoreSize_;
    int64_t nSingleCoreSize_;

    int64_t tailM_;
    int64_t tailN_;

    int64_t tailL1M_;
    int64_t tailL1N_;

    int64_t mBlockOffset_;
    int64_t nBlockOffset_;

    int64_t kAL1Offset_;
    int64_t kBL1Offset_;
    int64_t kAL1Len_;
    int64_t kBL1Len_;
    int64_t vecKBL1Len_;
    int64_t nBL1Offset_;
    int64_t aGmOffset_;
    int64_t bL1Offset_;
    int64_t aL1Offset_;
    int64_t scaleAL1DataSize_;
    int64_t scaleBL1DataSize_;
    int64_t scaleAGmOffset_;
    int64_t scaleBGmOffset_;

    int64_t curML0Idx_;
    int64_t curNL0Idx_;
    int64_t curML1Idx_;
    int64_t curNL1Idx_;
    int64_t curStepM_;
    int64_t curStepN_;
    int64_t mIter_;
    int64_t nIter_;
    int64_t baseUseM_;
    int64_t baseUseN_;
    int64_t mAL1Len_;
    int64_t nBL1Len_;
    int64_t vecNBL1Len_;

    uint16_t repeatTimes_;
    int32_t vsstbConfig_;
    int64_t outDimOffset_;
    uint32_t groupNumBub_;

    uint32_t predictTailVcvt_;
    uint32_t predictTailCalcFirst_;
    uint32_t predictTailCalcSecond_;

    uint64_t ubScalePongFlag_;
    uint64_t vecWeightInLen_;
    uint64_t vecScaleOffsetLen_;
    uint64_t vecWeightOutLen_;
    int64_t kAl1Factor_;
    int64_t kBl1Factor_;

    uint64_t ubBufIdx_;
    int64_t idx_ = -1;
    uint8_t curAL1BufIdx_ = 0;
    uint64_t curBL1BufIdx_ = 0;
    static constexpr uint64_t QUADRUPLE_BUFFER = 4;
    uint8_t vecPingpong_ = SINGLE_BUFFER;
    bool isFirstIter_ = true;
    bool twoVectorCoreSplitN_ = false;
    bool twoVectorCoreSplitK_ = false;
    int8_t biasPingPong_ = 1;
    int8_t biasIdx_ = 0;

    int8_t scaleAIdx_ = 0;
    int8_t scaleBIdx_ = 0;
    int8_t scaleAFactor_ = 1;
    int8_t scaleBFactor_ = 1;

    GlobalTensor<xType> xGlobal_;
    GlobalTensor<wType> wGlobal_;
    GlobalTensor<xType> addGlobal_;
    GlobalTensor<scaleType> x1ScaleGlobal_;
    GlobalTensor<scaleType> x2ScaleGlobal_;
    GlobalTensor<biasType> biasGlobal_;
    GlobalTensor<yType> yGlobal_;
    GlobalTensor<uint64_t> quantScaleGlobal_;

    LocalTensor<wType> weightInUb_;
    LocalTensor<scaleType> scaleInUb_;
    LocalTensor<xType> offsetInUb_;
    LocalTensor<xType> weightOutUb_;
    LocalTensor<uint64_t> scaleMaskUb_;
    LocalTensor<xType> aL1LocalBuf0_;
    LocalTensor<xType> aL1LocalBuf1_;
    LocalTensor<xType> aL1LocalBuf2_;
    LocalTensor<xType> aL1LocalBuf3_;
    LocalTensor<xType> bL1LocalBuf0_;
    LocalTensor<xType> bL1LocalBuf1_;
    LocalTensor<xType> bL1LocalBuf2_;
    LocalTensor<xType> bL1LocalBuf3_;
    LocalTensor<xType> l1Local_;
    LocalTensor<scaleType> scaleAL1Buf0_;
    LocalTensor<scaleType> scaleAL1Buf1_;
    LocalTensor<scaleType> scaleAL1Buf2_;
    LocalTensor<scaleType> scaleAL1Buf3_;
    LocalTensor<scaleType> scaleBL1Buf0_;
    LocalTensor<scaleType> scaleBL1Buf1_;
    LocalTensor<scaleType> scaleBL1Buf2_;
    LocalTensor<scaleType> scaleBL1Buf3_;
    LocalTensor<biasType> biasL1Buf0_;
    LocalTensor<biasType> biasL1Buf1_;

    __local_mem__ int8_t* weightInUbBaseAddr_;
    __local_mem__ scaleType* scaleBaseAddr1_;
    __local_mem__ xType* offsetBaseAddr1_;
    __local_mem__ xType* weightOutUbAddr_;
    __local_mem__ xType* weightOutUbAddr1_;

protected:
    static constexpr uint64_t FLAG_ID_MAX = 16;
    static constexpr uint64_t L1_BUFFER_SIZE = 512 * 1024;
    static constexpr uint64_t L1_BUFFER_HALF_SIZE = 256 * 1024;
    static constexpr int32_t REPEAT_STRIDE_UINT = 5;  // 位移位数, 32
    static constexpr uint64_t DOUBLE_BUFFER = 2;
    static constexpr uint64_t SINGLE_BUFFER = 1;
    static constexpr uint64_t SYNC_AIV_AIC_FLAG = 2;
    static constexpr uint8_t SYNC_MODE4 = 4;
    static constexpr uint64_t INT4_DTYPE_PARAM = 1;  // 位移位数, 2
    static constexpr uint64_t VCVT_PARAM = 1;        // 位移位数, 2
    static constexpr int32_t VEC_MAX_ELEM_B16 = VECTOR_REG_WIDTH / sizeof(xType);
    static constexpr int32_t VEC_MAX_ELEM_B8 = VECTOR_REG_WIDTH / sizeof(xType);
    static constexpr int64_t ONE_BLK_ELEM_B16 = ONE_BLK_SIZE / sizeof(xType);
    static constexpr int64_t ONE_BLK_ELEM_B8 = ONE_BLK_SIZE / sizeof(xType);
    static constexpr int32_t BLOCK_CUBE_INT4 = BLOCK_CUBE >> INT4_DTYPE_PARAM;
    static constexpr int32_t BLOCK_NUM_REG = VECTOR_REG_WIDTH / ONE_BLK_SIZE;

    TBuf<QuePosition::TSCM> l1Tbuf_;

     AscendC::TEventID eventIdsScaleAMte1ToMte2_[DOUBLE_BUFFER];
     AscendC::TEventID eventIdsScaleBMte1ToMte2_[DOUBLE_BUFFER];
};

enum class IterateOrder : uint32_t { ORDER_M = 0, ORDER_N, UNDEF };

/*
 *  功能：在 per-channel、per-group 场景下，获取 vecScaleLen 的值
 */
template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, QuantType antiQuantType, typename scaleType, bool weightNz>
__aicore__ inline uint64_t
QuantBatchMatmulV4RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset, antiQuantType,
                                      scaleType, weightNz>::GetVecScaleLen(uint64_t vecPingpong)
{
    uint64_t vecScaleLen;
    if constexpr (antiQuantType == QuantType::PER_GROUP || antiQuantType == QuantType::MX) {
        if constexpr (bTrans) {
            vecScaleLen = tiling_->BL1Pingpong * tiling_->nBubSize *
                          CeilAlign(CeilDiv(tiling_->kBubSize, tiling_->groupSize), ONE_BLK_SIZE) * sizeof(scaleType);
        } else {
            // 申请 bL1BufNum * groupNumBub * nBubSize 大小的空间
            vecScaleLen = tiling_->BL1Pingpong * CeilDiv(tiling_->kBubSize, tiling_->groupSize) * tiling_->nBubSize *
                          sizeof(scaleType);
        }
    } else {
        // per-channel, 目前仅考虑 A16W8 场景
        vecScaleLen = vecPingpong * tiling_->nBubSize * sizeof(scaleType);
    }
    return vecScaleLen;
}

/*
 *  功能：申请各输入的 global buffer，A 矩阵与 B 矩阵在 L1 上的空间，B 矩阵、scale、offset 在 UB 上的空间
 */
template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, QuantType antiQuantType, typename scaleType, bool weightNz>
__aicore__ inline void
QuantBatchMatmulV4RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset, antiQuantType,
                                      scaleType, weightNz>::InitInputs(GM_ADDR x1, GM_ADDR x2,
                                                                                     GM_ADDR x1_scale, GM_ADDR x2_scale,
                                                                                     GM_ADDR x2_offset, GM_ADDR y_scale,
                                                                                     GM_ADDR y_offset, GM_ADDR bias,
                                                                                     GM_ADDR y, GM_ADDR workspace)
{
    // 申请 Global Buffer
    xGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ xType*>(x1));
    wGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ wType*>(x2));
    yGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ yType*>(y));
    biasGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ biasType*>(bias));
    x1ScaleGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ scaleType*>(x1_scale));
    x2ScaleGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ scaleType*>(x2_scale));
    addGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ xType*>(x2_offset));
    quantScaleGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(y_scale));

    // 申请 L1 Buffer 以及 B 矩阵在 UB 上的输入与输出 buffer 空间
    if constexpr(antiQuantType == QuantType::MX) {
        pipe_->InitBuffer(l1Tbuf_, L1_BUFFER_SIZE);
    } else {
        pipe_->InitBuffer(l1Tbuf_, L1_BUFFER_SIZE -
                                   tiling_->matmulTiling.isBias * tiling_->matmulTiling.baseN * sizeof(biasType) -
                                   tiling_->matmulTiling.baseN * sizeof(uint64_t));
    }

    if (likely(tiling_->BL1Pingpong == QUADRUPLE_BUFFER)) {
        // 目前仅考虑B 矩阵转置场景
        if constexpr (bTrans) {
            if constexpr (weightNz) { // MX A8W4
                vecWeightInLen_ = (tiling_->BL1Pingpong * tiling_->nBubSize * tiling_->kBubSize) >> INT4_DTYPE_PARAM;
                vecWeightOutLen_ = tiling_->BL1Pingpong * tiling_->nBubSize * tiling_->kBubSize * sizeof(xType);
            } else {
                vecWeightInLen_ = tiling_->BL1Pingpong * (tiling_->nBubSize * CeilAlign(tiling_->kBubSize, OFFSET_64)) >>
                                INT4_DTYPE_PARAM;
                vecWeightOutLen_ = tiling_->BL1Pingpong * (CeilAlign(tiling_->nBubSize, BLOCK_CUBE) + 1) *
                                CeilAlign(tiling_->kBubSize, ONE_BLK_SIZE);
            }
        } else {
            if constexpr (weightNz) {
                vecWeightInLen_ = (tiling_->BL1Pingpong * tiling_->nBubSize * tiling_->kBubSize) >> INT4_DTYPE_PARAM;
                vecWeightOutLen_ = tiling_->BL1Pingpong * tiling_->nBubSize * tiling_->kBubSize * sizeof(xType);
            }
        }
    } else {
        if constexpr (bTrans) {
            if constexpr (weightNz) {
                vecWeightInLen_ = (vecPingpong_ * tiling_->nBubSize * tiling_->kBubSize) >> INT4_DTYPE_PARAM;
                vecWeightOutLen_ = vecPingpong_ * tiling_->nBubSize * tiling_->kBubSize * sizeof(xType);
            } else {
                vecPingpong_ = Min(tiling_->BL1Pingpong, DOUBLE_BUFFER);
                vecWeightInLen_ =
                    vecPingpong_ * (tiling_->nBubSize * CeilAlign(tiling_->kBubSize, OFFSET_64)) >> INT4_DTYPE_PARAM;
                vecWeightOutLen_ = vecPingpong_ * (CeilAlign(tiling_->nBubSize, BLOCK_CUBE) + 1) *
                                CeilAlign(tiling_->kBubSize, ONE_BLK_SIZE);
            }
        } else {
            vecPingpong_ = tiling_->BL1Pingpong;
            if constexpr (weightNz) {
                vecWeightInLen_ = (vecPingpong_ * tiling_->nBubSize * tiling_->kBubSize) >> INT4_DTYPE_PARAM;
                vecWeightOutLen_ = vecPingpong_ * tiling_->nBubSize * tiling_->kBubSize * sizeof(xType);
            }
        }
    }

    // 申请 scale 和 offset 在 UB 上的 buffer 空间
    if constexpr (antiQuantType == QuantType::PER_TENSOR) {
        scaleValue_ = x2ScaleGlobal_.GetValue(0);
        if constexpr (hasAntiQuantOffset) {
            offsetValue_ = addGlobal_.GetValue(0);
        }
    } else {
        vecScaleOffsetLen_ = GetVecScaleLen(vecPingpong_);
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, QuantType antiQuantType, typename scaleType, bool weightNz>
__aicore__ inline void
QuantBatchMatmulV4RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset, antiQuantType,
                                      scaleType, weightNz>::InitL1Params()
{
    kAL1Size_ = tiling_->matmulTiling.stepKa * tiling_->matmulTiling.baseK;
    kBL1Size_ = tiling_->matmulTiling.stepKb * tiling_->matmulTiling.baseK;
    int64_t maxKL1 = Max(kAL1Size_, kBL1Size_);
    kAl1Factor_ = CeilDiv(maxKL1, kBL1Size_);
    kBl1Factor_ = CeilDiv(maxKL1, kAL1Size_);
    scaleAFactor_ = tiling_->matmulTiling.mxTypePara & 0xff;
    scaleBFactor_ = (tiling_->matmulTiling.mxTypePara >> 8) & 0xff; // 高8位表示scaleBFactor_

    mAL1Size_ = tiling_->matmulTiling.stepM * tiling_->matmulTiling.baseM;
    nBL1Size_ = tiling_->matmulTiling.stepN * tiling_->matmulTiling.baseN;
    aL1DataSize_ = mAL1Size_ * kAL1Size_;
    bL1DataSize_ = nBL1Size_ * kBL1Size_;
    scaleAL1DataSize_ = aL1DataSize_ * scaleAFactor_ / GROUP_SIZE_32;
    scaleBL1DataSize_ = bL1DataSize_ * scaleBFactor_ / GROUP_SIZE_32;

    uint64_t singleM = CeilAlign(CeilDiv(tiling_->mSize, tiling_->cubeNumBlocksM), 16);
    uint64_t singleN;
    // singleN按c0对齐
    if constexpr (weightNz && !bTrans) {
        // weightNz非转置 shape为(n1, k1, k0, n0) n0 = 32B/sizeof(A8) = 32
        singleN = CeilAlign(CeilDiv(tiling_->nSize, tiling_->cubeNumBlocksN), C0_SIZE_B8);
    } else {
        // 其它转置场景 shape为(k1, n1, n0, k0) n0为固定值16
        singleN = CeilAlign(CeilDiv(tiling_->nSize, tiling_->cubeNumBlocksN), BLOCK_CUBE);
    }

    kSingleCoreIterNum_ = CeilDiv(tiling_->kSize, Min(kAL1Size_, kBL1Size_));

    int64_t mTailCoreSize_ = tiling_->mSize - (tiling_->cubeNumBlocksM - 1) * singleM;  // 尾核 m 大小
    int64_t nTailCoreSize_ = tiling_->nSize - (tiling_->cubeNumBlocksN - 1) * singleN;  // 尾核 n 大小

    mSingleCoreSize_ = mDimIdx_ == tiling_->cubeNumBlocksM - 1 ? mTailCoreSize_ : singleM;  // 单核内 m 方向大小
    nSingleCoreSize_ = nDimIdx_ == tiling_->cubeNumBlocksN - 1 ? nTailCoreSize_ : singleN;  // 单核内 n 方向大小

    tailL1M_ = mSingleCoreSize_ % mAL1Size_;  // 单核内l1上m方向尾块
    if (tailL1M_ == 0) {
        tailL1M_ = mAL1Size_;
    }

    tailL1N_ = nSingleCoreSize_ % nBL1Size_;  // 单核内l1上n方向尾块
    if (tailL1N_ == 0) {
        tailL1N_ = nBL1Size_;
    }

    mBlockOffset_ = mDimIdx_ * singleM;  // m 方向核间偏移
    nBlockOffset_ = nDimIdx_ * singleN;  // n 方向核间偏移
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, QuantType antiQuantType, typename scaleType, bool weightNz>
__aicore__ inline void
QuantBatchMatmulV4RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset, antiQuantType,
                                      scaleType, weightNz>::InitTilingData()
{
    InitL1Params();

    mIter_ = CeilDiv(mSingleCoreSize_, tiling_->matmulTiling.baseM);  // 单核内一共有basem块数量
    nIter_ = CeilDiv(nSingleCoreSize_, tiling_->matmulTiling.baseN);  // 单核内一共有basen块数量
    biasPingPong_ = Min(nIter_, DOUBLE_BUFFER);

    tailM_ = mSingleCoreSize_ % tiling_->matmulTiling.baseM;  // 单核内l0上m方向尾块
    if (tailM_ == 0) {
        tailM_ = tiling_->matmulTiling.baseM;
    }

    tailN_ = nSingleCoreSize_ % tiling_->matmulTiling.baseN;  // 单核内l0上n方向尾块
    if (tailN_ == 0) {
        tailN_ = tiling_->matmulTiling.baseN;
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, QuantType antiQuantType, typename scaleType, bool weightNz>
__aicore__ inline void
QuantBatchMatmulV4RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset, antiQuantType,
                                      scaleType, weightNz>::CopyInTensorWeight(int64_t bubKOffset,
                                                                                             int64_t bubNOffset,
                                                                                             int32_t bubKLen,
                                                                                             int32_t bubNLen)
{
    DataCopyExtParams intriParams;
    intriParams.dstStride = 0;
    DataCopyPadExtParams<wType> padParams;
    int64_t gmOffset;  // GM 上 B 矩阵的地址偏移（以元素个数为单位）
    if constexpr (bTrans) {
        if constexpr (IsSameType<wType, fp4x2_e2m1_t>::value) {
            if constexpr (weightNz && (antiQuantType == QuantType::MX)) { // MXA8W4, NZ,(k1,n1,n0,k0)
                intriParams.blockCount = bubKLen / C0_SIZE_B8;     // k1
                intriParams.blockLen = bubNLen * BLOCK_CUBE;  // n1 * n0 * k0
                int64_t nAlignSize = CeilAlign(tiling_->nSize, BLOCK_CUBE);
                intriParams.srcStride = (nAlignSize - bubNLen) * BLOCK_CUBE;
                gmOffset = (bubKOffset * nAlignSize + bubNOffset * C0_SIZE_B8);
            } else {
                intriParams.blockCount = bubNLen;
                intriParams.blockLen = bubKLen >> INT4_DTYPE_PARAM;
                intriParams.srcStride = (tiling_->kSize - bubKLen) >> INT4_DTYPE_PARAM;
                gmOffset = bubNOffset * tiling_->kSize + bubKOffset;
            }
        } else {
#ifdef __CCE_KT_TEST__
            ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "not support yet"); });
#endif
        }
    } else {  // B 矩阵非转置
        intriParams.blockCount = bubNLen / C0_SIZE_B8;     // n1
        intriParams.blockLen = bubKLen * BLOCK_CUBE;  // k1 * k0 * n0
        int64_t kAlignSize = CeilAlign(tiling_->kSize, BLOCK_CUBE);
        intriParams.srcStride = (kAlignSize - bubKLen) * BLOCK_CUBE;
        gmOffset = (bubNOffset * kAlignSize + bubKOffset * C0_SIZE_B8);
    }

    // UB 上存储反量化前 B 矩阵的空间的地址偏移（以元素个数为单位）
    uint64_t weightInOffset;
    if constexpr (IsSameType<wType, int8_t>::value) {
        weightInOffset = ubBufIdx_ * (vecWeightInLen_ / vecPingpong_);
    } else {
        weightInOffset = ubBufIdx_ * (vecWeightInLen_ << INT4_DTYPE_PARAM) / tiling_->BL1Pingpong;
    }
    DataCopyPad(weightInUb_[weightInOffset], wGlobal_[gmOffset], intriParams, padParams);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, QuantType antiQuantType, typename scaleType, bool weightNz>
__aicore__ inline void
QuantBatchMatmulV4RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset, antiQuantType,
                                      scaleType, weightNz>::CopyInScaleOffset(int64_t bubNOffset, int32_t bubNLen,
                                                                              int32_t bubNLoopIdx, int32_t bubKLoopIdx,
                                                                              int64_t bubKOffset, int64_t bubKFactor)
{
    if constexpr (antiQuantType == QuantType::PER_TENSOR || antiQuantType == QuantType::MX) {
        return;
    }

    DataCopyExtParams intriParams;
    intriParams.dstStride = 0;
    DataCopyPadExtParams<scaleType> padParams;
    DataCopyPadExtParams<xType> padParamsOffset;
    int64_t gmOffset;  // GM 上 scale 和 offset 的地址偏移（以元素个数为单位）
    int64_t ubOffset;  // UB 上 scale 和 offset 的地址偏移（以元素个数为单位）

    if constexpr (bTrans) {
        if constexpr (antiQuantType == QuantType::PER_GROUP) {
            int64_t totalGroupNum = CeilDiv(tiling_->kSize, tiling_->groupSize);
            intriParams.blockCount = bubNLen;
            intriParams.blockLen = groupNumBub_ * sizeof(scaleType);
            intriParams.srcStride = (totalGroupNum - groupNumBub_) * sizeof(scaleType);
            gmOffset = bubNOffset * totalGroupNum + (bubKOffset / tiling_->groupSize);
            ubOffset = ubBufIdx_ * (vecScaleOffsetLen_ / sizeof(scaleType) / tiling_->BL1Pingpong);
        }
    } else {  // B 矩阵非转置
        intriParams.blockLen = bubNLen * sizeof(scaleType);
        intriParams.srcStride = (tiling_->nSize - bubNLen) * sizeof(scaleType);
        if constexpr (antiQuantType == QuantType::PER_GROUP) {
            // k_offset + n_offset
            gmOffset = bubKOffset / tiling_->groupSize * tiling_->nSize + bubNOffset;
            ubOffset = ubBufIdx_ * (vecScaleOffsetLen_ / sizeof(scaleType) / tiling_->BL1Pingpong);
            intriParams.blockCount = groupNumBub_;
        } else {
            intriParams.blockCount = 1;
        }
    }
    DataCopyPad(scaleInUb_[ubOffset], x2ScaleGlobal_[gmOffset], intriParams, padParams);
    if constexpr (hasAntiQuantOffset) {
        DataCopyPad(offsetInUb_[ubOffset], addGlobal_[gmOffset], intriParams, padParamsOffset);
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, QuantType antiQuantType, typename scaleType, bool weightNz>
__aicore__ inline void
QuantBatchMatmulV4RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset, antiQuantType,
                                      scaleType, weightNz>::ComputeAntiquantParam(int32_t innerExtend,
                                                                                                int32_t bubKOffset,
                                                                                                int32_t outerExtend)
{
    static_assert(SupportType<wType, fp4x2_e2m1_t>(), "only support fp4x2_e2m1_t");

    if constexpr (IsSameType<wType, fp4x2_e2m1_t>::value) {
        uint32_t calTailCount_ = innerExtend % VECTOR_REG_WIDTH;
        if (calTailCount_) {
            predictTailVcvt_ = calTailCount_;
            predictTailCalcFirst_ = Min(calTailCount_, VEC_MAX_ELEM_B8);
            predictTailCalcSecond_ = Max(0, calTailCount_ - VEC_MAX_ELEM_B8);
        } else {
            predictTailVcvt_ = innerExtend;
            predictTailCalcFirst_ = VEC_MAX_ELEM_B8;
            predictTailCalcSecond_ = VEC_MAX_ELEM_B8;
        }
        // not supported yet
    } else {
        uint32_t calTailCount_ = innerExtend % VECTOR_REG_WIDTH;
        if (calTailCount_) {
            predictTailVcvt_ = calTailCount_;
            predictTailCalcFirst_ = Min(calTailCount_, VEC_MAX_ELEM_B16);
            predictTailCalcSecond_ = Max(0, calTailCount_ - VEC_MAX_ELEM_B16);
        } else {
            predictTailVcvt_ = innerExtend;
            predictTailCalcFirst_ = VEC_MAX_ELEM_B16;
            predictTailCalcSecond_ = VEC_MAX_ELEM_B16;
        }
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, QuantType antiQuantType, typename scaleType, bool weightNz>
__aicore__ inline void
QuantBatchMatmulV4RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset, antiQuantType,
                                      scaleType, weightNz>::AntiQuantComputeKNGroupNz(
                                    int32_t bubKOffset, int32_t bubNLen, int32_t bubKLen)
{
    // 处理一块 (kBub, nBub), 数据格式为 (nBub1, kBub1, kBub0, nBub0)
    // 每次处理 256 个数，按照 256 个数 (256B) 输出, 对于每个 256B, 间隔 1024B 存放, 避免 bank 冲突
    ParamsGroupKN<xType, wType, scaleType> params;
    params.scaleMaskBaseAddr = (__local_mem__ uint8_t*)scaleMaskUb_.GetPhyAddr();
    params.groupNumUb = bubKLen / tiling_->groupSize;
    params.vLLoopNumInGroup = tiling_->groupSize * C0_SIZE_B8 / VECTOR_REG_WIDTH;
    params.n1LoopNum = CeilDiv(bubNLen, C0_SIZE_B8);
    params.bubNLen = bubNLen;
    params.scaleN1Stride = C0_SIZE_B8;
    params.weightInGroupIdStride = tiling_->groupSize * C0_SIZE_B8 / 2;
    params.weighInN1Stride = params.weightInGroupIdStride * params.groupNumUb;
    params.scaleBaseAddr0 = scaleBaseAddr1_;
    params.scaleBaseAddr1 = params.scaleBaseAddr0 + BLOCK_CUBE;
    params.weightInBaseAddr0 = weightInUbBaseAddr_;
    params.weightInBaseAddr1 = params.weightInBaseAddr0 + OFFSET_64;
    params.weightOutBaseAddr = weightOutUbAddr_;
    params.weightOutVlStride = VECTOR_REG_WIDTH * tiling_->BL1Pingpong;
    params.weightOutGroupIdStride = params.vLLoopNumInGroup * params.weightOutVlStride;
    params.weightOutN1Stride = params.groupNumUb * params.weightOutGroupIdStride;
    AscendC::VF_CALL<AntiquantW4PergroupKN<xType, wType, scaleType, hasAntiQuantOffset>>(params);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, QuantType antiQuantType, typename scaleType, bool weightNz>
__aicore__ inline void
QuantBatchMatmulV4RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset, antiQuantType,
                                      scaleType, weightNz>::AntiQuantComputeNKGroup(int32_t bubKOffset,
                                                                                                  int32_t bubNLen,
                                                                                                  int32_t bubKLen)
{
    if (tiling_->groupSize == GROUP_SIZE_32) {
        ParamsGroupSize32<xType, wType, scaleType> params;
        params.outerExtend = bubNLen;
        params.innerExtend = CeilDiv(CeilAlign(bubKLen, UB_ALIGN_SIZE_FOR_4BITS), AscendC::VECTOR_REG_WIDTH);
        params.dataBlockStride = CeilAlign(bubNLen, AscendC::BLOCK_CUBE) + 1;
        params.repeatStride = params.dataBlockStride * OFFSET_8;
        params.outDimOffset =
            AscendC::ONE_BLOCK_SIZE - params.innerExtend * params.repeatStride * AscendC::ONE_BLOCK_SIZE;
        params.outerStrideScale = CeilAlign(params.innerExtend * OFFSET_8, AscendC::BLOCK_CUBE);
        params.outerStrideWeight = bubKLen >> 1;
        params.maskWeight = bubKLen;
        params.offsetBaseAddr = offsetBaseAddr1_;
        params.scaleBaseAddr = scaleBaseAddr1_;
        params.weightInBaseAddr0 = weightInUbBaseAddr_;
        params.weightInBaseAddr1 = params.weightInBaseAddr0 + OFFSET_64;
        params.weightOutBaseAddr = weightOutUbAddr_;
        AscendC::VF_CALL<AntiquantW4Pergroup32NK<xType, wType, scaleType, hasAntiQuantOffset>>(params);
    } else {
        // not support yet
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, QuantType antiQuantType, typename scaleType, bool weightNz>
__aicore__ inline void
QuantBatchMatmulV4RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset, antiQuantType,
                                      scaleType, weightNz>::CopyVecOut2L1WeightNz(int64_t l1Offset,
                                                                                        LocalTensor<xType> ubLocal,
                                                                                        int32_t bubKLen,
                                                                                        int32_t bubNLen)
{
    DataCopyParams params;
    params.blockLen = BLOCK_NUM_REG;
    if (twoVectorCoreSplitN_) {
        // l1 (k1, 2*n1, n0, k0)
        // v0 (k1, n1, n0, k0)
        // v1 (k1, n1, n0, k0)
        if (bubNLen < bubKLen) {
            params.blockCount = (BLOCK_CUBE >> 1) * bubKLen * sizeof(xType) / VECTOR_REG_WIDTH;
            params.srcStride = (CeilDiv(vecNBL1Len_, BLOCK_CUBE) * 2 - 1) * BLOCK_NUM_REG;
            params.dstStride = (nBL1Len_ * BLOCK_CUBE - VECTOR_REG_WIDTH) / ONE_BLK_SIZE;
            for (int32_t idxN = 0; idxN < CeilDiv(bubNLen, BLOCK_CUBE >> 1); idxN++) {
                DataCopy(l1Local_[l1Offset], ubLocal, params);
                l1Offset += VECTOR_REG_WIDTH;
            }
        } else {
            params.blockCount = bubNLen * BLOCK_CUBE * sizeof(xType) / VECTOR_REG_WIDTH;
            params.srcStride = (tiling_->BL1Pingpong - 1) * BLOCK_NUM_REG;
            params.dstStride = 0;
            for (int32_t idxK = 0; idxK < CeilDiv(bubKLen, BLOCK_CUBE); idxK++) {
                DataCopy(l1Local_[l1Offset], ubLocal, params);
                l1Offset += (nBL1Len_ - bubNLen) * BLOCK_CUBE / ONE_BLK_SIZE;
            }
        }
    } else {
        // 一块 bubNLen * bubKLen一共有多少256B
        params.blockCount = bubNLen * bubKLen * sizeof(xType) / VECTOR_REG_WIDTH;
        // 每256B跳1024B(4buffer) / 512B(2buffer)
        params.srcStride = (tiling_->BL1Pingpong - 1) * BLOCK_NUM_REG;
        params.dstStride = 0;  // dst地址连续

        DataCopy(l1Local_[l1Offset], ubLocal, params);
    }
}
template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, QuantType antiQuantType, typename scaleType, bool weightNz>
__aicore__ inline void
QuantBatchMatmulV4RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset, antiQuantType,
                                      scaleType, weightNz>::CopyVecOut2L1(int64_t l1Offset,
                                                                                        LocalTensor<xType> ubLocal,
                                                                                        int32_t bubKLen,
                                                                                        int32_t bubNLen)
{
    DataCopyParams params;
    if constexpr (bTrans) {
        if constexpr (IsSameType<wType, fp4x2_e2m1_t>::value) {
            if constexpr (weightNz && (antiQuantType == QuantType::MX)) {
                CopyVecOut2L1WeightNz(l1Offset, ubLocal, bubKLen, bubNLen);
            } else {
                params.blockLen = bubNLen;
                params.blockCount = CeilDiv(bubKLen, BLOCK_CUBE * 2);
                params.srcStride = 1 + CeilAlign(bubNLen, BLOCK_CUBE) - bubNLen;
                params.dstStride = CeilAlign(nBL1Len_, BLOCK_CUBE) - bubNLen;
                DataCopy(l1Local_[l1Offset], ubLocal, params);
            }
        } else {
            // "only support transpose_weight=True in s8"
        }
    } else {  // B 矩阵非转置
        if constexpr (IsSameType<wType, int8_t>::value) {
            params.blockLen = bubKLen;
            params.blockCount = CeilDiv(bubNLen, BLOCK_CUBE);
            params.srcStride = 1;
            params.dstStride = kBL1Size_ - bubKLen;
            DataCopy(l1Local_[l1Offset], ubLocal, params);
        } else {  // A16W4, NZ 格式输入, (n1, k1, k0, n0)
            params.blockLen = BLOCK_NUM_REG;
            if (twoVectorCoreSplitK_) {
                // l1 (n1, 2*k1, k0, n0)
                // v0 (n1, k1, k0, n0)
                // v1 (n1, k1, k0, n0)
                if (bubKLen < bubNLen) {
                    params.blockCount = (BLOCK_CUBE >> 1) * bubNLen * sizeof(xType) / VECTOR_REG_WIDTH;
                    params.srcStride = (CeilDiv(vecKBL1Len_, BLOCK_CUBE) * 2 - 1) * BLOCK_NUM_REG;
                    params.dstStride = (kBL1Len_ * BLOCK_CUBE - VECTOR_REG_WIDTH) / ONE_BLK_SIZE;
                    for (int32_t idxK = 0; idxK < CeilDiv(bubKLen, BLOCK_CUBE >> 1); idxK++) {
                        DataCopy(l1Local_[l1Offset], ubLocal, params);
                        l1Offset += VECTOR_REG_WIDTH;
                    }
                } else {
                    params.blockCount = bubKLen * BLOCK_CUBE * sizeof(xType) / VECTOR_REG_WIDTH;
                    params.srcStride = (tiling_->BL1Pingpong - 1) * BLOCK_NUM_REG;
                    params.dstStride = 0;
                    for (int32_t idxN = 0; idxN < CeilDiv(bubNLen, BLOCK_CUBE); idxN++) {
                        DataCopy(l1Local_[l1Offset], ubLocal, params);
                        l1Offset += (kBL1Len_ - bubKLen) * BLOCK_CUBE / ONE_BLK_SIZE;
                    }
                }
            } else {
                // 一块 bubNLen * bubKLen一共有多少256B
                params.blockCount = bubNLen * bubKLen * sizeof(xType) / VECTOR_REG_WIDTH;
                // 每256B跳1024B(4buffer) / 512B(2buffer)
                params.srcStride = (tiling_->BL1Pingpong - 1) * BLOCK_NUM_REG;
                params.dstStride = 0;  // dst地址连续

                DataCopy(l1Local_[l1Offset], ubLocal, params);
            }
        }
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, QuantType antiQuantType, typename scaleType, bool weightNz>
__aicore__ inline void
QuantBatchMatmulV4RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset, antiQuantType,
                                      scaleType, weightNz>::MovWeightAndCompute(int64_t bubKOffset,
                                                                                              int64_t bubNOffset,
                                                                                              int32_t bubKLen,
                                                                                              int32_t bubNLen)
{
    CopyInTensorWeight(bubKOffset, bubNOffset, bubKLen, bubNLen);
    SetFlag<HardEvent::MTE2_V>(ubBufIdx_);
    if (idx_ >= tiling_->BL1Pingpong) {
        WaitFlag<HardEvent::MTE3_V>(ubBufIdx_);
    }
    WaitFlag<HardEvent::MTE2_V>(ubBufIdx_);
    AntiQuantCompute(bubKOffset, bubNLen, bubKLen);
    SetFlag<HardEvent::V_MTE3>(ubBufIdx_);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, QuantType antiQuantType, typename scaleType, bool weightNz>
__aicore__ inline void
QuantBatchMatmulV4RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset, antiQuantType,
                                      scaleType, weightNz>::BL1ProcessNK(uint64_t curBL1BufIdx,
                                                                                       int64_t nBL1Offset,
                                                                                       int64_t kBL1Offset,
                                                                                       int32_t kL1Len, int32_t nL0Len)
{
    int64_t bubNFactor = CeilDiv(vecNBL1Len_, tiling_->nBubSize);
    int64_t bubKFactor = CeilDiv(vecKBL1Len_, tiling_->kBubSize);
    for (int32_t bubNLoopIdx = 0; bubNLoopIdx < bubNFactor; bubNLoopIdx++) {
        int64_t vecNBL1Offset = bubNLoopIdx * tiling_->nBubSize;
        int64_t bubNOffset = nBL1Offset + vecNBL1Offset;
        int32_t bubNLen = Min(vecNBL1Len_ - vecNBL1Offset, tiling_->nBubSize);
        for (int32_t bubKLoopIdx = 0; bubKLoopIdx < bubKFactor; bubKLoopIdx++) {
            int64_t vecKBL1Offset = bubKLoopIdx * tiling_->kBubSize;
            int64_t bubKOffset = kBL1Offset + vecKBL1Offset;

            idx_ += 1;
            ubBufIdx_ = idx_ % vecPingpong_;
            if (idx_ >= vecPingpong_) {
                WaitFlag<HardEvent::V_MTE2>(ubBufIdx_);
            }

            int32_t bubKLen = Min(vecKBL1Len_ - vecKBL1Offset, tiling_->kBubSize);
            groupNumBub_ = GetVecGn(bubKOffset, bubKLen);
            CopyInScaleOffset(bubNOffset, bubNLen, bubNLoopIdx, bubKLoopIdx, bubKOffset, bubKFactor);
            MovWeightAndCompute(bubKOffset, bubNOffset, bubKLen, bubNLen);

            SetFlag<HardEvent::V_MTE2>(ubBufIdx_);
            uint64_t weightOutUbOffset = ubBufIdx_ * (vecWeightOutLen_ / sizeof(xType) / vecPingpong_);
            WaitFlag<HardEvent::V_MTE3>(ubBufIdx_);
            int64_t kl1Offset = vecKBL1Offset;
            if (AscendC::GetSubBlockIdx() == 1 && tiling_->vecCoreParallel == 1) {
                kl1Offset += tiling_->kBubSize;
            }
            int64_t l1Offset;
            if constexpr (antiQuantType == QuantType::MX) {
                l1Offset = (curBL1BufIdx & 0x1) * L1_BUFFER_HALF_SIZE / sizeof(xType) + nBL1Len_ * kl1Offset +
                           vecNBL1Offset * BLOCK_CUBE * 2;
            } else {
                l1Offset = nBL1Len_ * kl1Offset + vecNBL1Offset * BLOCK_CUBE * 2 + curBL1BufIdx * bL1DataSize_;
            }
            if constexpr (weightNz && (antiQuantType == QuantType::MX)) {
                CopyVecOut2L1(l1Offset, weightOutUb_[ubBufIdx_ * VEC_MAX_ELEM_B8], bubKLen, bubNLen);
            } else {
                CopyVecOut2L1(l1Offset, weightOutUb_[weightOutUbOffset], bubKLen, bubNLen);
            }
            SetFlag<HardEvent::MTE3_V>(ubBufIdx_);
        }
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, QuantType antiQuantType, typename scaleType, bool weightNz>
__aicore__ inline void
QuantBatchMatmulV4RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset, antiQuantType,
                                      scaleType, weightNz>::BL1ProcessNK4Buffer(uint64_t curBL1BufIdx,
                                                                                              int64_t nBL1Offset,
                                                                                              int64_t kBL1Offset,
                                                                                              int32_t kL1Len,
                                                                                              int32_t nL0Len)
{
    int64_t l1Offset =
        (curBL1BufIdx & 0x1) * Max(L1_BUFFER_HALF_SIZE / sizeof(xType), DOUBLE_BUFFER * bL1DataSize_ + aL1DataSize_) +
        ((curBL1BufIdx & 0x2) > 1) * bL1DataSize_;

    idx_ += 1;
    ubBufIdx_ = idx_ & (tiling_->BL1Pingpong - 1);
    if (idx_ >= tiling_->BL1Pingpong) {
        WaitFlag<HardEvent::V_MTE2>(ubBufIdx_);
    }
    int32_t bubKLen = Min(vecKBL1Len_, tiling_->kBubSize);
    int32_t bubNLen = Min(vecNBL1Len_, tiling_->nBubSize);
    groupNumBub_ = GetVecGn(kBL1Offset, bubKLen);
    CopyInScaleOffset(nBL1Offset, bubNLen, 0, 0, kBL1Offset, 1);
    MovWeightAndCompute(kBL1Offset, nBL1Offset, bubKLen, bubNLen);
    SetFlag<HardEvent::V_MTE2>(ubBufIdx_);
    uint64_t weightOutUbOffset = ubBufIdx_ * (vecWeightOutLen_ / sizeof(xType) / tiling_->BL1Pingpong);
    WaitFlag<HardEvent::V_MTE3>(ubBufIdx_);
    int64_t nl1Offset = 0;
    int64_t kl1Offset = 0;
    if (AscendC::GetSubBlockIdx() == 1 && twoVectorCoreSplitK_ && kBL1Len_ > bubKLen) {
        // 存在尾块kBL1Len_= bubKLen的情况，此时两个vec核处理同一块BL1buffer,往同一个地址搬运
        kl1Offset += CeilDiv(CeilDiv(kBL1Len_, tiling_->kBubSize), AIV_NUM) * tiling_->kBubSize;
    } else if (AscendC::GetSubBlockIdx() == 1 && twoVectorCoreSplitN_ && nBL1Len_ > bubNLen) {
        nl1Offset += CeilDiv(CeilDiv(nBL1Len_, tiling_->nBubSize), AIV_NUM) * tiling_->nBubSize;
    }
    l1Offset += nl1Offset * ONE_BLK_SIZE + kl1Offset * CeilAlign(nBL1Len_, BLOCK_CUBE);
    if constexpr (weightNz && (antiQuantType == QuantType::MX)) {
        CopyVecOut2L1(l1Offset, weightOutUb_[ubBufIdx_ * VEC_MAX_ELEM_B8], bubKLen, bubNLen);
    } else {
        CopyVecOut2L1(l1Offset, weightOutUb_[weightOutUbOffset], bubKLen, bubNLen);
    }
    SetFlag<HardEvent::MTE3_V>(ubBufIdx_);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, QuantType antiQuantType, typename scaleType, bool weightNz>
__aicore__ inline void
QuantBatchMatmulV4RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset, antiQuantType,
                                      scaleType, weightNz>::BL1ProcessKN(uint64_t curBL1BufIdx,
                                                                                       int64_t nBL1Offset,
                                                                                       int64_t kBL1Offset,
                                                                                       int32_t kL1Len, int32_t nL0Len)
{
    // l1 space: bp0 bp1
    int64_t l1Offset;
    if constexpr (antiQuantType == QuantType::MX) {
        l1Offset = curBL1BufIdx * bL1DataSize_;
    } else {
        l1Offset = (curBL1BufIdx & 0x1) *
                       Max(L1_BUFFER_HALF_SIZE / sizeof(xType), DOUBLE_BUFFER * bL1DataSize_ + aL1DataSize_) +
                   ((curBL1BufIdx & 0x2) > 1) * bL1DataSize_;
    }
    // AIV-0 / AIV-1 上一块 UB buffer 对应于一块 BL1 buffer 的搬运与计算过程
    int32_t bubKLen = Min(vecKBL1Len_, tiling_->kBubSize);
    groupNumBub_ = GetVecGn(kBL1Offset, bubKLen);
    idx_ += 1;
    ubBufIdx_ = idx_ & (tiling_->BL1Pingpong - 1);
    if (idx_ >= tiling_->BL1Pingpong) {
        WaitFlag<HardEvent::V_MTE2>(ubBufIdx_);
    }
    int32_t bubNLen = Min(vecNBL1Len_, tiling_->nBubSize);
    CopyInScaleOffset(nBL1Offset, bubNLen, 0, 0, kBL1Offset, 1);
    MovWeightAndCompute(kBL1Offset, nBL1Offset, bubKLen, bubNLen);


    SetFlag<HardEvent::V_MTE2>(ubBufIdx_);
    WaitFlag<HardEvent::V_MTE3>(ubBufIdx_);

    int64_t nl1Offset = 0;
    int64_t kl1Offset = 0;
    if (AscendC::GetSubBlockIdx() == 1 && twoVectorCoreSplitK_ && kBL1Len_ > bubKLen) {
        // 存在尾块kBL1Len_= bubKLen的情况，此时两个vec核处理同一块BL1buffer,往同一个地址搬运
        kl1Offset += CeilDiv(CeilDiv(kBL1Len_, tiling_->kBubSize), AIV_NUM) * tiling_->kBubSize;
    } else if (AscendC::GetSubBlockIdx() == 1 && twoVectorCoreSplitN_ && nBL1Len_ > bubNLen) {
        nl1Offset += CeilDiv(CeilDiv(nBL1Len_, tiling_->nBubSize), AIV_NUM) * tiling_->nBubSize;
    }
    l1Offset += nl1Offset * CeilAlign(kBL1Len_, BLOCK_CUBE) + kl1Offset * BLOCK_CUBE;
    CopyVecOut2L1(l1Offset, weightOutUb_[ubBufIdx_ * VECTOR_REG_WIDTH], bubKLen, bubNLen);
    SetFlag<HardEvent::MTE3_V>(ubBufIdx_);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, QuantType antiQuantType, typename scaleType, bool weightNz>
__aicore__ inline void
QuantBatchMatmulV4RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset, antiQuantType,
                                      scaleType, weightNz>::BL1Process(uint64_t curBL1BufIdx,
                                                                                     int64_t nBL1Offset,
                                                                                     int64_t kBL1Offset, int32_t kL1Len,
                                                                                     int32_t nL0Len)
{
    if constexpr (bTrans) {
        if (tiling_->BL1Pingpong == QUADRUPLE_BUFFER) {
            BL1ProcessNK4Buffer(curBL1BufIdx, nBL1Offset, kBL1Offset, kL1Len, nL0Len);
        } else {
            BL1ProcessNK(curBL1BufIdx, nBL1Offset, kBL1Offset, kL1Len, nL0Len);
        }
    } else {
        BL1ProcessKN(curBL1BufIdx, nBL1Offset, kBL1Offset, kL1Len, nL0Len);
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, QuantType antiQuantType, typename scaleType, bool weightNz>
__aicore__ inline void
QuantBatchMatmulV4RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset, antiQuantType,
                                      scaleType, weightNz>::UpdateIdxAndStep()
{
    curML0Idx_ = curML1Idx_;
    curNL0Idx_ = curNL1Idx_;
    curStepM_ =
        (mIter_ - curML0Idx_) > tiling_->matmulTiling.stepM ? tiling_->matmulTiling.stepM : (mIter_ - curML1Idx_);
    curStepN_ =
        (nIter_ - curNL0Idx_) > tiling_->matmulTiling.stepN ? tiling_->matmulTiling.stepN : (nIter_ - curNL1Idx_);
}

/*
 * 功能：该函数作用为通过每次移动一个 baseM 或 baseN，并更新当前L0和L1的index以及对应当前的使用大小（包含尾块）
 */
template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, QuantType antiQuantType, typename scaleType, bool weightNz>
__aicore__ inline bool
QuantBatchMatmulV4RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset, antiQuantType,
                                      scaleType, weightNz>::IterMatmulOut()
{
    if (unlikely(isFirstIter_)) {
        curML0Idx_ = 0;
        curNL0Idx_ = 0;
        curML1Idx_ = 0;
        curNL1Idx_ = 0;
        isFirstIter_ = false;
        // 考虑尾核
        curStepM_ =
            (mIter_ - curML0Idx_) > tiling_->matmulTiling.stepM ? tiling_->matmulTiling.stepM : (mIter_ - curML1Idx_);
        curStepN_ =
            (nIter_ - curNL0Idx_) > tiling_->matmulTiling.stepN ? tiling_->matmulTiling.stepN : (nIter_ - curNL1Idx_);
    } else if (likely(tiling_->matmulTiling.iterateOrder == static_cast<int>(IterateOrder::ORDER_N))) {
        if (++curML0Idx_ >= curML1Idx_ + curStepM_) {
            curML0Idx_ = curML1Idx_;
            if (++curNL0Idx_ >= curNL1Idx_ + curStepN_) {
                curML1Idx_ += curStepM_;
                if (curNL0Idx_ >= nIter_ && curML1Idx_ >= mIter_) {
                    return false;
                }
                if (curML1Idx_ >= mIter_) {
                    curML1Idx_ = 0;
                    curNL1Idx_ += curStepN_;
                }
                UpdateIdxAndStep();
            }
        }
    } else {
        if (++curNL0Idx_ >= curNL1Idx_ + curStepN_) {
            curNL0Idx_ = curNL1Idx_;
            if (++curML0Idx_ >= curML1Idx_ + curStepM_) {
                curNL1Idx_ += curStepN_;
                if (curML0Idx_ >= mIter_ && curNL1Idx_ >= nIter_) {
                    return false;
                }
                if (curNL1Idx_ >= nIter_) {
                    curNL1Idx_ = 0;
                    curML1Idx_ += curStepM_;
                }
                UpdateIdxAndStep();
            }
        }
    }

    baseUseM_ = (curML0Idx_ + 1 == mIter_) ? tailM_ : tiling_->matmulTiling.baseM;
    baseUseN_ = (curNL0Idx_ + 1 == nIter_) ? tailN_ : tiling_->matmulTiling.baseN;
    mAL1Len_ = (curML1Idx_ + curStepM_ >= mIter_) ? tailL1M_ : mAL1Size_;
    nBL1Len_ = (curNL1Idx_ + curStepN_ >= nIter_) ? tailL1N_ : nBL1Size_;
    return true;
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, QuantType antiQuantType, typename scaleType, bool weightNz>
__aicore__ inline int64_t
QuantBatchMatmulV4RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset, antiQuantType,
                                      scaleType, weightNz>::CeilDiv(int64_t x, int64_t y)
{
#ifdef __CCE_KT_TEST__
    ASCENDC_ASSERT(y != 0, { KERNEL_LOG(KERNEL_ERROR, "divide 0: x(%ld) y(%ld)", x, y); });
#endif
    if (y == 0) {
        return 0;
    }
    return (x + y - 1) / y;
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, QuantType antiQuantType, typename scaleType, bool weightNz>
__aicore__ inline int64_t
QuantBatchMatmulV4RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset, antiQuantType,
                                      scaleType, weightNz>::CeilAlign(int64_t x, int64_t y)
{
#ifdef __CCE_KT_TEST__
    ASCENDC_ASSERT(y != 0, { KERNEL_LOG(KERNEL_ERROR, "divide 0: x(%ld) y(%ld)", x, y); });
#endif
    if (y == 0) {
        return 0;
    }
    return (x + y - 1) / y * y;
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, QuantType antiQuantType, typename scaleType, bool weightNz>
__aicore__ inline int64_t
QuantBatchMatmulV4RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset, antiQuantType,
                                      scaleType, weightNz>::FloorAlign(int64_t x, int64_t y)
{
#ifdef __CCE_KT_TEST__
    ASCENDC_ASSERT(y != 0, { KERNEL_LOG(KERNEL_ERROR, "divide 0: x(%ld) y(%ld)", x, y); });
#endif
    if (y == 0) {
        return 0;
    }
    return x / y * y;
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, QuantType antiQuantType, typename scaleType, bool weightNz>
__aicore__ inline int64_t
QuantBatchMatmulV4RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset, antiQuantType,
                                      scaleType, weightNz>::GetVecGn(int32_t bubKOffset, int32_t bubKLen)
{
    if constexpr (antiQuantType == QuantType::PER_GROUP || antiQuantType == QuantType::MX) {
        return CeilDiv(bubKLen, tiling_->groupSize);
    } else {
        return 1;
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, QuantType antiQuantType, typename scaleType, bool weightNz>
__aicore__ inline int32_t
QuantBatchMatmulV4RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset, antiQuantType,
                                      scaleType, weightNz>::Min(int32_t a, int32_t b)
{
    return a > b ? b : a;
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, QuantType antiQuantType, typename scaleType, bool weightNz>
__aicore__ inline int32_t
QuantBatchMatmulV4RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset, antiQuantType,
                                      scaleType, weightNz>::Max(int32_t a, int32_t b)
{
    return a > b ? a : b;
}
template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, QuantType antiQuantType, typename scaleType, bool weightNz>
__aicore__ inline void
QuantBatchMatmulV4RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset, antiQuantType,
                                      scaleType, weightNz>::TwoVectorCoreSplit()
{
    uint64_t nBubFactor = CeilDiv(nBL1Len_, tiling_->nBubSize);
    uint64_t kBubFactor = CeilDiv(kBL1Len_, tiling_->kBubSize);
    if constexpr (weightNz && (antiQuantType == QuantType::MX)) {
        if (kBubFactor > 1) { // 优先在K方向切分
            twoVectorCoreSplitK_ = true;
            vecKBL1Len_ = CeilDiv(kBubFactor, AIV_NUM) * tiling_->kBubSize;
            if (AscendC::GetSubBlockIdx() == 1) {
                kBL1Offset_ += vecKBL1Len_;
                vecKBL1Len_ = Min(tiling_->kSize - kBL1Offset_, kBL1Len_ - vecKBL1Len_);
            } else {
                vecKBL1Len_ = Min(tiling_->kSize - kBL1Offset_, vecKBL1Len_);
            }
        } else if (nBubFactor > 1) {  // 优先在N方向切分
            twoVectorCoreSplitN_ = true;
            vecNBL1Len_ = CeilDiv(nBubFactor, AIV_NUM) * tiling_->nBubSize;
            if (AscendC::GetSubBlockIdx() == 1) {
                nBL1Offset_ += vecNBL1Len_;
                vecNBL1Len_ = Min(tiling_->nSize - nBL1Offset_, nBL1Len_ - vecNBL1Len_);
            } else {
                vecNBL1Len_ = Min(tiling_->nSize - nBL1Offset_, vecNBL1Len_);
            }
        } else {  // 仅使用 AIV-0 核
#ifdef __CCE_KT_TEST__
            ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "just use a single aiv core is not supported yet"); });
#endif
        }
    } else {
        if (nBubFactor > 1) {  // 优先在N方向切分
            twoVectorCoreSplitN_ = true;
            vecNBL1Len_ = CeilDiv(nBubFactor, AIV_NUM) * tiling_->nBubSize;
            if (AscendC::GetSubBlockIdx() == 1) {
                nBL1Offset_ += vecNBL1Len_;
                vecNBL1Len_ = Min(tiling_->nSize - nBL1Offset_, nBL1Len_ - vecNBL1Len_);
            } else {
                vecNBL1Len_ = Min(tiling_->nSize - nBL1Offset_, vecNBL1Len_);
            }
        } else if (kBubFactor > 1) {
            twoVectorCoreSplitK_ = true;
            vecKBL1Len_ = CeilDiv(kBubFactor, AIV_NUM) * tiling_->kBubSize;
            if (AscendC::GetSubBlockIdx() == 1) {
                kBL1Offset_ += vecKBL1Len_;
                vecKBL1Len_ = Min(tiling_->kSize - kBL1Offset_, kBL1Len_ - vecKBL1Len_);
            } else {
                vecKBL1Len_ = Min(tiling_->kSize - kBL1Offset_, vecKBL1Len_);
            }
        } else {  // 仅使用 AIV-0 核
#ifdef __CCE_KT_TEST__
            ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "just use a single aiv core is not supported yet"); });
#endif
        }
    }
}
/*
 * 功能：从 UB 上获取一块 BL1 buffer 的数据
 */
template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, QuantType antiQuantType, typename scaleType, bool weightNz>
__aicore__ inline void
QuantBatchMatmulV4RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset, antiQuantType,
                                      scaleType, weightNz>::CopyUb2L1()
{
    vecKBL1Len_ = kBL1Len_;
    vecNBL1Len_ = nBL1Len_;
    if (likely(tiling_->BL1Pingpong == QUADRUPLE_BUFFER)) {
        TwoVectorCoreSplit();
        VectorProcess();
    } else if (tiling_->BL1Pingpong == DOUBLE_BUFFER) {
        // curBL1BufIdx_ == 0 && AscendC::GetSubBlockIdx() == 0 或者
        // curBL1BufIdx_ == 1 && AscendC::GetSubBlockIdx() == 1 时进入 VectorProcess 函数
        if (curBL1BufIdx_ == AscendC::GetSubBlockIdx()) {
            VectorProcess();
        }
    } else if (tiling_->vecCoreParallel == 1) {
        if (AscendC::GetSubBlockIdx() == 1) {
            kBL1Offset_ += tiling_->kBubSize;
            vecKBL1Len_ = Min(tiling_->kSize - kBL1Offset_, kBL1Size_);  // AIV-1 需要对应去搬运的 kBL1 大小
        } else {
            vecKBL1Len_ = Min(tiling_->kSize - kBL1Offset_, tiling_->kBubSize);  // AIV-0 需要对应去搬运的 kBL1 大小
        }
        VectorProcess();
    } else if (AscendC::GetSubBlockIdx() == 0) {
        VectorProcess();
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, QuantType antiQuantType, typename scaleType, bool weightNz>
__aicore__ inline void
QuantBatchMatmulV4RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset, antiQuantType,
                                      scaleType, weightNz>::VectorProcess()
{
    WaitForCube();
    BL1Process(curBL1BufIdx_, nBL1Offset_, kBL1Offset_, kBL1Len_, baseUseN_);
    NotifyCube();
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, QuantType antiQuantType, typename scaleType, bool weightNz>
__aicore__ inline void QuantBatchMatmulV4RegBaseCommonKernel<
    xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset, antiQuantType, scaleType,
    weightNz>::PostProcess(int32_t kFactorIdx, AscendC::TEventID eventIdsMte1ToMte2[MAX_AL1_BUF_NUM])
{
    if ASCEND_IS_AIC {
        if ((kFactorIdx + 1) % kBl1Factor_ == 0 || (kFactorIdx + 1) == kSingleCoreIterNum_) {
            NotifyVector(curBL1BufIdx_);
            curBL1BufIdx_ = (curBL1BufIdx_ + 1) % tiling_->BL1Pingpong;
        }
        // AL1
        // k非全载时，每次跨kAl1Factor_前一拍做同步，考虑mk合轴后开db时kAL1可能存在尾块，因此在最后一次k循环也要加同步
        // AL1 k全载时，仅当完成一次n方向iter后做同步
        if ((kFactorIdx + 1) % kAl1Factor_ == 0 || (kFactorIdx + 1) == kSingleCoreIterNum_) {
            SetFlag<HardEvent::MTE1_MTE2>(eventIdsMte1ToMte2[curAL1BufIdx_]);
            curAL1BufIdx_ = (curAL1BufIdx_ + 1) % tiling_->AL1Pingpong;
        }
        if constexpr (antiQuantType == QuantType::MX) {
            if (tiling_->matmulTiling.isBias && kFactorIdx == kSingleCoreIterNum_ - 1) {
                biasIdx_ = (biasIdx_ + 1) % biasPingPong_;
            }
            if ((kFactorIdx + 1) % scaleAFactor_ == 0 || (kFactorIdx + 1) == kSingleCoreIterNum_) {
                SetFlag<HardEvent::MTE1_MTE2>(eventIdsScaleAMte1ToMte2_[scaleAIdx_]);
                scaleAIdx_ = (scaleAIdx_ + 1) % DOUBLE_BUFFER;
            }
            if ((kFactorIdx + 1) % scaleBFactor_ == 0 || (kFactorIdx + 1) == kSingleCoreIterNum_) {
                SetFlag<HardEvent::MTE1_MTE2>(eventIdsScaleBMte1ToMte2_[scaleBIdx_]);
                scaleBIdx_ = (scaleBIdx_ + 1) % DOUBLE_BUFFER;
            }
        }
    }
    if ASCEND_IS_AIV {
        if ((kFactorIdx + 1) % kBl1Factor_ == 0 || (kFactorIdx + 1) == kSingleCoreIterNum_) {
            curBL1BufIdx_ = (curBL1BufIdx_ + 1) % tiling_->BL1Pingpong;
        }
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, QuantType antiQuantType, typename scaleType, bool weightNz>
__aicore__ inline void QuantBatchMatmulV4RegBaseCommonKernel<
    xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset, antiQuantType, scaleType,
    weightNz>::InitSync(AscendC::TEventID eventIdsMte1ToMte2[MAX_AL1_BUF_NUM])
{
    if ASCEND_IS_AIC {
        for (int32_t idx = 0; idx < tiling_->BL1Pingpong; idx++) {
            NotifyVector(idx);
        }

        for (int32_t idx = 0; idx < tiling_->AL1Pingpong; idx++) {
            eventIdsMte1ToMte2[idx] = GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>();
            SetFlag<HardEvent::MTE1_MTE2>(eventIdsMte1ToMte2[idx]);
        }
        if constexpr (antiQuantType == QuantType::MX) {
            eventIdsScaleAMte1ToMte2_[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>();
            eventIdsScaleAMte1ToMte2_[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>();
            eventIdsScaleBMte1ToMte2_[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>();
            eventIdsScaleBMte1ToMte2_[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>();
            SetFlag<HardEvent::MTE1_MTE2>(eventIdsScaleAMte1ToMte2_[0]);
            SetFlag<HardEvent::MTE1_MTE2>(eventIdsScaleAMte1ToMte2_[1]);
            SetFlag<HardEvent::MTE1_MTE2>(eventIdsScaleBMte1ToMte2_[0]);
            SetFlag<HardEvent::MTE1_MTE2>(eventIdsScaleBMte1ToMte2_[1]);
        }
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, QuantType antiQuantType, typename scaleType, bool weightNz>
__aicore__ inline void QuantBatchMatmulV4RegBaseCommonKernel<
    xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset, antiQuantType, scaleType,
    weightNz>::EndSync(AscendC::TEventID eventIdsMte1ToMte2[MAX_AL1_BUF_NUM])
{
    if ASCEND_IS_AIV {
        if (tiling_->BL1Pingpong == QUADRUPLE_BUFFER) {
            int8_t buffNum = Min(nIter_ * tiling_->kSize / tiling_->kBubSize, tiling_->BL1Pingpong);
            for (int8_t idx = 0; idx < buffNum; idx++) {
                WaitFlag<HardEvent::V_MTE2>(idx);
                WaitFlag<HardEvent::MTE3_V>(idx);
            }
        } else {
            if (idx_ > 0) {
                for (int8_t idx = 0; idx < vecPingpong_; idx++) {
                    WaitFlag<HardEvent::V_MTE2>(idx);
                    WaitFlag<HardEvent::MTE3_V>(idx);
                }
            } else if (idx_ == 0) {
                WaitFlag<HardEvent::V_MTE2>(0);
                WaitFlag<HardEvent::MTE3_V>(0);
            }
        }

        if (tiling_->BL1Pingpong == QUADRUPLE_BUFFER) {
            WaitForCube();
            WaitForCube();
            WaitForCube();
            WaitForCube();
            return;
        }

        if ((tiling_->BL1Pingpong == 1) && (tiling_->vecCoreParallel == 0)) {
            if (AscendC::GetSubBlockIdx() == 0) {
                WaitForCube();
            }
            return;
        }

        if ((tiling_->BL1Pingpong == 1) && (tiling_->vecCoreParallel == 1)) {
            WaitForCube();
            return;
        }

        if (tiling_->BL1Pingpong == DOUBLE_BUFFER) {
            WaitForCube();
            return;
        }
    } else {
        for (int8_t idx = 0; idx < tiling_->AL1Pingpong; idx++) {
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_MTE2>(eventIdsMte1ToMte2[idx]);
        }
        if constexpr (antiQuantType == QuantType::MX) {
            WaitFlag<HardEvent::MTE1_MTE2>(eventIdsScaleAMte1ToMte2_[0]);
            WaitFlag<HardEvent::MTE1_MTE2>(eventIdsScaleAMte1ToMte2_[1]);
            WaitFlag<HardEvent::MTE1_MTE2>(eventIdsScaleBMte1ToMte2_[0]);
            WaitFlag<HardEvent::MTE1_MTE2>(eventIdsScaleBMte1ToMte2_[1]);
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_MTE2>(eventIdsScaleAMte1ToMte2_[0]);
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_MTE2>(eventIdsScaleAMte1ToMte2_[1]);
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_MTE2>(eventIdsScaleBMte1ToMte2_[0]);
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_MTE2>(eventIdsScaleBMte1ToMte2_[1]);
        }
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, QuantType antiQuantType, typename scaleType, bool weightNz>
__aicore__ inline void
QuantBatchMatmulV4RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset, antiQuantType,
                                      scaleType, weightNz>::AntiQuantComputeNormal(int32_t bubKOffset,
                                                                                                 int32_t bubNLen,
                                                                                                 int32_t bubKLen)
{
    static_assert(SupportType<wType, fp4x2_e2m1_t>(), "only support fp4x2_e2m1_t");

    uint16_t outExtend;
    uint32_t scaleExtend;
    uint32_t mainGroupNum = bubKLen / this->tiling_->groupSize;
    if constexpr (bTrans) {
        outExtend = static_cast<uint16_t>(bubNLen);
        if constexpr (antiQuantType == QuantType::PER_GROUP || antiQuantType == QuantType::MX) {
            scaleExtend = CeilAlign(groupNumBub_, ONE_BLK_ELEM_B16);
        } else {
            scaleExtend = ONE_BLK_ELEM_B16;
        }
        ComputeAntiquantParam(bubKLen, bubKOffset, bubNLen);
    } else {
        outExtend = static_cast<uint16_t>(bubKLen);
        scaleExtend = VECTOR_REG_WIDTH;
        ComputeAntiquantParam(bubNLen, bubKOffset, bubKLen);
    }
    uint16_t innerExtend = CeilDiv(CeilAlign(bubKLen, UB_ALIGN_SIZE_FOR_4BITS), VECTOR_REG_WIDTH_FOR_4BITS);
    uint32_t dataBlockStride = CeilAlign(bubNLen, BLOCK_CUBE) + 1;
    uint32_t repeatStride = dataBlockStride * BLOCK_CUBE;
    int32_t outDimOffset = AscendC::ONE_BLOCK_SIZE - innerExtend * repeatStride * AscendC::ONE_BLOCK_SIZE;
    int32_t vsstbConfig = (dataBlockStride << BLOCK_CUBE) | (repeatStride & MASK_16_BITS);
    uint32_t maskB8Tail0 = 0;
    uint32_t maskB8Tail1 = 0;
    uint32_t maskWeight0Tmp = 0;
    uint32_t maskWeight1Tmp = 0;
    MicroAPI::MaskReg MaskRegB8Tail0;
    MicroAPI::MaskReg MaskRegB8Tail1;
    maskB8Tail0 = Min(bubKLen % VECTOR_REG_WIDTH_FOR_4BITS, VECTOR_REG_WIDTH) +
                  bubKLen / VECTOR_REG_WIDTH_FOR_4BITS * VECTOR_REG_WIDTH;
    maskB8Tail1 = Max(bubKLen % VECTOR_REG_WIDTH_FOR_4BITS - VECTOR_REG_WIDTH, 0) +
                  bubKLen / VECTOR_REG_WIDTH_FOR_4BITS * VECTOR_REG_WIDTH;

    __local_mem__ int8_t* weightInUbBaseAddr = weightInUbBaseAddr_;
    __local_mem__ xType* weightOutUbAddr = weightOutUbAddr_;
    __local_mem__ xType* weightOutUbAddr1 = weightOutUbAddr1_;

#ifndef __CCE_KT_TEST__
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<uint8_t> wDIntlv0, wDIntlv1, wLoad0, wLoad1, wLoad2, wLoad3, sAnd0, sAnd1, wShr, wShl, s1,
            wOr0, wOr1, wdup1, wdup4;
        MicroAPI::RegTensor<int8_t> wdup0, wdup2, wdup3;
        MicroAPI::MaskReg preg = MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();
        MicroAPI::Duplicate<int8_t, AscendC::MicroAPI::MaskMergeMode::ZEROING>(wdup0, DUP_CONFIG_2, preg);
        MicroAPI::Duplicate<uint8_t, AscendC::MicroAPI::MaskMergeMode::ZEROING>(wdup1, DUP_CONFIG_MODE_1C, preg);
        MicroAPI::Duplicate<int8_t, AscendC::MicroAPI::MaskMergeMode::ZEROING>(wdup2, DUP_CONFIG_2, preg);
        MicroAPI::Duplicate<int8_t, AscendC::MicroAPI::MaskMergeMode::ZEROING>(wdup3, DUP_CONFIG_4, preg);
        MicroAPI::Duplicate<uint8_t, AscendC::MicroAPI::MaskMergeMode::ZEROING>(wdup4, DUP_FLAG_80, preg);
        // 一次处理一个N轴
        for (uint16_t outIdx = 0; outIdx < outExtend; ++outIdx) {
            // 一次处理K轴512个element groupSize=32,需要加载16个Scale
            maskWeight0Tmp = maskB8Tail0;
            maskWeight1Tmp = maskB8Tail1;
            for (uint16_t repeatIdx = 0; repeatIdx < innerExtend; ++repeatIdx) {
                MaskRegB8Tail0 = MicroAPI::UpdateMask<uint8_t>(maskWeight0Tmp);
                MaskRegB8Tail1 = MicroAPI::UpdateMask<uint8_t>(maskWeight1Tmp);
                MicroAPI::AddrReg aregWeightB8 =
                    MicroAPI::CreateAddrReg<uint8_t>(outIdx, bubKLen >> 1, repeatIdx, VEC_MAX_ELEM_B8);
                MicroAPI::DataCopy(wLoad0, (__local_mem__ uint8_t*&)weightInUbBaseAddr, aregWeightB8);
                // 提取E/M
                MicroAPI::ShiftRight(wShr, wLoad0, wdup0, preg);  // vr1
                MicroAPI::And(wShr, wShr, wdup1, preg);           // vr1
                MicroAPI::ShiftLeft(wShl, wLoad0, wdup2, preg);   // vr2
                MicroAPI::And(wShl, wShl, wdup1, preg);           // vr2
                // 提取S
                MicroAPI::ShiftLeft(s1, wLoad0, wdup3, preg);  // vr3
                MicroAPI::And(sAnd0, s1, wdup4, preg);         // vr3
                MicroAPI::And(sAnd1, wLoad0, wdup4, preg);     // vr4
                // 合并S/E/M
                MicroAPI::Or(wOr0, wShr, sAnd1, preg);  // odd
                MicroAPI::Or(wOr1, wShl, sAnd0, preg);  // even
                MicroAPI::Interleave(wDIntlv0, wDIntlv1, wOr1, wOr0);
                MicroAPI::DataCopy<uint8_t, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                                   MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    (__local_mem__ uint8_t*&)weightOutUbAddr, wDIntlv0, dataBlockStride, repeatStride, MaskRegB8Tail0);
                MicroAPI::DataCopy<uint8_t, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                                   MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    (__local_mem__ uint8_t*&)weightOutUbAddr1, wDIntlv1, dataBlockStride, repeatStride, MaskRegB8Tail1);
            }
            weightOutUbAddr += outDimOffset;
            weightOutUbAddr1 += outDimOffset;
        }
    }
#endif
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, QuantType antiQuantType, typename scaleType, bool weightNz>
__aicore__ inline void
QuantBatchMatmulV4RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset, antiQuantType,
                                      scaleType, weightNz>::AntiQuantComputeNKMxNz(int32_t bubKOffset,
                                                                                                 int32_t bubNLen,
                                                                                                 int32_t bubKLen)
{
    static_assert(SupportType<wType, fp4x2_e2m1_t>(), "only support fp4x2_e2m1_t");
    uint32_t shiftLeftSize = 0;
    uint32_t andMask = 0;
    if constexpr (IsSameType<wType, fp4x2_e2m1_t>::value) {
        shiftLeftSize = E2M1_SHIFT_LEFT_SIZE;
        andMask = E2M1_AND_MASK;
    }
    uint16_t innerExtend = CeilDiv(bubKLen * bubNLen, VECTOR_REG_WIDTH);
    uint32_t maskB8Tail0 = 0;
    uint32_t maskB8Tail1 = 0;
    uint32_t maskWeight0Tmp = 0;
    uint32_t innerDstExtend = VECTOR_REG_WIDTH * tiling_->BL1Pingpong;
    uint32_t innerSrcExtend = VECTOR_REG_WIDTH >> 1;
    MicroAPI::MaskReg MaskRegB8Tail0;
    maskB8Tail0 = bubKLen;
    __local_mem__ int8_t* weightInUbBaseAddr = weightInUbBaseAddr_;
    __local_mem__ xType* weightOutUbAddr = weightOutUbAddr_;
    __local_mem__ xType* weightOutUbAddr1 = weightOutUbAddr1_;

#ifndef __CCE_KT_TEST__
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<int8_t> wdup0, wdup1, wdup2, wLoad0, wShl, wShr0, wShr1, wSel0, sAnd0;
        MicroAPI::MaskReg preg = MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg pregVsel = MicroAPI::CreateMask<uint16_t, AscendC::MicroAPI::MaskPattern::ALL>();

        MicroAPI::Duplicate<int8_t, AscendC::MicroAPI::MaskMergeMode::ZEROING>(wdup0, shiftLeftSize, preg);
        MicroAPI::Duplicate<int8_t, AscendC::MicroAPI::MaskMergeMode::ZEROING>(wdup1, SHIFT_RIGHT_SIZE, preg);
        MicroAPI::Duplicate<int8_t, AscendC::MicroAPI::MaskMergeMode::ZEROING>(wdup2, andMask, preg);

        // 一次处理一个N轴
        for (uint16_t repeatIdx = 0; repeatIdx < innerExtend; ++repeatIdx) {
            MicroAPI::AddrReg aregWeightB8In = MicroAPI::CreateAddrReg<uint8_t>(repeatIdx, innerSrcExtend);
            MicroAPI::AddrReg aregWeightB8Out = MicroAPI::CreateAddrReg<uint8_t>(repeatIdx, innerDstExtend);
            MicroAPI::DataCopy<uint8_t, MicroAPI::LoadDist::DIST_US_B8>(
                (MicroAPI::RegTensor<uint8_t>&)wLoad0, (__local_mem__ uint8_t*&)weightInUbBaseAddr, aregWeightB8In);
            MicroAPI::ShiftRight(wShr0, wLoad0, wdup0, preg);
            MicroAPI::ShiftLeft(wShl, wLoad0, wdup1, preg);
            MicroAPI::ShiftRight(wShr1, wShl, wdup0, preg);
            MicroAPI::Select(wSel0, wShr1, wShr0, pregVsel);
            MicroAPI::And(sAnd0, wSel0, wdup2, preg);
            MicroAPI::DataCopy<uint8_t, MicroAPI::StoreDist::DIST_NORM_B8>(
                (__local_mem__ uint8_t*&)weightOutUbAddr, (MicroAPI::RegTensor<uint8_t>&)sAnd0, aregWeightB8Out, preg);
        }
    }
#endif
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, QuantType antiQuantType, typename scaleType, bool weightNz>
__aicore__ inline void
QuantBatchMatmulV4RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset, antiQuantType,
                                      scaleType, weightNz>::AntiQuantCompute(int32_t bubKOffset,
                                                                                           int32_t bubNLen,
                                                                                           int32_t bubKLen)
{
    // 以下表征 offset 的变量单位均为元素个数
    uint64_t weightOutUbOffset;
    uint64_t weightInUbOffset;  // B矩阵反量化前数据类型为 INT4 时, 在 UB 上的存储格式仍是 INT8
    uint64_t scaleUbOffset;
    if constexpr (IsSameType<wType, fp4x2_e2m1_t>::value) {  // A8W4
        if constexpr (weightNz) {
            weightOutUbOffset = ubBufIdx_ * VEC_MAX_ELEM_B8;
        } else {
            weightOutUbOffset = ubBufIdx_ * (vecWeightOutLen_ / sizeof(xType) / this->tiling_->BL1Pingpong);
        }
        weightInUbOffset = ubBufIdx_ * (vecWeightInLen_ << INT4_DTYPE_PARAM) / this->tiling_->BL1Pingpong;
        scaleUbOffset = ubBufIdx_ * (vecScaleOffsetLen_ / sizeof(scaleType) / this->tiling_->BL1Pingpong);
    } else {  // A16W8
        weightOutUbOffset = ubBufIdx_ * (vecWeightOutLen_ / sizeof(xType) / vecPingpong_);
        weightInUbOffset = ubBufIdx_ * (vecWeightInLen_ / sizeof(int8_t) / vecPingpong_);
        scaleUbOffset = ubScalePongFlag_ * (vecScaleOffsetLen_ / sizeof(scaleType) / vecPingpong_);
    }

    weightInUbBaseAddr_ = (__local_mem__ int8_t*)weightInUb_[weightInUbOffset].GetPhyAddr();
    scaleBaseAddr1_ = (__local_mem__ scaleType*)scaleInUb_[scaleUbOffset].GetPhyAddr();
    if constexpr (hasAntiQuantOffset) {
        offsetBaseAddr1_ = (__local_mem__ xType*)offsetInUb_[scaleUbOffset].GetPhyAddr();
    }
    weightOutUbAddr_ = (__local_mem__ xType*)weightOutUb_[weightOutUbOffset].GetPhyAddr();

    if constexpr (bTrans) {
        if constexpr (antiQuantType == QuantType::PER_GROUP || antiQuantType == QuantType::MX) {
            if constexpr (!weightNz) {
                uint16_t blockStride = CeilAlign(bubNLen, BLOCK_CUBE) + 1;
                weightOutUbAddr1_ = weightOutUbAddr_ + VEC_MAX_ELEM_B8 * blockStride;
                uint16_t repeatStride = blockStride * CeilDiv(Min(bubKLen, VEC_MAX_ELEM_B8), BLOCK_CUBE);
                vsstbConfig_ = (blockStride << 16u) | (repeatStride & 0xFFFFU);
                repeatTimes_ = CeilDiv(bubKLen >> INT4_DTYPE_PARAM, VECTOR_REG_WIDTH);
                outDimOffset_ = ONE_BLK_ELEM_B8 - repeatTimes_ * repeatStride * ONE_BLK_ELEM_B8;
    #ifndef __CCE_KT_TEST__
                if constexpr (antiQuantType == QuantType::MX) {
                    AntiQuantComputeNormal(bubKOffset, bubNLen, bubKLen);
                } else if constexpr (antiQuantType == QuantType::PER_GROUP && !weightNz) {
                    AntiQuantComputeNKGroup(bubKOffset, bubNLen, bubKLen);
                }
    #endif
            } else {
                if constexpr (antiQuantType == QuantType::MX) {
                    AntiQuantComputeNKMxNz(bubKOffset, bubNLen, bubKLen);
                }
            }
        } else {
            // not supported yet
        }
    } else {
        if constexpr (antiQuantType == QuantType::PER_GROUP && weightNz) {
                AntiQuantComputeKNGroupNz(bubKOffset, bubNLen, bubKLen);
        } else {
            // not supported yet
        }
    }
}
}  // namespace QuantBatchMatmulV4

#endif  // QUANT_BATCH_MATMUL_V4_REG_BASE_COMMON_H
