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
 * \file quant_batch_matmul_v4_msd.h
 * \brief
 */

#ifndef QUANT_BATCH_MATMUL_V4_MSD_H
#define QUANT_BATCH_MATMUL_V4_MSD_H
#include "../inc/kernel_utils.h"
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "lib/matmul_intf.h"

#include "quant_batch_matmul_v4_tiling_data.h"

using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;
using AscendC::DataCopyExtParams;
using AscendC::DataCopyPadExtParams;
using AscendC::GetBlockIdx;
using AscendC::GetTaskRation;
using AscendC::GlobalTensor;
using AscendC::int4b_t;
using AscendC::LocalTensor;
using AscendC::QuePosition;
using AscendC::RoundMode;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::TPosition;
using AscendC::TQue;
using matmul::MatmulType;
using AscendC::MatmulImpl;
using AscendC::GetSubBlockIdx;
using AscendC::SyncAll;
using AscendC::CacheMode;
using AscendC::BroadCast;
using AscendC::PipeBarrier;
using AscendC::BinaryRepeatParams;

using AscendC::GetBlockNum;
using AscendC::HardEvent;
using AscendC::SetFlag;
using AscendC::WaitFlag;

enum class QuantType : std::uint8_t {
    K_C = 5,    //pertoken叠加perchannel
    K_G = 6     //pertoken叠加pergroup
};

constexpr uint64_t SYNC_AIV_TO_AIC = 3;
constexpr uint64_t SYNC_AIC_TO_AIV = 5;
constexpr uint32_t MM_BASE_BLOCK_OFFSET = 32768;
constexpr uint32_t UB_BLOCK_SIZE = 32;

template <typename T>
__aicore__ inline void DataCopyPad2DA8W4(const LocalTensor<T> dst, const GlobalTensor<T> src, uint32_t dim1, uint32_t dim0,
                                     uint32_t srcDim0) {
    DataCopyExtParams params;
    params.blockCount = dim1;
    params.blockLen = dim0 * sizeof(T);
    params.srcStride = (srcDim0 - dim0) * sizeof(T);
    params.dstStride = 0;

    DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
    DataCopyPad(dst, src, params, padParams);
}

template <typename T>
__aicore__ inline void DataCopyPad2DA8W4(const GlobalTensor<T> dst, const LocalTensor<T> src, uint32_t dim1, uint32_t dim0,
                                     uint32_t srcDim0, uint32_t dstDim0) {
    DataCopyExtParams params;
    params.blockCount = dim1;
    params.blockLen = dim0 * sizeof(T);
    // 32: ub访问粒度为32B
    params.srcStride = (srcDim0 - dim0) * sizeof(T) / 32;
    params.dstStride = (dstDim0 - dim0) * sizeof(T);
    DataCopyPad(dst, src, params);
}

constexpr uint32_t AND_ONE_REPEAT_LENGTH = 128; // and operation max length
constexpr uint32_t DOUBLE_ROWS = 2; // in8 to in4, one rows will change two rows
class QuantBatchMatmulV4MsdPre{
public:
    __aicore__ inline QuantBatchMatmulV4MsdPre(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workSpace,
                                const QuantBatchMatmulV4MsdTilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();
    __aicore__ inline void VectorCompute();
    __aicore__ inline void DataCopyGmToUb(const LocalTensor<int8_t> dst, const GlobalTensor<int8_t> src);
    __aicore__ inline void DataCopyUbToGm(const GlobalTensor<int8_t> dst, const LocalTensor<int8_t> src);
private:
    TQue<QuePosition::VECIN, 1> vecInQueueX_;
    TQue<QuePosition::VECOUT, 1> vecOutQueueA1_;
    TQue<QuePosition::VECOUT, 1> vecOutQueueA2_;
    TQue<QuePosition::VECOUT, 1> vecOutQueueA3_;
    TQue<QuePosition::VECOUT, 1> vecOutQueueA4_;
    TQue<QuePosition::VECOUT, 1> vecOutQueueA5_;
    TQue<QuePosition::VECOUT, 1> vecOutQueue0F_;

    LocalTensor<int8_t> xTensor_;
    LocalTensor<half> xHighHalfTensor_;
    LocalTensor<half> xLowHalfTensor_;
    LocalTensor<half> xLowHalfTensor2_;
    LocalTensor<int4b_t> xHighI4Tensor_;
    LocalTensor<int4b_t> xLowI4Tensor_;
    LocalTensor<int16_t> xLowI16Tensor_;
    GlobalTensor<int8_t> xGlobal_;
    GlobalTensor<int8_t> yGlobal_;

    uint32_t kSize_;
    uint32_t blockDim_;
    uint32_t coreIdx_;
    uint32_t groupNum_;
    uint32_t mSize_;
    uint32_t alignKSize_;
    const QuantBatchMatmulV4MsdTilingData *tilingData_;
    TPipe *pipe_;
};

 __aicore__ inline void QuantBatchMatmulV4MsdPre::Init(GM_ADDR x, GM_ADDR y, GM_ADDR workSpace,
                                                     const QuantBatchMatmulV4MsdTilingData * tilingData, TPipe *tPipe) {
    xGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(x));
    yGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(y));
    kSize_ = tilingData->kSize;
    alignKSize_ = ops::CeilDiv(kSize_, UB_BLOCK_SIZE) * UB_BLOCK_SIZE;
    tilingData_ = tilingData;
    pipe_ = tPipe;
    pipe_->InitBuffer(vecInQueueX_, 1, alignKSize_ * sizeof(int8_t));
    pipe_->InitBuffer(vecOutQueueA1_, 1, alignKSize_ * sizeof(int4b_t));
    pipe_->InitBuffer(vecOutQueueA2_, 1, alignKSize_ * sizeof(int4b_t));
    pipe_->InitBuffer(vecOutQueueA3_, 1, alignKSize_ * sizeof(half));
    pipe_->InitBuffer(vecOutQueueA4_, 1, alignKSize_ * sizeof(half));
    pipe_->InitBuffer(vecOutQueueA5_, 1, alignKSize_ * sizeof(half));
    constexpr int BUFFER_SIZE_256B = AND_ONE_REPEAT_LENGTH * sizeof(int16_t);
    pipe_->InitBuffer(vecOutQueue0F_, 1, BUFFER_SIZE_256B);
    coreIdx_ = GetBlockIdx();
    blockDim_ = GetBlockNum() * GetTaskRation();
    mSize_ = tilingData->mSize;
}

 __aicore__ inline void QuantBatchMatmulV4MsdPre::Process() {
    xTensor_ = vecInQueueX_.AllocTensor<int8_t>();
    xHighI4Tensor_ = vecOutQueueA1_.AllocTensor<int4b_t>();
    xLowI4Tensor_ = vecOutQueueA2_.AllocTensor<int4b_t>();
    xHighHalfTensor_ = vecOutQueueA3_.AllocTensor<half>();
    xLowHalfTensor_ = vecOutQueueA4_.AllocTensor<half>();
    xLowHalfTensor2_ = vecOutQueueA5_.AllocTensor<half>();
    xLowI16Tensor_ = vecOutQueue0F_.AllocTensor<int16_t>();
    VectorCompute();
    vecInQueueX_.FreeTensor(xTensor_);
    vecOutQueueA1_.FreeTensor(xHighI4Tensor_);
    vecOutQueueA2_.FreeTensor(xLowI4Tensor_);
    vecOutQueueA3_.FreeTensor(xHighHalfTensor_);
    vecOutQueueA4_.FreeTensor(xLowHalfTensor_);
    vecOutQueueA5_.FreeTensor(xLowHalfTensor2_);
    vecOutQueue0F_.FreeTensor(xLowI16Tensor_);
}

__aicore__ inline void QuantBatchMatmulV4MsdPre::DataCopyGmToUb(const LocalTensor<int8_t> dst, const GlobalTensor<int8_t> src) {
    DataCopyPadExtParams<int8_t> padParams;
    DataCopyExtParams xTensorParams{1, static_cast<uint32_t>(kSize_), 1 , 1, 0};
    DataCopyPad(dst, src, xTensorParams, padParams);
}

__aicore__ inline void QuantBatchMatmulV4MsdPre::DataCopyUbToGm(const GlobalTensor<int8_t> dst, const LocalTensor<int8_t> src) {
    DataCopyExtParams xTensorParams{1, static_cast<uint32_t>(kSize_ / 2), 1, 1, 0};
    DataCopyPad(dst, src, xTensorParams);
}

__aicore__ inline void QuantBatchMatmulV4MsdPre::VectorCompute() {
    Duplicate(xLowI16Tensor_, static_cast<int16_t>(0x0F0F),AND_ONE_REPEAT_LENGTH);
    const half OneEight = static_cast<half>(0.0625);
    const uint32_t halfKSize = kSize_ / DOUBLE_ROWS;
    const uint32_t repeatTimes = (alignKSize_ / sizeof(int16_t) ) / AND_ONE_REPEAT_LENGTH;
    const uint32_t last = (alignKSize_ / sizeof(int16_t)) % AND_ONE_REPEAT_LENGTH;
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
    const half MINUS_EIGHT = static_cast<half>(-8);
    for(uint32_t xLoop = coreIdx_; xLoop < mSize_; xLoop += blockDim_) {
        uint64_t offset = xLoop * kSize_;
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        DataCopyGmToUb(xTensor_, xGlobal_[offset]);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Cast(xHighHalfTensor_, xTensor_, AscendC::RoundMode::CAST_NONE, kSize_);
        PipeBarrier<PIPE_V>();
        Muls(xHighHalfTensor_, xHighHalfTensor_, OneEight, kSize_);
        PipeBarrier<PIPE_V>();
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
        Cast(xHighI4Tensor_, xHighHalfTensor_, AscendC::RoundMode::CAST_FLOOR, kSize_);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyUbToGm(yGlobal_[offset], xHighI4Tensor_.ReinterpretCast<int8_t>());
        SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
        if (repeatTimes > 0) {
            And(xLowHalfTensor_.ReinterpretCast<int16_t>(), xTensor_.ReinterpretCast<int16_t>(), xLowI16Tensor_, AND_ONE_REPEAT_LENGTH, repeatTimes, {1,1,1,8,8,0});
            PipeBarrier<PIPE_V>();      
        }
        if (last > 0) {
            const uint32_t andOffset = AND_ONE_REPEAT_LENGTH * repeatTimes;
            And(xLowHalfTensor_.ReinterpretCast<int16_t>()[andOffset], xTensor_.ReinterpretCast<int16_t>()[andOffset], xLowI16Tensor_, last, 1, {1,1,1,8,8,0});
            PipeBarrier<PIPE_V>();
        }
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        Cast(xLowHalfTensor2_.ReinterpretCast<half>(), xLowHalfTensor_.ReinterpretCast<int8_t>(), AscendC::RoundMode::CAST_NONE, kSize_);
        PipeBarrier<PIPE_V>();
        Adds(xHighHalfTensor_, xLowHalfTensor2_, MINUS_EIGHT, kSize_);
        PipeBarrier<PIPE_V>();
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        Cast(xLowI4Tensor_, xHighHalfTensor_.ReinterpretCast<half>(), AscendC::RoundMode::CAST_NONE, kSize_);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
        DataCopyUbToGm(yGlobal_[offset + halfKSize], xLowI4Tensor_.ReinterpretCast<int8_t>());
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
}

template <typename xType, typename wType, typename scaleType, typename yType, QuantType quantType, bool weightNz = false>
class QuantBatchMatmulV4Msd
{
public:
    __aicore__ inline QuantBatchMatmulV4Msd(){};
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR x1_scale, GM_ADDR x2_scale,
                                GM_ADDR y_scale, GM_ADDR x1_offset, GM_ADDR x2_offset, GM_ADDR y_offset, GM_ADDR y,
                                GM_ADDR workspace, const QuantBatchMatmulV4MsdTilingData* tilingData, TPipe* tPipe);
    __aicore__ inline void Process();
    __aicore__ inline void InitUbBuffer();
    __aicore__ inline void MMCompute(uint32_t mIdx, uint32_t nIdx, uint64_t workSpaceOffset);
    __aicore__ inline void VectorCompute(uint32_t mIdx, uint32_t nIdx, uint64_t workSpaceOffset);
    __aicore__ inline void ComputeDequant(uint32_t mIdx, uint32_t curVecBaseM, uint32_t alignBaseN, uint32_t curVecBaseN, uint32_t offsetM);
    __aicore__ inline void DataCopyYOffset(uint32_t curBaseN, uint32_t alignBaseN, uint64_t yOffset);
    __aicore__ inline void DataCopyX1ScaleAndBrcb(uint32_t mIdx, uint32_t curBaseM, uint32_t alignBaseN, uint32_t offsetM);

private:
    const uint32_t HALF_ALIGN = 16;
    GlobalTensor<xType> x1Global_;
    GlobalTensor<wType> x2Global_;
    GlobalTensor<half> mmOutGlobal_;
    GlobalTensor<scaleType> x1ScaleGlobal_;
    GlobalTensor<uint64_t> x2ScaleGlobal_;
    GlobalTensor<float> yOffsetGlobal_;
    GlobalTensor<yType> yGlobal_;

    // define the que
    TQue<QuePosition::VECIN, 1> vecInQueue_;
    TQue<QuePosition::VECOUT, 1> vecOutQueue_;
    TQue<QuePosition::VECIN, 1> x1ScaleInQueue_;
    TQue<QuePosition::VECIN, 1> yOffsetInQueue_;
    TBuf<TPosition::VECCALC> tmpBuff_;
    LocalTensor<float> yOffsetInUb_;
    LocalTensor<float> buffer1_;
    LocalTensor<float> buffer2_;
    LocalTensor<float> buffer3_;
    LocalTensor<float> buffer4_;
    LocalTensor<uint8_t> buffer5_;

    // tilingData
    uint32_t nSize_;
    uint32_t mSize_;
    uint32_t kSize_;
    uint32_t baseM_;
    uint32_t baseN_;

    uint32_t subBlockIdx_;
    uint32_t coreIdx_;
    uint32_t groupSize_;
    uint32_t groupNum_;
    uint32_t cubeCount = 0;
    uint32_t vecCount = 0;
    uint32_t blockDimN_;
    uint32_t blockDimM_;
    uint32_t x1ScaleComputeSize_;
    uint32_t workSpaceOffset_ = 0;
    uint32_t quantGroupSize;
    uint32_t cubeCount_ = 0;
    uint32_t vecCount_ = 0;
    TPipe *pipe_;
    const QuantBatchMatmulV4MsdTilingData *tilingData_;
    const TCubeTiling* matmulTiling_;
    using inputX1Type = MatmulType<TPosition::GM, CubeFormat::ND, int4b_t, false>;
    using inputX2Type = MatmulType<TPosition::GM, CubeFormat::ND, int4b_t, false>;
    using inputBiasType = MatmulType<TPosition::GM, CubeFormat::ND, int32_t, false>;
    using outputYType = MatmulType<TPosition::GM, CubeFormat::ND, half, false>;
    MatmulImpl<inputX1Type, inputX2Type, outputYType, inputBiasType, CFG_MDL> mmObj_;
};

template <typename xType, typename wType, typename scaleType, typename yType, QuantType quantType, bool weightNz >
__aicore__ inline void QuantBatchMatmulV4Msd<xType, wType, scaleType, yType, quantType, weightNz >::Init
                            (GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR x1_scale, GM_ADDR x2_scale,
                            GM_ADDR y_scale, GM_ADDR x1_offset, GM_ADDR x2_offset, GM_ADDR y_offset, GM_ADDR y,
                            GM_ADDR workspace, const QuantBatchMatmulV4MsdTilingData* tilingData, TPipe* tPipe) {
    x1Global_.SetGlobalBuffer(reinterpret_cast<__gm__ xType*>(x1));
    x2Global_.SetGlobalBuffer(reinterpret_cast<__gm__ wType*>(x2));
    mmOutGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(workspace));
    x1ScaleGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ scaleType*>(x1_scale));
    x2ScaleGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(x2_scale));
    yOffsetGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y_offset));
    yGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ yType*>(y));

    tilingData_ = tilingData;
    matmulTiling_ = &(tilingData->matmulTiling);
    baseM_ = matmulTiling_->baseM;
    baseN_ = matmulTiling_->baseN;
    if constexpr (quantType == QuantType::K_G) {
        groupSize_ = tilingData_->groupSize;
        groupNum_ = tilingData_->kSize / groupSize_;  // 约束为整除关系
    }
    subBlockIdx_ = GetSubBlockIdx();
    coreIdx_ = GetBlockIdx();
    nSize_ = tilingData_->nSize;
    kSize_ = tilingData_->kSize;
    mSize_ = tilingData_->mSize;

    mmObj_.SetSubBlockIdx(0);
    mmObj_.Init(matmulTiling_, pipe_);
    if ASCEND_IS_AIV {
        if (GetTaskRation() != 0) {
            coreIdx_ /= GetTaskRation();
        }
    }
    pipe_ = tPipe;
    InitUbBuffer();
}


template <typename xType, typename wType, typename scaleType, typename yType, QuantType quantType, bool weightNz >
__aicore__ inline void QuantBatchMatmulV4Msd<xType, wType, scaleType, yType, quantType, weightNz >::InitUbBuffer(){
    if ASCEND_IS_AIC {
        return;
    }
    pipe_->InitBuffer(yOffsetInQueue_, 1, baseN_ * sizeof(scaleType));
    pipe_->InitBuffer(x1ScaleInQueue_, 1, ops::CeilDiv(tilingData_->vBaseM * sizeof(scaleType), static_cast<unsigned long>(32)) * 32);
    pipe_->InitBuffer(vecInQueue_, 1, tilingData_->ubCalSize * 2 * sizeof(half));
    pipe_->InitBuffer(vecOutQueue_, 1, tilingData_->ubCalSize * sizeof(yType));
    pipe_->InitBuffer(tmpBuff_, tilingData_->ubRestBytes);
    uint32_t ubCalSizeFloat = tilingData_->ubCalSize * sizeof(float);
    // ub分配，依次划分中间结果，划分方式参考设计文档
    buffer1_ = tmpBuff_.GetWithOffset<float>(tilingData_->ubCalSize * 2, 0);

    buffer5_ = tmpBuff_.GetWithOffset<uint8_t>(2 * ubCalSizeFloat, 0);
    uint32_t offset = ubCalSizeFloat * 2;
    buffer2_ = tmpBuff_.GetWithOffset<float>(tilingData_->ubCalSize, offset);
    offset += ubCalSizeFloat;
    buffer3_ = tmpBuff_.GetWithOffset<float>(tilingData_->ubCalSize, offset);
    buffer4_ = tmpBuff_.GetWithOffset<float>(tilingData_->ubCalSize, 0);
}

template <typename xType, typename wType, typename scaleType, typename yType, QuantType quantType, bool weightNz >
__aicore__ inline void QuantBatchMatmulV4Msd<xType, wType, scaleType, yType, quantType, weightNz >::Process()
{
    mmObj_.SetOrgShape(mSize_ * 2 , nSize_, kSize_);
    blockDimN_ = ops::CeilDiv(nSize_, baseN_);
    blockDimM_ = ops::CeilDiv(mSize_ * 2, baseM_);
    uint32_t curCount = blockDimN_ * blockDimM_;
    uint32_t curBlock = coreIdx_;

    while (curBlock < curCount) {
        uint32_t mIdx = curBlock / blockDimN_;
        uint32_t nIdx = curBlock % blockDimN_;
        workSpaceOffset_ = MM_BASE_BLOCK_OFFSET * (coreIdx_ + (cubeCount % tilingData_->parallNum) * tilingData_->coreNum);
        MMCompute(mIdx, nIdx, workSpaceOffset_);

        if ASCEND_IS_AIV {
            VectorCompute(mIdx, nIdx, workSpaceOffset_);
        }
        curBlock += tilingData_->coreNum;
    }
}

template <typename xType, typename wType, typename scaleType, typename yType, QuantType quantType, bool weightNz >
__aicore__ inline void QuantBatchMatmulV4Msd<xType, wType, scaleType, yType, quantType, weightNz >::MMCompute(uint32_t mIdx, uint32_t nIdx, uint64_t workSpaceOffset)
{
    if ASCEND_IS_AIC {
        uint64_t x1Offset = mIdx * baseM_ * kSize_;
        uint64_t x2Offset = nIdx * baseN_;
        uint32_t curSingleN = baseN_;
        if (nIdx == blockDimN_ - 1) {
            curSingleN = nSize_ - x2Offset;
        }
        uint32_t curSingleM = baseM_;
        if (mIdx == blockDimM_ - 1) {
            curSingleM = mSize_ * 2 - mIdx * baseM_;
        }


        if (cubeCount >= tilingData_->parallNum) {
            CrossCoreWaitFlag(SYNC_AIV_TO_AIC);
        }
        if constexpr (quantType == QuantType::K_C) {
            mmObj_.SetSingleShape(curSingleM, curSingleN, kSize_);
            mmObj_.SetTensorA(x1Global_[x1Offset]);
            auto weightSlice = x2Global_[x2Offset];
            if (blockDimM_ == 1) {
                weightSlice.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            }
            mmObj_.SetTensorB(weightSlice);
            mmObj_.SetQuantVector(x2ScaleGlobal_[x2Offset]);
            mmObj_.Iterate();
            mmObj_.GetTensorC(mmOutGlobal_[workSpaceOffset], 0, true);
        } else {
            mmObj_.SetSingleShape(curSingleM, curSingleN, groupSize_);
            GlobalTensor<wType> weightSlice;
            for (uint32_t loopK = 0; loopK < groupNum_; loopK++) {
                mmObj_.SetTensorA(x1Global_[x1Offset + loopK * groupSize_]);
                auto weightSlice = x2Global_[x2Offset + loopK * groupSize_ * nSize_];
                if (blockDimM_ == 1) {
                    weightSlice.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
                }
                mmObj_.SetTensorB(weightSlice);
                mmObj_.SetQuantVector(x2ScaleGlobal_[loopK * nSize_ + x2Offset]);
                mmObj_.Iterate();
                mmObj_.GetTensorC(mmOutGlobal_[workSpaceOffset], loopK == 0 ? 0 : 1, true);
            }
        }
        CrossCoreSetFlag<2, PIPE_FIX>(SYNC_AIC_TO_AIV);
    }
    cubeCount++;
}

template <typename xType, typename wType, typename scaleType, typename yType, QuantType quantType, bool weightNz >
__aicore__ inline void QuantBatchMatmulV4Msd<xType, wType, scaleType, yType, quantType, weightNz >::VectorCompute(uint32_t mIdx, uint32_t nIdx, uint64_t workSpaceOffset)
{
    uint32_t curCubeSingleN = baseN_;
    if (nIdx == blockDimN_ - 1) {
        curCubeSingleN = nSize_ - nIdx * baseN_;
    }
    uint32_t curCubeSingleM = baseM_ / 2; // 2: 2 lines int4 to 1 line int8
    uint64_t outOffset = mIdx * curCubeSingleM * tilingData_->nSize + nIdx * baseN_;
    if (mIdx == blockDimM_ - 1) {
        curCubeSingleM =  mSize_ - mIdx * curCubeSingleM;
    }
    uint32_t vecBaseM = tilingData_->ubCalSize / baseN_;
    vecBaseM = vecBaseM < curCubeSingleM ? vecBaseM : curCubeSingleM;
    uint32_t curVecBaseN = baseN_;
    uint64_t yOffset =  nIdx * baseN_;
    uint32_t taskRation = GetTaskRation();
    CrossCoreWaitFlag(SYNC_AIC_TO_AIV);
    for (uint32_t offsetN = 0; offsetN < curCubeSingleN; offsetN += baseN_) {
        if (offsetN + baseN_ >= curCubeSingleN) curVecBaseN = curCubeSingleN - offsetN;
        uint32_t alignBaseN = ops::CeilDiv(curVecBaseN, uint32_t(16)) * 16;  //  16: num half in 32B ub block
        DataCopyYOffset(curVecBaseN, alignBaseN, yOffset + offsetN);
        uint32_t curVecBaseM = vecBaseM;
        uint64_t mmOutOffset = workSpaceOffset + offsetN * baseM_;
        for (uint32_t offsetM = 0; offsetM < curCubeSingleM; offsetM += vecBaseM) {
            vecCount++;
            if (taskRation != 0 && vecCount % taskRation != subBlockIdx_) { continue; }
            if (offsetM + vecBaseM >= curCubeSingleM) { curVecBaseM = curCubeSingleM - offsetM; }
            LocalTensor<half> mmOutLocal = vecInQueue_.AllocTensor<half>();
            DataCopyPad2DA8W4(mmOutLocal, mmOutGlobal_[mmOutOffset + offsetM * 2 * curVecBaseN],
                          curVecBaseM, curVecBaseN, curVecBaseN * 2);   // 2: 2 lines int4 to 1 line int8
            DataCopyPad2DA8W4(mmOutLocal[curVecBaseM * alignBaseN],
                          mmOutGlobal_[mmOutOffset + (offsetM * 2 + 1) * curVecBaseN],
                          curVecBaseM, curVecBaseN, curVecBaseN * 2);    // 2: 2 lines int4 to 1 line int8
            vecInQueue_.EnQue(mmOutLocal);
            ComputeDequant(mIdx, curVecBaseM, alignBaseN, curVecBaseN, offsetM);
            LocalTensor<yType> yLocal = vecOutQueue_.DeQue<yType>();
            DataCopyPad2DA8W4(yGlobal_[outOffset + offsetM * nSize_ + offsetN], yLocal,
                        curVecBaseM, curVecBaseN, alignBaseN, nSize_);
            vecOutQueue_.FreeTensor(yLocal);
        }
        yOffsetInQueue_.FreeTensor(yOffsetInUb_);
    }
    CrossCoreSetFlag<2, PIPE_MTE2>(SYNC_AIV_TO_AIC);
}

template <typename xType, typename wType, typename scaleType, typename yType, QuantType quantType, bool weightNz >
__aicore__ inline void QuantBatchMatmulV4Msd<xType, wType, scaleType, yType, quantType, weightNz >::ComputeDequant(uint32_t mIdx,
    uint32_t curVecBaseM, uint32_t alignBaseN, uint32_t curVecBaseN, uint32_t offsetM)
{
    uint32_t computeSize = curVecBaseM * alignBaseN;
    LocalTensor<half> mmOutInUb = vecInQueue_.DeQue<half>();
    uint32_t castSize= computeSize + ops::CeilDiv(computeSize, HALF_ALIGN)*HALF_ALIGN;
    Cast(buffer1_, mmOutInUb, RoundMode::CAST_NONE, castSize);  // 2: in pre process, x has been splited to [2m,n]
    PipeBarrier<PIPE_V>();
    vecInQueue_.FreeTensor(mmOutInUb);
    const float RIGHT_MOVE = 16.0f;         // right move int4 to int8
    Muls(buffer2_, buffer1_, RIGHT_MOVE, computeSize);
    PipeBarrier<PIPE_V>();
    uint32_t addStartAddr = ops::CeilDiv(computeSize, HALF_ALIGN)*HALF_ALIGN;
    Add(buffer3_, buffer1_[addStartAddr], buffer2_, computeSize);
    PipeBarrier<PIPE_V>();
    uint32_t loop = alignBaseN / 64; // 256B为64个float，alignBaseN需约束为64倍数
    uint8_t blkStride = static_cast<uint8_t>(alignBaseN * sizeof(float) / 32);  //32: 单位32B
    BinaryRepeatParams param(1, 1, 1, blkStride, blkStride, 0);
    uint64_t mask = 64;     // float is 32 bit, mask continous mode [1,64], set all params involved in the computation
    for (uint32_t i = 0; i < loop; i++) {
        uint32_t offset = i * 64; // 每次64个元素
        Add(buffer2_[offset], buffer3_[offset], yOffsetInUb_[offset], mask, curVecBaseM, param);
    }
    PipeBarrier<PIPE_V>();
    uint64_t last = alignBaseN % 64;
    if(last > 0) {
        uint32_t offset = loop *64;
        Add(buffer2_[offset], buffer3_[offset], yOffsetInUb_[offset], last, curVecBaseM, param);
    }
    PipeBarrier<PIPE_V>();

    DataCopyX1ScaleAndBrcb(mIdx, curVecBaseM, alignBaseN, offsetM);

    Mul(buffer4_, buffer2_, buffer3_, computeSize);
    PipeBarrier<PIPE_V>();

    LocalTensor<yType> yLocalInUb = vecOutQueue_.AllocTensor<yType>();
    // Cast后获得最终输出
    Cast(yLocalInUb, buffer4_, RoundMode::CAST_RINT, computeSize);
    PipeBarrier<PIPE_V>();
    vecOutQueue_.EnQue(yLocalInUb);
}

template <typename xType, typename wType, typename scaleType, typename yType, QuantType quantType, bool weightNz >
__aicore__ inline void QuantBatchMatmulV4Msd<xType, wType, scaleType, yType, quantType, weightNz >::DataCopyYOffset(uint32_t curBaseN, uint32_t alignBaseN, uint64_t yOffset)
{
    DataCopyPadExtParams<float> padParams;
    DataCopyExtParams yOffsetParams{1, static_cast<uint32_t>(curBaseN * sizeof(float)), 1, 1, 0};
    LocalTensor<float> yOffsetLocal = yOffsetInQueue_.AllocTensor<float>();
    DataCopyPad(yOffsetLocal, yOffsetGlobal_[yOffset], yOffsetParams, padParams);
    yOffsetInQueue_.EnQue(yOffsetLocal);
    yOffsetInUb_ = yOffsetInQueue_.DeQue<float>();
}

template <typename xType, typename wType, typename scaleType, typename yType, QuantType quantType, bool weightNz >
__aicore__ inline void QuantBatchMatmulV4Msd<xType, wType, scaleType, yType, quantType, weightNz >::DataCopyX1ScaleAndBrcb(uint32_t mIdx,
        uint32_t curBaseM, uint32_t alignBaseN, uint32_t offsetM)
{
    uint64_t x1ScaleOffset = mIdx * baseM_ / 2 + offsetM; //2: m方向两行合并为1行
    uint32_t alignBaseM = ops::CeilDiv(curBaseM, uint32_t(8)) * 8;  //  8: num int32_t in 32B ub block
    // GM拷贝per token scale
    DataCopyPadExtParams<float> padParams;
    DataCopyExtParams x1ScaleParams{1, static_cast<uint32_t>(curBaseM * sizeof(float)), 0, 0, 0};
    LocalTensor<float> x1ScaleLocal = x1ScaleInQueue_.AllocTensor<float>();
    DataCopyPad(x1ScaleLocal, x1ScaleGlobal_[x1ScaleOffset], x1ScaleParams, padParams);

    x1ScaleInQueue_.EnQue(x1ScaleLocal);

    x1ScaleLocal = x1ScaleInQueue_.DeQue<float>();
    auto scaleTmp = x1ScaleLocal;

    const uint32_t broadCastDst[2] = {curBaseM, alignBaseN};
    const uint32_t broadCastSrc[2] = {curBaseM, 1};
    BroadCast<float, 2, 1>(buffer3_, scaleTmp, broadCastDst, broadCastSrc, buffer5_);
    x1ScaleInQueue_.FreeTensor(x1ScaleLocal);
}
#endif