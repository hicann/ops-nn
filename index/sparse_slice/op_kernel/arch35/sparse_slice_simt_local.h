/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file sparse_slice_element_base.h
 * \brief
 */

#ifndef CANN_SPARSE_SLICE_SIMT_H
#define CANN_SPARSE_SLICE_SIMT_H

#include "sparse_slice_common.h"
#include "kernel_operator.h"
namespace SparseSlice {
using namespace AscendC;

constexpr int32_t THREAD_NUM = 256;
constexpr int32_t BINARY_HALF = 2;
constexpr int32_t INDICES_NUM = 19968;
constexpr uint32_t MAX_INDICES = 24;
constexpr uint32_t DIGIT_ALIGN = 16;
static constexpr int64_t DIGIT_TEN = 10;
static constexpr int64_t DIGIT_NINETEEN = 19;

template <typename NUM_T>
__simt_callee__ __aicore__ inline void SimtComputeReduction(__ubuf__ NUM_T* tmpValNum_, int64_t countBR)
{
    if(countBR == 0){
        tmpValNum_[0] = 0;
    }
    while(countBR > 1){
        int64_t halfBR = countBR / BINARY_HALF;
        int64_t res = countBR % BINARY_HALF;
        if(Simt::GetThreadIdx() < halfBR){
            tmpValNum_[Simt::GetThreadIdx()] += tmpValNum_[Simt::GetThreadIdx() + halfBR + res];
        }
        Simt::ThreadBarrier();
        countBR = halfBR + res;
    }
}

template <typename NUM_T>
__simt_callee__ __aicore__ inline void SimtComputePrefixSum(__ubuf__ NUM_T* temp)
{
    for (int stride = 1; stride < Simt::GetThreadNum(); stride *= BINARY_HALF) {
        if (Simt::GetThreadIdx() >= stride) {
            temp[Simt::GetThreadIdx() + Simt::GetThreadNum()] = temp[Simt::GetThreadIdx()] + temp[Simt::GetThreadIdx() - stride];
        } else {
            temp[Simt::GetThreadIdx() + Simt::GetThreadNum()] = temp[Simt::GetThreadIdx()];
        }
        Simt::ThreadBarrier();
        stride *= BINARY_HALF;
        if (Simt::GetThreadIdx() >= stride) {
            temp[Simt::GetThreadIdx()] = temp[Simt::GetThreadIdx() + Simt::GetThreadNum()] + temp[Simt::GetThreadIdx() + Simt::GetThreadNum() - stride];
        } else {
            temp[Simt::GetThreadIdx()] = temp[Simt::GetThreadIdx() + Simt::GetThreadNum()];
        }
        Simt::ThreadBarrier();
    }
}

template <typename I_T, typename V_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void SparseSliceSimtCompute(
    __ubuf__ I_T* indices, __gm__ V_T* values, __gm__ I_T* shape,
    __gm__ I_T* start, __gm__ I_T* size, __gm__ I_T* yIndices, 
    __gm__ V_T* yValues, __gm__ I_T* yShape, 
    __gm__ int64_t* tmpValNumGM, __ubuf__ int64_t* tmpValNum, 
    __ubuf__ int64_t* tmpValNumBlock, const int64_t nnzNum, const int64_t rankNum, int64_t startNNZ, int64_t endNNZ, int64_t nnzNumBlock, __ubuf__ I_T* yShape_,
    __ubuf__ bool* tmpValBlock_, __ubuf__ int64_t* indicesPos)
    {
        if(Simt::GetBlockIdx() == Simt::GetBlockNum()-1 && Simt::GetThreadIdx() < rankNum){
            yShape[Simt::GetThreadIdx()] = yShape_[Simt::GetThreadIdx()];
        }

        // 每个thread处理的位置
        int64_t avgNNZThread = nnzNumBlock / Simt::GetThreadNum();
        int64_t tailThread = nnzNumBlock - avgNNZThread * Simt::GetThreadNum();
        int64_t startNNZThread = startNNZ + avgNNZThread * Simt::GetThreadIdx() + (Simt::GetThreadIdx() > tailThread ? tailThread : Simt::GetThreadIdx());
        int64_t nnzNumThread = avgNNZThread + (Simt::GetThreadIdx() < tailThread ? 1 : 0);
        int64_t endNNZThread = startNNZThread + nnzNumThread;
        indicesPos[Simt::GetThreadIdx()] = startNNZThread;
        indicesPos[Simt::GetThreadIdx() + Simt::GetThreadNum()] = endNNZThread;

        int64_t threadVal = 0;
        for(int64_t i = startNNZThread-startNNZ; i < endNNZThread-startNNZ; i++){
            bool val = true;
            for(int64_t j = 0; j < rankNum; j++){
                int64_t tmp = indices[i*rankNum + j] - start[j];
                if(tmp < 0 || tmp >= yShape_[j]){
                    val = false;
                    break;
                }
            }
            if(val){
                threadVal += 1;
                tmpValBlock_[i] = true;
            }
        }
        tmpValNum[Simt::GetThreadIdx()] = threadVal;
        Simt::ThreadBarrier();
        
        SimtComputePrefixSum<int64_t>(tmpValNum);
        Simt::ThreadBarrier();
        if(Simt::GetThreadIdx() == 0){
            tmpValNumGM[Simt::GetBlockIdx()] = tmpValNum[Simt::GetThreadNum()-1];
        }
    }

template <typename I_T, typename V_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void SparseSliceSimtGather(
    __ubuf__ I_T* indices, __gm__ V_T* values, __gm__ I_T* shape,
    __gm__ I_T* start, __gm__ I_T* size, __gm__ volatile I_T* yIndices, 
    __gm__ volatile V_T* yValues, __gm__ volatile I_T* yShape, __gm__ volatile I_T* outputShape1,
    __gm__ int64_t* tmpValNumGM, __ubuf__ int64_t* tmpValNum, 
    __ubuf__ int64_t* tmpValNumBlock, const int64_t nnzNum, const int64_t rankNum, int64_t startNNZ, int64_t endNNZ, int64_t nnzNumBlock,
    __ubuf__ bool* tmpValBlock_, __ubuf__ int64_t* indicesPos)
    {
        // 每个block处理的nnz数据位置
        int64_t startNNZThread = indicesPos[Simt::GetThreadIdx()];
        int64_t endNNZThread = indicesPos[Simt::GetThreadIdx() + Simt::GetThreadNum()];

        if(Simt::GetThreadIdx() < Simt::GetBlockNum()){
            tmpValNumBlock[Simt::GetThreadIdx()] = tmpValNumGM[Simt::GetThreadIdx()*1];
        }
        Simt::ThreadBarrier();

        SimtComputeReduction<int64_t>(tmpValNumBlock, Simt::GetBlockIdx());
        if(Simt::GetBlockIdx() == Simt::GetBlockNum()-1 && Simt::GetThreadIdx() == Simt::GetThreadNum()-1){
            int64_t total = tmpValNumBlock[0] + tmpValNumBlock[Simt::GetBlockIdx()];
            outputShape1[DIGIT_ZERO] = Y_INDICES_SHAPE_DIM_BASE;
            outputShape1[DIGIT_ONE] = total;
            outputShape1[DIGIT_TWO] = rankNum;
            outputShape1[DIGIT_NINE] = DIGIT_ONE;
            outputShape1[DIGIT_TEN] = total;
            outputShape1[DIGIT_EIGHTEEN] = DIGIT_ONE;
            outputShape1[DIGIT_NINETEEN] = rankNum;
        }        

        // 计算每个block内线程的额外offset
        int64_t addrThreadOffset = 0;
        if(Simt::GetThreadIdx() != 0){
            addrThreadOffset = tmpValNum[Simt::GetThreadIdx()-1];
        }
        int64_t addrThreadOut = tmpValNumBlock[0] + addrThreadOffset;

        for(int64_t i = startNNZThread-startNNZ; i < endNNZThread-startNNZ; i++){
            if(tmpValBlock_[i]){
                for(int64_t j = 0; j < rankNum; j++){
                    yIndices[addrThreadOut * rankNum + j] = indices[(i)*rankNum + j] - start[j];
                }
                yValues[addrThreadOut] = values[i+startNNZ];
                addrThreadOut += 1;
            }
        }        
    }


template <typename I_T, typename V_T>
class SparseSliceSimt : public SparseSliceBase{
    public:
    __aicore__ inline SparseSliceSimt(){
        blockNum_ = GetBlockNum();
    }

    __aicore__ inline void Init(
        GM_ADDR indices, GM_ADDR values, GM_ADDR shape, GM_ADDR start, GM_ADDR size,
        GM_ADDR yIndices, GM_ADDR yValues, GM_ADDR yshape, 
        GM_ADDR outputShape1, GM_ADDR workspace, const SparseSliceTilingData& tilingData, 
        TPipe *inPipe);
    __aicore__ inline void Process();
    __aicore__ inline void CopyIndices();

    private:
    TPipe* pipe;

    GlobalTensor<I_T> indices_;
    GlobalTensor<V_T> values_;
    GlobalTensor<I_T> shape_;
    GlobalTensor<I_T> start_;
    GlobalTensor<I_T> size_;

    GlobalTensor<I_T> yIndices_;
    GlobalTensor<V_T> yValues_;
    GlobalTensor<I_T> yShapeTensor_;
    GlobalTensor<I_T> outputShape1_;

    GlobalTensor<int64_t> tmpValNumGM_;

    LocalTensor<int64_t> tmpValNum_;
    LocalTensor<int64_t> tmpValNumBlock_;
    LocalTensor<int64_t> indicesUB_;
    LocalTensor<I_T> yShapeUB_;
    LocalTensor<bool> tmpValBlock_;
    LocalTensor<int64_t>  indicesPos_;

    TBuf<TPosition::VECCALC> tmpValNumBuffer_;
    TBuf<TPosition::VECCALC> tmpValNumBlockBuffer_;
    TBuf<TPosition::VECCALC> tmpValBlockBuffer_;
    TBuf<TPosition::VECCALC> yShapeBuffer_;
    TBuf<TPosition::VECCALC> indicesPosBuffer_;
    TBuf<TPosition::VECIN> indicesUBBuffer_;

    int64_t nnzNum = 0;
    int64_t rankNum = 0;
    int64_t blockNum_= 0;
    int64_t startNNZ = 0;
    int64_t endNNZ = 0;
    int64_t nnzNumBlock = 0;
};

template <typename I_T, typename V_T>
__aicore__ inline void SparseSliceSimt<I_T, V_T>::CopyIndices(){
    // 每个block处理的nnz数据位置
    int64_t avgNNZ = nnzNum / GetBlockNum();
    int64_t tailBlock = nnzNum - avgNNZ * GetBlockNum();//4-(10-8)
    int64_t startNNZ = avgNNZ * GetBlockIdx() + (GetBlockIdx() > tailBlock ? tailBlock : GetBlockIdx());
    int64_t nnzNumBlock = avgNNZ + (GetBlockIdx() < tailBlock ? 1 : 0);
    DataCopyExtParams copyParamsIndices{ static_cast<uint16_t>(1), static_cast<uint32_t>(rankNum * nnzNumBlock * static_cast<int32_t>(sizeof(I_T))),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0) };
    DataCopyPadExtParams<I_T> padParams{false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
        static_cast<int64_t>(0) };
    DataCopyPad<I_T>(indicesUB_[0], indices_[startNNZ * rankNum], copyParamsIndices, padParams);
}

template <typename I_T, typename V_T>
__aicore__ inline void SparseSliceSimt<I_T, V_T>::Init(
    GM_ADDR indices, GM_ADDR values, GM_ADDR shape, GM_ADDR start, GM_ADDR size,
    GM_ADDR yIndices, GM_ADDR yValues, GM_ADDR yShape, 
    GM_ADDR outputShape1, GM_ADDR workspace, const SparseSliceTilingData& tilingData, 
    TPipe *inPipe){
        pipe = inPipe;
        SparseSliceBase::ParseTilingData(tilingData);
        nnzNum = tilingData.valueNumbers;
        rankNum = tilingData.rankNumbers;

        indices_.SetGlobalBuffer((__gm__ I_T*)(indices));
        indices_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        
        values_.SetGlobalBuffer((__gm__ V_T*)(values));
        shape_.SetGlobalBuffer((__gm__ I_T*)(shape));
        start_.SetGlobalBuffer((__gm__ I_T*)(start));
        size_.SetGlobalBuffer((__gm__ I_T*)(size));
        yIndices_.SetGlobalBuffer((__gm__ I_T*)(yIndices));
        yValues_.SetGlobalBuffer((__gm__ V_T*)(yValues));
        yShapeTensor_.SetGlobalBuffer((__gm__ I_T*)(yShape));
        outputShape1_.SetGlobalBuffer((__gm__ I_T*)(outputShape1));

        tmpValNumGM_.SetGlobalBuffer((__gm__ int64_t*)(workspace), blockNum_ * DIGIT_ALIGN);

        pipe->InitBuffer(tmpValNumBuffer_, DIGIT_TWO * THREAD_NUM * sizeof(int64_t));
        tmpValNum_ = tmpValNumBuffer_.Get<int64_t>();
        pipe->InitBuffer(tmpValNumBlockBuffer_, blockNum_ * sizeof(int64_t));
        tmpValNumBlock_ = tmpValNumBlockBuffer_.Get<int64_t>();
        pipe->InitBuffer(indicesUBBuffer_, INDICES_NUM * sizeof(int64_t));
        indicesUB_ = indicesUBBuffer_.Get<int64_t>();
        pipe->InitBuffer(tmpValBlockBuffer_, INDICES_NUM * sizeof(bool));
        tmpValBlock_ = tmpValBlockBuffer_.Get<bool>();
        pipe->InitBuffer(yShapeBuffer_, MAX_INDICES * sizeof(int64_t));
        yShapeUB_ = yShapeBuffer_.Get<int64_t>();
        pipe->InitBuffer(indicesPosBuffer_, DIGIT_TWO * THREAD_NUM * sizeof(int64_t));
        indicesPos_ = indicesPosBuffer_.Get<int64_t>();
        Duplicate(tmpValNum_, static_cast<int64_t>(0), DIGIT_TWO * THREAD_NUM);
        Duplicate(tmpValNumBlock_, static_cast<int64_t>(0), blockNum_);
        Duplicate(tmpValBlock_, false, INDICES_NUM);
        for (int32_t i = 0; i < rankNumbers_; i++) {
            yShapeUB_(i) = tilingData.yShape[i];
        }
    }

template <typename I_T, typename V_T>
__aicore__ inline void SparseSliceSimt<I_T, V_T>::Process(){
    CopyIndices();
    // 每个block处理的nnz数据位置
    int64_t avgNNZ = nnzNum / GetBlockNum();
    int64_t tailBlock = nnzNum - avgNNZ * GetBlockNum();
    startNNZ = avgNNZ * GetBlockIdx() + (GetBlockIdx() > tailBlock ? tailBlock : GetBlockIdx());
    nnzNumBlock = avgNNZ + (GetBlockIdx() < tailBlock ? 1 : 0);
    endNNZ = startNNZ + nnzNumBlock;

    PipeBarrier<PIPE_ALL>();

    Simt::VF_CALL<SparseSliceSimtCompute<I_T, V_T>>(Simt::Dim3(THREAD_NUM),
    (__ubuf__ I_T*) indicesUB_.GetPhyAddr(), (__gm__ V_T*) values_.GetPhyAddr(), (__gm__ I_T*) shape_.GetPhyAddr(),
    (__gm__ I_T*) start_.GetPhyAddr(), (__gm__ I_T*) size_.GetPhyAddr(), (__gm__ I_T*) yIndices_.GetPhyAddr(), 
    (__gm__ V_T*) yValues_.GetPhyAddr(), (__gm__ I_T*) yShapeTensor_.GetPhyAddr(), 
    (__gm__ int64_t*) tmpValNumGM_.GetPhyAddr(), (__ubuf__ int64_t*) tmpValNum_.GetPhyAddr(), 
    (__ubuf__ int64_t*) tmpValNumBlock_.GetPhyAddr(), nnzNum, rankNum, startNNZ, endNNZ, nnzNumBlock, (__ubuf__ I_T*) yShapeUB_.GetPhyAddr(),
    (__ubuf__ bool*) tmpValBlock_.GetPhyAddr(), (__ubuf__ int64_t*) indicesPos_.GetPhyAddr());
    PipeBarrier<PIPE_ALL>();
    SyncAll();
    Simt::VF_CALL<SparseSliceSimtGather<I_T, V_T>>(Simt::Dim3(THREAD_NUM),
    (__ubuf__ I_T*) indicesUB_.GetPhyAddr(), (__gm__ V_T*) values_.GetPhyAddr(), (__gm__ I_T*) shape_.GetPhyAddr(),
    (__gm__ I_T*) start_.GetPhyAddr(), (__gm__ I_T*) size_.GetPhyAddr(), (__gm__ volatile I_T*) yIndices_.GetPhyAddr(), 
    (__gm__ volatile V_T*) yValues_.GetPhyAddr(), (__gm__ volatile I_T*) yShapeTensor_.GetPhyAddr(), (__gm__ volatile I_T*) outputShape1_.GetPhyAddr(),
    (__gm__ int64_t*) tmpValNumGM_.GetPhyAddr(), (__ubuf__ int64_t*) tmpValNum_.GetPhyAddr(), 
    (__ubuf__ int64_t*) tmpValNumBlock_.GetPhyAddr(), nnzNum, rankNum, startNNZ, endNNZ, nnzNumBlock,
    (__ubuf__ bool*) tmpValBlock_.GetPhyAddr(), (__ubuf__ int64_t*) indicesPos_.GetPhyAddr());
}

}
#endif