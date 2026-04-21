/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Shi Xiangyang <@shi-xiangyang225>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software: you can redistribute it and/or modify it.
 * Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * See the LICENSE file at the root of the repository for the full text of the License.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */
/*!
 * \file sync_bn_training_update_v2.h
 * \brief
 */
#ifndef __SYNC_BN_TRAINING_UPDATE_V2_H__
#define __SYNC_BN_TRAINING_UPDATE_V2_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "sync_bn_training_update_v2_tiling_data.h"
#include "sync_bn_training_update_v2_tiling_key.h"
namespace NsSyncBnTrainingUpdateV2 {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t DATA_CACHE_CLEAN_NEED = 64;
constexpr int32_t ALIGN32_NEED = 32 / sizeof(float);
constexpr int32_t SLOT = DATA_CACHE_CLEAN_NEED / sizeof(float); // 每个slot占用的float数
constexpr float EPSILON = 1e-6;
constexpr bool isReuse = false;
template <typename T>
class SyncBnTrainingUpdateV2 {
public:
    __aicore__ inline SyncBnTrainingUpdateV2(){};
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR x3, GM_ADDR y1, GM_ADDR y2, GM_ADDR y3, GM_ADDR workspace, SyncBnTrainingUpdateV2TilingData* tilingData);
    __aicore__ inline void Process();
private:
    __aicore__ inline void CopyIn(uint32_t progress);
    __aicore__ inline void CopyOut(uint32_t progress);
    __aicore__ inline void ReduceU(uint32_t progress); 
    __aicore__ inline void ReduceV(uint32_t progress);
    __aicore__ inline void SumAndSyncAllU();
    __aicore__ inline void SumAndSyncAllV();
    __aicore__ inline void Normalization(uint32_t progress);
    __aicore__ inline void ReduceUProcess();
    __aicore__ inline void ReduceVProcess();
    __aicore__ inline void NormalizationProcess();
    __aicore__ inline void UpdateRunning();
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX1;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY1;
    TBuf<QuePosition::VECCALC> TmpBuf;
    TBuf<QuePosition::VECCALC> SyncSumBuf;
    TBuf<QuePosition::VECCALC> LocalSumBuf;
    TBuf<QuePosition::VECCALC> ReduceBuf;
    TBuf<QuePosition::VECCALC> MathBuf;
    TBuf<QuePosition::VECCALC> BaseBuf;
    TBuf<QuePosition::VECCALC> GatherBuf;

    GlobalTensor<T> x1Gm;
    GlobalTensor<T> x2Gm;
    GlobalTensor<T> x3Gm;
    GlobalTensor<T> y1Gm;
    GlobalTensor<T> y2Gm;
    GlobalTensor<T> y3Gm;
    GlobalTensor<T> workGm; 

    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;

    uint32_t Hdim;
    uint32_t Wdim;
    uint32_t Cdim;
    uint32_t Ndim;
    float momentum;
    uint32_t channelIdx = 0;
    uint32_t StarchannelIdx;
    uint32_t prenum;
    uint32_t Starprenum;
    uint32_t WorkOff;
    LocalTensor<float> localSum;
    LocalTensor<uint32_t> gatherPattern;
    GatherMaskParams params;
    int32_t mask;// 每次repeat操作mask个元素，总的数据计算量为repeatTimes * mask个元素。
    uint32_t vecSize;
};

template <typename T>
__aicore__ inline void SyncBnTrainingUpdateV2<T>::Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR x3, GM_ADDR y1, GM_ADDR y2, GM_ADDR y3, GM_ADDR workspace, SyncBnTrainingUpdateV2TilingData* tilingData)
{
    ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
    uint32_t coreNum = GetBlockIdx();
    uint32_t globalBufferIndex = tilingData->bigCoreDataNum * GetBlockIdx();
    this->tileDataNum = tilingData->tileDataNum;
    this->momentum = tilingData->momentum;
    this->Hdim = tilingData->Hdim;
    this->Wdim = tilingData->Wdim;
    this->Cdim = tilingData->Cdim;
    this->Ndim = tilingData->Ndim;
    this->WorkOff = this->Cdim * SLOT; // workspace 偏移量
    if (coreNum < tilingData->tailBlockNum) {
        this->coreDataNum = tilingData->bigCoreDataNum;
        this->tileNum = tilingData->finalBigTileNum;
        this->tailDataNum = tilingData->bigTailDataNum;
    }
    else {
        this->coreDataNum = tilingData->smallCoreDataNum;
        this->tileNum = tilingData->finalSmallTileNum;
        this->tailDataNum = tilingData->smallTailDataNum;
        globalBufferIndex -= (tilingData->bigCoreDataNum - tilingData->smallCoreDataNum) * (GetBlockIdx() - tilingData->tailBlockNum);
    }
    x1Gm.SetGlobalBuffer((__gm__ T*)x1 + globalBufferIndex, this->coreDataNum);
    y1Gm.SetGlobalBuffer((__gm__ T*)y1 + globalBufferIndex, this->coreDataNum);
    x2Gm.SetGlobalBuffer((__gm__ T*)x2 , this->Cdim);
    y2Gm.SetGlobalBuffer((__gm__ T*)y2 , this->Cdim);
    x3Gm.SetGlobalBuffer((__gm__ T*)x3 , this->Cdim);
    y3Gm.SetGlobalBuffer((__gm__ T*)y3 , this->Cdim);
    workGm.SetGlobalBuffer((__gm__ T*)workspace, this->Cdim * SLOT * 2); // MEAN VAR
    pipe.InitBuffer(inQueueX1, BUFFER_NUM, this->tileDataNum * sizeof(T) * ALIGN32_NEED); // ALIGN32_NEED个float一组
    pipe.InitBuffer(outQueueY1, BUFFER_NUM, this->tileDataNum * sizeof(T));  

    pipe.InitBuffer(TmpBuf, this->Hdim * this->Wdim * sizeof(T));
    pipe.InitBuffer(MathBuf, this->Hdim * this->Wdim * sizeof(T));
    pipe.InitBuffer(SyncSumBuf, this->Cdim * SLOT * sizeof(float));
    pipe.InitBuffer(ReduceBuf, SLOT * sizeof(float));
    pipe.InitBuffer(LocalSumBuf, this->Cdim * SLOT * sizeof(float));
    pipe.InitBuffer(BaseBuf, this->Hdim * this->Wdim * sizeof(float));
    pipe.InitBuffer(GatherBuf,  sizeof(uint32_t) * ALIGN32_NEED);

    this->gatherPattern = GatherBuf.Get<uint32_t>();
    Duplicate(gatherPattern, static_cast<uint32_t>(0), ALIGN32_NEED);
    gatherPattern.SetValue(0, static_cast<uint32_t>(1));
    this->localSum = LocalSumBuf.Get<float>();
    Duplicate(this->localSum, 0.0f, this->Cdim);
    this->vecSize = this->Hdim * this->Wdim; // 每次处理的向量长度
    this->mask = static_cast<int32_t>(ALIGN32_NEED);
    // gather参数初始化
    params.src0BlockStride   = 1;
    params.repeatTimes       = this->vecSize;
    params.src0RepeatStride  = 1;
    params.src1RepeatStride  = 0;
    this->StarchannelIdx = (globalBufferIndex % (this->Cdim * this->Hdim * this->Wdim)) / (this->Hdim * this->Wdim);// 核起始位置需要处理的channelIdx
    this->Starprenum = this->vecSize - (globalBufferIndex % (this->Hdim * this->Wdim));// 核起始位置需要处理的prenum
    if(this->Starprenum == this->vecSize){
        this->Starprenum = 0;
    }
}

template <typename T>
__aicore__ inline void SyncBnTrainingUpdateV2<T>::CopyIn(uint32_t progress)
{
    LocalTensor<T> x1Local = inQueueX1.AllocTensor<T>();
    DataCopyExtParams copyParams{static_cast<uint16_t>(this->processDataNum), static_cast<uint32_t>(sizeof(float)), 0, 0, 0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位
    DataCopyPadExtParams<float> padParams{true, 0, 7, 0};
    DataCopyPad(x1Local, x1Gm[progress * this->tileDataNum], copyParams, padParams); // 从GM->VECIN搬运
    inQueueX1.EnQue(x1Local);
}

template <typename T>
__aicore__ inline void SyncBnTrainingUpdateV2<T>::CopyOut(uint32_t progress)
{
    LocalTensor<T> y1Local = outQueueY1.DeQue<T>();
    DataCopy(y1Gm[progress * this->tileDataNum], y1Local, this->processDataNum);
    outQueueY1.FreeTensor(y1Local);
}

template <typename T>
__aicore__ inline void SyncBnTrainingUpdateV2<T>::ReduceU(uint32_t progress)
{
    LocalTensor<T> tileLocal = inQueueX1.DeQue<T>();
    LocalTensor<T> subLocal = MathBuf.Get<T>();
    LocalTensor<float> ReduceLocal = ReduceBuf.Get<float>();
    const uint32_t shape[] = { 1, this->Hdim * this->Wdim};//只支持二维数组
    uint32_t remainder = this->processDataNum;
    uint32_t off = 0;
    AscendC::TEventID eventID = GetTPipePtr()->AllocEventID<AscendC::HardEvent::V_S>();
    AscendC::SetFlag<AscendC::HardEvent::V_S>(eventID);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(eventID);
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::V_S>(eventID);
    if(prenum != 0){
        float pretmp = 0;
        for(uint32_t i = 0; i < prenum; ++i){
            pretmp += tileLocal.GetValue(i * ALIGN32_NEED);
        }
        this->localSum.SetValue(channelIdx, pretmp + this->localSum.GetValue(channelIdx));
        remainder -= prenum;
        channelIdx = (channelIdx + 1) % this->Cdim;
        off += prenum;
        prenum = 0;
    }
    for(; remainder >= this->vecSize && this-> vecSize != 0; ){
        uint64_t rsvdCnt = 0;
        GatherMask(subLocal, tileLocal[off * ALIGN32_NEED], this->gatherPattern, true, this->mask, params, rsvdCnt);
        PipeBarrier<PIPE_V>();

        Duplicate(ReduceLocal, 0.0f, 1);
        PipeBarrier<PIPE_V>();

        ReduceSum<float, Pattern::Reduce::AR, isReuse>(ReduceLocal, subLocal, shape, true);
        int32_t eventIDvToS = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S));
        AscendC::SetFlag<AscendC::HardEvent::V_S>(eventIDvToS);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(eventIDvToS);
        
        this->localSum.SetValue(channelIdx, this->localSum.GetValue(channelIdx) + ReduceLocal.GetValue(0));

        remainder -= this->vecSize;
        channelIdx = (channelIdx + 1) % this->Cdim;
        off += this->vecSize;
    }
    if(remainder != 0 ){
        float lasttmp = 0;
        for(uint32_t i = 0; i < remainder; ++i){
            lasttmp += tileLocal.GetValue(off * ALIGN32_NEED + i * ALIGN32_NEED);
        }
        localSum.SetValue(channelIdx, this->localSum.GetValue(channelIdx) + lasttmp);
        prenum = this->vecSize - remainder;
        remainder = 0;
    }
    inQueueX1.FreeTensor(tileLocal);
}

template <typename T>
__aicore__ inline void SyncBnTrainingUpdateV2<T>::ReduceV(uint32_t progress)
{
    LocalTensor<T> tileLocal = inQueueX1.DeQue<T>();
    LocalTensor<T> tmpLocal = TmpBuf.Get<T>();
    LocalTensor<T> subLocal = MathBuf.Get<T>();
    LocalTensor<T> ReduceLocal = ReduceBuf.Get<T>();

    const uint32_t shape[] = { 1, this->Hdim * this->Wdim};
    uint32_t remainder = this->processDataNum;
    uint32_t off = 0;

    AscendC::TEventID eventID = GetTPipePtr()->AllocEventID<AscendC::HardEvent::V_S>();
    AscendC::SetFlag<AscendC::HardEvent::V_S>(eventID);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(eventID);
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::V_S>(eventID);

    if(prenum != 0){
        float pretmp = 0;
        for(uint32_t i = 0; i < prenum; ++i){
            float mean_val = workGm.GetValue(channelIdx * SLOT);
            pretmp += (tileLocal.GetValue(i * ALIGN32_NEED) - mean_val) * (tileLocal.GetValue(i * ALIGN32_NEED) - mean_val);
        }
        this->localSum.SetValue(channelIdx, pretmp + this->localSum.GetValue(channelIdx));
        remainder -= prenum;
        channelIdx = (channelIdx + 1)%this->Cdim;
        off += prenum;
        prenum = 0;
    }
    for(; remainder >= vecSize; ){
        uint64_t rsvdCnt = 0;//GatherMask参数，输出数量
        AscendC::GatherMask (tmpLocal, tileLocal[off * ALIGN32_NEED], this->gatherPattern, true, mask, params, rsvdCnt);
        PipeBarrier<PIPE_V>();
        Duplicate(subLocal, workGm.GetValue(channelIdx * SLOT), vecSize);//全局均值
        PipeBarrier<PIPE_V>();
        Sub(tmpLocal, tmpLocal, subLocal, vecSize);//减去均值
        PipeBarrier<PIPE_V>();
        Power(tmpLocal, tmpLocal, 2.0f, vecSize);//平方
        PipeBarrier<PIPE_V>();
        ReduceSum<float, Pattern::Reduce::AR, isReuse>(ReduceLocal, tmpLocal, shape, true);

        int32_t eventIDvToS = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S));
        AscendC::SetFlag<AscendC::HardEvent::V_S>(eventIDvToS);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(eventIDvToS);

        this->localSum.SetValue(channelIdx, this->localSum.GetValue(channelIdx) + ReduceLocal.GetValue(0));
        remainder -= vecSize;
        channelIdx = (channelIdx + 1)%this->Cdim;
        off += vecSize;
    }
    if(remainder != 0 ){
        float lasttmp = 0;
        for(uint32_t i = 0; i < remainder; ++i){
            float mean_val = workGm.GetValue(channelIdx * SLOT);
            lasttmp += (tileLocal.GetValue(off * ALIGN32_NEED + i * ALIGN32_NEED) - mean_val) * (tileLocal.GetValue(off * ALIGN32_NEED + i * ALIGN32_NEED) - mean_val);        
        }
        this->localSum.SetValue(channelIdx, lasttmp + this->localSum.GetValue(channelIdx));
        prenum = vecSize - remainder;
        remainder = 0;
    }
    inQueueX1.FreeTensor(tileLocal);
}

template <typename T>
__aicore__ inline void SyncBnTrainingUpdateV2<T>::SumAndSyncAllU()
{
    LocalTensor<float> SyncSumLocal = SyncSumBuf.Get<float>();
    Duplicate(SyncSumLocal, 0.0f, this->WorkOff);           // 清零
    for(uint32_t i = 0; i < this->Cdim; ++i){
        SyncSumLocal.SetValue(i * SLOT, this->localSum.GetValue(i));
    }

    SetAtomicAdd<float>();
    DataCopy(workGm, SyncSumLocal, this->WorkOff);   // 本地数据写回GM
    SetAtomicNone();

    SyncAll();
    AscendC::PipeBarrier<PIPE_V>();

    if (GetBlockIdx() == 0) {
        for(uint32_t i = 0; i < this->Cdim; ++i){
            workGm.SetValue(i * SLOT, workGm.GetValue(i * SLOT) / (this->Ndim * this->Hdim * this->Wdim));
            AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(workGm[i * SLOT]);
        }
    }
    SyncAll();
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void SyncBnTrainingUpdateV2<T>::SumAndSyncAllV()
{
    LocalTensor<float> SyncSumLocal = SyncSumBuf.Get<float>();
    Duplicate(SyncSumLocal, 0.0f, this->WorkOff);           // 清零
    for(uint32_t i = 0; i < this->Cdim; ++i){
        SyncSumLocal.SetValue(i * SLOT, this->localSum.GetValue(i));
    }
    SetAtomicAdd<float>();
    DataCopy(workGm[this->WorkOff], SyncSumLocal, this->WorkOff);      
    SetAtomicNone();
    
    SyncAll();
    AscendC::PipeBarrier<PIPE_V>();

    if (GetBlockIdx() == 0) {//Div只支持localtensor
        for(uint32_t i = 0; i < this->Cdim; ++i){
            workGm.SetValue(this->WorkOff + i * SLOT, workGm.GetValue(this->WorkOff + i * SLOT) / (this->Ndim * this->Hdim * this->Wdim));
            AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(workGm[this->WorkOff + i * SLOT]);
        }   
    }

    SyncAll();
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void SyncBnTrainingUpdateV2<T>::Normalization(uint32_t progress)
{
    LocalTensor<T> tileLocal = inQueueX1.DeQue<T>();
    LocalTensor<T> tileOut = outQueueY1.AllocTensor<T>();
    LocalTensor<T> subLocal = MathBuf.Get<T>();
    uint32_t remainder = this->processDataNum;
    uint32_t off = 0;
    PipeBarrier<PIPE_V>();
    if(prenum != 0){
        for(uint32_t i = 0; i < prenum; ++i){
            Duplicate(subLocal, workGm.GetValue(this->WorkOff + channelIdx * SLOT), 1);
            PipeBarrier<PIPE_V>();
            Adds(subLocal, subLocal, EPSILON, 1);
            PipeBarrier<PIPE_V>();
            Sqrt(subLocal, subLocal, 1);
            PipeBarrier<PIPE_V>();
            AscendC::TEventID eventID1 = GetTPipePtr()->AllocEventID<AscendC::HardEvent::V_S>();
            AscendC::SetFlag<AscendC::HardEvent::V_S>(eventID1);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(eventID1);
            tileOut.SetValue(i, (tileLocal.GetValue(i * ALIGN32_NEED) - workGm.GetValue(channelIdx * SLOT)) / subLocal.GetValue(0));
        }
        remainder -= prenum;
        channelIdx = (channelIdx + 1) % this->Cdim;
        off += prenum;
        prenum = 0;
    }
    for(; remainder >= vecSize; ){
        uint64_t rsvdCnt = 0;//GatherMask参数，输出数量
        AscendC::TEventID eventID2 = GetTPipePtr()->AllocEventID<AscendC::HardEvent::S_V>();
        AscendC::SetFlag<AscendC::HardEvent::S_V>(eventID2);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(eventID2);
        AscendC::GatherMask (tileOut[off], tileLocal[off * ALIGN32_NEED], this->gatherPattern, true, mask, params, rsvdCnt);
        PipeBarrier<PIPE_V>();
        Duplicate(subLocal, workGm.GetValue(channelIdx * SLOT), vecSize);
        PipeBarrier<PIPE_V>();
        Sub(tileOut[off], tileOut[off], subLocal, vecSize);
        PipeBarrier<PIPE_V>();
        LocalTensor<float> base = BaseBuf.Get<float>();
        Duplicate(base, workGm.GetValue(this->WorkOff + channelIdx * SLOT), vecSize);
        PipeBarrier<PIPE_V>();
        Adds(base, base, EPSILON, vecSize);
        PipeBarrier<PIPE_V>();
        Sqrt(base, base, vecSize);
        PipeBarrier<PIPE_V>();
        Div(tileOut[off], tileOut[off], base, vecSize);
        PipeBarrier<PIPE_V>();
        remainder -= vecSize;
        channelIdx = (channelIdx + 1)%this->Cdim;
        off += vecSize;
    }
    if(remainder != 0 ){
        for(uint32_t i = 0; i < remainder; ++i){
            Duplicate(subLocal, workGm.GetValue(this->WorkOff + channelIdx * SLOT), 1);
            PipeBarrier<PIPE_V>();
            Adds(subLocal, subLocal, EPSILON, 1);
            PipeBarrier<PIPE_V>();
            Sqrt(subLocal, subLocal, 1);
            PipeBarrier<PIPE_V>();
            AscendC::TEventID eventID3 = GetTPipePtr()->AllocEventID<AscendC::HardEvent::V_S>();
            AscendC::SetFlag<AscendC::HardEvent::V_S>(eventID3);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(eventID3);
            tileOut.SetValue(off + i , (tileLocal.GetValue(off * ALIGN32_NEED + i * ALIGN32_NEED) - workGm.GetValue(channelIdx * SLOT)) / subLocal.GetValue(0));
        }
        prenum = vecSize - remainder;
        remainder = 0;
    }
    outQueueY1.EnQue(tileOut);
    inQueueX1.FreeTensor(tileLocal);
}

template <typename T>
__aicore__ inline void SyncBnTrainingUpdateV2<T>::ReduceUProcess(){
    prenum = Starprenum;
    channelIdx = StarchannelIdx;
    int32_t loopCount = this->tileNum;
    this->processDataNum = this->tileDataNum;
    for (int32_t i = 0; i < loopCount - 1; i++) {
        CopyIn(i);
        ReduceU(i);    
    }
    this->processDataNum = this->tailDataNum;
    CopyIn(loopCount - 1);
    ReduceU(loopCount - 1);

    SumAndSyncAllU();
}

template <typename T>
__aicore__ inline void SyncBnTrainingUpdateV2<T>::ReduceVProcess(){
    prenum = Starprenum;
    channelIdx = StarchannelIdx;
    Duplicate(this->localSum, 0.0f, this->Cdim); 
    int32_t loopCount = this->tileNum;
    this->processDataNum = this->tileDataNum;
    for (int32_t i = 0; i < loopCount - 1; i++) {
        CopyIn(i);
        ReduceV(i);
    }
    this->processDataNum = this->tailDataNum;
    CopyIn(loopCount - 1);
    ReduceV(loopCount - 1);

    SumAndSyncAllV();
}

template <typename T>
__aicore__ inline void SyncBnTrainingUpdateV2<T>::NormalizationProcess(){
    prenum = Starprenum;
    channelIdx = StarchannelIdx;
    uint32_t loopCount = this->tileNum;
    this->processDataNum = this->tileDataNum;
    for (int32_t i = 0; i < loopCount - 1; i++) {
        CopyIn(i);
        Normalization(i);
        CopyOut(i);
    }
    this->processDataNum = this->tailDataNum;
    CopyIn(loopCount - 1);
    Normalization(loopCount - 1);
    CopyOut(loopCount - 1);
}

template <typename T>
__aicore__ inline void SyncBnTrainingUpdateV2<T>::UpdateRunning()
{
    if(GetBlockIdx() == 0){
        for(uint32_t i = 0; i < this->Cdim; ++i){
            float new_running_mean = this->momentum * x2Gm.GetValue(i) + (1 - this->momentum) * workGm.GetValue(i * SLOT);
            y2Gm.SetValue(i, new_running_mean);
            float new_running_var = this->momentum * x3Gm.GetValue(i) + (1 - this->momentum) * workGm.GetValue(this->WorkOff + i * SLOT);
            y3Gm.SetValue(i, new_running_var);
        }
        for(uint32_t i = 0; i < (this->Cdim * sizeof(T) / DATA_CACHE_CLEAN_NEED) + 1 ; ++i){
            AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(y2Gm[i * SLOT]);
            AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(y3Gm[i * SLOT]);
        }
    }
}

template <typename T>
__aicore__ inline void SyncBnTrainingUpdateV2<T>::Process()
{
    ReduceUProcess();
    ReduceVProcess();
    NormalizationProcess();
    UpdateRunning();
}
}
#endif // SYNC_BN_TRAINING_UPDATE_H