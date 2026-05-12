/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Shi Xiangyang <@shi-xiangyang225>
 * - Pei Haobo<@xiaopei-1>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lp_norm_v3.h
 * \brief
 */
#ifndef __LP_NORM_V3_H__
#define __LP_NORM_V3_H__
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lp_norm_v3_tiling_data.h"
#include "lp_norm_v3_tiling_key.h"

namespace NsLpNormV3 {
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t DATA_CACHE_CLEAN_NEED = 64 ;//64B
constexpr int32_t SLOT_STRIDE = DATA_CACHE_CLEAN_NEED / sizeof(float);
constexpr uint32_t POS_INF_BITS = 0x7F800000u;
constexpr uint32_t NEG_INF_BITS = 0xFF800000u;
template <typename T, uint32_t schMode>
class LpNormV3 {
public:
    __aicore__ inline LpNormV3(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, LpNormV3TilingData* tilingData);
    __aicore__ inline void Process();
private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Reduce(int32_t progress);
    __aicore__ inline void ReduceInf(int32_t progress);
    __aicore__ inline void SumAndSyncAll();
    __aicore__ inline void SumAndSyncAllInf();
    __aicore__ inline void Normalization(int32_t progress);
    __aicore__ inline void NormalProcess();
    __aicore__ inline void InfProcess();
    __aicore__ inline bool IsPosInf(float val);
    __aicore__ inline bool IsNegInf(float val);
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    GlobalTensor<float> workGm;
    TBuf<TPosition::VECCALC> poweredBuf;  // 存储abs(x)^p结果
    TBuf<TPosition::VECCALC> tmp_norm;    // 存储范数向量
    TBuf<TPosition::VECCALC> tmp_base;    // 存储sum+epsilon
    TBuf<TPosition::VECCALC> row_base;
    TBuf<TPosition::VECCALC> col_base;
    TBuf<TPosition::VECCALC> all_base;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
    uint32_t globalBufferIndex;
    bool isposinf;
    bool isneginf;
    float p;
    uint32_t rows;
    uint32_t cols;
    float epsilon = 1e-6;
    LocalTensor<float> localSum;
};
template <typename T, uint32_t schMode>
__aicore__ inline bool LpNormV3<T, schMode>::IsPosInf(float val)
{
    uint32_t bits = *reinterpret_cast<uint32_t*>(&val);
    return bits == POS_INF_BITS;
}

template <typename T, uint32_t schMode>
__aicore__ inline bool LpNormV3<T, schMode>::IsNegInf(float val)
{
    uint32_t bits = *reinterpret_cast<uint32_t*>(&val);
    return bits == NEG_INF_BITS;
}

template <typename T, uint32_t schMode>
__aicore__ inline void LpNormV3<T, schMode>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, LpNormV3TilingData* tilingData)
{
    ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
    uint32_t coreNum = GetBlockIdx();
    this->globalBufferIndex = tilingData->bigCoreDataNum * GetBlockIdx();
    this->tileDataNum = tilingData->tileDataNum;
    this->p = tilingData->p;
    this->rows = tilingData->rows;
    this->cols = tilingData->cols;
    this->isposinf = IsPosInf(tilingData->p);
    this->isneginf = IsNegInf(tilingData->p);
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
    xGm.SetGlobalBuffer((__gm__ T*)x + globalBufferIndex, this->coreDataNum);
    yGm.SetGlobalBuffer((__gm__ T*)y + globalBufferIndex, this->coreDataNum);
    // 全局工作区内存绑定（严格按axis分配）
    uint32_t workGmSize = 0;
    if constexpr (schMode == LP_NORM_AXIS_NONE){
        workGmSize = 1 * SLOT_STRIDE;
    }else if constexpr (schMode == LP_NORM_AXIS_0){
        workGmSize = rows * SLOT_STRIDE;
    }else if constexpr (schMode == LP_NORM_AXIS_1){
        workGmSize = cols * SLOT_STRIDE;
    }
    //多核操作时，更新全局缓存以64B为一个整体，若多核在64B内同时操作会导致随机覆写，故workGm设置大小为范数数量*16（范数以float存储）
    workGm.SetGlobalBuffer((__gm__ float*)workspace , workGmSize);

    // 分配本地缓冲区（双缓冲）
    pipe.InitBuffer(inQueue, BUFFER_NUM, this->tileDataNum * sizeof(T));
    pipe.InitBuffer(outQueue, BUFFER_NUM, this->tileDataNum * sizeof(T));
    // 初始化计算缓冲区
    pipe.InitBuffer(poweredBuf, tileDataNum * sizeof(float));  // 存储abs(x)^p结果
   
    if constexpr (schMode == LP_NORM_AXIS_NONE){
        pipe.InitBuffer(tmp_norm, tileDataNum * sizeof(T));    // 存储范数结果（向量归一化适配）
    }else{
        pipe.InitBuffer(tmp_norm, tileDataNum * sizeof(float));
    }

    pipe.InitBuffer(tmp_base, workGmSize);             // 存储sum+epsilon（范数计算用）+ 原子加操作临时缓冲
    if constexpr (schMode == LP_NORM_AXIS_0) {
        pipe.InitBuffer(row_base, rows * sizeof(float));
        this->localSum = row_base.Get<float>();
        if(this->isposinf){
            Duplicate(this->localSum, -1 * this->p, rows);
        }else if(this->isneginf){
            Duplicate(this->localSum, -1 * this->p, rows);
        }else{
            Duplicate(this->localSum, 0.0f, rows);
        }
    } else if constexpr (schMode == LP_NORM_AXIS_1) {
        pipe.InitBuffer(col_base, cols * sizeof(float));
        this->localSum = col_base.Get<float>();
        if(this->isposinf){
            Duplicate(this->localSum, -1 * this->p, cols);
        }else if(this->isneginf){
            Duplicate(this->localSum, -1 * this->p, cols);
        }else{
            Duplicate(this->localSum, 0.0f, cols);
        }
    }else if constexpr (schMode == LP_NORM_AXIS_NONE){
        pipe.InitBuffer(all_base, 8 * sizeof(float));
        this->localSum = all_base.Get<float>();
        if(this->isposinf){
            Duplicate(this->localSum, -1 * this->p, 1);
        }else if(this->isneginf){
            Duplicate(this->localSum, -1 * this->p, 1);
        }else{
            Duplicate(this->localSum, 0.0f, 1);
        }
    }
}

template <typename T, uint32_t schMode>
__aicore__ inline void LpNormV3<T, schMode>::CopyIn(int32_t progress)
{
    LocalTensor<T> xLocal = inQueue.AllocTensor<T>();
    DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processDataNum);
    inQueue.EnQue(xLocal);
}

template <typename T, uint32_t schMode>
__aicore__ inline void LpNormV3<T, schMode>::CopyOut(int32_t progress)
{
    LocalTensor<T> yLocal = outQueue.DeQue<T>();
    DataCopy(yGm[progress * this->tileDataNum], yLocal, this->processDataNum);
    outQueue.FreeTensor(yLocal);
}

template <typename T, uint32_t schMode>
__aicore__ inline void LpNormV3<T, schMode>::Reduce(int32_t progress)
{
    LocalTensor<T> tileLocal = inQueue.DeQue<T>();
    // half -> float 转换
    LocalTensor<float> tileFloat = poweredBuf.Get<float>();
    if constexpr ( IsSameType<T, half>::value){
        Cast(tileFloat, tileLocal, RoundMode::CAST_NONE, this->processDataNum);
    }else{
        tileFloat = tileLocal.template ReinterpretCast<float>();
    }
    // abs^p
    Abs(tileFloat, tileFloat, this->processDataNum);
    Power(tileFloat, tileFloat, this->p, this->processDataNum);
    // 累加到localSum
    if constexpr (schMode == LP_NORM_AXIS_NONE){
        float tileSum = 0.0f;
        for (uint32_t i = 0; i < this->processDataNum; ++i) {
            tileSum += tileFloat.GetValue(i);
        }
        this->localSum.SetValue(0, localSum.GetValue(0) + tileSum);
    }else if constexpr (schMode == LP_NORM_AXIS_0){
        for (uint32_t i = 0; i < this->processDataNum; ++i) {
            uint32_t globalIdx = globalBufferIndex + progress * this->tileDataNum + i;
            uint32_t rowIdx = globalIdx / this->cols;
            float prev = localSum.GetValue(rowIdx);
            localSum.SetValue(rowIdx, prev + tileFloat.GetValue(i));
        }
    }else if constexpr (schMode == LP_NORM_AXIS_1){
        for (uint32_t i = 0; i < this->processDataNum; ++i) {
            uint32_t globalIdx = globalBufferIndex + progress * this->tileDataNum + i;
            uint32_t colIdx = globalIdx % this->cols;
            float prev = localSum.GetValue(colIdx);
            localSum.SetValue(colIdx, prev + tileFloat.GetValue(i));
        }
    }
    inQueue.FreeTensor(tileLocal);
}
template <typename T, uint32_t schMode>
__aicore__ inline void LpNormV3<T, schMode>::ReduceInf(int32_t progress)
{
    LocalTensor<T> tileLocal = inQueue.DeQue<T>();
    // half -> float 转换
    LocalTensor<float> tileFloat = poweredBuf.Get<float>();
    if constexpr ( IsSameType<T, half>::value){
        Cast(tileFloat, tileLocal, RoundMode::CAST_NONE, this->processDataNum);
    }else{
        tileFloat = tileLocal.template ReinterpretCast<float>();
    }
    Abs(tileFloat, tileFloat, this->processDataNum);
    // 累加到localSum
    if constexpr (schMode == LP_NORM_AXIS_NONE){
        if(this->isposinf){//正无穷范数
            uint32_t negInfTmp = NEG_INF_BITS;
            float tileSum = *reinterpret_cast<float*>(&negInfTmp);
            for (uint32_t i = 0; i < this->processDataNum; ++i) {
                if( tileFloat.GetValue(i) > tileSum ){
                    tileSum = tileFloat.GetValue(i);
                }
            }
            this->localSum.SetValue(0, tileSum > localSum.GetValue(0) ? tileSum : localSum.GetValue(0));
        }else{
            uint32_t posInfTmp = POS_INF_BITS;
            float tileSum = *reinterpret_cast<float*>(&posInfTmp);
            for (uint32_t i = 0; i < this->processDataNum; ++i) {
                if( tileFloat.GetValue(i) < tileSum ){
                    tileSum = tileFloat.GetValue(i);
                }
            }
            this->localSum.SetValue(0, tileSum < localSum.GetValue(0) ? tileSum : localSum.GetValue(0));
        }
        
    }else {
        if(this->isposinf){
            if constexpr (schMode == LP_NORM_AXIS_0){
                for (uint32_t i = 0; i < this->processDataNum; ++i) {
                    uint32_t globalIdx = globalBufferIndex + progress * this->tileDataNum + i;
                    uint32_t Idx = globalIdx / this->cols;
                    float prev = localSum.GetValue(Idx);
                    localSum.SetValue(Idx, prev > tileFloat.GetValue(i) ? prev : tileFloat.GetValue(i));
                }
            }else{
                for (uint32_t i = 0; i < this->processDataNum; ++i) {
                    uint32_t globalIdx = globalBufferIndex + progress * this->tileDataNum + i;
                    uint32_t Idx = globalIdx % this->cols;
                    float prev = localSum.GetValue(Idx);
                    localSum.SetValue(Idx, prev > tileFloat.GetValue(i) ? prev : tileFloat.GetValue(i));
                }
            }
        }else{
            if constexpr (schMode == LP_NORM_AXIS_0){
                for (uint32_t i = 0; i < this->processDataNum; ++i) {
                    uint32_t globalIdx = globalBufferIndex + progress * this->tileDataNum + i;
                    uint32_t Idx = globalIdx / this->cols;
                    float prev = localSum.GetValue(Idx);
                    localSum.SetValue(Idx, prev < tileFloat.GetValue(i) ? prev : tileFloat.GetValue(i));
                }
            }else{
                for (uint32_t i = 0; i < this->processDataNum; ++i) {
                    uint32_t globalIdx = globalBufferIndex + progress * this->tileDataNum + i;
                    uint32_t Idx = globalIdx % this->cols;
                    float prev = localSum.GetValue(Idx);
                    localSum.SetValue(Idx, prev < tileFloat.GetValue(i) ? prev : tileFloat.GetValue(i));
                }
            }
        }
    }
    inQueue.FreeTensor(tileLocal);
}

template <typename T, uint32_t schMode>
__aicore__ inline void LpNormV3<T, schMode>::SumAndSyncAllInf()
{
    LocalTensor<float> base = tmp_base.Get<float>();
    uint32_t totalWorkSize = SLOT_STRIDE;
    if constexpr (schMode == LP_NORM_AXIS_NONE){
        Duplicate(base, 0.0f, 8);           // 清零
        base.SetValue(0, localSum.GetValue(0) + this->epsilon);      // 只放第一个值
        if(this->isposinf){//max
            AscendC::SetAtomicMax<float>();
        }else{//min
            AscendC::SetAtomicMin<float>();
        }
        SetAtomicAdd<float>();
        DataCopy(workGm[0], base, 8);       // 搬 8 个 float (32B 对齐)
        SetAtomicNone();
    }else if constexpr (schMode == LP_NORM_AXIS_0){
        totalWorkSize *= this->rows;
        for (uint32_t r = 0; r < this->rows; ++r) {
            uint32_t index = r * SLOT_STRIDE;
            base.SetValue(index, localSum.GetValue(r)+ this->epsilon);
        }
        if(this->isposinf){//max
            AscendC::SetAtomicMax<float>();
        }else{//min
            AscendC::SetAtomicMin<float>();
        }
        DataCopy(workGm, base, totalWorkSize);
        SetAtomicNone();
    }else if constexpr (schMode == LP_NORM_AXIS_1){
        totalWorkSize *= this->cols;
        for (uint32_t c = 0; c < this->cols; ++c) {
            uint32_t index = c * SLOT_STRIDE;
            base.SetValue(index, localSum.GetValue(c)+ this->epsilon);
        }
        if(this->isposinf){//max
            AscendC::SetAtomicMax<float>();
        }else{//min
            AscendC::SetAtomicMin<float>();
        }
        DataCopy(workGm, base, totalWorkSize);
        SetAtomicNone();
    }
    SyncAll();
    PipeBarrier<PIPE_V>();
    // 无穷范数直接使用最大/最小值，无需额外运算
    if ( GetBlockIdx() == 0) {//核0刷新缓存
        if constexpr (schMode == LP_NORM_AXIS_NONE){
            DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(workGm[0]);
        }else if constexpr (schMode == LP_NORM_AXIS_0){
            for (uint32_t r = 0; r < this->rows; ++r) {
                DataCacheCleanAndInvalid<float,CacheLine::SINGLE_CACHE_LINE,DcciDst::CACHELINE_OUT>(workGm[r * SLOT_STRIDE]);
            }
        }else if constexpr (schMode == LP_NORM_AXIS_1){
            for (uint32_t c = 0; c < this->cols; ++c) {
                DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(workGm[c * SLOT_STRIDE]);
            }
        }
    }
    PipeBarrier<PIPE_V>();
    SyncAll();
}

template <typename T, uint32_t schMode>
__aicore__ inline void LpNormV3<T, schMode>::SumAndSyncAll()
{
    LocalTensor<float> base = tmp_base.Get<float>();
     uint32_t totalWorkSize = SLOT_STRIDE;
    if constexpr (schMode == LP_NORM_AXIS_NONE){
        Duplicate(base, 0.0f, 8);           // 清零
        base.SetValue(0, localSum.GetValue(0));      // 只放第一个值
        SetAtomicAdd<float>();
        DataCopy(workGm[0], base, 8);       // 搬 8 个 float (32B 对齐)
        SetAtomicNone();
    }else if constexpr (schMode == LP_NORM_AXIS_0){
        totalWorkSize *= this->rows;
        for (uint32_t r = 0; r < this->rows; ++r) {
            uint32_t index = r * SLOT_STRIDE;
            base.SetValue(index, localSum.GetValue(r));
        }
        SetAtomicAdd<float>();
        DataCopy(workGm, base, totalWorkSize);
        SetAtomicNone();
    }else if constexpr (schMode == LP_NORM_AXIS_1){
        totalWorkSize *= this->cols;
        for (uint32_t c = 0; c < this->cols; ++c) {
            uint32_t index = c * SLOT_STRIDE;
            base.SetValue(index, localSum.GetValue(c));
        }
        SetAtomicAdd<float>();
        DataCopy(workGm, base, totalWorkSize);  // 写入到独立槽位
        SetAtomicNone();
    }
    SyncAll();
    PipeBarrier<PIPE_V>();
    if ( GetBlockIdx() == 0) {//核0计算开方

        if constexpr (schMode == LP_NORM_AXIS_NONE){
            base.SetValue(0, workGm.GetValue(0) + this->epsilon);
            Ln(base, base, 1);
            Muls(base, base, 1.0f / this->p, 1);
            Exp(base, base, 1);
            workGm.SetValue(0, base.GetValue(0));
            DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(workGm[0]);
        }else if constexpr (schMode == LP_NORM_AXIS_0){
            for (uint32_t r = 0; r < this->rows; ++r) {
                uint32_t index = r * SLOT_STRIDE;
                base.SetValue(0, workGm.GetValue(r * SLOT_STRIDE) + this->epsilon);
                Ln(base, base, 1);
                Muls(base, base, 1.0f / this->p, 1);
                Exp(base, base, 1);
                workGm.SetValue(r * SLOT_STRIDE, base.GetValue(0));
                DataCacheCleanAndInvalid<float,CacheLine::SINGLE_CACHE_LINE,DcciDst::CACHELINE_OUT>(workGm[r * SLOT_STRIDE]);
            }
        }else if constexpr (schMode == LP_NORM_AXIS_1){
            for (uint32_t c = 0; c < this->cols; ++c) {
                uint32_t index = c * SLOT_STRIDE;
                base.SetValue(0, workGm.GetValue(index) + this->epsilon);
                Ln(base, base, 1);
                Muls(base, base, 1.0f / this->p, 1);
                Exp(base, base, 1);
                workGm.SetValue(index, base.GetValue(0));
                DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(workGm[c * SLOT_STRIDE]);
            }
        }
    }
    PipeBarrier<PIPE_V>();
    SyncAll();
}

template <typename T, uint32_t schMode>
__aicore__ inline void LpNormV3<T, schMode>::Normalization(int32_t progress)
{
    LocalTensor<T> tileLocal = inQueue.DeQue<T>();
    LocalTensor<T> tileOut = outQueue.AllocTensor<T>();
    if constexpr (schMode == LP_NORM_AXIS_NONE){
        LocalTensor<T> normTensor = tmp_norm.Get<T>();
        Duplicate(normTensor, static_cast<T>(workGm.GetValue(0)), this->processDataNum);
        Div(tileOut, tileLocal, normTensor, this->processDataNum);
    }else if constexpr (schMode == LP_NORM_AXIS_0){
        for (uint32_t i = 0; i < this->processDataNum; ++i) {
            uint32_t globalIdx = globalBufferIndex + progress * this->tileDataNum + i;
            uint32_t rowIdx = globalIdx / this->cols;
            float norm = workGm.GetValue(rowIdx * SLOT_STRIDE);
            tileOut.SetValue(i, static_cast<T>( static_cast<float>( tileLocal.GetValue(i) ) / norm ));
        }
    }else if constexpr (schMode == LP_NORM_AXIS_1){
        for (uint32_t i = 0; i < this->processDataNum; ++i) {
            uint32_t globalIdx = globalBufferIndex + progress * this->tileDataNum + i;
            uint32_t colIdx = globalIdx % this->cols;
            float norm = workGm.GetValue(colIdx * SLOT_STRIDE);
            tileOut.SetValue(i, static_cast<T>( static_cast<float>( tileLocal.GetValue(i) ) / norm ));
        }
    }
    outQueue.EnQue(tileOut);
    inQueue.FreeTensor(tileLocal);
}

template <typename T, uint32_t schMode>
__aicore__ inline void LpNormV3<T, schMode>::NormalProcess()
{
    int32_t loopCount = this->tileNum;
    this->processDataNum = this->tileDataNum;
    for (int32_t i = 0; i < loopCount-1; i++) {
        CopyIn(i);
        Reduce(i);
    }
    this->processDataNum = this->tailDataNum;
    CopyIn(loopCount-1);
    Reduce(loopCount-1);

    SumAndSyncAll();

    this->processDataNum = this->tileDataNum;
    for (int32_t i = 0; i < loopCount-1; i++) {
        CopyIn(i);
        Normalization(i);
        CopyOut(i);
    }
    this->processDataNum = this->tailDataNum;
    CopyIn(loopCount-1);
    Normalization(loopCount-1);
    CopyOut(loopCount-1);
}

template <typename T, uint32_t schMode>
__aicore__ inline void LpNormV3<T, schMode>::InfProcess()
{
    int32_t loopCount = this->tileNum;
    this->processDataNum = this->tileDataNum;
    for (int32_t i = 0; i < loopCount-1; i++) {
        CopyIn(i);
        ReduceInf(i);
    }
    this->processDataNum = this->tailDataNum;
    CopyIn(loopCount-1);
    ReduceInf(loopCount-1);

    SumAndSyncAllInf();

    this->processDataNum = this->tileDataNum;
    for (int32_t i = 0; i < loopCount-1; i++) {
        CopyIn(i);
        Normalization(i);
        CopyOut(i);
    }
    this->processDataNum = this->tailDataNum;
    CopyIn(loopCount-1);
    Normalization(loopCount-1);
    CopyOut(loopCount-1);
}

template <typename T, uint32_t schMode>
__aicore__ inline void LpNormV3<T, schMode>::Process()
{
    if( !(this->isposinf) && !(this->isneginf)){ //当P不为负无穷和正无穷时
        NormalProcess();
    }else{
        InfProcess();
    }
}
} // namespace NsLpNormV3
#endif // LP_NORM_V3_H