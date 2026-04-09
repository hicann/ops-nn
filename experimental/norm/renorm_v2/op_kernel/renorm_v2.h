/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Cao Xiaojuan <@c15503545287>
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
 * \file renorm_v2.h
 * \brief 
 */
#ifndef __RENORM_V2_H__
#define __RENORM_V2_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "renorm_v2_tiling_data.h"
#include "renorm_v2_tiling_key.h"

namespace NsRenormV2 {
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t DATA_CACHE_CLEAN_NEED = 64;  // 64B
constexpr int32_t SLOT_STRIDE = DATA_CACHE_CLEAN_NEED / sizeof(float);

template <typename T>
class RenormV2 {
public:
    __aicore__ inline RenormV2(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, RenormV2TilingData* tilingData);
    __aicore__ inline void Process();
    
private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Reduce(int32_t progress, int32_t axis);
    __aicore__ inline void SumAndSyncAll(int32_t axis);
    __aicore__ inline void RenormScale(int32_t progress, int32_t axis);
    
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    GlobalTensor<float> workGm;
    TBuf<TPosition::VECCALC> poweredBuf;   // 存储abs(x)^p结果
    TBuf<TPosition::VECCALC> tmp_base;     // 临时缓冲区
    TBuf<TPosition::VECCALC> row_base;     // 行累加缓冲区
    TBuf<TPosition::VECCALC> col_base;     // 列累加缓冲区
    TBuf<TPosition::VECCALC> scale_buf;    // 缩放因子缓冲区
    
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
    uint32_t globalBufferIndex;
    
    float p;           // 范数阶数
    int32_t axis;      // 范数计算轴（0或1）
    uint32_t rows;     // 行数
    uint32_t cols;     // 列数
    float epsilon = 1e-6;  // 防止除零的小常数
    float max_norm;    // 最大范数值
    
    LocalTensor<float> localSum;
};

template <typename T>
__aicore__ inline void RenormV2<T>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, RenormV2TilingData* tilingData)
{
    ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
    uint32_t coreNum = GetBlockIdx();
    
    this->globalBufferIndex = tilingData->bigCoreDataNum * GetBlockIdx();
    this->tileDataNum = tilingData->tileDataNum;
    this->p = tilingData->p;
    this->axis = tilingData->axis;
    this->rows = tilingData->rows;
    this->cols = tilingData->cols;
    this->max_norm = tilingData->max_norm;
    
    if (coreNum < tilingData->tailBlockNum) {
        this->coreDataNum = tilingData->bigCoreDataNum;
        this->tileNum = tilingData->finalBigTileNum;
        this->tailDataNum = tilingData->bigTailDataNum;
    } else {
        this->coreDataNum = tilingData->smallCoreDataNum;
        this->tileNum = tilingData->finalSmallTileNum;
        this->tailDataNum = tilingData->smallTailDataNum;
        globalBufferIndex -= (tilingData->bigCoreDataNum - tilingData->smallCoreDataNum) * 
                            (GetBlockIdx() - tilingData->tailBlockNum);
    }
    
    xGm.SetGlobalBuffer((__gm__ T*)x + globalBufferIndex, this->coreDataNum);
    yGm.SetGlobalBuffer((__gm__ T*)y + globalBufferIndex, this->coreDataNum);
    
    // 全局工作区内存分配
    uint32_t workGmSize = 0;
    if (axis == 0) {
        workGmSize = rows * SLOT_STRIDE;  // 每行一个64B槽位
    } else if (axis == 1) {
        workGmSize = cols * SLOT_STRIDE;  // 每列一个64B槽位
    }
    
    workGm.SetGlobalBuffer((__gm__ float*)workspace, workGmSize);
    
    // 分配本地缓冲区（双缓冲）
    pipe.InitBuffer(inQueue, BUFFER_NUM, this->tileDataNum * sizeof(T));
    pipe.InitBuffer(outQueue, BUFFER_NUM, this->tileDataNum * sizeof(T));
    
    // 初始化计算缓冲区
    pipe.InitBuffer(poweredBuf, tileDataNum * sizeof(float));
    pipe.InitBuffer(tmp_base, workGmSize);
    pipe.InitBuffer(scale_buf, tileDataNum * sizeof(float));
    
    // 初始化累加缓冲区
    if (axis == 0) {
        pipe.InitBuffer(row_base, rows * sizeof(float));
        this->localSum = row_base.Get<float>();
        Duplicate(this->localSum, 0.0f, rows);
    } else if (axis == 1) {
        pipe.InitBuffer(col_base, cols * sizeof(float));
        this->localSum = col_base.Get<float>();
        Duplicate(this->localSum, 0.0f, cols);
    }
}

template <typename T>
__aicore__ inline void RenormV2<T>::CopyIn(int32_t progress)
{
    LocalTensor<T> xLocal = inQueue.AllocTensor<T>();
    DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processDataNum);
    inQueue.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void RenormV2<T>::CopyOut(int32_t progress)
{
    LocalTensor<T> yLocal = outQueue.DeQue<T>();
    DataCopy(yGm[progress * this->tileDataNum], yLocal, this->processDataNum);
    outQueue.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void RenormV2<T>::Reduce(int32_t progress, int32_t axis)
{
    LocalTensor<T> tileLocal = inQueue.DeQue<T>();
    LocalTensor<float> tileFloat = poweredBuf.Get<float>();
    
    // 转换为float
    if constexpr (IsSameType<T, half>::value) {
        Cast(tileFloat, tileLocal, RoundMode::CAST_NONE, this->processDataNum);
    } else {
        tileFloat = tileLocal;
    }
    
    // 计算abs(x)^p
    Abs(tileFloat, tileFloat, this->processDataNum);
    Power(tileFloat, tileFloat, this->p, this->processDataNum);
    
    // 累加到localSum
    if (axis == 0) {
        // 行累加
        for (uint32_t i = 0; i < this->processDataNum; ++i) {
            uint32_t globalIdx = globalBufferIndex + progress * this->tileDataNum + i;
            uint32_t rowIdx = globalIdx / this->cols;
            float prev = localSum.GetValue(rowIdx);
            localSum.SetValue(rowIdx, prev + tileFloat.GetValue(i));
        }
    } else if (axis == 1) {
        // 列累加
        for (uint32_t i = 0; i < this->processDataNum; ++i) {
            uint32_t globalIdx = globalBufferIndex + progress * this->tileDataNum + i;
            uint32_t colIdx = globalIdx % this->cols;
            float prev = localSum.GetValue(colIdx);
            localSum.SetValue(colIdx, prev + tileFloat.GetValue(i));
        }
    }
    
    inQueue.FreeTensor(tileLocal);
}

template <typename T>
__aicore__ inline void RenormV2<T>::SumAndSyncAll(int32_t axis)
{
    LocalTensor<float> base = tmp_base.Get<float>();
    uint32_t totalWorkSize = SLOT_STRIDE;
    if (axis == 0){
        totalWorkSize *= this->rows;
        for (uint32_t r = 0; r < this->rows; ++r) {
            uint32_t index = r * SLOT_STRIDE;
            base.SetValue(index, localSum.GetValue(r));
        }
        SetAtomicAdd<float>();
        DataCopy(workGm, base, totalWorkSize);
        SetAtomicNone();
    }else if(axis == 1){ 
        totalWorkSize *= this->cols;
        for (uint32_t c = 0; c < this->cols; ++c) {
            uint32_t index = c * SLOT_STRIDE;
            base.SetValue(index, localSum.GetValue(c));
        }
        SetAtomicAdd<float>();
        DataCopy(workGm, base, totalWorkSize);  // 写入到独立槽位
        SetAtomicNone();
    }

    PipeBarrier<PIPE_V>();
    if ( GetBlockIdx() == 0) {//核0计算开方
        if( axis == 0){
            for (uint32_t r = 0; r < this->rows; ++r) {
                uint32_t index = r * SLOT_STRIDE;
                base.SetValue(0, workGm.GetValue(r * SLOT_STRIDE));
                Ln(base, base, 1);
                Muls(base, base, 1.0f / this->p, 1);
                Exp(base, base, 1);
                workGm.SetValue(r * SLOT_STRIDE, base.GetValue(0));
                DataCacheCleanAndInvalid<float,CacheLine::SINGLE_CACHE_LINE,DcciDst::CACHELINE_OUT>(workGm[r * SLOT_STRIDE]);
            }
        }else if ( axis == 1){
            for (uint32_t c = 0; c < this->cols; ++c) {
                uint32_t index = c * SLOT_STRIDE;
                base.SetValue(0, workGm.GetValue(index));
                Ln(base, base, 1);
                Muls(base, base, 1.0f / this->p, 1);
                Exp(base, base, 1);
                workGm.SetValue(index, base.GetValue(0));
                DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(workGm[c * SLOT_STRIDE]);
            }
        }
    }
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void RenormV2<T>::RenormScale(int32_t progress, int32_t axis)
{
    LocalTensor<T> tileLocal = inQueue.DeQue<T>();
    LocalTensor<T> tileOut = outQueue.AllocTensor<T>();
    LocalTensor<float> scaleTensor = scale_buf.Get<float>();
    
    if (axis == 0) {
        // 行方向范数约束
        for (uint32_t i = 0; i < this->processDataNum; ++i) {
            uint32_t globalIdx = globalBufferIndex + progress * this->tileDataNum + i;
            uint32_t rowIdx = globalIdx / this->cols;
            float norm = workGm.GetValue(rowIdx * SLOT_STRIDE);
            float scale = 1.0f;
            
            // 如果范数超过最大范数，则进行缩放
            if (norm > this->max_norm) {
                scale = this->max_norm / norm ;
            }
            
            // 计算缩放因子
            scaleTensor.SetValue(i, scale);
        }
        
        // 应用缩放
        if constexpr (IsSameType<T, half>::value) {
            LocalTensor<float> tileFloat = poweredBuf.Get<float>();
            Cast(tileFloat, tileLocal, RoundMode::CAST_NONE, this->processDataNum);
            AscendC::PipeBarrier<PIPE_V>();
            Mul(tileFloat, tileFloat, scaleTensor, this->processDataNum);
            AscendC::PipeBarrier<PIPE_V>();
            Cast(tileOut, tileFloat, RoundMode::CAST_NONE, this->processDataNum);
        } else {
            Mul(tileOut, tileLocal, scaleTensor, this->processDataNum);
        }
        
    } else if (axis == 1) {
        // 列方向范数约束
        for (uint32_t i = 0; i < this->processDataNum; ++i) {
            uint32_t globalIdx = globalBufferIndex + progress * this->tileDataNum + i;
            uint32_t colIdx = globalIdx % this->cols;
            float norm = workGm.GetValue(colIdx * SLOT_STRIDE);
            float scale = 1.0f;
            
            // 如果范数超过最大范数，则进行缩放
            if (norm > this->max_norm) {
                scale = this->max_norm / norm;
            }
            
            // 计算缩放因子（暂时存储）
            scaleTensor.SetValue(i, scale);
        }
        
        // 应用缩放
        if constexpr (IsSameType<T, half>::value) {
            LocalTensor<float> tileFloat = poweredBuf.Get<float>();
            Cast(tileFloat, tileLocal, RoundMode::CAST_NONE, this->processDataNum);
            AscendC::PipeBarrier<PIPE_V>();
            Mul(tileFloat, tileFloat, scaleTensor, this->processDataNum);
            AscendC::PipeBarrier<PIPE_V>();
            Cast(tileOut, tileFloat, RoundMode::CAST_NONE, this->processDataNum);
        } else {
            Mul(tileOut, tileLocal, scaleTensor, this->processDataNum);
        }
    }
    
    outQueue.EnQue(tileOut);
    inQueue.FreeTensor(tileLocal);
}

template <typename T>
__aicore__ inline void RenormV2<T>::Process()
{
    // Renorm处理流程
    int32_t loopCount = this->tileNum;
    this->processDataNum = this->tileDataNum;
    
    // 阶段1：计算范数（累加abs(x)^p）
    for (int32_t i = 0; i < loopCount-1; i++) {
        CopyIn(i);
        Reduce(i, this->axis);
    }
    this->processDataNum = this->tailDataNum;
    CopyIn(loopCount-1);
    Reduce(loopCount-1, this->axis);
    
    // 阶段2：计算范数值并同步到所有核心
    SumAndSyncAll(this->axis);
    
    // 阶段3：应用Renorm缩放
    this->processDataNum = this->tileDataNum;
    for (int32_t i = 0; i < loopCount-1; i++) {
        CopyIn(i);
        RenormScale(i, this->axis);
        CopyOut(i);
    }
    this->processDataNum = this->tailDataNum;
    CopyIn(loopCount-1);
    RenormScale(loopCount-1, this->axis);
    CopyOut(loopCount-1);
}

} // namespace NsRenormV2
#endif // __RENORM_V2_H__ 