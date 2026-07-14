/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
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
 * \file layer_normalization_grad.h
 * \brief LayerNormalizationGrad 算子 Kernel 实现
 *
 * 计算公式：
 *   x_hat[i]  = (x[i] - mean[i]) * rstd[i]
 *   dxhat[i]  = dy[i] * gamma
 *   ds[i]     = sum(dxhat[i] * x_hat[i])
 *   db[i]     = sum(dxhat[i])
 *   dx[i]     = rstd[i] * (dxhat[i] - (db[i] + ds[i] * x_hat[i]) / D)
 *   dgamma    = sum(dy * x_hat, axis=0)
 *   dbeta     = sum(dy, axis=0)
 *
 * KernelLayerNormalizationGrad<float> 为直接 float 实现，
 * KernelLayerNormalizationGradCast<T, Mode> 为 half/bfloat16 的 float 累加器实现。
 */

#ifndef LAYER_NORMALIZATION_GRAD_H_
#define LAYER_NORMALIZATION_GRAD_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "layer_normalization_grad_tiling_data.h"
#include "layer_normalization_grad_tiling_key.h"

namespace NsLayerNormalizationGrad {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

__aicore__ inline void WaitVectorScalarSync()
{
    int32_t eventId = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S));
    AscendC::SetFlag<AscendC::HardEvent::V_S>(eventId);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(eventId);
}

__aicore__ inline void WaitVectorToMte3Sync()
{
    int32_t eventId = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventId);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventId);
}

__aicore__ inline void WaitVectorToMte2Sync()
{
    int32_t eventId = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE2));
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventId);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventId);
}

__aicore__ inline void WaitMte2ToVectorSync()
{
    int32_t eventId = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventId);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventId);
}

__aicore__ inline void WaitMte3ToVectorSync()
{
    int32_t eventId = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventId);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventId);
}

__aicore__ inline void WaitMte3ToMte2Sync()
{
    int32_t eventId = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
}

template <typename T>
__aicore__ inline float ToFloatValue(const T& value)
{
    if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
        return AscendC::ToFloat(value);
    } else {
        return static_cast<float>(value);
    }
}

// ============================================================================
// KernelLayerNormalizationGrad<float> : 直接 float 实现
// ============================================================================
template <typename T>
class KernelLayerNormalizationGrad {
public:
    __aicore__ inline KernelLayerNormalizationGrad() {}

    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR x, GM_ADDR gamma, GM_ADDR mean, GM_ADDR rstd, GM_ADDR dx,
                                GM_ADDR dgamma, GM_ADDR dbeta, GM_ADDR workspace,
                                const LayerNormalizationGradTilingData* tilingData)
    {
        this->N = tilingData->N;
        this->D = tilingData->D;
        this->DPadded = tilingData->DPadded;
        this->bufferNum = tilingData->bufferNum;
        this->colSplitMode = tilingData->colSplitMode;
        this->tileCols = tilingData->tileCols;
        this->tileColsAligned = tilingData->tileColsAligned;
        this->numColTiles = tilingData->numColTiles;
        this->workspaceTileStride = tilingData->workspaceTileStride;
        this->maxCoreRows = tilingData->maxCoreRows;

        this->invD = 1.0f / static_cast<float>(static_cast<int32_t>(this->D));

        uint64_t coreIdx = static_cast<uint64_t>(AscendC::GetBlockIdx());
        this->multiCore = (tilingData->usedCoreNum > 1);

        if (!this->multiCore) {
            this->startRow = 0;
            this->coreRows = this->N;
        } else if (coreIdx < tilingData->tailCoreRows) {
            this->coreRows = tilingData->rowsPerCore + 1;
            this->startRow = coreIdx * (tilingData->rowsPerCore + 1);
        } else {
            this->coreRows = tilingData->rowsPerCore;
            this->startRow = tilingData->tailCoreRows * (tilingData->rowsPerCore + 1) +
                             (coreIdx - tilingData->tailCoreRows) * tilingData->rowsPerCore;
        }

        dyGm.SetGlobalBuffer((__gm__ T*)dy, this->N * this->D);
        xGm.SetGlobalBuffer((__gm__ T*)x, this->N * this->D);
        dxGm.SetGlobalBuffer((__gm__ T*)dx, this->N * this->D);
        gammaGm.SetGlobalBuffer((__gm__ T*)gamma, this->D);
        meanGm.SetGlobalBuffer((__gm__ T*)mean, this->N);
        rstdGm.SetGlobalBuffer((__gm__ T*)rstd, this->N);
        dgammaGm.SetGlobalBuffer((__gm__ T*)dgamma, this->D);
        dbetaGm.SetGlobalBuffer((__gm__ T*)dbeta, this->D);

        const uint64_t alignElems64B = 64u / sizeof(T);
        this->wsOffset = ((this->DPadded + alignElems64B - 1u) / alignElems64B) * alignElems64B;

        if (this->multiCore) {
            GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
            this->usedCoreNum = tilingData->usedCoreNum;
            if (this->colSplitMode) {
                workspaceGm.SetGlobalBuffer((__gm__ float*)userWorkspace,
                                            this->usedCoreNum * this->numColTiles * 2 * this->workspaceTileStride);
            } else {
                workspaceGm.SetGlobalBuffer((__gm__ float*)userWorkspace, 2 * this->usedCoreNum * this->wsOffset);
                dgammaGmF.SetGlobalBuffer((__gm__ float*)dgamma, this->D);
                dbetaGmF.SetGlobalBuffer((__gm__ float*)dbeta, this->D);
            }
        }

        if (this->colSplitMode) {
            const uint32_t tileElems = static_cast<uint32_t>(this->tileColsAligned);
            pipe.InitBuffer(inQueueDy, BUFFER_NUM, tileElems * sizeof(T));
            pipe.InitBuffer(inQueueX, BUFFER_NUM, tileElems * sizeof(T));
            pipe.InitBuffer(outQueueDxLarge, BUFFER_NUM, tileElems * sizeof(T));

            pipe.InitBuffer(gammaBuf, tileElems * sizeof(T));
            pipe.InitBuffer(dgammaBuf, tileElems * sizeof(T));
            pipe.InitBuffer(dbetaBuf, tileElems * sizeof(T));
            pipe.InitBuffer(xhatBuf, tileElems * sizeof(T));
            pipe.InitBuffer(dxhatBuf, tileElems * sizeof(T));
            pipe.InitBuffer(tmpBuf, tileElems * sizeof(T));
            pipe.InitBuffer(rowDsBuf, static_cast<uint32_t>(this->maxCoreRows * sizeof(float)));
            pipe.InitBuffer(rowDbBuf, static_cast<uint32_t>(this->maxCoreRows * sizeof(float)));

            const uint64_t elemPerRepeat = 256u / sizeof(T);
            const uint64_t elemPerBlock = 32u / sizeof(T);
            const uint64_t firstRepeats = (this->tileColsAligned + elemPerRepeat - 1u) / elemPerRepeat;
            const uint64_t alignedRepeats = ((firstRepeats + elemPerBlock - 1u) / elemPerBlock) * elemPerBlock;
            const uint64_t reduceElems = alignedRepeats * elemPerBlock;
            pipe.InitBuffer(reduceBuf, static_cast<uint32_t>(reduceElems * sizeof(T)));
            return;
        }

        pipe.InitBuffer(inQueueDy, BUFFER_NUM, this->DPadded * sizeof(T));
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->DPadded * sizeof(T));
        pipe.InitBuffer(outQueueDx, BUFFER_NUM, this->DPadded * sizeof(T));

        pipe.InitBuffer(gammaBuf, this->DPadded * sizeof(T));
        pipe.InitBuffer(dgammaBuf, this->wsOffset * sizeof(T));
        pipe.InitBuffer(dbetaBuf, this->wsOffset * sizeof(T));
        pipe.InitBuffer(xhatBuf, this->DPadded * sizeof(T));
        pipe.InitBuffer(dxhatBuf, this->DPadded * sizeof(T));
        pipe.InitBuffer(tmpBuf, this->wsOffset * sizeof(T));

        const uint64_t elemPerRepeat = 256u / sizeof(T);
        const uint64_t elemPerBlock = 32u / sizeof(T);
        const uint64_t firstRepeats = (this->DPadded + elemPerRepeat - 1u) / elemPerRepeat;
        const uint64_t alignedRepeats = ((firstRepeats + elemPerBlock - 1u) / elemPerBlock) * elemPerBlock;
        const uint64_t reduceElems = alignedRepeats * elemPerBlock;
        pipe.InitBuffer(reduceBuf, static_cast<uint32_t>(reduceElems * sizeof(T)));

        AscendC::LocalTensor<T> gammaLocal = gammaBuf.Get<T>();
        {
            uint32_t dataBytes = static_cast<uint32_t>(this->D * sizeof(T));
            AscendC::DataCopyExtParams copyParams{1, dataBytes, 0, 0, 0};
            AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
            AscendC::DataCopyPad(gammaLocal, gammaGm[0], copyParams, padParams);
        }
        AscendC::PipeBarrier<PIPE_MTE2>();
        WaitMte2ToVectorSync();
        if (this->DPadded > this->D) {
            AscendC::Duplicate(gammaLocal[this->D], static_cast<T>(0), static_cast<int32_t>(this->DPadded - this->D));
        }

        AscendC::LocalTensor<T> dgammaLocal = dgammaBuf.Get<T>();
        AscendC::LocalTensor<T> dbetaLocal = dbetaBuf.Get<T>();
        AscendC::Duplicate(dgammaLocal, static_cast<T>(0), this->wsOffset);
        AscendC::Duplicate(dbetaLocal, static_cast<T>(0), this->wsOffset);
    }

    __aicore__ inline void Process()
    {
        if (this->colSplitMode) {
            ProcessLargeD();
            return;
        }

        for (uint64_t i = 0; i < this->coreRows; ++i) {
            CopyIn(this->startRow + i);
            Compute(this->startRow + i);
            CopyOut(this->startRow + i);
        }

        AscendC::LocalTensor<T> dgammaLocal = dgammaBuf.Get<T>();
        AscendC::LocalTensor<T> dbetaLocal = dbetaBuf.Get<T>();

        if (this->multiCore) {
            AscendC::LocalTensor<float> dgammaFloat = dgammaBuf.Get<float>();
            AscendC::LocalTensor<float> dbetaFloat = dbetaBuf.Get<float>();
            AscendC::LocalTensor<float> reduceFloat = tmpBuf.Get<float>();
            uint64_t coreIdx = static_cast<uint64_t>(AscendC::GetBlockIdx());
            AscendC::PipeBarrier<PIPE_V>();
            WaitVectorToMte3Sync();
            uint64_t workspaceBase = coreIdx * 2 * this->wsOffset;
            AscendC::DataCopy(workspaceGm[workspaceBase], dgammaFloat, this->wsOffset);
            AscendC::DataCopy(workspaceGm[workspaceBase + this->wsOffset], dbetaFloat, this->wsOffset);
            AscendC::PipeBarrier<PIPE_MTE3>();
            AscendC::SyncAll();

            if (coreIdx == 0) {
                AscendC::Duplicate(dgammaFloat, 0.0f, this->wsOffset);
                AscendC::Duplicate(dbetaFloat, 0.0f, this->wsOffset);
                for (uint64_t c = 0; c < this->usedCoreNum; ++c) {
                    uint64_t sliceBase = c * 2 * this->wsOffset;
                    AscendC::DataCopy(reduceFloat, workspaceGm[sliceBase], this->wsOffset);
                    AscendC::PipeBarrier<PIPE_MTE2>();
                    WaitMte2ToVectorSync();
                    AscendC::Add(dgammaFloat, dgammaFloat, reduceFloat, this->wsOffset);
                    AscendC::DataCopy(reduceFloat, workspaceGm[sliceBase + this->wsOffset], this->wsOffset);
                    AscendC::PipeBarrier<PIPE_MTE2>();
                    WaitMte2ToVectorSync();
                    AscendC::Add(dbetaFloat, dbetaFloat, reduceFloat, this->wsOffset);
                }
                AscendC::PipeBarrier<PIPE_V>();
                WaitVectorToMte3Sync();
                {
                    uint32_t dataBytes = static_cast<uint32_t>(this->D * sizeof(float));
                    AscendC::DataCopyExtParams writeParams{1, dataBytes, 0, 0, 0};
                    AscendC::DataCopyPad(dgammaGmF[0], dgammaFloat, writeParams);
                    AscendC::DataCopyPad(dbetaGmF[0], dbetaFloat, writeParams);
                }
            }
        } else {
            AscendC::PipeBarrier<PIPE_V>();
            WaitVectorToMte3Sync();
            {
                uint32_t dataBytes = static_cast<uint32_t>(this->D * sizeof(T));
                AscendC::DataCopyExtParams writeParams{1, dataBytes, 0, 0, 0};
                AscendC::DataCopyPad(dgammaGm[0], dgammaLocal, writeParams);
                AscendC::DataCopyPad(dbetaGm[0], dbetaLocal, writeParams);
            }
        }
    }

private:
    __aicore__ inline uint64_t GetTileColStart(uint64_t tileIdx) { return tileIdx * this->tileCols; }

    __aicore__ inline uint32_t GetTileColCount(uint64_t tileIdx)
    {
        uint64_t colStart = GetTileColStart(tileIdx);
        uint64_t remain = this->D - colStart;
        return static_cast<uint32_t>(remain < this->tileCols ? remain : this->tileCols);
    }

    __aicore__ inline uint32_t GetTileAlignedColCount(uint32_t colCount)
    {
        const uint32_t alignElems = static_cast<uint32_t>(32u / sizeof(T));
        return ((colCount + alignElems - 1u) / alignElems) * alignElems;
    }

    __aicore__ inline void LoadGammaTile(uint64_t colStart, uint32_t colCount, uint32_t tileAlignedCount)
    {
        AscendC::LocalTensor<T> gammaLocal = gammaBuf.Get<T>();
        uint32_t padElems = tileAlignedCount - colCount;
        AscendC::DataCopyExtParams copyParams{1, colCount * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{padElems != 0, 0, static_cast<uint8_t>(padElems), static_cast<T>(0)};
        AscendC::DataCopyPad(gammaLocal, gammaGm[colStart], copyParams, padParams);
        AscendC::PipeBarrier<PIPE_MTE2>();
        WaitMte2ToVectorSync();
    }

    __aicore__ inline void CopyInTile(uint64_t rowIdx, uint64_t colStart, uint32_t colCount, uint32_t tileAlignedCount)
    {
        uint64_t offset = rowIdx * this->D + colStart;
        uint32_t padElems = tileAlignedCount - colCount;
        AscendC::DataCopyExtParams copyParams{1, colCount * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{padElems != 0, 0, static_cast<uint8_t>(padElems), static_cast<T>(0)};

        AscendC::LocalTensor<T> dyLocal = inQueueDy.AllocTensor<T>();
        AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        AscendC::DataCopyPad(dyLocal, dyGm[offset], copyParams, padParams);
        AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        inQueueDy.EnQue(dyLocal);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void ProcessLargeD()
    {
        AscendC::LocalTensor<float> rowDsLocal = rowDsBuf.Get<float>();
        AscendC::LocalTensor<float> rowDbLocal = rowDbBuf.Get<float>();
        AscendC::Duplicate(rowDsLocal, 0.0f, this->maxCoreRows);
        AscendC::Duplicate(rowDbLocal, 0.0f, this->maxCoreRows);
        AscendC::PipeBarrier<PIPE_V>();

        for (uint64_t tileIdx = 0; tileIdx < this->numColTiles; ++tileIdx) {
            uint64_t colStart = GetTileColStart(tileIdx);
            uint32_t colCount = GetTileColCount(tileIdx);
            uint32_t tileAlignedCount = GetTileAlignedColCount(colCount);
            LoadGammaTile(colStart, colCount, tileAlignedCount);
            for (uint64_t localRow = 0; localRow < this->coreRows; ++localRow) {
                uint64_t rowIdx = this->startRow + localRow;
                CopyInTile(rowIdx, colStart, colCount, tileAlignedCount);
                AccumulateRowScalars(localRow, rowIdx, tileAlignedCount);
            }
        }

        for (uint64_t tileIdx = 0; tileIdx < this->numColTiles; ++tileIdx) {
            uint64_t colStart = GetTileColStart(tileIdx);
            uint32_t colCount = GetTileColCount(tileIdx);
            uint32_t tileAlignedCount = GetTileAlignedColCount(colCount);
            AscendC::LocalTensor<T> dgammaLocal = dgammaBuf.Get<T>();
            AscendC::LocalTensor<T> dbetaLocal = dbetaBuf.Get<T>();
            AscendC::Duplicate(dgammaLocal, static_cast<T>(0), tileAlignedCount);
            AscendC::Duplicate(dbetaLocal, static_cast<T>(0), tileAlignedCount);
            AscendC::PipeBarrier<PIPE_V>();
            if (tileIdx > 0) {
                AscendC::PipeBarrier<PIPE_MTE3>();
                WaitMte3ToMte2Sync();
            }
            LoadGammaTile(colStart, colCount, tileAlignedCount);
            for (uint64_t localRow = 0; localRow < this->coreRows; ++localRow) {
                uint64_t rowIdx = this->startRow + localRow;
                CopyInTile(rowIdx, colStart, colCount, tileAlignedCount);
                ComputeDxAndTileGrad(localRow, rowIdx, tileAlignedCount);
                CopyOutTile(rowIdx, colStart, colCount);
            }
            WaitMte3ToVectorSync();

            if (this->multiCore) {
                FlushTileAccumsToWorkspace(tileIdx, tileAlignedCount);
                WaitMte3ToVectorSync();
            } else {
                WriteTileAccumsToOutput(colStart, colCount);
                WaitMte3ToVectorSync();
            }
        }

        if (this->multiCore) {
            AscendC::PipeBarrier<PIPE_MTE3>();
            AscendC::SyncAll();
            ReduceWorkspaceTilesAndWrite();
        }
    }

    __aicore__ inline void AccumulateRowScalars(uint64_t localRow, uint64_t rowIdx, uint32_t tileAlignedCount)
    {
        AscendC::LocalTensor<T> dyLocal = inQueueDy.DeQue<T>();
        AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        AscendC::LocalTensor<T> gammaLocal = gammaBuf.Get<T>();
        AscendC::LocalTensor<T> xhatLocal = xhatBuf.Get<T>();
        AscendC::LocalTensor<T> dxhatLocal = dxhatBuf.Get<T>();
        AscendC::LocalTensor<T> tmpLocal = tmpBuf.Get<T>();
        AscendC::LocalTensor<T> reduceLocal = reduceBuf.Get<T>();
        AscendC::LocalTensor<float> rowDsLocal = rowDsBuf.Get<float>();
        AscendC::LocalTensor<float> rowDbLocal = rowDbBuf.Get<float>();

        float meanVal = ToFloatValue(meanGm.GetValue(rowIdx));
        float rstdVal = ToFloatValue(rstdGm.GetValue(rowIdx));
        AscendC::Adds(xhatLocal, xLocal, static_cast<T>(-meanVal), tileAlignedCount);
        AscendC::Muls(xhatLocal, xhatLocal, static_cast<T>(rstdVal), tileAlignedCount);
        AscendC::Mul(dxhatLocal, dyLocal, gammaLocal, tileAlignedCount);

        AscendC::Mul(tmpLocal, dxhatLocal, xhatLocal, tileAlignedCount);
        AscendC::ReduceSum<T>(reduceLocal, tmpLocal, reduceLocal, static_cast<int32_t>(tileAlignedCount));
        WaitVectorScalarSync();
        float ds = ToFloatValue(reduceLocal.GetValue(0));
        rowDsLocal.SetValue(localRow, rowDsLocal.GetValue(localRow) + ds);

        AscendC::ReduceSum<T>(reduceLocal, dxhatLocal, reduceLocal, static_cast<int32_t>(tileAlignedCount));
        WaitVectorScalarSync();
        float db = ToFloatValue(reduceLocal.GetValue(0));
        rowDbLocal.SetValue(localRow, rowDbLocal.GetValue(localRow) + db);

        inQueueDy.FreeTensor(dyLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void ComputeDxAndTileGrad(uint64_t localRow, uint64_t rowIdx, uint32_t tileAlignedCount)
    {
        AscendC::LocalTensor<T> dyLocal = inQueueDy.DeQue<T>();
        AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        AscendC::LocalTensor<T> gammaLocal = gammaBuf.Get<T>();
        AscendC::LocalTensor<T> dgammaLocal = dgammaBuf.Get<T>();
        AscendC::LocalTensor<T> dbetaLocal = dbetaBuf.Get<T>();
        AscendC::LocalTensor<T> xhatLocal = xhatBuf.Get<T>();
        AscendC::LocalTensor<T> dxhatLocal = dxhatBuf.Get<T>();
        AscendC::LocalTensor<T> tmpLocal = tmpBuf.Get<T>();
        AscendC::LocalTensor<float> rowDsLocal = rowDsBuf.Get<float>();
        AscendC::LocalTensor<float> rowDbLocal = rowDbBuf.Get<float>();

        float meanVal = ToFloatValue(meanGm.GetValue(rowIdx));
        float rstdVal = ToFloatValue(rstdGm.GetValue(rowIdx));
        AscendC::Adds(xhatLocal, xLocal, static_cast<T>(-meanVal), tileAlignedCount);
        AscendC::Muls(xhatLocal, xhatLocal, static_cast<T>(rstdVal), tileAlignedCount);
        AscendC::Mul(dxhatLocal, dyLocal, gammaLocal, tileAlignedCount);

        AscendC::Mul(tmpLocal, dyLocal, xhatLocal, tileAlignedCount);
        AscendC::Add(dgammaLocal, dgammaLocal, tmpLocal, tileAlignedCount);
        AscendC::Add(dbetaLocal, dbetaLocal, dyLocal, tileAlignedCount);

        float dsOverD = rowDsLocal.GetValue(localRow) * this->invD;
        float dbOverD = rowDbLocal.GetValue(localRow) * this->invD;
        AscendC::Muls(xhatLocal, xhatLocal, static_cast<T>(dsOverD), tileAlignedCount);
        AscendC::Adds(xhatLocal, xhatLocal, static_cast<T>(dbOverD), tileAlignedCount);
        AscendC::LocalTensor<T> dxLocal = outQueueDxLarge.AllocTensor<T>();
        AscendC::Sub(dxLocal, dxhatLocal, xhatLocal, tileAlignedCount);
        AscendC::Muls(dxLocal, dxLocal, static_cast<T>(rstdVal), tileAlignedCount);
        outQueueDxLarge.EnQue(dxLocal);

        inQueueDy.FreeTensor(dyLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOutTile(uint64_t rowIdx, uint64_t colStart, uint32_t colCount)
    {
        uint64_t offset = rowIdx * this->D + colStart;
        AscendC::LocalTensor<T> dxLocal = outQueueDxLarge.DeQue<T>();
        AscendC::DataCopyExtParams writeParams{1, colCount * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        AscendC::PipeBarrier<PIPE_V>();
        WaitVectorToMte3Sync();
        AscendC::DataCopyPad(dxGm[offset], dxLocal, writeParams);
        AscendC::PipeBarrier<PIPE_MTE3>();
        WaitMte3ToVectorSync();
        outQueueDxLarge.FreeTensor(dxLocal);
    }

    __aicore__ inline void FlushTileAccumsToWorkspace(uint64_t tileIdx, uint32_t tileAlignedCount)
    {
        uint64_t coreIdx = static_cast<uint64_t>(AscendC::GetBlockIdx());
        uint64_t sliceBase = (coreIdx * this->numColTiles + tileIdx) * 2ULL * this->workspaceTileStride;
        AscendC::LocalTensor<T> dgammaLocal = dgammaBuf.Get<T>();
        AscendC::LocalTensor<T> dbetaLocal = dbetaBuf.Get<T>();
        AscendC::PipeBarrier<PIPE_V>();
        WaitVectorToMte3Sync();
        AscendC::DataCopy(workspaceGm[sliceBase], dgammaLocal, tileAlignedCount);
        AscendC::DataCopy(workspaceGm[sliceBase + this->workspaceTileStride], dbetaLocal, tileAlignedCount);
    }

    __aicore__ inline void WriteTileAccumsToOutput(uint64_t colStart, uint32_t colCount)
    {
        AscendC::LocalTensor<T> dgammaLocal = dgammaBuf.Get<T>();
        AscendC::LocalTensor<T> dbetaLocal = dbetaBuf.Get<T>();
        AscendC::DataCopyExtParams writeParams{1, colCount * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        AscendC::PipeBarrier<PIPE_V>();
        WaitVectorToMte3Sync();
        AscendC::DataCopyPad(dgammaGm[colStart], dgammaLocal, writeParams);
        AscendC::DataCopyPad(dbetaGm[colStart], dbetaLocal, writeParams);
    }

    __aicore__ inline void ReduceWorkspaceTilesAndWrite()
    {
        uint64_t coreIdx = static_cast<uint64_t>(AscendC::GetBlockIdx());
        if (coreIdx != 0) {
            return;
        }

        AscendC::LocalTensor<T> dgammaLocal = dgammaBuf.Get<T>();
        AscendC::LocalTensor<T> dbetaLocal = dbetaBuf.Get<T>();
        AscendC::LocalTensor<T> tmpLocal = tmpBuf.Get<T>();

        for (uint64_t tileIdx = 0; tileIdx < this->numColTiles; ++tileIdx) {
            uint64_t colStart = GetTileColStart(tileIdx);
            uint32_t colCount = GetTileColCount(tileIdx);
            uint32_t tileAlignedCount = GetTileAlignedColCount(colCount);
            AscendC::Duplicate(dgammaLocal, static_cast<T>(0), tileAlignedCount);
            AscendC::Duplicate(dbetaLocal, static_cast<T>(0), tileAlignedCount);

            for (uint64_t c = 0; c < this->usedCoreNum; ++c) {
                uint64_t sliceBase = (c * this->numColTiles + tileIdx) * 2ULL * this->workspaceTileStride;
                AscendC::DataCopy(tmpLocal, workspaceGm[sliceBase], tileAlignedCount);
                AscendC::PipeBarrier<PIPE_MTE2>();
                WaitMte2ToVectorSync();
                AscendC::Add(dgammaLocal, dgammaLocal, tmpLocal, tileAlignedCount);

                AscendC::PipeBarrier<PIPE_V>();
                WaitVectorToMte2Sync();
                AscendC::DataCopy(tmpLocal, workspaceGm[sliceBase + this->workspaceTileStride], tileAlignedCount);
                AscendC::PipeBarrier<PIPE_MTE2>();
                WaitMte2ToVectorSync();
                AscendC::Add(dbetaLocal, dbetaLocal, tmpLocal, tileAlignedCount);

                AscendC::PipeBarrier<PIPE_V>();
                WaitVectorToMte2Sync();
            }

            AscendC::DataCopyExtParams writeParams{1, colCount * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
            AscendC::PipeBarrier<PIPE_V>();
            WaitVectorToMte3Sync();
            AscendC::DataCopyPad(dgammaGm[colStart], dgammaLocal, writeParams);
            AscendC::DataCopyPad(dbetaGm[colStart], dbetaLocal, writeParams);
            WaitMte3ToVectorSync();
        }
    }

    __aicore__ inline void CopyIn(uint64_t rowIdx)
    {
        uint64_t offset = rowIdx * this->D;

        AscendC::LocalTensor<T> dyLocal = inQueueDy.AllocTensor<T>();
        AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        {
            uint32_t dataBytes = static_cast<uint32_t>(this->D * sizeof(T));
            uint32_t padElems = static_cast<uint32_t>(this->DPadded - this->D);
            AscendC::DataCopyExtParams copyParams{1, dataBytes, 0, 0, 0};
            AscendC::DataCopyPadExtParams<T> padParams{true, 0, static_cast<uint8_t>(padElems), static_cast<T>(0)};
            AscendC::DataCopyPad(dyLocal, dyGm[offset], copyParams, padParams);
            AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        }
        inQueueDy.EnQue(dyLocal);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint64_t rowIdx)
    {
        AscendC::LocalTensor<T> dyLocal = inQueueDy.DeQue<T>();
        AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();

        AscendC::LocalTensor<T> gammaLocal = gammaBuf.Get<T>();
        AscendC::LocalTensor<T> dgammaLocal = dgammaBuf.Get<T>();
        AscendC::LocalTensor<T> dbetaLocal = dbetaBuf.Get<T>();
        AscendC::LocalTensor<T> xhatLocal = xhatBuf.Get<T>();
        AscendC::LocalTensor<T> dxhatLocal = dxhatBuf.Get<T>();
        AscendC::LocalTensor<T> tmpLocal = tmpBuf.Get<T>();

        float meanVal = ToFloatValue(meanGm.GetValue(rowIdx));
        float rstdVal = ToFloatValue(rstdGm.GetValue(rowIdx));
        AscendC::Adds(xhatLocal, xLocal, static_cast<T>(-meanVal), this->DPadded);
        AscendC::Muls(xhatLocal, xhatLocal, static_cast<T>(rstdVal), this->DPadded);

        AscendC::Mul(dxhatLocal, dyLocal, gammaLocal, this->DPadded);

        AscendC::Mul(tmpLocal, dyLocal, xhatLocal, this->DPadded);
        AscendC::Add(dgammaLocal, dgammaLocal, tmpLocal, this->DPadded);

        AscendC::Add(dbetaLocal, dbetaLocal, dyLocal, this->DPadded);

        AscendC::Mul(tmpLocal, dxhatLocal, xhatLocal, this->DPadded);
        AscendC::LocalTensor<T> reduceLocal = reduceBuf.Get<T>();
        AscendC::ReduceSum<T>(reduceLocal, tmpLocal, reduceLocal, static_cast<int32_t>(this->DPadded));
        WaitVectorScalarSync();
        float ds_f = ToFloatValue(reduceLocal.GetValue(0));

        AscendC::ReduceSum<T>(reduceLocal, dxhatLocal, reduceLocal, static_cast<int32_t>(this->DPadded));
        WaitVectorScalarSync();
        float db_f = ToFloatValue(reduceLocal.GetValue(0));

        float dsOverD = ds_f * this->invD;
        float dbOverD = db_f * this->invD;
        AscendC::Muls(xhatLocal, xhatLocal, static_cast<T>(dsOverD), this->DPadded);
        AscendC::Adds(xhatLocal, xhatLocal, static_cast<T>(dbOverD), this->DPadded);
        AscendC::Sub(dxhatLocal, dxhatLocal, xhatLocal, this->DPadded);
        AscendC::Muls(dxhatLocal, dxhatLocal, static_cast<T>(rstdVal), this->DPadded);

        AscendC::LocalTensor<T> dxLocal = outQueueDx.AllocTensor<T>();
        AscendC::DataCopy(dxLocal, dxhatLocal, this->DPadded);
        outQueueDx.EnQue(dxLocal);

        inQueueDy.FreeTensor(dyLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint64_t rowIdx)
    {
        uint64_t offset = rowIdx * this->D;
        AscendC::LocalTensor<T> dxLocal = outQueueDx.DeQue<T>();
        {
            uint32_t dataBytes = static_cast<uint32_t>(this->D * sizeof(T));
            AscendC::DataCopyExtParams writeParams{1, dataBytes, 0, 0, 0};
            AscendC::DataCopyPad(dxGm[offset], dxLocal, writeParams);
        }
        outQueueDx.FreeTensor(dxLocal);
    }

private:
    AscendC::TPipe pipe;

    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueDy;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueDx;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 2> outQueueDxLarge;

    AscendC::TBuf<AscendC::TPosition::VECCALC> gammaBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> dgammaBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> dbetaBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xhatBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> dxhatBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rowDsBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rowDbBuf;

    AscendC::GlobalTensor<T> dyGm;
    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> gammaGm;
    AscendC::GlobalTensor<T> meanGm;
    AscendC::GlobalTensor<T> rstdGm;
    AscendC::GlobalTensor<T> dxGm;
    AscendC::GlobalTensor<T> dgammaGm;
    AscendC::GlobalTensor<T> dbetaGm;
    AscendC::GlobalTensor<float> workspaceGm;
    AscendC::GlobalTensor<float> dgammaGmF;
    AscendC::GlobalTensor<float> dbetaGmF;

    uint64_t N;
    uint64_t D;
    uint64_t DPadded;
    uint32_t bufferNum;
    uint64_t wsOffset;
    uint64_t startRow;
    uint64_t coreRows;
    uint64_t usedCoreNum;
    uint32_t colSplitMode;
    uint64_t tileCols;
    uint64_t tileColsAligned;
    uint64_t numColTiles;
    uint64_t workspaceTileStride;
    uint64_t maxCoreRows;
    bool multiCore;
    float invD;
};

// ============================================================================
// KernelLayerNormalizationGradCast<T, CastOutMode> : half/bfloat16 的 float 累加器实现
// ============================================================================
template <typename T, AscendC::RoundMode CastOutMode>
class KernelLayerNormalizationGradCast {
public:
    __aicore__ inline KernelLayerNormalizationGradCast() {}

    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR x, GM_ADDR gamma, GM_ADDR mean, GM_ADDR rstd, GM_ADDR dx,
                                GM_ADDR dgamma, GM_ADDR dbeta, GM_ADDR workspace,
                                const LayerNormalizationGradTilingData* tilingData)
    {
        this->N = tilingData->N;
        this->D = tilingData->D;
        this->DPadded = tilingData->DPadded;
        this->bufferNum = tilingData->bufferNum;
        this->colSplitMode = tilingData->colSplitMode;
        this->tileCols = tilingData->tileCols;
        this->tileColsAligned = tilingData->tileColsAligned;
        this->numColTiles = tilingData->numColTiles;
        this->workspaceTileStride = tilingData->workspaceTileStride;
        this->maxCoreRows = tilingData->maxCoreRows;
        this->invD = 1.0f / static_cast<float>(static_cast<int32_t>(this->D));
        this->wsOffset = ((this->DPadded + 15u) / 16u) * 16u;
        this->usedCoreNum = tilingData->usedCoreNum;

        uint64_t coreIdx = static_cast<uint64_t>(AscendC::GetBlockIdx());
        this->multiCore = (tilingData->usedCoreNum > 1);

        if (!this->multiCore) {
            this->startRow = 0;
            this->coreRows = this->N;
        } else if (coreIdx < tilingData->tailCoreRows) {
            this->coreRows = tilingData->rowsPerCore + 1;
            this->startRow = coreIdx * (tilingData->rowsPerCore + 1);
        } else {
            this->coreRows = tilingData->rowsPerCore;
            this->startRow = tilingData->tailCoreRows * (tilingData->rowsPerCore + 1) +
                             (coreIdx - tilingData->tailCoreRows) * tilingData->rowsPerCore;
        }

        dyGm.SetGlobalBuffer((__gm__ T*)dy, this->N * this->D);
        xGm.SetGlobalBuffer((__gm__ T*)x, this->N * this->D);
        gammaGm.SetGlobalBuffer((__gm__ T*)gamma, this->D);
        meanGm.SetGlobalBuffer((__gm__ T*)mean, this->N);
        rstdGm.SetGlobalBuffer((__gm__ T*)rstd, this->N);
        dxGm.SetGlobalBuffer((__gm__ T*)dx, this->N * this->D);
        dgammaGm.SetGlobalBuffer((__gm__ T*)dgamma, this->D);
        dbetaGm.SetGlobalBuffer((__gm__ T*)dbeta, this->D);

        GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
        if (this->multiCore) {
            if (this->colSplitMode) {
                workspaceGm.SetGlobalBuffer((__gm__ float*)userWorkspace,
                                            this->usedCoreNum * this->numColTiles * 2 * this->workspaceTileStride);
            } else {
                workspaceGm.SetGlobalBuffer((__gm__ float*)userWorkspace, 2 * this->usedCoreNum * this->wsOffset);
            }
        }

        if (this->colSplitMode) {
            const uint32_t tileElems = static_cast<uint32_t>(this->tileColsAligned);
            pipe.InitBuffer(inQueueDy, BUFFER_NUM, tileElems * sizeof(T));
            pipe.InitBuffer(inQueueX, BUFFER_NUM, tileElems * sizeof(T));
            pipe.InitBuffer(outQueueDxLarge, 2, tileElems * sizeof(T));

            pipe.InitBuffer(gammaBuf, tileElems * sizeof(float));
            pipe.InitBuffer(dgammaBuf, tileElems * sizeof(float));
            pipe.InitBuffer(dbetaBuf, tileElems * sizeof(float));
            pipe.InitBuffer(xhatBuf, tileElems * sizeof(float));
            pipe.InitBuffer(dxhatBuf, tileElems * sizeof(float));
            pipe.InitBuffer(tmpBuf, tileElems * sizeof(float));
            pipe.InitBuffer(rowDsBuf, static_cast<uint32_t>(this->maxCoreRows * sizeof(float)));
            pipe.InitBuffer(rowDbBuf, static_cast<uint32_t>(this->maxCoreRows * sizeof(float)));

            const uint64_t elemPerRepeat = 256u / sizeof(float);
            const uint64_t elemPerBlock = 32u / sizeof(float);
            const uint64_t firstRepeats = (this->tileColsAligned + elemPerRepeat - 1u) / elemPerRepeat;
            const uint64_t alignedRepeats = ((firstRepeats + elemPerBlock - 1u) / elemPerBlock) * elemPerBlock;
            const uint64_t reduceElems = alignedRepeats * elemPerBlock;
            pipe.InitBuffer(reduceBuf, static_cast<uint32_t>(reduceElems * sizeof(float)));
            return;
        }

        pipe.InitBuffer(inQueueDy, BUFFER_NUM, this->DPadded * sizeof(T));
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->DPadded * sizeof(T));
        pipe.InitBuffer(outQueueDx, BUFFER_NUM, this->DPadded * sizeof(T));

        pipe.InitBuffer(gammaBuf, this->DPadded * sizeof(float));
        pipe.InitBuffer(dgammaBuf, this->wsOffset * sizeof(float));
        pipe.InitBuffer(dbetaBuf, this->wsOffset * sizeof(float));
        pipe.InitBuffer(xhatBuf, this->DPadded * sizeof(float));
        pipe.InitBuffer(dxhatBuf, this->DPadded * sizeof(float));
        pipe.InitBuffer(tmpBuf, this->wsOffset * sizeof(float));

        const uint64_t elemPerRepeat = 256u / sizeof(float);
        const uint64_t elemPerBlock = 32u / sizeof(float);
        const uint64_t firstRepeats = (this->DPadded + elemPerRepeat - 1u) / elemPerRepeat;
        const uint64_t alignedRepeats = ((firstRepeats + elemPerBlock - 1u) / elemPerBlock) * elemPerBlock;
        const uint64_t reduceElems = alignedRepeats * elemPerBlock;
        pipe.InitBuffer(reduceBuf, static_cast<uint32_t>(reduceElems * sizeof(float)));

        AscendC::LocalTensor<T> gammaLoad = inQueueDy.AllocTensor<T>();
        {
            uint32_t dataBytes = static_cast<uint32_t>(this->D * sizeof(T));
            uint32_t padElems = static_cast<uint32_t>(this->DPadded - this->D);
            AscendC::DataCopyExtParams copyParams{1, dataBytes, 0, 0, 0};
            AscendC::DataCopyPadExtParams<T> padParams{true, 0, static_cast<uint8_t>(padElems), static_cast<T>(0)};
            AscendC::DataCopyPad(gammaLoad, gammaGm[0], copyParams, padParams);
        }
        inQueueDy.EnQue(gammaLoad);
        gammaLoad = inQueueDy.DeQue<T>();
        AscendC::Cast(gammaBuf.Get<float>(), gammaLoad, AscendC::RoundMode::CAST_NONE, this->DPadded);
        AscendC::PipeBarrier<PIPE_V>();
        inQueueDy.FreeTensor(gammaLoad);

        AscendC::Duplicate(dgammaBuf.Get<float>(), 0.0f, this->wsOffset);
        AscendC::Duplicate(dbetaBuf.Get<float>(), 0.0f, this->wsOffset);
    }

    __aicore__ inline void Process()
    {
        if (this->colSplitMode) {
            ProcessLargeD();
            return;
        }

        for (uint64_t i = 0; i < this->coreRows; ++i) {
            CopyIn(this->startRow + i);
            Compute(this->startRow + i);
            CopyOut(this->startRow + i);
        }

        AscendC::LocalTensor<float> dgammaLocal = dgammaBuf.Get<float>();
        AscendC::LocalTensor<float> dbetaLocal = dbetaBuf.Get<float>();

        if (this->multiCore) {
            AscendC::LocalTensor<float> reduceLocal = tmpBuf.Get<float>();
            uint64_t coreIdx = static_cast<uint64_t>(AscendC::GetBlockIdx());
            AscendC::PipeBarrier<PIPE_V>();
            WaitVectorToMte3Sync();
            uint64_t workspaceBase = coreIdx * 2 * this->wsOffset;
            AscendC::DataCopy(workspaceGm[workspaceBase], dgammaLocal, this->wsOffset);
            AscendC::DataCopy(workspaceGm[workspaceBase + this->wsOffset], dbetaLocal, this->wsOffset);
            AscendC::PipeBarrier<PIPE_MTE3>();
            AscendC::SyncAll();

            if (coreIdx == 0) {
                AscendC::Duplicate(dgammaLocal, 0.0f, this->wsOffset);
                AscendC::Duplicate(dbetaLocal, 0.0f, this->wsOffset);
                for (uint64_t c = 0; c < this->usedCoreNum; ++c) {
                    uint64_t sliceBase = c * 2 * this->wsOffset;
                    AscendC::DataCopy(reduceLocal, workspaceGm[sliceBase], this->wsOffset);
                    AscendC::PipeBarrier<PIPE_MTE2>();
                    WaitMte2ToVectorSync();
                    AscendC::Add(dgammaLocal, dgammaLocal, reduceLocal, this->wsOffset);
                    AscendC::DataCopy(reduceLocal, workspaceGm[sliceBase + this->wsOffset], this->wsOffset);
                    AscendC::PipeBarrier<PIPE_MTE2>();
                    WaitMte2ToVectorSync();
                    AscendC::Add(dbetaLocal, dbetaLocal, reduceLocal, this->wsOffset);
                }
                AscendC::PipeBarrier<PIPE_V>();
                WriteFloatAccumToGm(dgammaLocal, dgammaGm);
                WriteFloatAccumToGm(dbetaLocal, dbetaGm);
            }
        } else {
            WriteFloatAccumToGm(dgammaLocal, dgammaGm);
            WriteFloatAccumToGm(dbetaLocal, dbetaGm);
        }
    }

private:
    __aicore__ inline uint64_t GetTileColStart(uint64_t tileIdx) { return tileIdx * this->tileCols; }

    __aicore__ inline uint32_t GetTileColCount(uint64_t tileIdx)
    {
        uint64_t colStart = GetTileColStart(tileIdx);
        uint64_t remain = this->D - colStart;
        return static_cast<uint32_t>(remain < this->tileCols ? remain : this->tileCols);
    }

    __aicore__ inline uint32_t GetTileAlignedColCount(uint32_t colCount)
    {
        const uint32_t alignElems = static_cast<uint32_t>(32u / sizeof(T));
        return ((colCount + alignElems - 1u) / alignElems) * alignElems;
    }

    __aicore__ inline void LoadGammaTile(uint64_t colStart, uint32_t colCount, uint32_t tileAlignedCount)
    {
        AscendC::LocalTensor<T> gammaLoad = inQueueDy.AllocTensor<T>();
        uint32_t padElems = tileAlignedCount - colCount;
        AscendC::DataCopyExtParams copyParams{1, colCount * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{padElems != 0, 0, static_cast<uint8_t>(padElems), static_cast<T>(0)};
        AscendC::DataCopyPad(gammaLoad, gammaGm[colStart], copyParams, padParams);
        inQueueDy.EnQue(gammaLoad);
        gammaLoad = inQueueDy.DeQue<T>();
        AscendC::PipeBarrier<PIPE_MTE2>();
        WaitMte2ToVectorSync();
        AscendC::Cast(gammaBuf.Get<float>(), gammaLoad, AscendC::RoundMode::CAST_NONE, tileAlignedCount);
        AscendC::PipeBarrier<PIPE_V>();
        inQueueDy.FreeTensor(gammaLoad);
    }

    __aicore__ inline void CopyInTile(uint64_t rowIdx, uint64_t colStart, uint32_t colCount, uint32_t tileAlignedCount)
    {
        uint64_t offset = rowIdx * this->D + colStart;
        uint32_t padElems = tileAlignedCount - colCount;
        AscendC::DataCopyExtParams copyParams{1, colCount * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{padElems != 0, 0, static_cast<uint8_t>(padElems), static_cast<T>(0)};

        AscendC::LocalTensor<T> dyLocal = inQueueDy.AllocTensor<T>();
        AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        AscendC::DataCopyPad(dyLocal, dyGm[offset], copyParams, padParams);
        AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        inQueueDy.EnQue(dyLocal);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void ProcessLargeD()
    {
        AscendC::LocalTensor<float> rowDsLocal = rowDsBuf.Get<float>();
        AscendC::LocalTensor<float> rowDbLocal = rowDbBuf.Get<float>();
        AscendC::Duplicate(rowDsLocal, 0.0f, this->maxCoreRows);
        AscendC::Duplicate(rowDbLocal, 0.0f, this->maxCoreRows);
        AscendC::PipeBarrier<PIPE_V>();

        for (uint64_t tileIdx = 0; tileIdx < this->numColTiles; ++tileIdx) {
            uint64_t colStart = GetTileColStart(tileIdx);
            uint32_t colCount = GetTileColCount(tileIdx);
            uint32_t tileAlignedCount = GetTileAlignedColCount(colCount);
            LoadGammaTile(colStart, colCount, tileAlignedCount);
            for (uint64_t localRow = 0; localRow < this->coreRows; ++localRow) {
                uint64_t rowIdx = this->startRow + localRow;
                CopyInTile(rowIdx, colStart, colCount, tileAlignedCount);
                AccumulateRowScalars(localRow, rowIdx, tileAlignedCount);
            }
        }

        for (uint64_t tileIdx = 0; tileIdx < this->numColTiles; ++tileIdx) {
            uint64_t colStart = GetTileColStart(tileIdx);
            uint32_t colCount = GetTileColCount(tileIdx);
            uint32_t tileAlignedCount = GetTileAlignedColCount(colCount);
            AscendC::LocalTensor<float> dgammaLocal = dgammaBuf.Get<float>();
            AscendC::LocalTensor<float> dbetaLocal = dbetaBuf.Get<float>();
            AscendC::Duplicate(dgammaLocal, 0.0f, tileAlignedCount);
            AscendC::Duplicate(dbetaLocal, 0.0f, tileAlignedCount);
            AscendC::PipeBarrier<PIPE_V>();
            if (tileIdx > 0) {
                AscendC::PipeBarrier<PIPE_MTE3>();
                WaitMte3ToMte2Sync();
            }
            LoadGammaTile(colStart, colCount, tileAlignedCount);
            for (uint64_t localRow = 0; localRow < this->coreRows; ++localRow) {
                uint64_t rowIdx = this->startRow + localRow;
                CopyInTile(rowIdx, colStart, colCount, tileAlignedCount);
                ComputeDxAndTileGrad(localRow, rowIdx, tileAlignedCount);
                CopyOutTile(rowIdx, colStart, colCount);
            }
            WaitMte3ToVectorSync();

            if (this->multiCore) {
                FlushTileAccumsToWorkspace(tileIdx, tileAlignedCount);
                WaitMte3ToVectorSync();
            } else {
                WriteTileAccumsToOutput(colStart, colCount);
                WaitMte3ToVectorSync();
            }
        }

        if (this->multiCore) {
            AscendC::PipeBarrier<PIPE_MTE3>();
            AscendC::SyncAll();
            ReduceWorkspaceTilesAndWrite();
        }
    }

    __aicore__ inline void AccumulateRowScalars(uint64_t localRow, uint64_t rowIdx, uint32_t tileAlignedCount)
    {
        AscendC::LocalTensor<T> dyLocal = inQueueDy.DeQue<T>();
        AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        AscendC::LocalTensor<float> gammaLocal = gammaBuf.Get<float>();
        AscendC::LocalTensor<float> xhatLocal = xhatBuf.Get<float>();
        AscendC::LocalTensor<float> dxhatLocal = dxhatBuf.Get<float>();
        AscendC::LocalTensor<float> tmpLocal = tmpBuf.Get<float>();
        AscendC::LocalTensor<float> reduceLocal = reduceBuf.Get<float>();
        AscendC::LocalTensor<float> rowDsLocal = rowDsBuf.Get<float>();
        AscendC::LocalTensor<float> rowDbLocal = rowDbBuf.Get<float>();

        AscendC::Cast(xhatLocal, xLocal, AscendC::RoundMode::CAST_NONE, tileAlignedCount);
        AscendC::Cast(dxhatLocal, dyLocal, AscendC::RoundMode::CAST_NONE, tileAlignedCount);
        AscendC::PipeBarrier<PIPE_V>();

        float meanVal = ToFloatValue(meanGm.GetValue(rowIdx));
        float rstdVal = ToFloatValue(rstdGm.GetValue(rowIdx));
        AscendC::Adds(xhatLocal, xhatLocal, -meanVal, tileAlignedCount);
        AscendC::Muls(xhatLocal, xhatLocal, rstdVal, tileAlignedCount);
        AscendC::Mul(dxhatLocal, dxhatLocal, gammaLocal, tileAlignedCount);

        AscendC::Mul(tmpLocal, dxhatLocal, xhatLocal, tileAlignedCount);
        AscendC::ReduceSum<float>(reduceLocal, tmpLocal, reduceLocal, static_cast<int32_t>(tileAlignedCount));
        WaitVectorScalarSync();
        rowDsLocal.SetValue(localRow, rowDsLocal.GetValue(localRow) + reduceLocal.GetValue(0));

        AscendC::ReduceSum<float>(reduceLocal, dxhatLocal, reduceLocal, static_cast<int32_t>(tileAlignedCount));
        WaitVectorScalarSync();
        rowDbLocal.SetValue(localRow, rowDbLocal.GetValue(localRow) + reduceLocal.GetValue(0));

        inQueueDy.FreeTensor(dyLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void ComputeDxAndTileGrad(uint64_t localRow, uint64_t rowIdx, uint32_t tileAlignedCount)
    {
        AscendC::LocalTensor<T> dyLocal = inQueueDy.DeQue<T>();
        AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        AscendC::LocalTensor<float> gammaLocal = gammaBuf.Get<float>();
        AscendC::LocalTensor<float> dgammaLocal = dgammaBuf.Get<float>();
        AscendC::LocalTensor<float> dbetaLocal = dbetaBuf.Get<float>();
        AscendC::LocalTensor<float> xhatLocal = xhatBuf.Get<float>();
        AscendC::LocalTensor<float> dxhatLocal = dxhatBuf.Get<float>();
        AscendC::LocalTensor<float> tmpLocal = tmpBuf.Get<float>();
        AscendC::LocalTensor<float> rowDsLocal = rowDsBuf.Get<float>();
        AscendC::LocalTensor<float> rowDbLocal = rowDbBuf.Get<float>();

        AscendC::Cast(xhatLocal, xLocal, AscendC::RoundMode::CAST_NONE, tileAlignedCount);
        AscendC::Cast(dxhatLocal, dyLocal, AscendC::RoundMode::CAST_NONE, tileAlignedCount);
        AscendC::PipeBarrier<PIPE_V>();

        float meanVal = ToFloatValue(meanGm.GetValue(rowIdx));
        float rstdVal = ToFloatValue(rstdGm.GetValue(rowIdx));
        AscendC::Adds(xhatLocal, xhatLocal, -meanVal, tileAlignedCount);
        AscendC::Muls(xhatLocal, xhatLocal, rstdVal, tileAlignedCount);

        AscendC::Add(dbetaLocal, dbetaLocal, dxhatLocal, tileAlignedCount);
        AscendC::Mul(tmpLocal, dxhatLocal, xhatLocal, tileAlignedCount);
        AscendC::Add(dgammaLocal, dgammaLocal, tmpLocal, tileAlignedCount);
        AscendC::Mul(dxhatLocal, dxhatLocal, gammaLocal, tileAlignedCount);

        float dsOverD = rowDsLocal.GetValue(localRow) * this->invD;
        float dbOverD = rowDbLocal.GetValue(localRow) * this->invD;
        AscendC::Muls(xhatLocal, xhatLocal, dsOverD, tileAlignedCount);
        AscendC::Adds(xhatLocal, xhatLocal, dbOverD, tileAlignedCount);
        AscendC::Sub(dxhatLocal, dxhatLocal, xhatLocal, tileAlignedCount);
        AscendC::Muls(dxhatLocal, dxhatLocal, rstdVal, tileAlignedCount);

        AscendC::LocalTensor<T> dxLocal = outQueueDxLarge.AllocTensor<T>();
        AscendC::Cast(dxLocal, dxhatLocal, CastOutMode, tileAlignedCount);
        AscendC::PipeBarrier<PIPE_V>();
        outQueueDxLarge.EnQue(dxLocal);

        inQueueDy.FreeTensor(dyLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOutTile(uint64_t rowIdx, uint64_t colStart, uint32_t colCount)
    {
        uint64_t offset = rowIdx * this->D + colStart;
        AscendC::LocalTensor<T> dxLocal = outQueueDxLarge.DeQue<T>();
        AscendC::DataCopyExtParams writeParams{1, colCount * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        AscendC::PipeBarrier<PIPE_V>();
        WaitVectorToMte3Sync();
        AscendC::DataCopyPad(dxGm[offset], dxLocal, writeParams);
        AscendC::PipeBarrier<PIPE_MTE3>();
        WaitMte3ToVectorSync();
        outQueueDxLarge.FreeTensor(dxLocal);
    }

    __aicore__ inline void FlushTileAccumsToWorkspace(uint64_t tileIdx, uint32_t tileAlignedCount)
    {
        uint64_t coreIdx = static_cast<uint64_t>(AscendC::GetBlockIdx());
        uint64_t sliceBase = (coreIdx * this->numColTiles + tileIdx) * 2ULL * this->workspaceTileStride;
        AscendC::LocalTensor<float> dgammaLocal = dgammaBuf.Get<float>();
        AscendC::LocalTensor<float> dbetaLocal = dbetaBuf.Get<float>();
        AscendC::PipeBarrier<PIPE_V>();
        WaitVectorToMte3Sync();
        AscendC::DataCopy(workspaceGm[sliceBase], dgammaLocal, tileAlignedCount);
        AscendC::DataCopy(workspaceGm[sliceBase + this->workspaceTileStride], dbetaLocal, tileAlignedCount);
    }

    __aicore__ inline void WriteFloatTileToGm(uint64_t colStart, uint32_t colCount,
                                              const AscendC::LocalTensor<float>& src, AscendC::GlobalTensor<T>& dst)
    {
        uint32_t tileAlignedCount = GetTileAlignedColCount(colCount);
        AscendC::LocalTensor<T> outLocal = outQueueDxLarge.AllocTensor<T>();
        AscendC::Cast(outLocal, src, CastOutMode, tileAlignedCount);
        AscendC::PipeBarrier<PIPE_V>();
        WaitVectorToMte3Sync();
        AscendC::DataCopyExtParams writeParams{1, colCount * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPad(dst[colStart], outLocal, writeParams);
        AscendC::PipeBarrier<PIPE_MTE3>();
        WaitMte3ToVectorSync();
        outQueueDxLarge.FreeTensor(outLocal);
    }

    __aicore__ inline void WriteTileAccumsToOutput(uint64_t colStart, uint32_t colCount)
    {
        WriteFloatTileToGm(colStart, colCount, dgammaBuf.Get<float>(), dgammaGm);
        WriteFloatTileToGm(colStart, colCount, dbetaBuf.Get<float>(), dbetaGm);
    }

    __aicore__ inline void ReduceWorkspaceTilesAndWrite()
    {
        uint64_t coreIdx = static_cast<uint64_t>(AscendC::GetBlockIdx());
        if (coreIdx != 0) {
            return;
        }

        AscendC::LocalTensor<float> dgammaLocal = dgammaBuf.Get<float>();
        AscendC::LocalTensor<float> dbetaLocal = dbetaBuf.Get<float>();
        AscendC::LocalTensor<float> tmpLocal = tmpBuf.Get<float>();

        for (uint64_t tileIdx = 0; tileIdx < this->numColTiles; ++tileIdx) {
            uint64_t colStart = GetTileColStart(tileIdx);
            uint32_t colCount = GetTileColCount(tileIdx);
            uint32_t tileAlignedCount = GetTileAlignedColCount(colCount);
            AscendC::Duplicate(dgammaLocal, 0.0f, tileAlignedCount);
            AscendC::Duplicate(dbetaLocal, 0.0f, tileAlignedCount);

            for (uint64_t c = 0; c < this->usedCoreNum; ++c) {
                uint64_t sliceBase = (c * this->numColTiles + tileIdx) * 2ULL * this->workspaceTileStride;
                AscendC::DataCopy(tmpLocal, workspaceGm[sliceBase], tileAlignedCount);
                AscendC::PipeBarrier<PIPE_MTE2>();
                WaitMte2ToVectorSync();
                AscendC::Add(dgammaLocal, dgammaLocal, tmpLocal, tileAlignedCount);

                AscendC::PipeBarrier<PIPE_V>();
                WaitVectorToMte2Sync();
                AscendC::DataCopy(tmpLocal, workspaceGm[sliceBase + this->workspaceTileStride], tileAlignedCount);
                AscendC::PipeBarrier<PIPE_MTE2>();
                WaitMte2ToVectorSync();
                AscendC::Add(dbetaLocal, dbetaLocal, tmpLocal, tileAlignedCount);

                AscendC::PipeBarrier<PIPE_V>();
                WaitVectorToMte2Sync();
            }

            WriteTileAccumsToOutput(colStart, colCount);
        }
    }

    __aicore__ inline void CopyIn(uint64_t rowIdx)
    {
        uint64_t offset = rowIdx * this->D;

        AscendC::LocalTensor<T> dyLocal = inQueueDy.AllocTensor<T>();
        AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        {
            uint32_t dataBytes = static_cast<uint32_t>(this->D * sizeof(T));
            uint32_t padElems = static_cast<uint32_t>(this->DPadded - this->D);
            AscendC::DataCopyExtParams copyParams{1, dataBytes, 0, 0, 0};
            AscendC::DataCopyPadExtParams<T> padParams{true, 0, static_cast<uint8_t>(padElems), static_cast<T>(0)};
            AscendC::DataCopyPad(dyLocal, dyGm[offset], copyParams, padParams);
            AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        }
        inQueueDy.EnQue(dyLocal);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint64_t rowIdx)
    {
        AscendC::LocalTensor<T> dyLocal = inQueueDy.DeQue<T>();
        AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();

        AscendC::LocalTensor<float> gammaLocal = gammaBuf.Get<float>();
        AscendC::LocalTensor<float> dgammaLocal = dgammaBuf.Get<float>();
        AscendC::LocalTensor<float> dbetaLocal = dbetaBuf.Get<float>();
        AscendC::LocalTensor<float> xhatLocal = xhatBuf.Get<float>();
        AscendC::LocalTensor<float> dxhatLocal = dxhatBuf.Get<float>();
        AscendC::LocalTensor<float> tmpLocal = tmpBuf.Get<float>();

        AscendC::Cast(xhatLocal, xLocal, AscendC::RoundMode::CAST_NONE, this->DPadded);
        AscendC::Cast(dxhatLocal, dyLocal, AscendC::RoundMode::CAST_NONE, this->DPadded);
        AscendC::PipeBarrier<PIPE_V>();

        float meanVal = ToFloatValue(meanGm.GetValue(rowIdx));
        float rstdVal = ToFloatValue(rstdGm.GetValue(rowIdx));
        AscendC::Adds(xhatLocal, xhatLocal, -meanVal, this->DPadded);
        AscendC::Muls(xhatLocal, xhatLocal, rstdVal, this->DPadded);

        AscendC::Add(dbetaLocal, dbetaLocal, dxhatLocal, this->DPadded);
        AscendC::Mul(tmpLocal, dxhatLocal, xhatLocal, this->DPadded);
        AscendC::Add(dgammaLocal, dgammaLocal, tmpLocal, this->DPadded);
        AscendC::Mul(dxhatLocal, dxhatLocal, gammaLocal, this->DPadded);

        AscendC::LocalTensor<float> reduceLocal = reduceBuf.Get<float>();
        AscendC::Mul(tmpLocal, dxhatLocal, xhatLocal, this->DPadded);
        AscendC::ReduceSum<float>(reduceLocal, tmpLocal, reduceLocal, static_cast<int32_t>(this->DPadded));
        WaitVectorScalarSync();
        float ds_f = reduceLocal.GetValue(0);

        AscendC::ReduceSum<float>(reduceLocal, dxhatLocal, reduceLocal, static_cast<int32_t>(this->DPadded));
        WaitVectorScalarSync();
        float db_f = reduceLocal.GetValue(0);

        AscendC::Muls(xhatLocal, xhatLocal, ds_f * this->invD, this->DPadded);
        AscendC::Adds(xhatLocal, xhatLocal, db_f * this->invD, this->DPadded);
        AscendC::Sub(dxhatLocal, dxhatLocal, xhatLocal, this->DPadded);
        AscendC::Muls(dxhatLocal, dxhatLocal, rstdVal, this->DPadded);

        AscendC::LocalTensor<T> dxLocal = outQueueDx.AllocTensor<T>();
        AscendC::Cast(dxLocal, dxhatLocal, CastOutMode, this->DPadded);
        AscendC::PipeBarrier<PIPE_V>();
        outQueueDx.EnQue(dxLocal);

        inQueueDy.FreeTensor(dyLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint64_t rowIdx)
    {
        uint64_t offset = rowIdx * this->D;
        AscendC::LocalTensor<T> dxLocal = outQueueDx.DeQue<T>();
        {
            uint32_t dataBytes = static_cast<uint32_t>(this->D * sizeof(T));
            AscendC::DataCopyExtParams writeParams{1, dataBytes, 0, 0, 0};
            AscendC::DataCopyPad(dxGm[offset], dxLocal, writeParams);
        }
        outQueueDx.FreeTensor(dxLocal);
    }

    __aicore__ inline void WriteFloatAccumToGm(const AscendC::LocalTensor<float>& src, AscendC::GlobalTensor<T>& dst)
    {
        AscendC::LocalTensor<T> outLocal = outQueueDx.AllocTensor<T>();
        AscendC::Cast(outLocal, src, CastOutMode, this->DPadded);
        AscendC::PipeBarrier<PIPE_V>();
        WaitVectorToMte3Sync();
        outQueueDx.EnQue(outLocal);
        outLocal = outQueueDx.DeQue<T>();
        {
            uint32_t dataBytes = static_cast<uint32_t>(this->D * sizeof(T));
            AscendC::DataCopyExtParams writeParams{1, dataBytes, 0, 0, 0};
            AscendC::DataCopyPad(dst[0], outLocal, writeParams);
        }
        outQueueDx.FreeTensor(outLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueDy;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueDx;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueDxLarge;

    AscendC::TBuf<AscendC::TPosition::VECCALC> gammaBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> dgammaBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> dbetaBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xhatBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> dxhatBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rowDsBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rowDbBuf;

    AscendC::GlobalTensor<T> dyGm;
    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> gammaGm;
    AscendC::GlobalTensor<T> meanGm;
    AscendC::GlobalTensor<T> rstdGm;
    AscendC::GlobalTensor<T> dxGm;
    AscendC::GlobalTensor<T> dgammaGm;
    AscendC::GlobalTensor<T> dbetaGm;
    AscendC::GlobalTensor<float> workspaceGm;

    uint64_t N;
    uint64_t D;
    uint64_t DPadded;
    uint32_t bufferNum;
    uint64_t wsOffset;
    uint64_t startRow;
    uint64_t coreRows;
    uint64_t usedCoreNum;
    uint32_t colSplitMode;
    uint64_t tileCols;
    uint64_t tileColsAligned;
    uint64_t numColTiles;
    uint64_t workspaceTileStride;
    uint64_t maxCoreRows;
    bool multiCore;
    float invD;
};

// Template specializations for half and bfloat16 using float accumulators
template <>
class KernelLayerNormalizationGrad<half>
    : public KernelLayerNormalizationGradCast<half, AscendC::RoundMode::CAST_NONE> {};

template <>
class KernelLayerNormalizationGrad<bfloat16_t>
    : public KernelLayerNormalizationGradCast<bfloat16_t, AscendC::RoundMode::CAST_RINT> {};

} // namespace NsLayerNormalizationGrad
#endif // LAYER_NORMALIZATION_GRAD_H_
