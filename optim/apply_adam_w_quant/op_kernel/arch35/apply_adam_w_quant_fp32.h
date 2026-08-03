/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file apply_adam_w_quant_fp32.h
 * \brief
 */
#ifndef APPLY_ADAM_W_QUANT_FP32_H_
#define APPLY_ADAM_W_QUANT_FP32_H_

#include "apply_adam_w_quant_base.h"

namespace ApplyAdamWQuantNS {
using namespace AscendC;

using namespace ApplyAdamWQuantNS;

template <typename T, typename U>
class ApplyAdamWQuant {
public:
    __aicore__ inline ApplyAdamWQuant(){};
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR grad, GM_ADDR m, GM_ADDR v, GM_ADDR qmap_m, GM_ADDR qmap_v,
                                GM_ADDR absmax_m, GM_ADDR absmax_v, GM_ADDR step, GM_ADDR var_ref, GM_ADDR m_ref,
                                GM_ADDR v_ref, GM_ADDR absmax_m_ref, GM_ADDR absmax_v_ref,
                                const ApplyAdamWQuantRegbaseTilingData* tilingData)
    {
        this->blockIdx = AscendC::GetBlockIdx();
        ParseTilingDataFp32(tilingData);

        if (this->blockIdx < notLastCoreNum) {
            gmOffset = this->blockIdx * notLastPreCoreRowWork * blockSize * oneCoreDoBlockNumPerRow;
            absmaxOffset = this->blockIdx * notLastPreCoreRowWork * oneCoreDoBlockNumPerRow;
        } else {
            gmOffset = (notLastCoreNum * notLastPreCoreRowWork +
                        (this->blockIdx - notLastCoreNum) * lastPreCoreRowWork) *
                       blockSize * oneCoreDoBlockNumPerRow;
            absmaxOffset = (notLastCoreNum * notLastPreCoreRowWork +
                            (this->blockIdx - notLastCoreNum) * lastPreCoreRowWork) *
                           oneCoreDoBlockNumPerRow;
        }
        this->singleBlockNum = oneCoreDoBlockNumPerRow;
        this->singleSize = blockSize * oneCoreDoBlockNumPerRow;
        varGm.SetGlobalBuffer((__gm__ T*)var + gmOffset);
        gradGm.SetGlobalBuffer((__gm__ T*)grad + gmOffset);
        stateMGm.SetGlobalBuffer((__gm__ uint8_t*)m + gmOffset);
        stateVGm.SetGlobalBuffer((__gm__ uint8_t*)v + gmOffset);
        qMapMGm.SetGlobalBuffer((__gm__ T*)qmap_m);
        qMapVGm.SetGlobalBuffer((__gm__ T*)qmap_v);
        absMaxMGm.SetGlobalBuffer((__gm__ T*)absmax_m + absmaxOffset);
        absMaxVGm.SetGlobalBuffer((__gm__ T*)absmax_v + absmaxOffset);
        stepGm.SetGlobalBuffer((__gm__ U*)step, 1);
        varRefGm.SetGlobalBuffer((__gm__ T*)var_ref + gmOffset);
        stateMRefGm.SetGlobalBuffer((__gm__ uint8_t*)m_ref + gmOffset);
        stateVRefGm.SetGlobalBuffer((__gm__ uint8_t*)v_ref + gmOffset);
        absMaxMRefGm.SetGlobalBuffer((__gm__ T*)absmax_m_ref + absmaxOffset);
        absMaxVRefGm.SetGlobalBuffer((__gm__ T*)absmax_v_ref + absmaxOffset);
        step_ = static_cast<float>(stepGm.GetValue(0));

        pipe.InitBuffer(varBuf, singleSize * sizeof(T));
        pipe.InitBuffer(gradBuf, singleSize * sizeof(T));
        pipe.InitBuffer(stateMBuf, singleSize * sizeof(T));
        pipe.InitBuffer(stateVBuf, singleSize * sizeof(T));
        pipe.InitBuffer(qMapMBuf, Q_MAP_SIZE * sizeof(T));
        pipe.InitBuffer(qMapVBuf, Q_MAP_SIZE * sizeof(T));
        pipe.InitBuffer(absMaxMBuf, oneCoreDoBlockNumPerRow * sizeof(T));
        pipe.InitBuffer(absMaxVBuf, oneCoreDoBlockNumPerRow * sizeof(T));
        pipe.InitBuffer(calcBuf, CALC_BUF_NUM * singleSize * sizeof(float));

        this->qMapM = qMapMBuf.Get<T>();
        this->qMapV = qMapVBuf.Get<T>();
        this->stateM = stateMBuf.Get<uint8_t>();
        this->stateV = stateVBuf.Get<uint8_t>();
        this->absMaxM = absMaxMBuf.Get<T>();
        this->absMaxV = absMaxVBuf.Get<T>();
        this->var = varBuf.Get<T>();
        this->grad = gradBuf.Get<T>();

        step_ += 1;
        uint32_t count = AscendC::ONE_BLK_SIZE / sizeof(float);
        AscendC::LocalTensor<float> stepTensor = calcBuf.GetWithOffset<float>(count, 0);
        AscendC::LocalTensor<float> resTensor = calcBuf.GetWithOffset<float>(count, AscendC::ONE_BLK_SIZE);
        AscendC::Duplicate<float>(stepTensor, step_, AscendC::ONE_BLK_SIZE / sizeof(float));
        AscendC::PipeBarrier<PIPE_V>();

        float correction1 = 1 - PowS(resTensor, beta1, stepTensor);
        this->correction2 = sqrt(1 - PowS(resTensor, beta2, stepTensor));
        this->stepSize = -lr * this->correction2 / correction1;
    }
    __aicore__ inline void Process()
    {
        DataCopyIn<T>(qMapM, qMapMGm, Q_MAP_SIZE);
        DataCopyIn<T>(qMapV, qMapVGm, Q_MAP_SIZE);

        int64_t perCoreRowWork = lastPreCoreRowWork;
        if (this->blockIdx < notLastCoreNum) {
            perCoreRowWork += 1;
        }
        uint64_t blockOffset = 0;
        uint64_t absMaxBlockOffset = 0;
        for (int64_t n = 0; n < perCoreRowWork; n++) {
            bool isLastRow = static_cast<uint64_t>(this->blockIdx + 1) == useNumCore && n + 1 == perCoreRowWork;
            singleBlockNum = isLastRow ? lastCoreLastBlock : oneCoreDoBlockNumPerRow;
            singleSize = blockSize * singleBlockNum;
            uint32_t validSize = isLastRow ? singleSize - blockSize + lastBlockSize : singleSize;

            DataCopyIn<uint8_t>(stateM, stateMGm[blockOffset], validSize);
            DataCopyIn<uint8_t>(stateV, stateVGm[blockOffset], validSize);

            DataCopyIn<T>(absMaxM, absMaxMGm[absMaxBlockOffset], singleBlockNum);
            DataCopyIn<T>(absMaxV, absMaxVGm[absMaxBlockOffset], singleBlockNum);

            PipeSync<AscendC::HardEvent::MTE2_V>();

            DeQuantFp32(stateM, qMapM, absMaxM, calcBuf, singleBlockNum, validSize);
            DeQuantFp32(stateV, qMapV, absMaxV, calcBuf, singleBlockNum, validSize);

            DataCopyIn<T>(var, varGm[blockOffset], validSize);
            DataCopyIn<T>(grad, gradGm[blockOffset], validSize);

            PipeSync<AscendC::HardEvent::MTE2_V>();

            UpdateStateAndParamFp32(stateM.template ReinterpretCast<T>(), stateV.template ReinterpretCast<T>(), grad,
                                    var, validSize);
            PipeSync<AscendC::HardEvent::V_MTE3>();

            DataCopyOut<T>(varRefGm[blockOffset], var, validSize);

            QuantMFp32(stateM.template ReinterpretCast<T>(), qMapM, stateM, absMaxM, grad, calcBuf, singleBlockNum);
            QuantVFp32(stateV.template ReinterpretCast<T>(), qMapV, stateV, absMaxV, calcBuf, singleBlockNum);
            PipeSync<AscendC::HardEvent::V_MTE3>();

            DataCopyOut<T>(absMaxMRefGm[absMaxBlockOffset], absMaxM, singleBlockNum);
            DataCopyOut<T>(absMaxVRefGm[absMaxBlockOffset], absMaxV, singleBlockNum);
            DataCopyOut<uint8_t>(stateMRefGm[blockOffset], stateM, validSize);
            DataCopyOut<uint8_t>(stateVRefGm[blockOffset], stateV, validSize);
            PipeSync<AscendC::HardEvent::MTE3_MTE2>();

            blockOffset += singleSize;
            absMaxBlockOffset += singleBlockNum;
        }
    }

private:
    __aicore__ inline void ParseTilingDataFp32(const ApplyAdamWQuantRegbaseTilingData* tilingData)
    {
        useNumCore = tilingData->use_num_core;
        lastPreCoreRowWork = tilingData->last_pre_core_row_work;
        notLastCoreNum = tilingData->not_last_core_num;
        notLastPreCoreRowWork = tilingData->not_last_pre_core_row_work;
        lastCoreLastBlock = tilingData->last_core_last_block;
        lastBlockSize = tilingData->last_block_size;
        oneCoreDoBlockNumPerRow = tilingData->one_core_do_block_num_per_row;

        lr = tilingData->lr;
        beta1 = tilingData->beta1;
        beta2 = tilingData->beta2;
        weightDecay = tilingData->weight_decay;
        eps = tilingData->eps;
        gnormScale = tilingData->gnorm_scale;
        blockSize = tilingData->block_size;
    }

    __aicore__ inline void BinarySearchFp32(const AscendC::LocalTensor<T>& normState,
                                            const AscendC::LocalTensor<T>& qMap, AscendC::LocalTensor<uint8_t>& qState,
                                            AscendC::TBuf<>& calcBuf, int32_t singleBlockNum)
    {
        uint32_t offset = 0;
        AscendC::LocalTensor<int32_t> lowerPivot = calcBuf.GetWithOffset<int32_t>(singleSize, offset);
        offset += singleSize * sizeof(int32_t);
        AscendC::LocalTensor<int32_t> upperPivot = calcBuf.GetWithOffset<int32_t>(singleSize, offset);
        offset += singleSize * sizeof(int32_t);
        AscendC::LocalTensor<int32_t> pivot = calcBuf.GetWithOffset<int32_t>(singleSize, offset);
        offset += singleSize * sizeof(int32_t);
        AscendC::LocalTensor<int32_t> gatherOffset = calcBuf.GetWithOffset<int32_t>(singleSize, offset);
        offset += singleSize * sizeof(int32_t);
        AscendC::LocalTensor<T> stateTmp = calcBuf.GetWithOffset<T>(singleSize, offset);
        offset += singleSize * sizeof(T);
        AscendC::LocalTensor<uint8_t> higgerMask = calcBuf.GetWithOffset<uint8_t>(singleSize / PER_UINT8_8BITS, offset);

        AscendC::Duplicate<int32_t>(lowerPivot, 0, singleSize);
        AscendC::Duplicate<int32_t>(upperPivot, Q_MAP_SIZE - 1, singleSize);
        AscendC::Duplicate<int32_t>(pivot, (Q_MAP_SIZE - 1) >> 1, singleSize);

        T midValue = qMap.GetValue((Q_MAP_SIZE - 1) >> 1);

        PipeSync<AscendC::HardEvent::S_V>();

        AscendC::Duplicate<T>(stateTmp, midValue, singleSize);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Compare<T, uint8_t>(higgerMask, normState, stateTmp, AscendC::CMPMODE::GT, singleSize);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Select<float, uint8_t>(
            lowerPivot.template ReinterpretCast<float>(), higgerMask, pivot.template ReinterpretCast<float>(),
            lowerPivot.template ReinterpretCast<float>(), AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, REPEAT_NUM,
            singleSize / REPEAT_NUM, {1, 1, 1, 8, 8, 8});
        AscendC::Select<float, uint8_t>(
            upperPivot.template ReinterpretCast<float>(), higgerMask, upperPivot.template ReinterpretCast<float>(),
            pivot.template ReinterpretCast<float>(), AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, REPEAT_NUM,
            singleSize / REPEAT_NUM, {1, 1, 1, 8, 8, 8});
        AscendC::PipeBarrier<PIPE_V>();

        for (int32_t i = 0; i < REPEAT_7_TIMES; ++i) {
            AscendC::Add<int32_t>(pivot, lowerPivot, upperPivot, singleSize);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ShiftRight<int32_t>(pivot, pivot, 1, singleSize);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Muls<int32_t>(gatherOffset, pivot, sizeof(T), singleSize);
            AscendC::PipeBarrier<PIPE_V>();

#ifdef ASCENDC_CPU_DEBUG
            for (int i = 0; i < singleSize / Q_MAP_SIZE; i++) {
                AscendC::Gather<T>(stateTmp[i * Q_MAP_SIZE], qMap,
                                   gatherOffset.template ReinterpretCast<uint32_t>()[i * Q_MAP_SIZE], 0, Q_MAP_SIZE);
            }
#else
            AscendC::Gather<T>(stateTmp, qMap, gatherOffset.template ReinterpretCast<uint32_t>(), 0, singleSize);
#endif
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Compare<T, uint8_t>(higgerMask, normState, stateTmp, AscendC::CMPMODE::GT, singleSize);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Select<float, uint8_t>(
                lowerPivot.template ReinterpretCast<float>(), higgerMask, pivot.template ReinterpretCast<float>(),
                lowerPivot.template ReinterpretCast<float>(), AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, REPEAT_NUM,
                singleSize / REPEAT_NUM, {1, 1, 1, 8, 8, 8});

            AscendC::Select<float, uint8_t>(
                upperPivot.template ReinterpretCast<float>(), higgerMask, upperPivot.template ReinterpretCast<float>(),
                pivot.template ReinterpretCast<float>(), AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, REPEAT_NUM,
                singleSize / REPEAT_NUM, {1, 1, 1, 8, 8, 8});
            AscendC::PipeBarrier<PIPE_V>();
        }

        AscendC::LocalTensor<T> lowerState = stateTmp;
        AscendC::LocalTensor<T> upperState = pivot.template ReinterpretCast<T>();

        AscendC::Muls<int32_t>(gatherOffset, lowerPivot, sizeof(T), singleSize);
        AscendC::PipeBarrier<PIPE_V>();
#ifdef ASCENDC_CPU_DEBUG
        for (int i = 0; i < singleSize / Q_MAP_SIZE; i++) {
            AscendC::Gather<T>(lowerState[i * Q_MAP_SIZE], qMap,
                               gatherOffset.template ReinterpretCast<uint32_t>()[i * Q_MAP_SIZE], 0, Q_MAP_SIZE);
        }
#else
        AscendC::Gather<T>(lowerState, qMap, gatherOffset.template ReinterpretCast<uint32_t>(), 0, singleSize);
#endif
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub<T>(lowerState, normState, lowerState, singleSize);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Abs<T>(lowerState, lowerState, singleSize);

        AscendC::Muls<int32_t>(gatherOffset, upperPivot, sizeof(T), singleSize);
        AscendC::PipeBarrier<PIPE_V>();

#ifdef ASCENDC_CPU_DEBUG
        for (int i = 0; i < singleSize / Q_MAP_SIZE; i++) {
            AscendC::Gather<T>(upperState[i * Q_MAP_SIZE], qMap,
                               gatherOffset.template ReinterpretCast<uint32_t>()[i * Q_MAP_SIZE], 0, Q_MAP_SIZE);
        }
#else
        AscendC::Gather<T>(upperState, qMap, gatherOffset.template ReinterpretCast<uint32_t>(), 0, singleSize);
#endif
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub<T>(upperState, normState, upperState, singleSize);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Abs<T>(upperState, upperState, singleSize);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Compare<T, uint8_t>(higgerMask, lowerState, upperState, AscendC::CMPMODE::LE, singleSize);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Select<float, uint8_t>(
            pivot.template ReinterpretCast<float>(), higgerMask, lowerPivot.template ReinterpretCast<float>(),
            upperPivot.template ReinterpretCast<float>(), AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, REPEAT_NUM,
            singleSize / REPEAT_NUM, {1, 1, 1, 8, 8, 8});
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast<int16_t, int32_t>(stateTmp.template ReinterpretCast<int16_t>(), pivot,
                                        AscendC::RoundMode::CAST_NONE, singleSize);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast<half, int16_t>(pivot.template ReinterpretCast<half>(),
                                     stateTmp.template ReinterpretCast<int16_t>(), AscendC::RoundMode::CAST_NONE,
                                     singleSize);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast<uint8_t, half>(qState, pivot.template ReinterpretCast<half>(), AscendC::RoundMode::CAST_NONE,
                                     singleSize);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void BinarySearchScaledFp32(const AscendC::LocalTensor<T>& state,
                                                  const AscendC::LocalTensor<T>& qMap,
                                                  const AscendC::LocalTensor<T>& absMax,
                                                  AscendC::LocalTensor<uint8_t>& qState, AscendC::TBuf<>& calcBuf,
                                                  int32_t singleBlockNum)
    {
        uint32_t offset = 0;
        AscendC::LocalTensor<int32_t> lowerPivot = calcBuf.GetWithOffset<int32_t>(singleSize, offset);
        offset += singleSize * sizeof(int32_t);
        AscendC::LocalTensor<int32_t> upperPivot = calcBuf.GetWithOffset<int32_t>(singleSize, offset);
        offset += singleSize * sizeof(int32_t);
        AscendC::LocalTensor<int32_t> pivot = calcBuf.GetWithOffset<int32_t>(singleSize, offset);
        offset += singleSize * sizeof(int32_t);
        AscendC::LocalTensor<int32_t> gatherOffset = calcBuf.GetWithOffset<int32_t>(singleSize, offset);
        offset += singleSize * sizeof(int32_t);
        AscendC::LocalTensor<T> stateTmp = calcBuf.GetWithOffset<T>(singleSize, offset);
        offset += singleSize * sizeof(T);
        AscendC::LocalTensor<uint8_t> higgerMask = calcBuf.GetWithOffset<uint8_t>(singleSize / PER_UINT8_8BITS, offset);
        AscendC::LocalTensor<T> absMaxBrcb = gatherOffset.template ReinterpretCast<T>();

        uint32_t srcShape[2] = {static_cast<uint32_t>(singleBlockNum), 1};
        uint32_t dstShape[2] = {static_cast<uint32_t>(singleBlockNum), blockSize};

        AscendC::Duplicate<int32_t>(lowerPivot, 0, singleSize);
        AscendC::Duplicate<int32_t>(upperPivot, Q_MAP_SIZE - 1, singleSize);
        AscendC::Duplicate<int32_t>(pivot, (Q_MAP_SIZE - 1) >> 1, singleSize);

        T midValue = qMap.GetValue((Q_MAP_SIZE - 1) >> 1);

        PipeSync<AscendC::HardEvent::S_V>();

        AscendC::Duplicate<T>(stateTmp, midValue, singleSize);
        AscendC::Broadcast<T, BROADCAST_DIM2, BROADCAST_AXIS1>(absMaxBrcb, absMax, dstShape, srcShape);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul<T>(stateTmp, stateTmp, absMaxBrcb, singleSize);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Compare<T, uint8_t>(higgerMask, state, stateTmp, AscendC::CMPMODE::GT, singleSize);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Select<float, uint8_t>(
            lowerPivot.template ReinterpretCast<float>(), higgerMask, pivot.template ReinterpretCast<float>(),
            lowerPivot.template ReinterpretCast<float>(), AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, REPEAT_NUM,
            singleSize / REPEAT_NUM, {1, 1, 1, 8, 8, 8});
        AscendC::Select<float, uint8_t>(
            upperPivot.template ReinterpretCast<float>(), higgerMask, upperPivot.template ReinterpretCast<float>(),
            pivot.template ReinterpretCast<float>(), AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, REPEAT_NUM,
            singleSize / REPEAT_NUM, {1, 1, 1, 8, 8, 8});
        AscendC::PipeBarrier<PIPE_V>();

        for (int32_t i = 0; i < REPEAT_7_TIMES; ++i) {
            AscendC::Add<int32_t>(pivot, lowerPivot, upperPivot, singleSize);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ShiftRight<int32_t>(pivot, pivot, 1, singleSize);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Muls<int32_t>(gatherOffset, pivot, sizeof(T), singleSize);
            AscendC::PipeBarrier<PIPE_V>();

#ifdef ASCENDC_CPU_DEBUG
            for (int i = 0; i < singleSize / Q_MAP_SIZE; i++) {
                AscendC::Gather<T>(stateTmp[i * Q_MAP_SIZE], qMap,
                                   gatherOffset.template ReinterpretCast<uint32_t>()[i * Q_MAP_SIZE], 0, Q_MAP_SIZE);
            }
#else
            AscendC::Gather<T>(stateTmp, qMap, gatherOffset.template ReinterpretCast<uint32_t>(), 0, singleSize);
#endif
            AscendC::Broadcast<T, BROADCAST_DIM2, BROADCAST_AXIS1>(absMaxBrcb, absMax, dstShape, srcShape);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul<T>(stateTmp, stateTmp, absMaxBrcb, singleSize);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Compare<T, uint8_t>(higgerMask, state, stateTmp, AscendC::CMPMODE::GT, singleSize);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Select<float, uint8_t>(
                lowerPivot.template ReinterpretCast<float>(), higgerMask, pivot.template ReinterpretCast<float>(),
                lowerPivot.template ReinterpretCast<float>(), AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, REPEAT_NUM,
                singleSize / REPEAT_NUM, {1, 1, 1, 8, 8, 8});

            AscendC::Select<float, uint8_t>(
                upperPivot.template ReinterpretCast<float>(), higgerMask, upperPivot.template ReinterpretCast<float>(),
                pivot.template ReinterpretCast<float>(), AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, REPEAT_NUM,
                singleSize / REPEAT_NUM, {1, 1, 1, 8, 8, 8});
            AscendC::PipeBarrier<PIPE_V>();
        }

        AscendC::LocalTensor<T> lowerState = stateTmp;
        AscendC::LocalTensor<T> upperState = pivot.template ReinterpretCast<T>();

        AscendC::Muls<int32_t>(gatherOffset, lowerPivot, sizeof(T), singleSize);
        AscendC::PipeBarrier<PIPE_V>();
#ifdef ASCENDC_CPU_DEBUG
        for (int i = 0; i < singleSize / Q_MAP_SIZE; i++) {
            AscendC::Gather<T>(lowerState[i * Q_MAP_SIZE], qMap,
                               gatherOffset.template ReinterpretCast<uint32_t>()[i * Q_MAP_SIZE], 0, Q_MAP_SIZE);
        }
#else
        AscendC::Gather<T>(lowerState, qMap, gatherOffset.template ReinterpretCast<uint32_t>(), 0, singleSize);
#endif
        AscendC::Broadcast<T, BROADCAST_DIM2, BROADCAST_AXIS1>(absMaxBrcb, absMax, dstShape, srcShape);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul<T>(lowerState, lowerState, absMaxBrcb, singleSize);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub<T>(lowerState, state, lowerState, singleSize);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Abs<T>(lowerState, lowerState, singleSize);

        AscendC::Muls<int32_t>(gatherOffset, upperPivot, sizeof(T), singleSize);
        AscendC::PipeBarrier<PIPE_V>();

#ifdef ASCENDC_CPU_DEBUG
        for (int i = 0; i < singleSize / Q_MAP_SIZE; i++) {
            AscendC::Gather<T>(upperState[i * Q_MAP_SIZE], qMap,
                               gatherOffset.template ReinterpretCast<uint32_t>()[i * Q_MAP_SIZE], 0, Q_MAP_SIZE);
        }
#else
        AscendC::Gather<T>(upperState, qMap, gatherOffset.template ReinterpretCast<uint32_t>(), 0, singleSize);
#endif
        AscendC::Broadcast<T, BROADCAST_DIM2, BROADCAST_AXIS1>(absMaxBrcb, absMax, dstShape, srcShape);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul<T>(upperState, upperState, absMaxBrcb, singleSize);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub<T>(upperState, state, upperState, singleSize);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Abs<T>(upperState, upperState, singleSize);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Compare<T, uint8_t>(higgerMask, lowerState, upperState, AscendC::CMPMODE::LE, singleSize);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Select<float, uint8_t>(
            pivot.template ReinterpretCast<float>(), higgerMask, lowerPivot.template ReinterpretCast<float>(),
            upperPivot.template ReinterpretCast<float>(), AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, REPEAT_NUM,
            singleSize / REPEAT_NUM, {1, 1, 1, 8, 8, 8});
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast<int16_t, int32_t>(stateTmp.template ReinterpretCast<int16_t>(), pivot,
                                        AscendC::RoundMode::CAST_NONE, singleSize);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast<half, int16_t>(pivot.template ReinterpretCast<half>(),
                                     stateTmp.template ReinterpretCast<int16_t>(), AscendC::RoundMode::CAST_NONE,
                                     singleSize);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast<uint8_t, half>(qState, pivot.template ReinterpretCast<half>(), AscendC::RoundMode::CAST_NONE,
                                     singleSize);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CheckSignFp32(const AscendC::LocalTensor<T>& normState, const AscendC::LocalTensor<T>& qMap,
                                         AscendC::LocalTensor<uint8_t>& qState, AscendC::TBuf<>& calcBuf,
                                         int32_t singleBlockNum)
    {
        // Match golden exactly: if normalized state and selected qmap have opposite
        // signs, move the code by one step toward the normalized state's sign.
        // Use int16 arithmetic (A2-style) to avoid float rounding in index conversion.
        uint32_t offset = 0;
        AscendC::LocalTensor<int32_t> gatherOffset = calcBuf.GetWithOffset<int32_t>(singleSize, offset);
        offset += singleSize * sizeof(int32_t);
        AscendC::LocalTensor<T> dq = calcBuf.GetWithOffset<T>(singleSize, offset);
        offset += singleSize * sizeof(T);
        AscendC::LocalTensor<int16_t> idxI = calcBuf.GetWithOffset<int16_t>(singleSize, offset);
        offset += singleSize * sizeof(int16_t);
        AscendC::LocalTensor<T> zero = calcBuf.GetWithOffset<T>(singleSize, offset);
        offset += singleSize * sizeof(T);
        AscendC::LocalTensor<int16_t> adjustedI = calcBuf.GetWithOffset<int16_t>(singleSize, offset);
        offset += singleSize * sizeof(int16_t);
        const int32_t maskBytes = singleSize / PER_UINT8_8BITS;
        AscendC::LocalTensor<uint8_t> normPosMask = calcBuf.GetWithOffset<uint8_t>(maskBytes, offset);
        offset += maskBytes;
        AscendC::LocalTensor<uint8_t> normNegMask = calcBuf.GetWithOffset<uint8_t>(maskBytes, offset);
        offset += maskBytes;
        AscendC::LocalTensor<uint8_t> qPosMask = calcBuf.GetWithOffset<uint8_t>(maskBytes, offset);
        offset += maskBytes;
        AscendC::LocalTensor<uint8_t> qNegMask = calcBuf.GetWithOffset<uint8_t>(maskBytes, offset);
        AscendC::LocalTensor<half> halfBuf = gatherOffset.template ReinterpretCast<half>();

        const int32_t selRepeat = singleSize / REPEAT_NUM_128;

        // idxI = int16(qState) — exact since values are 0-255
        AscendC::Cast<half, uint8_t>(halfBuf, qState, AscendC::RoundMode::CAST_NONE, singleSize);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast<int16_t, half>(idxI, halfBuf, AscendC::RoundMode::CAST_RINT, singleSize);
        AscendC::PipeBarrier<PIPE_V>();

        // dq = qMap[idxI]
        AscendC::Cast<int32_t, int16_t>(gatherOffset, idxI, AscendC::RoundMode::CAST_NONE, singleSize);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls<int32_t>(gatherOffset, gatherOffset, sizeof(T), singleSize);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Gather<T>(dq, qMap, gatherOffset.template ReinterpretCast<uint32_t>(), 0, singleSize);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Duplicate<T>(zero, static_cast<T>(0.0f), singleSize);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Compare<T, uint8_t>(normPosMask, normState, zero, AscendC::CMPMODE::GT, singleSize);
        AscendC::Compare<T, uint8_t>(normNegMask, normState, zero, AscendC::CMPMODE::LT, singleSize);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Compare<T, uint8_t>(qPosMask, dq, zero, AscendC::CMPMODE::GT, singleSize);
        AscendC::Compare<T, uint8_t>(qNegMask, dq, zero, AscendC::CMPMODE::LT, singleSize);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::And(normPosMask, normPosMask, qNegMask, maskBytes);
        AscendC::And(normNegMask, normNegMask, qPosMask, maskBytes);
        AscendC::PipeBarrier<PIPE_V>();

        // Adjust idxI by ±1 in int16 space (avoids float rounding)
        AscendC::Adds<int16_t>(adjustedI, idxI, static_cast<int16_t>(1), singleSize);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Select<int16_t, uint8_t>(idxI, normPosMask, adjustedI, idxI, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE,
                                          REPEAT_NUM_128, selRepeat, {1, 1, 1, 8, 8, 8});
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds<int16_t>(adjustedI, idxI, static_cast<int16_t>(-1), singleSize);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Select<int16_t, uint8_t>(idxI, normNegMask, adjustedI, idxI, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE,
                                          REPEAT_NUM_128, selRepeat, {1, 1, 1, 8, 8, 8});
        AscendC::PipeBarrier<PIPE_V>();

        // Clamp to [0, 255]
        AscendC::Maxs<int16_t>(idxI, idxI, static_cast<int16_t>(0), singleSize);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mins<int16_t>(idxI, idxI, static_cast<int16_t>(Q_MAP_SIZE - 1), singleSize);
        AscendC::PipeBarrier<PIPE_V>();

        // int16 → half → uint8 (exact, no rounding)
        AscendC::Cast<half, int16_t>(halfBuf, idxI, AscendC::RoundMode::CAST_NONE, singleSize);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast<uint8_t, half>(qState, halfBuf, AscendC::RoundMode::CAST_NONE, singleSize);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void NormlizeFp32(const AscendC::LocalTensor<T>& state, const AscendC::LocalTensor<T>& absMax,
                                        AscendC::TBuf<>& calcBuf, int32_t singleBlockNum)
    {
        AscendC::LocalTensor<T> absState = calcBuf.GetWithOffset<T>(singleSize, 0);
        AscendC::LocalTensor<T> absMaxTmp = calcBuf.GetWithOffset<T>(singleSize, singleSize * sizeof(T));
        AscendC::LocalTensor<T> absMaxBrcb = calcBuf.GetWithOffset<T>(singleSize, 2 * singleSize * sizeof(T));

        AscendC::Abs<T>(absState, state, singleSize);
        AscendC::PipeBarrier<PIPE_V>();

        uint32_t mask = AscendC::ONE_REPEAT_BYTE_SIZE / sizeof(T);
        uint32_t repeat = singleSize / mask;
        AscendC::BlockReduceMax<T>(absMaxTmp, absState, repeat, mask, 1, 1, STRIDE_8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::WholeReduceMax<T>(absMax, absMaxTmp, blockSize / (AscendC::ONE_BLK_SIZE / sizeof(T)), singleBlockNum,
                                   1, 1, PER_4NUM_ONEMAX, AscendC::ReduceOrder::ORDER_ONLY_VALUE);
        AscendC::PipeBarrier<PIPE_V>();

        uint32_t srcShape[2] = {static_cast<uint32_t>(singleBlockNum), 1};
        uint32_t dstShape[2] = {static_cast<uint32_t>(singleBlockNum), blockSize};
        AscendC::Broadcast<T, BROADCAST_DIM2, BROADCAST_AXIS1>(absMaxBrcb, absMax, dstShape, srcShape);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Div<T>(state, state, absMaxBrcb, singleSize);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CalcAbsMaxFp32(const AscendC::LocalTensor<T>& state, const AscendC::LocalTensor<T>& absMax,
                                          AscendC::TBuf<>& calcBuf, int32_t singleBlockNum)
    {
        AscendC::LocalTensor<T> absState = calcBuf.GetWithOffset<T>(singleSize, 0);
        AscendC::LocalTensor<T> absMaxTmp = calcBuf.GetWithOffset<T>(singleSize, singleSize * sizeof(T));

        AscendC::Abs<T>(absState, state, singleSize);
        AscendC::PipeBarrier<PIPE_V>();

        uint32_t mask = AscendC::ONE_REPEAT_BYTE_SIZE / sizeof(T);
        uint32_t repeat = singleSize / mask;
        AscendC::BlockReduceMax<T>(absMaxTmp, absState, repeat, mask, 1, 1, STRIDE_8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::WholeReduceMax<T>(absMax, absMaxTmp, blockSize / (AscendC::ONE_BLK_SIZE / sizeof(T)), singleBlockNum,
                                   1, 1, PER_4NUM_ONEMAX, AscendC::ReduceOrder::ORDER_ONLY_VALUE);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void QuantMFp32(const AscendC::LocalTensor<T>& state, const AscendC::LocalTensor<T>& qMap,
                                      AscendC::LocalTensor<uint8_t>& qState, const AscendC::LocalTensor<T>& absMax,
                                      AscendC::LocalTensor<T>& normStateTmp, AscendC::TBuf<>& calcBuf,
                                      int32_t singleBlockNum)
    {
        NormlizeFp32(state, absMax, calcBuf, singleBlockNum);

        AscendC::Adds<T>(normStateTmp, state, static_cast<T>(0.0f), singleSize);
        AscendC::PipeBarrier<PIPE_V>();

        BinarySearchFp32(state, qMap, qState, calcBuf, singleBlockNum);

        CheckSignFp32(normStateTmp, qMap, qState, calcBuf, singleBlockNum);
    }

    __aicore__ inline void QuantVFp32(const AscendC::LocalTensor<T>& state, const AscendC::LocalTensor<T>& qMap,
                                      AscendC::LocalTensor<uint8_t>& qState, const AscendC::LocalTensor<T>& absMax,
                                      AscendC::TBuf<>& calcBuf, int32_t singleBlockNum)
    {
        NormlizeFp32(state, absMax, calcBuf, singleBlockNum);
        BinarySearchFp32(state, qMap, qState, calcBuf, singleBlockNum);
    }

    __aicore__ inline void DeQuantFp32(const AscendC::LocalTensor<uint8_t>& state, const AscendC::LocalTensor<T>& qMap,
                                       AscendC::LocalTensor<T>& absMax, AscendC::TBuf<>& calcBuf,
                                       int32_t singleBlockNum, uint32_t validSize)
    {
        uint32_t offset = 0;
        AscendC::LocalTensor<T> dqState = calcBuf.GetWithOffset<T>(singleSize, offset);
        offset += singleSize * sizeof(T);
        AscendC::LocalTensor<int32_t> gatherOffset = calcBuf.GetWithOffset<int32_t>(singleSize, offset);
        offset += singleSize * sizeof(int32_t);
        AscendC::LocalTensor<T> absMaxBrcb = calcBuf.GetWithOffset<T>(singleSize, offset);

        AscendC::Cast<half, uint8_t>(gatherOffset[singleSize >> 1].template ReinterpretCast<half>(), state,
                                     AscendC::RoundMode::CAST_NONE, singleSize);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast<int32_t, half>(gatherOffset, gatherOffset[singleSize >> 1].template ReinterpretCast<half>(),
                                     AscendC::RoundMode::CAST_RINT, singleSize);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls<int32_t>(gatherOffset, gatherOffset, sizeof(T), singleSize);
        AscendC::PipeBarrier<PIPE_V>();
#ifdef ASCENDC_CPU_DEBUG
        for (int i = 0; i < singleSize / Q_MAP_SIZE; i++) {
            AscendC::Gather<T>(dqState[i * Q_MAP_SIZE], qMap,
                               gatherOffset.template ReinterpretCast<uint32_t>()[i * Q_MAP_SIZE], 0, Q_MAP_SIZE);
        }
#else
        AscendC::Gather<T>(dqState, qMap, gatherOffset.template ReinterpretCast<uint32_t>(), 0, singleSize);
#endif

        uint32_t srcShape[2] = {static_cast<uint32_t>(singleBlockNum), 1};
        uint32_t dstShape[2] = {static_cast<uint32_t>(singleBlockNum), blockSize};
        AscendC::Broadcast<T, BROADCAST_DIM2, BROADCAST_AXIS1>(absMaxBrcb, absMax, dstShape, srcShape);
        AscendC::PipeBarrier<PIPE_V>();
        if (validSize < static_cast<uint32_t>(singleSize)) {
            AscendC::Duplicate<T>(state.template ReinterpretCast<T>(), static_cast<T>(0), singleSize);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::Mul<T>(state.template ReinterpretCast<T>(), dqState, absMaxBrcb, validSize);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void UpdateStateAndParamFp32(const AscendC::LocalTensor<T>& dqStateM,
                                                   const AscendC::LocalTensor<T>& dqStateV,
                                                   AscendC::LocalTensor<T>& grad, AscendC::LocalTensor<T>& var,
                                                   uint32_t validSize)
    {
        constexpr uint32_t oneRepeatSize = AscendC::GetVecLen() / sizeof(T);
        uint16_t repeatTimes = AscendC::CeilDivision(validSize, oneRepeatSize);
        float weightDecayFactor = weightDecay > 0.0f ? 1.0f - lr * weightDecay : 1.0f;
        asc_vf_call<AdamWQuantUpdateVF<T>>((__ubuf__ T*)dqStateM.GetPhyAddr(), (__ubuf__ T*)dqStateV.GetPhyAddr(),
                                           (__ubuf__ T*)var.GetPhyAddr(), (__ubuf__ T*)grad.GetPhyAddr(), gnormScale,
                                           beta1, 1.0f - beta1, beta2, 1.0f - beta2, eps * correction2, stepSize,
                                           weightDecayFactor, validSize, oneRepeatSize, repeatTimes);
    }

private:
    TPipe pipe;

    AscendC::TBuf<AscendC::QuePosition::VECCALC> varBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> gradBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> stateMBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> stateVBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> qMapMBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> qMapVBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> absMaxMBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> absMaxVBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> calcBuf;

    AscendC::GlobalTensor<T> varGm;
    AscendC::GlobalTensor<T> varRefGm;
    AscendC::GlobalTensor<T> gradGm;
    AscendC::GlobalTensor<uint8_t> stateMGm;
    AscendC::GlobalTensor<uint8_t> stateMRefGm;
    AscendC::GlobalTensor<uint8_t> stateVGm;
    AscendC::GlobalTensor<uint8_t> stateVRefGm;
    AscendC::GlobalTensor<T> qMapMGm;
    AscendC::GlobalTensor<T> qMapVGm;
    AscendC::GlobalTensor<T> absMaxMGm;
    AscendC::GlobalTensor<T> absMaxMRefGm;
    AscendC::GlobalTensor<T> absMaxVGm;
    AscendC::GlobalTensor<T> absMaxVRefGm;
    AscendC::GlobalTensor<U> stepGm;

    AscendC::LocalTensor<T> qMapM;
    AscendC::LocalTensor<T> qMapV;
    AscendC::LocalTensor<uint8_t> stateM;
    AscendC::LocalTensor<uint8_t> stateV;
    AscendC::LocalTensor<T> absMaxM;
    AscendC::LocalTensor<T> absMaxV;
    AscendC::LocalTensor<T> var;
    AscendC::LocalTensor<T> grad;

    int32_t blockIdx;
    float step_ = 0;

    int32_t singleBlockNum;
    int32_t singleSize;
    uint64_t useNumCore = 0;
    uint64_t lastPreCoreRowWork = 0;
    uint64_t notLastCoreNum = 0;
    uint64_t notLastPreCoreRowWork = 0;
    uint64_t lastCoreLastBlock = 0;
    uint64_t lastBlockSize = 0;
    int64_t oneCoreDoBlockNumPerRow = 0;

    float stepSize;
    float correction2;
    float lr;
    float beta1;
    float beta2;
    float weightDecay;
    float eps;
    float gnormScale;
    uint32_t blockSize;

    uint64_t gmOffset = 0;
    uint64_t absmaxOffset = 0;
};
} // namespace ApplyAdamWQuantNS
#endif // APPLY_ADAM_W_QUANT_FP32_H_
