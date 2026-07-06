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
 * \file inplace_index_add_with_sorted_fix.h
 * \brief A5 (ascend950) Fix kernel — bufCnt=3, float32 compute, Cast at CopyIn/CopyOut boundary
 *
 *   CopySelfIn:  Load T→staging, MicroAPI VF Cast T→float32→compute area
 *   CopyValueIn: Same: Load T→staging, MicroAPI VF Cast T→float32→compute area
 *   Compute:     Pure float32 VF (T=float, DIST_NORM, zero Cast inside VF)
 *   CopyOut:     AscendC Cast float32→T, DataCopy→VECOUT→GM
 */
#ifndef INPLACE_INDEX_ADD_WITH_SORTED_ARCH35_FIX_H_
#define INPLACE_INDEX_ADD_WITH_SORTED_ARCH35_FIX_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"
#include "inplace_index_add_with_sorted_struct.h"

using namespace AscendC;

constexpr int64_t BUFFER_NUM = 2;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t NUM_TWO = 2;
constexpr int64_t INDEX_UB_NUM = 1536;
constexpr uint32_t B32_REP_SIZE = platform::GetVRegSize() / sizeof(float);  // = 64

constexpr MicroAPI::CastTrait castTraitB162B32 = {
    MicroAPI::RegLayout::ZERO,
    MicroAPI::SatMode::UNKNOWN,
    MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::UNKNOWN,
};

__aicore__ inline int64_t CeilAlignFix(int64_t val, int64_t align)
{
    return ((val + align - 1) / align) * align;
}

// ============================================================================
// A5 VF functions — pure float32 (T=float), DIST_NORM, zero Cast inside VF
//   §6.6: oneRepeat = B32_REP_SIZE = 64 float elements per VF iteration
// ============================================================================

template <typename T>
__simd_vf__ inline void IndexAddSimdVf(
    __ubuf__ T* selfUb, __ubuf__ T* valueUb,
    __ubuf__ T* outUb, uint32_t count, uint32_t oneRepeat, uint16_t repeat)
{
    MicroAPI::RegTensor<T> selfReg, valueReg, outReg;
    MicroAPI::MaskReg mask;

    for (uint16_t r = 0; r < repeat; ++r) {
        uint32_t offset = r * oneRepeat;
        uint32_t curCount = (r == repeat - 1) ? (count - r * oneRepeat) : oneRepeat;
        mask = MicroAPI::UpdateMask<T>(curCount);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(selfReg,  selfUb  + offset);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(valueReg, valueUb + offset);
        MicroAPI::Add(outReg, selfReg, valueReg, mask);
        MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(outUb + offset, outReg, mask);
    }
}

template <typename T>
__simd_vf__ inline void IndexAddSimdVfAlpha(
    __ubuf__ T* selfUb, __ubuf__ T* valueUb,
    __ubuf__ T* outUb, uint32_t count, uint32_t oneRepeat, uint16_t repeat, float alphaVal)
{
    MicroAPI::RegTensor<T> selfReg, valueReg, scaledReg, outReg;
    MicroAPI::MaskReg mask;

    for (uint16_t r = 0; r < repeat; ++r) {
        uint32_t offset = r * oneRepeat;
        uint32_t curCount = (r == repeat - 1) ? (count - r * oneRepeat) : oneRepeat;
        mask = MicroAPI::UpdateMask<T>(curCount);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(selfReg,  selfUb  + offset);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(valueReg, valueUb + offset);
        MicroAPI::Muls(scaledReg, valueReg, alphaVal, mask);
        MicroAPI::Add(outReg, selfReg, scaledReg, mask);
        MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(outUb + offset, outReg, mask);
    }
}

template <typename T>
class InplaceIndexAddWithSortedFix
{
public:
    __aicore__ inline InplaceIndexAddWithSortedFix(
        TPipe* pipeIn, const InplaceIndexAddWithSortedTilingData* __restrict tilingData)
    {
        pipe = pipeIn;
        coreId = GetBlockIdx();

        usedCoreNum = tilingData->usedCoreNum;
        enableAlpha = tilingData->enableAlpha;
        eachIndexCountFix = tilingData->eachIndexCount;
        lastIndexCountFix = tilingData->lastIndexCount;
        inputCountFix = tilingData->inputCount;
        indicesCountFix = tilingData->indicesCount;
        updatesCountFix = tilingData->updatesCount;
        updatesOneTime = tilingData->updatesOneTime;
        maxSize = tilingData->maxSize;
        eachNumFix = tilingData->eachNum;
        eachLoopFix = tilingData->eachLoop;
        eachTailFix = tilingData->eachTail;
        eachUBIndexRound = tilingData->eachUBIndexRound;
        eachUBIndexCount = tilingData->eachUBIndexCount;
        eachUBIndexTail = tilingData->eachUBIndexTail;
        lastUBIndexRound = tilingData->lastUBIndexRound;
        lastUBIndexCount = tilingData->lastUBIndexCount;
        lastUBIndexTail = tilingData->lastUBIndexTail;

        currentUBIndexRound = coreId == (usedCoreNum - 1) ? lastUBIndexRound : eachUBIndexRound;
        currentIndexCount = coreId == (usedCoreNum - 1) ? lastUBIndexCount : eachUBIndexCount;
        currentUBIndexTail = coreId == (usedCoreNum - 1) ? lastUBIndexTail : eachUBIndexTail;
    }

    __aicore__ inline void Init(GM_ADDR var, GM_ADDR value, GM_ADDR sorted_indices, GM_ADDR pos, GM_ADDR alpha)
    {
        inputGmFix.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(var), inputCountFix);
        valueGmFix.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(value), updatesCountFix);

        if (enableAlpha == 1) {
            alphaGmFix.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(alpha), 1);
            alphaDataFix = alphaGmFix.GetValue(0);
            if (alphaDataFix == static_cast<float>(1.0)) {
                enableAlpha = 0;
            }
        }
        
        indicesGmFix.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sorted_indices), indicesCountFix);
        posGmFix.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(pos), indicesCountFix);
        finalIndex = indicesGmFix.GetValue(indicesCountFix - 1);
        
        pipe->InitBuffer(inQueueSelfFix, BUFFER_NUM, maxSize * sizeof(float));
        pipe->InitBuffer(inQueueValueFix, BUFFER_NUM, maxSize * sizeof(float));
        pipe->InitBuffer(inQueueIndex, 1, NUM_TWO * INDEX_UB_NUM * sizeof(int32_t));
        pipe->InitBuffer(outQueueOutFix, BUFFER_NUM, maxSize * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        if (coreId != 0) {
            int64_t lastOffset = eachIndexCountFix * coreId - 1;
            lastCoreIndex = indicesGmFix.GetValue(lastOffset);
        }

        for (int64_t i = 0; i < eachLoopFix; ++i) {
            currentEachNum = i == eachLoopFix - 1 ? eachTailFix : eachNumFix;
            currentEachNumAlign = CeilAlignFix(currentEachNum, BLOCK_SIZE);
            AvgProcess(i);
            if (coreId == usedCoreNum - 1 && lastCoreIndex != finalIndex) {
                CopyOut(finalIndex, i, currentEachNum);
                continue;
            } else if (coreId == usedCoreNum - 1) {
                continue;
            }
            SameIndexProcess(i);
        }
    }

private:
    __aicore__ inline void AvgProcess(int64_t loopIdx)
    {
        int32_t updateIndex, updatePos, lastUpdateIndex = -1;
        bool firstMov = true;
        for (int64_t idxRound = 0; idxRound < currentUBIndexRound; ++idxRound) {
            currentEachIndex = idxRound == currentUBIndexRound - 1 ? currentUBIndexTail : currentIndexCount;
            indexLocal = inQueueIndex.AllocTensor<int32_t>();
            int64_t indexOffset = coreId * eachIndexCountFix + idxRound * eachUBIndexCount;

            DataCopyPadExtParams<int32_t> tPadParams = {false, 0, 0, static_cast<int32_t>(0)};
            DataCopyExtParams updatesExtParams = {
                (uint16_t)1, static_cast<uint32_t>(currentEachIndex * sizeof(int32_t)), 0, 0, 0};
            DataCopyPad(indexLocal, indicesGmFix[indexOffset], updatesExtParams, tPadParams);
            DataCopyPad(indexLocal[INDEX_UB_NUM], posGmFix[indexOffset], updatesExtParams, tPadParams);

            event_t eventIDMTE2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
            SetFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
            WaitFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);

            for (int64_t j = 0; j < currentEachIndex; ++j) {
                int64_t indicesOffset = INDEX_UB_NUM + j;
                updateIndex = indexLocal.GetValue(j);
                if (coreId != 0 && updateIndex == lastCoreIndex) {
                    continue;
                }
                bool ifSyncIn = firstMov || updateIndex != lastUpdateIndex;
                bool ifSyncOut = (!firstMov && updateIndex != lastUpdateIndex);
                if (ifSyncOut) {
                    CopyOut(lastUpdateIndex, loopIdx, currentEachNum);
                }
                updatePos = indexLocal.GetValue(indicesOffset);

                event_t eventIDSToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
                SetFlag<HardEvent::S_MTE2>(eventIDSToMTE2);
                WaitFlag<HardEvent::S_MTE2>(eventIDSToMTE2);

                if (ifSyncIn) {
                    CopySelfIn(updateIndex, loopIdx, currentEachNum);
                }
                CopyValueIn(updatePos, loopIdx, currentEachNum);
                Compute(currentEachNumAlign);
                lastUpdateIndex = updateIndex;
                firstMov = false;
            }
            inQueueIndex.FreeTensor(indexLocal);
        }
    }

    __aicore__ inline void SameIndexProcess(int64_t loopIdx)
    {
        int32_t lastIndex = indicesGmFix.GetValue(eachIndexCountFix * (coreId + 1) - 1);
        int32_t nextIndex = indicesGmFix.GetValue(eachIndexCountFix * (coreId + 1));

        bool ifContinue = lastIndex == nextIndex;
        int32_t updatePos = 0;
        int64_t offset = 0;

        if ((coreId != 0 && lastCoreIndex == lastIndex)) {
            return;
        }
        if (!ifContinue) {
            CopyOut(lastIndex, loopIdx, currentEachNum);
            return;
        }
        while (ifContinue) {
            int64_t indicesOffset = eachIndexCountFix * (coreId + 1) + offset;
            updatePos = posGmFix.GetValue(indicesOffset);
            nextIndex = indicesGmFix.GetValue(indicesOffset);

            event_t eventIDSToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
            SetFlag<HardEvent::S_MTE2>(eventIDSToMTE2);
            WaitFlag<HardEvent::S_MTE2>(eventIDSToMTE2);

            CopyValueIn(updatePos, loopIdx, currentEachNum);
            Compute(currentEachNumAlign);
            offset += 1;
            indicesOffset = eachIndexCountFix * (coreId + 1) + offset;
            if (indicesOffset >= indicesCountFix) {
                break;
            }
            nextIndex = indicesGmFix.GetValue(indicesOffset);
            ifContinue = lastIndex == nextIndex;
        }
        CopyOut(lastIndex, loopIdx, currentEachNum);
        offset = 0;
        ifContinue = true;
    }

    // ---- §6.5.1 CopySelfIn: T→float32 via MicroAPI VF ----
    __aicore__ inline void CopySelfIn(int32_t index, int64_t progress, int64_t dataLen)
    {
        // 1. Alloc float buffer
        inputLocal = inQueueSelfFix.AllocTensor<float>();

        // 2. ReinterpretCast<T>, load T data to staging area (upper half)
        LocalTensor<T> inputLocalT = inputLocal.template ReinterpretCast<T>();
        int64_t inputOffset = CeilAlignFix(dataLen, BLOCK_UB_SIZE);

        DataCopyPadExtParams<T> tPadParams = {false, 0, 0, static_cast<T>(0)};
        DataCopyExtParams updatesExtParams = {(uint16_t)1, static_cast<uint32_t>(dataLen * sizeof(T)), 0, 0, 0};
        DataCopyPad(inputLocalT[inputOffset],
                    inputGmFix[index * updatesOneTime + progress * maxSize],
                    updatesExtParams, tPadParams);

        // 3. Sync MTE2 → V (ensure T staging data is ready for VF Cast)
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);

        int64_t count = CeilAlignFix(dataLen, BLOCK_UB_SIZE);
        uint16_t loops = AscendC::CeilDivision(count * sizeof(float), platform::GetVRegSize());
        uint32_t loopsStride = platform::GetVRegSize() / sizeof(float);  // = 64

        // 4. MicroAPI VF: T staging → DIST_UNPACK_B16 → Cast float32 → DataCopy to float compute area
        __VEC_SCOPE__ {
            __local_mem__ float* dst = reinterpret_cast<__local_mem__ float*>(inputLocal.GetPhyAddr());
            __local_mem__ T* src = reinterpret_cast<__local_mem__ T*>(inputLocalT.GetPhyAddr()) + inputOffset;
            uint32_t sreg = static_cast<uint32_t>(count);

            MicroAPI::MaskReg mask;
            MicroAPI::RegTensor<T> aReg;
            MicroAPI::RegTensor<float> bReg;

            for (uint16_t i = 0; i < loops; i++) {
                mask = MicroAPI::UpdateMask<float>(sreg);
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(aReg, src + i * loopsStride);
                MicroAPI::Cast<float, T, castTraitB162B32>(bReg, aReg, mask);
                MicroAPI::DataCopy(dst + i * loopsStride, bReg, mask);
            }
        }

        // 5. EnQue/DeQue float (sync MTE2 DMA), alloc VECOUT buffer
        inQueueSelfFix.EnQue<float>(inputLocal);
        inputLocal = inQueueSelfFix.DeQue<float>();
    }

    // ---- §6.5.2 CopyValueIn: T→float32 via MicroAPI VF (same as CopySelfIn) ----
    __aicore__ inline void CopyValueIn(int32_t index, int64_t progress, int64_t dataLen)
    {
        // 1. Alloc float buffer
        valueLocal = inQueueValueFix.AllocTensor<float>();

        // 2. ReinterpretCast<T>, load T data to staging area
        LocalTensor<T> valueLocalT = valueLocal.template ReinterpretCast<T>();
        int64_t valueOffset = CeilAlignFix(dataLen, BLOCK_UB_SIZE);
        DataCopyPadExtParams<T> tPadParams = {false, 0, 0, static_cast<T>(0)};
        DataCopyExtParams updatesExtParams = {(uint16_t)1, static_cast<uint32_t>(dataLen * sizeof(T)), 0, 0, 0};
        DataCopyPad(valueLocalT[valueOffset],
                    valueGmFix[index * updatesOneTime + progress * maxSize],
                    updatesExtParams, tPadParams);

        // 3. Sync MTE2 → V
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);

        int64_t count = CeilAlignFix(dataLen, BLOCK_UB_SIZE);
        uint16_t loops = AscendC::CeilDivision(count * sizeof(float), platform::GetVRegSize());
        uint32_t loopsStride = platform::GetVRegSize() / sizeof(float);  // = 64

        // 4. MicroAPI VF: T → float32 Cast
        __VEC_SCOPE__ {
            __local_mem__ float* dst = reinterpret_cast<__local_mem__ float*>(valueLocal.GetPhyAddr());
            __local_mem__ T* src = reinterpret_cast<__local_mem__ T*>(valueLocalT.GetPhyAddr()) + valueOffset;
            uint32_t sreg = static_cast<uint32_t>(count);

            MicroAPI::MaskReg mask;
            MicroAPI::RegTensor<T> aReg;
            MicroAPI::RegTensor<float> bReg;
            for (uint16_t i = 0; i < loops; i++) {
                mask = MicroAPI::UpdateMask<float>(sreg);
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(aReg, src + i * loopsStride);
                MicroAPI::Cast<float, T, castTraitB162B32>(bReg, aReg, mask);
                MicroAPI::DataCopy(dst + i * loopsStride, bReg, mask);
            }
        }

        // 5. EnQue float (no DeQue — Compute handles it)
        inQueueValueFix.EnQue<float>(valueLocal);
    }

    // ---- §6.5.3 Compute: pure float32 VF (T=float, zero Cast) ----
    __aicore__ inline void Compute(int64_t dataLenAlign)
    {
        valueLocal = inQueueValueFix.DeQue<float>();

        __ubuf__ float* selfUb  = reinterpret_cast<__ubuf__ float*>(inputLocal.GetPhyAddr());
        __ubuf__ float* valueUb = reinterpret_cast<__ubuf__ float*>(valueLocal.GetPhyAddr());

        uint32_t count = static_cast<uint32_t>(dataLenAlign);
        uint16_t repeat = AscendC::CeilDivision(count, B32_REP_SIZE);

        // VF writes result back to selfUb (in-place float32 accumulation)
        if (enableAlpha == 1) {
            AscendC::VF_CALL<IndexAddSimdVfAlpha<float>>(selfUb, valueUb, selfUb, count, B32_REP_SIZE, repeat, alphaDataFix);
        } else {
            AscendC::VF_CALL<IndexAddSimdVf<float>>(selfUb, valueUb, selfUb, count, B32_REP_SIZE, repeat);
        }

        inQueueValueFix.FreeTensor(valueLocal);
    }

    // ---- §6.5.4 CopyOut: AscendC Cast float32→T, then VECOUT→GM ----
    __aicore__ inline void CopyOut(int32_t index, int64_t progress, int64_t dataLen)
    {
        // wait for VF's add end
        PipeBarrier<PIPE_V>();

        LocalTensor<T> outT = outQueueOutFix.AllocTensor<T>();
        Cast(outT, inputLocal, RoundMode::CAST_RINT, CeilAlignFix(dataLen, BLOCK_SIZE));
        inQueueSelfFix.FreeTensor(inputLocal);
        outQueueOutFix.EnQue(outT);

        outT = outQueueOutFix.DeQue<T>();
        DataCopyExtParams updatesExtParams = {(uint16_t)1, static_cast<uint32_t>(dataLen * sizeof(T)), 0, 0, 0};
        DataCopyPad(inputGmFix[index * updatesOneTime + progress * maxSize], outT, updatesExtParams);
        outQueueOutFix.FreeTensor(outT);
    }

private:
    TPipe* pipe;
    GlobalTensor<T> inputGmFix;
    GlobalTensor<T> valueGmFix;
    GlobalTensor<float> alphaGmFix;
    GlobalTensor<int32_t> indicesGmFix;
    GlobalTensor<int32_t> posGmFix;
    LocalTensor<float> inputLocal;
    LocalTensor<float> valueLocal;
    LocalTensor<int32_t> indexLocal;

    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueSelfFix;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueValueFix;
    TQue<QuePosition::VECIN, 1> inQueueIndex;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueOutFix;

    static constexpr int64_t BLOCK_UB_SIZE = 32 / sizeof(T);  // T elem align for ReinterpretCast

    int64_t coreId;
    int32_t usedCoreNum;
    int32_t enableAlpha;
    int64_t eachIndexCountFix;
    int64_t lastIndexCountFix;
    int64_t inputCountFix;
    int64_t indicesCountFix;
    int64_t updatesCountFix;
    int64_t updatesOneTime;
    int64_t maxSize;
    int64_t eachNumFix;
    int64_t eachLoopFix;
    int64_t eachTailFix;
    int64_t currentEachIndex;
    int32_t finalIndex;
    int64_t currentUBIndexRound;
    int64_t currentIndexCount;
    int64_t currentUBIndexTail;
    int64_t currentEachNum;
    int64_t currentEachNumAlign;
    int32_t lastCoreIndex = -1;
    int64_t eachUBIndexRound = 1;
    int64_t eachUBIndexCount = 1;
    int64_t eachUBIndexTail = 0;
    int64_t lastUBIndexRound = 1;
    int64_t lastUBIndexCount = 1;
    int64_t lastUBIndexTail = 0;
    float alphaDataFix;
};

#endif // INPLACE_INDEX_ADD_WITH_SORTED_ARCH35_FIX_H_
