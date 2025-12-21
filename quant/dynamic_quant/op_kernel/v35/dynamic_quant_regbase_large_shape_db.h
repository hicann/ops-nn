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
 * \file dynamic_quant_large_shape_db.h
 * \brief
 */
#ifndef DYNAMIC_QUANT_REGBASE_LARGE_SHAPE_DB_H
#define DYNAMIC_QUANT_REGBASE_LARGE_SHAPE_DB_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace DynamicQuantNDOpt2 {
using namespace AscendC;

#ifndef INFINITY
#define INFINITY (__builtin_inff())
#endif

constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t DOUBLE_BUFFER_NUM = 2;
constexpr uint32_t FIFTEEN = 15;
constexpr uint32_t SIXTEEN = 16;
constexpr uint32_t THIRTY_ONE = 31;
constexpr uint32_t THIRTY_TWO = 32;
constexpr uint32_t SIXTY_THREE = 63;
constexpr uint32_t SIXTY_FOUR = 64;
constexpr uint32_t TWO_FIVE_SIX = 256;
constexpr float FP8_E5M2_MAX_VALUE = 57344.0f;
constexpr float FP8_E4M3FN_MAX_VALUE = 448.0f;
constexpr float HIFLOAT8_MAX_VALUE = 32768.0f;
constexpr float INT8_MAX_VALUE = 127.0f;
constexpr float INT4_MAX_VALUE = 7.0f;
constexpr float FP8_E5M2_MAX_VALUE_NO_SYM = 114688.0f;
constexpr float FP8_E4M3FN_MAX_VALUE_NO_SYM = 896.0f;
constexpr float HIFLOAT8_MAX_VALUE_NO_SYM = 65536.0f;
constexpr float INT8_MAX_VALUE_NO_SYM = 255.0f;
constexpr float INT4_MAX_VALUE_NO_SYM = 15.0f;
constexpr float MIN_FLOAT_VALUE = -INFINITY;
constexpr float MAX_FLOAT_VALUE = INFINITY;

template <typename T, typename yDtype, int64_t hasSmooth, bool isSymmetrical = true>
class DynamicQuantLargeShapeDb {
public:
    using yCopyDtype = std::conditional_t<IsSameType<yDtype, int4b_t>::value, uint8_t, yDtype>;
    __aicore__ inline DynamicQuantLargeShapeDb(TPipe *pipe) { pPipe = pipe; }
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR smooth_scales, GM_ADDR y, GM_ADDR scale, GM_ADDR offset,
                                GM_ADDR workSpace, const DynamicQuantTilingData& tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void SetMaxValue();
    __aicore__ inline void SetSymMaxValue();
    __aicore__ inline void SetNoSymMaxValue();
    __aicore__ inline void ProcessLoop(uint32_t i, uint32_t j);
    __aicore__ inline void CopyInByEle(int64_t offset, uint32_t loopIndex, uint32_t elementNum, uint8_t rightPadding);
    __aicore__ inline void ComputeMaxScale(uint32_t elementNum, __local_mem__ float* scaleLocalAddr,
                                           __local_mem__ float* offsetLocalAddr);
    __aicore__ inline void ComputeMaxScaleAndYTail(uint32_t elementNum, __local_mem__ float* scaleLocalAddr,
                                                   __local_mem__ float* offsetLocalAddr);
    template <bool isFinal = false>
    __aicore__ inline void ComputeMaxScaleVF(__local_mem__ T* inLocalAddr, __local_mem__ T* smoothLocalAddr,
                                             __local_mem__ float* maxLocalAddr, __local_mem__ float* minLocalAddr,
                                             uint32_t elementNum);
    __aicore__ inline void ComputeY(uint32_t elementNum, __local_mem__ float* scaleLocalAddr,
                                    __local_mem__ float* offsetLocalAddr);
    __aicore__ inline void ComputeYVF(__local_mem__ T* inLocalAddr, __local_mem__ T* smoothLocalAddr,
                                      __local_mem__ yCopyDtype* outAddr, __local_mem__ float* scaleLocalAddr,
                                      __local_mem__ float* offsetLocalAddr, uint32_t elementNum);
    __aicore__ inline void ParseTilingData(const DynamicQuantTilingData& tilingData);
    __aicore__ inline void CopyOutScale(int64_t offset);
    __aicore__ inline void CopyOutY(int64_t offset, uint32_t element);

private:
    TPipe* pPipe = nullptr;
    /* tiling data */
    DynamicQuantTilingData tilingData_;

    /* ascendc variable */
    TQue<QuePosition::VECIN, DOUBLE_BUFFER_NUM> inQueue;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER_NUM> smoothQueue;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER_NUM> outQueue;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER_NUM> scaleQueue;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER_NUM> offsetQueue;

    /* global memory address */
    GlobalTensor<T> inGm;
    GlobalTensor<T> smoothGm;
    GlobalTensor<yCopyDtype> outGm;
    GlobalTensor<float> scaleGm;
    GlobalTensor<float> offsetGm;

    uint32_t blockIdx;
    uint32_t sizeFloatLen;
    uint32_t outAlignLen;
    uint32_t rowPerHeadCore;
    uint32_t rowPerTailCore;
    uint32_t lenHead;
    uint32_t lenTail;
    uint32_t outLen;
    uint32_t outLenHead;
    uint32_t outLenTail;
    uint32_t loopCnt = 0;
    uint32_t loopCntHead;
    uint32_t loopCntTail;
    uint8_t rightPadding = 0;

    int64_t offsetBase = 0;
    int64_t srcOffset = 0;
    float maxValue = 0.0;
    float scaleMaxValue = 0.0;
    float offsetMaxValue = 0.0;

    constexpr static AscendC::MicroAPI::CastTrait castTrait0 = {
        AscendC::MicroAPI::RegLayout::ZERO,
        AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN};
    constexpr static AscendC::MicroAPI::CastTrait castTraitF32toI16 = {
        AscendC::MicroAPI::RegLayout::ZERO,
        AscendC::MicroAPI::SatMode::NO_SAT,
        AscendC::MicroAPI::MaskMergeMode::ZEROING,
        RoundMode::CAST_RINT};
    constexpr static AscendC::MicroAPI::CastTrait castTraitI16toF16 = {
        AscendC::MicroAPI::RegLayout::ZERO,
        AscendC::MicroAPI::SatMode::NO_SAT,
        AscendC::MicroAPI::MaskMergeMode::ZEROING,
        RoundMode::CAST_ROUND};
    constexpr static AscendC::MicroAPI::CastTrait castTraitF16toI8 = {
        AscendC::MicroAPI::RegLayout::ZERO,
        AscendC::MicroAPI::SatMode::NO_SAT,
        AscendC::MicroAPI::MaskMergeMode::ZEROING,
        RoundMode::CAST_TRUNC};
    constexpr static AscendC::MicroAPI::CastTrait castTrait32tofp8 = {
        AscendC::MicroAPI::RegLayout::ZERO,
        AscendC::MicroAPI::SatMode::NO_SAT,
        AscendC::MicroAPI::MaskMergeMode::ZEROING,
        RoundMode::CAST_RINT};
    constexpr static AscendC::MicroAPI::CastTrait castTrait32toh8 = {
        AscendC::MicroAPI::RegLayout::ZERO,
        AscendC::MicroAPI::SatMode::NO_SAT,
        AscendC::MicroAPI::MaskMergeMode::ZEROING,
        RoundMode::CAST_ROUND};
};

template <typename T, typename yDtype, int64_t hasSmooth, bool isSymmetrical>
__aicore__ inline void DynamicQuantLargeShapeDb<T, yDtype, hasSmooth, isSymmetrical>::Init(
    GM_ADDR x, GM_ADDR smooth_scales, GM_ADDR y, GM_ADDR scale, GM_ADDR offset,
    GM_ADDR workSpace, const DynamicQuantTilingData& tilingData)
{
    blockIdx = GetBlockIdx();

    ParseTilingData(tilingData);

    rowPerHeadCore = tilingData_.rowPerHeadCore;
    rowPerTailCore = tilingData_.rowPerTailCore;

    // 如果是大shape，一次只算一行，所以不存在尾行
    if (blockIdx < tilingData_.headCoreNum) {
        loopCnt = rowPerHeadCore;
    }
    if (blockIdx >= tilingData_.headCoreNum && blockIdx < tilingData_.coreNum) {
        loopCnt = rowPerTailCore;
    }

    // 尾轴切分，最后剩余的部分
    uint32_t sizeTailLen = (tilingData_.innerLoopTail + FIFTEEN) / SIXTEEN * SIXTEEN;
    rightPadding = sizeTailLen - tilingData_.innerLoopTail;
    loopCntHead = loopCnt / THIRTY_TWO;
    loopCntTail = loopCnt % THIRTY_TWO;
    lenHead = rowPerHeadCore * tilingData_.rowLen;
    lenTail = rowPerTailCore * tilingData_.rowLen;

    if constexpr (IsSameType<yDtype, int4b_t>::value) {
        outAlignLen = (tilingData_.innerLoopEle + SIXTY_THREE) / SIXTY_FOUR * SIXTY_FOUR;
        outLenHead = lenHead >> 1;
        outLenTail = lenTail >> 1;
    } else {
        outAlignLen = tilingData_.innerLoopEle;
        outLenHead = lenHead;
        outLenTail = lenTail;
    }

    if (blockIdx < tilingData_.headCoreNum) {
        inGm.SetGlobalBuffer((__gm__ T*)x + blockIdx * lenHead, lenHead);
        outGm.SetGlobalBuffer((__gm__ yCopyDtype*)y + blockIdx * outLenHead, outLenHead);
        scaleGm.SetGlobalBuffer((__gm__ float*)scale + blockIdx * rowPerHeadCore, rowPerHeadCore);
        if constexpr (!isSymmetrical) {
            offsetGm.SetGlobalBuffer((__gm__ float*)offset + blockIdx * rowPerHeadCore, rowPerHeadCore);
        }
    } else {
        inGm.SetGlobalBuffer(
            (__gm__ T*)x + tilingData_.headCoreNum * lenHead + (blockIdx - tilingData_.headCoreNum) * lenTail, lenTail);
        outGm.SetGlobalBuffer(
            (__gm__ yCopyDtype*)y + tilingData_.headCoreNum * outLenHead + (blockIdx - tilingData_.headCoreNum) * outLenTail,
            outLenTail);
        scaleGm.SetGlobalBuffer((__gm__ float*)scale + tilingData_.headCoreNum * rowPerHeadCore +
                                (blockIdx - tilingData_.headCoreNum) * rowPerTailCore,
                                rowPerTailCore);
        if constexpr (!isSymmetrical) {
            offsetGm.SetGlobalBuffer((__gm__ float*)offset + tilingData_.headCoreNum * rowPerHeadCore +
                                     (blockIdx - tilingData_.headCoreNum) * rowPerTailCore,
                                     rowPerTailCore);
        }
    }

    // innerLoopEle已经保证了32Bytes对齐
    if constexpr (hasSmooth == 1) {
        smoothGm.SetGlobalBuffer((__gm__ T*)smooth_scales);
        pPipe->InitBuffer(smoothQueue, DOUBLE_BUFFER_NUM, tilingData_.innerLoopEle * sizeof(T));
    }
    pPipe->InitBuffer(inQueue, DOUBLE_BUFFER_NUM, tilingData_.innerLoopEle * sizeof(T));
    pPipe->InitBuffer(outQueue, DOUBLE_BUFFER_NUM, tilingData_.innerLoopEle * sizeof(yCopyDtype));
    pPipe->InitBuffer(scaleQueue, DOUBLE_BUFFER_NUM, TWO_FIVE_SIX);
    if constexpr (!isSymmetrical) {
        pPipe->InitBuffer(offsetQueue, DOUBLE_BUFFER_NUM, TWO_FIVE_SIX);
    }

    SetMaxValue();
}

template <typename T, typename yDtype, int64_t hasSmooth, bool isSymmetrical>
__aicore__ inline void DynamicQuantLargeShapeDb<T, yDtype, hasSmooth, isSymmetrical>::SetMaxValue()
{
    if constexpr (isSymmetrical) {
        SetSymMaxValue();
    } else {
        SetNoSymMaxValue();
    }
}

template <typename T, typename yDtype, int64_t hasSmooth, bool isSymmetrical>
__aicore__ inline void DynamicQuantLargeShapeDb<T, yDtype, hasSmooth, isSymmetrical>::SetSymMaxValue()
{
    if constexpr (IsSameType<yDtype, int8_t>::value) {
        scaleMaxValue = static_cast<float>(1.0) / INT8_MAX_VALUE;
    } else if constexpr (IsSameType<yDtype, int4b_t>::value) {
        scaleMaxValue = static_cast<float>(1.0) / INT4_MAX_VALUE;
    } else if constexpr (IsSameType<yDtype, fp8_e5m2_t>::value) {
        scaleMaxValue = static_cast<float>(1.0) / FP8_E5M2_MAX_VALUE;
    } else if constexpr (IsSameType<yDtype, fp8_e4m3fn_t>::value) {
        scaleMaxValue = static_cast<float>(1.0) / FP8_E4M3FN_MAX_VALUE;
    } else if constexpr (IsSameType<yDtype, hifloat8_t>::value) {
        scaleMaxValue = static_cast<float>(1.0) / HIFLOAT8_MAX_VALUE;
    }
}

template <typename T, typename yDtype, int64_t hasSmooth, bool isSymmetrical>
__aicore__ inline void DynamicQuantLargeShapeDb<T, yDtype, hasSmooth, isSymmetrical>::SetNoSymMaxValue()
{
    if constexpr (IsSameType<yDtype, int8_t>::value) {
        scaleMaxValue = static_cast<float>(1.0) / INT8_MAX_VALUE_NO_SYM;
        offsetMaxValue = INT8_MAX_VALUE;
    } else if constexpr (IsSameType<yDtype, int4b_t>::value) {
        scaleMaxValue = static_cast<float>(1.0) / INT4_MAX_VALUE_NO_SYM;
        offsetMaxValue = INT4_MAX_VALUE;
    } else if constexpr (IsSameType<yDtype, fp8_e5m2_t>::value) {
        scaleMaxValue = static_cast<float>(1.0) / FP8_E5M2_MAX_VALUE_NO_SYM;
        offsetMaxValue = FP8_E5M2_MAX_VALUE;
    } else if constexpr (IsSameType<yDtype, fp8_e4m3fn_t>::value) {
        scaleMaxValue = static_cast<float>(1.0) / FP8_E4M3FN_MAX_VALUE_NO_SYM;
        offsetMaxValue = FP8_E4M3FN_MAX_VALUE;
    } else if constexpr (IsSameType<yDtype, hifloat8_t>::value) {
        scaleMaxValue = static_cast<float>(1.0) / HIFLOAT8_MAX_VALUE_NO_SYM;
        offsetMaxValue = HIFLOAT8_MAX_VALUE;
    }
}

template <typename T, typename yDtype, int64_t hasSmooth, bool isSymmetrical>
__aicore__ inline void DynamicQuantLargeShapeDb<T, yDtype, hasSmooth, isSymmetrical>::ParseTilingData(const DynamicQuantTilingData& tilingData) {
    tilingData_.coreNum = tilingData.coreNum;
    tilingData_.rowLen = tilingData.rowLen;
    tilingData_.headCoreNum = tilingData.headCoreNum;
    tilingData_.rowPerHeadCore = tilingData.rowPerHeadCore;
    tilingData_.rowPerTailCore = tilingData.rowPerTailCore;
    tilingData_.multiRowNumTailCore = tilingData.multiRowNumTailCore;
    tilingData_.multiRowNumHeadCore = tilingData.multiRowNumHeadCore;
    tilingData_.innerLoopEle = tilingData.innerLoopEle;
    tilingData_.innerLoopTimes = tilingData.innerLoopTimes;
    tilingData_.innerLoopTail = tilingData.innerLoopTail;
    tilingData_.groupNum = tilingData.groupNum;
    tilingData_.alignGroupNum = tilingData.alignGroupNum;
    tilingData_.hasSmooth = tilingData.hasSmooth;
}

template <typename T, typename yDtype, int64_t hasSmooth, bool isSymmetrical>
__aicore__ inline void DynamicQuantLargeShapeDb<T, yDtype, hasSmooth, isSymmetrical>::Process()
{
    if (blockIdx >= tilingData_.coreNum) {
        return;
    }

    for (uint32_t i = 0; i < loopCntHead; i++) {
        for (uint32_t j = 0; j < THIRTY_TWO; j++) {
            ProcessLoop(i, j);
        }
    }

    for (uint32_t i = 0; i < loopCntTail; i++) {
        ProcessLoop(loopCntHead, i);
    }
}


template <typename T, typename yDtype, int64_t hasSmooth, bool isSymmetrical>
__aicore__ inline void DynamicQuantLargeShapeDb<T, yDtype, hasSmooth, isSymmetrical>::ProcessLoop(uint32_t i, uint32_t j)
{
    LocalTensor<float> scaleLocal = scaleQueue.AllocTensor<float>();
    __local_mem__ float* scaleLocalAddr = (__local_mem__ float*)scaleLocal.GetPhyAddr();
    LocalTensor<float> offsetLocal;
    __local_mem__ float* offsetLocalAddr;

    if constexpr (isSymmetrical) {
        AscendC::Duplicate(scaleLocal, (float)0.0, 64, 1, 1, 8);
    } else {
        offsetLocal = offsetQueue.AllocTensor<float>();
        offsetLocalAddr = (__local_mem__ float*)offsetLocal.GetPhyAddr();
        AscendC::Duplicate(scaleLocal, MIN_FLOAT_VALUE, 64, 1, 1, 8);
        AscendC::Duplicate(offsetLocal, MAX_FLOAT_VALUE, 64, 1, 1, 8);
    }

    offsetBase = i * THIRTY_TWO + j;
    srcOffset = offsetBase * tilingData_.rowLen;
    for (uint32_t innerLoopIndex = 0; innerLoopIndex < tilingData_.innerLoopTimes; innerLoopIndex++) {
        CopyInByEle(srcOffset, innerLoopIndex, tilingData_.innerLoopEle, 0);
        ComputeMaxScale(tilingData_.innerLoopEle, scaleLocalAddr, offsetLocalAddr);
        srcOffset += tilingData_.innerLoopEle;
    }
    if(tilingData_.innerLoopTail > 0) { // 如果核内切的有尾块
        CopyInByEle(srcOffset, tilingData_.innerLoopTimes, tilingData_.innerLoopTail, rightPadding);
        ComputeMaxScaleAndYTail(tilingData_.innerLoopTail, scaleLocalAddr, offsetLocalAddr);
        CopyOutY(srcOffset, tilingData_.innerLoopTail);
    }

    srcOffset = offsetBase * tilingData_.rowLen;
    for (uint32_t innerLoopIndex = 0; innerLoopIndex < tilingData_.innerLoopTimes; innerLoopIndex++) {
        CopyInByEle(srcOffset, innerLoopIndex, tilingData_.innerLoopEle, 0);
        ComputeY(tilingData_.innerLoopEle, scaleLocalAddr, offsetLocalAddr);
        CopyOutY(srcOffset, tilingData_.innerLoopEle);
        srcOffset += tilingData_.innerLoopEle;
    }
    if constexpr (!isSymmetrical) {
        offsetQueue.EnQue<float>(offsetLocal);
    }
    scaleQueue.EnQue<float>(scaleLocal);
    CopyOutScale(offsetBase);
}

template <typename T, typename yDtype, int64_t hasSmooth, bool isSymmetrical>
__aicore__ inline void DynamicQuantLargeShapeDb<T, yDtype, hasSmooth, isSymmetrical>::CopyInByEle(int64_t offset, uint32_t loopIndex, uint32_t elementNum, uint8_t rightPadding) {
    DataCopyParams copyParams{1, (uint16_t)(elementNum * sizeof(T)), 0, 0};
    DataCopyPadParams padParams{true, 0, rightPadding, 0};

    LocalTensor<T> inLocal = inQueue.AllocTensor<T>();
    DataCopyPad(inLocal, inGm[offset], copyParams, padParams);
    inQueue.EnQue(inLocal);

    if constexpr (hasSmooth == 1) {
        LocalTensor<T> smoothLocal = smoothQueue.AllocTensor<T>();
        DataCopyPad(smoothLocal, smoothGm[loopIndex * tilingData_.innerLoopEle], copyParams, padParams);
        smoothQueue.EnQue(smoothLocal);
    }
}

template <typename T, typename yDtype, int64_t hasSmooth, bool isSymmetrical>
__aicore__ inline void DynamicQuantLargeShapeDb<T, yDtype, hasSmooth, isSymmetrical>::ComputeMaxScale(
    uint32_t elementNum, __local_mem__ float* scaleLocalAddr, __local_mem__ float* offsetLocalAddr)
{
    LocalTensor<T> inLocal = inQueue.DeQue<T>();
    __local_mem__ T* inLocalAddr = (__local_mem__ T*)inLocal.GetPhyAddr();
    LocalTensor<T> smoothLocal;
    __local_mem__ T* smoothLocalAddr;

    if constexpr (hasSmooth == 1) {
        smoothLocal = smoothQueue.DeQue<T>();
        smoothLocalAddr = (__local_mem__ T*)smoothLocal.GetPhyAddr();
    }

    ComputeMaxScaleVF(inLocalAddr, smoothLocalAddr, scaleLocalAddr, offsetLocalAddr, elementNum);

    if constexpr (hasSmooth == 1) {
        smoothQueue.FreeTensor(smoothLocal);
    }

    inQueue.FreeTensor(inLocal);
}

template <typename T, typename yDtype, int64_t hasSmooth, bool isSymmetrical>
__aicore__ inline void DynamicQuantLargeShapeDb<T, yDtype, hasSmooth, isSymmetrical>::ComputeMaxScaleAndYTail(
    uint32_t elementNum, __local_mem__ float* scaleLocalAddr, __local_mem__ float* offsetLocalAddr)
{
    LocalTensor<T> inLocal = inQueue.DeQue<T>();
    __local_mem__ T* inLocalAddr = (__local_mem__ T*)inLocal.GetPhyAddr();
    LocalTensor<yCopyDtype> outLocal = outQueue.AllocTensor<yCopyDtype>();
    __local_mem__ yCopyDtype* outAddr = (__local_mem__ yCopyDtype*)outLocal.GetPhyAddr();
    LocalTensor<T> smoothLocal;
    __local_mem__ T* smoothLocalAddr;

    if constexpr (hasSmooth == 1) {
        smoothLocal = smoothQueue.DeQue<T>();
        smoothLocalAddr = (__local_mem__ T*)smoothLocal.GetPhyAddr();
    }

    ComputeMaxScaleVF<true>(inLocalAddr, smoothLocalAddr, scaleLocalAddr, offsetLocalAddr, elementNum);
    ComputeYVF(inLocalAddr, smoothLocalAddr, outAddr, scaleLocalAddr, offsetLocalAddr, elementNum);

    if constexpr (hasSmooth == 1) {
        smoothQueue.FreeTensor(smoothLocal);
    }

    inQueue.FreeTensor(inLocal);
    outQueue.EnQue(outLocal);
}

template <typename T, typename yDtype, int64_t hasSmooth, bool isSymmetrical>
template <bool isFinal>
__aicore__ inline void DynamicQuantLargeShapeDb<T, yDtype, hasSmooth, isSymmetrical>::ComputeMaxScaleVF(
    __local_mem__ T* inLocalAddr, __local_mem__ T* smoothLocalAddr,
    __local_mem__ float* maxLocalAddr, __local_mem__ float* minLocalAddr,
    uint32_t elementNum)
{
    uint32_t dtypeSize = sizeof(float);
    uint32_t VL = AscendC::VECTOR_REG_WIDTH / dtypeSize;
    uint16_t vfLoopNum = (elementNum + VL - 1) / VL;
    uint32_t tailNum = (elementNum % VL == 0) ? VL : elementNum % VL;
    uint32_t maskNum;

    __VEC_SCOPE__ {
        AscendC::MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vreg0;
        AscendC::MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vreg1;
        AscendC::MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vreg2;
        AscendC::MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vreg3;
        AscendC::MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vreg5;
        AscendC::MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vreg6;
        AscendC::MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vregMax;
        AscendC::MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vreg8;
        AscendC::MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vregMin;
        AscendC::MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vregMaxTail;
        AscendC::MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vregMinTail;
        static constexpr AscendC::MicroAPI::DivSpecificMode mode = {AscendC::MicroAPI::MaskMergeMode::ZEROING, true};

        AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(vreg6, maxLocalAddr);
        if constexpr (!isSymmetrical) {
            AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(vreg8, minLocalAddr);
        }

        AscendC::MicroAPI::MaskReg mask;
        AscendC::MicroAPI::MaskReg maskAll = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();

        for (uint16_t i = 0; i < static_cast<uint16_t>(vfLoopNum - 1); i++)
        {
            AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vreg0, inLocalAddr + i * VL);
            AscendC::MicroAPI::Cast<float, T, castTrait0>(vreg1, vreg0, maskAll);
            if constexpr (hasSmooth == 1) {
                AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vreg2, smoothLocalAddr + i * VL);
                AscendC::MicroAPI::Cast<float, T, castTrait0>(vreg3, vreg2, maskAll);
                AscendC::MicroAPI::Mul(vreg1, vreg1, vreg3, maskAll);
            }

            if constexpr (isSymmetrical) {
                AscendC::MicroAPI::Abs(vreg5, vreg1, maskAll);
                AscendC::MicroAPI::Muls(vreg5, vreg5, scaleMaxValue, maskAll);
                AscendC::MicroAPI::Max(vreg6, vreg5, vreg6, maskAll);
            } else {
                AscendC::MicroAPI::Max(vreg6, vreg1, vreg6, maskAll);
                AscendC::MicroAPI::Min(vreg8, vreg1, vreg8, maskAll);
            }
        }
        {
            if constexpr (isSymmetrical) {
                AscendC::MicroAPI::ReduceMax<float>(vregMax, vreg6, maskAll);
            } else {
                AscendC::MicroAPI::ReduceMax<float>(vregMax, vreg6, maskAll);
                AscendC::MicroAPI::ReduceMin<float>(vregMin, vreg8, maskAll);
            }
            mask = AscendC::MicroAPI::UpdateMask<float>(tailNum);
            AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vreg0, inLocalAddr + (vfLoopNum - 1) * VL);
            AscendC::MicroAPI::Cast<float, T, castTrait0>(vreg1, vreg0, mask);
            if constexpr (hasSmooth == 1) {
                AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vreg2, smoothLocalAddr + (vfLoopNum - 1) * VL);
                AscendC::MicroAPI::Cast<float, T, castTrait0>(vreg3, vreg2, mask);
                AscendC::MicroAPI::Mul(vreg1, vreg1, vreg3, mask);
            }
            if constexpr (isSymmetrical) {
                AscendC::MicroAPI::Abs(vreg5, vreg1, mask);
                AscendC::MicroAPI::Muls(vreg5, vreg5, scaleMaxValue, mask);
                AscendC::MicroAPI::Max(vreg6, vreg5, vreg6, mask);
                AscendC::MicroAPI::ReduceMax<float>(vregMaxTail, vreg6, mask);
                AscendC::MicroAPI::Max(vregMax, vregMax, vregMaxTail, mask);
            } else {
                AscendC::MicroAPI::Max(vreg6, vreg1, vreg6, mask);
                AscendC::MicroAPI::Min(vreg8, vreg1, vreg8, mask);
                AscendC::MicroAPI::ReduceMax<float>(vregMaxTail, vreg6, mask);
                AscendC::MicroAPI::ReduceMin<float>(vregMinTail, vreg8, mask);
                AscendC::MicroAPI::Max(vregMax, vregMax, vregMaxTail, mask);
                AscendC::MicroAPI::Min(vregMin, vregMin, vregMinTail, mask);
            }
        }

        if constexpr (isSymmetrical) {
            AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(maxLocalAddr, vregMax, maskAll);
        } else {
            if constexpr (isFinal) {
                AscendC::MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vreg9;
                AscendC::MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vreg10;
                AscendC::MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vreg11;
                AscendC::MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vreg12;
                AscendC::MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vreg13;
                AscendC::MicroAPI::Sub(vreg9, vregMax, vregMin, maskAll);
                AscendC::MicroAPI::Muls(vreg10, vreg9, scaleMaxValue, maskAll); // scale

                AscendC::MicroAPI::Div<float,&mode>(vreg11, vregMax, vreg10, maskAll);
                AscendC::MicroAPI::Muls(vreg12, vreg11, -1, maskAll);
                AscendC::MicroAPI::Adds(vreg13, vreg12, offsetMaxValue, maskAll);  // offset

                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(maxLocalAddr, vreg10, maskAll);
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(minLocalAddr, vreg13, maskAll);
            } else {
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(maxLocalAddr, vregMax, maskAll);
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(minLocalAddr, vregMin, maskAll);
            }
        }
    }
}

template <typename T, typename yDtype, int64_t hasSmooth, bool isSymmetrical>
__aicore__ inline void DynamicQuantLargeShapeDb<T, yDtype, hasSmooth, isSymmetrical>::ComputeY(
    uint32_t elementNum, __local_mem__ float* scaleLocalAddr, __local_mem__ float* offsetLocalAddr)
{
LocalTensor<T> inLocal = inQueue.DeQue<T>();
    __local_mem__ T* inLocalAddr = (__local_mem__ T*)inLocal.GetPhyAddr();
    LocalTensor<yCopyDtype> outLocal = outQueue.AllocTensor<yCopyDtype>();
    __local_mem__ yCopyDtype* outAddr = (__local_mem__ yCopyDtype*)outLocal.GetPhyAddr();
    LocalTensor<T> smoothLocal;
    __local_mem__ T* smoothLocalAddr;

    if constexpr (hasSmooth == 1) {
        smoothLocal = smoothQueue.DeQue<T>();
        smoothLocalAddr = (__local_mem__ T*)smoothLocal.GetPhyAddr();
    }

    ComputeYVF(inLocalAddr, smoothLocalAddr, outAddr, scaleLocalAddr, offsetLocalAddr, elementNum);

    if constexpr (hasSmooth == 1) {
        smoothQueue.FreeTensor(smoothLocal);
    }

    inQueue.FreeTensor(inLocal);
    outQueue.EnQue(outLocal);
}

template <typename T, typename yDtype, int64_t hasSmooth, bool isSymmetrical>
__aicore__ inline void DynamicQuantLargeShapeDb<T, yDtype, hasSmooth, isSymmetrical>::ComputeYVF(
    __local_mem__ T* inLocalAddr, __local_mem__ T* smoothLocalAddr,
    __local_mem__ yCopyDtype* outAddr, __local_mem__ float* scaleLocalAddr,
    __local_mem__ float* offsetLocalAddr, uint32_t elementNum)
{
    uint32_t dtypeSize = sizeof(float);
    uint32_t VL = AscendC::VECTOR_REG_WIDTH / dtypeSize;
    uint16_t vfLoopNum = (elementNum + VL - 1) / VL;

    __VEC_SCOPE__ {
        AscendC::MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vreg0;
        AscendC::MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vreg1;
        AscendC::MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vreg2;
        AscendC::MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vreg3;
        AscendC::MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vreg4;
        AscendC::MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vreg5;
        AscendC::MicroAPI::RegTensor<int16_t, MicroAPI::RegTraitNumOne> vreg6;
        AscendC::MicroAPI::RegTensor<half, MicroAPI::RegTraitNumOne> vreg7;
        AscendC::MicroAPI::RegTensor<yCopyDtype, MicroAPI::RegTraitNumOne> vreg8;
        AscendC::MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vreg9;

        AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(vreg9, scaleLocalAddr);

        AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg mask2 = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::H>();
        for (uint16_t i = 0; i < vfLoopNum; i++)
        {
            auto addr = outAddr + i * VL;
            AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vreg0, inLocalAddr + i * VL);
            AscendC::MicroAPI::Cast<float, T, castTrait0>(vreg1, vreg0, mask);

            if constexpr (hasSmooth == 1) {
                AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vreg2, smoothLocalAddr + i * VL);
                AscendC::MicroAPI::Cast<float, T, castTrait0>(vreg3, vreg2, mask);
                AscendC::MicroAPI::Mul(vreg4, vreg1, vreg3, mask);
                AscendC::MicroAPI::Div(vreg5, vreg4, vreg9, mask);
            } else {
                AscendC::MicroAPI::Div(vreg5, vreg1, vreg9, mask);
            }
            if constexpr (!isSymmetrical) {
                AscendC::MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vregOffset;
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(vregOffset, offsetLocalAddr);
                AscendC::MicroAPI::Add(vreg5, vreg5, vregOffset, mask);
            }

            if constexpr (IsSameType<yDtype, int8_t>::value) {
                AscendC::MicroAPI::Cast<int16_t, float, castTraitF32toI16>(vreg6, vreg5, mask);
                AscendC::MicroAPI::Cast<half, int16_t, castTraitI16toF16>(vreg7, vreg6, mask);
                AscendC::MicroAPI::Cast<yDtype, half, castTraitF16toI8>(vreg8, vreg7, mask);
            } else if constexpr (IsSameType<yDtype, int4b_t>::value) {
                AscendC::MicroAPI::RegTensor<uint16_t, MicroAPI::RegTraitNumOne> vreg20;
                AscendC::MicroAPI::Cast<int16_t, float, castTraitF32toI16>(vreg6, vreg5, mask);
                AscendC::MicroAPI::Cast<half, int16_t, castTraitI16toF16>(vreg7, vreg6, mask);
                AscendC::MicroAPI::Pack(vreg20, (AscendC::MicroAPI::RegTensor<uint32_t>&)vreg7);
                AscendC::MicroAPI::Cast<int4x2_t, half, castTraitF16toI8>(
                    (AscendC::MicroAPI::RegTensor<int4x2_t>&)vreg8,
                    (AscendC::MicroAPI::RegTensor<half>&)vreg20, mask);
                addr = outAddr + (i * VL) / 2;
            } else if constexpr (IsSameType<yDtype, hifloat8_t>::value) {
                AscendC::MicroAPI::Cast<yDtype, float, castTrait32toh8>(vreg8, vreg5, mask);
            } else if constexpr (IsSameType<yDtype, fp8_e4m3fn_t>::value || IsSameType<yDtype, fp8_e5m2_t>::value)  {
                AscendC::MicroAPI::Cast<yDtype, float, castTrait32tofp8>(vreg8, vreg5, mask);
            }

            if constexpr (IsSameType<yDtype, int4b_t>::value) {
                AscendC::MicroAPI::DataCopy<yCopyDtype, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(addr, vreg8, mask2);
            } else {
                AscendC::MicroAPI::DataCopy<yCopyDtype, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(addr, vreg8, mask);
            }
        }
    }
}

template <typename T, typename yDtype, int64_t hasSmooth, bool isSymmetrical>
__aicore__ inline void DynamicQuantLargeShapeDb<T, yDtype, hasSmooth, isSymmetrical>::CopyOutScale(int64_t offset) {
    LocalTensor<float> scaleLocal = scaleQueue.DeQue<float>();
    DataCopyParams copyParams{1, (uint16_t)(sizeof(float)), 0, 0};
    DataCopyPad(scaleGm[offset], scaleLocal, copyParams);
    scaleQueue.FreeTensor(scaleLocal);
    if constexpr (!isSymmetrical) {
        LocalTensor<float> offsetLocal = offsetQueue.DeQue<float>();
        DataCopyPad(offsetGm[offset], offsetLocal, copyParams);
        offsetQueue.FreeTensor(offsetLocal);
    }
}

template <typename T, typename yDtype, int64_t hasSmooth, bool isSymmetrical>
__aicore__ inline void DynamicQuantLargeShapeDb<T, yDtype, hasSmooth, isSymmetrical>::CopyOutY(int64_t offset, uint32_t element) {
    LocalTensor<yCopyDtype> outLocal = outQueue.DeQue<yCopyDtype>();
    DataCopyExtParams copyExtParams{(uint16_t)1, (uint16_t)(element * sizeof(yCopyDtype)), 0, 0, 0};
    if constexpr (IsSameType<yDtype, int4b_t>::value) {
        copyExtParams.blockLen = copyExtParams.blockLen >> 1;
        uint32_t offset2 = offset / 2;
        DataCopyPad(outGm[offset2], outLocal, copyExtParams);
    } else {
        DataCopyPad(outGm[offset], outLocal, copyExtParams);
    }
    outQueue.FreeTensor(outLocal);
}

}  // namespace DynamicQuantNDOpt2
#endif  // DYNAMIC_QUANT_REGBASE_LARGE_SHAPE_DB_H