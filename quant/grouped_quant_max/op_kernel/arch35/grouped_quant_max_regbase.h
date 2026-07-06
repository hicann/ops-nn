/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file grouped_quant_max_regbase.h
 * @brief GroupedQuantMax Kernel
 *
 */

#ifndef GROUPED_QUANT_MAX_REGBASE_H
#define GROUPED_QUANT_MAX_REGBASE_H
#define FLOAT_OVERFLOW_MODE_CTRL 60

#include "kernel_operator.h"
#include "grouped_quant_max_tiling_data.h"
#include "grouped_quant_max_common.h"
#include "../inc/platform.h"

namespace GroupedQuantMax {
using namespace AscendC;

template <typename T, typename U, uint64_t RoundMode>
class GroupedQuantMaxRegbase : public GroupedQuantMaxBase<T, U, RoundMode> {
public:
    __aicore__ inline GroupedQuantMaxRegbase(const GroupedQuantMaxTilingData* tilingData) : tilingData_(tilingData){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR scale, GM_ADDR groupList, GM_ADDR y, GM_ADDR amax,
                                GM_ADDR workspace);
    __aicore__ inline void Process();
    __aicore__ inline void MergeAmax();

private:
    __aicore__ inline void CopyXAndCompute(int64_t dataCount, int64_t offset, int64_t groupIndex);
    __aicore__ inline void CopyInX(int64_t xLen, int64_t xInOffset);
    __aicore__ inline void Compute(int64_t dataCount, int64_t groupIndex);
    __aicore__ inline void CopyOutY(int64_t yLen, int64_t yOutOffset);

private:
    constexpr static int32_t bufferNum_ = 2;
    constexpr static int32_t Num_1 = 1;
    constexpr static int64_t WS_RESERVE = 1024 * 1024;
    constexpr static int64_t ALIGN_BYTES = 32;
    constexpr static int64_t VF_BYTES = platform::GetVRegSize();
    constexpr static int64_t FLOAT_ALIGN_ELEM = ALIGN_BYTES / sizeof(float); // 8

    TPipe pipe_;
    TQue<QuePosition::VECIN, bufferNum_> inQueueX_;
    TQue<QuePosition::VECOUT, bufferNum_> outQueueY_;
    TBuf<TPosition::VECCALC> scaleBuf_;
    TBuf<TPosition::VECCALC> groupListBuf_;
    TBuf<TPosition::VECCALC> maxBuf_;
    TBuf<TPosition::VECCALC> zeroBuf_;
    TQue<QuePosition::VECIN, Num_1> inQueueGroupMax_;
    TQue<QuePosition::VECOUT, Num_1> outQueueAmax_;

    GlobalTensor<T> xGm_;
    GlobalTensor<float> scaleGm_;
    GlobalTensor<int64_t> groupListGm_;
    GlobalTensor<U> yGm_;
    GlobalTensor<T> amaxGm_;
    GlobalTensor<float> coreGroupMaxGm_;
    LocalTensor<float> scaleLocal;
    LocalTensor<int64_t> groupListLocal;
    LocalTensor<float> maxLocal;
    LocalTensor<float> zeroLocal;

    const GroupedQuantMaxTilingData* tilingData_;
    int32_t coreIdx_ = 0;
    int64_t blockLen_ = 1;
    float scaleValue_ = 0.0f;
    int64_t numGroups_ = 0;
    int64_t dim1_ = 0;
    uint16_t VL_ = 0;
    int64_t totalCoreNum_;
};

template <typename T, typename U, uint64_t RoundMode>
__aicore__ inline void GroupedQuantMaxRegbase<T, U, RoundMode>::Init(GM_ADDR x, GM_ADDR scale, GM_ADDR groupList,
                                                                     GM_ADDR y, GM_ADDR amax, GM_ADDR workspace)
{
    SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
    coreIdx_ = GetBlockIdx();
    numGroups_ = tilingData_->numGroups;
    dim1_ = tilingData_->dim1;
    totalCoreNum_ = tilingData_->totalCoreNum;
    VL_ = VF_BYTES / sizeof(float);

    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    scaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scale));
    groupListGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(groupList));
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(y));
    amaxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(amax));
    coreGroupMaxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace + WS_RESERVE),
                                    numGroups_ * totalCoreNum_);

    pipe_.InitBuffer(inQueueX_, bufferNum_, tilingData_->ubFactor * sizeof(T));
    pipe_.InitBuffer(outQueueY_, bufferNum_, tilingData_->ubFactor * sizeof(U));
    pipe_.InitBuffer(scaleBuf_, numGroups_ * sizeof(float));
    pipe_.InitBuffer(groupListBuf_, numGroups_ * sizeof(int64_t));
    pipe_.InitBuffer(maxBuf_, numGroups_ * sizeof(float));
    pipe_.InitBuffer(zeroBuf_, numGroups_ * sizeof(float));
    pipe_.InitBuffer(
        inQueueGroupMax_, Num_1,
        ((numGroups_ + FLOAT_ALIGN_ELEM - 1) / FLOAT_ALIGN_ELEM * FLOAT_ALIGN_ELEM) * totalCoreNum_ * sizeof(float));
    pipe_.InitBuffer(outQueueAmax_, Num_1,
                     ((numGroups_ + FLOAT_ALIGN_ELEM - 1) / FLOAT_ALIGN_ELEM * FLOAT_ALIGN_ELEM) * sizeof(T));

    scaleLocal = scaleBuf_.Get<float>();
    groupListLocal = groupListBuf_.Get<int64_t>();
    maxLocal = maxBuf_.Get<float>();
    zeroLocal = zeroBuf_.Get<float>();

    DataCopyExtParams scaleCopyParams{1, static_cast<uint32_t>(numGroups_ * sizeof(float)), 0, 0, 0};
    DataCopyPad(scaleLocal, scaleGm_[0], scaleCopyParams, {false, 0, 0, 0});

    DataCopyExtParams groupListCopyParams{1, static_cast<uint32_t>(numGroups_ * sizeof(int64_t)), 0, 0, 0};
    DataCopyPad(groupListLocal, groupListGm_[0], groupListCopyParams, {false, 0, 0, 0});

    Duplicate<float>(maxLocal, 0.0f, numGroups_);

    int32_t eventIDMTE2ToV = GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V);
    SetFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2ToV);
    WaitFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2ToV);

    Duplicate<float>(zeroLocal, 0.0f, numGroups_);

    int32_t eventIDMTE3ToV = GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3);
    SetFlag<AscendC::HardEvent::V_S>(eventIDMTE3ToV);
    WaitFlag<AscendC::HardEvent::V_S>(eventIDMTE3ToV);

    DataCopyExtParams copyParams{1, static_cast<uint32_t>(numGroups_ * sizeof(float)), 0, 0, 0};
    DataCopyPad<float>(coreGroupMaxGm_[coreIdx_], zeroLocal, copyParams);
}

template <typename T, typename U, uint64_t RoundMode>
__aicore__ inline void GroupedQuantMaxRegbase<T, U, RoundMode>::Process()
{
    if (coreIdx_ >= tilingData_->actualCoreNum) {
        return;
    }

    groupListLocal = groupListBuf_.Get<int64_t>();
    scaleLocal = scaleBuf_.Get<float>();

    int64_t coreStart = 0;
    if (coreIdx_ < tilingData_->actualCoreNum - tilingData_->blockTailCoreNum) {
        blockLen_ = tilingData_->blockFactor;
        coreStart = coreIdx_ * blockLen_;
    } else {
        blockLen_ = tilingData_->blockTailFactor;
        coreStart = (tilingData_->actualCoreNum - tilingData_->blockTailCoreNum) * tilingData_->blockFactor +
                    (coreIdx_ - (tilingData_->actualCoreNum - tilingData_->blockTailCoreNum)) * blockLen_;
    }

    int64_t coreEnd = coreStart + blockLen_;
    int64_t firstGroup = -1;
    int64_t lastGroup = -1;

    for (int64_t g = 0; g < numGroups_; ++g) {
        int64_t groupEnd = groupListLocal.GetValue(g) * dim1_;
        if (firstGroup == -1 && groupEnd > coreStart) {
            firstGroup = g;
        }
        if (groupEnd >= coreEnd) {
            lastGroup = g;
            break;
        }
    }

    firstGroup = (firstGroup < 0) ? 0 : firstGroup;
    for (int64_t g = firstGroup; g <= lastGroup; ++g) {
        int64_t groupStart = (g == 0) ? 0 : groupListLocal.GetValue(g - 1) * dim1_;
        int64_t groupEnd = groupListLocal.GetValue(g) * dim1_;
        int64_t segStart = (coreStart > groupStart) ? coreStart : groupStart;
        int64_t segEnd = (coreEnd < groupEnd) ? coreEnd : groupEnd;
        int64_t segLen = segEnd - segStart;

        if (segLen <= 0) {
            continue;
        }

        scaleValue_ = scaleLocal.GetValue(g);
        int64_t ubFactor = tilingData_->ubFactor;
        int64_t loopNum = segLen / ubFactor;
        int64_t loopTail = segLen % ubFactor;

        for (int64_t i = 0; i < loopNum; ++i) {
            CopyXAndCompute(ubFactor, segStart + i * ubFactor, g);
        }
        if (loopTail != 0) {
            CopyXAndCompute(loopTail, segStart + loopNum * ubFactor, g);
        }
    }

    int32_t eventIDVToS = GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S);
    SetFlag<AscendC::HardEvent::V_S>(eventIDVToS);
    WaitFlag<AscendC::HardEvent::V_S>(eventIDVToS);

    DataCopyExtParams copyParams{1, static_cast<uint32_t>(numGroups_ * sizeof(float)), 0, 0, 0};
    DataCopyPad<float>(coreGroupMaxGm_[coreIdx_ * numGroups_], maxLocal, copyParams);
}

template <typename T, typename U, uint64_t RoundMode>
__aicore__ inline void GroupedQuantMaxRegbase<T, U, RoundMode>::CopyXAndCompute(int64_t dataCount, int64_t offset,
                                                                                int64_t groupIndex)
{
    CopyInX(dataCount, offset);
    Compute(dataCount, groupIndex);
    CopyOutY(dataCount, offset);
}

template <typename T, typename U, uint64_t RoundMode>
__aicore__ inline void GroupedQuantMaxRegbase<T, U, RoundMode>::CopyInX(int64_t xLen, int64_t xInOffset)
{
    LocalTensor<T> xLocal = inQueueX_.AllocTensor<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(xLen * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
    DataCopyPad<T>(xLocal, xGm_[xInOffset], copyParams, padParams);
    inQueueX_.EnQue(xLocal);
}

template <typename T, typename U, uint64_t RoundMode>
__aicore__ inline void GroupedQuantMaxRegbase<T, U, RoundMode>::Compute(int64_t dataCount, int64_t groupIndex)
{
    LocalTensor<T> xLocal = inQueueX_.DeQue<T>();
    LocalTensor<U> outLocal = outQueueY_.AllocTensor<U>();

    __local_mem__ T* xLocalAddr = (__local_mem__ T*)xLocal.GetPhyAddr();
    __local_mem__ U* outLocalAddr = (__local_mem__ U*)outLocal.GetPhyAddr();
    __local_mem__ float* maxAddr = (__local_mem__ float*)maxLocal.GetPhyAddr();

    uint16_t vfLoopNum = (dataCount + VL_ - 1) / VL_;

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<T> vregX;
        AscendC::Reg::RegTensor<float> vregFloatAbsT;
        AscendC::Reg::RegTensor<float> vregTileMaxFp32;
        AscendC::Reg::RegTensor<float> vregFloatX;
        AscendC::Reg::RegTensor<float> vregS;
        AscendC::Reg::RegTensor<float> vregFloatY;
        AscendC::Reg::RegTensor<U> vregY;

        AscendC::Reg::MaskReg mask = AscendC::Reg::CreateMask<float>();
        AscendC::Reg::MaskReg maskAll = AscendC::Reg::CreateMask<float>();
        uint32_t countOne = 1;
        AscendC::Reg::MaskReg maskOne = AscendC::Reg::UpdateMask<float>(countOne);

        float currentGroupMax = maxLocal.GetValue(groupIndex);
        AscendC::Reg::Duplicate<float>(vregTileMaxFp32, currentGroupMax, mask);
        AscendC::Reg::Duplicate<float>(vregS, scaleValue_, mask);

        uint32_t count = dataCount;
        for (uint16_t i = 0; i < vfLoopNum; i++) {
            mask = AscendC::Reg::UpdateMask<float>(count);

            if constexpr (IsSameType<T, float>::value) {
                AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(vregFloatX, xLocalAddr + i * VL_);
            } else if constexpr (IsSameType<T, half>::value) {
                AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vregX, xLocalAddr + i * VL_);
                AscendC::Reg::Cast<float, half, GroupedQuantMaxBase<T, U, RoundMode>::CAST_TRAIT_HALF_TO_FP32>(
                    vregFloatX, vregX, mask);
            } else if constexpr (IsSameType<T, bfloat16_t>::value) {
                AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vregX,
                                                                                             xLocalAddr + i * VL_);
                AscendC::Reg::Cast<float, bfloat16_t, GroupedQuantMaxBase<T, U, RoundMode>::CAST_TRAIT_BF16_TO_FP32>(
                    vregFloatX, vregX, mask);
            }

            AscendC::Reg::Abs<float>(vregFloatAbsT, vregFloatX, mask);
            AscendC::Reg::Max<float, AscendC::Reg::MaskMergeMode::MERGING>(vregTileMaxFp32, vregTileMaxFp32,
                                                                           vregFloatAbsT, mask);
            AscendC::Reg::Mul(vregFloatY, vregFloatX, vregS, mask);

            if constexpr (IsSameType<U, hifloat8_t>::value) {
                AscendC::Reg::Cast<U, float, GroupedQuantMaxBase<T, U, RoundMode>::CAST_TRAIT_FP32_TO_HIFP8>(
                    vregY, vregFloatY, mask);
                AscendC::Reg::StoreAlign<U, Reg::StoreDist::DIST_PACK4_B32>(outLocalAddr + i * VL_, vregY, mask);
            } else if constexpr (IsSameType<U, fp8_e5m2_t>::value) {
                AscendC::Reg::Cast<U, float, GroupedQuantMaxBase<T, U, RoundMode>::CAST_TRAIT_FP32_TO_FP8E5M2>(
                    vregY, vregFloatY, mask);
                AscendC::Reg::StoreAlign<U, Reg::StoreDist::DIST_PACK4_B32>(outLocalAddr + i * VL_, vregY, mask);
            } else if constexpr (IsSameType<U, fp8_e4m3fn_t>::value) {
                AscendC::Reg::Cast<U, float, GroupedQuantMaxBase<T, U, RoundMode>::CAST_TRAIT_FP32_TO_FP8E4M3>(
                    vregY, vregFloatY, mask);
                AscendC::Reg::StoreAlign<U, Reg::StoreDist::DIST_PACK4_B32>(outLocalAddr + i * VL_, vregY, mask);
            }
        }

        AscendC::Reg::Reduce<Reg::ReduceType::MAX>(vregTileMaxFp32, vregTileMaxFp32, maskAll);
        AscendC::Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
            ((__local_mem__ float*)maxAddr + groupIndex), vregTileMaxFp32, maskOne);
    }

    inQueueX_.FreeTensor(xLocal);
    outQueueY_.EnQue(outLocal);
}

template <typename T, typename U, uint64_t RoundMode>
__aicore__ inline void GroupedQuantMaxRegbase<T, U, RoundMode>::CopyOutY(int64_t yLen, int64_t yOutOffset)
{
    LocalTensor<U> outLocal = outQueueY_.DeQue<U>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(yLen * sizeof(U)), 0, 0, 0};
    DataCopyPad(yGm_[yOutOffset], outLocal, copyParams);
    outQueueY_.FreeTensor(outLocal);
}

template <typename T, typename U, uint64_t RoundMode>
__aicore__ inline void GroupedQuantMaxRegbase<T, U, RoundMode>::MergeAmax()
{
    if (coreIdx_ >= 1) {
        return;
    }

    LocalTensor<float> groupMaxLocal = inQueueGroupMax_.AllocTensor<float>();
    LocalTensor<T> amaxLocal = outQueueAmax_.AllocTensor<T>();

    uint8_t padNum = (numGroups_ % FLOAT_ALIGN_ELEM == 0) ? 0 : FLOAT_ALIGN_ELEM - (numGroups_ % FLOAT_ALIGN_ELEM);
    DataCopyExtParams copyParams{static_cast<uint16_t>(totalCoreNum_),
                                 static_cast<uint32_t>(numGroups_ * sizeof(float)), 0, 0, 0};
    DataCopyPad(groupMaxLocal, coreGroupMaxGm_, copyParams, {true, 0, padNum, 0});

    inQueueGroupMax_.EnQue(groupMaxLocal);
    groupMaxLocal = inQueueGroupMax_.DeQue<float>();

    __local_mem__ float* groupMaxLocalAddr = (__local_mem__ float*)groupMaxLocal.GetPhyAddr();
    __local_mem__ T* amaxLocalAddr = (__local_mem__ T*)amaxLocal.GetPhyAddr();

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<float> vregFloatMax;
        AscendC::Reg::RegTensor<float> vregFloatTmp;
        AscendC::Reg::RegTensor<T> vregFloatOutAmax;

        AscendC::Reg::MaskReg mask = AscendC::Reg::CreateMask<float>();
        AscendC::Reg::MaskReg maskU8 = AscendC::Reg::CreateMask<uint8_t>();

        uint32_t count = numGroups_;
        uint16_t vfLoopNum = (numGroups_ + totalCoreNum_ - 1) / totalCoreNum_;
        uint32_t numGroupsAlign32_ = (numGroups_ + FLOAT_ALIGN_ELEM - 1) / FLOAT_ALIGN_ELEM * FLOAT_ALIGN_ELEM;
        uint16_t totalCoreNumLoop_ = static_cast<uint16_t>(totalCoreNum_);

        for (uint16_t i = 0; i < vfLoopNum; ++i) {
            mask = AscendC::Reg::UpdateMask<float>(count);
            AscendC::Reg::Duplicate<float>(vregFloatMax, 0.0f, mask);
            for (uint16_t j = 0; j < totalCoreNumLoop_; ++j) {
                AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(
                    vregFloatTmp, groupMaxLocalAddr + i * VL_ + j * numGroupsAlign32_);
                AscendC::Reg::Max<float, AscendC::Reg::MaskMergeMode::ZEROING>(vregFloatMax, vregFloatMax, vregFloatTmp,
                                                                               mask);
            }

            if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
                AscendC::Reg::Cast<T, float, GroupedQuantMaxBase<T, U, RoundMode>::CAST_TRAIT_FP32_TO_X_DETYPE>(
                    vregFloatOutAmax, vregFloatMax, maskU8);
                AscendC::Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(amaxLocalAddr + i * VL_, vregFloatOutAmax,
                                                                           mask);
            } else if constexpr (IsSameType<T, float>::value) {
                AscendC::Reg::StoreAlign<T>(amaxLocalAddr + i * VL_, vregFloatMax, mask);
            }
        }
    }
    outQueueAmax_.EnQue(amaxLocal);
    amaxLocal = outQueueAmax_.DeQue<T>();

    DataCopyExtParams copyParamsOut{1, static_cast<uint32_t>(numGroups_ * sizeof(T)), 0, 0, 0};
    DataCopyPad(amaxGm_, amaxLocal, copyParamsOut);

    inQueueGroupMax_.FreeTensor(groupMaxLocal);
    outQueueAmax_.FreeTensor(amaxLocal);
}

} // namespace GroupedQuantMax
#endif // GROUPED_QUANT_MAX_REGBASE_H
