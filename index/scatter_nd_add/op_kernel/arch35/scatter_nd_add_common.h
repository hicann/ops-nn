/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scatter_nd_add_common.h
 * \brief scatter_nd_add
 */
#ifndef ASCENDC_SCATTER_ND_ADD_COMMON_H_
#define ASCENDC_SCATTER_ND_ADD_COMMON_H_

#include "kernel_operator.h"
#include "../inc/platform.h"
#include "scatter_nd_add_struct.h"
#include "../inc/load_store_utils.h"
#include "./indices_sort_utils.h"

namespace ScatterNdAdd {
using namespace AscendC;

#ifdef __DAV_FPGA__
constexpr uint32_t THREAD_NUM = 128;
constexpr uint32_t THREAD_NUM_LAUNCH_BOUND = 512;
#else
constexpr uint32_t THREAD_NUM = 1024;
constexpr uint32_t THREAD_NUM_LAUNCH_BOUND = 1024;
#endif
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr uint64_t UB_AGLIN_VALUE = 32;
constexpr uint16_t MAX_RANK_COUNT = 7;
constexpr uint64_t SORT_PAD_NUM = 2;
constexpr uint16_t MIN_SAME_IDX_ACCM_COUNT = 256;
constexpr uint16_t MAX_SHAPE_RANK = 8;
constexpr uint16_t INDICE_RANK_TWO = 2;
constexpr float SORT_HIST_THRESHOLD = 0.01f;
constexpr uint32_t HASH_SCORE_BUF_SIZE = 128;
constexpr uint32_t CAST_0 = 0;
constexpr uint32_t CAST_1 = 1;
constexpr uint32_t CAST_2 = 2;
constexpr uint32_t CAST_3 = 3;
constexpr uint32_t CAST_4 = 4;
constexpr uint32_t CAST_5 = 5;
constexpr int64_t VFLEN_INT64 = platform::GetVRegSize() / sizeof(int64_t);
constexpr int64_t VFLEN_INT32 = platform::GetVRegSize() / sizeof(int32_t);
constexpr int64_t VFLEN_INT16 = platform::GetVRegSize() / sizeof(int16_t);
constexpr int64_t VFLEN_INT16HALF = platform::GetVRegSize() / sizeof(int16_t) / TWO;
constexpr int64_t VFLEN_UINT8 = platform::GetVRegSize() / sizeof(uint8_t);
constexpr int64_t VFLEN_UINT8HALFHALF = platform::GetVRegSize() / sizeof(uint8_t) / FOUR;
constexpr uint32_t U8_MAX = 255;

static constexpr SortConfig sortConfig{SortType::RADIX_SORT, false};

__aicore__ inline uint32_t ROUND_UP32(uint32_t x)
{
    if (x % UB_AGLIN_VALUE != 0) {
        return (x / UB_AGLIN_VALUE + 1) * UB_AGLIN_VALUE;
    }
    return x;
}

template <typename T, typename U, typename CAST_T = U, uint32_t castType = CAST_0>
class ScatterNdAddBase
{
public:
    int64_t indicesFactor_ = 0;
    int64_t afterAxisFactor_ = 0;
    int64_t afterAxis_ = 0;
    int64_t indexRankSize_ = 0;
    int64_t eachCoreAfterAxisCount_ = 0;
    int64_t eachCoreIndexCount_ = 0;
    int64_t shiftOffset_ = UB_AGLIN_VALUE / sizeof(CAST_T);
    uint32_t uniqueIdNum_ = 0;
    float maxScore_ = static_cast<float>(0);

    AscendC::GlobalTensor<U> indicesGm_;
    AscendC::GlobalTensor<T> updatesGm_;
    AscendC::GlobalTensor<T> yGm_;

    TBuf<QuePosition::VECCALC> indicesBuf_;
    TBuf<QuePosition::VECCALC> outOfstBuf_;
    TBuf<QuePosition::VECCALC> strideBuf_;
    TBuf<QuePosition::VECCALC> maxScoreBuf_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> dataQueue_;

    TBuf<QuePosition::VECCALC> sortIndicesQue_;
    TBuf<QuePosition::VECCALC> castIndicesQue_;
    TBuf<QuePosition::VECCALC> castTmpIndicesQue_;
    TQue<QuePosition::VECIN, 1> updatesOriginIdexQue_;
    TQue<QuePosition::VECIN, 1> uniqueIdCountQue_;
    TQue<QuePosition::VECOUT, 1> updateSumIdxQue_;
    TQue<QuePosition::VECOUT, 1> updateSumQue_;

    using IndexRegType = typename std::conditional<
        IsSameType<U, int64_t>::value,
        typename AscendC::MicroAPI::RegTensor<uint64_t, AscendC::MicroAPI::RegTraitNumTwo>,
        typename AscendC::MicroAPI::RegTensor<uint32_t>>::type;
    using InnerRegType = typename std::conditional<
        IsSameType<U, int64_t>::value,
        typename AscendC::MicroAPI::RegTensor<int64_t, AscendC::MicroAPI::RegTraitNumTwo>,
        typename AscendC::MicroAPI::RegTensor<int32_t>>::type;

    using selRegType = typename std::conditional<IsSameType<T, bool>::value, int8_t, T>::type;

    __aicore__ inline void InitBaseBuffer(
        TPipe& pipe, uint32_t indicesNumber, GM_ADDR indices, GM_ADDR updates, GM_ADDR y)
    {
        indicesGm_.SetGlobalBuffer((__gm__ U*)(indices));
        updatesGm_.SetGlobalBuffer((__gm__ T*)(updates));
        yGm_.SetGlobalBuffer((__gm__ T*)(y));

        pipe.InitBuffer(strideBuf_, MAX_RANK_COUNT * sizeof(U));
        pipe.InitBuffer(dataQueue_, 1, indicesFactor_ * afterAxisFactor_ * sizeof(T));
        pipe.InitBuffer(outOfstBuf_, indicesFactor_ * sizeof(U));
        pipe.InitBuffer(indicesBuf_, indicesFactor_ * indexRankSize_ * sizeof(U));
        pipe.InitBuffer(maxScoreBuf_, HASH_SCORE_BUF_SIZE * sizeof(float));

        pipe.InitBuffer(updatesOriginIdexQue_, 1, indicesFactor_ * sizeof(uint32_t));
        pipe.InitBuffer(
            uniqueIdCountQue_, 1, ops::CeilAlign((indicesFactor_ + 1) * sizeof(int32_t), UB_AGLIN_VALUE));
        pipe.InitBuffer(
            updateSumIdxQue_, 1, ops::CeilAlign((indicesFactor_ + 1) * sizeof(U), UB_AGLIN_VALUE));
        pipe.InitBuffer(updateSumQue_, 1, indicesFactor_ * afterAxisFactor_ * sizeof(float));

        if constexpr (castType == CAST_0) {
            pipe.InitBuffer(sortIndicesQue_, ops::CeilAlign(indicesFactor_ * sizeof(U) + SORT_PAD_NUM * UB_AGLIN_VALUE, UB_AGLIN_VALUE));
        } else {
            pipe.InitBuffer(sortIndicesQue_, ops::CeilAlign(indicesFactor_ * sizeof(CAST_T) + SORT_PAD_NUM * UB_AGLIN_VALUE, UB_AGLIN_VALUE));
            pipe.InitBuffer(castIndicesQue_, ops::CeilAlign(indicesFactor_ * sizeof(CAST_T), UB_AGLIN_VALUE));
        }
    }

    template <typename PARAM_T>
    __aicore__ inline void CopyIn(
        const LocalTensor<PARAM_T>& dstTensor, const GlobalTensor<PARAM_T>& srcTensor, int64_t dataLen)
    {
        DataCopyExtParams copyParams = {
            static_cast<uint16_t>(1), static_cast<uint32_t>(dataLen * sizeof(PARAM_T)), static_cast<uint32_t>(0),
            static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
        DataCopyPadExtParams<PARAM_T> padParams = {
            false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<PARAM_T>(0)};
        DataCopyPad(dstTensor, srcTensor, copyParams, padParams);
    }

    template <typename PARAM_T>
    __aicore__ inline void CopyOut(
        const GlobalTensor<PARAM_T>& dstTensor, const LocalTensor<PARAM_T>& srcTensor, int64_t dataLen)
    {
        DataCopyExtParams copyParams = {
            static_cast<uint16_t>(1), static_cast<uint32_t>(dataLen * sizeof(PARAM_T)), static_cast<uint32_t>(0),
            static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
        DataCopyPad(dstTensor, srcTensor, copyParams);
    }

    __aicore__ inline void ComputeOutOfset(
        const LocalTensor<U> indicesLocal, const LocalTensor<U> outOfstLocal, int32_t indicesLen, int32_t rankSize)
    {
        LocalTensor<U> strideLocal = strideBuf_.Get<U>();
        __local_mem__ U* indicesLocalPtr = ((__local_mem__ U*)indicesLocal.GetPhyAddr());
        __local_mem__ U* outOfstLocalPtr = ((__local_mem__ U*)outOfstLocal.GetPhyAddr());

        uint32_t dataLen = indicesLen;
        uint32_t vfLen = platform::GetVRegSize() / sizeof(int32_t);
        uint32_t indicesLenTimes = ops::CeilDiv(dataLen, vfLen);
        uint16_t loopCnt = static_cast<uint16_t>(indicesLenTimes);
        uint16_t rankSizeLoops = static_cast<uint16_t>(rankSize);

        __VEC_SCOPE__
        {
            InnerRegType inReg;
            InnerRegType outReg;
            InnerRegType orderReg;
            IndexRegType indexReg;
            AscendC::MicroAPI::MaskReg pregLoop;

            for (uint16_t i = 0; i < loopCnt; i++) {
                if constexpr (IsSameType<U, int64_t>::value) {
                    pregLoop = AscendC::MicroAPI::UpdateMask<U, AscendC::MicroAPI::RegTraitNumTwo>(dataLen);
                } else {
                    pregLoop = AscendC::MicroAPI::UpdateMask<U>(dataLen);
                }
                AscendC::MicroAPI::Duplicate(outReg, 0, pregLoop);
                AscendC::MicroAPI::Arange(orderReg, i * vfLen);
                AscendC::MicroAPI::Muls(orderReg, orderReg, rankSize, pregLoop);
                for (uint16_t dim = 0; dim < rankSizeLoops; dim++) {
                    U strideValue = strideLocal(dim);
                    indexReg = (IndexRegType&)orderReg;
                    AscendC::MicroAPI::DataCopyGather(inReg, indicesLocalPtr, indexReg, pregLoop);
                    AscendC::MicroAPI::Muls(inReg, inReg, strideValue, pregLoop);
                    AscendC::MicroAPI::Add(outReg, inReg, outReg, pregLoop);
                    AscendC::MicroAPI::Adds(orderReg, orderReg, (U)(1), pregLoop);
                }
                auto outOfstAddr = outOfstLocalPtr + i * vfLen;
                AscendC::MicroAPI::DataCopy(outOfstAddr, outReg, pregLoop);
            }
        }
        return;
    }

    __aicore__ inline void  ComputeUniqueIdNumInt64(__local_mem__ CAST_T* indicesAddr, __local_mem__ int32_t* uniqueIdCountsAddr, uint16_t loopCnt, int64_t dataLen)
    {
        uint32_t counter = dataLen + 1;
        AscendC::MicroAPI::RegTensor<int32_t> orderReg, selReg;
        AscendC::MicroAPI::RegTensor<CAST_T> sortedIdxReg, sortedIdxShiftOneReg;
        AscendC::MicroAPI::MaskReg cmpMask, maskReg, maskHalf;
        AscendC::MicroAPI::UnalignReg u0, uOut;
        for (uint16_t i = 0; i < loopCnt; ++i) {
            AscendC::MicroAPI::Arange(orderReg, i * VFLEN_INT64);
            maskReg = AscendC::MicroAPI::UpdateMask<CAST_T>(counter);
            auto startAddr = indicesAddr + i * VFLEN_INT64;
            DataCopy(sortedIdxReg, startAddr);
            AscendC::MicroAPI::DataCopyUnAlignPre(u0, startAddr - 1);
            AscendC::MicroAPI::DataCopyUnAlign<CAST_T>(sortedIdxShiftOneReg, u0, startAddr - 1);
            AscendC::MicroAPI::Compare<CAST_T, CMPMODE::NE>(cmpMask, sortedIdxReg, sortedIdxShiftOneReg, maskReg);
            AscendC::MicroAPI::MaskPack<AscendC::MicroAPI::HighLowPart::LOWEST>(maskHalf, cmpMask);
            AscendC::MicroAPI::GatherMask<int32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(selReg, orderReg, maskHalf);
            AscendC::MicroAPI::DataCopyUnAlign<int32_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                uniqueIdCountsAddr, selReg, uOut);
        }
        AscendC::MicroAPI::DataCopyUnAlignPost(uniqueIdCountsAddr, uOut);
    }

    __aicore__ inline void  ComputeUniqueIdNumInt32(__local_mem__ CAST_T* indicesAddr, __local_mem__ int32_t* uniqueIdCountsAddr, uint16_t loopCnt, int64_t dataLen)
    {
        uint32_t counter = dataLen + 1;
        AscendC::MicroAPI::RegTensor<int32_t> orderReg, selReg;
        AscendC::MicroAPI::RegTensor<CAST_T> sortedIdxReg, sortedIdxShiftOneReg;
        AscendC::MicroAPI::MaskReg cmpMask, maskReg;
        AscendC::MicroAPI::UnalignReg u0, uOut;
        for (uint16_t i = 0; i < loopCnt; ++i) {
            AscendC::MicroAPI::Arange(orderReg, i * VFLEN_INT32);
            maskReg = AscendC::MicroAPI::UpdateMask<CAST_T>(counter);
            auto startAddr = indicesAddr + i * VFLEN_INT32;
            DataCopy(sortedIdxReg, startAddr);
            AscendC::MicroAPI::DataCopyUnAlignPre(u0, startAddr - 1);
            AscendC::MicroAPI::DataCopyUnAlign<CAST_T>(sortedIdxShiftOneReg, u0, startAddr - 1);
            AscendC::MicroAPI::Compare<CAST_T, CMPMODE::NE>(cmpMask, sortedIdxReg, sortedIdxShiftOneReg, maskReg);
            AscendC::MicroAPI::GatherMask<int32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(selReg, orderReg, cmpMask);
            AscendC::MicroAPI::DataCopyUnAlign<int32_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                uniqueIdCountsAddr, selReg, uOut);
        }
        AscendC::MicroAPI::DataCopyUnAlignPost(uniqueIdCountsAddr, uOut);
    }

    __aicore__ inline void  ComputeUniqueIdNumInt16(__local_mem__ CAST_T* indicesAddr, __local_mem__ int32_t* uniqueIdCountsAddr, uint16_t loopCnt, int64_t dataLen)
    {
        uint32_t counter = dataLen + 1;
        AscendC::MicroAPI::RegTensor<int32_t> orderReg, orderReg2, selReg, selReg2;
        AscendC::MicroAPI::RegTensor<CAST_T> sortedIdxReg, sortedIdxShiftOneReg;
        AscendC::MicroAPI::MaskReg cmpMask, maskReg, maskDouble1, maskDouble2;
        AscendC::MicroAPI::UnalignReg u0, uOut;
        for (uint16_t i = 0; i < loopCnt; ++i) {
            AscendC::MicroAPI::Arange(orderReg, i * VFLEN_INT16);
            AscendC::MicroAPI::Arange(orderReg2, i * VFLEN_INT16 + VFLEN_INT16HALF);
            maskReg = AscendC::MicroAPI::UpdateMask<CAST_T>(counter);
            auto startAddr = indicesAddr + i * VFLEN_INT16;
            DataCopy(sortedIdxReg, startAddr);
            AscendC::MicroAPI::DataCopyUnAlignPre(u0, startAddr - 1);
            AscendC::MicroAPI::DataCopyUnAlign<CAST_T>(sortedIdxShiftOneReg, u0, startAddr - 1);
            AscendC::MicroAPI::Compare<CAST_T, CMPMODE::NE>(cmpMask, sortedIdxReg, sortedIdxShiftOneReg, maskReg);
            AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::LOWEST>(maskDouble1, cmpMask);
            AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::HIGHEST>(maskDouble2, cmpMask);
            AscendC::MicroAPI::GatherMask<int32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(selReg, orderReg, maskDouble1);
            AscendC::MicroAPI::GatherMask<int32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(selReg2, orderReg2, maskDouble2);
            AscendC::MicroAPI::DataCopyUnAlign<int32_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(uniqueIdCountsAddr, selReg, uOut);
            AscendC::MicroAPI::DataCopyUnAlign<int32_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(uniqueIdCountsAddr, selReg2, uOut);
        }
        AscendC::MicroAPI::DataCopyUnAlignPost(uniqueIdCountsAddr, uOut);
    }

    __aicore__ inline void  ComputeUniqueIdNumUint8(__local_mem__ CAST_T* indicesAddr, __local_mem__ int32_t* uniqueIdCountsAddr, uint16_t loopCnt, int64_t dataLen)
    {
        uint32_t counter = dataLen + 1;
        AscendC::MicroAPI::RegTensor<int32_t> orderReg, orderReg2, orderReg3, orderReg4;
        AscendC::MicroAPI::RegTensor<int32_t> selReg, selReg2, selReg3, selReg4;
        AscendC::MicroAPI::RegTensor<CAST_T> sortedIdxReg, sortedIdxShiftOneReg;
        AscendC::MicroAPI::MaskReg cmpMask, maskReg, maskFour1, maskFour2, maskFour3, maskFour4;
        AscendC::MicroAPI::UnalignReg u0, uOut;
        for (uint16_t i = 0; i < loopCnt; ++i) {
            AscendC::MicroAPI::Arange(orderReg, i * VFLEN_UINT8);
            AscendC::MicroAPI::Arange(orderReg2, i * VFLEN_UINT8 + VFLEN_UINT8HALFHALF);
            AscendC::MicroAPI::Arange(orderReg3, i * VFLEN_UINT8 + VFLEN_UINT8HALFHALF * TWO);
            AscendC::MicroAPI::Arange(orderReg4, i * VFLEN_UINT8 + VFLEN_UINT8HALFHALF * THREE);
            maskReg = AscendC::MicroAPI::UpdateMask<CAST_T>(counter);
            auto startAddr = indicesAddr + i * VFLEN_UINT8;
            DataCopy(sortedIdxReg, startAddr);
            AscendC::MicroAPI::DataCopyUnAlignPre(u0, startAddr - 1);
            AscendC::MicroAPI::DataCopyUnAlign<CAST_T>(sortedIdxShiftOneReg, u0, startAddr - 1);
            AscendC::MicroAPI::Compare<CAST_T, CMPMODE::NE>(cmpMask, sortedIdxReg, sortedIdxShiftOneReg, maskReg);
            AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::LOWEST>(maskFour3, cmpMask);
            AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::HIGHEST>(maskFour4, cmpMask);
            AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::LOWEST>(maskFour1, maskFour3);
            AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::HIGHEST>(maskFour2, maskFour3);
            AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::LOWEST>(maskFour3, maskFour4);
            AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::HIGHEST>(maskFour4, maskFour4);
            AscendC::MicroAPI::GatherMask<int32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(selReg, orderReg, maskFour1);
            AscendC::MicroAPI::GatherMask<int32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(selReg2, orderReg2, maskFour2);
            AscendC::MicroAPI::GatherMask<int32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(selReg3, orderReg3, maskFour3);
            AscendC::MicroAPI::GatherMask<int32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(selReg4, orderReg4, maskFour4);
            AscendC::MicroAPI::DataCopyUnAlign<int32_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(uniqueIdCountsAddr, selReg, uOut);
            AscendC::MicroAPI::DataCopyUnAlign<int32_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(uniqueIdCountsAddr, selReg2, uOut);
            AscendC::MicroAPI::DataCopyUnAlign<int32_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(uniqueIdCountsAddr, selReg3, uOut);
            AscendC::MicroAPI::DataCopyUnAlign<int32_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(uniqueIdCountsAddr, selReg4, uOut);
        }
        AscendC::MicroAPI::DataCopyUnAlignPost(uniqueIdCountsAddr, uOut);
    }

    __aicore__ inline uint32_t
    ComputeUniqueIdNum(LocalTensor<CAST_T> indicesLocal, LocalTensor<int32_t> uniqueIdCountLocal, int64_t dataLen)
    {
        __local_mem__ CAST_T* indicesAddr = (__local_mem__ CAST_T*)indicesLocal[(UB_AGLIN_VALUE / sizeof(CAST_T))].GetPhyAddr();
        __local_mem__ int32_t* uniqueIdCountsAddr = (__local_mem__ int32_t*)uniqueIdCountLocal.GetPhyAddr();

        int64_t vfLen = platform::GetVRegSize() / sizeof(CAST_T);
        uint16_t loopCnt = ops::CeilDiv(dataLen + 1, vfLen);
        uint32_t counter = dataLen + 1;
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::ClearSpr<AscendC::SpecialPurposeReg::AR>();
            if constexpr (std::is_same<int64_t, CAST_T>::value) {
                ComputeUniqueIdNumInt64(indicesAddr, uniqueIdCountsAddr, loopCnt, dataLen);
            } else if constexpr (std::is_same<int32_t, CAST_T>::value) {
                ComputeUniqueIdNumInt32(indicesAddr, uniqueIdCountsAddr, loopCnt, dataLen);
            } else if constexpr (std::is_same<int16_t, CAST_T>::value) {
                ComputeUniqueIdNumInt16(indicesAddr, uniqueIdCountsAddr, loopCnt, dataLen);
            } else {
                ComputeUniqueIdNumUint8(indicesAddr, uniqueIdCountsAddr, loopCnt, dataLen);
            }
        }
        uint32_t uniqueIdNum = ((AscendC::MicroAPI::GetSpr<AscendC::SpecialPurposeReg::AR>()) / sizeof(int32_t)) - 1;
        return uniqueIdNum;
    }

    __aicore__ inline uint32_t
    ComputeUniqueIdNumDeter(LocalTensor<U> indicesLocal, LocalTensor<int32_t> uniqueIdCountLocal, int64_t dataLen)
    {
        __local_mem__ U* indicesAddr = (__local_mem__ U*)indicesLocal[shiftOffset_].GetPhyAddr();
        __local_mem__ int32_t* uniqueIdCountsAddr = (__local_mem__ int32_t*)uniqueIdCountLocal.GetPhyAddr();

        int64_t vfLen = platform::GetVRegSize() / sizeof(U);
        uint16_t loopCnt = ops::CeilDiv(dataLen + 1, vfLen);
        uint32_t counter = dataLen + 1;
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<int32_t> orderReg;
            AscendC::MicroAPI::RegTensor<U> sortedIdxReg;
            AscendC::MicroAPI::RegTensor<U> sortedIdxShiftOneReg;
            AscendC::MicroAPI::RegTensor<int32_t> selReg;
            AscendC::MicroAPI::MaskReg cmpMask;
            AscendC::MicroAPI::MaskReg maskReg;
            AscendC::MicroAPI::UnalignReg u0;
            AscendC::MicroAPI::UnalignReg uOut;
            AscendC::MicroAPI::ClearSpr<AscendC::SpecialPurposeReg::AR>();

            for (uint16_t i = 0; i < loopCnt; ++i) {
                AscendC::MicroAPI::Arange(orderReg, i * vfLen);
                maskReg = AscendC::MicroAPI::UpdateMask<U>(counter);
                auto startAddr = indicesAddr + i * vfLen;
                DataCopy(sortedIdxReg, startAddr);
                AscendC::MicroAPI::DataCopyUnAlignPre(u0, startAddr - 1);
                AscendC::MicroAPI::DataCopyUnAlign<U>(sortedIdxShiftOneReg, u0, startAddr - 1);
                AscendC::MicroAPI::Compare<U, CMPMODE::NE>(cmpMask, sortedIdxReg, sortedIdxShiftOneReg, maskReg);
                if constexpr (std::is_same<int64_t, U>::value) {
                    AscendC::MicroAPI::MaskReg maskHalf;
                    AscendC::MicroAPI::MaskPack<AscendC::MicroAPI::HighLowPart::LOWEST>(maskHalf, cmpMask);
                    AscendC::MicroAPI::GatherMask<int32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(
                        selReg, orderReg, maskHalf);
                } else {
                    AscendC::MicroAPI::GatherMask<int32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(
                        selReg, orderReg, cmpMask);
                }
                AscendC::MicroAPI::DataCopyUnAlign<int32_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    uniqueIdCountsAddr, selReg, uOut);
            }
            AscendC::MicroAPI::DataCopyUnAlignPost(uniqueIdCountsAddr, uOut);
        }
        uint32_t uniqueIdNum = ((AscendC::MicroAPI::GetSpr<AscendC::SpecialPurposeReg::AR>()) / sizeof(int32_t)) - 1;
        return uniqueIdNum;
    }

    __aicore__ inline void  ComputeUinqueIdTimes(LocalTensor<int32_t> uniqueIdCountLocal, uint32_t uniqueIdNum)
    {
        __local_mem__ int32_t* uniqueIdCountsAddr = (__local_mem__ int32_t*)uniqueIdCountLocal.GetPhyAddr();
        uint32_t vfLen = platform::GetVRegSize() / sizeof(int32_t);
        uint16_t loopSize = ops::CeilDiv(uniqueIdNum, vfLen);
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<int32_t> preReg;
            AscendC::MicroAPI::RegTensor<int32_t> postReg;
            AscendC::MicroAPI::RegTensor<int32_t> subReg;
            AscendC::MicroAPI::UnalignReg uIn;
            AscendC::MicroAPI::MaskReg maskReg;
            for (uint16_t i = 0; i < loopSize; ++i) {
                maskReg = AscendC::MicroAPI::UpdateMask<int32_t>(uniqueIdNum);
                auto startAddr = uniqueIdCountsAddr + i * vfLen;
                auto startAddrOfstOne = startAddr + 1;
                DataCopy(preReg, startAddr);
                AscendC::MicroAPI::DataCopyUnAlignPre(uIn, startAddrOfstOne);
                AscendC::MicroAPI::DataCopyUnAlign<int32_t>(postReg, uIn, startAddrOfstOne, vfLen);
                AscendC::MicroAPI::Sub(subReg, postReg, preReg, maskReg);
                DataCopy(startAddr, subReg, maskReg);
            }
        }
    }

    __aicore__ inline void IndicesSortCast(LocalTensor<U> indicesLocal, LocalTensor<CAST_T> indicesCastLocal,
                                            LocalTensor<int32_t> indicesCastTmpLocal, uint32_t indicesCount)
    {
        if constexpr (castType == CAST_4) {  // int32 Cast uint8
            CompareScalar(indicesCastLocal, indicesLocal, static_cast<U>(0), CMPMODE::GE, indicesCount);
            Select(indicesLocal, indicesCastLocal, indicesLocal, static_cast<U>(U8_MAX), SELMODE::VSEL_TENSOR_SCALAR_MODE, indicesCount);
            Cast<CAST_T, U>(indicesCastLocal, indicesLocal, RoundMode::CAST_NONE, indicesCount);
        } else if constexpr (castType == CAST_3) {  // int64 Cast int16
            Cast<int32_t, U>(indicesCastTmpLocal, indicesLocal, RoundMode::CAST_NONE, indicesCount);
            Cast<CAST_T, int32_t>(indicesCastLocal, indicesCastTmpLocal, RoundMode::CAST_NONE, indicesCount);
        } else if constexpr (castType == CAST_5) {  // int64 Cast uint8
            CompareScalar(indicesCastLocal, indicesLocal, static_cast<U>(0), CMPMODE::GE, indicesCount);
            Select(indicesLocal, indicesCastLocal, indicesLocal, static_cast<U>(U8_MAX), SELMODE::VSEL_TENSOR_SCALAR_MODE, indicesCount);
            Cast<int32_t, U>(indicesCastTmpLocal, indicesLocal, RoundMode::CAST_NONE, indicesCount);
            Cast<CAST_T, int32_t>(indicesCastLocal, indicesCastTmpLocal, RoundMode::CAST_NONE, indicesCount);
        } else {    // CAST_1 + CAST_2, int32 Cast int16 + int64 Cast int32
            Cast<CAST_T, U>(indicesCastLocal, indicesLocal, RoundMode::CAST_NONE, indicesCount);
        }
    }

    __aicore__ inline void  ComputeSumWithCast(
        LocalTensor<int32_t> uniqueIdCountLocal, LocalTensor<uint32_t> updatesOriginIdexLocal,
        LocalTensor<T> updatesLocal, LocalTensor<T> updateSumLocal, uint32_t uniqueIdNum, int64_t colLen)
    {
        __local_mem__ T* updatesAddr = (__local_mem__ T*)updatesLocal.GetPhyAddr();
        __local_mem__ float* updateSumAddr = (__local_mem__ float*)updateSumLocal.GetPhyAddr();

        uint32_t vfLen = platform::GetVRegSize() / sizeof(float);
        int32_t loopSize = ops::CeilDiv(static_cast<uint32_t>(colLen), vfLen);
        int32_t idLocation = 0;
        int64_t colLenAlignSize = ops::CeilAlign(colLen * sizeof(T), UB_AGLIN_VALUE) / sizeof(T);
        int64_t colLenAlignFp32 = ops::CeilAlign(colLen * sizeof(float), UB_AGLIN_VALUE) / sizeof(float);

        __VEC_SCOPE__
        {
            for (uint16_t i = 0; i < static_cast<uint16_t>(uniqueIdNum); i++) {
                AscendC::MicroAPI::RegTensor<float> sumReg;
                AscendC::MicroAPI::RegTensor<float> castReg;
                AscendC::MicroAPI::MaskReg maskReg;
                AscendC::MicroAPI::MaskReg zeroMask = AscendC::MicroAPI::CreateMask<float>();
                uint32_t maskLen = static_cast<uint32_t>(colLen);
                uint16_t idRepeatTimes = static_cast<uint16_t>(uniqueIdCountLocal(i));
                for (uint16_t j = 0; j < static_cast<uint16_t>(loopSize); j++) {
                    maskReg = AscendC::MicroAPI::UpdateMask<float>(maskLen);
                    AscendC::MicroAPI::Duplicate(sumReg, (float)0, zeroMask);
                    for (uint16_t k = 0; k < idRepeatTimes; k++) {
                        auto updatesOffet = updatesOriginIdexLocal(idLocation + k) * colLenAlignSize + j * vfLen;
                        ops::LoadOneTensorForDtypeT<T>(updatesAddr, castReg, maskReg, updatesOffet);
                        AscendC::MicroAPI::Add(sumReg, sumReg, castReg, maskReg);
                    }
                    auto updateSumAddrOfst = i * colLenAlignFp32 + j * vfLen;
                    ops::StoreOneTensorForDtypeT<float>(updateSumAddr, sumReg, maskReg, updateSumAddrOfst);
                }
                idLocation += idRepeatTimes;
            }
        }
    }

    __aicore__ inline void  ComputeSumWithOutCast(
        LocalTensor<int32_t> uniqueIdCountLocal, LocalTensor<uint32_t> updatesOriginIdexLocal,
        LocalTensor<T> updatesLocal, LocalTensor<T> updateSumLocal, uint32_t uniqueIdNum, int64_t colLen)
    {
        __local_mem__ selRegType* updatesAddr = (__local_mem__ selRegType*)updatesLocal.GetPhyAddr();
        __local_mem__ selRegType* updateSumAddr = (__local_mem__ selRegType*)updateSumLocal.GetPhyAddr();

        uint32_t vfLen = platform::GetVRegSize() / sizeof(selRegType);
        int32_t loopSize = (colLen + vfLen - 1) / vfLen;
        int32_t idLocation = 0;
        int64_t colLenAlignSize = ops::CeilAlign(colLen * sizeof(selRegType), UB_AGLIN_VALUE) / sizeof(selRegType);
        __VEC_SCOPE__
        {
            for (uint16_t i = 0; i < static_cast<uint16_t>(uniqueIdNum); i++) {
                AscendC::MicroAPI::RegTensor<selRegType> sumReg;
                AscendC::MicroAPI::RegTensor<selRegType> updateReg;
                AscendC::MicroAPI::MaskReg maskReg;
                AscendC::MicroAPI::MaskReg zeroMask = AscendC::MicroAPI::CreateMask<selRegType>();
                uint32_t maskLen = static_cast<uint32_t>(colLen);
                uint16_t idRepeatTimes = static_cast<uint16_t>(uniqueIdCountLocal(i));
                for (uint16_t j = 0; j < static_cast<uint16_t>(loopSize); j++) {
                    maskReg = AscendC::MicroAPI::UpdateMask<selRegType>(maskLen);
                    AscendC::MicroAPI::Duplicate(sumReg, (selRegType)0, zeroMask);
                    for (uint16_t k = 0; k < idRepeatTimes; k++) {
                        auto updatesOffet = updatesOriginIdexLocal(idLocation + k) * colLenAlignSize + j * vfLen;
                        auto startAddr = updatesAddr + updatesOffet;
                        AscendC::MicroAPI::DataCopy(updateReg, startAddr);
                        AscendC::MicroAPI::Add(sumReg, sumReg, updateReg, maskReg);
                    }
                    auto updateSumAddrOfst = updateSumAddr + i * colLenAlignSize + j * vfLen;
                    AscendC::MicroAPI::DataCopy(updateSumAddrOfst, sumReg, maskReg);
                }
                idLocation += idRepeatTimes;
            }
        }
    }

    __aicore__ inline void  ComputeIdxCount(
        LocalTensor<U> outOfstLocal, int64_t rowLen, int64_t colLen)
    {
        LocalTensor<CAST_T> sortIndicesLocal = sortIndicesQue_.Get<CAST_T>();
        LocalTensor<int32_t> uniqueIdCountLocal = uniqueIdCountQue_.AllocTensor<int32_t>();
        LocalTensor<uint32_t> updatesOriginIdexLocal = updatesOriginIdexQue_.AllocTensor<uint32_t>();
        LocalTensor<CAST_T> shiftSortLocal = sortIndicesLocal[shiftOffset_];
        if constexpr (castType == CAST_0) {
            AscendC::Sort<U, false, sortConfig>(
                shiftSortLocal, updatesOriginIdexLocal, outOfstLocal, static_cast<uint32_t>(rowLen));
            Duplicate(sortIndicesLocal, (U)-1, shiftOffset_);
        } else {
            LocalTensor<CAST_T> indicesCastLocal = castIndicesQue_.Get<CAST_T>();
            IndicesSortCast(outOfstLocal, indicesCastLocal, uniqueIdCountLocal, rowLen);
            AscendC::Sort<CAST_T, false, sortConfig>(
                shiftSortLocal, updatesOriginIdexLocal, indicesCastLocal, static_cast<uint32_t>(rowLen));
            Duplicate(sortIndicesLocal, (CAST_T)-1, shiftOffset_);
        }
        event_t eventIdV2S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdV2S);
        WaitFlag<HardEvent::V_S>(eventIdV2S);

        shiftSortLocal(rowLen) = -1;
        PipeBarrier<PIPE_V>();

        uniqueIdNum_ = ComputeUniqueIdNum(sortIndicesLocal, uniqueIdCountLocal, rowLen);
        uniqueIdCountQue_.FreeTensor(uniqueIdCountLocal);
        updatesOriginIdexQue_.FreeTensor(updatesOriginIdexLocal);
    }

    __aicore__ inline void  ComputeSumForUniqueIdx(
        LocalTensor<U> outOfstLocal, LocalTensor<T> updatesLocal, int64_t rowLen, int64_t colLen)
    {
        LocalTensor<CAST_T> sortIndicesLocal = sortIndicesQue_.Get<CAST_T>();
        LocalTensor<int32_t> uniqueIdCountLocal = uniqueIdCountQue_.AllocTensor<int32_t>();
        LocalTensor<uint32_t> updatesOriginIdexLocal = updatesOriginIdexQue_.AllocTensor<uint32_t>();
        LocalTensor<CAST_T> shiftSortLocal = sortIndicesLocal[shiftOffset_];
        if constexpr (castType == CAST_0) {
            AscendC::Sort<U, false, sortConfig>(
                shiftSortLocal, updatesOriginIdexLocal, outOfstLocal, static_cast<uint32_t>(rowLen));
            Duplicate(sortIndicesLocal, (U)-1, shiftOffset_);
        } else {
            LocalTensor<CAST_T> indicesCastLocal = castIndicesQue_.Get<CAST_T>();
            IndicesSortCast(outOfstLocal, indicesCastLocal, uniqueIdCountLocal, rowLen);
            AscendC::Sort<CAST_T, false, sortConfig>(
                shiftSortLocal, updatesOriginIdexLocal, indicesCastLocal, static_cast<uint32_t>(rowLen));
            Duplicate(sortIndicesLocal, (CAST_T)-1, shiftOffset_);
        }
        event_t eventIdV2S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdV2S);
        WaitFlag<HardEvent::V_S>(eventIdV2S);

        shiftSortLocal(rowLen) = -1;
        PipeBarrier<PIPE_V>();

        LocalTensor<U> updateSumIdxLocal = updateSumIdxQue_.AllocTensor<U>();
        LocalTensor<T> updateSumLocal = updateSumQue_.AllocTensor<T>();
        uniqueIdNum_ = ComputeUniqueIdNum(sortIndicesLocal, uniqueIdCountLocal, rowLen);
        event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);

        for (uint32_t idx = 0; idx < uniqueIdNum_; idx++) {
            auto offset = uniqueIdCountLocal(idx);
            updateSumIdxLocal(idx) = static_cast<U>(shiftSortLocal(offset));
        }
        event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventIdSToV);
        WaitFlag<HardEvent::S_V>(eventIdSToV);
        ComputeUinqueIdTimes(uniqueIdCountLocal, uniqueIdNum_);
        PipeBarrier<PIPE_V>();

        if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
            ComputeSumWithCast(
                uniqueIdCountLocal, updatesOriginIdexLocal, updatesLocal, updateSumLocal, uniqueIdNum_, colLen);
        } else {
            ComputeSumWithOutCast(
                uniqueIdCountLocal, updatesOriginIdexLocal, updatesLocal, updateSumLocal, uniqueIdNum_, colLen);
        }

        updateSumIdxQue_.EnQue(updateSumIdxLocal);
        updateSumQue_.EnQue(updateSumLocal);
        uniqueIdCountQue_.FreeTensor(uniqueIdCountLocal);
        updatesOriginIdexQue_.FreeTensor(updatesOriginIdexLocal);
    }

    __aicore__ inline void  ComputeUpdateSum(int64_t rowLen, int64_t colLen)
    {
        LocalTensor<int32_t> uniqueIdCountLocal = uniqueIdCountQue_.DeQue<int32_t>();
        LocalTensor<uint32_t> updatesOriginIdexLocal = updatesOriginIdexQue_.DeQue<uint32_t>();
        LocalTensor<T> updatesLocal = dataQueue_.DeQue<T>();

        LocalTensor<T> updateSumLocal = updateSumQue_.AllocTensor<T>();

        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

        if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
            ComputeSumWithCast(
                uniqueIdCountLocal, updatesOriginIdexLocal, updatesLocal, updateSumLocal, uniqueIdNum_, colLen);
        } else {
            ComputeSumWithOutCast(
                uniqueIdCountLocal, updatesOriginIdexLocal, updatesLocal, updateSumLocal, uniqueIdNum_, colLen);
        }

        updatesOriginIdexQue_.EnQue(updatesOriginIdexLocal);
        uniqueIdCountQue_.EnQue(uniqueIdCountLocal);

        updateSumQue_.EnQue(updateSumLocal);
        dataQueue_.EnQue(updatesLocal);
    }

    __aicore__ inline void  SortIndices(
        LocalTensor<U> outOfstLocal, int64_t rowLen)
    {
        LocalTensor<CAST_T> sortIndicesLocal = sortIndicesQue_.Get<CAST_T>();
        LocalTensor<int32_t> uniqueIdCountLocal = uniqueIdCountQue_.AllocTensor<int32_t>();
        LocalTensor<uint32_t> updatesOriginIdexLocal = updatesOriginIdexQue_.AllocTensor<uint32_t>();
        LocalTensor<CAST_T> shiftSortLocal = sortIndicesLocal[(UB_AGLIN_VALUE / sizeof(CAST_T))];
        if constexpr (castType == CAST_0) {
            AscendC::Sort<U, false, sortConfig>(
                shiftSortLocal, updatesOriginIdexLocal, outOfstLocal, static_cast<uint32_t>(rowLen));
            Duplicate(sortIndicesLocal, (U)-1, shiftOffset_);
        } else {
            LocalTensor<CAST_T> indicesCastLocal = castIndicesQue_.Get<CAST_T>();
            IndicesSortCast(outOfstLocal, indicesCastLocal, uniqueIdCountLocal, rowLen);
            AscendC::Sort<CAST_T, false, sortConfig>(
                shiftSortLocal, updatesOriginIdexLocal, indicesCastLocal, static_cast<uint32_t>(rowLen));
            Duplicate(sortIndicesLocal, (CAST_T)-1, shiftOffset_);
        }

        event_t eventIdV2S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdV2S);
        WaitFlag<HardEvent::V_S>(eventIdV2S);

        shiftSortLocal(rowLen) = -1;
        PipeBarrier<PIPE_V>();

        LocalTensor<U> updateSumIdxLocal = updateSumIdxQue_.AllocTensor<U>();
        
        uniqueIdNum_ = ComputeUniqueIdNum(sortIndicesLocal, uniqueIdCountLocal, rowLen);
        event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);

        for (uint32_t idx = 0; idx < uniqueIdNum_; idx++) {
            auto offset = uniqueIdCountLocal(idx);
            updateSumIdxLocal(idx) = static_cast<U>(shiftSortLocal(offset));
        }
        event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventIdSToV);
        WaitFlag<HardEvent::S_V>(eventIdSToV);
        ComputeUinqueIdTimes(uniqueIdCountLocal, uniqueIdNum_);
        PipeBarrier<PIPE_V>();

        updatesOriginIdexQue_.EnQue(updatesOriginIdexLocal);
        uniqueIdCountQue_.EnQue(uniqueIdCountLocal);

        updateSumIdxQue_.EnQue(updateSumIdxLocal);
    }

    __aicore__ inline void  ExecConvertAlign(
        LocalTensor<T> updatesLocal, LocalTensor<float> updateSumLocal, int64_t rowLen, int64_t colLen)
    {
        __local_mem__ T* updatesAddr = (__local_mem__ T*)updatesLocal.GetPhyAddr();
        __local_mem__ float* updateSumAddr = (__local_mem__ float*)updateSumLocal.GetPhyAddr();

        uint32_t vfLen = platform::GetVRegSize() / sizeof(float);
        uint16_t loopSize = ops::CeilDiv(static_cast<uint32_t>(colLen), vfLen);
        int64_t colLenAlignSize = ops::CeilAlign(colLen * sizeof(T), UB_AGLIN_VALUE) / sizeof(T);
        int64_t colLenAlignFp32 = ops::CeilAlign(colLen * sizeof(float), UB_AGLIN_VALUE) / sizeof(float);

        __VEC_SCOPE__
        {
            for (uint16_t i = 0; i < static_cast<uint16_t>(rowLen); i++) {
                AscendC::MicroAPI::RegTensor<float> castReg;
                AscendC::MicroAPI::MaskReg maskReg;
                uint32_t maskLen = static_cast<uint32_t>(colLen);
                for (uint16_t j = 0; j < static_cast<uint16_t>(loopSize); j++) {
                    maskReg = AscendC::MicroAPI::UpdateMask<float>(maskLen);
                    auto updateSumAddrOfst = i * colLenAlignFp32 + j * vfLen;
                    auto updateAddrOfet = i * colLenAlignSize + j * vfLen;
                    ops::LoadOneTensorForDtypeT<float>(updateSumAddr, castReg, maskReg, updateSumAddrOfst);
                    ops::StoreOneTensorForDtypeT<T>(updatesAddr, castReg, maskReg, updateAddrOfet);
                }
            }
        }
    }

    __aicore__ inline void CopyOutSplitIndices(
        LocalTensor<U> ofstLocal, LocalTensor<T> dataLocal, int64_t rowLen, int64_t colLen, int64_t colIdx)
    {
        int64_t colLenAlignSize = ops::CeilAlign(colLen * sizeof(T), UB_AGLIN_VALUE) / sizeof(T);
        for (int64_t i = 0; i < rowLen; i++) {
            int64_t rowOfset = ofstLocal(i) * afterAxis_;
            int64_t outOfset = rowOfset + colIdx * afterAxisFactor_;
            if constexpr (IsSameType<T, bool>::value) {
                SetAtomicMax<int8_t>();
                CopyOut<bool>(yGm_[outOfset], dataLocal[i * colLenAlignSize], colLen);
                SetAtomicNone();
            } else {
                SetAtomicAdd<T>();
                CopyOut<T>(yGm_[outOfset], dataLocal[i * colLenAlignSize], colLen);
                SetAtomicNone();
            }
        }
    }

    __aicore__ inline void CopyIndiceInSplitIndices(int64_t rowIdx, int64_t rowLen)
    {
        LocalTensor<U> indicesLocal = indicesBuf_.Get<U>();
        LocalTensor<U> outOfstLocal = outOfstBuf_.Get<U>();
        LocalTensor<float> dstLocal = maxScoreBuf_.Get<float>();

        int64_t rankSize = indexRankSize_;
        int64_t indicesOfset = GetBlockIdx() * eachCoreIndexCount_ + rowIdx * indicesFactor_;
        this->template CopyIn<U>(indicesLocal, indicesGm_[indicesOfset * rankSize], rowLen * rankSize);
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        this->ComputeOutOfset(indicesLocal, outOfstLocal, rowLen, rankSize);
        if constexpr (IsSameType<U, int32_t>::value) {
            IndexStatisticInt32(outOfstLocal, dstLocal, this->maxScore_, rowLen, afterAxis_);
        } else {
            IndexStatisticInt64(outOfstLocal, dstLocal, this->maxScore_, rowLen, afterAxis_);
        }
    }

    __aicore__ inline void CopyUpdatesInSplitIndices(
        int64_t rowIdx, int64_t colIdx, int64_t rowLen, int64_t colLen)
    {
        LocalTensor<T> updatesLocal = dataQueue_.AllocTensor<T>();
        int64_t indicesOfset = GetBlockIdx() * eachCoreIndexCount_ + rowIdx * indicesFactor_;
        DataCopyExtParams copyParams = {
            static_cast<uint16_t>(rowLen), static_cast<uint32_t>(colLen * sizeof(T)),
            static_cast<uint32_t>((afterAxis_ - colLen) * sizeof(T)), static_cast<uint32_t>(0),
            static_cast<uint32_t>(0)};
        DataCopyPadExtParams<T> updatePadParams = {false, 0, 0, 0};
        int64_t rowOfset = indicesOfset * afterAxis_;
        int64_t updatesOfset = rowOfset + colIdx * afterAxisFactor_;
        DataCopyPad(updatesLocal, updatesGm_[updatesOfset], copyParams, updatePadParams);
        dataQueue_.EnQue(updatesLocal);
    }

    __aicore__ inline void ComputeOutSplitAfter(int64_t colIdx, int64_t rowLen, int64_t colLen)
    {
        LocalTensor<T> updatesLocal = dataQueue_.DeQue<T>();
        LocalTensor<U> updateSumIdxLocal = updateSumIdxQue_.DeQue<U>();
        if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
            LocalTensor<float> updateSumLocal = updateSumQue_.DeQue<float>();
            ExecConvertAlign(updatesLocal, updateSumLocal, rowLen, colLen);
            event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            CopyOutSplitAfter(updateSumIdxLocal, updatesLocal, uniqueIdNum_, colLen, colIdx);
            updateSumQue_.FreeTensor(updateSumLocal);
        } else {
            LocalTensor<T> updateSumLocal = updateSumQue_.DeQue<T>();
            CopyOutSplitAfter(updateSumIdxLocal, updateSumLocal, uniqueIdNum_, colLen, colIdx);
            updateSumQue_.FreeTensor(updateSumLocal);
        }

        dataQueue_.FreeTensor(updatesLocal);
        updateSumIdxQue_.EnQue(updateSumIdxLocal);
    }

    __aicore__ inline void ComputeOutSplitIndices(int64_t colIdx, int64_t rowLen, int64_t colLen)
    {
        LocalTensor<U> updateSumIdxLocal = updateSumIdxQue_.DeQue<U>();
        LocalTensor<T> updatesLocal = dataQueue_.DeQue<T>();

        if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
            LocalTensor<float> updateSumLocal = updateSumQue_.DeQue<float>();
            ExecConvertAlign(updatesLocal, updateSumLocal, rowLen, colLen);
            event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

            CopyOutSplitIndices(updateSumIdxLocal, updatesLocal, uniqueIdNum_, colLen, colIdx);
            updateSumQue_.FreeTensor(updateSumLocal);
        } else {
            LocalTensor<T> updateSumLocal = updateSumQue_.DeQue<T>();
            CopyOutSplitIndices(updateSumIdxLocal, updateSumLocal, uniqueIdNum_, colLen, colIdx);
            updateSumQue_.FreeTensor(updateSumLocal);
        }

        dataQueue_.FreeTensor(updatesLocal);
        updateSumIdxQue_.EnQue(updateSumIdxLocal);
    }

    __aicore__ inline void CopyOutSplitAfter(
        LocalTensor<U> ofstLocal, LocalTensor<T> dataLocal, int64_t rowLen, int64_t colLen, int64_t colIdx)
    {
        int64_t colLenAlignSize = ops::CeilAlign(colLen * sizeof(T), UB_AGLIN_VALUE) / sizeof(T);
        for (int64_t i = 0; i < rowLen; i++) {
            int64_t rowOfset = ofstLocal(i) * afterAxis_;
            int64_t outOfset = rowOfset + GetBlockIdx() * eachCoreAfterAxisCount_ + colIdx * afterAxisFactor_;
            if constexpr (IsSameType<T, bool>::value) {
                SetAtomicMax<int8_t>();
                CopyOut<bool>(yGm_[outOfset], dataLocal[i * colLenAlignSize], colLen);
                SetAtomicNone();
            } else {
                SetAtomicAdd<T>();
                CopyOut<T>(yGm_[outOfset], dataLocal[i * colLenAlignSize], colLen);
                SetAtomicNone();
            }
        }
    }

    __aicore__ inline void CopyOutSplitAfterDuplicateIdx(
        LocalTensor<U> ofstLocal, LocalTensor<T> dataLocal, int64_t rowLen, int64_t colLen, int64_t colIdx)
    {
        int64_t colLenAlignSize = ops::CeilAlign(colLen * sizeof(T), UB_AGLIN_VALUE) / sizeof(T);
        for (int64_t i = 0; i < rowLen; i++) {
            int64_t rowOfset = ofstLocal(i) * afterAxis_;
            int64_t outOfset = rowOfset + GetBlockIdx() * eachCoreAfterAxisCount_ + colIdx * afterAxisFactor_;
            PipeBarrier<PIPE_MTE3>();
            SetAtomicAdd<T>();
            CopyOut<T>(yGm_[outOfset], dataLocal[i * colLenAlignSize], colLen);
            SetAtomicNone();
        }
    }

    __aicore__ inline void CopyIndiceInSplitAfter(int64_t rowIdx, int64_t rowLen)
    {
        LocalTensor<U> indicesLocal = indicesBuf_.Get<U>();
        LocalTensor<U> outOfstLocal = outOfstBuf_.Get<U>();
        LocalTensor<float> dstLocal = maxScoreBuf_.Get<float>();

        int64_t rankSize = indexRankSize_;
        int64_t indicesOfset = rowIdx * indicesFactor_;
        CopyIn<U>(indicesLocal, indicesGm_[indicesOfset * rankSize], rowLen * rankSize);
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        ComputeOutOfset(indicesLocal, outOfstLocal, rowLen, rankSize);
        if constexpr (IsSameType<U, int32_t>::value) {
            IndexStatisticInt32(outOfstLocal, dstLocal, maxScore_, rowLen, afterAxis_);
        } else {
            IndexStatisticInt64(outOfstLocal, dstLocal, maxScore_, rowLen, afterAxis_);
        }
    }

    __aicore__ inline void CopyUpdatesInSplitAfter(int64_t rowIdx, int64_t colIdx, int64_t rowLen, int64_t colLen)
    {
        LocalTensor<T> updatesLocal = dataQueue_.AllocTensor<T>();
        int64_t indicesOfset = rowIdx * indicesFactor_;
        DataCopyExtParams copyParams = {
            static_cast<uint16_t>(rowLen), static_cast<uint32_t>(colLen * sizeof(T)),
            static_cast<uint32_t>((afterAxis_ - colLen) * sizeof(T)), static_cast<uint32_t>(0),
            static_cast<uint32_t>(0)};
        DataCopyPadExtParams<T> updatePadParams = {false, 0, 0, 0};
        int64_t rowOfset = indicesOfset * afterAxis_;
        int64_t updatesOfset = rowOfset + GetBlockIdx() * eachCoreAfterAxisCount_ + colIdx * afterAxisFactor_;
        DataCopyPad(updatesLocal, updatesGm_[updatesOfset], copyParams, updatePadParams);
        dataQueue_.EnQue(updatesLocal);
    }

    __aicore__ inline void ProcessAfterSingleNonSort(int64_t rowIdx, int64_t colIdx, int64_t rowLen, int64_t colLen)
    {
        LocalTensor<U> indicesLocal = indicesBuf_.Get<U>();
        LocalTensor<U> outOfstLocal = outOfstBuf_.Get<U>();
        LocalTensor<T> updatesLocal = dataQueue_.AllocTensor<T>();

        int64_t colLenAlignSize = ops::CeilAlign(colLen * sizeof(T), UB_AGLIN_VALUE) / sizeof(T);
        int64_t colLenAlignFp32 = ops::CeilAlign(colLen * sizeof(float), UB_AGLIN_VALUE) / sizeof(float);
        int64_t indicesOfset = rowIdx * indicesFactor_;

        CopyIn<U>(indicesLocal, indicesGm_[indicesOfset * indexRankSize_], rowLen * indexRankSize_);
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        ComputeOutOfset(indicesLocal, outOfstLocal, rowLen, indexRankSize_);
        event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);

        DataCopyExtParams copyParams = {
            static_cast<uint16_t>(rowLen), static_cast<uint32_t>(colLen * sizeof(T)),
            static_cast<uint32_t>((afterAxis_ - colLen) * sizeof(T)), static_cast<uint32_t>(0),
            static_cast<uint32_t>(0)};
        DataCopyPadExtParams<T> updatePadParams = {false, 0, 0, 0};
        int64_t rowOfset = indicesOfset * afterAxis_;
        int64_t updatesOfset = rowOfset + GetBlockIdx() * eachCoreAfterAxisCount_ + colIdx * afterAxisFactor_;
        DataCopyPad(updatesLocal, updatesGm_[updatesOfset], copyParams, updatePadParams);

        event_t eventIdMte2ToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
        SetFlag<HardEvent::MTE2_MTE3>(eventIdMte2ToMte3);
        WaitFlag<HardEvent::MTE2_MTE3>(eventIdMte2ToMte3);

        dataQueue_.EnQue(updatesLocal);
    }

    __aicore__ inline void ProcessAfterSingleLoop(int64_t rowIdx, int64_t colIdx, int64_t rowLen, int64_t colLen)
    {
        LocalTensor<U> indicesLocal = indicesBuf_.Get<U>();
        LocalTensor<T> updatesLocal = dataQueue_.AllocTensor<T>();
        LocalTensor<U> outOfstLocal = outOfstBuf_.Get<U>();
        
        int64_t indicesOfset = rowIdx * indicesFactor_;
        CopyIn<U>(indicesLocal, indicesGm_[indicesOfset * indexRankSize_], rowLen * indexRankSize_);
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

        ComputeOutOfset(indicesLocal, outOfstLocal, rowLen, indexRankSize_);

        event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
        DataCopyExtParams copyParams = {static_cast<uint16_t>(rowLen),
                                        static_cast<uint32_t>(colLen * sizeof(T)),
                                        static_cast<uint32_t>((afterAxis_ - colLen) * sizeof(T)),
                                        static_cast<uint32_t>(0),
                                        static_cast<uint32_t>(0)};

        DataCopyPadExtParams<T> updatePadParams = {false, 0, 0, 0};

        int64_t rowOfset = indicesOfset * afterAxis_;
        int64_t updatesOfset = rowOfset + GetBlockIdx() * eachCoreAfterAxisCount_ + colIdx * afterAxisFactor_;
        DataCopyPad(updatesLocal, updatesGm_[updatesOfset], copyParams, updatePadParams);

        event_t eventIdMte2ToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
        SetFlag<HardEvent::MTE2_MTE3>(eventIdMte2ToMte3);
        WaitFlag<HardEvent::MTE2_MTE3>(eventIdMte2ToMte3);

        if (rowLen < MIN_SAME_IDX_ACCM_COUNT) {
            ComputeIdxCount(outOfstLocal, rowLen, colLen);
            if (uniqueIdNum_ == rowLen) {
                CopyOutSplitAfter(outOfstLocal, updatesLocal, rowLen, colLen, colIdx);
            } else {
                CopyOutSplitAfterDuplicateIdx(outOfstLocal, updatesLocal, rowLen, colLen, colIdx);
            }
            dataQueue_.FreeTensor(updatesLocal);
            return;
        }

        ComputeSumForUniqueIdx(outOfstLocal, updatesLocal, rowLen, colLen);
        LocalTensor<U> updateSumIdxLocal = updateSumIdxQue_.DeQue<U>();
        if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
            LocalTensor<float> updateSumLocal = updateSumQue_.DeQue<float>();
            ExecConvertAlign(updatesLocal, updateSumLocal, rowLen, colLen);

            event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

            CopyOutSplitAfter(updateSumIdxLocal, updatesLocal, uniqueIdNum_, colLen, colIdx);
            updateSumQue_.FreeTensor(updateSumLocal);
        } else {
            LocalTensor<T> updateSumLocal = updateSumQue_.DeQue<T>();
            CopyOutSplitAfter(updateSumIdxLocal, updateSumLocal, uniqueIdNum_, colLen, colIdx);
            
            updateSumQue_.FreeTensor(updateSumLocal);
        }

        dataQueue_.FreeTensor(updatesLocal);
        updateSumIdxQue_.FreeTensor(updateSumIdxLocal);
    }
};
} // namespace ScatterNdAdd

#endif