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
 * \file scatter_nd_common_base.h
 * \brief scatter_nd_common_base
 */
#ifndef ASCENDC_SCATTER_ND_COMMON_BASE_H_
#define ASCENDC_SCATTER_ND_COMMON_BASE_H_

#include "kernel_operator.h"
#include "../inc/platform.h"
#include "scatter_nd_common_struct.h"
#include "../inc/load_store_utils.h"
#include "./indices_sort_utils.h"

namespace ScatterNdCommon {
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
constexpr float SORT_HIST_THRESHOLD = 0.03f;
constexpr uint32_t HASH_SCORE_BUF_SIZE = 128;
constexpr uint32_t CAST_0 = 0;
constexpr uint32_t CAST_1 = 1;
constexpr uint32_t CAST_2 = 2;
constexpr uint32_t CAST_3 = 3;
constexpr uint32_t CAST_4 = 4;
constexpr uint32_t CAST_5 = 5;
constexpr uint8_t MODE_MAX = 1; // 1: scatter_nd_max;
constexpr uint8_t MODE_MIN = 0; // 0: scatter_nd_min;
constexpr float FLOAT32_MAX = 3.4028235e+38f;
constexpr half FLOAT16_MAX = 65504.0f;
constexpr bfloat16_t BFLOAT16_MAX = 3.3895314e+38f;
constexpr float FLOAT32_MIN = -3.4028235e+38f;
constexpr half FLOAT16_MIN = -65504.0f;
constexpr bfloat16_t BFLOAT16_MIN = -3.3895314e+38f;
constexpr int64_t VFLEN_INT64 = platform::GetVRegSize() / sizeof(int64_t);
constexpr int64_t VFLEN_INT32 = platform::GetVRegSize() / sizeof(int32_t);
constexpr int64_t VFLEN_INT16 = platform::GetVRegSize() / sizeof(int16_t);
constexpr int64_t VFLEN_INT16HALF = platform::GetVRegSize() / sizeof(int16_t) / TWO;
constexpr int64_t VFLEN_UINT8 = platform::GetVRegSize() / sizeof(uint8_t);
constexpr int64_t VFLEN_UINT8HALFHALF = platform::GetVRegSize() / sizeof(uint8_t) / FOUR;
constexpr uint32_t U8_MAX = 255;

constexpr MicroAPI::CastTrait castTraitB322B64 = {
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

static constexpr SortConfig sortConfig{SortType::RADIX_SORT, false};

__aicore__ inline uint32_t ROUND_UP32(uint32_t x)
{
    if (x % UB_AGLIN_VALUE != 0) {
        return (x / UB_AGLIN_VALUE + 1) * UB_AGLIN_VALUE;
    }
    return x;
}

template <typename TYPE>
__aicore__ inline constexpr TYPE GetDtypeMax()
{
    TYPE dtypeMax = 0;
    if constexpr (IsSameType<TYPE, int32_t>::value) {
        dtypeMax = INT32_MAX;
    } else if constexpr (IsSameType<TYPE, int64_t>::value) {
        dtypeMax = INT64_MAX;
    } else if constexpr (IsSameType<TYPE, uint32_t>::value) {
        dtypeMax = UINT32_MAX;
    } else if constexpr (IsSameType<TYPE, uint64_t>::value) {
        dtypeMax = UINT64_MAX;
    } else if constexpr (IsSameType<TYPE, bfloat16_t>::value) {
        dtypeMax = BFLOAT16_MAX;
    } else if constexpr (IsSameType<TYPE, half>::value) {
        dtypeMax = FLOAT16_MAX;
    } else if constexpr (IsSameType<TYPE, float>::value) {
        dtypeMax = FLOAT32_MAX;
    } else if constexpr (IsSameType<TYPE, int8_t>::value) {
        dtypeMax = INT8_MAX;
    } else if constexpr (IsSameType<TYPE, int16_t>::value) {
        dtypeMax = INT16_MAX;
    }
    return dtypeMax;
}

template <typename TYPE>
__aicore__ inline constexpr TYPE GetDtypeMin()
{
    TYPE dtypeMin = 0;
    if constexpr (IsSameType<TYPE, int32_t>::value) {
        dtypeMin = INT32_MIN;
    } else if constexpr (IsSameType<TYPE, int64_t>::value) {
        dtypeMin = INT64_MIN;
    } else if constexpr (IsSameType<TYPE, uint32_t>::value) {
        dtypeMin = 0;
    } else if constexpr (IsSameType<TYPE, uint64_t>::value) {
        dtypeMin = 0;
    } else if constexpr (IsSameType<TYPE, bfloat16_t>::value) {
        dtypeMin = BFLOAT16_MIN;
    } else if constexpr (IsSameType<TYPE, half>::value) {
        dtypeMin = FLOAT16_MIN;
    } else if constexpr (IsSameType<TYPE, float>::value) {
        dtypeMin = FLOAT32_MIN;
    } else if constexpr (IsSameType<TYPE, int8_t>::value) {
        dtypeMin = INT8_MIN;
    } else if constexpr (IsSameType<TYPE, int16_t>::value) {
        dtypeMin = INT16_MIN;
    }
    return dtypeMin;
}

template <typename T, typename U, typename CAST_T = U, typename OFFSET_T = U, uint32_t castType = CAST_0, uint8_t Mode = MODE_MAX>
class ScatterNdCommonBase
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
    TBuf<QuePosition::VECCALC> outputShapeBuf_;
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
        IsSameType<OFFSET_T, int64_t>::value,
        typename AscendC::MicroAPI::RegTensor<int64_t, AscendC::MicroAPI::RegTraitNumTwo>,
        typename AscendC::MicroAPI::RegTensor<int32_t>>::type;

    using selRegType = typename std::conditional<IsSameType<T, bool>::value, int8_t, T>::type;

    using COMPUTE_TYPE = typename std::conditional<IsSameType<T, bool>::value, int8_t, T>::type;

    __aicore__ inline void InitBaseBuffer(
        TPipe& pipe, uint32_t indicesNumber, GM_ADDR indices, GM_ADDR updates, GM_ADDR y, int64_t singleCol = 0)
    {
        indicesGm_.SetGlobalBuffer((__gm__ U*)(indices));
        updatesGm_.SetGlobalBuffer((__gm__ T*)(updates));
        yGm_.SetGlobalBuffer((__gm__ T*)(y));

        pipe.InitBuffer(strideBuf_, MAX_RANK_COUNT * sizeof(OFFSET_T));
        pipe.InitBuffer(outputShapeBuf_, MAX_SHAPE_RANK * sizeof(U));
        pipe.InitBuffer(outOfstBuf_, indicesFactor_ * sizeof(OFFSET_T));
        pipe.InitBuffer(indicesBuf_, indicesFactor_ * indexRankSize_ * sizeof(U));
        pipe.InitBuffer(maxScoreBuf_, HASH_SCORE_BUF_SIZE * sizeof(float));

        pipe.InitBuffer(updatesOriginIdexQue_, 1, indicesFactor_ * sizeof(uint32_t));
        pipe.InitBuffer(
            uniqueIdCountQue_, 1, ops::CeilAlign((indicesFactor_ + 1) * sizeof(int32_t), UB_AGLIN_VALUE));
        pipe.InitBuffer(
            updateSumIdxQue_, 1, ops::CeilAlign((indicesFactor_ + 1) * sizeof(OFFSET_T), UB_AGLIN_VALUE));
        if (singleCol) {
            pipe.InitBuffer(dataQueue_, DOUBLE_BUFFER, afterAxisFactor_ * sizeof(T));
            pipe.InitBuffer(updateSumQue_, DOUBLE_BUFFER, afterAxisFactor_ * sizeof(T));
        } else {
            pipe.InitBuffer(dataQueue_, 1, indicesFactor_ * afterAxisFactor_ * sizeof(T));
            pipe.InitBuffer(updateSumQue_, 1, indicesFactor_ * afterAxisFactor_ * sizeof(T));
        }

        if constexpr (castType == CAST_0) {
            pipe.InitBuffer(sortIndicesQue_, ops::CeilAlign(indicesFactor_ * sizeof(OFFSET_T) + SORT_PAD_NUM * UB_AGLIN_VALUE, UB_AGLIN_VALUE));
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

    __simd_vf__ inline void ComputeOutOfsetVF(__ubuf__ U* indicesLocalPtr, __ubuf__ OFFSET_T* outOfstLocalPtr, __ubuf__ OFFSET_T* strideLocalPtr,
                                              __ubuf__ U* outputShapeLocalPtr, int32_t rankSize, uint32_t dataLen, uint32_t vfLen, uint16_t loopCnt, uint16_t rankSizeLoops)
    {
        InnerRegType inReg;
        InnerRegType outReg;
        InnerRegType orderReg;
        InnerRegType selectReg;
        IndexRegType indexReg;
        AscendC::MicroAPI::MaskReg pregLoop;
        AscendC::MicroAPI::MaskReg cmpMask;
        AscendC::MicroAPI::MaskReg invalidMask;

        for (uint16_t i = 0; i < loopCnt; i++) {
            if constexpr (IsSameType<OFFSET_T, int64_t>::value) {
                pregLoop = AscendC::MicroAPI::UpdateMask<OFFSET_T, AscendC::MicroAPI::RegTraitNumTwo>(dataLen);
                invalidMask = AscendC::MicroAPI::CreateMask<OFFSET_T, MicroAPI::MaskPattern::ALLF, AscendC::MicroAPI::RegTraitNumTwo>();
            } else {
                pregLoop = AscendC::MicroAPI::UpdateMask<OFFSET_T>(dataLen);
                invalidMask = AscendC::MicroAPI::CreateMask<OFFSET_T, MicroAPI::MaskPattern::ALLF>();
            }
            AscendC::MicroAPI::Duplicate(outReg, 0, pregLoop);
            AscendC::MicroAPI::Arange(orderReg, i * vfLen);
            AscendC::MicroAPI::Muls(orderReg, orderReg, rankSize, pregLoop);
            for (uint16_t dim = 0; dim < rankSizeLoops; dim++) {
                OFFSET_T strideValue = strideLocalPtr[dim];
                U outputShapeValue = outputShapeLocalPtr[dim];
                indexReg = (IndexRegType&)orderReg;

                if constexpr (IsSameType<U, int32_t>::value && IsSameType<OFFSET_T, int64_t>::value) {
                    AscendC::MicroAPI::RegTensor<int32_t> castReg;
                    AscendC::MicroAPI::DataCopyGather(castReg, indicesLocalPtr, indexReg, pregLoop);
                    MicroAPI::Cast<int64_t, int32_t, castTraitB322B64>(inReg, castReg, pregLoop);
                } else {
                    AscendC::MicroAPI::DataCopyGather(inReg, indicesLocalPtr, indexReg, pregLoop);
                }
                AscendC::MicroAPI::CompareScalar<OFFSET_T, CMPMODE::LT>(cmpMask, inReg, static_cast<OFFSET_T>(0), pregLoop);
                AscendC::MicroAPI::Or(invalidMask, invalidMask, cmpMask, pregLoop);
                AscendC::MicroAPI::CompareScalar<OFFSET_T, CMPMODE::GE>(cmpMask, inReg, static_cast<OFFSET_T>(outputShapeValue), pregLoop);
                AscendC::MicroAPI::Or(invalidMask, invalidMask, cmpMask, pregLoop);

                AscendC::MicroAPI::Muls(inReg, inReg, strideValue, pregLoop);
                AscendC::MicroAPI::Add(outReg, inReg, outReg, pregLoop);
                AscendC::MicroAPI::Adds(orderReg, orderReg, (OFFSET_T)(1), pregLoop);
            }
            AscendC::MicroAPI::Duplicate(selectReg, static_cast<OFFSET_T>(-2), pregLoop);
            AscendC::MicroAPI::Select(outReg, selectReg, outReg, invalidMask);
            auto outOfstAddr = outOfstLocalPtr + i * vfLen;
            AscendC::MicroAPI::DataCopy(outOfstAddr, outReg, pregLoop);
        }
    }

    __aicore__ inline void ComputeOutOfset(
        const LocalTensor<U> indicesLocal, const LocalTensor<OFFSET_T> outOfstLocal, int32_t indicesLen, int32_t rankSize)
    {
        LocalTensor<OFFSET_T> strideLocal = strideBuf_.Get<OFFSET_T>();
        LocalTensor<U> outputShapeLocal = outputShapeBuf_.Get<U>();

        __local_mem__ U* indicesLocalPtr = ((__local_mem__ U*)indicesLocal.GetPhyAddr());
        __local_mem__ OFFSET_T* outOfstLocalPtr = ((__local_mem__ OFFSET_T*)outOfstLocal.GetPhyAddr());
        __local_mem__ OFFSET_T* strideLocalPtr = ((__local_mem__ OFFSET_T*)strideLocal.GetPhyAddr());
        __local_mem__ U* outputShapeLocalPtr = ((__local_mem__ U*)outputShapeLocal.GetPhyAddr());

        uint32_t dataLen = indicesLen;
        uint32_t vfLen = platform::GetVRegSize() / sizeof(int32_t);
        uint32_t indicesLenTimes = ops::CeilDiv(dataLen, vfLen);
        uint16_t loopCnt = static_cast<uint16_t>(indicesLenTimes);
        uint16_t rankSizeLoops = static_cast<uint16_t>(rankSize);

        ComputeOutOfsetVF(indicesLocalPtr, outOfstLocalPtr, strideLocalPtr, outputShapeLocalPtr, rankSize, dataLen, vfLen, loopCnt, rankSizeLoops);
        return;
    }

    __simd_callee__ inline void  ComputeUniqueIdNumInt64(__local_mem__ CAST_T* indicesAddr, __local_mem__ int32_t* uniqueIdCountsAddr, uint16_t loopCnt, int64_t dataLen)
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

    __simd_callee__ inline void  ComputeUniqueIdNumInt32(__local_mem__ CAST_T* indicesAddr, __local_mem__ int32_t* uniqueIdCountsAddr, uint16_t loopCnt, int64_t dataLen)
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

    __simd_callee__ inline void  ComputeUniqueIdNumInt16(__local_mem__ CAST_T* indicesAddr, __local_mem__ int32_t* uniqueIdCountsAddr, uint16_t loopCnt, int64_t dataLen)
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
            AscendC::MicroAPI::DataCopyUnAlign<int32_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(uniqueIdCountsAddr, selReg, uOut);
            AscendC::MicroAPI::GatherMask<int32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(selReg2, orderReg2, maskDouble2);
            AscendC::MicroAPI::DataCopyUnAlign<int32_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(uniqueIdCountsAddr, selReg2, uOut);
        }
        AscendC::MicroAPI::DataCopyUnAlignPost(uniqueIdCountsAddr, uOut);
    }

    __simd_callee__ inline void  ComputeUniqueIdNumUint8(__local_mem__ CAST_T* indicesAddr, __local_mem__ int32_t* uniqueIdCountsAddr, uint16_t loopCnt, int64_t dataLen)
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
            AscendC::MicroAPI::DataCopyUnAlign<int32_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(uniqueIdCountsAddr, selReg, uOut);
            AscendC::MicroAPI::GatherMask<int32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(selReg2, orderReg2, maskFour2);
            AscendC::MicroAPI::DataCopyUnAlign<int32_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(uniqueIdCountsAddr, selReg2, uOut);
            AscendC::MicroAPI::GatherMask<int32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(selReg3, orderReg3, maskFour3);
            AscendC::MicroAPI::DataCopyUnAlign<int32_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(uniqueIdCountsAddr, selReg3, uOut);
            AscendC::MicroAPI::GatherMask<int32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(selReg4, orderReg4, maskFour4);
            AscendC::MicroAPI::DataCopyUnAlign<int32_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(uniqueIdCountsAddr, selReg4, uOut);
        }
        AscendC::MicroAPI::DataCopyUnAlignPost(uniqueIdCountsAddr, uOut);
    }

    
    __simd_vf__ inline void ComputeUniqueIdNumVF(__ubuf__ CAST_T* indicesAddr, __ubuf__ int32_t* uniqueIdCountsAddr, int64_t dataLen, uint16_t loopCnt)
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

    __aicore__ inline uint32_t
    ComputeUniqueIdNum(LocalTensor<CAST_T> indicesLocal, LocalTensor<int32_t> uniqueIdCountLocal, int64_t dataLen)
    {
        __local_mem__ CAST_T* indicesAddr = (__local_mem__ CAST_T*)indicesLocal[(UB_AGLIN_VALUE / sizeof(CAST_T))].GetPhyAddr();
        __local_mem__ int32_t* uniqueIdCountsAddr = (__local_mem__ int32_t*)uniqueIdCountLocal.GetPhyAddr();

        int64_t vfLen = platform::GetVRegSize() / sizeof(CAST_T);
        uint16_t loopCnt = ops::CeilDiv(dataLen + 1, vfLen);

        ComputeUniqueIdNumVF(indicesAddr, uniqueIdCountsAddr, dataLen, loopCnt);

        uint32_t uniqueIdNum = ((AscendC::MicroAPI::GetSpr<AscendC::SpecialPurposeReg::AR>()) / sizeof(int32_t)) - 1;
        return uniqueIdNum;
    }

    __simd_vf__ inline void  ComputeUinqueIdTimesVF(__ubuf__ int32_t* uniqueIdCountsAddr, uint32_t uniqueIdNum, uint32_t vfLen, uint16_t loopSize)
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
    
    __aicore__ inline void  ComputeUinqueIdTimes(LocalTensor<int32_t> uniqueIdCountLocal, uint32_t uniqueIdNum)
    {
        __local_mem__ int32_t* uniqueIdCountsAddr = (__local_mem__ int32_t*)uniqueIdCountLocal.GetPhyAddr();
        uint32_t vfLen = platform::GetVRegSize() / sizeof(int32_t);
        uint16_t loopSize = ops::CeilDiv(uniqueIdNum, vfLen);
        ComputeUinqueIdTimesVF(uniqueIdCountsAddr, uniqueIdNum, vfLen, loopSize);
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

    __simd_vf__ inline void  ComputeSumWithOutCastVF(__ubuf__ selRegType* updatesAddr, __ubuf__ selRegType* updateSumAddr, __ubuf__ int32_t* uniqueIdCountAddr,
                                                     __ubuf__ uint32_t* updatesOriginIdexAddr, uint32_t uniqueIdNum, int64_t colLen, uint32_t vfLen, int32_t loopSize,
                                                     int32_t idLocation, int64_t colLenAlignSize, selRegType dupNum)
    {
        for (uint16_t i = 0; i < static_cast<uint16_t>(uniqueIdNum); i++) {
            AscendC::MicroAPI::RegTensor<selRegType> sumReg;
            AscendC::MicroAPI::RegTensor<selRegType> updateReg;
            AscendC::MicroAPI::MaskReg maskReg;
            AscendC::MicroAPI::MaskReg zeroMask = AscendC::MicroAPI::CreateMask<selRegType>();
            uint32_t maskLen = static_cast<uint32_t>(colLen);
            uint16_t idRepeatTimes = static_cast<uint16_t>(uniqueIdCountAddr[i]);
            for (uint16_t j = 0; j < static_cast<uint16_t>(loopSize); j++) {
                maskReg = AscendC::MicroAPI::UpdateMask<selRegType>(maskLen);
                AscendC::MicroAPI::Duplicate(sumReg, dupNum, zeroMask);
                for (uint16_t k = 0; k < idRepeatTimes; k++) {
                    auto updatesOffet = updatesOriginIdexAddr[idLocation + k] * colLenAlignSize + j * vfLen;
                    auto startAddr = updatesAddr + updatesOffet;
                    AscendC::MicroAPI::DataCopy(updateReg, startAddr);
                    if constexpr (Mode == MODE_MAX) {
                        AscendC::MicroAPI::Max(sumReg, sumReg, updateReg, maskReg);
                    } else {
                        AscendC::MicroAPI::Min(sumReg, sumReg, updateReg, maskReg);
                    }
                }
                auto updateSumAddrOfst = updateSumAddr + i * colLenAlignSize + j * vfLen;
                AscendC::MicroAPI::DataCopy(updateSumAddrOfst, sumReg, maskReg);
            }
            idLocation += idRepeatTimes;
        }
    }
    
    __aicore__ inline void  ComputeSumWithOutCast(
        LocalTensor<int32_t> uniqueIdCountLocal, LocalTensor<uint32_t> updatesOriginIdexLocal,
        LocalTensor<T> updatesLocal, LocalTensor<T> updateSumLocal, uint32_t uniqueIdNum, int64_t colLen)
    {
        __local_mem__ selRegType* updatesAddr = (__local_mem__ selRegType*)updatesLocal.GetPhyAddr();
        __local_mem__ selRegType* updateSumAddr = (__local_mem__ selRegType*)updateSumLocal.GetPhyAddr();
        __local_mem__ int32_t* uniqueIdCountAddr = (__local_mem__ int32_t*)uniqueIdCountLocal.GetPhyAddr();
        __local_mem__ uint32_t* updatesOriginIdexAddr = (__local_mem__ uint32_t*)updatesOriginIdexLocal.GetPhyAddr();

        uint32_t vfLen = platform::GetVRegSize() / sizeof(selRegType);
        int32_t loopSize = (colLen + vfLen - 1) / vfLen;
        int32_t idLocation = 0;
        int64_t colLenAlignSize = ops::CeilAlign(colLen * sizeof(selRegType), UB_AGLIN_VALUE) / sizeof(selRegType);
        
        selRegType dupNum = 0;
        if constexpr (Mode == MODE_MAX) {
            dupNum = GetDtypeMin<selRegType>();
        } else {
            dupNum = GetDtypeMax<selRegType>();
        }

        ComputeSumWithOutCastVF(updatesAddr, updateSumAddr, uniqueIdCountAddr, updatesOriginIdexAddr,
                                uniqueIdNum, colLen, vfLen, loopSize, idLocation, colLenAlignSize, dupNum);
    }

    __aicore__ inline void  ComputeUpdateSum(int64_t rowLen, int64_t colLen) // rowLen 不用
    {
        LocalTensor<int32_t> uniqueIdCountLocal = uniqueIdCountQue_.DeQue<int32_t>(); // 存放每个排好序的 非重复的 indices的个数
        LocalTensor<uint32_t> updatesOriginIdexLocal = updatesOriginIdexQue_.DeQue<uint32_t>(); // 排序之后的indices的索引
        LocalTensor<T> updatesLocal = dataQueue_.DeQue<T>();

        LocalTensor<T> updateSumLocal = updateSumQue_.AllocTensor<T>();

        ComputeSumWithOutCast(uniqueIdCountLocal, updatesOriginIdexLocal, updatesLocal, updateSumLocal, uniqueIdNum_, colLen);

        updatesOriginIdexQue_.EnQue(updatesOriginIdexLocal);
        uniqueIdCountQue_.EnQue(uniqueIdCountLocal);

        updateSumQue_.EnQue(updateSumLocal);
        dataQueue_.EnQue(updatesLocal);
    }

    __aicore__ inline void  SortIndices(
        LocalTensor<OFFSET_T> outOfstLocal, int64_t rowLen)
    {
        LocalTensor<CAST_T> sortIndicesLocal = sortIndicesQue_.Get<CAST_T>(); // CAST_0 情况下 CAST_T = OFFSET_T
        LocalTensor<int32_t> uniqueIdCountLocal = uniqueIdCountQue_.AllocTensor<int32_t>();
        LocalTensor<uint32_t> updatesOriginIdexLocal = updatesOriginIdexQue_.AllocTensor<uint32_t>();
        LocalTensor<CAST_T> shiftSortLocal = sortIndicesLocal[(UB_AGLIN_VALUE / sizeof(CAST_T))];
        if constexpr (castType == CAST_0) {
            AscendC::Sort<CAST_T, false, sortConfig>( // 排序之后的indices, 排序之后的indices的索引, 原始indices
                shiftSortLocal, updatesOriginIdexLocal, outOfstLocal, static_cast<uint32_t>(rowLen));
            Duplicate(sortIndicesLocal, (CAST_T)-1, shiftOffset_);
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

        LocalTensor<OFFSET_T> updateSumIdxLocal = updateSumIdxQue_.AllocTensor<OFFSET_T>();
        
        uniqueIdNum_ = ComputeUniqueIdNum(sortIndicesLocal, uniqueIdCountLocal, rowLen);
        event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);

        for (uint32_t idx = 0; idx < uniqueIdNum_; idx++) {
            auto offset = uniqueIdCountLocal(idx); // uniqueIdCountLocal 存放排好序的 非重复的 indices的索引
            updateSumIdxLocal(idx) = static_cast<OFFSET_T>(shiftSortLocal(offset)); // 存放排好序的 非重复的 indices值
        }
        event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventIdSToV);
        WaitFlag<HardEvent::S_V>(eventIdSToV);
        ComputeUinqueIdTimes(uniqueIdCountLocal, uniqueIdNum_); // uniqueIdCountLocal 存放每个排好序的 非重复的 indices的个数
        PipeBarrier<PIPE_V>();

        updatesOriginIdexQue_.EnQue(updatesOriginIdexLocal);
        uniqueIdCountQue_.EnQue(uniqueIdCountLocal);

        updateSumIdxQue_.EnQue(updateSumIdxLocal);
    }

    __aicore__ inline void CopyOutSplitIndices(
        LocalTensor<OFFSET_T> ofstLocal, LocalTensor<T> dataLocal, int64_t rowLen, int64_t colLen, int64_t colIdx)
    {
        int64_t colLenAlignSize = ops::CeilAlign(colLen * sizeof(T), UB_AGLIN_VALUE) / sizeof(T);
        for (int64_t i = 0; i < rowLen; i++) {
            if (ofstLocal(i) < 0) {
                continue;
            }
            int64_t rowOfset = ofstLocal(i) * afterAxis_;
            int64_t outOfset = rowOfset + colIdx * afterAxisFactor_;
            if constexpr (IsSameType<T, bool>::value) {
                if constexpr (Mode == MODE_MAX) {
                    SetAtomicMax<int8_t>();
                } else {
                    SetAtomicMin<int8_t>();
                }
                CopyOut<bool>(yGm_[outOfset], dataLocal[i * colLenAlignSize], colLen);
                SetAtomicNone();
            } else {
                if constexpr (Mode == MODE_MAX) {
                    SetAtomicMax<T>();
                } else {
                    SetAtomicMin<T>();
                }
                CopyOut<T>(yGm_[outOfset], dataLocal[i * colLenAlignSize], colLen);
                SetAtomicNone();
            }
        }
    }

    __aicore__ inline void CopyIndiceInSplitIndices(int64_t rowIdx, int64_t rowLen)
    {
        LocalTensor<U> indicesLocal = indicesBuf_.Get<U>();
        LocalTensor<OFFSET_T> outOfstLocal = outOfstBuf_.Get<OFFSET_T>();
        LocalTensor<float> dstLocal = maxScoreBuf_.Get<float>();

        int64_t rankSize = indexRankSize_;
        int64_t indicesOfset = GetBlockIdx() * eachCoreIndexCount_ + rowIdx * indicesFactor_;
        this->template CopyIn<U>(indicesLocal, indicesGm_[indicesOfset * rankSize], rowLen * rankSize);
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        this->ComputeOutOfset(indicesLocal, outOfstLocal, rowLen, rankSize);
        if constexpr (IsSameType<OFFSET_T, int32_t>::value) {
            IndexStatisticInt32(outOfstLocal, dstLocal, this->maxScore_, rowLen, afterAxis_);
        } else {
            IndexStatisticInt64(outOfstLocal, dstLocal, this->maxScore_, rowLen, afterAxis_);
        }
    }

    __simd_vf__ inline void SingleColAddVF(__ubuf__ selRegType* updatesAddr, __ubuf__ COMPUTE_TYPE* updateSumAddr, int64_t colLen, uint32_t vfLen, int32_t loopSize)
    {
        AscendC::MicroAPI::RegTensor<COMPUTE_TYPE> sumReg;
        AscendC::MicroAPI::RegTensor<COMPUTE_TYPE> updateReg;
        AscendC::MicroAPI::MaskReg maskReg;
        uint32_t maskLen = static_cast<uint32_t>(colLen);
        for (uint16_t j = 0; j < static_cast<uint16_t>(loopSize); j++) {
            maskReg = AscendC::MicroAPI::UpdateMask<COMPUTE_TYPE>(maskLen);
            AscendC::MicroAPI::DataCopy(sumReg, updateSumAddr + j * vfLen);
            AscendC::MicroAPI::DataCopy(updateReg, updatesAddr + j * vfLen);
            if constexpr (Mode == MODE_MAX) {
                AscendC::MicroAPI::Max(sumReg, sumReg, updateReg, maskReg);
            } else {
                AscendC::MicroAPI::Min(sumReg, sumReg, updateReg, maskReg);
            }
            auto updateSumAddrOfst = updateSumAddr + j * vfLen;
            AscendC::MicroAPI::DataCopy(updateSumAddrOfst, sumReg, maskReg);
        }
    }

    __aicore__ inline void SingleColAdd(LocalTensor<COMPUTE_TYPE> updateSumLocal, int64_t colLen) {
        LocalTensor<T> updatesLocal = dataQueue_.DeQue<T>();
        __local_mem__ selRegType* updatesAddr = (__local_mem__ selRegType*)updatesLocal.GetPhyAddr();
        __local_mem__ COMPUTE_TYPE* updateSumAddr = (__local_mem__ COMPUTE_TYPE*)updateSumLocal.GetPhyAddr();
        uint32_t vfLen = platform::GetVRegSize() / sizeof(COMPUTE_TYPE);
        int32_t loopSize = (colLen + vfLen - 1) / vfLen;
        SingleColAddVF(updatesAddr, updateSumAddr, colLen, vfLen, loopSize);
        dataQueue_.FreeTensor(updatesLocal);
    }

    __aicore__ inline void CopyOutSingleCol(LocalTensor<T> outLocal, int64_t outOffset, int64_t colLen)
    {
        if constexpr (IsSameType<T, bool>::value) {
            if constexpr (Mode == MODE_MAX) {
                SetAtomicMax<int8_t>();
            } else {
                SetAtomicMin<int8_t>();
            }
            CopyOut<bool>(yGm_[outOffset], outLocal, colLen);
            SetAtomicNone();
        } else {
            if constexpr (Mode == MODE_MAX) {
                SetAtomicMax<T>();
            } else {
                SetAtomicMin<T>();
            }
            CopyOut<T>(yGm_[outOffset], outLocal, colLen);
            SetAtomicNone();
        }
    }

    __aicore__ inline void CopyInUpdates(int64_t updatesOffset, int64_t rowLen, int64_t colLen) {
        LocalTensor<T> updatesLocal = dataQueue_.AllocTensor<T>();
        DataCopyExtParams copyParams = {
            static_cast<uint16_t>(rowLen), static_cast<uint32_t>(colLen * sizeof(T)),
            static_cast<uint32_t>((afterAxis_ - colLen) * sizeof(T)), static_cast<uint32_t>(0),
            static_cast<uint32_t>(0)};
        DataCopyPadExtParams<T> updatePadParams = {false, 0, 0, 0};
        DataCopyPad(updatesLocal, updatesGm_[updatesOffset], copyParams, updatePadParams);
        dataQueue_.EnQue(updatesLocal);
    }

    __aicore__ inline void ComputeSortInOutSingleCol(
        int64_t rowIdx, int64_t colIdx, int64_t rowLen, int64_t colLen) // rowLen 不用
    {
        // rowIdx 可以改为indicesOffset， splitAfter和SplitBefore通用 colIdx 同理
        int64_t indicesOffset = GetBlockIdx() * eachCoreIndexCount_ + rowIdx * indicesFactor_;
        event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
        WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);

        event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);

        LocalTensor<int32_t> uniqueIdCountLocal = uniqueIdCountQue_.DeQue<int32_t>();
        LocalTensor<uint32_t> updatesOriginIdexLocal = updatesOriginIdexQue_.DeQue<uint32_t>(); // 排序之后的indices的索引
        LocalTensor<OFFSET_T> ofstLocal = updateSumIdxQue_.DeQue<OFFSET_T>();
        int64_t updatesOffset = indicesOffset * afterAxis_ + colIdx * afterAxisFactor_;
        int32_t idLocation = 0;
        COMPUTE_TYPE dupNum = 0;
        if constexpr (Mode == MODE_MAX) {
            dupNum = GetDtypeMin<COMPUTE_TYPE>();
        } else {
            dupNum = GetDtypeMax<COMPUTE_TYPE>();
        }
        for (int32_t i = 0; i < uniqueIdNum_; i++) {
            if (ofstLocal(i) < 0) {
                continue;
            }
            LocalTensor<COMPUTE_TYPE> updateSumLocal = updateSumQue_.AllocTensor<COMPUTE_TYPE>();
            AscendC::Duplicate<COMPUTE_TYPE>(updateSumLocal, dupNum, colLen);
            int32_t idRepeatTimes = uniqueIdCountLocal(i);
            for (int32_t k = 0; k < idRepeatTimes; k++) {
                // 兼容splitAfter
                int64_t curOffset = updatesOffset + updatesOriginIdexLocal(idLocation) * afterAxis_;
                event_t eventIdVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
                SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
                WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
                CopyInUpdates(curOffset, 1, colLen); // 每个索引单行搬入update
                event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                SingleColAdd(updateSumLocal, colLen);
                idLocation += 1;
            }

            updateSumQue_.EnQue(updateSumLocal);
            int64_t outOffset = ofstLocal(i) * afterAxis_ + colIdx * afterAxisFactor_;
            LocalTensor<T> outLocal = updateSumQue_.DeQue<T>();
            CopyOutSingleCol(outLocal, outOffset, colLen);
            updateSumQue_.FreeTensor(outLocal);
        }
        updatesOriginIdexQue_.EnQue(updatesOriginIdexLocal);
        uniqueIdCountQue_.EnQue(uniqueIdCountLocal);
        updateSumIdxQue_.EnQue(ofstLocal);
    }

    __aicore__ inline void CopyInAndOutSingleCol(LocalTensor<OFFSET_T> outOffsetLocal, int64_t rowIdx, int64_t colIdx, int64_t rowLen, int64_t colLen)
    {
        for (int32_t i = 0; i < rowLen; i++) {
            if (outOffsetLocal(i) < 0) {
                continue;
            }
            int64_t indicesOfset = GetBlockIdx() * eachCoreIndexCount_ + rowIdx * indicesFactor_ + i;
            int64_t updatesOffset = indicesOfset * afterAxis_ + colIdx * afterAxisFactor_;
            CopyInUpdates(updatesOffset, 1, colLen);
            LocalTensor<T> dataLocal = dataQueue_.DeQue<T>();
            int64_t outOffset = outOffsetLocal(i) * afterAxis_ + colIdx * afterAxisFactor_;
            CopyOutSingleCol(dataLocal, outOffset, colLen);
            dataQueue_.FreeTensor(dataLocal);
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

    __aicore__ inline void ComputeOutSplitAfter(int64_t colIdx, int64_t rowLen, int64_t colLen) // rowLen 不用
    {
        LocalTensor<T> updatesLocal = dataQueue_.DeQue<T>();
        LocalTensor<OFFSET_T> updateSumIdxLocal = updateSumIdxQue_.DeQue<OFFSET_T>();

        LocalTensor<T> updateSumLocal = updateSumQue_.DeQue<T>();
        CopyOutSplitAfter(updateSumIdxLocal, updateSumLocal, uniqueIdNum_, colLen, colIdx);
        updateSumQue_.FreeTensor(updateSumLocal);

        dataQueue_.FreeTensor(updatesLocal);
        updateSumIdxQue_.EnQue(updateSumIdxLocal);
    }

    __aicore__ inline void ComputeOutSplitIndices(int64_t colIdx, int64_t rowLen, int64_t colLen) // rowLen 不用
    {
        LocalTensor<OFFSET_T> updateSumIdxLocal = updateSumIdxQue_.DeQue<OFFSET_T>();
        LocalTensor<T> updatesLocal = dataQueue_.DeQue<T>();

        LocalTensor<T> updateSumLocal = updateSumQue_.DeQue<T>();
        CopyOutSplitIndices(updateSumIdxLocal, updateSumLocal, uniqueIdNum_, colLen, colIdx);
        updateSumQue_.FreeTensor(updateSumLocal);

        dataQueue_.FreeTensor(updatesLocal);
        updateSumIdxQue_.EnQue(updateSumIdxLocal);
    }

    __aicore__ inline void CopyOutSplitAfter(
        LocalTensor<OFFSET_T> ofstLocal, LocalTensor<T> dataLocal, int64_t rowLen, int64_t colLen, int64_t colIdx)
    {
        int64_t colLenAlignSize = ops::CeilAlign(colLen * sizeof(T), UB_AGLIN_VALUE) / sizeof(T);
        for (int64_t i = 0; i < rowLen; i++) {
            if (ofstLocal(i) < 0) {
                continue;
            }
            int64_t rowOfset = ofstLocal(i) * afterAxis_;
            int64_t outOfset = rowOfset + GetBlockIdx() * eachCoreAfterAxisCount_ + colIdx * afterAxisFactor_;
            if constexpr (IsSameType<T, bool>::value) {
                if constexpr (Mode == MODE_MAX) {
                    SetAtomicMax<int8_t>();
                } else {
                    SetAtomicMin<int8_t>();
                }
                CopyOut<bool>(yGm_[outOfset], dataLocal[i * colLenAlignSize], colLen);
                SetAtomicNone();
            } else {
                if constexpr (Mode == MODE_MAX) {
                    SetAtomicMax<T>();
                } else {
                    SetAtomicMin<T>();
                }
                CopyOut<T>(yGm_[outOfset], dataLocal[i * colLenAlignSize], colLen);
                SetAtomicNone();
            }
        }
    }

    __aicore__ inline void CopyIndiceInSplitAfter(int64_t rowIdx, int64_t rowLen)
    {
        LocalTensor<U> indicesLocal = indicesBuf_.Get<U>();
        LocalTensor<OFFSET_T> outOfstLocal = outOfstBuf_.Get<OFFSET_T>();
        LocalTensor<float> dstLocal = maxScoreBuf_.Get<float>();

        int64_t rankSize = indexRankSize_;
        int64_t indicesOfset = rowIdx * indicesFactor_;
        CopyIn<U>(indicesLocal, indicesGm_[indicesOfset * rankSize], rowLen * rankSize);
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        ComputeOutOfset(indicesLocal, outOfstLocal, rowLen, rankSize);
        if constexpr (IsSameType<OFFSET_T, int32_t>::value) {
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

};

} // namespace ScatterNdCommon

#endif
