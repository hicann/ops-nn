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
 * \file copy_pad_impl.h
 * \brief
 */

#ifndef COPY_PAD_IMPL_H
#define COPY_PAD_IMPL_H

#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "kernel_operator.h"

namespace CopyPad
{

using namespace AscendC;

static constexpr int64_t SCATTER_THREHOLD = 4;
const uint32_t COPY_SINGLE_ROW = 0;
const uint32_t SCATTER_ONE_DIM = 1;
const uint32_t SCATTER_TWO_DIMS = 2;
const uint32_t SCATTER_THREE_DIMS = 3;
const uint32_t SCATTER_FOUR_DIMS = 4;
const uint32_t SCATTER_FIVE_DIMS = 5;

const int64_t DIM0 = 0;  // 第一个非连续轴从这里开始
const int64_t DIM1 = 1;
const int64_t DIM2 = 2;
const int64_t DIM3 = 3;
const int64_t DIM4 = 4;
constexpr int32_t ONE = 1;
constexpr int32_t TWO = 2;
constexpr int32_t THREE = 3;
constexpr int32_t FOUR = 4;
constexpr int32_t FIVE = 5;

constexpr int32_t B64 = 8;
constexpr int32_t B8 = 1;

constexpr int32_t SPLIT_COLS = 1;
constexpr int32_t SPLIT_ROWS = 2;
constexpr int32_t SPLIT_DEPTHS = 3;
constexpr int32_t SPLIT_BATCHS = 4;
const uint32_t DEFAULT_MODE_NEG_INF = 0;
const uint32_t DEFAULT_MODE_ZERO = 1;

constexpr uint16_t FLOAT16_NEG_INF = 0xFC00;      // -inf 0xFC00
constexpr uint32_t FLOAT32_NEG_INF = 0xFF800000;  // -inf 0xFF800000
constexpr uint16_t BFLOAT16_NEG_INF = 0xFF80;     // -inf 0xFF80
constexpr int32_t MIN_INT32 = -2147483648;
constexpr int64_t MIN_INT64 = -9223372036854775807LL-1;
constexpr uint8_t MIN_UINT8 = 0;
constexpr int16_t MIN_INT16 = -32768;
constexpr uint16_t MIN_UINT16 = 0;
constexpr int8_t MIN_INT8 = -128;

struct CopyPadShapeInfo {
    uint32_t inSize[FIVE] = {1};
    uint32_t srcStride[FIVE] = {1};
    uint32_t dstStride[FIVE] = {1};
    uint32_t expectStart[FIVE] = {0};
    uint32_t totalSize = 0;
};


template <typename T>
struct indexTypeGet {
    using type =
        typename std::conditional<std::is_same<T, uint32_t>::value, int32_t,
                                  typename std::conditional<std::is_same<T, uint16_t>::value, int16_t,
                                                            typename std::conditional<std::is_same<T, uint64_t>::value,
                                                                                      int64_t, T>::type>::type>::type;
};

template <typename T>
__aicore__ inline constexpr T GetDefaultNegInf()
{
    T negInf = 0;
    if constexpr (std::is_same<T, int32_t>::value) {
        negInf = MIN_INT32;
    } else if constexpr (std::is_same<T, int64_t>::value) {
        negInf = MIN_INT64;
    } else if constexpr (std::is_same<T, uint8_t>::value) {
        negInf = MIN_UINT8;
    } else if constexpr (std::is_same<T, int16_t>::value) {
        negInf = MIN_INT16;
    } else if constexpr (std::is_same<T, int8_t>::value) {
        negInf = MIN_INT8;
    } else if constexpr (std::is_same<T, uint16_t>::value) {
        negInf = MIN_UINT16;
    } else if constexpr (std::is_same<T, float>::value) {
        negInf = *reinterpret_cast<const float*>(&FLOAT32_NEG_INF);
    } else {
        negInf = *reinterpret_cast<const half*>(&FLOAT16_NEG_INF);
    }
    return negInf;
}

template <typename T>
__aicore__ inline uint32_t GetCopyMode(const CopyPadShapeInfo& params, int32_t splitMode)
{
    uint32_t maxScatterElm = platform::GetVRegSize() / sizeof(T);
    if constexpr (sizeof(T) == sizeof(int64_t)) {
        maxScatterElm = platform::GetVRegSize() / sizeof(int32_t);
    }
    uint32_t copyeMode = COPY_SINGLE_ROW;
    if (params.inSize[DIM0] <= SCATTER_THREHOLD * maxScatterElm) {
        copyeMode = SCATTER_ONE_DIM;
    } else {
        copyeMode = COPY_SINGLE_ROW;
    }
    if (params.inSize[DIM0] <= maxScatterElm && splitMode >= SPLIT_ROWS) {
        copyeMode = SCATTER_TWO_DIMS;
    }
    if (params.inSize[DIM0] * params.inSize[DIM1] <= maxScatterElm && splitMode >= SPLIT_DEPTHS) {
        copyeMode = SCATTER_THREE_DIMS;
    }
    if (params.inSize[DIM0] * params.inSize[DIM1] * params.inSize[DIM2] <= maxScatterElm && splitMode >= SPLIT_BATCHS) {
        copyeMode = SCATTER_FOUR_DIMS;
    }
    if (params.inSize[DIM0] * params.inSize[DIM1] * params.inSize[DIM2] * params.inSize[DIM3] <= maxScatterElm &&
        splitMode > SPLIT_BATCHS) {
        copyeMode = SCATTER_FIVE_DIMS;
    }
    return copyeMode;
}

template <typename T, uint32_t DEFAULT_MODE>
__aicore__ inline void CustomDuplicate(__local_mem__ T* dstAddr, uint32_t calNum, int16_t loop)
{
    uint32_t sreg = calNum;
    using RegDstT = typename std::conditional<sizeof(T) == B64, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>,
                                              MicroAPI::RegTensor<T>>::type;
    RegDstT v0;
    // if use vaule batter then mode, like T value
    if constexpr (DEFAULT_MODE == DEFAULT_MODE_NEG_INF) {
        if constexpr (std::is_same<T, bfloat16_t>::value) {
            MicroAPI::Duplicate((MicroAPI::RegTensor<uint16_t>&)v0, BFLOAT16_NEG_INF);
        } else {
            T value = GetDefaultNegInf<T>();
            MicroAPI::Duplicate(v0, value);
        }
    } else {
        MicroAPI::Duplicate(v0, (T)0);
    }
    if constexpr (sizeof(T) == B64) {
        constexpr int16_t repeatElm = TWO * platform::GetVRegSize() / sizeof(T);
        for (int16_t i = 0; i < loop; i++) {
            MicroAPI::MaskReg preg = MicroAPI::UpdateMask<T, MicroAPI::RegTraitNumTwo>(sreg);
            MicroAPI::DataCopy(dstAddr + i * repeatElm, v0, preg);
        }
    } else {
        constexpr int16_t repeatElm = platform::GetVRegSize() / sizeof(T);
        for (int16_t i = 0; i < loop; i++) {
            MicroAPI::MaskReg preg = MicroAPI::UpdateMask<T>(sreg);
            MicroAPI::AddrReg offset = MicroAPI::CreateAddrReg<T>(i, repeatElm);
            MicroAPI::DataCopy(dstAddr, v0, offset, preg);
        }
    }
}

template <typename T>
__aicore__ inline void CustomCopy(const __local_mem__ T* dstAddr, const __local_mem__ T* srcAddr,
                                  const CopyPadShapeInfo& params, int16_t loopCols, int16_t tailCols,
                                  int16_t repeatElm)
{
    using RegDstT = typename std::conditional<sizeof(T) == B64, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>,
                                              MicroAPI::RegTensor<T>>::type;
    RegDstT v0;
    MicroAPI::UnalignReg u0;

    for (int16_t i = 0; i < params.inSize[DIM3]; i++) {
        auto srcAddr1 = (__local_mem__ T*)srcAddr + i * params.srcStride[DIM3];
        auto dstAddr1 = (__local_mem__ T*)dstAddr +  i * params.dstStride[DIM3];
        for (int16_t d = 0; d < params.inSize[DIM2]; d++) {
            auto srcAddr2 = srcAddr1 +  d * params.srcStride[DIM2];
            auto dstAddr2 = dstAddr1 + d * params.dstStride[DIM2];
            for (int16_t j = 0; j < params.inSize[DIM1]; j++) {
                auto curSrcAddr = srcAddr2 + j * params.srcStride[DIM1];
                auto curDstAddr = dstAddr2 + j * params.dstStride[DIM1];
                for (int16_t k = 0; k < loopCols; k++) {
                    MicroAPI::DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(v0, curSrcAddr, repeatElm);
                    MicroAPI::DataCopyUnAlign(curDstAddr, v0, u0, repeatElm);
                }
                MicroAPI::DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(v0, curSrcAddr, repeatElm);
                MicroAPI::DataCopyUnAlign(curDstAddr, v0, u0, tailCols);
                MicroAPI::DataCopyUnAlignPost(curDstAddr, u0, 0);
            }
        }
    }
}

template <typename T, typename U>
__aicore__ inline void CustomCopyByScatterSingleRow(const __local_mem__ T* dstAddr, const __local_mem__ T* srcAddr,
                                                    const CopyPadShapeInfo& params, int16_t loopCols, int16_t elems,
                                                    int16_t repeatElm)
{
    using RegDstT = typename std::conditional<sizeof(T) == B64, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>,
                                              MicroAPI::RegTensor<T>>::type;
    RegDstT v0;
    MicroAPI::RegTensor<U> sIndex;
    MicroAPI::MaskReg preg;
    using regType = typename indexTypeGet<U>::type;
    MicroAPI::Arange((MicroAPI::RegTensor<regType>&)sIndex, 0);
    for (int16_t i = 0; i < params.inSize[DIM3]; i++) {
        auto dstAddr2 = (__local_mem__ T*)dstAddr + i * params.dstStride[DIM3];
        auto srcAddr0 = (__local_mem__ T*)srcAddr + i * params.srcStride[DIM3];
        for (int16_t d = 0; d < params.inSize[DIM2]; d++) {
            auto dstAddr3 = dstAddr2 + d * params.dstStride[DIM2];
            auto srcAddr1 = srcAddr0 + d * params.srcStride[DIM2];
            uint32_t sreg = elems;
            for (int16_t j = 0; j < loopCols; j++) {
                auto curDstAddr = dstAddr3 + j * repeatElm;
                auto curSrcAddr = srcAddr1 + j * repeatElm;
                if constexpr (sizeof(T) == B64) {
                    preg = MicroAPI::UpdateMask<U, MicroAPI::RegTraitNumTwo>(sreg);
                } else {
                    preg = MicroAPI::UpdateMask<U>(sreg);
                }
                for (int16_t k = 0; k < params.inSize[DIM1]; k++) {
                    if constexpr (sizeof(T) == B8) {
                        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK_B8>(
                            v0, curSrcAddr + k * params.srcStride[DIM1]);
                    } else {
                        MicroAPI::DataCopy(v0, curSrcAddr + k * params.srcStride[DIM1]);
                    }
                    MicroAPI::DataCopyScatter(curDstAddr + k * params.dstStride[DIM1], v0, sIndex, preg);
                }
            }
        }
    }
}

template <typename T, typename U>
__aicore__ inline void CustomCopyByScatterTwoDims(const __local_mem__ T* dstAddr, const __local_mem__ T* srcAddr,
                                                  MicroAPI::RegTensor<U> index, const CopyPadShapeInfo& params,
                                                  uint32_t srcStrideDim2, uint32_t dstStrideDim2,
                                                  int16_t loopRows, uint32_t repeatElm, uint32_t tailElm)
{
    using RegDstT = typename std::conditional<sizeof(T) == B64, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>,
                                              MicroAPI::RegTensor<T>>::type;
    RegDstT vd1;
    MicroAPI::RegTensor<U> v1, v2, v3, v4, v5, v6, v7, v8;
    MicroAPI::RegTensor<U> gIndex;
    uint32_t sreg = repeatElm;
    uint32_t tailSreg = tailElm;
    MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<U, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg preg;
    MicroAPI::MaskReg tailPreg;
    if constexpr (sizeof(T) == B64) {
        preg = MicroAPI::UpdateMask<T, MicroAPI::RegTraitNumTwo>(sreg);
        tailPreg = MicroAPI::UpdateMask<T, MicroAPI::RegTraitNumTwo>(tailSreg);
    } else {
        preg = MicroAPI::UpdateMask<U>(sreg);
        tailPreg = MicroAPI::UpdateMask<U>(tailSreg);
    }
    MicroAPI::RegTensor<U> vd0;
    using regType = typename indexTypeGet<U>::type;
    MicroAPI::Arange((MicroAPI::RegTensor<regType>&)gIndex, 0);
    __local_mem__ T* curDstAddr = (__local_mem__ T*)dstAddr;
    for (uint32_t k = 0; k < params.inSize[DIM4]; k++) {
        MicroAPI::Adds(v1, index, k * params.dstStride[DIM4], maskAll);
        MicroAPI::Adds(v2, gIndex, k * params.srcStride[DIM4], maskAll);
        for (int16_t i = 0; i < params.inSize[DIM3]; i++) {
            MicroAPI::Adds(v3, v1, i * params.dstStride[DIM3], maskAll);
            MicroAPI::Adds(v4, v2, i * params.srcStride[DIM3], maskAll);
            for (int16_t d = 0; d < params.inSize[DIM2]; d++) {
                MicroAPI::Adds(v5, v3, d * params.dstStride[DIM2], maskAll);
                MicroAPI::Adds(v6, v4, d * params.srcStride[DIM2], maskAll);
                for (int16_t j = 0; j < loopRows; j++) {
                    MicroAPI::Adds(v7, v5, j * dstStrideDim2, preg);
                    MicroAPI::Adds(v8, v6, j * srcStrideDim2, preg);
                    MicroAPI::DataCopyGather(vd1, srcAddr, v8, preg);
                    if constexpr (sizeof(T) == B8) {
                        MicroAPI::DataCopyScatter(curDstAddr, (MicroAPI::RegTensor<T>&)vd1, v7, preg);
                    } else {
                        MicroAPI::DataCopyScatter(curDstAddr, vd1, v7, preg);
                    }
                }
                MicroAPI::Adds(v7, v5, loopRows * dstStrideDim2, tailPreg);
                MicroAPI::Adds(v8, v6, loopRows * srcStrideDim2, tailPreg);
                MicroAPI::DataCopyGather(vd1, srcAddr, v8, tailPreg);
                if constexpr (sizeof(T) == B8) {
                    MicroAPI::DataCopyScatter(curDstAddr, (MicroAPI::RegTensor<T>&)vd1, v7, tailPreg);
                } else {
                    MicroAPI::DataCopyScatter(curDstAddr, vd1, v7, tailPreg);
                }
            }
        }
    }
}

template <typename T, typename U>
__aicore__ inline void CustomCopyByScatterThreeDims(const __local_mem__ T* dstAddr, const __local_mem__ T* srcAddr,
                                                    MicroAPI::RegTensor<U> index, const CopyPadShapeInfo& params,
                                                    uint32_t srcStrideDim3, uint32_t dstStrideDim3,
                                                    int16_t loopDeps, uint32_t repeatElm, uint32_t tailElm)
{
    using RegDstT = typename std::conditional<sizeof(T) == B64, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>,
                                              MicroAPI::RegTensor<T>>::type;
    RegDstT vd1;
    MicroAPI::RegTensor<U> v1, v2, v3, v4, v5, v6;
    MicroAPI::RegTensor<U> gIndex;
    uint32_t sreg = repeatElm;
    uint32_t tailSreg = tailElm;
    MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<U, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg preg;
    MicroAPI::MaskReg tailPreg;
    if constexpr (sizeof(T) == B64) {
        preg = MicroAPI::UpdateMask<T, MicroAPI::RegTraitNumTwo>(sreg);
        tailPreg = MicroAPI::UpdateMask<T, MicroAPI::RegTraitNumTwo>(tailSreg);
    } else {
        preg = MicroAPI::UpdateMask<U>(sreg);
        tailPreg = MicroAPI::UpdateMask<U>(tailSreg);
    }
    using regType = typename indexTypeGet<U>::type;
    MicroAPI::RegTensor<U> vd0;
    MicroAPI::Arange((MicroAPI::RegTensor<regType>&)gIndex, 0);
    __local_mem__ T* curDstAddr = (__local_mem__ T*)dstAddr;
    for (int16_t i = 0; i < params.inSize[DIM4]; i++) {
        MicroAPI::Adds(v1, index, i * params.dstStride[DIM4], maskAll);
        MicroAPI::Adds(v2, gIndex, i * params.srcStride[DIM4], maskAll);
        for (int16_t d = 0; d < params.inSize[DIM3]; d++) {
            MicroAPI::Adds(v3, v1, d * params.dstStride[DIM3], maskAll);
            MicroAPI::Adds(v4, v2, d * params.srcStride[DIM3], maskAll);
            for (int16_t j = 0; j < loopDeps; j++) {
                MicroAPI::Adds(v5, v3, j * dstStrideDim3, preg);
                MicroAPI::Adds(v6, v4, j * srcStrideDim3, preg);
                MicroAPI::DataCopyGather(vd1, srcAddr, v6, preg);
                if constexpr (sizeof(T) == B8) {
                    MicroAPI::DataCopyScatter(curDstAddr, (MicroAPI::RegTensor<T>&)vd1, v5, preg);
                } else {
                    MicroAPI::DataCopyScatter(curDstAddr, vd1, v5, preg);
                }
            }
            MicroAPI::Adds(v5, v3, loopDeps * dstStrideDim3, tailPreg);
            MicroAPI::Adds(v6, v4, loopDeps * srcStrideDim3, tailPreg);
            MicroAPI::DataCopyGather(vd1, srcAddr, v6, tailPreg);
            if constexpr (sizeof(T) == B8) {
                MicroAPI::DataCopyScatter(curDstAddr, (MicroAPI::RegTensor<T>&)vd1, v5, tailPreg);
            } else {
                MicroAPI::DataCopyScatter(curDstAddr, vd1, v5, tailPreg);
            }
        }
    }
}

template <typename T, typename U>
__aicore__ inline void CustomCopyByScatterFourDims(const __local_mem__ T* dstAddr, const __local_mem__ T* srcAddr,
                                                   MicroAPI::RegTensor<U> index, const CopyPadShapeInfo& params,
                                                   uint32_t srcStrideDim4, uint32_t dstStrideDim4,
                                                   int16_t loopDim4, uint32_t repeatElm, uint32_t tailElm)
{
    using RegDstT = typename std::conditional<sizeof(T) == B64, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>,
                                              MicroAPI::RegTensor<T>>::type;
    RegDstT vd1;
    MicroAPI::RegTensor<U> v1, v2, v3, v4, v5, v6;
    MicroAPI::RegTensor<U> gIndex;
    uint32_t sreg = repeatElm;
    uint32_t tailSreg = tailElm;
    MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<U, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg preg;
    MicroAPI::MaskReg tailPreg;
    if constexpr (sizeof(T) == B64) {
        preg = MicroAPI::UpdateMask<T, MicroAPI::RegTraitNumTwo>(sreg);
        tailPreg = MicroAPI::UpdateMask<T, MicroAPI::RegTraitNumTwo>(tailSreg);
    } else {
        preg = MicroAPI::UpdateMask<U>(sreg);
        tailPreg = MicroAPI::UpdateMask<U>(tailSreg);
    }
    using regType = typename indexTypeGet<U>::type;
    MicroAPI::RegTensor<U> vd0;
    MicroAPI::Arange((MicroAPI::RegTensor<regType>&)gIndex, 0);
    __local_mem__ T* curDstAddr = (__local_mem__ T*)dstAddr;
    for (int16_t i = 0; i < params.inSize[DIM4]; i++) {
        MicroAPI::Adds(v1, index, i * params.dstStride[DIM4], maskAll);
        MicroAPI::Adds(v2, gIndex, i * params.srcStride[DIM4], maskAll);
        for (int16_t j = 0; j < loopDim4; j++) {
            MicroAPI::Adds(v3, v1, j * dstStrideDim4, preg);
            MicroAPI::Adds(v4, v2, j * srcStrideDim4, preg);
            MicroAPI::DataCopyGather(vd1, srcAddr, v4, preg);
            if constexpr (sizeof(T) == B8) {
                MicroAPI::DataCopyScatter(curDstAddr, (MicroAPI::RegTensor<T>&)vd1, v3, preg);
            } else {
                MicroAPI::DataCopyScatter(curDstAddr, vd1, v3, preg);
            }
        }
        MicroAPI::Adds(v3, v1, loopDim4 * dstStrideDim4, tailPreg);
        MicroAPI::Adds(v4, v2, loopDim4 * srcStrideDim4, tailPreg);
        MicroAPI::DataCopyGather(vd1, srcAddr, v4, tailPreg);
        if constexpr (sizeof(T) == B8) {
            MicroAPI::DataCopyScatter(curDstAddr, (MicroAPI::RegTensor<T>&)vd1, v3, tailPreg);
        } else {
            MicroAPI::DataCopyScatter(curDstAddr, vd1, v3, tailPreg);
        }
    }
}

template <typename T, typename U>
__aicore__ inline void CustomCopyByScatterFiveDims(const __local_mem__ T* dstAddr, const __local_mem__ T* srcAddr,
                                                   MicroAPI::RegTensor<U> index, const CopyPadShapeInfo& params,
                                                   uint32_t srcStrideDim5, uint32_t dstStrideDim5,
                                                   int16_t loopDim5, uint32_t repeatElm, uint32_t tailElm)
{
    using RegDstT = typename std::conditional<sizeof(T) == B64, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>,
                                              MicroAPI::RegTensor<T>>::type;
    RegDstT vd1;
    MicroAPI::RegTensor<U> v1, v2, v3, v4, v5, v6;
    MicroAPI::RegTensor<U> gIndex;
    uint32_t sreg = repeatElm;
    uint32_t tailSreg = tailElm;
    MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<U, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg preg;
    MicroAPI::MaskReg tailPreg;
    if constexpr (sizeof(T) == B64) {
        preg = MicroAPI::UpdateMask<T, MicroAPI::RegTraitNumTwo>(sreg);
        tailPreg = MicroAPI::UpdateMask<T, MicroAPI::RegTraitNumTwo>(tailSreg);
    } else {
        preg = MicroAPI::UpdateMask<U>(sreg);
        tailPreg = MicroAPI::UpdateMask<U>(tailSreg);
    }
    using regType = typename indexTypeGet<U>::type;
    MicroAPI::RegTensor<U> vd0;
    MicroAPI::Arange((MicroAPI::RegTensor<regType>&)gIndex, 0);
    __local_mem__ T* curDstAddr = (__local_mem__ T*)dstAddr;
    for (int16_t j = 0; j < loopDim5; j++) {  // 用dim2 计算loopDeps
        MicroAPI::Adds(v1, index, j * dstStrideDim5, preg);
        MicroAPI::Adds(v2, gIndex, j * srcStrideDim5, preg);
        MicroAPI::DataCopyGather(vd1, srcAddr, v2, preg);
        if constexpr (sizeof(T) == B8) {
            MicroAPI::DataCopyScatter(curDstAddr, (MicroAPI::RegTensor<T>&)vd1, v1, preg);
        } else {
            MicroAPI::DataCopyScatter(curDstAddr, vd1, v1, preg);
        }
    }
    MicroAPI::Adds(v1, index, loopDim5 * dstStrideDim5, tailPreg);
    MicroAPI::Adds(v2, gIndex, loopDim5 * srcStrideDim5, tailPreg);
    MicroAPI::DataCopyGather(vd1, srcAddr, v2, tailPreg);
    if constexpr (sizeof(T) == B8) {
        MicroAPI::DataCopyScatter(curDstAddr, (MicroAPI::RegTensor<T>&)vd1, v1, tailPreg);
    } else {
        MicroAPI::DataCopyScatter(curDstAddr, vd1, v1, tailPreg);
    }
}

template <typename T, typename U>
__aicore__ inline void ScatterOneDim(const __local_mem__ T* dstAddr, const __local_mem__ T* srcAddr,
                                     const CopyPadShapeInfo& params)
{
    constexpr uint32_t repeatElm = platform::GetVRegSize() / sizeof(U);
    int16_t realColsElm = params.inSize[DIM0];
    int16_t preColsLoop = static_cast<int16_t>((realColsElm + repeatElm - 1) / repeatElm);
    uint32_t dstOffset = params.expectStart[DIM4] * params.dstStride[DIM4] +
                     params.expectStart[DIM3] * params.dstStride[DIM3] +
                     params.expectStart[DIM2] * params.dstStride[DIM2] +
                     params.expectStart[DIM1] * params.dstStride[DIM1] + params.expectStart[DIM0];
    for (int i = 0; i < params.inSize[DIM4]; i++) {
        __local_mem__ T* curSrcAddr = (__local_mem__ T*)srcAddr + i * params.srcStride[DIM4];
        __local_mem__ T* curDstAddr =
            (__local_mem__ T*)dstAddr + i * params.dstStride[DIM4] + dstOffset;
        __VEC_SCOPE__
        {
            CustomCopyByScatterSingleRow<T, U>(curDstAddr, curSrcAddr, params, preColsLoop, realColsElm, repeatElm);
        }
    }
}

template <typename T, typename U>
__aicore__ inline void ScatterTwoDims(const __local_mem__ T* dstAddr, const __local_mem__ T* srcAddr,
                                      LocalTensor<U>& indexLocal, const CopyPadShapeInfo& params,
                                      uint32_t maxRepeatElms)
{
    uint32_t onceCopyRow = maxRepeatElms / params.inSize[DIM0];
    uint32_t dstOffset = params.expectStart[DIM4] * params.dstStride[DIM4] +
                         params.expectStart[DIM3] * params.dstStride[DIM3] +
                         params.expectStart[DIM2] * params.dstStride[DIM2] +
                         params.expectStart[DIM1] * params.dstStride[DIM1] + params.expectStart[DIM0];

    uint32_t loopRows = params.inSize[DIM1] / onceCopyRow;
    uint32_t repeatElm = onceCopyRow * params.inSize[DIM0];
    uint32_t tailRepeatElm = (params.inSize[DIM1] - loopRows * onceCopyRow) * params.inSize[DIM0];
    uint32_t srcStrideDim2 = onceCopyRow * params.inSize[DIM0];
    uint32_t dstStrideDim2 = onceCopyRow * params.dstStride[DIM1];
    auto indexAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    auto curDstAddr = dstAddr + dstOffset;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<U> v0;
        MicroAPI::DataCopy(v0, indexAddr);
        CustomCopyByScatterTwoDims<T, U>(curDstAddr, srcAddr, v0, params, srcStrideDim2, dstStrideDim2,
                                         loopRows, repeatElm, tailRepeatElm);
    }
}

template <typename T, typename U>
__aicore__ inline void ScatterThreeDims(const __local_mem__ T* dstAddr, const __local_mem__ T* srcAddr,
                                        LocalTensor<U>& indexLocal, const CopyPadShapeInfo& params,
                                        uint32_t maxRepeatElms)
{
    uint32_t oneDepElms = params.inSize[DIM0] * params.inSize[DIM1];
    uint32_t onceCopyDep = maxRepeatElms / oneDepElms;
    uint32_t dstOffset = params.expectStart[DIM4] * params.dstStride[DIM4] +
                         params.expectStart[DIM3] * params.dstStride[DIM3] +
                         params.expectStart[DIM2] * params.dstStride[DIM2] +
                         params.expectStart[DIM1] * params.dstStride[DIM1] + params.expectStart[DIM0];
    uint32_t loopDeps = params.inSize[DIM2] / onceCopyDep;
    uint32_t repeatElm = onceCopyDep * oneDepElms;
    uint32_t tailRepeatElm = (params.inSize[DIM2] - loopDeps * onceCopyDep) * oneDepElms;
    uint32_t srcStrideDim3 = onceCopyDep * oneDepElms;
    uint32_t dstStrideDim3 = onceCopyDep * params.dstStride[DIM2];
    auto indexAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    auto curDstAddr = dstAddr + dstOffset;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<U> v0;
        MicroAPI::DataCopy(v0, indexAddr);
        CustomCopyByScatterThreeDims<T, U>(curDstAddr, srcAddr, v0, params, srcStrideDim3, dstStrideDim3,
                                           loopDeps, repeatElm, tailRepeatElm);
    }
}

template <typename T, typename U>
__aicore__ inline void ScatterFourDims(const __local_mem__ T* dstAddr, const __local_mem__ T* srcAddr,
                                       LocalTensor<U>& indexLocal, const CopyPadShapeInfo& params,
                                       uint32_t maxRepeatElms)
{
    uint32_t oneDimElms = params.inSize[DIM0] * params.inSize[DIM1] * params.inSize[DIM2];
    uint32_t onceCopyDim4 = maxRepeatElms / oneDimElms;
    uint32_t dstOffset = params.expectStart[DIM4] * params.dstStride[DIM4] +
                         params.expectStart[DIM3] * params.dstStride[DIM3] +
                         params.expectStart[DIM2] * params.dstStride[DIM2] +
                         params.expectStart[DIM1] * params.dstStride[DIM1] + params.expectStart[DIM0];
    uint32_t loopDim4 = params.inSize[DIM3] / onceCopyDim4;
    uint32_t repeatElm = onceCopyDim4 * oneDimElms;
    uint32_t tailRepeatElm = (params.inSize[DIM3] - loopDim4 * onceCopyDim4) * oneDimElms;
    uint32_t srcStrideDim4 = onceCopyDim4 * oneDimElms;
    uint32_t dstStrideDim4 = onceCopyDim4 * params.dstStride[DIM3];
    auto indexAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    auto curDstAddr = dstAddr + dstOffset;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<U> v0;
        MicroAPI::DataCopy(v0, indexAddr);
        CustomCopyByScatterFourDims<T, U>(curDstAddr, srcAddr, v0, params, srcStrideDim4, dstStrideDim4,
                                          loopDim4, repeatElm, tailRepeatElm);
    }
}

template <typename T, typename U>
__aicore__ inline void ScatterFiveDims(const __local_mem__ T* dstAddr, const __local_mem__ T* srcAddr,
                                       LocalTensor<U>& indexLocal, const CopyPadShapeInfo& params,
                                       uint32_t maxRepeatElms)
{
    uint32_t oneDimElms = params.inSize[DIM0] * params.inSize[DIM1] * params.inSize[DIM2] * params.inSize[DIM3];
    uint32_t onceCopyDim5 = maxRepeatElms / oneDimElms;
    uint32_t dstOffset = params.expectStart[DIM4] * params.dstStride[DIM4] +
                         params.expectStart[DIM3] * params.dstStride[DIM3] +
                         params.expectStart[DIM2] * params.dstStride[DIM2] +
                         params.expectStart[DIM1] * params.dstStride[DIM1] + params.expectStart[DIM0];
    uint32_t loopDim5 = params.inSize[DIM4] / onceCopyDim5;
    uint32_t repeatElm = onceCopyDim5 * oneDimElms;
    uint32_t tailRepeatElm = (params.inSize[DIM4] - loopDim5 * onceCopyDim5) * oneDimElms;
    uint32_t srcStrideDim5 = onceCopyDim5 * oneDimElms;
    uint32_t dstStrideDim5 = onceCopyDim5 * params.dstStride[DIM4];
    auto indexAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    auto curDstAddr = dstAddr + dstOffset;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<U> v0;
        MicroAPI::DataCopy(v0, indexAddr);
        CustomCopyByScatterFiveDims<T, U>(curDstAddr, srcAddr, v0, params, srcStrideDim5, dstStrideDim5,
                                          loopDim5, repeatElm, tailRepeatElm);
    }
}

template <typename T, typename U>
__aicore__ inline void CopySingleDim(const __local_mem__ T* dstAddr, const __local_mem__ T* srcAddr,
                                     const CopyPadShapeInfo& params)
{
    constexpr uint32_t repeatElm = platform::GetVRegSize() / sizeof(T);
    int16_t realColsElm = params.inSize[DIM0];
    int16_t preColsLoop = static_cast<int16_t>(realColsElm / repeatElm);
    uint32_t tailPreCols = realColsElm - preColsLoop * repeatElm;
    uint32_t dstOffset = params.expectStart[DIM4] * params.dstStride[DIM4] +
                     params.expectStart[DIM3] * params.dstStride[DIM3] +
                     params.expectStart[DIM2] * params.dstStride[DIM2] +
                     params.expectStart[DIM1] * params.dstStride[DIM1] + params.expectStart[DIM0];
    for (int i = 0; i < params.inSize[DIM4]; i++) {
        __local_mem__ T* curSrcAddr = (__local_mem__ T*)srcAddr + i * params.srcStride[DIM4];
        __local_mem__ T* curDstAddr =
            (__local_mem__ T*)dstAddr + i * params.dstStride[DIM4] + dstOffset;
        __VEC_SCOPE__
        {
            CustomCopy(curDstAddr, curSrcAddr, params, preColsLoop, tailPreCols, repeatElm);
        }
    }
}

template <typename T, typename U, uint32_t DEFAULT_MODE>
__aicore__ inline void CopyAndPad(LocalTensor<T>& inLocal, LocalTensor<T>& outLocal, LocalTensor<U>& scatterLocal,
                                  const CopyPadShapeInfo& params, uint32_t copyMode)  // copy row copy 自己算
{
    __local_mem__ T* inLocalAddr = (__local_mem__ T*)inLocal.GetPhyAddr();
    __local_mem__ T* outLocalAddr = (__local_mem__ T*)outLocal.GetPhyAddr();
    uint32_t dupRepeatElm = platform::GetVRegSize() / sizeof(T);
    if constexpr (sizeof(T) == sizeof(int64_t)) {
        dupRepeatElm = platform::GetVRegSize() / sizeof(int32_t);
    }

    uint32_t totalDupNum = params.totalSize;
    int16_t dupLoop = static_cast<int16_t>((totalDupNum + dupRepeatElm - 1) / dupRepeatElm);
    __VEC_SCOPE__
    {
        CustomDuplicate<T, DEFAULT_MODE>(outLocalAddr, totalDupNum, dupLoop);
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_STORE>();
    }
    if (copyMode == SCATTER_ONE_DIM) {
        ScatterOneDim<T, U>(outLocalAddr, inLocalAddr, params);
    } else if (copyMode == SCATTER_TWO_DIMS) {
        ScatterTwoDims<T, U>(outLocalAddr, inLocalAddr, scatterLocal, params, dupRepeatElm);
    } else if (copyMode == SCATTER_THREE_DIMS) {
        ScatterThreeDims<T, U>(outLocalAddr, inLocalAddr, scatterLocal, params, dupRepeatElm);
    } else if (copyMode == SCATTER_FOUR_DIMS) {
        ScatterFourDims<T, U>(outLocalAddr, inLocalAddr, scatterLocal, params, dupRepeatElm);
    } else if (copyMode == SCATTER_FIVE_DIMS) {
        ScatterFiveDims<T, U>(outLocalAddr, inLocalAddr, scatterLocal, params, dupRepeatElm);
    } else {
        CopySingleDim<T, U>(outLocalAddr, inLocalAddr, params);
    }
}

// eg: scatter and copy pad
/**
    CopyPadShapeInfo shapeInfo = {
        {channels, cols, rows, n, 1},
        {1, channels, cols * channels, rows * cols * channels, 1},
        {1, channels, inColElms, inRows * inColsElms, 1},
        {0, expColStart, expRowStart, 0, 0},
        totalSize,
    };
    uint32_t copyMode = GetCopyMode<T>(shapeInfo, splitMode)
    GenScatterIndex<U, FOUR>(shapeInfo, indexLocal);
    CopyAndPad<M, U, DEFAULT_MODE_NEG_INF>(inLocal, outLocal, indexLocal, shapeInfo, copyMode);

**/

}  // namespace CopyPad

#endif  // COPY_PAD_IMPL_H