/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file embedding_simd_base.h
 * \brief SIMD base kernel template for Embedding operator.
 *        Provides common buffer init, copy in/out and utility functions
 *        shared by contiguous (EmbeddingSimdTwoDim) and non-contiguous
 *        (EmbeddingSimdNoContiguous) SIMD kernels.
 */
#ifndef EMBEDDING_SIMD_BASE_H
#define EMBEDDING_SIMD_BASE_H

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_vec_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"

namespace Embedding {
using namespace AscendC;

constexpr int32_t BUFFER_NUM_SIMD = 2;

template <typename INDICES_T>
class EmbeddingSimdBase {
public:
    GlobalTensor<int8_t> xGm_;
    GlobalTensor<INDICES_T> indicesGm_;
    GlobalTensor<int8_t> yGm_;
    TPipe* pipe_ = nullptr;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM_SIMD> inQueue_;
    TBuf<QuePosition::VECCALC> indexBuf_;

    int64_t indicesOffsetBase_ = -1;
    int64_t curIndexSize_ = 0;

    __aicore__ inline EmbeddingSimdBase() = default;

    __aicore__ inline void InitBaseBuffer(TPipe* pipe, int64_t maxElement, int64_t dtypeSize, int64_t indiceFactor,
                                          GM_ADDR x, GM_ADDR indices, GM_ADDR y)
    {
        pipe_ = pipe;
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(x));
        indicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ INDICES_T*>(indices));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(y));
        pipe_->InitBuffer(inQueue_, BUFFER_NUM_SIMD, maxElement * dtypeSize);
        pipe_->InitBuffer(indexBuf_, indiceFactor * sizeof(INDICES_T));
    }

    template <typename T>
    __aicore__ inline void CopyInContiguous(LocalTensor<T> xLocal, GlobalTensor<T> xGm, int64_t offset, uint32_t nBurst,
                                            uint32_t copyLen)
    {
        DataCopyPadExtParams<T> dataCopyPadExtParams;
        dataCopyPadExtParams.isPad = false;
        dataCopyPadExtParams.leftPadding = 0;
        dataCopyPadExtParams.rightPadding = 0;
        dataCopyPadExtParams.paddingValue = 0;

        DataCopyExtParams dataCoptExtParams;
        dataCoptExtParams.blockCount = nBurst;
        dataCoptExtParams.blockLen = copyLen * sizeof(T);
        dataCoptExtParams.srcStride = 0;
        dataCoptExtParams.dstStride = 0;
        DataCopyPad(xLocal, xGm[offset], dataCoptExtParams, dataCopyPadExtParams);
    }

    template <typename T>
    __aicore__ inline void CopyInNoContiguous(LocalTensor<T> dstTensor, GlobalTensor<T>& srcTensor, int64_t offset,
                                              uint32_t dataCount, uint32_t dataLen, uint32_t srcStride)
    {
        DataCopyExtParams dataCoptExtParams;
        dataCoptExtParams.blockCount = dataCount;
        dataCoptExtParams.blockLen = dataLen * sizeof(T);
        dataCoptExtParams.srcStride = srcStride * sizeof(T);
        dataCoptExtParams.dstStride = 0;
        DataCopyPadExtParams<T> dataCopyPadExtParams;
        dataCopyPadExtParams.isPad = false;
        dataCopyPadExtParams.leftPadding = 0;
        dataCopyPadExtParams.rightPadding = 0;
        dataCopyPadExtParams.paddingValue = 0;
        DataCopyPad<T, PaddingMode::Compact>(dstTensor, srcTensor[offset], dataCoptExtParams, dataCopyPadExtParams);
    }

    __aicore__ inline void CopyOut(int64_t offset, uint32_t nBurst, uint32_t copyLen)
    {
        DataCopyExtParams dataCoptExtParams;
        dataCoptExtParams.blockCount = nBurst;
        dataCoptExtParams.blockLen = copyLen;
        dataCoptExtParams.srcStride = 0;
        dataCoptExtParams.dstStride = 0;
        LocalTensor<int8_t> yLocal = inQueue_.DeQue<int8_t>();
        event_t eventIdVtoMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVtoMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVtoMTE3);
        DataCopyPad(yGm_[offset], yLocal, dataCoptExtParams);
        inQueue_.FreeTensor(yLocal);
    }

    __aicore__ inline void GetYStartYEnd(int64_t& yStart, int64_t& yEnd, int64_t blockFactor, int64_t tailBlockFactor)
    {
        if (GetBlockIdx() < tailBlockFactor) {
            yStart = (blockFactor + 1) * GetBlockIdx();
            yEnd = yStart + blockFactor + 1;
        } else {
            yStart = blockFactor * GetBlockIdx() + tailBlockFactor;
            yEnd = yStart + blockFactor;
        }
    }

    __aicore__ inline int64_t GetColsAlign(int64_t innerSize, int64_t dtypeSize)
    {
        return Ops::Base::CeilAlign(innerSize * dtypeSize, static_cast<int64_t>(Ops::Base::GetUbBlockSize()));
    }
};
} // namespace Embedding
#endif // EMBEDDING_SIMD_BASE_H
