/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 /*!
 * \file concat_offset_simt.h
 * \brief
 */

#ifndef CONCAT_OFFSET_SIMT_H
#define CONCAT_OFFSET_SIMT_H

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "concat_offset_struct.h"
#include "kernel_operator.h"

#ifdef __DAV_FPGA__
constexpr uint32_t THREAD_NUM_LAUNCH_BOUND = 512;
#else
constexpr uint32_t THREAD_NUM_LAUNCH_BOUND = 2048;
#endif

constexpr uint32_t TBUF_SIZE = 8192;

namespace ConcatOffset {
using namespace AscendC;


template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_LAUNCH_BOUND) inline void SimtComputer(GM_ADDR x, GM_ADDR y, uint32_t m0, uint32_t shift0, __local_mem__ T* tmpLocal,
                                                     uint32_t perTensorShapeSize, uint32_t concatDim, uint32_t needCalNum)
{
    for (uint32_t curCalIdx = static_cast<uint32_t>(Simt::GetThreadIdx()); curCalIdx < needCalNum; curCalIdx += static_cast<uint32_t>(Simt::GetThreadNum()))
    {
        uint32_t curCalIdx_y = Simt::UintDiv(curCalIdx, m0, shift0);  // threadIdx_y
        uint32_t curCalIdx_x = curCalIdx - perTensorShapeSize * curCalIdx_y; // threadIdx_x
        if (curCalIdx_x == 0) {
            __gm__ uint64_t* xDataAddr = reinterpret_cast<__gm__ uint64_t*>(x);
            uint64_t xDataPtrOffset = *xDataAddr;
            __gm__ uint64_t* xDataPtr = xDataAddr + (xDataPtrOffset >> 3);

            tmpLocal[curCalIdx_y] = reinterpret_cast<__gm__ T*>(*(xDataPtr + curCalIdx_y))[concatDim];
        }
        Simt::ThreadBarrier();
        T value = 0;
        if (curCalIdx_x == concatDim) {
            for (int32_t i = 0; i < curCalIdx_y; ++i) {
                value += tmpLocal[i];
            }
        }
        __gm__ uint64_t* yDataAddr = reinterpret_cast<__gm__ uint64_t*>(y);
        uint64_t yDataPtrOffset = *yDataAddr;
        __gm__ uint64_t* yDataPtr = yDataAddr + (yDataPtrOffset >> 3);

        reinterpret_cast<__gm__ T*>(*(yDataPtr + curCalIdx_y))[curCalIdx_x] = value;
    }
}


template <typename T>
class ConcatOffsetSimt {
public:
  __aicore__ inline ConcatOffsetSimt(){};
  __aicore__ inline void Init(const ConcatOffsetTilingData* tilingData);
  __aicore__ inline void Process(GM_ADDR x, GM_ADDR y);

private:
    TPipe pipe_;
    TBuf<QuePosition::VECCALC> tmpBuf_;
    const ConcatOffsetTilingData* tilingData_ = nullptr;
};


template <typename T>
__aicore__ inline void ConcatOffsetSimt<T>::Init(const ConcatOffsetTilingData* tilingData) 
{
    tilingData_ = tilingData;
    pipe_.InitBuffer(tmpBuf_, TBUF_SIZE);
}


template <typename T>
__aicore__ inline void ConcatOffsetSimt<T>::Process(GM_ADDR x, GM_ADDR y)
{
    int32_t blockIdx = static_cast<int32_t>(GetBlockIdx());
    if (blockIdx >= 1) {
        return;
    }

    LocalTensor<T> tmpLocal = tmpBuf_.Get<T>();
    uint32_t perTensorShapeSize = static_cast<uint32_t>(tilingData_->perTensorShapeSize);

    // fast division
    uint32_t m0 = 0;
    uint32_t shift0 = 0;
    GetUintDivMagicAndShift(m0, shift0, perTensorShapeSize);

    uint32_t concatDim = static_cast<uint32_t>(tilingData_->concatDim);
    uint32_t needCalNum = static_cast<uint32_t>(tilingData_->needCalNum);
    int32_t threadNum = static_cast<int32_t>(tilingData_->threadNum);

    AscendC::Simt::VF_CALL<SimtComputer<T>>(Simt::Dim3(threadNum), x, y, m0, shift0, (__local_mem__ T*) (tmpLocal.GetPhyAddr()),
                                                                         perTensorShapeSize, concatDim, needCalNum);
}

}  // namespace ConcatOffset
#endif  // CONCAT_OFFSET_SIMT_H
