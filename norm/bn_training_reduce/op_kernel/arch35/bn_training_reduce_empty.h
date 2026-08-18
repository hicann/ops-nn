/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BN_TRAINING_REDUCE_EMPTY_H_
#define BN_TRAINING_REDUCE_EMPTY_H_

#include "bn_training_reduce_tiling_data.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

namespace NsBNTrainingReduce {

using namespace AscendC;

constexpr uint32_t kEmptyRepF32 = 256U / sizeof(float);

template <bool hasPostElewise>
__simd_vf__ inline void DuplicateEmptyROutputVfImpl(__ubuf__ float* output, float emptyROutputValue, uint32_t count,
                                                    uint16_t repeats)
{
    float finalValue = emptyROutputValue;
    if constexpr (hasPostElewise) {
        finalValue = emptyROutputValue;
    }
    AscendC::Reg::RegTensor<float> value;
    AscendC::Reg::Duplicate(value, finalValue);
    AscendC::Reg::MaskReg mask;
    uint32_t remaining = count;
    for (uint16_t i = 0; i < repeats; ++i) {
        const int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(kEmptyRepF32);
        mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::StoreAlign(output + off, value, mask);
    }
}

template <typename DType>
class BNTrainingReduceEmpty {
public:
    __aicore__ inline void Init(GM_ADDR sum, GM_ADDR squareSum, const BNTrainingReduceTilingData* td)
    {
        usedCoreNum_ = td->usedCoreNum;
        if (static_cast<int64_t>(GetBlockIdx()) >= static_cast<int64_t>(usedCoreNum_)) {
            return;
        }

        aTotal_ = td->axisShape[0];
        aUbFactor_ = td->aUbFactor;
        aBigCoreCnt_ = td->aBigCoreCnt;
        aBigCoreLoopCnt_ = td->aBigCoreLoopCnt;
        aSmallCoreLoopCnt_ = td->aSmallCoreLoopCnt;
        sumGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sum));
        squareSumGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(squareSum));
        pipe_.InitBuffer(outQue_, 1, td->postReduceUbSize);
    }

    __aicore__ inline void Process(int32_t outputIdx)
    {
        const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
        if (blockIdx >= static_cast<int64_t>(usedCoreNum_)) {
            return;
        }

        int64_t loopStart = 0;
        int64_t loopEnd = 0;
        if (blockIdx < static_cast<int64_t>(aBigCoreCnt_)) {
            loopStart = blockIdx * aBigCoreLoopCnt_;
            loopEnd = loopStart + aBigCoreLoopCnt_;
        } else {
            loopStart = static_cast<int64_t>(aBigCoreCnt_) * aBigCoreLoopCnt_ +
                        (blockIdx - static_cast<int64_t>(aBigCoreCnt_)) * aSmallCoreLoopCnt_;
            loopEnd = loopStart + aSmallCoreLoopCnt_;
        }

        for (int64_t loop = loopStart; loop < loopEnd; ++loop) {
            const int64_t outOff = loop * aUbFactor_;
            const int64_t remain = aTotal_ - outOff;
            const int64_t aLen = remain < aUbFactor_ ? remain : aUbFactor_;
            DuplicateEmptyROutput(outputIdx, outOff, aLen);
        }
    }

private:
    __aicore__ inline void DuplicateEmptyROutput(int32_t outputIdx, int64_t outOff, int64_t aLen)
    {
        auto out = outQue_.AllocTensor<float>();
        __ubuf__ float* output = reinterpret_cast<__ubuf__ float*>(out.GetPhyAddr());
        const uint32_t count = static_cast<uint32_t>(aLen);
        const uint16_t repeats = static_cast<uint16_t>((count + kEmptyRepF32 - 1U) / kEmptyRepF32);
        asc_vf_call<DuplicateEmptyROutputVfImpl<false>>(output, 0.0F, count, repeats);
        outQue_.EnQue(out);

        auto deq = outQue_.DeQue<float>();
        DataCopyExtParams params = {};
        params.blockLen = static_cast<uint32_t>(aLen * sizeof(float));
        params.blockCount = 1;
        params.srcStride = 0;
        params.dstStride = 0;
        params.rsv = 0;
        if (outputIdx == 0) {
            DataCopyPad(sumGm_[outOff], deq, params);
        } else {
            DataCopyPad(squareSumGm_[outOff], deq, params);
        }
        outQue_.FreeTensor(deq);
    }

    int32_t usedCoreNum_ = 0;
    int64_t aTotal_ = 0;
    int64_t aUbFactor_ = 0;
    int32_t aBigCoreCnt_ = 0;
    int64_t aBigCoreLoopCnt_ = 0;
    int64_t aSmallCoreLoopCnt_ = 0;
    GlobalTensor<float> sumGm_;
    GlobalTensor<float> squareSumGm_;
    TPipe pipe_;
    TQue<QuePosition::VECOUT, 1> outQue_;
};

} // namespace NsBNTrainingReduce

#endif // BN_TRAINING_REDUCE_EMPTY_H_
