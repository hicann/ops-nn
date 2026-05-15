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
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/**
 * \file hard_sigmoid_grad_v3.h
 * \brief HardSigmoidGradV3 kernel class definition (arch32)
 */
#ifndef HARD_SIGMOID_GRAD_V3_H
#define HARD_SIGMOID_GRAD_V3_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "hard_sigmoid_grad_v3_tiling_data.h"
#include "hard_sigmoid_grad_v3_tiling_key.h"

namespace NsHardSigmoidGradV3 {

constexpr int64_t DATA_COPY_ALIGN_BYTES = 32;
constexpr int64_t COMPARE_ALIGN_BYTES = 256;
constexpr float LOWER_BOUND = -3.0f;
constexpr float UPPER_BOUND = 3.0f;
constexpr float SCALE_VALUE = 1.0f / 6.0f;

__aicore__ inline int64_t AlignUp(int64_t value, int64_t align)
{
    if (align == 0) {
        return value;
    }
    if (align < 0) {
        return value;
    }
    return ((value + align - 1) / align) * align;
}

template <typename T, int BUFFER_MODE>
class HardSigmoidGradV3 {
    static constexpr int32_t BUFFER_NUM = BUFFER_MODE ? 2 : 1;
    static constexpr bool IS_BF16 = std::is_same<T, bfloat16_t>::value;

public:
    __aicore__ inline HardSigmoidGradV3() {}

    __aicore__ inline void Init(GM_ADDR gradOutput, GM_ADDR self, GM_ADDR gradInput,
                                const HardSigmoidGradV3TilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInFull(int64_t progress);
    __aicore__ inline void CopyInTail(int64_t progress, int64_t validLength);
    __aicore__ inline void ComputeFull();
    __aicore__ inline void ComputeTail(int64_t validLength);
    __aicore__ inline void CopyOutFull(int64_t progress);
    __aicore__ inline void CopyOutTail(int64_t progress, int64_t validLength);

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> gradQueue;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> selfQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outputQueue;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> compareMaskBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> gradFloatBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> selfFloatBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> outFloatBuf;

    AscendC::GlobalTensor<T> gradGM;
    AscendC::GlobalTensor<T> selfGM;
    AscendC::GlobalTensor<T> outputGM;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T, int BUFFER_MODE>
__aicore__ inline void HardSigmoidGradV3<T, BUFFER_MODE>::Init(
    GM_ADDR gradOutput, GM_ADDR self, GM_ADDR gradInput, const HardSigmoidGradV3TilingData* tilingData)
{
    int64_t blockIdx = AscendC::GetBlockIdx();
    int64_t remainderLength = tilingData->totalNum - tilingData->blockFactor * blockIdx;
    blockLength_ = (remainderLength > tilingData->blockFactor) ? tilingData->blockFactor : remainderLength;
    ubLength_ = tilingData->ubFactor;

    gradGM.SetGlobalBuffer((__gm__ T*)gradOutput + tilingData->blockFactor * blockIdx, blockLength_);
    selfGM.SetGlobalBuffer((__gm__ T*)self + tilingData->blockFactor * blockIdx, blockLength_);
    outputGM.SetGlobalBuffer((__gm__ T*)gradInput + tilingData->blockFactor * blockIdx, blockLength_);

    pipe.InitBuffer(gradQueue, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(selfQueue, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueue, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(compareMaskBuf, ubLength_ * sizeof(uint8_t));
    if constexpr (IS_BF16) {
        pipe.InitBuffer(gradFloatBuf, ubLength_ * sizeof(float));
        pipe.InitBuffer(selfFloatBuf, ubLength_ * sizeof(float));
        pipe.InitBuffer(outFloatBuf, ubLength_ * sizeof(float));
    }
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void HardSigmoidGradV3<T, BUFFER_MODE>::CopyInFull(int64_t progress)
{
    AscendC::LocalTensor<T> gradLocal = gradQueue.template AllocTensor<T>();
    AscendC::LocalTensor<T> selfLocal = selfQueue.template AllocTensor<T>();
    AscendC::DataCopyExtParams copyParams {1, static_cast<uint32_t>(ubLength_ * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> padParams {false, 0, 0, static_cast<T>(0)};
    AscendC::DataCopyPad(gradLocal, gradGM[progress * ubLength_], copyParams, padParams);
    AscendC::DataCopyPad(selfLocal, selfGM[progress * ubLength_], copyParams, padParams);
    gradQueue.EnQue(gradLocal);
    selfQueue.EnQue(selfLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void HardSigmoidGradV3<T, BUFFER_MODE>::CopyInTail(int64_t progress, int64_t validLength)
{
    int64_t alignNum = DATA_COPY_ALIGN_BYTES / static_cast<int64_t>(sizeof(T));
    int64_t alignedLength = AlignUp(validLength, alignNum);
    AscendC::DataCopyExtParams copyParams {1, static_cast<uint32_t>(validLength * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> padParams {
        true, 0, static_cast<uint8_t>(alignedLength - validLength), static_cast<T>(0)};

    AscendC::LocalTensor<T> gradLocal = gradQueue.template AllocTensor<T>();
    AscendC::LocalTensor<T> selfLocal = selfQueue.template AllocTensor<T>();
    AscendC::DataCopyPad(gradLocal, gradGM[progress * ubLength_], copyParams, padParams);
    AscendC::DataCopyPad(selfLocal, selfGM[progress * ubLength_], copyParams, padParams);
    gradQueue.EnQue(gradLocal);
    selfQueue.EnQue(selfLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void HardSigmoidGradV3<T, BUFFER_MODE>::ComputeFull()
{
    AscendC::LocalTensor<T> gradLocal = gradQueue.template DeQue<T>();
    AscendC::LocalTensor<T> selfLocal = selfQueue.template DeQue<T>();
    AscendC::LocalTensor<T> resultLocal = outputQueue.template AllocTensor<T>();

    if constexpr (IS_BF16) {
        AscendC::LocalTensor<float> gradFloat = gradFloatBuf.Get<float>();
        AscendC::LocalTensor<float> selfFloat = selfFloatBuf.Get<float>();
        AscendC::LocalTensor<float> outFloat = outFloatBuf.Get<float>();
        AscendC::Cast(gradFloat, gradLocal, AscendC::RoundMode::CAST_NONE, static_cast<uint32_t>(ubLength_));
        AscendC::Cast(selfFloat, selfLocal, AscendC::RoundMode::CAST_NONE, static_cast<uint32_t>(ubLength_));
        AscendC::PipeBarrier<PIPE_V>();
        for (int64_t idx = 0; idx < ubLength_; ++idx) {
            float selfValue = selfFloat.GetValue(idx);
            float gradValue = gradFloat.GetValue(idx);
            float outValue = 0.0f;
            if (selfValue > LOWER_BOUND && selfValue < UPPER_BOUND) {
                outValue = gradValue * SCALE_VALUE;
            }
            outFloat.SetValue(idx, outValue);
        }
        AscendC::Cast(resultLocal, outFloat, AscendC::RoundMode::CAST_ROUND, static_cast<uint32_t>(ubLength_));
    } else {
        AscendC::LocalTensor<uint8_t> compareMask = compareMaskBuf.Get<uint8_t>();
        AscendC::Muls(resultLocal, gradLocal, static_cast<T>(SCALE_VALUE), static_cast<uint32_t>(ubLength_));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Abs(selfLocal, selfLocal, static_cast<uint32_t>(ubLength_));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::CompareScalar(compareMask, selfLocal, static_cast<T>(UPPER_BOUND), AscendC::CMPMODE::LT,
                               static_cast<uint32_t>(ubLength_));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Select<T, uint8_t>(resultLocal, compareMask, resultLocal, static_cast<T>(0),
                                    AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE,
                                    static_cast<uint32_t>(ubLength_));
    }

    outputQueue.template EnQue<T>(resultLocal);
    gradQueue.FreeTensor(gradLocal);
    selfQueue.FreeTensor(selfLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void HardSigmoidGradV3<T, BUFFER_MODE>::ComputeTail(int64_t validLength)
{
    AscendC::LocalTensor<T> gradLocal = gradQueue.template DeQue<T>();
    AscendC::LocalTensor<T> selfLocal = selfQueue.template DeQue<T>();
    AscendC::LocalTensor<T> resultLocal = outputQueue.template AllocTensor<T>();

    if constexpr (IS_BF16) {
        AscendC::LocalTensor<float> gradFloat = gradFloatBuf.Get<float>();
        AscendC::LocalTensor<float> selfFloat = selfFloatBuf.Get<float>();
        AscendC::LocalTensor<float> outFloat = outFloatBuf.Get<float>();
        AscendC::Cast(gradFloat, gradLocal, AscendC::RoundMode::CAST_NONE, static_cast<uint32_t>(validLength));
        AscendC::Cast(selfFloat, selfLocal, AscendC::RoundMode::CAST_NONE, static_cast<uint32_t>(validLength));
        AscendC::PipeBarrier<PIPE_V>();
        for (int64_t idx = 0; idx < validLength; ++idx) {
            float selfValue = selfFloat.GetValue(idx);
            float gradValue = gradFloat.GetValue(idx);
            float outValue = 0.0f;
            if (selfValue > LOWER_BOUND && selfValue < UPPER_BOUND) {
                outValue = gradValue * SCALE_VALUE;
            }
            outFloat.SetValue(idx, outValue);
        }
        AscendC::Cast(resultLocal, outFloat, AscendC::RoundMode::CAST_ROUND, static_cast<uint32_t>(validLength));
    } else {
        for (int64_t idx = 0; idx < validLength; ++idx) {
            float selfValue = static_cast<float>(selfLocal.GetValue(idx));
            float gradValue = static_cast<float>(gradLocal.GetValue(idx));
            float outValue = 0.0f;
            if (selfValue > LOWER_BOUND && selfValue < UPPER_BOUND) {
                outValue = gradValue * SCALE_VALUE;
            }
            resultLocal.SetValue(idx, static_cast<T>(outValue));
        }
    }

    outputQueue.template EnQue<T>(resultLocal);
    gradQueue.FreeTensor(gradLocal);
    selfQueue.FreeTensor(selfLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void HardSigmoidGradV3<T, BUFFER_MODE>::CopyOutFull(int64_t progress)
{
    AscendC::LocalTensor<T> resultLocal = outputQueue.template DeQue<T>();
    AscendC::DataCopyExtParams copyParams {1, static_cast<uint32_t>(ubLength_ * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPad(outputGM[progress * ubLength_], resultLocal, copyParams);
    outputQueue.FreeTensor(resultLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void HardSigmoidGradV3<T, BUFFER_MODE>::CopyOutTail(int64_t progress, int64_t validLength)
{
    AscendC::LocalTensor<T> resultLocal = outputQueue.template DeQue<T>();
    AscendC::DataCopyExtParams copyParams {1, static_cast<uint32_t>(validLength * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPad(outputGM[progress * ubLength_], resultLocal, copyParams);
    outputQueue.FreeTensor(resultLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void HardSigmoidGradV3<T, BUFFER_MODE>::Process()
{
    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }

    int64_t fullTileNum = blockLength_ / ubLength_;
    if (fullTileNum > 0) {
        CopyInFull(0);
        for (int64_t i = 0; i < fullTileNum - 1; ++i) {
            CopyInFull(i + 1);
            ComputeFull();
            CopyOutFull(i);
        }
        ComputeFull();
        CopyOutFull(fullTileNum - 1);
    }

    int64_t tailLength = blockLength_ - fullTileNum * ubLength_;
    if (tailLength > 0) {
        CopyInTail(fullTileNum, tailLength);
        ComputeTail(tailLength);
        CopyOutTail(fullTileNum, tailLength);
    }
}

} // namespace NsHardSigmoidGradV3

#endif // HARD_SIGMOID_GRAD_V3_H
