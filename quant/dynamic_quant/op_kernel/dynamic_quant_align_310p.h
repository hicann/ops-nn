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
 * \file dynamic_quant_align_310p.h
 * \brief
 */
#ifndef DYNAMIC_QUANT_ALIGN_H
#define DYNAMIC_QUANT_ALIGN_H

#include "kernel_operator.h"

namespace DynamicQuantNDOpt {

#if __CCE_AICORE__ < 220
constexpr uint32_t DYNAMIC_QUANT_BUFFER_NUM_ONE = 1;
constexpr uint32_t DYNAMIC_QUANT_BUFFER_NUM_TWO = 2;
constexpr uint32_t DYNAMIC_QUANT_SIZE_OF_INT8 = 1;
constexpr uint32_t DYNAMIC_QUANT_SIZE_OF_HALF = 2;
constexpr uint32_t DYNAMIC_QUANT_SIZE_OF_FLOAT = 4;
constexpr uint32_t DYNAMIC_QUANT_ALIGN_SIZE = 32;
constexpr uint32_t DYNAMIC_QUANT_ALIGN_NUM_X = 16;
constexpr uint32_t DYNAMIC_QUANT_ALIGN_NUM_SCALE = 8;
constexpr uint32_t DYNAMIC_QUANT_MAX_VALUE = 127;
constexpr uint32_t DYNAMIC_QUANT_MIN_VALUE = -128;
constexpr uint32_t DYNAMIC_QUANT_ASYMMETRIC_VALUE = 255;
constexpr uint32_t DYNAMIC_QUANT_HEADSPACE = 10240;
constexpr uint32_t DYNAMIC_QUANT_COPY_ROW_SCALE = 5;
constexpr uint32_t DYNAMIC_QUANT_FP16_BUF_SCALE = 2;

using AscendC::HardEvent;

class DynamicQuantAlign310p
{
public:
    __aicore__ inline DynamicQuantAlign310p()
    {}
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, GM_ADDR scale, GM_ADDR offset, const DynamicQuantTilingData* __restrict tilingData)
    {
        InitParams(tilingData, offset);
        InitBuffer(x, y, scale, offset);
        InitQueue();
    }
    __aicore__ inline void Process()
    {
        ProcessWithHeadDoublePipLineForAlign();
        ProcessWithHeadLastComputeAndCopyOutForAlign();
    }

    __aicore__ inline void CopyIn(uint64_t progress, uint32_t calCount)
    {
        // alloc tensor from queue memory
        AscendC::LocalTensor<half> inLocal = inQueue_.AllocTensor<half>();
        // copy progress_th tile from global tensor to local tensor
        DataCopy(inLocal, inGm_[progress * lenCopyRow_], calCount);
        // enque input tensors to VECIN queue
        inQueue_.EnQue(inLocal);
    }

    __aicore__ inline void Compute(uint32_t calRow, uint32_t sizeZ)
    {
        // deque input tensors from VECIN queue
        AscendC::LocalTensor<float> scaleLocal = scaleQueue_.AllocTensor<float>();
        AscendC::LocalTensor<int8_t> outLocal = outQueue_.AllocTensor<int8_t>();
        AscendC::LocalTensor<half> temp = fp16Buf_.Get<half>();
        AscendC::LocalTensor<half> inLocal = inQueue_.DeQue<half>();

        uint32_t idx;
        float maxValue;
        float scale;
        for (uint32_t i = 0; i < calRow; i++) {
            idx = i * sizeX_;
            Abs(temp, inLocal[idx], sizeH_);
            AscendC::PipeBarrier<PIPE_V>();

            ReduceMax(temp, temp, temp, sizeH_, false);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            maxValue = static_cast<float>(temp.GetValue(0));
            if (maxValue <= DYNAMIC_QUANT_EPSINON) {
                scale = 0.0f;
            } else {
                scale = DYNAMIC_QUANT_MAX_VALUE / maxValue;
            }
            scaleLocal.SetValue(i, maxValue);
            Muls(inLocal[idx], inLocal[idx], static_cast<half>(scale), sizeH_);
            AscendC::PipeBarrier<PIPE_V>();
            Cast(outLocal[i * sizeZ], inLocal[idx], AscendC::RoundMode::CAST_ROUND, sizeH_);
        }
        Muls(scaleLocal, scaleLocal, static_cast<float>(1.0f / DYNAMIC_QUANT_MAX_VALUE), calRow);
        // enque the output tensor to VECOUT queue
        outQueue_.EnQue<int8_t>(outLocal);
        scaleQueue_.EnQue<float>(scaleLocal);
        inQueue_.FreeTensor(inLocal);
    }

    __aicore__ inline void CopyOut(uint64_t progress, uint32_t calCount)
    {
        // deque output tensor from VECOUT queue
        AscendC::LocalTensor<int8_t> outLocal = outQueue_.DeQue<int8_t>();
        AscendC::LocalTensor<float> scaleLocal = scaleQueue_.DeQue<float>();
        AscendC::LocalTensor<float> offsetLocal;
        if (asymmetric_) {
            offsetLocal = offsetQueue_.DeQue<float>();
        }
        // copy progress_th tile from local tensor to global tensor
        DataCopy(outGm_[progress * lenCopyRow_], outLocal, calCount);
        DataCopy(scaleGm_[progress * numCopyRow_], scaleLocal, numCopyRow_);
        if (asymmetric_) {
            DataCopy(offsetGm_[progress * numCopyRow_], offsetLocal, numCopyRow_);
            offsetQueue_.FreeTensor(offsetLocal);
        }
        // free output tensor for reuse
        outQueue_.FreeTensor(outLocal);
        scaleQueue_.FreeTensor(scaleLocal);
    }

    __aicore__ inline void InitParams(const DynamicQuantTilingData* __restrict tilingData, GM_ADDR offset)
    {
        blockIdx_ = AscendC::GetBlockIdx();
        coreNum_ = tilingData->coreNum;
        sizeH_ = tilingData->sizeH;
        sizeX_ = tilingData->sizeX;
        numCopyRow_ = tilingData->numCopyRow;
        numHeadCore_ = tilingData->numHeadCore;
        numTailCore_ = tilingData->numTailCore;
        numHeadTimes_ = tilingData->numHeadTimes;
        numTailTimes_ = tilingData->numTailTimes;

        lenCopyRow_ = sizeH_ * numCopyRow_;
        numHeadRow_ = numHeadTimes_ * numCopyRow_;
        numTailRow_ = numTailTimes_ * numCopyRow_;
        lenHead_ = numHeadRow_ * sizeH_;
        lenTail_ = numTailRow_ * sizeH_;

        numLastTailRow_ = tilingData->numLastTailRow;
        lenLastTail_ = numLastTailRow_ * sizeH_;
        isTailLoop_ = numLastTailRow_ > 0 && blockIdx_ == coreNum_ - 1;
        if (offset == nullptr) {
            asymmetric_ = false;
        } else {
            asymmetric_ = true;
        }

        if (blockIdx_ < numHeadCore_) {
            loopCount_ = numHeadTimes_;
        } else {
            loopCount_ = numTailTimes_;
        }
    }

    __aicore__ inline void InitBuffer(GM_ADDR x, GM_ADDR z, GM_ADDR scale, GM_ADDR offset)
    {
        if (isTailLoop_) {
            inGm_.SetGlobalBuffer(
                (__gm__ half*)x + numHeadCore_ * lenHead_ + (blockIdx_ - numHeadCore_) * lenTail_,
                lenTail_ + lenLastTail_);
            outGm_.SetGlobalBuffer(
                (__gm__ int8_t*)z + numHeadCore_ * lenHead_ + (blockIdx_ - numHeadCore_) * lenTail_,
                lenTail_ + lenLastTail_);
            scaleGm_.SetGlobalBuffer(
                (__gm__ float*)scale + numHeadCore_ * numHeadRow_ + (blockIdx_ - numHeadCore_) * numTailRow_,
                numTailRow_ + numCopyRow_);
            if (asymmetric_) {
                offsetGm_.SetGlobalBuffer(
                    (__gm__ float*)offset + numHeadCore_ * numHeadRow_ + (blockIdx_ - numHeadCore_) * numTailRow_,
                    numTailRow_ + numCopyRow_);
            }
        } else {
            if (blockIdx_ < numHeadCore_) {
                inGm_.SetGlobalBuffer((__gm__ half*)x + blockIdx_ * lenHead_, lenHead_);
                outGm_.SetGlobalBuffer((__gm__ int8_t*)z + blockIdx_ * lenHead_, lenHead_);
                scaleGm_.SetGlobalBuffer((__gm__ float*)scale + blockIdx_ * numHeadRow_, numHeadRow_);
                if (asymmetric_) {
                    offsetGm_.SetGlobalBuffer((__gm__ float*)offset + blockIdx_ * numHeadRow_, numHeadRow_);
                }
            } else {
                inGm_.SetGlobalBuffer(
                    (__gm__ half*)x + numHeadCore_ * lenHead_ + (blockIdx_ - numHeadCore_) * lenTail_, lenTail_);
                outGm_.SetGlobalBuffer(
                    (__gm__ int8_t*)z + numHeadCore_ * lenHead_ + (blockIdx_ - numHeadCore_) * lenTail_, lenTail_);
                scaleGm_.SetGlobalBuffer(
                    (__gm__ float*)scale + numHeadCore_ * numHeadRow_ + (blockIdx_ - numHeadCore_) * numTailRow_,
                    numTailRow_);
                if (asymmetric_) {
                    offsetGm_.SetGlobalBuffer(
                        (__gm__ float*)offset + numHeadCore_ * numHeadRow_ + (blockIdx_ - numHeadCore_) * numTailRow_,
                        numTailRow_);
                }
            }
        }
    }

    __aicore__ inline void InitQueue()
    {
        pipe_.InitBuffer(inQueue_, DYNAMIC_QUANT_BUFFER_NUM_TWO, lenCopyRow_ * DYNAMIC_QUANT_SIZE_OF_HALF);
        pipe_.InitBuffer(outQueue_, DYNAMIC_QUANT_BUFFER_NUM_ONE, lenCopyRow_ * DYNAMIC_QUANT_SIZE_OF_INT8);
        pipe_.InitBuffer(scaleQueue_, DYNAMIC_QUANT_BUFFER_NUM_ONE, numCopyRow_ * DYNAMIC_QUANT_SIZE_OF_FLOAT);
        pipe_.InitBuffer(fp16Buf_, sizeH_ * DYNAMIC_QUANT_SIZE_OF_HALF);
        pipe_.InitBuffer(fp32Buf_, sizeH_ * DYNAMIC_QUANT_SIZE_OF_FLOAT);
        if (asymmetric_) {
            pipe_.InitBuffer(offsetQueue_, DYNAMIC_QUANT_BUFFER_NUM_ONE, numCopyRow_ * DYNAMIC_QUANT_SIZE_OF_FLOAT);
        }
    }

    __aicore__ inline void ProcessWithHeadDoublePipLineForAlign()
    {
        if (loopCount_ > 0) {
            CopyIn(0, lenCopyRow_);
            for (uint64_t i = 1; i < loopCount_; i++) {
                CopyIn(i, lenCopyRow_);
                ExcuteCompute();
                CopyOut(i - 1, lenCopyRow_);
            }
        }
    }

    __aicore__ inline void ProcessWithHeadLastComputeAndCopyOutForAlign()
    {
        if (loopCount_ > 0) {
            ExcuteCompute();
            CopyOut(loopCount_ - 1, lenCopyRow_);
        }
    }

    __aicore__ inline void ExcuteCompute()
    {
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
        if constexpr (std::is_same<half, bfloat16_t>::value) {
            ComputeBF16(numCopyRow_, sizeH_);
            return;
        }
#endif
        if (asymmetric_) {
            ComputeAsymmetric(numCopyRow_, sizeH_);
        } else {
            Compute(numCopyRow_, sizeH_);
        }
    }

    __aicore__ inline void ComputeBF16(uint32_t calRow, uint32_t sizeZ)
    {
        // deque input tensors from VECIN queue
        AscendC::LocalTensor<float> scaleLocal = scaleQueue_.AllocTensor<float>();
        AscendC::LocalTensor<int8_t> outLocal = outQueue_.AllocTensor<int8_t>();
        AscendC::LocalTensor<float> temp = fp32Buf_.Get<float>();
        AscendC::LocalTensor<half> tempf16 = fp16Buf_.Get<half>();
        AscendC::LocalTensor<float> tempf32 = fp32Buf_.Get<float>();
        AscendC::LocalTensor<half> inLocal = inQueue_.DeQue<half>();

        uint32_t idx;
        float maxValue;
        float scale;
        for (uint32_t i = 0; i < calRow; i++) {
            idx = i * sizeX_;
            // bf16 -> fp32
            Cast(tempf32, inLocal[idx], AscendC::RoundMode::CAST_NONE, sizeH_);
            AscendC::PipeBarrier<PIPE_V>();
            Abs(temp, tempf32, sizeH_);
            AscendC::PipeBarrier<PIPE_V>();

            ReduceMax(temp, temp, temp, sizeH_, false);
            AscendC::SetFlag<HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<HardEvent::V_S>(EVENT_ID0);
            maxValue = temp.GetValue(0);
            if (maxValue <= DYNAMIC_QUANT_EPSINON || maxValue == 0.0f) {
                scale = 0.0f;
            } else {
                scale = DYNAMIC_QUANT_MAX_VALUE / maxValue;
            }
            scaleLocal.SetValue(i, maxValue);
            Cast(tempf32, inLocal[idx], AscendC::RoundMode::CAST_NONE, sizeH_);
            AscendC::PipeBarrier<PIPE_V>();
            Muls(tempf32, tempf32, scale, sizeH_);
            AscendC::PipeBarrier<PIPE_V>();
            // fp32 -> f16
            Cast(tempf16, tempf32, AscendC::RoundMode::CAST_RINT, sizeH_);
            AscendC::PipeBarrier<PIPE_V>();
            // f16 -> int8
            Cast(outLocal[i * sizeZ], tempf16, AscendC::RoundMode::CAST_ROUND, sizeH_);
        }
        Muls(scaleLocal, scaleLocal, static_cast<float>(1.0f / DYNAMIC_QUANT_MAX_VALUE), calRow);
        // enque the output tensor to VECOUT queue
        outQueue_.EnQue<int8_t>(outLocal);
        scaleQueue_.EnQue<float>(scaleLocal);
        inQueue_.FreeTensor(inLocal);
    }

    __aicore__ inline void ComputeAsymmetric(uint32_t calRow, uint32_t sizeZ)
    {
        // deque input tensors from VECIN queue
        AscendC::LocalTensor<float> scaleLocal = scaleQueue_.AllocTensor<float>();
        AscendC::LocalTensor<float> offsetLocal = offsetQueue_.AllocTensor<float>();
        AscendC::LocalTensor<int8_t> outLocal = outQueue_.AllocTensor<int8_t>();
        AscendC::LocalTensor<half> temp = fp16Buf_.Get<half>();
        AscendC::LocalTensor<half> inLocal = inQueue_.DeQue<half>();

        uint32_t idx;
        float maxValue;
        float minValue;
        float subValue;
        float scale;
        float offset;
        for (uint32_t i = 0; i < calRow; i++) {
            idx = i * sizeX_;
            ReduceMax(temp, inLocal[idx], temp, sizeH_, false);
            AscendC::SetFlag<HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<HardEvent::V_S>(EVENT_ID0);
            maxValue = static_cast<float>(temp.GetValue(0));

            ReduceMin(temp, inLocal[idx], temp, sizeH_, false);
            AscendC::SetFlag<HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<HardEvent::V_S>(EVENT_ID0);
            minValue = static_cast<float>(temp.GetValue(0));

            subValue = maxValue - minValue;
            if (subValue <= DYNAMIC_QUANT_EPSINON) {
                scale = 0.0f;
            } else {
                scale = DYNAMIC_QUANT_ASYMMETRIC_VALUE / subValue;
            }
            scaleLocal.SetValue(i, subValue / DYNAMIC_QUANT_ASYMMETRIC_VALUE);

            offset = -(maxValue + minValue) * scale * 0.5f; // 0.5f: half of (maxValue + minValue)
            offsetLocal.SetValue(i, offset);

            Muls(inLocal[idx], inLocal[idx], static_cast<half>(scale), sizeH_);
            AscendC::PipeBarrier<PIPE_V>();
            Adds(inLocal[idx], inLocal[idx], static_cast<half>(offset), sizeH_);
            AscendC::PipeBarrier<PIPE_V>();
            Cast(outLocal[i * sizeZ], inLocal[idx], AscendC::RoundMode::CAST_ROUND, sizeH_);
        }
        // enque the output tensor to VECOUT queue
        outQueue_.EnQue<int8_t>(outLocal);
        scaleQueue_.EnQue<float>(scaleLocal);
        offsetQueue_.EnQue<float>(offsetLocal);
        inQueue_.FreeTensor(inLocal);
    }

public:
    AscendC::TPipe pipe_;
    // create queues for input, in this case depth is equal to buffer num
    AscendC::TQue<AscendC::QuePosition::VECIN, DYNAMIC_QUANT_BUFFER_NUM_TWO> inQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, DYNAMIC_QUANT_BUFFER_NUM_ONE> outQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, DYNAMIC_QUANT_BUFFER_NUM_ONE> scaleQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, DYNAMIC_QUANT_BUFFER_NUM_ONE> offsetQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp16Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp32Buf_;

    AscendC::GlobalTensor<half> inGm_;
    AscendC::GlobalTensor<int8_t> outGm_;
    AscendC::GlobalTensor<float> scaleGm_;
    AscendC::GlobalTensor<float> offsetGm_;

    uint32_t blockIdx_;
    uint32_t coreNum_;
    uint32_t sizeH_;
    uint32_t sizeX_;
    uint32_t numCopyRow_;
    uint32_t lenCopyRow_;

    uint32_t numHeadCore_;
    uint32_t numTailCore_;
    uint32_t numHeadTimes_;
    uint32_t numTailTimes_;

    uint32_t lenHead_;
    uint32_t lenTail_;
    uint64_t loopCount_;
    uint32_t numHeadRow_;
    uint32_t numTailRow_;

    uint32_t numLastTailRow_;
    uint32_t lenLastTail_;
    bool isTailLoop_;
    bool asymmetric_; // true: asymmetric, false: symmetric
};
#endif
} // namespace DynamicQuantNDOpt
#endif
