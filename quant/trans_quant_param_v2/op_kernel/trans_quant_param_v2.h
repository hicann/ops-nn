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
 * \file trans_quant_param_v2.h
 * \brief
 */
#ifndef TRANS_QUANT_PARAM_V2_H
#define TRANS_QUANT_PARAM_V2_H

#include "kernel_operator.h"
#include "kernel_operator_intf.h"

namespace AscendC {

constexpr uint64_t DEQ_SCALE_MUL = 0xFFFFE000;
constexpr uint64_t QUANT_SCALE = 0x1ULL << 46;
constexpr uint64_t QUANT_MASK_0 = 0x1FFULL;
constexpr int32_t UB_ALIGN_SIZE = 32;
constexpr int32_t OFFSET_DEVIATION = 37;
constexpr int32_t SPLIT_SIZE = 65536;
constexpr int32_t MAX_INT9 = 255;
constexpr int32_t MIN_INT9 = -256;
constexpr int32_t FP19_SIGN_SHIFT = 18;
constexpr int32_t FP19_TAIL_SHIFT = 13;
constexpr int32_t FP19_EXP_SHIFT = 10;
constexpr int32_t FP32_SIGN_SHIFT = 31;
constexpr int32_t FP32_EXP_SHIFT = 23;
constexpr int32_t FP32_MANTISSA_LEN = 0x7FFFFF;
constexpr int32_t FP19_TAIL_LEN = 0xFFF;

__aicore__ inline constexpr bool IsDataCopyPadSupport()
{
#if __CCE_AICORE__ == 220
    return true;
#else
    return false;
#endif
}

template <bool forAtomicAdd = false, typename T>
__aicore__ inline void SafeDataCopy(
    const AscendC::GlobalTensor<T>& dstGlobal, const AscendC::LocalTensor<T>& srcLocal, const int64_t& calCount,
    bool recoverUbTailFormat = false)
{
    constexpr int typeSize = sizeof(T);                                // 元素字节数
    constexpr int numElemsPerBlock = AscendC::ONE_BLK_SIZE / typeSize; // 32byte元素数
    if constexpr (IsDataCopyPadSupport() && sizeof(T) < 8) { // 如果支持DataCopyPad则直接DataCopyPad拷贝
        AscendC::DataCopyParams copyParams{1, static_cast<uint16_t>(calCount * typeSize), 0, 0};
        DataCopyPad(dstGlobal, srcLocal, copyParams);
    } else {
        if (likely(!(calCount % numElemsPerBlock))) { // 对齐则直接DataCopy拷贝
            struct AscendC::DataCopyParams copyParams;
            copyParams.blockLen = calCount / AscendC::AscendCUtils::GetC0Count(typeSize);
            DataCopy(dstGlobal, srcLocal, copyParams);
        } else { // 如果既不支持DataCopyPad也不对齐则地址回退拷贝
            const int numAlignedBlocks = calCount / numElemsPerBlock * numElemsPerBlock; // 对齐部分
            if (calCount * typeSize < AscendC::ONE_BLK_SIZE) {
                DataCopy(dstGlobal, srcLocal, numElemsPerBlock);
                return; // 此处依然有内存踩踏
            }
            DataCopy(dstGlobal, srcLocal, numAlignedBlocks);
            event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_S));
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(eventID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(eventID);
            const int rollbackEleCount = calCount - numAlignedBlocks;          // 计算需要回退处理byte数
            const size_t rollbackDstIdx = numAlignedBlocks - numElemsPerBlock; // 操作回退的block元素索引
            const size_t rollbackSrcIdx = rollbackDstIdx + rollbackEleCount;   // 回退来源元素索引
            if constexpr (!forAtomicAdd) {
                for (int i = 0; i < numElemsPerBlock;
                     ++i) { // 将整个非对齐的尾块以一个block的size复制填充到前一个block中
                    srcLocal.SetValue((rollbackDstIdx + i), srcLocal.GetValue(rollbackSrcIdx + i)); // 重造local buf
                }
            } else {
                const size_t setZeroEleCount = numElemsPerBlock - rollbackEleCount; // 需要置0的元素量
                for (int i = 0; i < setZeroEleCount; ++i) {
                    srcLocal.SetValue((rollbackDstIdx + i), 0); // Atomic模式下，回退部分置0，使得回退部分不会重复加
                }
                for (int i = setZeroEleCount; i < numElemsPerBlock; ++i) { // 将整个非对齐的尾块复制填充到前一个block中
                    srcLocal.SetValue((rollbackDstIdx + i), srcLocal.GetValue(rollbackSrcIdx + i)); // 重造local buf
                }
                DataCopy(dstGlobal[calCount - numElemsPerBlock], srcLocal[rollbackDstIdx], numElemsPerBlock);
                return; // AtomicAdd模式下 暂不支持recoverUbTailFormat，待后续扩展支持
            }
            DataCopy(dstGlobal[calCount - numElemsPerBlock], srcLocal[rollbackDstIdx], numElemsPerBlock);
            if (recoverUbTailFormat) { // 还原回滚现场
                event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventID);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventID);
                DataCopy(
                    srcLocal[rollbackDstIdx], dstGlobal[rollbackDstIdx],
                    numElemsPerBlock); // 还原用于回退的block内容
            }
        }
    }
}

class TransQuantParamV2 {
public:
    __aicore__ inline TransQuantParamV2(){};
    __aicore__ inline void Init(
        GM_ADDR scale, GM_ADDR offset, GM_ADDR y, GM_ADDR workspace, const TransQuantParamV2TilingData* __restrict__ tilingData,
        TPipe* tPipe)
    {
        if ASCEND_IS_AIC {
            return;
        }
        scaleLength_ = tilingData->scaleLength;
        offsetLength_ = tilingData->offsetLength;
        roundMode_ = tilingData->roundMode;
        // init global buffer
        scaleGm_.SetGlobalBuffer((__gm__ float*)scale);
        offsetGm_.SetGlobalBuffer((__gm__ float*)offset);
        yGm_.SetGlobalBuffer((__gm__ uint64_t*)y);
        pipe_ = tPipe;
    }

    /** main logical function
     */
    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIC {
            return;
        }
        if (GetBlockIdx() > 0) {
            return;
        }
        CalOffsetValueShapeOne();
        CalScaleValueShapeOne();
        uint32_t eachLength = SPLIT_SIZE / sizeof(uint64_t);
        uint32_t loops = Max<uint32_t>(scaleLength_, offsetLength_) / eachLength;
        uint32_t tailLength = Max<uint32_t>(scaleLength_, offsetLength_) - loops * eachLength;
        uint32_t alignedLength = Align<uint32_t>(eachLength * sizeof(float), UB_ALIGN_SIZE) / sizeof(float);
        pipe_->InitBuffer(
            offsetUb_, 2 * alignedLength * sizeof(float)); // need 2 times space for float and int32_t tensor
        pipe_->InitBuffer(resUb_, SPLIT_SIZE);
        LocalTensor<uint64_t> resTensor = resUb_.Get<uint64_t>(eachLength);
        DataCopyParams ub2GmParams{1, SPLIT_SIZE / UB_ALIGN_SIZE, 0, 0};
        for (uint32_t loopidx = 0; loopidx < loops; ++loopidx) {
            for (uint32_t idx = 0; idx < eachLength; ++idx) {
                CalQuantPreScale(eachLength * loopidx + idx, idx, resTensor);
                if (offsetLength_ == 1) {
                    resTensor.SetValue(idx, resTensor.GetValue(idx) | offetInt9Bit_);
                }
            }
            if (offsetLength_ > 1) {
                CalOffset(eachLength, static_cast<uint64_t>(eachLength) * loopidx, resTensor);
            }
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopy(yGm_[static_cast<uint64_t>(eachLength) * loopidx], resTensor, ub2GmParams);
            SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        }
        PipeBarrier<PIPE_ALL>();
        for (uint32_t idx = 0; idx < tailLength; ++idx) {
            CalQuantPreScale(eachLength * loops + idx, idx, resTensor);
            if (offsetLength_ == 1) {
                resTensor.SetValue(idx, resTensor.GetValue(idx) | offetInt9Bit_);
            }
        }
        if (tailLength != 0) {
            if (offsetLength_ > 1) {
                CalOffset(tailLength, static_cast<uint64_t>(eachLength) * loops, resTensor);
            }
            DataCopyParams ub2GmParams1;
            SetCopyParams(ub2GmParams1);
            ub2GmParams1.blockLen = tailLength * sizeof(uint64_t);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopyGlobal(yGm_[static_cast<uint64_t>(eachLength) * loops], resTensor, ub2GmParams1);
        }
    }

protected:
    TPipe* pipe_;
    GlobalTensor<float> scaleGm_;
    GlobalTensor<float> offsetGm_;
    GlobalTensor<uint64_t> yGm_;

    TBuf<> resUb_;
    TBuf<> offsetUb_;

    uint32_t scaleLength_;
    uint32_t offsetLength_;
    uint32_t roundMode_;

    uint64_t offetInt9Bit_ = 0;
    uint64_t scalequantPre_ = 0;

    template <typename T>
    __aicore__ inline void DataCopyLocal(const LocalTensor<T>& dst, const GlobalTensor<T>& src, DataCopyParams& params)
    {
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ > 200)
        DataCopyPadParams padParams;
        DataCopyPad(dst, src, params, padParams);
#else
        params.blockLen = Align<uint32_t>(params.blockLen, UB_ALIGN_SIZE) / UB_ALIGN_SIZE;
        DataCopy(dst, src, params);
#endif
    }

    template <typename T>
    __aicore__ inline void DataCopyGlobal(const GlobalTensor<T>& dst, const LocalTensor<T>& src, DataCopyParams& params)
    {
#if defined(__DAV_C310__)
        DataCopyPad(dst, src, params);
#elif defined(__CCE_AICORE__) && (__CCE_AICORE__ > 200)
        DataCopyPad(dst, src, {params.blockCount, params.blockLen, 0, 0, 0});
#else
        params.blockLen = params.blockLen / sizeof(T);
        SafeDataCopy(dst, src, params.blockLen);
#endif
    }

    __aicore__ inline void CalQuantPreScale(uint32_t gmIdx, uint32_t idx, LocalTensor<uint64_t> resTensor)
    {
        if (scaleLength_ == 1) {
            resTensor.SetValue(idx, scalequantPre_);
            return;
        }
        float scaleOri = scaleGm_.GetValue(gmIdx);
        uint32_t uint32Scale = *(reinterpret_cast<uint32_t*>(&scaleOri));
        uint64_t quantPre = CalQuantPreScaleValue(uint32Scale);
        resTensor.SetValue(idx, quantPre);
    }

    __aicore__ inline void CalOffsetValueShapeOne()
    {
        if (offsetLength_ != 1) {
            return;
        }
        uint32_t alignedLength = Align<uint32_t>(1 * sizeof(float), UB_ALIGN_SIZE) / sizeof(float);
        pipe_->InitBuffer(offsetUb_, 2 * alignedLength * sizeof(float)); // 需要2倍空间给 float和int32_t tensor
        LocalTensor<float> offsetFp32_ = offsetUb_.Get<float>(alignedLength);
        LocalTensor<int32_t> offsetInt32_ = offsetUb_.Get<int32_t>(alignedLength);
        DataCopyParams gm2UbParams;
        gm2UbParams.blockLen = 1 * sizeof(float);
        SetCopyParams(gm2UbParams);
        DataCopyLocal(offsetFp32_, offsetGm_, gm2UbParams);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Cast(offsetInt32_, offsetFp32_, RoundMode::CAST_RINT, 1); // round to nearest, tie to even
        PipeBarrier<PIPE_V>();
        Maxs(offsetInt32_, offsetInt32_, MIN_INT9, 1);
        PipeBarrier<PIPE_V>();
        Mins(offsetInt32_, offsetInt32_, MAX_INT9, 1);
        PipeBarrier<PIPE_ALL>();
        int offsetVal = offsetInt32_.GetValue(0);
        offetInt9Bit_ = (static_cast<uint64_t>(offsetVal) & QUANT_MASK_0) << OFFSET_DEVIATION;
    }

    __aicore__ inline void CalScaleValueShapeOne()
    {
        if (scaleLength_ != 1) {
            return;
        }
        float scaleOri = scaleGm_.GetValue(0);
        uint32_t uint32Scale = *(reinterpret_cast<uint32_t*>(&scaleOri));
        scalequantPre_ = CalQuantPreScaleValue(uint32Scale);
    }

    __aicore__ inline uint64_t CalQuantPreScaleValue(uint32_t uint32Scale)
    {
        uint64_t quantPre = 0;
        if (roundMode_ == 0) {
            quantPre = (uint32Scale & DEQ_SCALE_MUL) | QUANT_SCALE; // 取高19位
        } else {
            uint32_t fp19Scale = 0;
            uint32_t sign = (uint32Scale >> FP32_SIGN_SHIFT) & 0x1;
            uint32_t exponent = (uint32Scale >> FP32_EXP_SHIFT) & MAX_INT9;
            uint32_t mantissa = uint32Scale & FP32_MANTISSA_LEN;
            uint32_t fp19Mantissa = mantissa >> FP19_TAIL_SHIFT;
            uint32_t roundBit = (mantissa >> (FP19_TAIL_SHIFT - 1)) & 0x1;
            uint32_t stickyBits = mantissa & FP19_TAIL_LEN;
            if (exponent == 0) {
                fp19Scale = sign << FP19_SIGN_SHIFT;
            } else if (exponent == MAX_INT9) {
                fp19Scale = (sign << FP19_SIGN_SHIFT) | (exponent << FP19_EXP_SHIFT) | fp19Mantissa;
            } else {
                if ((roundBit != 0) && ((stickyBits != 0) || ((fp19Mantissa & 0x1) != 0))) {
                    fp19Mantissa += 1;
                    if (fp19Mantissa >= (1 << FP19_EXP_SHIFT)) {
                        fp19Mantissa = 0;
                        exponent += 1;
                    }
                }
                if (exponent > MAX_INT9) {
                    fp19Scale = (sign << FP19_SIGN_SHIFT) | (MAX_INT9 << FP19_EXP_SHIFT);
                } else {
                    fp19Scale = (sign << FP19_SIGN_SHIFT) | (exponent << FP19_EXP_SHIFT) | fp19Mantissa;
                }
            }
            quantPre = (fp19Scale << FP19_TAIL_SHIFT) | QUANT_SCALE;
        }
        return quantPre;
    }

    __aicore__ inline void SetOffsetValue(
        LocalTensor<uint64_t> resTensor, LocalTensor<int32_t> offsetInt32, LocalTensor<float> offsetFp32,
        uint32_t length)
    {
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Cast(offsetInt32, offsetFp32, RoundMode::CAST_RINT, length); // round to nearest, tie to even
        PipeBarrier<PIPE_V>();
        Maxs(offsetInt32, offsetInt32, MIN_INT9, length);
        PipeBarrier<PIPE_V>();
        Mins(offsetInt32, offsetInt32, MAX_INT9, length);
        PipeBarrier<PIPE_ALL>();
        for (uint32_t idx = 0; idx < length; ++idx) {
            int offsetVal = offsetInt32.GetValue(idx);
            uint64_t int9bits = (static_cast<uint64_t>(offsetVal) & QUANT_MASK_0) << OFFSET_DEVIATION;
            resTensor.SetValue(idx, resTensor.GetValue(idx) | int9bits);
        }
    }

    __aicore__ inline void CalOffset(uint32_t length, uint64_t gmOffset, LocalTensor<uint64_t> resTensor)
    {
        uint32_t alignedLength = Align<uint32_t>(length * sizeof(float), UB_ALIGN_SIZE) / sizeof(float);
        LocalTensor<float> offsetFp32_ = offsetUb_.Get<float>(alignedLength);
        LocalTensor<int32_t> offsetInt32_ = offsetUb_.Get<int32_t>(alignedLength);
        DataCopyParams gm2UbParams;
        gm2UbParams.blockLen = length * sizeof(float);
        SetCopyParams(gm2UbParams);
        DataCopyLocal(offsetFp32_, offsetGm_[gmOffset], gm2UbParams);
        SetOffsetValue(resTensor, offsetInt32_, offsetFp32_, length);
    }

    __aicore__ inline void SetCopyParams(DataCopyParams params)
    {
        params.blockCount = 1;
        params.srcStride = 0;
        params.dstStride = 0;
    }

    template <typename T>
    __aicore__ inline T Max(T a, T b)
    {
        return a > b ? a : b;
    }

    template <typename T>
    __aicore__ inline T Min(T a, T b)
    {
        return a > b ? b : a;
    }

    template <typename T>
    __aicore__ inline T Align(T a, T b = 16)
    {
        return (a + b - 1) / b * b;
    }

    __aicore__ inline int Float32ToInt9(float value)
    {
        int intValue = static_cast<int>(value);
        int int9Value = Max<int>(-256, Min<int>(255, intValue));
        return int9Value;
    }
};
} // namespace AscendC

#endif // TRANS_QUANT_PARAM_V2_H