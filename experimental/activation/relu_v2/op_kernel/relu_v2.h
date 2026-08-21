/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACTIVATION_RELU_V2_H_
#define ACTIVATION_RELU_V2_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "relu_v2_tiling_data.h"
#include "relu_v2_tiling_key.h"

namespace NsReluV2 {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 4;
// QUEUE_DEPTH 必须与 BUFFER_NUM 一致(=4)：流式算子单次 DMA 搬运耗时远超指令延迟，
// 深度4允许更多 in-flight MTE2/MTE3 请求，充分隐藏 HBM 延迟；
// 各 dtype 的 bufferCoefficient 均按深度4预留 UB。
constexpr int32_t QUEUE_DEPTH = 4;
// INT64 路径 UB 开销大(56B/元素)，深度加到4会使 tile 过小，保持深度2。
constexpr int32_t BUFFER_NUM_I64 = 2;

template <typename T, int32_t BUF_NUM>
__aicore__ inline void CopyInFromGm(TQue<TPosition::VECIN, BUF_NUM>& inQueueX, GlobalTensor<T>& xGm, int64_t progress,
                                    int64_t tileLength, int64_t curTileLength)
{
    LocalTensor<T> xLocal = inQueueX.template AllocTensor<T>();
    if ((curTileLength * static_cast<int64_t>(sizeof(T))) % 32 == 0) {
        DataCopy(xLocal, xGm[progress * tileLength], curTileLength);
    } else {
        // 尾部tile非32B对齐（如shape=[1]）：DataCopy不支持非对齐长度，改用DataCopyPad
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(curTileLength * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(xLocal, xGm[progress * tileLength], copyParams, padParams);
    }
    inQueueX.EnQue(xLocal);
}

template <typename T, int32_t BUF_NUM>
__aicore__ inline void CopyOutToGm(TQue<TPosition::VECOUT, BUF_NUM>& outQueueY, GlobalTensor<T>& yGm, int64_t progress,
                                   int64_t tileLength, int64_t curTileLength)
{
    LocalTensor<T> yLocal = outQueueY.template DeQue<T>();
    if ((curTileLength * static_cast<int64_t>(sizeof(T))) % 32 == 0) {
        DataCopy(yGm[progress * tileLength], yLocal, curTileLength);
    } else {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(curTileLength * sizeof(T)), 0, 0, 0};
        DataCopyPad(yGm[progress * tileLength], yLocal, copyParams);
    }
    outQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void CopyInFromGm(TQueBind<TPosition::VECIN, TPosition::VECOUT, QUEUE_DEPTH>& inOutQueue,
                                    GlobalTensor<T>& xGm, int64_t progress, int64_t tileLength, int64_t curTileLength)
{
    LocalTensor<T> xLocal = inOutQueue.AllocTensor<T>();
    if ((curTileLength * static_cast<int64_t>(sizeof(T))) % 32 == 0) {
        DataCopy(xLocal, xGm[progress * tileLength], curTileLength);
    } else {
        // 尾部tile非32B对齐（如shape=[1]）：DataCopy不支持非对齐长度，改用DataCopyPad
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(curTileLength * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(xLocal, xGm[progress * tileLength], copyParams, padParams);
    }
    inOutQueue.template EnQue<QuePosition::GM, QuePosition::VECIN, T>(xLocal);
}

template <typename T>
__aicore__ inline void CopyOutToGm(TQueBind<TPosition::VECIN, TPosition::VECOUT, QUEUE_DEPTH>& inOutQueue,
                                   GlobalTensor<T>& yGm, int64_t progress, int64_t tileLength, int64_t curTileLength)
{
    LocalTensor<T> yLocal = inOutQueue.template DeQue<QuePosition::VECOUT, QuePosition::GM, T>();
    if ((curTileLength * static_cast<int64_t>(sizeof(T))) % 32 == 0) {
        DataCopy(yGm[progress * tileLength], yLocal, curTileLength);
    } else {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(curTileLength * sizeof(T)), 0, 0, 0};
        DataCopyPad(yGm[progress * tileLength], yLocal, copyParams);
    }
    inOutQueue.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void InitBlockGm(GlobalTensor<T>& xGm, GlobalTensor<T>& yGm, int64_t& blockLength,
                                   int64_t& tileLength, GM_ADDR x, GM_ADDR y, const ReluV2TilingData* tilingData)
{
    int64_t blockIdx = GetBlockIdx();
    if (blockIdx < tilingData->formerNum) {
        blockLength = tilingData->formerLength;
        int64_t offset = tilingData->formerLength * blockIdx;
        xGm.SetGlobalBuffer((__gm__ T*)x + offset, tilingData->formerLength);
        yGm.SetGlobalBuffer((__gm__ T*)y + offset, tilingData->formerLength);
    } else {
        blockLength = tilingData->tailLength;
        int64_t offset = tilingData->formerLength * tilingData->formerNum;
        xGm.SetGlobalBuffer((__gm__ T*)x + offset, tilingData->tailLength);
        yGm.SetGlobalBuffer((__gm__ T*)y + offset, tilingData->tailLength);
    }

    // L2 Cache 策略：Relu 是读一次写一次、无复用的流式算子。
    // L2 Cache 写策略：Relu 是读一次写一次的流式算子。
    // 实测对 50-600MB 用例应走 L2 写合并（如 Test_1852、bf16 58MB），达到~30GB/s写带宽；
    // 仅超大数据（>1GB）才 bypass L2 避免缓存污染。当前阈值设为 128MB，覆盖绝大多数场景。
    constexpr int64_t L2_BYPASS_BYTES = 1024 * 1024 * 1024; // 1GB
    int64_t totalElements = tilingData->formerNum * tilingData->formerLength + tilingData->tailLength;
    if (totalElements * static_cast<int64_t>(sizeof(T)) > L2_BYPASS_BYTES) {
        xGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        yGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    }

    tileLength = tilingData->tileLength;
}

template <typename Derived, typename T>
class KernelReluV2Base {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluV2TilingData* tilingData, TPipe* pipeIn)
    {
        pipe_ = pipeIn;
        InitBlockGm(xGm_, yGm_, blockLength_, tileLength_, x, y, tilingData);
        if (blockLength_ <= tileLength_) {
            // 小shape单tile路径：只按实际数据量分配单个TBuf，
            // 避免4缓冲队列的初始化与UB占用开销。
            pipe_->InitBuffer(smallBuf_, blockLength_ * sizeof(T));
        } else {
            pipe_->InitBuffer(inOutQueue_, BUFFER_NUM, tileLength_ * sizeof(T));
        }
    }

    __aicore__ inline void Process()
    {
        if (blockLength_ <= tileLength_) {
            // 单tile快速路径：数据一次搬完，使用单个TBuf，
            // 跳过队列管理与 EnQue/DeQue 事件同步，最小化小shape标量开销。
            ProcessSingleTile();
            return;
        }
        int64_t tileNum = (blockLength_ + tileLength_ - 1) / tileLength_;
        if (tileNum == 0) {
            return;
        }
        auto& self = static_cast<Derived&>(*this);
        int64_t tailTileLength = blockLength_ - (tileNum - 1) * tileLength_;
        for (int64_t i = 0; i < tileNum - 1; ++i) {
            CopyIn(i, tileLength_);
            self.Compute(tileLength_);
            CopyOut(i, tileLength_);
        }
        CopyIn(tileNum - 1, tailTileLength);
        self.Compute(tailTileLength);
        CopyOut(tileNum - 1, tailTileLength);
    }

    __aicore__ inline void ProcessSingleTile()
    {
        auto& self = static_cast<Derived&>(*this);
        LocalTensor<T> xLocal = smallBuf_.template Get<T>();
        if ((blockLength_ * static_cast<int64_t>(sizeof(T))) % 32 == 0) {
            DataCopy(xLocal, xGm_[0], blockLength_);
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(blockLength_ * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
            DataCopyPad(xLocal, xGm_[0], copyParams, padParams);
        }
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        self.ComputeSingleTile(xLocal, blockLength_);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        if ((blockLength_ * static_cast<int64_t>(sizeof(T))) % 32 == 0) {
            DataCopy(yGm_[0], xLocal, blockLength_);
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(blockLength_ * sizeof(T)), 0, 0, 0};
            DataCopyPad(yGm_[0], xLocal, copyParams);
        }
    }

    __aicore__ inline void CopyIn(int64_t progress, int64_t curTileLength)
    {
        CopyInFromGm(inOutQueue_, xGm_, progress, tileLength_, curTileLength);
    }

    __aicore__ inline void CopyOut(int64_t progress, int64_t curTileLength)
    {
        CopyOutToGm(inOutQueue_, yGm_, progress, tileLength_, curTileLength);
    }

protected:
    TPipe* pipe_;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, QUEUE_DEPTH> inOutQueue_;
    TBuf<TPosition::VECCALC> smallBuf_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    int64_t blockLength_ = 0;
    int64_t tileLength_ = 0;
};

template <typename T>
class KernelReluV2 : public KernelReluV2Base<KernelReluV2<T>, T> {
public:
    __aicore__ inline KernelReluV2() {}

    __aicore__ inline void Compute(int64_t curTileLength)
    {
        LocalTensor<T> xLocal = this->inOutQueue_.template DeQue<QuePosition::GM, QuePosition::VECIN, T>();
        Relu(xLocal, xLocal, curTileLength);
        this->inOutQueue_.template EnQue<QuePosition::VECOUT, QuePosition::GM, T>(xLocal);
    }

    __aicore__ inline void ComputeSingleTile(LocalTensor<T>& xLocal, int64_t length) { Relu(xLocal, xLocal, length); }
};

template <typename T, typename MidT>
class KernelReluV2Upcast : public KernelReluV2Base<KernelReluV2Upcast<T, MidT>, T> {
public:
    __aicore__ inline KernelReluV2Upcast() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluV2TilingData* tilingData, TPipe* pipeIn)
    {
        KernelReluV2Base<KernelReluV2Upcast<T, MidT>, T>::Init(x, y, tilingData, pipeIn);
        int64_t bufLen = (this->blockLength_ <= this->tileLength_) ? this->blockLength_ : this->tileLength_;
        this->pipe_->InitBuffer(tmpBufX_, bufLen * sizeof(MidT));
    }

    __aicore__ inline void Compute(int64_t curTileLength)
    {
        LocalTensor<T> xLocal = this->inOutQueue_.template DeQue<QuePosition::GM, QuePosition::VECIN, T>();
        LocalTensor<MidT> xMid = tmpBufX_.Get<MidT>();
        Cast(xMid, xLocal, RoundMode::CAST_NONE, curTileLength);
        Relu(xMid, xMid, curTileLength);
        Cast(xLocal, xMid, RoundMode::CAST_RINT, curTileLength);
        this->inOutQueue_.template EnQue<QuePosition::VECOUT, QuePosition::GM, T>(xLocal);
    }

    __aicore__ inline void ComputeSingleTile(LocalTensor<T>& xLocal, int64_t length)
    {
        LocalTensor<MidT> xMid = tmpBufX_.Get<MidT>();
        Cast(xMid, xLocal, RoundMode::CAST_NONE, length);
        Relu(xMid, xMid, length);
        Cast(xLocal, xMid, RoundMode::CAST_RINT, length);
    }

private:
    TBuf<TPosition::VECCALC> tmpBufX_;
};

template <typename T>
class KernelReluV2VectorInt64 {
    static_assert(std::is_same_v<T, int64_t>, "Only support int64_t");
    using Int16 = int16_t;
    using Int32 = int32_t;
    using Int64 = int64_t;

#define SCAST static_cast
#define VCAST(Type, vec) vec.template ReinterpretCast<Type>()

    // 小shape阈值：低于此值使用标量降级
    static constexpr int64_t SCALAR_THRESHOLD = 256;

public:
    __aicore__ inline KernelReluV2VectorInt64() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluV2TilingData* tilingData, TPipe* pipeIn)
    {
        pipe_ = pipeIn;
        InitBlockGm(xGm_, yGm_, blockLength_, tileLength_, x, y, tilingData);
        pipe_->InitBuffer(inQueueX_, BUFFER_NUM_I64, tileLength_ * sizeof(T));
        pipe_->InitBuffer(outQueueY_, BUFFER_NUM_I64, tileLength_ * sizeof(T));

        // 掩码scratch缓冲区：4N个Int16（每个int64视为4个int16）
        pipe_->InitBuffer(tempBuf_, tileLength_ * sizeof(T));
        // Gather提取+展开索引：4N个uint32_t（与数据无关，仅依赖位置）
        pipe_->InitBuffer(expandIdxBuf_, tileLength_ * 4 * sizeof(uint32_t));
        // 索引生成长度：取 tile 容量与实际数据量的较小值，避免小shape下
        // 按满 tile 容量生成索引而做大量无用向量运算（Compute的curTileLength永远≤此）。
        int64_t idxLenInt64 = (blockLength_ < tileLength_) ? blockLength_ : tileLength_;
        int64_t lenInt16 = idxLenInt64 * 4;
        // 向量化生成展开索引，替代逐元素SetValue标量循环。
        // 目标：expandIdx[k] = ((k>>2)*4+3)*sizeof(int16) = (k>>2)*8 + 6
        // 此处生成 (k>>2)*8，剩余的 +6 由Gather的srcBaseAddr(=6字节)吸收。
        LocalTensor<Int32> idxI32 = expandIdxBuf_.Get<Int32>();
        CreateVecIndex(idxI32, SCAST<Int32>(0), SCAST<uint32_t>(lenInt16)); // 0,1,2,...
        ShiftRight(idxI32, idxI32, SCAST<Int32>(2), lenInt16);              // k>>2
        ShiftLeft(idxI32, idxI32, SCAST<Int32>(3), lenInt16);               // (k>>2)*8
    }

    __aicore__ inline void Process()
    {
        if (blockLength_ <= SCALAR_THRESHOLD) {
            ProcessScalar();
            return;
        }
        int64_t tileNum = (blockLength_ + tileLength_ - 1) / tileLength_;
        if (tileNum == 0) {
            return;
        }
        int64_t tailTileLength = blockLength_ - (tileNum - 1) * tileLength_;
        for (int64_t i = 0; i < tileNum - 1; ++i) {
            CopyIn(i, tileLength_);
            Compute(tileLength_);
            CopyOut(i, tileLength_);
        }
        CopyIn(tileNum - 1, tailTileLength);
        Compute(tailTileLength);
        CopyOut(tileNum - 1, tailTileLength);
    }

    // 标量降级处理：小shape场景
    __aicore__ inline void ProcessScalar()
    {
        int64_t tileNum = (blockLength_ + tileLength_ - 1) / tileLength_;

        for (int64_t tileIdx = 0; tileIdx < tileNum; ++tileIdx) {
            int64_t curTileLength = (tileIdx == tileNum - 1) ? (blockLength_ - tileIdx * tileLength_) : tileLength_;

            LocalTensor<Int64> xLocal = inQueueX_.AllocTensor<Int64>();
            DataCopyExtParams copyInParams{1, static_cast<uint32_t>(curTileLength * sizeof(Int64)), 0, 0, 0};
            DataCopyPadExtParams<Int64> padParams{false, 0, 0, 0};
            DataCopyPad(xLocal, xGm_[tileIdx * tileLength_], copyInParams, padParams);

            int64_t i = 0;
            for (; i + 3 < curTileLength; i += 4) {
                Int64 val0 = xLocal.GetValue(i);
                Int64 val1 = xLocal.GetValue(i + 1);
                Int64 val2 = xLocal.GetValue(i + 2);
                Int64 val3 = xLocal.GetValue(i + 3);
                xLocal.SetValue(i, (val0 > 0) ? val0 : SCAST<Int64>(0));
                xLocal.SetValue(i + 1, (val1 > 0) ? val1 : SCAST<Int64>(0));
                xLocal.SetValue(i + 2, (val2 > 0) ? val2 : SCAST<Int64>(0));
                xLocal.SetValue(i + 3, (val3 > 0) ? val3 : SCAST<Int64>(0));
            }
            for (; i < curTileLength; ++i) {
                Int64 val = xLocal.GetValue(i);
                xLocal.SetValue(i, (val > 0) ? val : SCAST<Int64>(0));
            }

            DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(curTileLength * sizeof(Int64)), 0, 0, 0};
            DataCopyPad(yGm_[tileIdx * tileLength_], xLocal, copyOutParams);
            inQueueX_.FreeTensor(xLocal);
        }
    }

    __aicore__ inline void CopyIn(int64_t progress, int64_t curTileLengthInt64)
    {
        CopyInFromGm(inQueueX_, xGm_, progress, tileLength_, curTileLengthInt64);
    }

    __aicore__ inline void Compute(int64_t curTileLengthInt64)
    {
        LocalTensor<T> xLocal = inQueueX_.DeQue<T>();
        LocalTensor<T> yLocal = outQueueY_.AllocTensor<T>();

        int64_t curLenInt16 = curTileLengthInt64 * 4;
        auto xInt16 = VCAST(Int16, xLocal);
        auto yInt16 = VCAST(Int16, yLocal);

        LocalTensor<Int16> rawMask = tempBuf_.Get<Int16>();
        LocalTensor<uint32_t> expandIdx = expandIdxBuf_.Get<uint32_t>();

        // 步骤1: Gather提取+展开合一。每个输出int16取其所属int64的最高int16
        //   （含bit15=int64真符号），得到4N个符号值。全程int16同视图，无跨视图hazard。
        //   srcBaseAddr=6字节：吸收索引中 +3个int16偏移（见Init）。
        Gather(rawMask, xInt16, expandIdx, 6U, SCAST<uint32_t>(curLenInt16));

        // 步骤2: 算术右移15位提取符号（正数→0x0000，负数→0xFFFF）。
        ShiftRight(rawMask, rawMask, SCAST<Int16>(15), curLenInt16);

        // 步骤3: 取反（正数→0xFFFF全一掩码，负数→0x0000全零掩码），int16 unary in-place安全。
        Not(rawMask, rawMask, curLenInt16);

        // 步骤4: 位与。dst=yInt16、src1=xInt16、src2=rawMask 三者互不相同。
        //   正数：x & 0xFFFF... = x（保留原值）；负数：x & 0x0000... = 0（归零）。
        And(yInt16, xInt16, rawMask, curLenInt16);

        outQueueY_.EnQue<T>(yLocal);
        inQueueX_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int64_t progress, int64_t curTileLengthInt64)
    {
        CopyOutToGm(outQueueY_, yGm_, progress, tileLength_, curTileLengthInt64);
    }

private:
    TPipe* pipe_;
    TQue<TPosition::VECIN, BUFFER_NUM_I64> inQueueX_;
    TQue<TPosition::VECOUT, BUFFER_NUM_I64> outQueueY_;
    TBuf<TPosition::VECCALC> tempBuf_;      // 掩码scratch缓冲区：4N个Int16
    TBuf<TPosition::VECCALC> expandIdxBuf_; // Gather提取+展开索引：4N个uint32_t
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    int64_t blockLength_ = 0;
    int64_t tileLength_ = 0;

#undef SCAST
#undef VCAST
};

} // namespace NsReluV2

#endif
