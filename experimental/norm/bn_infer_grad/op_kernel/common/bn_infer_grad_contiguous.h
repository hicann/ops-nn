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
* 我们正常的版权申明，下面是我们的备注
*
* NOTE: Portions of this code were AI-generated and have been
* technically reviewed for functional accuracy and security
*/

/*!
 * \file bn_infer_grad_contiguous.h
 * \brief BnInferGrad CONTIGUOUS 分支 Kernel 实现（TilingKey=0）
 *
 * 处理 NCHW/NHWC 格式。将 grads 展平后按 tile 切分，
 * 每个 tile 内将 inv_std 按通道展开后执行逐元素乘法。
 *
 * inv_std[c] = scale[c] / sqrt(batch_variance[c] + epsilon) 在 Init 中计算。
 *
 * 迭代三：支持 fp32/fp16/bf16 + NCHW/NHWC + 多核 + 边界处理。
 * fp16/bf16 使用混合精度路径：Cast(input->fp32) -> Mul -> Cast(fp32->output)。
 */
#ifndef BN_INFER_GRAD_CONTIGUOUS_H
#define BN_INFER_GRAD_CONTIGUOUS_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "bn_infer_grad_tiling_data.h"
#include "bn_infer_grad_tiling_key.h"

namespace NsBnInferGrad {

using namespace AscendC;

template <typename T>
class BnInferGradContiguous {
public:
    __aicore__ inline BnInferGradContiguous() {}

    __aicore__ inline void Init(GM_ADDR grads, GM_ADDR scale, GM_ADDR batchVariance,
                                 GM_ADDR xBackprop, GM_ADDR workspace,
                                 const BnInferGradTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t tileIdx);
    __aicore__ inline void Compute(int64_t tileIdx);
    __aicore__ inline void CopyOut(int64_t tileIdx);
    __aicore__ inline void ExpandInvStd(int64_t tileStart, int64_t tileLen);
    __aicore__ inline void ComputeInvStd();

private:
    static constexpr bool IS_FP32 = sizeof(T) == sizeof(float);

    TPipe pipe;
    TQue<QuePosition::VECIN, 2> inQueueGrads;
    TQue<QuePosition::VECOUT, 2> outQueueResult;
    TBuf<QuePosition::VECCALC> invStdBuf;
    TBuf<QuePosition::VECCALC> invStdExpandBuf;
    TBuf<QuePosition::VECCALC> scaleBuf;        // scale 数据
    TBuf<QuePosition::VECCALC> varianceBuf;     // batch_variance 数据
    // 以下仅在混合精度路径（fp16/bf16）中使用
    TBuf<QuePosition::VECCALC> castBuf;
    TBuf<QuePosition::VECCALC> resultFp32Buf;

    GlobalTensor<T> gradsGM;
    GlobalTensor<T> outputGM;
    GlobalTensor<float> scaleGM;
    GlobalTensor<float> batchVarianceGM;

    // Tiling 参数
    int64_t totalElements_ = 0;
    int64_t channelSize_ = 0;
    int64_t spatialSize_ = 0;
    int64_t formatMode_ = 0;
    int64_t tileLen_ = 0;
    int64_t numTiles_ = 0;
    int64_t lastTileLen_ = 0;
    int64_t alignedC_ = 0;
    float epsilon_ = 0.0001f;

    // 多核参数
    int64_t startOffset_ = 0;
    int64_t coreElements_ = 0;
};

template <typename T>
__aicore__ inline void BnInferGradContiguous<T>::Init(GM_ADDR grads, GM_ADDR scale, GM_ADDR batchVariance,
                                                       GM_ADDR xBackprop, GM_ADDR workspace,
                                                       const BnInferGradTilingData* tilingData)
{
    totalElements_ = tilingData->totalElements;
    channelSize_ = tilingData->channelSize;
    spatialSize_ = tilingData->spatialSize;
    formatMode_ = tilingData->formatMode;
    tileLen_ = tilingData->tileLen;
    alignedC_ = tilingData->alignedC;

    // 从 TilingData 恢复 epsilon
    int64_t epsBits = tilingData->epsilonBits;
    float epsVal;
    // 使用位级拷贝：低 32 位存储 float
    *(reinterpret_cast<int32_t*>(&epsVal)) = static_cast<int32_t>(epsBits);
    epsilon_ = epsVal;

    // 多核切分
    int64_t blockIdx = AscendC::GetBlockIdx();
    int64_t elemsPerCore = tilingData->elemsPerCore;
    int64_t tailCoreElems = tilingData->tailCoreElems;
    int64_t usedCoreNum = tilingData->usedCoreNum;

    startOffset_ = blockIdx * elemsPerCore;
    if (blockIdx < usedCoreNum - 1) {
        coreElements_ = elemsPerCore;
    } else {
        coreElements_ = tailCoreElems;
    }

    // 重新计算当前核的 tile 数
    numTiles_ = (coreElements_ + tileLen_ - 1) / tileLen_;
    lastTileLen_ = coreElements_ - (numTiles_ - 1) * tileLen_;
    if (lastTileLen_ <= 0) {
        lastTileLen_ = tileLen_;
    }

    // 边界处理：当前核无数据或空 tensor，跳过 buffer 初始化
    if (coreElements_ <= 0 || tileLen_ <= 0) {
        numTiles_ = 0;
        return;
    }

    // 设置 GM 指针
    gradsGM.SetGlobalBuffer((__gm__ T*)grads, totalElements_);
    outputGM.SetGlobalBuffer((__gm__ T*)xBackprop, totalElements_);
    scaleGM.SetGlobalBuffer((__gm__ float*)scale, channelSize_);
    batchVarianceGM.SetGlobalBuffer((__gm__ float*)batchVariance, channelSize_);

    // 初始化 Buffer
    pipe.InitBuffer(inQueueGrads, 2, tileLen_ * static_cast<int64_t>(sizeof(T)));
    pipe.InitBuffer(outQueueResult, 2, tileLen_ * static_cast<int64_t>(sizeof(T)));
    pipe.InitBuffer(invStdBuf, alignedC_ * static_cast<int64_t>(sizeof(float)));
    pipe.InitBuffer(invStdExpandBuf, tileLen_ * static_cast<int64_t>(sizeof(float)));
    pipe.InitBuffer(scaleBuf, alignedC_ * static_cast<int64_t>(sizeof(float)));
    pipe.InitBuffer(varianceBuf, alignedC_ * static_cast<int64_t>(sizeof(float)));

    if constexpr (!IS_FP32) {
        pipe.InitBuffer(castBuf, tileLen_ * static_cast<int64_t>(sizeof(float)));
        pipe.InitBuffer(resultFp32Buf, tileLen_ * static_cast<int64_t>(sizeof(float)));
    }

    // 计算 inv_std
    ComputeInvStd();
}

template <typename T>
__aicore__ inline void BnInferGradContiguous<T>::ComputeInvStd()
{
    // 搬入 scale 和 batch_variance
    LocalTensor<float> scaleLocal = scaleBuf.Get<float>();
    LocalTensor<float> varLocal = varianceBuf.Get<float>();
    LocalTensor<float> invStdLocal = invStdBuf.Get<float>();

    uint32_t count = static_cast<uint32_t>(alignedC_);

    // 零初始化 buffer，确保 padding 区域（channelSize..alignedC-1）为 0
    // 避免 garbage 导致 Sqrt(负数) 产生 NaN
    Duplicate(scaleLocal, 0.0f, count);
    Duplicate(varLocal, 0.0f, count);
    PipeBarrier<PIPE_ALL>();

    // 使用 alignedC（对齐到 32 字节）搬运，确保 DMA 至少一个完整数据块
    uint32_t copyBytes = static_cast<uint32_t>(channelSize_ * static_cast<int64_t>(sizeof(float)));

    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = copyBytes;
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;

    DataCopyPad(scaleLocal, scaleGM, copyParams, {false, 0, 0, 0});
    DataCopyPad(varLocal, batchVarianceGM, copyParams, {false, 0, 0, 0});
    PipeBarrier<PIPE_ALL>();

    // 计算 inv_std[c] = scale[c] / sqrt(batch_variance[c] + epsilon)
    Adds(varLocal, varLocal, epsilon_, count);
    Sqrt(varLocal, varLocal, count);
    Div(invStdLocal, scaleLocal, varLocal, count);
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void BnInferGradContiguous<T>::ExpandInvStd(int64_t tileStart, int64_t curTileLen)
{
    LocalTensor<float> invStdLocal = invStdBuf.Get<float>();
    LocalTensor<float> invStdExpanded = invStdExpandBuf.Get<float>();

    // Ascend C 向量指令（Duplicate/DataCopy）要求目标地址 32 字节对齐（即 float 偏移为 8 的倍数）。
    // 对于非对齐的 segment 边界，使用标量 SetValue 逐元素填充。
    constexpr int64_t ALIGN_ELEMS = 8;

    if (formatMode_ == 0) {
        // NCHW 格式：通道索引 c = (globalIdx / spatialSize) % channelSize
        // 同一通道的 spatialSize 个元素连续存储
        if (spatialSize_ >= ALIGN_ELEMS && (spatialSize_ % ALIGN_ELEMS == 0)) {
            // spatialSize 对齐到 8：所有 segment 起始位置都是 8 的倍数，可以安全使用 Duplicate
            int64_t pos = 0;
            int64_t globalIdx = tileStart;
            while (pos < curTileLen) {
                int64_t c = (globalIdx / spatialSize_) % channelSize_;
                int64_t remaining = spatialSize_ - (globalIdx % spatialSize_);
                int64_t segLen = remaining;
                if (segLen > curTileLen - pos) {
                    segLen = curTileLen - pos;
                }
                // segLen >= 8 因为 spatialSize 是 8 的倍数且 remaining <= spatialSize
                // 对于最后一个 segment 可能 segLen < 8，用 max(segLen, 8) 并确保不越界
                int64_t dupCount = segLen < ALIGN_ELEMS ? ALIGN_ELEMS : segLen;
                Duplicate(invStdExpanded[pos], invStdLocal.GetValue(c), static_cast<uint32_t>(dupCount));
                pos += segLen;
                globalIdx += segLen;
            }
        } else {
            // spatialSize 非对齐：segment 边界不在 8 的倍数位置，
            // 使用标量 SetValue 逐元素填充以确保正确性
            for (int64_t i = 0; i < curTileLen; i++) {
                int64_t globalIdx = tileStart + i;
                int64_t c = (globalIdx / spatialSize_) % channelSize_;
                invStdExpanded.SetValue(i, invStdLocal.GetValue(c));
            }
        }
    } else if (formatMode_ == 1) {
        // NHWC 格式：通道索引 c = globalIdx % channelSize
        int64_t numRepeats = curTileLen / channelSize_;
        int64_t remainElems = curTileLen % channelSize_;

        if (channelSize_ >= ALIGN_ELEMS && (channelSize_ % ALIGN_ELEMS == 0)) {
            // C 对齐到 8，可以安全使用 DataCopy
            uint32_t copyLen = static_cast<uint32_t>(channelSize_);
            for (int64_t r = 0; r < numRepeats; r++) {
                DataCopy(invStdExpanded[r * channelSize_], invStdLocal, copyLen);
            }
            // 处理不足一组 C 的尾部（remainElems 一定是 C 的子集，也对齐到 8）
            if (remainElems > 0) {
                // remainElems < channelSize，使用标量填充
                int64_t base = numRepeats * channelSize_;
                for (int64_t i = 0; i < remainElems; i++) {
                    int64_t c = (tileStart + base + i) % channelSize_;
                    invStdExpanded.SetValue(base + i, invStdLocal.GetValue(c));
                }
            }
        } else if (channelSize_ == 1) {
            // C=1：所有元素乘以同一个 inv_std 值
            int64_t dupCount = curTileLen < ALIGN_ELEMS ? ALIGN_ELEMS : curTileLen;
            Duplicate(invStdExpanded, invStdLocal.GetValue(0), static_cast<uint32_t>(dupCount));
        } else {
            // C 不对齐到 8：使用标量 SetValue 逐元素填充
            for (int64_t i = 0; i < curTileLen; i++) {
                int64_t c = (tileStart + i) % channelSize_;
                invStdExpanded.SetValue(i, invStdLocal.GetValue(c));
            }
        }
    }
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void BnInferGradContiguous<T>::CopyIn(int64_t tileIdx)
{
    int64_t curTileLen = (tileIdx < numTiles_ - 1) ? tileLen_ : lastTileLen_;
    int64_t gmOffset = startOffset_ + tileIdx * tileLen_;

    LocalTensor<T> gradsLocal = inQueueGrads.template AllocTensor<T>();

    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(curTileLen * static_cast<int64_t>(sizeof(T)));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPad(gradsLocal, gradsGM[gmOffset], copyParams, {false, 0, 0, 0});

    inQueueGrads.EnQue(gradsLocal);
}

template <typename T>
__aicore__ inline void BnInferGradContiguous<T>::Compute(int64_t tileIdx)
{
    int64_t curTileLen = (tileIdx < numTiles_ - 1) ? tileLen_ : lastTileLen_;
    int64_t tileStart = startOffset_ + tileIdx * tileLen_;

    // 对齐 curTileLen 到 8 的整数倍（向量操作要求）
    int64_t alignedTileLen = ((curTileLen + 7) / 8) * 8;
    uint32_t computeCount = static_cast<uint32_t>(alignedTileLen);

    LocalTensor<T> gradsLocal = inQueueGrads.template DeQue<T>();
    LocalTensor<T> resultLocal = outQueueResult.template AllocTensor<T>();

    // 展开 inv_std
    ExpandInvStd(tileStart, alignedTileLen);
    LocalTensor<float> invStdExpanded = invStdExpandBuf.Get<float>();

    if constexpr (IS_FP32) {
        // fp32 路径：直接乘
        LocalTensor<float> gradsFp32 = gradsLocal.template ReinterpretCast<float>();
        LocalTensor<float> resultFp32 = resultLocal.template ReinterpretCast<float>();
        Mul(resultFp32, gradsFp32, invStdExpanded, computeCount);
    } else {
        // fp16/bf16 混合精度路径：Cast -> Mul -> Cast
        LocalTensor<float> castLocal = castBuf.Get<float>();
        LocalTensor<float> resFp32Local = resultFp32Buf.Get<float>();

        Cast(castLocal, gradsLocal, RoundMode::CAST_NONE, computeCount);
        Mul(resFp32Local, castLocal, invStdExpanded, computeCount);
        Cast(resultLocal, resFp32Local, RoundMode::CAST_ROUND, computeCount);
    }

    outQueueResult.template EnQue<T>(resultLocal);
    inQueueGrads.FreeTensor(gradsLocal);
}

template <typename T>
__aicore__ inline void BnInferGradContiguous<T>::CopyOut(int64_t tileIdx)
{
    int64_t curTileLen = (tileIdx < numTiles_ - 1) ? tileLen_ : lastTileLen_;
    int64_t gmOffset = startOffset_ + tileIdx * tileLen_;

    LocalTensor<T> resultLocal = outQueueResult.template DeQue<T>();

    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(curTileLen * static_cast<int64_t>(sizeof(T)));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPad(outputGM[gmOffset], resultLocal, copyParams);

    outQueueResult.FreeTensor(resultLocal);
}

template <typename T>
__aicore__ inline void BnInferGradContiguous<T>::Process()
{
    // 边界处理：空 tensor 或当前核无数据
    if (coreElements_ <= 0 || numTiles_ <= 0) {
        return;
    }
    for (int64_t i = 0; i < numTiles_; i++) {
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
}

} // namespace NsBnInferGrad

#endif // BN_INFER_GRAD_CONTIGUOUS_H
