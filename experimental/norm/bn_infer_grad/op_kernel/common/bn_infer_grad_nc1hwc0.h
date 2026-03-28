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
 * \file bn_infer_grad_nc1hwc0.h
 * \brief BnInferGrad NC1HWC0 分支 Kernel 实现（TilingKey=1）
 *
 * 处理 NC1HWC0 五维格式 (N, C1, H, W, C0)。
 * 按 (n, c1) 任务分配到多核，每个任务处理 H*W*C0 个元素。
 * 对每个 (n, c1) 任务，按 tileHW 切分 H*W 维度，
 * 每个 tile 处理 tileHW * C0 个元素。
 *
 * inv_std[c] = scale[c] / sqrt(batch_variance[c] + epsilon)
 * 在 Init 中从 scale 和 batch_variance 计算。
 * 对每个 c1 段，取 inv_std[c1*C0 : c1*C0+C0]，
 * 重复 tileHW 次展开后与 grads 逐元素相乘。
 *
 * 迭代三：完整实现，fp32/fp16/bf16 + 多核 + 边界处理。
 * fp16/bf16 使用混合精度路径：Cast(input->fp32) -> Mul -> Cast(fp32->output)。
 */
#ifndef BN_INFER_GRAD_NC1HWC0_H
#define BN_INFER_GRAD_NC1HWC0_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "bn_infer_grad_tiling_data.h"
#include "bn_infer_grad_tiling_key.h"

namespace NsBnInferGrad {

using namespace AscendC;

template <typename T>
class BnInferGradNc1hwc0 {
public:
    __aicore__ inline BnInferGradNc1hwc0() {}

    __aicore__ inline void Init(GM_ADDR grads, GM_ADDR scale, GM_ADDR batchVariance,
                                 GM_ADDR xBackprop, GM_ADDR workspace,
                                 const BnInferGradTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t gmOffset, int64_t curTileElems);
    __aicore__ inline void Compute(int64_t c1Idx, int64_t curTileElems);
    __aicore__ inline void CopyOut(int64_t gmOffset, int64_t curTileElems);
    __aicore__ inline void ComputeInvStd();
    __aicore__ inline void ExpandInvStdForC1(int64_t c1Idx, int64_t curTileHW);

private:
    static constexpr bool IS_FP32 = sizeof(T) == sizeof(float);

    TPipe pipe;
    TQue<QuePosition::VECIN, 2> inQueueGrads;
    TQue<QuePosition::VECOUT, 2> outQueueResult;
    TBuf<QuePosition::VECCALC> invStdBuf;         // 完整 inv_std (alignedC)
    TBuf<QuePosition::VECCALC> invStdExpandBuf;    // 展开后 inv_std (tileHW * C0)
    TBuf<QuePosition::VECCALC> scaleBuf;           // scale 数据
    TBuf<QuePosition::VECCALC> varianceBuf;        // batch_variance 数据
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
    int64_t N_ = 0;
    int64_t C1_ = 0;
    int64_t C0_ = 0;
    int64_t alignedC_ = 0;
    int64_t alignedC0_ = 0;
    float epsilon_ = 0.0001f;

    // UB 切分参数
    int64_t tileHW_ = 0;
    int64_t numTilesHW_ = 0;
    int64_t lastTileHW_ = 0;
    int64_t tileLen_ = 0;  // = tileHW_ * C0_

    // 多核参数
    int64_t taskStart_ = 0;    // 当前核的起始任务索引
    int64_t coreTasks_ = 0;    // 当前核处理的任务数
};

template <typename T>
__aicore__ inline void BnInferGradNc1hwc0<T>::Init(GM_ADDR grads, GM_ADDR scale, GM_ADDR batchVariance,
                                                     GM_ADDR xBackprop, GM_ADDR workspace,
                                                     const BnInferGradTilingData* tilingData)
{
    totalElements_ = tilingData->totalElements;
    channelSize_ = tilingData->channelSize;
    spatialSize_ = tilingData->spatialSize;
    N_ = tilingData->N;
    C1_ = tilingData->C1;
    C0_ = tilingData->C0;
    alignedC_ = tilingData->alignedC;
    alignedC0_ = tilingData->alignedC0;
    tileHW_ = tilingData->tileHW;
    numTilesHW_ = tilingData->numTilesHW;
    lastTileHW_ = tilingData->lastTileHW;
    tileLen_ = tileHW_ * C0_;

    // 从 TilingData 恢复 epsilon
    int64_t epsBits = tilingData->epsilonBits;
    float epsVal;
    *(reinterpret_cast<int32_t*>(&epsVal)) = static_cast<int32_t>(epsBits);
    epsilon_ = epsVal;

    // 多核切分：按任务（N*C1 个任务）分配
    int64_t blockIdx = AscendC::GetBlockIdx();
    int64_t tasksPerCore = tilingData->tasksPerCore;
    int64_t tailCoreTasks = tilingData->tailCoreTasks;
    int64_t usedCoreNum = tilingData->usedCoreNum;

    taskStart_ = blockIdx * tasksPerCore;
    if (blockIdx < usedCoreNum - 1) {
        coreTasks_ = tasksPerCore;
    } else {
        coreTasks_ = tailCoreTasks;
    }

    // 边界处理：当前核无任务或空 tensor，跳过 buffer 初始化
    if (coreTasks_ <= 0 || tileLen_ <= 0) {
        coreTasks_ = 0;
        return;
    }

    // 设置 GM 指针
    gradsGM.SetGlobalBuffer((__gm__ T*)grads, totalElements_);
    outputGM.SetGlobalBuffer((__gm__ T*)xBackprop, totalElements_);
    scaleGM.SetGlobalBuffer((__gm__ float*)scale, channelSize_);
    batchVarianceGM.SetGlobalBuffer((__gm__ float*)batchVariance, channelSize_);

    // 初始化 Buffer
    // invStdExpandBuf 大小 = tileHW * C0（可能 lastTileHW < tileHW，用最大值分配）
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

    // 计算 inv_std（对所有通道）
    ComputeInvStd();
}

template <typename T>
__aicore__ inline void BnInferGradNc1hwc0<T>::ComputeInvStd()
{
    // 搬入 scale 和 batch_variance
    LocalTensor<float> scaleLocal = scaleBuf.Get<float>();
    LocalTensor<float> varLocal = varianceBuf.Get<float>();
    LocalTensor<float> invStdLocal = invStdBuf.Get<float>();

    uint32_t count = static_cast<uint32_t>(alignedC_);

    // 零初始化 buffer，确保 padding 区域为 0，避免 NaN
    Duplicate(scaleLocal, 0.0f, count);
    Duplicate(varLocal, 0.0f, count);
    PipeBarrier<PIPE_ALL>();

    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(channelSize_ * static_cast<int64_t>(sizeof(float)));
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
__aicore__ inline void BnInferGradNc1hwc0<T>::ExpandInvStdForC1(int64_t c1Idx, int64_t curTileHW)
{
    // 将 inv_std[c1*C0 : c1*C0+C0] 重复 curTileHW 次，填充 invStdExpanded
    LocalTensor<float> invStdLocal = invStdBuf.Get<float>();
    LocalTensor<float> invStdExpanded = invStdExpandBuf.Get<float>();

    int64_t c1Offset = c1Idx * C0_;

    // 对齐的 curTileHW * C0 长度
    int64_t expandLen = curTileHW * C0_;

    // 对每个 hw 位置，复制 C0 个 inv_std 值
    // 由于 C0 通常为 16 或 32（>=8 且为 32 字节对齐），可以直接 DataCopy
    uint32_t copyLen = static_cast<uint32_t>(C0_);
    for (int64_t r = 0; r < curTileHW; r++) {
        DataCopy(invStdExpanded[r * C0_], invStdLocal[c1Offset], copyLen);
    }
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void BnInferGradNc1hwc0<T>::CopyIn(int64_t gmOffset, int64_t curTileElems)
{
    LocalTensor<T> gradsLocal = inQueueGrads.template AllocTensor<T>();

    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(curTileElems * static_cast<int64_t>(sizeof(T)));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPad(gradsLocal, gradsGM[gmOffset], copyParams, {false, 0, 0, 0});

    inQueueGrads.EnQue(gradsLocal);
}

template <typename T>
__aicore__ inline void BnInferGradNc1hwc0<T>::Compute(int64_t c1Idx, int64_t curTileElems)
{
    // 对齐到 8 的整数倍
    int64_t alignedTileElems = ((curTileElems + 7) / 8) * 8;
    uint32_t computeCount = static_cast<uint32_t>(alignedTileElems);

    LocalTensor<T> gradsLocal = inQueueGrads.template DeQue<T>();
    LocalTensor<T> resultLocal = outQueueResult.template AllocTensor<T>();

    LocalTensor<float> invStdExpanded = invStdExpandBuf.Get<float>();

    if constexpr (IS_FP32) {
        LocalTensor<float> gradsFp32 = gradsLocal.template ReinterpretCast<float>();
        LocalTensor<float> resultFp32 = resultLocal.template ReinterpretCast<float>();
        Mul(resultFp32, gradsFp32, invStdExpanded, computeCount);
    } else {
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
__aicore__ inline void BnInferGradNc1hwc0<T>::CopyOut(int64_t gmOffset, int64_t curTileElems)
{
    LocalTensor<T> resultLocal = outQueueResult.template DeQue<T>();

    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(curTileElems * static_cast<int64_t>(sizeof(T)));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPad(outputGM[gmOffset], resultLocal, copyParams);

    outQueueResult.FreeTensor(resultLocal);
}

template <typename T>
__aicore__ inline void BnInferGradNc1hwc0<T>::Process()
{
    // 边界处理：空 tensor 或当前核无任务
    if (coreTasks_ <= 0) {
        return;
    }
    // 遍历当前核分配的任务
    for (int64_t t = 0; t < coreTasks_; t++) {
        int64_t taskIdx = taskStart_ + t;
        // 任务索引 -> (n, c1)
        int64_t c1Idx = taskIdx % C1_;

        // 该任务在 GM 中的基地址偏移（按展平后的元素索引）
        // NC1HWC0 layout: [N, C1, H, W, C0]
        // 第 taskIdx 个 (n, c1) 块的起始偏移 = taskIdx * spatialSize * C0
        int64_t taskBaseOffset = taskIdx * spatialSize_ * C0_;

        // 按 tileHW 切分 H*W 维度
        for (int64_t tileIdx = 0; tileIdx < numTilesHW_; tileIdx++) {
            int64_t curTileHW = (tileIdx < numTilesHW_ - 1) ? tileHW_ : lastTileHW_;
            int64_t curTileElems = curTileHW * C0_;
            int64_t tileOffset = tileIdx * tileHW_ * C0_;
            int64_t gmOffset = taskBaseOffset + tileOffset;

            // 展开当前 c1 段的 inv_std
            ExpandInvStdForC1(c1Idx, curTileHW);

            CopyIn(gmOffset, curTileElems);
            Compute(c1Idx, curTileElems);
            CopyOut(gmOffset, curTileElems);
        }
    }
}

} // namespace NsBnInferGrad

#endif // BN_INFER_GRAD_NC1HWC0_H
