/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_matmul_gelu_kernel.h
 * \brief AICore kernel implementation for y = gelu(x @ weight^T + bias).
 */

#ifndef OP_KERNEL_FUSED_MATMUL_GELU_H_
#define OP_KERNEL_FUSED_MATMUL_GELU_H_

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace FusedMatmulGelu {
using namespace AscendC;

constexpr uint64_t DATA_PER_BLOCK_B16 = 16;
constexpr uint64_t DATA_PER_BLOCK_B32 = 8;

constexpr float GELU_BETA = 0.044715f;
constexpr float GELU_ALPHA = -1.5957691f;
constexpr float GELU_ONE = 1.0f;

constexpr MatmulConfig MATMUL_CFG{false, false, true, 0, 0, 0, false, false, false, false,
                                  false, 0,     0,    0, 0, 0, 0,     0,     true};

template <typename T>
__aicore__ inline T CeilDiv(T a, T b)
{
    return b == 0 ? 0 : (a + b - 1) / b;
}

template <typename T>
__aicore__ inline T AlignUp(T a, T b)
{
    return b == 0 ? a : CeilDiv(a, b) * b;
}

template <typename T, uint64_t APPROXIMATE>
class FusedMatmulGeluOp {
public:
    __aicore__ inline FusedMatmulGeluOp() = default;

    __aicore__ inline void Init(const FusedMatmulGeluTilingData& __restrict tilingData, GM_ADDR x, GM_ADDR weight,
                                GM_ADDR bias, GM_ADDR y, GM_ADDR userWorkspace, TPipe* pipe);

    __aicore__ inline void Process();

private:
    __aicore__ inline void InitTilingData(const FusedMatmulGeluTilingData& __restrict tilingData);
    __aicore__ inline void MatmulProcess();
    __aicore__ inline void VectorProcess();
    __aicore__ inline void VectorChunkProcess(uint64_t offset, uint64_t count);
    __aicore__ inline void ComputeGeluTanh(uint64_t count);

private:
    using XType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using WeightType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T, true>;
    using YType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using BiasType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;

    matmul::MatmulImpl<XType, WeightType, YType, BiasType, MATMUL_CFG> mm_;

    TPipe* pipe_ = nullptr;

    GlobalTensor<T> xGm_;
    GlobalTensor<T> weightGm_;
    GlobalTensor<T> biasGm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<T> mmOutGm_;

    TBuf<TPosition::VECCALC> calcBuf_;
    LocalTensor<T> xLocal_;
    LocalTensor<T> biasLocal_;
    LocalTensor<float> xFp32_;
    LocalTensor<float> workFp32_;

    uint64_t m_ = 0;
    uint64_t k_ = 0;
    uint64_t n_ = 0;
    uint64_t totalElement_ = 0;
    uint64_t bufSize_ = 0;
    uint64_t cubeCoreNum_ = 0;
    uint64_t cubeCoreNumAligned_ = 0;
    uint64_t vecCoreNum_ = 0;
    uint64_t vecBlockIdx_ = 0;
    uint64_t vecTasksPerCore_ = 0;
    uint64_t vecTasksTailCore_ = 0;
    uint64_t elemsPerVecLoop_ = 0;
    uint64_t hasBias_ = 0;

    const TCubeTiling* __restrict mmTilingData_ = nullptr;

    uint64_t blockIdx_ = 0;
    uint64_t coreIdx_ = 0;
};

template <typename T, uint64_t APPROXIMATE>
__aicore__ inline void FusedMatmulGeluOp<T, APPROXIMATE>::InitTilingData(
    const FusedMatmulGeluTilingData& __restrict tilingData)
{
    m_ = tilingData.m;
    k_ = tilingData.k;
    n_ = tilingData.n;
    totalElement_ = tilingData.totalElement;
    bufSize_ = tilingData.bufSize;
    cubeCoreNum_ = tilingData.cubeCoreNum;
    cubeCoreNumAligned_ = tilingData.cubeCoreNumAligned;
    vecCoreNum_ = tilingData.vecCoreNum;
    vecTasksPerCore_ = tilingData.vecTasksPerCore;
    vecTasksTailCore_ = tilingData.vecTasksTailCore;
    elemsPerVecLoop_ = tilingData.elemsPerVecLoop;
    hasBias_ = tilingData.hasBias;
    mmTilingData_ = &tilingData.mmTiling;

    blockIdx_ = GetBlockIdx();
    coreIdx_ = blockIdx_;
    vecBlockIdx_ = blockIdx_;

    if ASCEND_IS_AIV {
        // For KERNEL_TYPE_MIX_AIC_1_2, multiple AIV sub-blocks can share the
        // same blockIdx. GetSubBlockIdx distinguishes the vector sub-core.
        uint64_t taskRatio = static_cast<uint64_t>(GetTaskRation());
        uint64_t subBlockIdx = static_cast<uint64_t>(GetSubBlockIdx());
        vecBlockIdx_ = blockIdx_ * taskRatio + subBlockIdx;

        // Fallback for schedulers that expose AIV cores as continuous blockIdx.
        if (vecBlockIdx_ >= vecCoreNum_) {
            if (blockIdx_ >= cubeCoreNum_ && (blockIdx_ - cubeCoreNum_) < vecCoreNum_) {
                vecBlockIdx_ = blockIdx_ - cubeCoreNum_;
            } else if (blockIdx_ >= cubeCoreNumAligned_ && (blockIdx_ - cubeCoreNumAligned_) < vecCoreNum_) {
                vecBlockIdx_ = blockIdx_ - cubeCoreNumAligned_;
            } else {
                vecBlockIdx_ = blockIdx_;
            }
        }

        coreIdx_ = vecBlockIdx_;
    }
    mm_.Init(mmTilingData_, pipe_);
}

template <typename T, uint64_t APPROXIMATE>
__aicore__ inline void FusedMatmulGeluOp<T, APPROXIMATE>::Init(const FusedMatmulGeluTilingData& __restrict tilingData,
                                                               GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR y,
                                                               GM_ADDR userWorkspace, TPipe* pipe)
{
    pipe_ = pipe;
    InitTilingData(tilingData);

    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    weightGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(weight));
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
    mmOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(userWorkspace));

    if (hasBias_ != 0) {
        biasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bias));
    }

    if ASCEND_IS_AIV {
        pipe_->InitBuffer(calcBuf_, bufSize_);

        uint64_t tAligned = AlignUp(elemsPerVecLoop_, DATA_PER_BLOCK_B16);
        uint64_t fAligned = AlignUp(elemsPerVecLoop_, DATA_PER_BLOCK_B32);
        uint64_t offset = 0;

        xLocal_ = calcBuf_.GetWithOffset<T>(tAligned, offset);
        offset += tAligned * sizeof(T);

        biasLocal_ = calcBuf_.GetWithOffset<T>(tAligned, offset);
        offset += tAligned * sizeof(T);

        xFp32_ = calcBuf_.GetWithOffset<float>(fAligned, offset);
        offset += fAligned * sizeof(float);

        workFp32_ = calcBuf_.GetWithOffset<float>(fAligned, offset);
    }
}

template <typename T, uint64_t APPROXIMATE>
__aicore__ inline void FusedMatmulGeluOp<T, APPROXIMATE>::MatmulProcess()
{
    uint64_t baseM = mmTilingData_->baseM;
    uint64_t baseN = mmTilingData_->baseN;
    uint64_t blockDimM = CeilDiv(m_, baseM);
    uint64_t blockDimN = CeilDiv(n_, baseN);
    uint64_t totalCubeBlocks = blockDimM * blockDimN;

    uint64_t loopCount = totalCubeBlocks / cubeCoreNum_;
    uint64_t loopRemain = totalCubeBlocks % cubeCoreNum_;
    if (coreIdx_ < loopRemain) {
        loopCount += 1;
    }

    // FP16 uses MatmulImpl SetBias, which has been verified.
    // BF16 disables MatmulImpl bias and adds bias in AIV epilogue.
    if constexpr (IsSameType<T, half>::value) {
        if (hasBias_ == 0) {
            mm_.DisableBias();
        }
    } else {
        mm_.DisableBias();
    }

    for (uint64_t loop = 0; loop < loopCount; ++loop) {
        uint64_t block = coreIdx_ + loop * cubeCoreNum_;
        uint64_t mIdx = block / blockDimN;
        uint64_t nIdx = block % blockDimN;

        uint64_t mOffset = mIdx * baseM;
        uint64_t nOffset = nIdx * baseN;

        uint64_t curM = (mIdx + 1 < blockDimM) ? baseM : (m_ - mOffset);
        uint64_t curN = (nIdx + 1 < blockDimN) ? baseN : (n_ - nOffset);

        uint64_t xOffset = mOffset * k_;
        uint64_t weightOffset = nOffset * k_;
        uint64_t outOffset = mOffset * n_ + nOffset;

        mm_.SetOrgShape(m_, n_, k_);
        mm_.SetSingleShape(curM, curN, k_);
        mm_.SetTensorA(xGm_[xOffset], false);
        mm_.SetTensorB(weightGm_[weightOffset], true);

        if constexpr (IsSameType<T, half>::value) {
            if (hasBias_ != 0) {
                mm_.SetBias(biasGm_[nOffset]);
            }
        }

        mm_.template IterateAll<false>(mmOutGm_[outOffset], 0, false);
    }
}

template <typename T, uint64_t APPROXIMATE>
__aicore__ inline void FusedMatmulGeluOp<T, APPROXIMATE>::ComputeGeluTanh(uint64_t count)
{
    Mul(workFp32_, xFp32_, xFp32_, count);
    PipeBarrier<PIPE_V>();

    Mul(workFp32_, workFp32_, xFp32_, count);
    PipeBarrier<PIPE_V>();

    Muls(workFp32_, workFp32_, GELU_BETA, count);
    PipeBarrier<PIPE_V>();

    Add(workFp32_, xFp32_, workFp32_, count);
    PipeBarrier<PIPE_V>();

    Muls(workFp32_, workFp32_, GELU_ALPHA, count);
    PipeBarrier<PIPE_V>();

    Exp(workFp32_, workFp32_, count);
    PipeBarrier<PIPE_V>();

    Adds(workFp32_, workFp32_, GELU_ONE, count);
    PipeBarrier<PIPE_V>();

    Div(xFp32_, xFp32_, workFp32_, count);
    PipeBarrier<PIPE_V>();
}

template <typename T, uint64_t APPROXIMATE>
__aicore__ inline void FusedMatmulGeluOp<T, APPROXIMATE>::VectorChunkProcess(uint64_t offset, uint64_t count)
{
    uint64_t alignedT = AlignUp(count, DATA_PER_BLOCK_B16);
    uint8_t rightPadT = static_cast<uint8_t>(alignedT - count);

    DataCopyExtParams copyInParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, rightPadT, T(0)};

    // Read one continuous chunk from matmul workspace.
    DataCopyPad(xLocal_, mmOutGm_[offset], copyInParams, padParams);

    // Ensure the MTE2 copy is complete before the vector pipeline reads xLocal_.
    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);

    Cast(xFp32_, xLocal_, RoundMode::CAST_NONE, count);
    PipeBarrier<PIPE_V>();

    if (hasBias_ != 0) {
        // FP16 bias has already been added by MatmulImpl.
        // BF16 bias is added here because BF16 MatmulImpl SetBias is not reliable in this path.
        if constexpr (!IsSameType<T, half>::value) {
            uint64_t processed = 0;
            uint64_t curOffset = offset;

            while (processed < count) {
                uint64_t col = curOffset % n_;
                uint64_t rowRemain = n_ - col;
                uint64_t curCount = (count - processed) < rowRemain ? (count - processed) : rowRemain;

                uint64_t alignedBias = AlignUp(curCount, DATA_PER_BLOCK_B16);
                uint8_t rightPadBias = static_cast<uint8_t>(alignedBias - curCount);

                DataCopyExtParams biasCopyParams{1, static_cast<uint32_t>(curCount * sizeof(T)), 0, 0, 0};
                DataCopyPadExtParams<T> biasPadParams{true, 0, rightPadBias, T(0)};

                DataCopyPad(biasLocal_, biasGm_[col], biasCopyParams, biasPadParams);

                // Ensure the BF16 bias copy is complete before vector conversion.
                SetFlag<HardEvent::MTE2_V>(0);
                WaitFlag<HardEvent::MTE2_V>(0);

                Cast(workFp32_, biasLocal_, RoundMode::CAST_NONE, curCount);
                PipeBarrier<PIPE_V>();

                Add(xFp32_[processed], xFp32_[processed], workFp32_, curCount);
                PipeBarrier<PIPE_V>();

                processed += curCount;
                curOffset += curCount;
            }
        }
    }
    ComputeGeluTanh(count);

    if constexpr (IsSameType<T, half>::value) {
        Cast(xLocal_, xFp32_, RoundMode::CAST_NONE, count);
    } else {
        Cast(xLocal_, xFp32_, RoundMode::CAST_RINT, count);
    }
    PipeBarrier<PIPE_V>();

    // Ensure vector conversion has completed before MTE3 reads xLocal_.
    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);

    // Write one continuous chunk to output.
    DataCopyPad(yGm_[offset], xLocal_, copyInParams);
    PipeBarrier<PIPE_ALL>();
}

template <typename T, uint64_t APPROXIMATE>
__aicore__ inline void FusedMatmulGeluOp<T, APPROXIMATE>::VectorProcess()
{
    if (totalElement_ == 0 || elemsPerVecLoop_ == 0 || vecCoreNum_ == 0) {
        return;
    }

    if (vecBlockIdx_ >= vecCoreNum_) {
        return;
    }

    uint64_t curTasks = vecTasksPerCore_;
    uint64_t startTask = 0;

    if (vecBlockIdx_ < vecTasksTailCore_) {
        curTasks += 1;
        startTask = vecBlockIdx_ * curTasks;
    } else {
        startTask = vecBlockIdx_ * vecTasksPerCore_ + vecTasksTailCore_;
    }

    for (uint64_t task = 0; task < curTasks; ++task) {
        uint64_t offset = (startTask + task) * elemsPerVecLoop_;
        if (offset >= totalElement_) {
            return;
        }

        uint64_t remain = totalElement_ - offset;
        uint64_t count = remain < elemsPerVecLoop_ ? remain : elemsPerVecLoop_;

        VectorChunkProcess(offset, count);
    }
}

template <typename T, uint64_t APPROXIMATE>
__aicore__ inline void FusedMatmulGeluOp<T, APPROXIMATE>::Process()
{
    if ASCEND_IS_AIC {
        MatmulProcess();
    }

    SyncAll<false>();

    if ASCEND_IS_AIV {
        VectorProcess();
    }
}

} // namespace FusedMatmulGelu

#endif // OP_KERNEL_FUSED_MATMUL_GELU_H_
