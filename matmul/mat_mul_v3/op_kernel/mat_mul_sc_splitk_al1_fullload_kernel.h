/**
* Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
* \file mat_mul_sc_splitk_al1_fullload_kernel.h
* \brief Single core split K with AL1 full load kernel implementation
*/
#ifndef __OP_KERNEL_MATMUL_V3_SC_SPLITK_AL1_FULLLOAD_KERNEL_H__
#define __OP_KERNEL_MATMUL_V3_SC_SPLITK_AL1_FULLLOAD_KERNEL_H__

#include "mat_mul_sc_splitk_al1_fullload_block.h"
#include "mat_mul_base_block.h"

using namespace AscendC;
using namespace matmul;

template <class A_TYPE, class B_TYPE, class L0C_TYPE, class C_TYPE, class BIAS_TYPE,
    class BLOCK_TYPE = MatmulSingleCoreSplitKAL1FullLoadBlock,
    const MatmulConfig &MM_CFG = MM_CFG_NO_PRELOAD>
class MatMulBaseKernelSCSplitKAL1FullLoad {
public:
    __aicore__ inline MatMulBaseKernelSCSplitKAL1FullLoad() {}

    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR workspaceGM, GM_ADDR biasGM,
        const MatmulTilingData *matmulTilingData, TPipe *pipe);

    __aicore__ inline void Process(GM_ADDR cGM, GM_ADDR srcAddr, TBuf<TPosition::VECCALC> &ubBuf);

    __aicore__ inline void End()
    {
        mm_.End();
    }

protected:
    BLOCK_TYPE block_;
    using A_T = typename A_TYPE::T;
    using B_T = typename B_TYPE::T;
    using C_T = typename C_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;
    using A_TYPE_NEW = MatmulType<AscendC::TPosition::TSCM, A_TYPE::format, A_T, A_TYPE::isTrans>;
    using L0C_T = typename L0C_TYPE::T;
    MatmulImpl<A_TYPE_NEW, B_TYPE, L0C_TYPE, BIAS_TYPE, MM_CFG> mm_;

    TQue<QuePosition::A1, 1> inQueueAL1_;
    GlobalTensor<A_T> aGlobal_;
    GlobalTensor<B_T> bGlobal_;
    GlobalTensor<L0C_T> cOutGlobal_;
    GlobalTensor<BiasT> biasGlobal_;
    TPipe *pipe_;

private:
    __aicore__ inline void InitInputs(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR workspaceGM, GM_ADDR biasGM);
    __aicore__ inline void SetOrgShape();
    __aicore__ inline void CopyInAL1();
    __aicore__ inline void CastCurrentNBaseTile(GM_ADDR cGM, GM_ADDR srcAddr, TBuf<TPosition::VECCALC> &ubBuf);
    __aicore__ inline void ProcessNInnerBlock(uint64_t kIndex, LocalTensor<A_T> &al1Local);
};

template <class A_TYPE, class B_TYPE, class L0C_TYPE, class C_TYPE, class BIAS_TYPE,
    class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void MatMulBaseKernelSCSplitKAL1FullLoad<A_TYPE, B_TYPE, L0C_TYPE, C_TYPE, BIAS_TYPE,
    BLOCK_TYPE, MM_CFG>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR workspaceGM, GM_ADDR biasGM,
    const MatmulTilingData *matmulTilingData, TPipe *pipe)
{
    block_.template Init<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(matmulTilingData);
    pipe_ = pipe;

    if ASCEND_IS_AIV {
        return;
    }

    SetAtomicNone();
    if (GetCurrentBlockIdx() >= block_.matmulTilingData_->matmulTiling.usedCoreNum) {
        return;
    }
    InitInputs(aGM, bGM, workspaceGM, biasGM);

    const uint64_t innerAlignedBlock = BLOCK_BYTE_SIZE / sizeof(A_T);
    const uint64_t mDim = static_cast<uint64_t>(
        block_.matmulTilingData_->matmulTiling.baseM * block_.matmulTilingData_->matmulTiling.stepM);
    const uint64_t kAL1Dim = block_.params_.kAL1FullLoadSize;
    const uint64_t mAligned = A_TYPE::isTrans ?
        MMV3CeilAlign(mDim, innerAlignedBlock) :
        MMV3CeilAlign(mDim, static_cast<uint64_t>(BLOCK_SIZE));
    const uint64_t kAL1Aligned = A_TYPE::isTrans ?
        MMV3CeilAlign(kAL1Dim, static_cast<uint64_t>(BLOCK_SIZE)) :
        MMV3CeilAlign(kAL1Dim, innerAlignedBlock);
    pipe_->InitBuffer(inQueueAL1_, 1, mAligned * kAL1Aligned * sizeof(A_T));

    mm_.SetSubBlockIdx(0);
    mm_.Init(&block_.matmulTilingData_->matmulTiling, pipe_);
    SetOrgShape();
}

template <class A_TYPE, class B_TYPE, class L0C_TYPE, class C_TYPE, class BIAS_TYPE,
    class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void MatMulBaseKernelSCSplitKAL1FullLoad<A_TYPE, B_TYPE, L0C_TYPE, C_TYPE, BIAS_TYPE,
    BLOCK_TYPE, MM_CFG>::InitInputs(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR workspaceGM, GM_ADDR biasGM)
{
    aGlobal_.SetGlobalBuffer((__gm__ A_T *)aGM,
        static_cast<uint64_t>(block_.matmulTilingData_->matmulTiling.M) *
        block_.matmulTilingData_->matmulTiling.Ka);
    bGlobal_.SetGlobalBuffer((__gm__ B_T *)bGM,
        static_cast<uint64_t>(block_.matmulTilingData_->matmulTiling.Kb) *
        block_.matmulTilingData_->matmulTiling.N);
    aGlobal_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    bGlobal_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    cOutGlobal_.SetGlobalBuffer((__gm__ L0C_T *)workspaceGM,
        static_cast<uint64_t>(block_.matmulTilingData_->matmulTiling.M) *
        block_.matmulTilingData_->matmulTiling.N);
    biasGlobal_.SetGlobalBuffer((__gm__ BiasT *)biasGM,
        block_.matmulTilingData_->matmulTiling.N);
}

template <class A_TYPE, class B_TYPE, class L0C_TYPE, class C_TYPE, class BIAS_TYPE,
    class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void MatMulBaseKernelSCSplitKAL1FullLoad<A_TYPE, B_TYPE, L0C_TYPE, C_TYPE, BIAS_TYPE,
    BLOCK_TYPE, MM_CFG>::SetOrgShape()
{
    if constexpr (A_TYPE::format == CubeFormat::NZ && B_TYPE::format == CubeFormat::NZ) {
        mm_.SetOrgShape(block_.params_.alignedOriM, block_.params_.alignedOriN,
            block_.params_.alignedKaSize, block_.params_.alignedKbSize, block_.params_.outNAlign);
    } else if constexpr (A_TYPE::format == CubeFormat::NZ) {
        mm_.SetOrgShape(block_.params_.alignedOriM, block_.matmulTilingData_->matmulTiling.N,
            block_.params_.alignedKaSize, block_.matmulTilingData_->matmulTiling.Kb,
            block_.params_.outNAlign);
    } else if constexpr (B_TYPE::format == CubeFormat::NZ) {
        mm_.SetOrgShape(block_.matmulTilingData_->matmulTiling.M, block_.params_.alignedOriN,
            block_.matmulTilingData_->matmulTiling.Ka, block_.params_.alignedKbSize,
            block_.params_.outNAlign);
    } else {
        mm_.SetOrgShape(block_.matmulTilingData_->matmulTiling.M,
            block_.matmulTilingData_->matmulTiling.N,
            block_.matmulTilingData_->matmulTiling.Ka,
            block_.matmulTilingData_->matmulTiling.Kb,
            block_.params_.outNAlign);
    }
}

template <class A_TYPE, class B_TYPE, class L0C_TYPE, class C_TYPE, class BIAS_TYPE,
    class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void MatMulBaseKernelSCSplitKAL1FullLoad<A_TYPE, B_TYPE, L0C_TYPE, C_TYPE, BIAS_TYPE,
    BLOCK_TYPE, MM_CFG>::CopyInAL1()
{
    LocalTensor<A_T> al1Local = inQueueAL1_.AllocTensor<A_T>();

    uint64_t baseM = static_cast<uint64_t>(block_.matmulTilingData_->matmulTiling.baseM);
    uint64_t stepM = static_cast<uint64_t>(block_.matmulTilingData_->matmulTiling.stepM);
    uint64_t mDim = baseM * stepM;
    uint64_t kDim = block_.params_.currentKBlockSize;

    if constexpr (A_TYPE::format == CubeFormat::NZ) {
        uint64_t alignM = MMV3CeilAlign(mDim, static_cast<uint64_t>(ALIGNED_H));
        DataCopyParams aNzParams;
        if constexpr (A_TYPE::isTrans) {
            aNzParams.blockCount = MMV3DivCeil(mDim, ALIGNED_H);
            aNzParams.blockLen = kDim;
            aNzParams.srcStride = block_.params_.alignedKaSize - kDim;
        } else {
            aNzParams.blockCount = MMV3DivCeil(kDim, block_.params_.c0Size);
            aNzParams.blockLen = alignM;
            aNzParams.srcStride = block_.params_.alignedOriM - alignM;
        }
        uint64_t gmOffset = block_.params_.currentKBlockStart;
        if constexpr (A_TYPE::isTrans) {
            gmOffset = block_.params_.currentKBlockStart * block_.params_.c0Size;
        } else {
            gmOffset = block_.params_.currentKBlockStart * block_.params_.alignedOriM;
        }
        DataCopy(al1Local, aGlobal_[gmOffset], aNzParams);
    } else {
        Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = 1;
        uint64_t nDim = A_TYPE::isTrans ? kDim : mDim;
        uint64_t dDim = A_TYPE::isTrans ? mDim : kDim;
        nd2nzParams.nValue = nDim;
        nd2nzParams.dValue = dDim;
        nd2nzParams.srcNdMatrixStride = 0;
        nd2nzParams.srcDValue = static_cast<uint64_t>(A_TYPE::isTrans ?
            block_.matmulTilingData_->matmulTiling.M :
            block_.matmulTilingData_->matmulTiling.Ka);
        nd2nzParams.dstNzC0Stride = MMV3CeilAlign(nDim, static_cast<uint64_t>(BLOCK_SIZE));
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = 0;

        uint64_t gmOffset = block_.params_.currentKBlockStart;
        if constexpr (A_TYPE::isTrans) {
            gmOffset = block_.params_.currentKBlockStart *
                static_cast<uint64_t>(block_.matmulTilingData_->matmulTiling.M);
        }
        DataCopy(al1Local, aGlobal_[gmOffset], nd2nzParams);
    }
    inQueueAL1_.EnQue(al1Local);
}

template <class A_TYPE, class B_TYPE, class L0C_TYPE, class C_TYPE, class BIAS_TYPE,
    class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void MatMulBaseKernelSCSplitKAL1FullLoad<A_TYPE, B_TYPE, L0C_TYPE, C_TYPE, BIAS_TYPE,
    BLOCK_TYPE, MM_CFG>::ProcessNInnerBlock(uint64_t kIndex, LocalTensor<A_T> &al1Local)
{
    if ASCEND_IS_AIC {
        mm_.SetSingleShape(block_.matmulTilingData_->matmulTiling.M,
            block_.params_.curCoreN,
            block_.params_.currentKBlockSize);
        mm_.SetTensorA(al1Local, block_.params_.isTransposeA);
        mm_.SetTensorB(bGlobal_[block_.offset_.offsetB], block_.params_.isTransposeB);
        if (kIndex == 0) {
            block_.params_.atomicAddFlag = false;
            if (block_.matmulTilingData_->matmulTiling.isBias) {
                mm_.SetBias(biasGlobal_[block_.offset_.offsetBias]);
            }
        } else {
            block_.params_.atomicAddFlag = true;
        }

        uint64_t actualOffsetC = block_.offset_.offsetC;
        if constexpr (sizeof(C_T) == sizeof(float)) {
            uint64_t alignedN = block_.params_.outNAlign;
            uint64_t realN = block_.matmulTilingData_->matmulTiling.N;
            if (alignedN != realN && alignedN != 0) {
                uint64_t nOff = actualOffsetC % alignedN;
                uint64_t mRow = actualOffsetC / alignedN;
                actualOffsetC = mRow * realN + nOff;
            }
        }
        mm_.IterateAll(cOutGlobal_[actualOffsetC], block_.params_.atomicAddFlag);
        mm_.ClearBias();
    }
}

template <class A_TYPE, class B_TYPE, class L0C_TYPE, class C_TYPE, class BIAS_TYPE,
    class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void MatMulBaseKernelSCSplitKAL1FullLoad<A_TYPE, B_TYPE, L0C_TYPE, C_TYPE, BIAS_TYPE,
    BLOCK_TYPE, MM_CFG>::CastCurrentNBaseTile(GM_ADDR cGM, GM_ADDR srcAddr, TBuf<TPosition::VECCALC> &ubBuf)
{
    uint64_t offset = block_.params_.coreOffsetN;
    uint64_t mRows = block_.matmulTilingData_->matmulTiling.M;
    uint64_t cols = block_.params_.curCoreN;
    uint64_t vMOffset = MMV3DivCeil(mRows, NUM_TWO);
    if (GetBlockIdx() % NUM_TWO == 1) {
        offset = offset + vMOffset * block_.matmulTilingData_->matmulTiling.N;
        vMOffset = mRows - vMOffset;
    }
    uint64_t singleSize = vMOffset * cols;
    Cast32to16V220(reinterpret_cast<__gm__ typename C_TYPE::T *>(cGM) + offset,
        reinterpret_cast<__gm__ float *>(srcAddr) + offset,
        singleSize,
        static_cast<uint32_t>(cols),
        block_.matmulTilingData_->matmulTiling.N,
        ubBuf);
    PipeBarrier<PIPE_ALL>();
}

template <class A_TYPE, class B_TYPE, class L0C_TYPE, class C_TYPE, class BIAS_TYPE,
    class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void MatMulBaseKernelSCSplitKAL1FullLoad<A_TYPE, B_TYPE, L0C_TYPE, C_TYPE, BIAS_TYPE,
    BLOCK_TYPE, MM_CFG>::Process(GM_ADDR cGM, GM_ADDR srcAddr, TBuf<TPosition::VECCALC> &ubBuf)
{
    block_.InitBlockIndex();
    const bool needFp32Cast = !block_.params_.isSingleKStrip && sizeof(C_T) != sizeof(float);
    if ASCEND_IS_AIC {
        mm_.SetHF32(block_.params_.isHf32, 1);
    }

    if ASCEND_IS_AIC {
        for (uint64_t kIndex = 0; kIndex < block_.params_.kAL1LoopNum; ++kIndex) {
            block_.UpdateBlockParamsK(kIndex);
            CopyInAL1();
            LocalTensor<A_T> al1Local = inQueueAL1_.DeQue<A_T>();
            block_.template CalcGMOffset<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(kIndex);
            ProcessNInnerBlock(kIndex, al1Local);
            inQueueAL1_.FreeTensor(al1Local);
        }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
        if (needFp32Cast) {
            NotifyEvent<PIPE_FIX>(AIC_SYNC_AIV_FLAG);
        }
#endif
        PipeBarrier<PIPE_ALL>();
    }

    if (needFp32Cast) {
        if ASCEND_IS_AIV {
            WaitFlagDevLocal(AIC_SYNC_AIV_FLAG);
            CastCurrentNBaseTile(cGM, srcAddr, ubBuf);
        }
    }

    PipeBarrier<PIPE_ALL>();
    SetAtomicNone();

    if ASCEND_IS_AIC {
        mm_.SetHF32(false, 0);
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE,
    class BLOCK_TYPE = MatmulSingleCoreSplitKAL1FullLoadBlock,
    const MatmulConfig &MM_CFG = MM_CFG_NO_PRELOAD>
class MatMulSingleCoreSplitKAL1FullLoadKernel {
    struct InnerParams {
        GM_ADDR alignedworkspaceGM;
        GM_ADDR cGM;
        bool n128Align = false;
        /// stepKa*baseK>=Ka 且 C 非 fp32：高阶 API 直接写 C GM，跳过 fp32 workspace + vector cast
        bool directOutput = false;
    };

public:
    __aicore__ inline MatMulSingleCoreSplitKAL1FullLoadKernel() {}
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM,
        GM_ADDR offsetWGM, GM_ADDR workspaceGM, const MatmulTilingData *matmulTilingData,
        TPipe *pipe);

    __aicore__ inline void InitWithMatmulWorkspace(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM,
        GM_ADDR offsetWGM, GM_ADDR workspaceGM, GM_ADDR matmulFp32WorkspaceGM,
        const MatmulTilingData *matmulTilingData, TPipe *pipe);

    __aicore__ inline void Process(uint8_t enAtomic = 0);
    __aicore__ inline void End()
    {
        if (innerParams_.directOutput) {
            directKernel_.End();
        } else {
            splitKernel_.End();
        }
    }

protected:
    using SplitL0CType = MatmulType<C_TYPE::pos, C_TYPE::format, float, C_TYPE::isTrans>;
    MatMulBaseKernelSCSplitKAL1FullLoad<A_TYPE, B_TYPE, C_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>
        directKernel_;
    MatMulBaseKernelSCSplitKAL1FullLoad<A_TYPE, B_TYPE, SplitL0CType, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>
        splitKernel_;

    TPipe *pipe_;
    TBuf<> ubBuf_;
    const MatmulTilingData *matmulTilingData_;
    InnerParams innerParams_;

private:
    __aicore__ inline void InitInputs(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM,
        GM_ADDR offsetWGM, GM_ADDR workspaceGM);
    __aicore__ inline void InitInner(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM,
        const MatmulTilingData *matmulTilingData, TPipe *pipe);
    __aicore__ inline bool ShouldSkipAivCore() const;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE,
    class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void MatMulSingleCoreSplitKAL1FullLoadKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE,
    BLOCK_TYPE, MM_CFG>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR workspaceGM,
    const MatmulTilingData *matmulTilingData, TPipe *pipe)
{
    InitWithMatmulWorkspace(aGM, bGM, cGM, biasGM, offsetWGM, workspaceGM, nullptr, matmulTilingData, pipe);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE,
    class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void MatMulSingleCoreSplitKAL1FullLoadKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE,
    BLOCK_TYPE, MM_CFG>::InitWithMatmulWorkspace(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR workspaceGM,
    GM_ADDR matmulFp32WorkspaceGM, const MatmulTilingData *matmulTilingData, TPipe *pipe)
{
    pipe_ = pipe;
    matmulTilingData_ = matmulTilingData;
    innerParams_.n128Align = (matmulTilingData_->matmulTiling.N % ALIGN_128_BYTE == 0);

    InitInputs(aGM, bGM, cGM, biasGM, offsetWGM, workspaceGM);
    if (matmulFp32WorkspaceGM != nullptr) {
        innerParams_.alignedworkspaceGM = matmulFp32WorkspaceGM;
    }

    const uint64_t kAL1FullLoadSize = static_cast<uint64_t>(matmulTilingData_->matmulTiling.stepKa) *
        static_cast<uint64_t>(matmulTilingData_->matmulTiling.baseK);
    using C_T = typename C_TYPE::T;
    innerParams_.directOutput = (kAL1FullLoadSize >= static_cast<uint64_t>(matmulTilingData_->matmulTiling.Ka)) &&
        sizeof(C_T) != sizeof(float);

    if ASCEND_IS_AIV {
        if (ShouldSkipAivCore()) {
            return;
        }
        pipe_->InitBuffer(ubBuf_, TOTAL_UB_SIZE);
    }

    InitInner(aGM, bGM, biasGM, matmulTilingData, pipe);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE,
    class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void MatMulSingleCoreSplitKAL1FullLoadKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE,
    BLOCK_TYPE, MM_CFG>::InitInner(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM,
    const MatmulTilingData *matmulTilingData, TPipe *pipe)
{
    if ASCEND_IS_AIC {
        if (GetBlockIdx() >= matmulTilingData_->matmulTiling.usedCoreNum) {
            return;
        }
        using C_T = typename C_TYPE::T;
        if constexpr (sizeof(C_T) == sizeof(float)) {
            innerParams_.alignedworkspaceGM = innerParams_.cGM;
        }
    }

    if (innerParams_.directOutput) {
        directKernel_.Init(aGM, bGM, innerParams_.cGM, biasGM, matmulTilingData, pipe);
    } else {
        splitKernel_.Init(aGM, bGM, innerParams_.alignedworkspaceGM, biasGM, matmulTilingData, pipe);
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE,
    class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void MatMulSingleCoreSplitKAL1FullLoadKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE,
    BLOCK_TYPE, MM_CFG>::InitInputs(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR workspaceGM)
{
    innerParams_.alignedworkspaceGM = reinterpret_cast<GM_ADDR>(
        ((reinterpret_cast<uint64_t>(workspaceGM + MAX_BLOCK_NUM * DEFAULT_BLOCK_LEN * sizeof(int32_t)) + 511) / 512) * 512);
    innerParams_.cGM = cGM;
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE,
    class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void MatMulSingleCoreSplitKAL1FullLoadKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE,
    BLOCK_TYPE, MM_CFG>::Process(uint8_t enAtomic)
{
    (void)enAtomic;
    using C_T = typename C_TYPE::T;
    if ASCEND_IS_AIV {
        if (ShouldSkipAivCore()) {
            return;
        }
        PipeBarrier<PIPE_ALL>();
    }

    if ASCEND_IS_AIC {
        if (GetBlockIdx() >= matmulTilingData_->matmulTiling.usedCoreNum) {
            return;
        }
    }

    if (innerParams_.directOutput) {
        directKernel_.Process(innerParams_.cGM, innerParams_.cGM, ubBuf_);
        return;
    }

    if constexpr (sizeof(C_T) == sizeof(float)) {
        splitKernel_.Process(innerParams_.cGM, innerParams_.alignedworkspaceGM, ubBuf_);
        return;
    }

    splitKernel_.Process(innerParams_.cGM, innerParams_.alignedworkspaceGM, ubBuf_);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE,
    class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline bool MatMulSingleCoreSplitKAL1FullLoadKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE,
    BLOCK_TYPE, MM_CFG>::ShouldSkipAivCore() const
{
    using C_T = typename C_TYPE::T;
    if constexpr (sizeof(C_T) == sizeof(float)) {
        return true;
    }
    if (innerParams_.directOutput) {
        return true;
    }
    return GetBlockIdx() >= matmulTilingData_->matmulTiling.usedCoreNum * NUM_AIV_TO_AIC_RATIO;
}

#endif // __OP_KERNEL_MATMUL_V3_SC_SPLITK_AL1_FULLLOAD_KERNEL_H__
