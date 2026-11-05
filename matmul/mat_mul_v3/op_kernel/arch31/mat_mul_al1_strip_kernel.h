/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MAT_MUL_V3_ARCH31_AL1_STRIP_KERNEL_H
#define MAT_MUL_V3_ARCH31_AL1_STRIP_KERNEL_H

#include "../mat_mul_base_block.h"
#include "mat_mul_v3_l1_policy.h"

using namespace AscendC;
using namespace matmul;

template <class B_TYPE, class C_TYPE, class BIAS_TYPE, const MatmulConfig& MM_CFG = MM_CFG_MDL>
class MatmulAl1StripKernelArch31 {
public:
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, const MatmulTilingData* tilingData, TPipe* pipe)
    {
        tilingData_ = tilingData;
        pipe_ = pipe;
        aGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(aGM),
                                 static_cast<uint64_t>(tilingData_->matmulTiling.M) * tilingData_->matmulTiling.Ka);
        bGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(bGM),
                                 static_cast<uint64_t>(tilingData_->matmulTiling.Kb) * tilingData_->matmulTiling.N);
        cGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(cGM),
                                 static_cast<uint64_t>(tilingData_->matmulTiling.M) * tilingData_->matmulTiling.N);
        mm_.SetSubBlockIdx(0);
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            return;
        }
        const uint64_t m = tilingData_->matmulTiling.M;
        const uint64_t n = tilingData_->matmulTiling.N;
        const uint64_t k = tilingData_->matmulTiling.Ka;
        const uint64_t stripM = tilingData_->matmulTiling.baseM;
        const uint64_t stripMAligned = CeilAlign(stripM, static_cast<uint64_t>(BLOCK_SIZE));
        const uint64_t kAligned = CeilAlign(k, static_cast<uint64_t>(BLOCK_BYTE_SIZE / sizeof(half)));
        pipe_->InitBuffer(a1Queue_, 1, stripMAligned * kAligned * sizeof(half));
        mm_.Init(&tilingData_->matmulTiling, pipe_);
        mm_.SetHF32(false, 0);
        SetAtomicNone();

        const uint64_t blockIdx = static_cast<uint64_t>(AscendC::GetBlockIdx());
        const uint64_t blockNum = static_cast<uint64_t>(AscendC::GetBlockNum());
        const uint64_t stripCount = (m + stripM - 1) / stripM;
        for (uint64_t stripIdx = blockIdx; stripIdx < stripCount; stripIdx += blockNum) {
            const uint64_t mOffset = stripIdx * stripM;
            const uint64_t useM = Min(stripM, m - mOffset);
            LocalTensor<half> a1 = a1Queue_.AllocTensor<half>();
            CopyA(a1, aGlobal_[mOffset * k], useM, k);
            a1Queue_.EnQue(a1);
            a1 = a1Queue_.DeQue<half>();
            mm_.SetSingleShape(useM, n, k);
            mm_.SetTensorA(a1, false);
            mm_.SetTensorB(bGlobal_, true);
            mm_.IterateAll(cGlobal_[mOffset * n], 0);
            a1Queue_.FreeTensor(a1);
        }
        mm_.End();
        PipeBarrier<PIPE_ALL>();
        mm_.SetHF32(false, 0);
    }

private:
    using A_TYPE_A1 = MatmulType<TPosition::TSCM, CubeFormat::ND, half, false>;
    __aicore__ inline uint64_t CeilAlign(uint64_t value, uint64_t alignment) const
    {
        return (value + alignment - 1) / alignment * alignment;
    }
    __aicore__ inline uint64_t Min(uint64_t lhs, uint64_t rhs) const { return lhs < rhs ? lhs : rhs; }
    __aicore__ inline void CopyA(LocalTensor<half> dst, const GlobalTensor<half>& src, uint64_t useM, uint64_t k)
    {
        Nd2NzParams copy;
        copy.ndNum = 1;
        copy.nValue = useM;
        copy.dValue = k;
        copy.srcNdMatrixStride = 0;
        copy.srcDValue = k;
        copy.dstNzC0Stride = CeilAlign(useM, static_cast<uint64_t>(BLOCK_SIZE));
        copy.dstNzNStride = 1;
        copy.dstNzMatrixStride = 0;
        DataCopy(dst, src, copy);
    }

    MatmulImpl<A_TYPE_A1, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG> mm_;
    TQue<QuePosition::A1, 1> a1Queue_;
    GlobalTensor<half> aGlobal_;
    GlobalTensor<half> bGlobal_;
    GlobalTensor<half> cGlobal_;
    const MatmulTilingData* tilingData_ = nullptr;
    TPipe* pipe_ = nullptr;
};

#endif
