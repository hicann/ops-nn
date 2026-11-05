/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MAT_MUL_V3_ARCH31_SPLITK_AL1_KERNEL_H
#define MAT_MUL_V3_ARCH31_SPLITK_AL1_KERNEL_H

#include "../mat_mul_base_block.h"
#include "mat_mul_v3_l1_policy.h"

using namespace AscendC;
using namespace matmul;

template <class B_TYPE, class C_TYPE, class BIAS_TYPE, const MatmulConfig& MM_CFG = MM_CFG_MDL>
class MatmulSplitKAl1KernelArch31 {
public:
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR workspaceGM,
                                const MatmulTilingData* tilingData, TPipe* pipe)
    {
        tilingData_ = tilingData;
        pipe_ = pipe;
        const uint64_t m = tilingData_->matmulTiling.M;
        const uint64_t n = tilingData_->matmulTiling.N;
        const uint64_t k = tilingData_->matmulTiling.Ka;
        aGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(aGM), m * k);
        bGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(bGM), n * k);
        cGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(cGM), m * n);
        cBitsGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(cGM), m * n);
        tempGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(workspaceGM), m * n);
        mm_.SetSubBlockIdx(0);
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            return;
        }
        if (static_cast<uint64_t>(AscendC::GetBlockIdx()) != 0U) {
            return;
        }

        const uint64_t m = tilingData_->matmulTiling.M;
        const uint64_t n = tilingData_->matmulTiling.N;
        const uint64_t k = tilingData_->matmulTiling.Ka;
        const uint64_t baseN = tilingData_->matmulTiling.baseN;
        const uint64_t baseK = tilingData_->matmulTiling.baseK;
        const uint64_t mAligned = CeilAlign(m, static_cast<uint64_t>(BLOCK_SIZE));
        const uint64_t stripK = FloorAlign(
            static_cast<uint64_t>(matmul_v3_arch31::kAl1StripBudgetBytes) / (mAligned * sizeof(half)), baseK);
        const uint64_t stripKAligned = CeilAlign(stripK, static_cast<uint64_t>(BLOCK_BYTE_SIZE / sizeof(half)));

        pipe_->InitBuffer(a1Queue_, 1, mAligned * stripKAligned * sizeof(half));
        pipe_->InitBuffer(addBuffer_, matmul_v3_arch31::kSplitKUbBytes);
        mm_.Init(&tilingData_->matmulTiling, pipe_);
        mm_.SetOrgShape(m, n, k, k, n);
        mm_.SetHF32(false, 0);
        SetAtomicNone();

        const uint64_t kLoops = (k + stripK - 1) / stripK;
        const uint64_t nLoops = (n + baseN - 1) / baseN;
        for (uint64_t kIndex = 0; kIndex < kLoops; ++kIndex) {
            const uint64_t kStart = kIndex * stripK;
            const uint64_t useK = Min(stripK, k - kStart);
            LocalTensor<half> a1 = a1Queue_.AllocTensor<half>();
            CopyA(a1, aGlobal_[kStart], m, useK, k, mAligned);
            a1Queue_.EnQue(a1);
            a1 = a1Queue_.DeQue<half>();

            for (uint64_t nIndex = 0; nIndex < nLoops; ++nIndex) {
                const uint64_t nStart = nIndex * baseN;
                const uint64_t useN = Min(baseN, n - nStart);
                mm_.SetSingleShape(m, useN, useK);
                mm_.SetTensorA(a1, false);
                mm_.SetTensorB(bGlobal_[nStart * k + kStart], true);
                if (kIndex == 0U) {
                    mm_.IterateAll(cGlobal_[nStart]);
                } else {
                    mm_.IterateAll(tempGlobal_[nStart]);
                }
            }
            a1Queue_.FreeTensor(a1);
            if (kIndex != 0U) {
                PipeBarrier<PIPE_ALL>();
                Accumulate(m * n);
                PipeBarrier<PIPE_ALL>();
            }
        }
        mm_.End();
        PipeBarrier<PIPE_ALL>();
        RepairExceptionalResults(m, n, k);
        SetAtomicNone();
        mm_.SetHF32(false, 0);
    }

private:
    using A_TYPE_A1 = MatmulType<TPosition::TSCM, CubeFormat::ND, half, false>;
    static constexpr uint64_t kChunkElements = matmul_v3_arch31::kSplitKUbBytes / sizeof(half) / 2;
    static constexpr uint64_t kRepairChunkElements = 1024;

    __aicore__ inline uint64_t CeilAlign(uint64_t value, uint64_t alignment) const
    {
        return (value + alignment - 1) / alignment * alignment;
    }
    __aicore__ inline uint64_t FloorAlign(uint64_t value, uint64_t alignment) const
    {
        return value / alignment * alignment;
    }
    __aicore__ inline uint64_t Min(uint64_t lhs, uint64_t rhs) const { return lhs < rhs ? lhs : rhs; }

    __aicore__ inline void CopyA(LocalTensor<half> dst, const GlobalTensor<half>& src, uint64_t m, uint64_t useK,
                                 uint64_t fullK, uint64_t mAligned)
    {
        Nd2NzParams copy;
        copy.ndNum = 1;
        copy.nValue = m;
        copy.dValue = useK;
        copy.srcNdMatrixStride = 0;
        copy.srcDValue = fullK;
        copy.dstNzC0Stride = mAligned;
        copy.dstNzNStride = 1;
        copy.dstNzMatrixStride = 0;
        DataCopy(dst, src, copy);
    }

    __aicore__ inline uint16_t FloatToHalfBits(float value) const
    {
        // The L311 cube FP16 store saturates overflow and flushes subnormals.
        // Encode RNE bits explicitly after FP32 strip accumulation.
        union FloatBits {
            float f;
            uint32_t u;
        } bits{value};
        const uint16_t sign = static_cast<uint16_t>((bits.u >> 16U) & 0x8000U);
        const uint32_t exponent = (bits.u >> 23U) & 0xFFU;
        const uint32_t mantissa = bits.u & 0x7FFFFFU;
        if (exponent == 0xFFU) {
            return static_cast<uint16_t>(sign | 0x7C00U | (mantissa == 0U ? 0U : 0x0200U));
        }
        const int32_t unbiased = static_cast<int32_t>(exponent) - 127;
        if (unbiased > 15) {
            return static_cast<uint16_t>(sign | 0x7C00U);
        }
        if (unbiased >= -14) {
            uint32_t encoded = static_cast<uint32_t>(unbiased + 15) << 10U;
            encoded |= mantissa >> 13U;
            const uint32_t remainder = mantissa & 0x1FFFU;
            if (remainder > 0x1000U || (remainder == 0x1000U && (encoded & 1U) != 0U)) {
                ++encoded;
            }
            return static_cast<uint16_t>(sign | (encoded >= 0x7C00U ? 0x7C00U : encoded));
        }
        if (unbiased < -25) {
            return sign;
        }
        const uint32_t significand = mantissa | 0x800000U;
        const uint32_t shift = static_cast<uint32_t>(-unbiased - 1);
        uint32_t encoded = significand >> shift;
        const uint32_t remainderMask = (1U << shift) - 1U;
        const uint32_t remainder = significand & remainderMask;
        const uint32_t halfway = 1U << (shift - 1U);
        if (remainder > halfway || (remainder == halfway && (encoded & 1U) != 0U)) {
            ++encoded;
        }
        return static_cast<uint16_t>(sign | encoded);
    }

    __aicore__ inline void Accumulate(uint64_t elements)
    {
        LocalTensor<half> lhs = addBuffer_.Get<half>();
        LocalTensor<half> rhs = lhs[kChunkElements];
        const event_t mte2ToV = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE2_V));
        const event_t vToMte3 = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE3));
        const event_t mte3ToMte2 = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2);
        for (uint64_t offset = 0; offset < elements; offset += kChunkElements) {
            const uint64_t useElements = Min(kChunkElements, elements - offset);
            WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2);
            DataCopy(lhs, cGlobal_[offset], useElements);
            DataCopy(rhs, tempGlobal_[offset], useElements);
            SetFlag<HardEvent::MTE2_V>(mte2ToV);
            WaitFlag<HardEvent::MTE2_V>(mte2ToV);
            Add(lhs, lhs, rhs, useElements);
            SetFlag<HardEvent::V_MTE3>(vToMte3);
            WaitFlag<HardEvent::V_MTE3>(vToMte3);
            DataCopy(cGlobal_[offset], lhs, useElements);
            SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2);
        }
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2);
        pipe_->ReleaseEventID<HardEvent::MTE2_V>(mte2ToV);
        pipe_->ReleaseEventID<HardEvent::V_MTE3>(vToMte3);
        pipe_->ReleaseEventID<HardEvent::MTE3_MTE2>(mte3ToMte2);
    }

    __aicore__ inline void RepairExceptionalResults(uint64_t m, uint64_t n, uint64_t k)
    {
        if (!HasExceptionalOutput(m * n)) {
            return;
        }
        for (uint64_t column = 0; column < n; ++column) {
            if (!HasNonzeroWeight(column, k)) {
                continue;
            }
            for (uint64_t row = 0; row < m; ++row) {
                const uint64_t outputOffset = row * n + column;
                const uint16_t magnitude = cBitsGlobal_.GetValue(outputOffset) & 0x7FFFU;
                if (magnitude != 0U && magnitude != 0x7BFFU) {
                    continue;
                }
                cBitsGlobal_.SetValue(outputOffset, FloatToHalfBits(RecomputeDot(row, column, k)));
            }
        }
    }

    __aicore__ inline bool HasExceptionalOutput(uint64_t elements)
    {
        LocalTensor<half> values = addBuffer_.Get<half>();
        LocalTensor<float> floatBuffer = addBuffer_.Get<float>();
        LocalTensor<float> magnitudes = floatBuffer[kRepairChunkElements];
        LocalTensor<float> reduce = floatBuffer[2 * kRepairChunkElements];
        LocalTensor<float> work = reduce[32];
        for (uint64_t offset = 0; offset < elements; offset += kRepairChunkElements) {
            const uint64_t count = Min(kRepairChunkElements, elements - offset);
            DataCopy(values, cGlobal_[offset], count);
            PipeBarrier<PIPE_ALL>();
            Cast(magnitudes, values, RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_ALL>();
            Abs(magnitudes, magnitudes, count);
            ReduceMin<float>(reduce, magnitudes, work, count, false);
            PipeBarrier<PIPE_ALL>();
            if (reduce.GetValue(0) == 0.0f) {
                return true;
            }
            ReduceMax<float>(reduce, magnitudes, work, count, false);
            PipeBarrier<PIPE_ALL>();
            if (reduce.GetValue(0) == 65504.0f) {
                return true;
            }
        }
        return false;
    }

    __aicore__ inline bool HasNonzeroWeight(uint64_t column, uint64_t k)
    {
        LocalTensor<half> halfBuffer = addBuffer_.Get<half>();
        LocalTensor<half> weightHalf = halfBuffer[kRepairChunkElements];
        LocalTensor<float> floatBuffer = addBuffer_.Get<float>();
        LocalTensor<float> weightFloat = floatBuffer[2 * kRepairChunkElements];
        LocalTensor<float> reduce = floatBuffer[4 * kRepairChunkElements];
        LocalTensor<float> work = reduce[32];
        for (uint64_t offset = 0; offset < k; offset += kRepairChunkElements) {
            const uint64_t count = Min(kRepairChunkElements, k - offset);
            DataCopy(weightHalf, bGlobal_[column * k + offset], count);
            PipeBarrier<PIPE_ALL>();
            Cast(weightFloat, weightHalf, RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_ALL>();
            Abs(weightFloat, weightFloat, count);
            ReduceMax<float>(reduce, weightFloat, work, count, false);
            PipeBarrier<PIPE_ALL>();
            if (reduce.GetValue(0) != 0.0f) {
                return true;
            }
        }
        return false;
    }

    __aicore__ inline float RecomputeDot(uint64_t row, uint64_t column, uint64_t k)
    {
        LocalTensor<half> halfBuffer = addBuffer_.Get<half>();
        LocalTensor<half> lhsHalf = halfBuffer;
        LocalTensor<half> rhsHalf = halfBuffer[kRepairChunkElements];
        LocalTensor<float> floatBuffer = addBuffer_.Get<float>();
        LocalTensor<float> lhsFloat = floatBuffer[kRepairChunkElements];
        LocalTensor<float> rhsFloat = floatBuffer[2 * kRepairChunkElements];
        LocalTensor<float> product = floatBuffer[3 * kRepairChunkElements];
        LocalTensor<float> reduce = floatBuffer[4 * kRepairChunkElements];
        LocalTensor<float> work = reduce[32];
        float sum = 0.0f;
        for (uint64_t offset = 0; offset < k; offset += kRepairChunkElements) {
            const uint64_t count = Min(kRepairChunkElements, k - offset);
            DataCopy(lhsHalf, aGlobal_[row * k + offset], count);
            DataCopy(rhsHalf, bGlobal_[column * k + offset], count);
            PipeBarrier<PIPE_ALL>();
            Cast(lhsFloat, lhsHalf, RoundMode::CAST_NONE, count);
            Cast(rhsFloat, rhsHalf, RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_ALL>();
            Mul(product, lhsFloat, rhsFloat, count);
            PipeBarrier<PIPE_ALL>();
            ReduceSum<float>(reduce, product, work, count);
            PipeBarrier<PIPE_ALL>();
            sum += reduce.GetValue(0);
        }
        return sum;
    }

    MatmulImpl<A_TYPE_A1, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG> mm_;
    TQue<QuePosition::A1, 1> a1Queue_;
    TBuf<TPosition::VECCALC> addBuffer_;
    GlobalTensor<half> aGlobal_;
    GlobalTensor<half> bGlobal_;
    GlobalTensor<half> cGlobal_;
    GlobalTensor<uint16_t> cBitsGlobal_;
    GlobalTensor<half> tempGlobal_;
    const MatmulTilingData* tilingData_ = nullptr;
    TPipe* pipe_ = nullptr;
};

#endif
