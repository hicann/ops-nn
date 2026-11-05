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
 * \file quant_batch_matmul_v4.cpp
 * \brief KirinX90/Kirin9030 arch31 S8S4 kernel
 */

#include "kernel_operator.h"
#include "quant_batch_matmul_v4_tiling_data.h"

using namespace AscendC;
#if defined(__has_include) && __has_include("micro_api/kernel_micro_intf.h")
#include "micro_api/kernel_micro_intf.h"
#else
namespace MicroAPI = AscendC::Reg;
#endif

namespace {
constexpr uint32_t TILE_M = 64;
constexpr uint32_t TILE_N = 128;
constexpr uint32_t A_FRAC_M = 16;
constexpr uint32_t A_FRAC_K = 32;
constexpr uint32_t B_FRAC_N = 16;
constexpr uint32_t B_FRAC_K = 64;
constexpr uint32_t B_FRACS_PER_TILE = TILE_N / B_FRAC_N;
constexpr uint32_t A_FRAC_BYTES = A_FRAC_M * A_FRAC_K;
constexpr uint32_t B_FRAC_BYTES = 512;
constexpr uint32_t INT4_NUMS_IN_INT32 = 8;
constexpr uint32_t INT32_PER_B_FRAC = B_FRAC_BYTES / sizeof(int32_t);
constexpr uint32_t L1_K_M1 = 256;

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3113)
constexpr uint32_t M1_K_L0A_LIMIT = 2048;
#elif defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003)
constexpr uint32_t M1_K_L0A_LIMIT = 4096;
#endif

constexpr uint8_t EVT_L1 = 4;
constexpr uint8_t EVT_L0AB = 0;
constexpr uint8_t EVT_L0C = 0;
constexpr uint8_t EVT_S_SYNC = 0;
constexpr uint8_t EVT_WS0 = 0;
constexpr uint8_t EVT_WS1 = 1;
constexpr uint8_t EVT_S1_0 = 0;
constexpr uint8_t EVT_S1_1 = 1;
constexpr uint8_t EVT_PAR0 = 0;
constexpr uint8_t EVT_PAR1 = 1;
constexpr uint8_t EVT_YOUT = 0;
constexpr uint8_t EVT_X1S = 2;
constexpr uint8_t EVT_BIAS = 4;
constexpr uint8_t EVT_S1_DRAIN = 0;
constexpr uint8_t EVT_FIXDRAIN = 0;
constexpr uint8_t EVT_NV_L1AB = 0;
constexpr uint8_t EVT_NV_L1AB_PONG = 1;
constexpr uint8_t EVT_NV_L0A = 3;
constexpr uint8_t EVT_NV_L0A_PONG = 4;
constexpr uint8_t EVT_NV_L1 = 1;
constexpr uint8_t EVT_NV_L1_PONG = 0;
constexpr uint8_t EVT_NV_DEQ_REL = 0;
constexpr uint8_t EVT_NV_DEQ_REL_PONG = 1;
constexpr uint8_t EVT_NV_L0C = 0;
constexpr uint8_t EVT_NV_L0C_PONG = 1;
constexpr uint8_t EVT_NV_INS = 0;
constexpr uint8_t EVT_NV_YOUT = 0;
constexpr uint8_t EVT_NV_YOUT_PONG = 1;
constexpr uint8_t EVT_NV_PGD = 1;
constexpr uint8_t EVT_NV_PGD_PONG = 0;
constexpr uint8_t EVT_NV_PVU = 1;
constexpr uint8_t EVT_NV_PVU_PONG = 0;
constexpr uint8_t EVT_NV_FXM = 0;
constexpr uint8_t EVT_NV_FXM_PONG = 1;
constexpr uint8_t EVT_NV_MFX = 0;
constexpr uint8_t EVT_NV_MFX_PONG = 1;
constexpr uint8_t EVT_NV_X1S = 2;
constexpr uint8_t EVT_NV_BIAS = 4;
constexpr uint32_t PER_VEG_NUM = 64;

__aicore__ inline uint32_t CeilDivU(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

__aicore__ inline uint32_t AlignU(uint32_t a, uint32_t b) { return CeilDivU(a, b) * b; }
} // namespace

class QuantBatchMatmulV4Arch31Kernel {
public:
    static constexpr MicroAPI::CastTrait castF32F16_ = {
        MicroAPI::RegLayout::ZERO,
        MicroAPI::SatMode::SAT,
        MicroAPI::MaskMergeMode::ZEROING,
        RoundMode::CAST_RINT,
    };
    static constexpr MicroAPI::CastTrait castF16F32_ = {
        MicroAPI::RegLayout::ZERO,
        MicroAPI::SatMode::SAT,
        MicroAPI::MaskMergeMode::ZEROING,
        RoundMode::CAST_NONE,
    };

    __aicore__ inline QuantBatchMatmulV4Arch31Kernel() {}
    __aicore__ inline void Init(GM_ADDR x1GmAddr, GM_ADDR x2GmAddr, GM_ADDR biasGmAddr, GM_ADDR x1ScaleGmAddr,
                                GM_ADDR x2ScaleGmAddr, GM_ADDR yGmAddr, GM_ADDR workspaceGmAddr,
                                const QuantBatchMatmulV4Arch31TilingData& tiling)
    {
        m_ = static_cast<uint32_t>(tiling.m);
        n_ = static_cast<uint32_t>(tiling.n);
        k_ = static_cast<uint32_t>(tiling.k);
        gs_ = static_cast<uint32_t>(tiling.groupSize);
        n1Total_ = CeilDivU(n_, B_FRAC_N);
        gt_ = CeilDivU(k_, gs_);
        mt_ = CeilDivU(m_, TILE_M);
        nt_ = CeilDivU(n_, TILE_N);
        hasBias_ = (tiling.hasBias != 0);

        x1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(x1GmAddr));
        x2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(x2GmAddr));
        biasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(biasGmAddr));
        x1ScaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x1ScaleGmAddr));
        x2ScaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(x2ScaleGmAddr));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(yGmAddr));
        wsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(workspaceGmAddr));

        InitBuffers();
        SetFlag<HardEvent::MTE1_MTE2>(EVT_L1);
        SetFlag<HardEvent::FIX_M>(EVT_L0C);
        SetFlag<HardEvent::MTE2_FIX>(EVT_WS0);
        SetFlag<HardEvent::MTE2_FIX>(EVT_WS1);
        SetFlag<HardEvent::FIX_MTE3>(EVT_S1_0);
        if (gt_ > 1U) {
            SetFlag<HardEvent::FIX_MTE3>(EVT_S1_1);
        }
        SetFlag<HardEvent::V_MTE2>(EVT_PAR0);
        SetFlag<HardEvent::V_MTE2>(EVT_PAR1);
        SetFlag<HardEvent::V_MTE2>(EVT_X1S);
        SetFlag<HardEvent::V_MTE2>(EVT_BIAS);
        SetFlag<HardEvent::MTE3_V>(EVT_YOUT);
    }

    __aicore__ inline void Process()
    {
        for (uint32_t mt = 0; mt < mt_; ++mt) {
            for (uint32_t nt = 0; nt < nt_; ++nt) {
                ProcessTile(mt, nt);
            }
        }

        WaitFlag<HardEvent::MTE1_MTE2>(EVT_L1);
        WaitFlag<HardEvent::FIX_M>(EVT_L0C);
        WaitFlag<HardEvent::MTE2_FIX>(EVT_WS0);
        WaitFlag<HardEvent::MTE2_FIX>(EVT_WS1);
        WaitFlag<HardEvent::FIX_MTE3>(EVT_S1_0);
        if (gt_ > 1U) {
            WaitFlag<HardEvent::FIX_MTE3>(EVT_S1_1);
        }
        WaitFlag<HardEvent::V_MTE2>(EVT_PAR0);
        WaitFlag<HardEvent::V_MTE2>(EVT_PAR1);
        WaitFlag<HardEvent::V_MTE2>(EVT_X1S);
        WaitFlag<HardEvent::V_MTE2>(EVT_BIAS);
        WaitFlag<HardEvent::MTE3_V>(EVT_YOUT);
    }

private:
    __aicore__ inline void InitBuffers()
    {
        const uint32_t k1tMax = gs_ / A_FRAC_K;
        const uint32_t jMax = gs_ / B_FRAC_K;
        const uint32_t m1Max = TILE_M / A_FRAC_M;
        pipe_.InitBuffer(aBufL1_, 2U * m1Max * k1tMax * A_FRAC_BYTES);
        pipe_.InitBuffer(bBufL1_, 2U * jMax * B_FRACS_PER_TILE * B_FRAC_BYTES);
        pipe_.InitBuffer(sBufL1_, 2U * TILE_N * sizeof(uint64_t));
        pipe_.InitBuffer(aBufL0_, m1Max * k1tMax * A_FRAC_BYTES);
        pipe_.InitBuffer(bBufL0_, jMax * B_FRACS_PER_TILE * B_FRAC_BYTES);
        pipe_.InitBuffer(cBufL0_, TILE_M * TILE_N * sizeof(int32_t));
        pipe_.InitBuffer(accUb_, TILE_M * TILE_N * sizeof(float));
        pipe_.InitBuffer(partialUb_, 2U * TILE_M * TILE_N * sizeof(half));
        pipe_.InitBuffer(stagingFp32Ub_, TILE_M * TILE_N * sizeof(float));
        pipe_.InitBuffer(yOutUb_, TILE_M * TILE_N * sizeof(half));
        pipe_.InitBuffer(deqCompactUb_, TILE_N * sizeof(uint64_t));
        pipe_.InitBuffer(x1ScaleUb_, TILE_M * sizeof(float));
        pipe_.InitBuffer(biasUb_, 512U);
    }

    __aicore__ inline void ProcessTile(uint32_t mt, uint32_t nt)
    {
        const uint32_t m0 = mt * TILE_M;
        const uint32_t n0 = nt * TILE_N;
        const uint32_t mAct = Min(TILE_M, m_ - m0);
        const uint32_t nAct = Min(TILE_N, n_ - n0);
        const uint32_t tAct = CeilDivU(nAct, B_FRAC_N);
        const uint32_t nFracAligned = tAct * B_FRAC_N;
        const uint32_t n1t = CeilDivU(mAct, A_FRAC_M);
        const uint32_t mFracAligned = n1t * A_FRAC_M;
        const uint32_t jMax = gs_ / B_FRAC_K;
        LocalTensor<float> acc = accUb_.Get<float>();
        LocalTensor<half> yOut = yOutUb_.Get<half>();

        const uint32_t gLoopCap = gt_;
        for (uint32_t g = 0; g < gLoopCap; ++g) {
            const uint32_t p = g % 2U;
            const uint32_t k0 = g * gs_;
            const uint32_t kAct = Min(gs_, k_ - k0);
            const uint32_t k1t = kAct / A_FRAC_K;
            const uint32_t jTotal = kAct / B_FRAC_K;
            const uint32_t j0 = g * jMax;

            LocalTensor<int8_t> aL1 = aBufL1_.Get<int8_t>();
            LocalTensor<int32_t> bL1I32 = bBufL1_.Get<int32_t>();
            LocalTensor<int64_t> sL1I64 = sBufL1_.Get<int64_t>();
            LocalTensor<int8_t> aL0 = aBufL0_.Get<int8_t>();
            LocalTensor<int8_t> bL0 = bBufL0_.Get<int8_t>();
            LocalTensor<int32_t> cL0 = cBufL0_.Get<int32_t>();
            LocalTensor<half> partial = partialUb_.Get<half>();
            LocalTensor<uint64_t> sL1U64 = sBufL1_.Get<uint64_t>();

            const uint32_t aSlot = p * (gs_ / A_FRAC_K) * (TILE_M / A_FRAC_M) * A_FRAC_BYTES;
            const uint32_t bSlot = p * jMax * B_FRACS_PER_TILE * B_FRAC_BYTES;
            const uint32_t s1Base = p * nAct;
            const uint32_t wsBase = p * TILE_M * TILE_N;

            WaitFlag<HardEvent::MTE1_MTE2>(EVT_L1);
            WaitFlag<HardEvent::FIX_MTE3>(EVT_S1_0 + p);
            LoadX1ToL1(aL1[aSlot], m0, k0, mAct, kAct, n1t);
            LoadX2ToL1(bL1I32[bSlot / sizeof(int32_t)], j0, n0, tAct, jTotal);
            LoadScaleToL1(sL1I64[s1Base], g, n0, nAct);
            SetFlag<HardEvent::MTE2_S>(EVT_L1);
            WaitFlag<HardEvent::MTE2_S>(EVT_L1);

            if (g != 0U) {
                WaitFlag<HardEvent::M_S>(EVT_S_SYNC);
            }
            LoadL1ToL0A(aL0, aL1, aSlot, n1t, k1t);
            LoadL1ToL0B(bL0, bBufL1_.Get<int8_t>(), bSlot, tAct, jTotal);
            SetFlag<HardEvent::MTE1_M>(EVT_L0AB);
            SetFlag<HardEvent::MTE1_MTE2>(EVT_L1);
            WaitFlag<HardEvent::MTE1_M>(EVT_L0AB);
            WaitFlag<HardEvent::FIX_M>(EVT_L0C);
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003)
            mad_s8s4(reinterpret_cast<__cc__ int32_t*>(cL0.GetPhyAddr()),
                     reinterpret_cast<__ca__ int8_t*>(aL0.GetPhyAddr()),
                     reinterpret_cast<__cb__ void*>(bL0.GetPhyAddr()), 0, static_cast<uint16_t>(mFracAligned),
                     static_cast<uint16_t>(kAct), static_cast<uint16_t>(nFracAligned), 0, false, false, true);
#endif
            PipeBarrier<PIPE_M>();
            SetFlag<HardEvent::M_S>(EVT_S_SYNC);
            SetFlag<HardEvent::M_FIX>(EVT_L0C);
            WaitFlag<HardEvent::M_FIX>(EVT_L0C);

            FixpipeParamsV220 fixParams{};
            fixParams.mSize = static_cast<uint16_t>(mAct);
            fixParams.nSize = static_cast<uint16_t>(nAct);
            fixParams.srcStride = static_cast<uint16_t>(mFracAligned);
            fixParams.dstStride = TILE_N;
            fixParams.ndNum = 1;
            fixParams.quantPre = QuantMode_t::VDEQF16;
            fixParams.deqScalar = 0;
            WaitFlag<HardEvent::MTE2_FIX>(EVT_WS0 + p);
            Fixpipe(wsGm_[wsBase], cL0, sL1U64[s1Base], fixParams);
            SetFlag<HardEvent::FIX_S>(EVT_FIXDRAIN);
            WaitFlag<HardEvent::FIX_S>(EVT_FIXDRAIN);
            SetFlag<HardEvent::FIX_M>(EVT_L0C);
            SetFlag<HardEvent::FIX_MTE2>(EVT_WS0 + p);
            SetFlag<HardEvent::FIX_MTE3>(EVT_S1_0 + p);
            WaitFlag<HardEvent::FIX_MTE2>(EVT_WS0 + p);
            WaitFlag<HardEvent::V_MTE2>(EVT_PAR0 + p);

            DataCopyExtParams partialParams;
            partialParams.blockCount = mAct;
            partialParams.blockLen = TILE_N * sizeof(half);
            partialParams.srcStride = 0;
            partialParams.dstStride = 0;
            DataCopyPad(partial[p * TILE_M * TILE_N], wsGm_[wsBase], partialParams, DataCopyPadExtParams<half>());
            SetFlag<HardEvent::MTE2_FIX>(EVT_WS0 + p);
            SetFlag<HardEvent::MTE2_V>(EVT_PAR0 + p);
            WaitFlag<HardEvent::MTE2_V>(EVT_PAR0 + p);

            LocalTensor<float> staging = stagingFp32Ub_.Get<float>();
            if (g == 0U) {
                Cast(acc, partial[p * TILE_M * TILE_N], RoundMode::CAST_NONE, static_cast<int32_t>(mAct * TILE_N));
            } else {
                Cast(staging, partial[p * TILE_M * TILE_N], RoundMode::CAST_NONE, static_cast<int32_t>(mAct * TILE_N));
                Add(acc, acc, staging, static_cast<int32_t>(mAct * TILE_N));
            }
            SetFlag<HardEvent::V_MTE2>(EVT_PAR0 + p);
        }
        WaitFlag<HardEvent::M_S>(EVT_S_SYNC);
        WaitFlag<HardEvent::MTE3_V>(EVT_YOUT);
        WaitFlag<HardEvent::V_MTE2>(EVT_X1S);
        LocalTensor<float> x1ScaleUb = x1ScaleUb_.Get<float>();
        DataCopyExtParams x1sParams;
        x1sParams.blockCount = 1;
        x1sParams.blockLen = static_cast<uint16_t>(mAct * sizeof(float));
        x1sParams.srcStride = 0;
        x1sParams.dstStride = 0;
        DataCopyPad(x1ScaleUb, x1ScaleGm_[m0], x1sParams, DataCopyPadExtParams<float>());
        SetFlag<HardEvent::MTE2_V>(EVT_X1S);
        WaitFlag<HardEvent::MTE2_V>(EVT_X1S);

        LocalTensor<half> biasUb = biasUb_.Get<half>();
        uint32_t biasWinStart = 0;
        uint32_t biasOffset = 0;
        if (hasBias_) {
            WaitFlag<HardEvent::V_MTE2>(EVT_BIAS);
            biasWinStart = (n_ >= TILE_N) ? (n_ - TILE_N) : 0U;
            biasWinStart = Min(n0, biasWinStart);
            const uint32_t biasWinLen = Min(TILE_N, n_ - biasWinStart);
            biasOffset = n0 - biasWinStart;
            DataCopyExtParams biasParams;
            biasParams.blockCount = 1;
            biasParams.blockLen = static_cast<uint16_t>(biasWinLen * sizeof(half));
            biasParams.srcStride = 0;
            biasParams.dstStride = 0;
            DataCopyPad(biasUb, biasGm_[biasWinStart], biasParams, DataCopyPadExtParams<half>());
            SetFlag<HardEvent::MTE2_V>(EVT_BIAS);
            WaitFlag<HardEvent::MTE2_V>(EVT_BIAS);
        }
        __VEC_SCOPE__
        {
            __ubuf__ float* accAddr = reinterpret_cast<__ubuf__ float*>(acc.GetPhyAddr());
            __ubuf__ half* yOutAddr = reinterpret_cast<__ubuf__ half*>(yOut.GetPhyAddr());
            __ubuf__ float* scaleAddr = reinterpret_cast<__ubuf__ float*>(x1ScaleUb.GetPhyAddr());
            __ubuf__ half* biasAddr = reinterpret_cast<__ubuf__ half*>(biasUb.GetPhyAddr());
            uint16_t perRowLoopNum = static_cast<uint16_t>(CeilDivU(nAct, PER_VEG_NUM));
            uint16_t mRows = static_cast<uint16_t>(mAct);
            uint16_t biasOn = static_cast<uint16_t>(hasBias_ ? 1U : 0U);
            uint16_t biasOff = static_cast<uint16_t>(biasOffset);
            for (uint16_t rowIdx = 0; rowIdx < mRows; rowIdx++) {
                MicroAPI::RegTensor<float> bcastReg;
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(bcastReg, scaleAddr + rowIdx);
                uint32_t remaining = nAct;
                for (uint16_t j = 0; j < perRowLoopNum; j++) {
                    MicroAPI::MaskReg preg = MicroAPI::UpdateMask<float>(remaining);
                    MicroAPI::RegTensor<float> accReg;
                    MicroAPI::DataCopy(accReg, accAddr + rowIdx * TILE_N + j * PER_VEG_NUM);
                    MicroAPI::Mul<float>(accReg, accReg, bcastReg, preg);
                    if (biasOn != 0U) {
                        MicroAPI::RegTensor<half> biasH;
                        MicroAPI::DataCopy<half, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                            biasH, biasAddr + biasOff + j * PER_VEG_NUM);
                        MicroAPI::RegTensor<float> biasF;
                        MicroAPI::Cast<float, half, castF16F32_>(biasF, biasH, preg);
                        MicroAPI::Add<float>(accReg, accReg, biasF, preg);
                    }
                    MicroAPI::RegTensor<half> yReg;
                    MicroAPI::Cast<half, float, castF32F16_>(yReg, accReg, preg);
                    MicroAPI::DataCopy<half, MicroAPI::StoreDist::DIST_PACK_B32>(
                        yOutAddr + rowIdx * TILE_N + j * PER_VEG_NUM, yReg, preg);
                }
            }
        }
        SetFlag<HardEvent::V_MTE2>(EVT_X1S);
        if (hasBias_) {
            SetFlag<HardEvent::V_MTE2>(EVT_BIAS);
        }
        SetFlag<HardEvent::V_MTE3>(EVT_YOUT);
        WaitFlag<HardEvent::V_MTE3>(EVT_YOUT);

        if (nAct == TILE_N && n_ == TILE_N) {
            DataCopyParams outCopyParams;
            outCopyParams.blockCount = static_cast<uint16_t>(mAct);
            outCopyParams.blockLen = static_cast<uint16_t>(nAct * sizeof(half) / 32);
            outCopyParams.srcStride = 0;
            outCopyParams.dstStride = 0;
            DataCopy(yGm_[static_cast<uint64_t>(m0) * n_ + n0], yOut, outCopyParams);
        } else {
            for (uint32_t row = 0; row < mAct; ++row) {
                DataCopyExtParams outParams;
                outParams.blockCount = 1;
                outParams.blockLen = static_cast<uint16_t>(nAct * sizeof(half));
                outParams.srcStride = 0;
                outParams.dstStride = 0;
                DataCopyPad(yGm_[static_cast<uint64_t>(m0 + row) * n_ + n0], yOut[row * TILE_N], outParams);
            }
        }
        SetFlag<HardEvent::MTE3_V>(EVT_YOUT);
    }

    __aicore__ inline void LoadX1ToL1(LocalTensor<int8_t> dst, uint32_t m0, uint32_t k0, uint32_t mAct, uint32_t kAct,
                                      uint32_t n1t)
    {
        Nd2NzParams nd2nz;
        nd2nz.ndNum = 1;
        nd2nz.nValue = static_cast<uint16_t>(mAct);
        nd2nz.dValue = static_cast<uint16_t>(kAct);
        nd2nz.srcNdMatrixStride = 0;
        nd2nz.srcDValue = static_cast<uint16_t>(k_);
        nd2nz.dstNzC0Stride = static_cast<uint16_t>(n1t * A_FRAC_M);
        nd2nz.dstNzNStride = 1;
        nd2nz.dstNzMatrixStride = 0;
        DataCopy(dst, x1Gm_[static_cast<uint64_t>(m0) * k_ + k0], nd2nz);
    }

    __aicore__ inline void LoadX2ToL1(LocalTensor<int32_t> dst, uint32_t j0, uint32_t n0, uint32_t tAct,
                                      uint32_t jTotal)
    {
        const uint32_t t0 = n0 / B_FRAC_N;
        const uint64_t srcBase = static_cast<uint64_t>(j0 * n1Total_ + t0) * INT32_PER_B_FRAC;
        DataCopyExtParams bParams;
        bParams.blockCount = static_cast<uint16_t>(jTotal);
        bParams.blockLen = static_cast<uint32_t>(tAct * B_FRAC_BYTES);
        bParams.srcStride = static_cast<uint32_t>((n1Total_ - tAct) * B_FRAC_BYTES);
        bParams.dstStride = 0;
        bParams.rsv = 0;
        DataCopyPad(dst, x2Gm_[srcBase], bParams, DataCopyPadExtParams<int32_t>());
    }

    __aicore__ inline void LoadScaleToL1(LocalTensor<int64_t> dst, uint32_t g, uint32_t n0, uint32_t nAct)
    {
        if (gt_ == 1U) {
            LocalTensor<int64_t> deqCompact = deqCompactUb_.Get<int64_t>();
            DataCopyExtParams sParams;
            sParams.blockCount = 1;
            sParams.blockLen = static_cast<uint32_t>(nAct * sizeof(uint64_t));
            sParams.srcStride = 0;
            sParams.dstStride = 0;
            DataCopyPad(deqCompact, x2ScaleGm_[static_cast<uint64_t>(n0)], sParams, DataCopyPadExtParams<int64_t>());
            SetFlag<HardEvent::MTE2_S>(EVT_S_SYNC);
            WaitFlag<HardEvent::MTE2_S>(EVT_S_SYNC);
            DataCopyParams ub2l1;
            ub2l1.blockCount = 1;
            ub2l1.blockLen = static_cast<uint16_t>(nAct * sizeof(uint64_t) / 32U);
            ub2l1.srcGap = 0;
            ub2l1.dstGap = 0;
            DataCopy(dst, deqCompact, ub2l1);
            SetFlag<HardEvent::MTE3_MTE2>(EVT_S_SYNC);
            WaitFlag<HardEvent::MTE3_MTE2>(EVT_S_SYNC);
            SetFlag<HardEvent::MTE3_S>(EVT_S1_DRAIN);
            WaitFlag<HardEvent::MTE3_S>(EVT_S1_DRAIN);
        } else {
            DataCopyParams gm2l1;
            gm2l1.blockCount = 1;
            gm2l1.blockLen = static_cast<uint16_t>(nAct * sizeof(uint64_t) / 32U);
            gm2l1.srcGap = 0;
            gm2l1.dstGap = 0;
            DataCopy(dst, x2ScaleGm_[static_cast<uint64_t>(g) * n_ + n0], gm2l1);
        }
    }

    __aicore__ inline void LoadL1ToL0A(LocalTensor<int8_t> aL0, LocalTensor<int8_t> aL1, uint32_t aSlot, uint32_t n1t,
                                       uint32_t k1t)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3113)
        const uint32_t totalFracs = n1t * k1t;
        LoadData2DParamsV2 params;
        params.mStartPosition = 0;
        params.kStartPosition = 0;
        params.mStep = 1;
        params.kStep = static_cast<uint16_t>(totalFracs);
        params.srcStride = 1;
        params.dstStride = 1;
        params.ifTranspose = false;
        LoadData(aL0[0], aL1[aSlot], params);
#elif defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003)
        for (uint32_t m1 = 0; m1 < n1t; ++m1) {
            LoadData2DParams params;
            params.startIndex = 0;
            params.repeatTimes = static_cast<uint8_t>(k1t);
            params.srcStride = static_cast<uint16_t>(n1t);
            params.dstGap = 0;
            params.ifTranspose = false;
            LoadData(aL0[m1 * k1t * A_FRAC_BYTES], aL1[aSlot + m1 * A_FRAC_BYTES], params);
        }
#endif
    }

    __aicore__ inline void LoadL1ToL0B(LocalTensor<int8_t> bL0, LocalTensor<int8_t> bL1, uint32_t bSlot, uint32_t tAct,
                                       uint32_t jTotal)
    {
        const uint32_t totalFracs = jTotal * tAct;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003)
        LoadData2DParams params;
        params.startIndex = 0;
        params.repeatTimes = static_cast<uint8_t>(totalFracs);
        params.srcStride = 1;
        params.dstGap = 0;
        params.ifTranspose = false;
        LoadData(bL0[0], bL1[bSlot], params);
#elif defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3113)
        LoadData2DParamsV2 params;
        params.mStartPosition = 0;
        params.kStartPosition = 0;
        params.mStep = 1;
        params.kStep = static_cast<uint16_t>(totalFracs);
        params.srcStride = 1;
        params.dstStride = 1;
        params.ifTranspose = false;
        LoadData(bL0[0], bL1[bSlot], params);
#endif
    }

    __aicore__ inline uint32_t Min(uint32_t a, uint32_t b) { return (a < b) ? a : b; }

private:
    GlobalTensor<int8_t> x1Gm_;
    GlobalTensor<int32_t> x2Gm_;
    GlobalTensor<half> biasGm_;
    GlobalTensor<float> x1ScaleGm_;
    GlobalTensor<int64_t> x2ScaleGm_;
    GlobalTensor<half> yGm_;
    GlobalTensor<half> wsGm_;

    TPipe pipe_;
    TBuf<TPosition::A1> aBufL1_;
    TBuf<TPosition::A1> bBufL1_;
    TBuf<TPosition::A1> sBufL1_;
    TBuf<TPosition::A2> aBufL0_;
    TBuf<TPosition::B2> bBufL0_;
    TBuf<TPosition::CO1> cBufL0_;
    TBuf<TPosition::VECCALC> accUb_;

    TBuf<TPosition::VECCALC> partialUb_;
    TBuf<TPosition::VECCALC> stagingFp32Ub_;
    TBuf<TPosition::VECCALC> yOutUb_;
    TBuf<TPosition::VECCALC> deqCompactUb_;
    TBuf<TPosition::VECCALC> x1ScaleUb_;
    TBuf<TPosition::VECCALC> biasUb_;

    uint32_t m_ = 0;
    uint32_t n_ = 0;
    uint32_t k_ = 0;
    uint32_t gs_ = 0;
    uint32_t n1Total_ = 0;
    uint32_t gt_ = 0;
    uint32_t mt_ = 0;
    uint32_t nt_ = 0;
    bool hasBias_ = false;
};

class QuantBatchMatmulV4NewTplBase {
public:
    static constexpr MicroAPI::CastTrait castF32F16_ = {
        MicroAPI::RegLayout::ZERO,
        MicroAPI::SatMode::SAT,
        MicroAPI::MaskMergeMode::ZEROING,
        RoundMode::CAST_RINT,
    };
    static constexpr MicroAPI::CastTrait castF16F32_ = {
        MicroAPI::RegLayout::ZERO,
        MicroAPI::SatMode::SAT,
        MicroAPI::MaskMergeMode::ZEROING,
        RoundMode::CAST_NONE,
    };

    __aicore__ inline QuantBatchMatmulV4NewTplBase() {}

protected:
    __aicore__ inline void InitCommon(GM_ADDR x1GmAddr, GM_ADDR x2GmAddr, GM_ADDR biasGmAddr, GM_ADDR x1ScaleGmAddr,
                                      GM_ADDR x2ScaleGmAddr, GM_ADDR yGmAddr,
                                      const QuantBatchMatmulV4Arch31TilingData& tiling)
    {
        m_ = static_cast<uint32_t>(tiling.m);
        n_ = static_cast<uint32_t>(tiling.n);
        k_ = static_cast<uint32_t>(tiling.k);
        gs_ = static_cast<uint32_t>(tiling.groupSize);
        hasBias_ = (tiling.hasBias != 0);
        n1Total_ = CeilDivU(n_, B_FRAC_N);
        gt_ = CeilDivU(k_, gs_);

        x1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(x1GmAddr));
        x2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(x2GmAddr));
        biasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(biasGmAddr));
        x1ScaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x1ScaleGmAddr));
        x2ScaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(x2ScaleGmAddr));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(yGmAddr));
    }

    __aicore__ inline void LoadL1ToL0A(LocalTensor<int8_t> aL0, LocalTensor<int8_t> aL1, uint32_t aSlot, uint32_t n1t,
                                       uint32_t k1t)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3113)
        const uint32_t totalFracs = n1t * k1t;
        LoadData2DParamsV2 params;
        params.mStartPosition = 0;
        params.kStartPosition = 0;
        params.mStep = 1;
        params.kStep = static_cast<uint16_t>(totalFracs);
        params.srcStride = 1;
        params.dstStride = 1;
        params.ifTranspose = false;
        LoadData(aL0[0], aL1[aSlot], params);
#elif defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003)
        for (uint32_t m1 = 0; m1 < n1t; ++m1) {
            LoadData2DParams params;
            params.startIndex = 0;
            params.repeatTimes = static_cast<uint8_t>(k1t);
            params.srcStride = static_cast<uint16_t>(n1t);
            params.dstGap = 0;
            params.ifTranspose = false;
            LoadData(aL0[m1 * k1t * A_FRAC_BYTES], aL1[aSlot + m1 * A_FRAC_BYTES], params);
        }
#endif
    }

    __aicore__ inline uint32_t Min(uint32_t a, uint32_t b) { return (a < b) ? a : b; }

    GlobalTensor<int8_t> x1Gm_;
    GlobalTensor<int32_t> x2Gm_;
    GlobalTensor<half> biasGm_;
    GlobalTensor<float> x1ScaleGm_;
    GlobalTensor<int64_t> x2ScaleGm_;
    GlobalTensor<half> yGm_;

    TPipe pipe_;
    TBuf<TPosition::VECCALC> deqCompactUb_;

    uint32_t m_ = 0;
    uint32_t n_ = 0;
    uint32_t k_ = 0;
    uint32_t gs_ = 0;
    uint32_t n1Total_ = 0;
    uint32_t gt_ = 0;
    bool hasBias_ = false;
};

class QuantBatchMatmulV4M1Kernel : public QuantBatchMatmulV4NewTplBase {
public:
    __aicore__ inline QuantBatchMatmulV4M1Kernel() {}
    __aicore__ inline void Init(GM_ADDR x1GmAddr, GM_ADDR x2GmAddr, GM_ADDR biasGmAddr, GM_ADDR x1ScaleGmAddr,
                                GM_ADDR x2ScaleGmAddr, GM_ADDR yGmAddr,
                                const QuantBatchMatmulV4Arch31TilingData& tiling)
    {
        InitCommon(x1GmAddr, x2GmAddr, biasGmAddr, x1ScaleGmAddr, x2ScaleGmAddr, yGmAddr, tiling);
        nt_ = CeilDivU(n_, TILE_N);
        kOverL0a_ = (k_ > M1_K_L0A_LIMIT);
        l1kEff_ = L1_K_M1;
        if (kOverL0a_) {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3113)
            if (k_ % 256U == 0U) {
                l1kEff_ = 256U;
            } else {
                l1kEff_ = 128U;
            }
#elif defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003)
            if (k_ % 512U == 0U) {
                l1kEff_ = 512U;
            } else if (k_ % 256U == 0U) {
                l1kEff_ = 256U;
            } else {
                l1kEff_ = 128U;
            }
#endif
        }
        k1t_ = k_ / l1kEff_;
        l0Loops_ = l1kEff_ / gs_;
        InitBuffers();
        InitTokens();
    }

    __aicore__ inline void Process()
    {
        LoadX1AllToL1ThenL0A();
        {
            LocalTensor<float> x1ScaleUb = x1ScaleUb_.Get<float>();
            DataCopyExtParams x1sParams;
            x1sParams.blockCount = 1;
            x1sParams.blockLen = static_cast<uint16_t>(sizeof(float));
            x1sParams.srcStride = 0;
            x1sParams.dstStride = 0;
            DataCopyPad(x1ScaleUb, x1ScaleGm_[0], x1sParams, DataCopyPadExtParams<float>());
            SetFlag<HardEvent::MTE2_V>(EVT_NV_X1S);
            WaitFlag<HardEvent::MTE2_V>(EVT_NV_X1S);
        }
        if (nt_ != 0U) {
            LoadDeqScaleForNTile(0U, 0U, Min(TILE_N, n_));
            LocalTensor<int8_t> bL0 = bBufL0_.Get<int8_t>();
            const uint32_t tAct0 = CeilDivU(Min(TILE_N, n_), B_FRAC_N);
            LoadX2GMToL0B(bL0, 0U, 0U, 0U, tAct0, l1kEff_ / B_FRAC_K);
            SetFlag<HardEvent::MTE2_M>(EVT_NV_L1AB);
        }
        for (uint32_t nt = 0; nt < nt_; ++nt) {
            const uint32_t n0 = nt * TILE_N;
            const uint32_t nAct = Min(TILE_N, n_ - n0);
            WaitFlag<HardEvent::MTE2_FIX>((nt % 2U == 0U) ? EVT_NV_L1 : EVT_NV_L1_PONG);
            const bool hasNextTile = (nt + 1U < nt_);
            uint32_t nextT0 = 0;
            uint32_t nextTAct = 0;
            if (hasNextTile) {
                const uint32_t nNext = (nt + 1U) * TILE_N;
                const uint32_t nNextAct = Min(TILE_N, n_ - nNext);
                if (nt >= 1U) {
                    WaitFlag<HardEvent::FIX_MTE2>(((nt + 1U) % 2U == 0U) ? EVT_NV_DEQ_REL : EVT_NV_DEQ_REL_PONG);
                }
                LoadDeqScaleForNTile((nt + 1U) % 2U, nNext, nNextAct);
                nextT0 = nNext / B_FRAC_N;
                nextTAct = CeilDivU(nNextAct, B_FRAC_N);
            }
            ProcessNTile(nt, n0, nAct, hasNextTile, nextT0, nextTAct);
        }
        WaitFlag<HardEvent::V_MTE2>(EVT_NV_BIAS);
        WaitFlag<HardEvent::MTE3_V>(EVT_NV_YOUT);
        WaitFlag<HardEvent::MTE3_V>(EVT_NV_YOUT_PONG);
        if (nt_ >= 2U) {
            WaitFlag<HardEvent::FIX_MTE2>(EVT_NV_DEQ_REL);
            WaitFlag<HardEvent::FIX_MTE2>(EVT_NV_DEQ_REL_PONG);
        } else if (nt_ == 1U) {
            WaitFlag<HardEvent::FIX_MTE2>(EVT_NV_DEQ_REL);
        }
        WaitFlag<HardEvent::M_MTE2>(EVT_NV_L1AB);
        WaitFlag<HardEvent::M_MTE2>(EVT_NV_L1AB_PONG);
        if (kOverL0a_) {
            WaitFlag<HardEvent::M_MTE1>(EVT_NV_L0A);
            WaitFlag<HardEvent::M_MTE1>(EVT_NV_L0A_PONG);
        }
    }

private:
    __aicore__ inline void InitBuffers()
    {
        pipe_.InitBuffer(aBufL1_, k_ * A_FRAC_M);
        pipe_.InitBuffer(bBufL0_, 2U * l1kEff_ * TILE_N / 2);
        pipe_.InitBuffer(sBufL1_, 2U * gt_ * TILE_N * sizeof(uint64_t));
        pipe_.InitBuffer(outBufL1_, 2U * A_FRAC_M * TILE_N * sizeof(half));
        if (kOverL0a_) {
            pipe_.InitBuffer(aBufL0_, 2U * l1kEff_ * A_FRAC_M);
        } else {
            pipe_.InitBuffer(aBufL0_, k_ * A_FRAC_M);
        }
        pipe_.InitBuffer(cBufL0_, 2U * A_FRAC_M * TILE_N * sizeof(int32_t));
        pipe_.InitBuffer(accUb_, A_FRAC_M * TILE_N * sizeof(float));
        pipe_.InitBuffer(partialUb_, 2U * A_FRAC_M * TILE_N * sizeof(half));
        pipe_.InitBuffer(stagingFp32Ub_, A_FRAC_M * TILE_N * sizeof(float));
        pipe_.InitBuffer(yOutUb_, 2U * A_FRAC_M * TILE_N * sizeof(half));
        pipe_.InitBuffer(deqCompactUb_, TILE_N * sizeof(uint64_t));
        pipe_.InitBuffer(x1ScaleUb_, A_FRAC_M * sizeof(float));
        pipe_.InitBuffer(biasUb_, 512U);
    }

    __aicore__ inline void InitTokens()
    {
        SetFlag<HardEvent::V_MTE2>(EVT_NV_BIAS);
        SetFlag<HardEvent::MTE3_V>(EVT_NV_YOUT);
        SetFlag<HardEvent::MTE3_V>(EVT_NV_YOUT_PONG);
        SetFlag<HardEvent::M_MTE2>(EVT_NV_L1AB_PONG);
        if (kOverL0a_) {
            SetFlag<HardEvent::M_MTE1>(EVT_NV_L0A);
            SetFlag<HardEvent::M_MTE1>(EVT_NV_L0A_PONG);
        }
    }

    __aicore__ inline void LoadX1AllToL1ThenL0A()
    {
        LocalTensor<int8_t> aL1 = aBufL1_.Get<int8_t>();
        LocalTensor<int8_t> aL0 = aBufL0_.Get<int8_t>();
        Nd2NzParams nd2nz;
        nd2nz.ndNum = 1;
        nd2nz.nValue = 1;
        nd2nz.dValue = static_cast<uint16_t>(k_);
        nd2nz.srcNdMatrixStride = 0;
        nd2nz.srcDValue = static_cast<uint16_t>(k_);
        nd2nz.dstNzC0Stride = static_cast<uint16_t>(A_FRAC_M);
        nd2nz.dstNzNStride = 1;
        nd2nz.dstNzMatrixStride = 0;
        DataCopy(aL1, x1Gm_[0], nd2nz);
        SetFlag<HardEvent::MTE2_MTE1>(EVT_NV_L1AB);
        WaitFlag<HardEvent::MTE2_MTE1>(EVT_NV_L1AB);

        if (kOverL0a_) {
            return;
        }
        const uint32_t k1tTotal = k_ / A_FRAC_K;
        LoadL1ToL0A(aL0, aL1, 0U, 1U, k1tTotal);
        SetFlag<HardEvent::MTE1_M>(EVT_NV_L1AB);
    }

    __aicore__ inline void LoadX1SegL1ToL0A(LocalTensor<int8_t> aL0, LocalTensor<int8_t> aL1, uint32_t srcOff,
                                            uint32_t dstOff, uint32_t k1tSeg)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3113)
        LoadData2DParamsV2 params;
        params.mStartPosition = 0;
        params.kStartPosition = 0;
        params.mStep = 1;
        params.kStep = static_cast<uint16_t>(k1tSeg);
        params.srcStride = 1;
        params.dstStride = 1;
        params.ifTranspose = false;
        LoadData(aL0[dstOff], aL1[srcOff], params);
#elif defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003)
        LoadData2DParams params;
        params.startIndex = 0;
        params.repeatTimes = static_cast<uint8_t>(k1tSeg);
        params.srcStride = 1;
        params.dstGap = 0;
        params.ifTranspose = false;
        LoadData(aL0[dstOff], aL1[srcOff], params);
#endif
    }

    __aicore__ inline void LoadDeqScaleForNTile(uint32_t slot, uint32_t n0, uint32_t nAct)
    {
        LocalTensor<int64_t> sL1All = sBufL1_.Get<int64_t>();
        LocalTensor<int64_t> sL1I64 = sL1All[slot * gt_ * TILE_N];
        if (gt_ == 1U) {
            LocalTensor<int64_t> deqCompact = deqCompactUb_.Get<int64_t>();
            DataCopyExtParams sParams;
            sParams.blockCount = 1;
            sParams.blockLen = static_cast<uint32_t>(nAct * sizeof(uint64_t));
            sParams.srcStride = 0;
            sParams.dstStride = 0;
            DataCopyPad(deqCompact, x2ScaleGm_[static_cast<uint64_t>(n0)], sParams, DataCopyPadExtParams<int64_t>());
            SetFlag<HardEvent::MTE2_S>(EVT_NV_BIAS);
            WaitFlag<HardEvent::MTE2_S>(EVT_NV_BIAS);
            DataCopyParams ub2l1;
            ub2l1.blockCount = 1;
            ub2l1.blockLen = static_cast<uint16_t>(nAct * sizeof(uint64_t) / 32U);
            ub2l1.srcStride = 0;
            ub2l1.dstStride = 0;
            DataCopy(sL1I64, deqCompact, ub2l1);
            SetFlag<HardEvent::MTE3_MTE2>(EVT_NV_BIAS);
            WaitFlag<HardEvent::MTE3_MTE2>(EVT_NV_BIAS);
        } else {
            DataCopyParams gm2l1;
            gm2l1.blockCount = static_cast<uint16_t>(gt_);
            gm2l1.blockLen = static_cast<uint16_t>(nAct * sizeof(uint64_t) / 32U);
            gm2l1.srcStride = static_cast<uint16_t>((n_ - nAct) * sizeof(uint64_t) / 32U);
            gm2l1.dstStride = 0;
            DataCopy(sL1I64, x2ScaleGm_[static_cast<uint64_t>(n0)], gm2l1);
        }
        SetFlag<HardEvent::MTE2_FIX>((slot == 0U) ? EVT_NV_L1 : EVT_NV_L1_PONG);
    }

    __aicore__ inline void LoadX2GMToL0B(LocalTensor<int8_t> bL0, uint32_t bSlot, uint32_t j0Start, uint32_t t0,
                                         uint32_t tAct, uint32_t jTotal)
    {
        const uint32_t bytesPerRow = tAct * B_FRAC_BYTES;
        __gm__ int8_t* gmByteBase = reinterpret_cast<__gm__ int8_t*>(const_cast<__gm__ int32_t*>(x2Gm_.GetPhyAddr()));

        for (uint32_t j = 0; j < jTotal; ++j) {
            const uint32_t j0 = j0Start + j;
            const uint64_t srcByteOffset = (static_cast<uint64_t>(j0) * n1Total_ + t0) * B_FRAC_BYTES;
            const uint32_t dstByteOffset = bSlot + static_cast<uint32_t>(j) * bytesPerRow;

            GlobalTensor<int8_t> srcView;
            srcView.SetGlobalBuffer(gmByteBase + srcByteOffset);
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3113)
            LoadData2DParamsV2 params;
            params.mStartPosition = 0;
            params.kStartPosition = 0;
            params.mStep = 1;
            params.kStep = static_cast<uint16_t>(tAct);
            params.srcStride = 1;
            params.dstStride = static_cast<uint16_t>(tAct);
            params.ifTranspose = false;
            params.sid = 0;
            LoadData(bL0[dstByteOffset], srcView, params);
#elif defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003)
            LoadData2DParams params;
            params.startIndex = 0;
            params.repeatTimes = static_cast<uint8_t>(tAct);
            params.srcStride = 1;
            params.dstGap = 0;
            params.sid = 0;
            params.ifTranspose = false;
            params.addrMode = 0;
            LoadData(bL0[dstByteOffset], srcView, params);
#endif
        }
    }

    __aicore__ inline void ProcessNTile(uint32_t nt, uint32_t n0, uint32_t nAct, bool hasNextTile, uint32_t nextT0,
                                        uint32_t nextTAct)
    {
        const uint32_t t0 = n0 / B_FRAC_N;
        const uint32_t tAct = CeilDivU(nAct, B_FRAC_N);
        const uint32_t nFracAligned = tAct * B_FRAC_N;
        const uint32_t jTotalL1 = l1kEff_ / B_FRAC_K;

        LocalTensor<uint64_t> sU64 = sBufL1_.Get<uint64_t>()[(nt % 2U) * gt_ * TILE_N];
        LocalTensor<int8_t> aL0 = aBufL0_.Get<int8_t>();
        LocalTensor<int8_t> aL1Local = aBufL1_.Get<int8_t>();
        LocalTensor<int8_t> bL0 = bBufL0_.Get<int8_t>();
        LocalTensor<int32_t> cL0 = cBufL0_.Get<int32_t>();
        LocalTensor<half> l1Out = outBufL1_.Get<half>();
        LocalTensor<half> partial = partialUb_.Get<half>();
        LocalTensor<float> acc = accUb_.Get<float>();
        LocalTensor<float> staging = stagingFp32Ub_.Get<float>();

        for (uint32_t j1 = 0; j1 < k1t_; ++j1) {
            const uint32_t seg = nt * k1t_ + j1;
            const uint32_t midPP = seg % 2U;
            const uint8_t bEvt = (midPP == 0U) ? EVT_NV_L1AB : EVT_NV_L1AB_PONG;
            const uint8_t bEvtL0a = (midPP == 0U) ? EVT_NV_L0A : EVT_NV_L0A_PONG;
            const uint32_t k1Base = j1 * l1kEff_;
            const uint32_t bSlot = midPP * l1kEff_ * TILE_N / 2;
            const uint32_t s1SegBase = j1 * (l1kEff_ / gs_) * TILE_N;
            WaitFlag<HardEvent::MTE2_M>(bEvt);
            if (kOverL0a_) {
                WaitFlag<HardEvent::M_MTE1>(bEvtL0a);
                LoadX1SegL1ToL0A(aL0, aL1Local, k1Base * A_FRAC_M, midPP * l1kEff_ * A_FRAC_M, l1kEff_ / A_FRAC_K);
                SetFlag<HardEvent::MTE1_M>(bEvt);
            }
            if (j1 + 1U < k1t_ || hasNextTile) {
                const uint32_t nextPP = (seg + 1U) % 2U;
                const uint8_t nEvt = (nextPP == 0U) ? EVT_NV_L1AB : EVT_NV_L1AB_PONG;
                WaitFlag<HardEvent::M_MTE2>(nEvt);
                const uint32_t nextJ0 = (j1 + 1U < k1t_) ? (j1 + 1U) * l1kEff_ / B_FRAC_K : 0U;
                const uint32_t nextTx = (j1 + 1U < k1t_) ? t0 : nextT0;
                const uint32_t nextTa = (j1 + 1U < k1t_) ? tAct : nextTAct;
                LoadX2GMToL0B(bL0, nextPP * l1kEff_ * TILE_N / 2, nextJ0, nextTx, nextTa, jTotalL1);
                SetFlag<HardEvent::MTE2_M>(nEvt);
            }
            SetFlag<HardEvent::FIX_M>(EVT_NV_L0C);
            SetFlag<HardEvent::FIX_M>(EVT_NV_L0C_PONG);
            SetFlag<HardEvent::MTE1_FIX>(EVT_NV_MFX);
            SetFlag<HardEvent::MTE1_FIX>(EVT_NV_MFX_PONG);
            SetFlag<HardEvent::V_MTE1>(EVT_NV_PVU);
            SetFlag<HardEvent::V_MTE1>(EVT_NV_PVU_PONG);
            for (uint32_t i0 = 0; i0 < l0Loops_; ++i0) {
                const uint32_t innerPP = (j1 * l0Loops_ + i0) % 2U;
                const uint8_t l0cEvt = (innerPP == 0U) ? EVT_NV_L0C : EVT_NV_L0C_PONG;
                const uint8_t pvuEvt = (innerPP == 0U) ? EVT_NV_PVU : EVT_NV_PVU_PONG;
                const uint8_t mfxEvt = (innerPP == 0U) ? EVT_NV_MFX : EVT_NV_MFX_PONG;
                const uint8_t fxmEvt = (innerPP == 0U) ? EVT_NV_FXM : EVT_NV_FXM_PONG;
                const uint8_t pgdEvt = (innerPP == 0U) ? EVT_NV_PGD : EVT_NV_PGD_PONG;
                const uint32_t l1OutSlot = innerPP * A_FRAC_M * TILE_N;
                const uint32_t l0aOffset = kOverL0a_ ? (midPP * l1kEff_ + i0 * gs_) * A_FRAC_M :
                                                       (k1Base + i0 * gs_) * A_FRAC_M;
                const uint32_t s1Offset = s1SegBase + i0 * TILE_N;
                const uint32_t l0bOffset = i0 * (gs_ / B_FRAC_K * tAct * B_FRAC_BYTES);
                if (kOverL0a_) {
                    if (i0 == 0U) {
                        WaitFlag<HardEvent::MTE1_M>(bEvt);
                    }
                } else if (nt == 0U && j1 == 0U && i0 == 0U) {
                    WaitFlag<HardEvent::MTE1_M>(EVT_NV_L1AB);
                }
                WaitFlag<HardEvent::FIX_M>(l0cEvt);
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003)
                mad_s8s4(reinterpret_cast<__cc__ int32_t*>(cL0.GetPhyAddr()) + innerPP * A_FRAC_M * TILE_N,
                         reinterpret_cast<__ca__ int8_t*>(aL0.GetPhyAddr() + l0aOffset),
                         reinterpret_cast<__cb__ void*>(bL0.GetPhyAddr() + bSlot + l0bOffset), 0,
                         static_cast<uint16_t>(A_FRAC_M), static_cast<uint16_t>(gs_),
                         static_cast<uint16_t>(nFracAligned), 0, false, false, true);
#endif
                SetFlag<HardEvent::M_FIX>(l0cEvt);
                WaitFlag<HardEvent::M_FIX>(l0cEvt);
                if (i0 == l0Loops_ - 1U) {
                    SetFlag<HardEvent::M_MTE2>(bEvt);
                    if (kOverL0a_) {
                        SetFlag<HardEvent::M_MTE1>(bEvtL0a);
                    }
                }
                WaitFlag<HardEvent::MTE1_FIX>(mfxEvt);
                FixpipeParamsV220 fixParams{};
                fixParams.mSize = 1;
                fixParams.nSize = static_cast<uint16_t>(nAct);
                fixParams.srcStride = static_cast<uint16_t>(A_FRAC_M);
                fixParams.dstStride = TILE_N;
                fixParams.ndNum = 1;
                fixParams.quantPre = QuantMode_t::VDEQF16;
                fixParams.deqScalar = 0;
                Fixpipe(l1Out[l1OutSlot], cL0[innerPP * A_FRAC_M * TILE_N], sU64[s1Offset], fixParams);
                SetFlag<HardEvent::FIX_M>(l0cEvt);
                SetFlag<HardEvent::FIX_MTE1>(fxmEvt);
                WaitFlag<HardEvent::FIX_MTE1>(fxmEvt);
                WaitFlag<HardEvent::V_MTE1>(pvuEvt);
                {
                    DataCopyParams l12ub;
                    l12ub.blockCount = 1;
                    l12ub.blockLen = static_cast<uint16_t>(nAct * sizeof(half) / 32U);
                    l12ub.srcStride = 0;
                    l12ub.dstStride = 0;
                    DataCopy(partial[innerPP * A_FRAC_M * TILE_N], l1Out[l1OutSlot], l12ub);
                }
                SetFlag<HardEvent::MTE1_FIX>(mfxEvt);
                SetFlag<HardEvent::MTE1_V>(pgdEvt);
                WaitFlag<HardEvent::MTE1_V>(pgdEvt);
                const bool isFirst = (j1 == 0U && i0 == 0U);
                if (isFirst) {
                    Cast(acc, partial[innerPP * A_FRAC_M * TILE_N], RoundMode::CAST_NONE, static_cast<int32_t>(TILE_N));
                } else {
                    Cast(staging, partial[innerPP * A_FRAC_M * TILE_N], RoundMode::CAST_NONE,
                         static_cast<int32_t>(TILE_N));
                    Add(acc, acc, staging, static_cast<int32_t>(TILE_N));
                }
                SetFlag<HardEvent::V_MTE1>(pvuEvt);
            }
            WaitFlag<HardEvent::FIX_M>(EVT_NV_L0C);
            WaitFlag<HardEvent::FIX_M>(EVT_NV_L0C_PONG);
            WaitFlag<HardEvent::MTE1_FIX>(EVT_NV_MFX);
            WaitFlag<HardEvent::MTE1_FIX>(EVT_NV_MFX_PONG);
            WaitFlag<HardEvent::V_MTE1>(EVT_NV_PVU);
            WaitFlag<HardEvent::V_MTE1>(EVT_NV_PVU_PONG);
        }
        SetFlag<HardEvent::FIX_MTE2>((nt % 2U == 0U) ? EVT_NV_DEQ_REL : EVT_NV_DEQ_REL_PONG);
        FinalizeTile(acc, n0, nAct, nt);
    }

    __aicore__ inline void FinalizeTile(LocalTensor<float> acc, uint32_t n0, uint32_t nAct, uint32_t nt)
    {
        const uint32_t outPP = nt % 2U;
        const uint8_t yEvt = (outPP == 0U) ? EVT_NV_YOUT : EVT_NV_YOUT_PONG;
        LocalTensor<half> yOut = yOutUb_.Get<half>()[outPP * A_FRAC_M * TILE_N];
        WaitFlag<HardEvent::MTE3_V>(yEvt);
        LocalTensor<float> x1ScaleUb = x1ScaleUb_.Get<float>();
        LocalTensor<half> biasUb = biasUb_.Get<half>();
        uint32_t biasWinStart = 0;
        uint32_t biasOffset = 0;
        if (hasBias_) {
            WaitFlag<HardEvent::V_MTE2>(EVT_NV_BIAS);
            biasWinStart = (n_ >= TILE_N) ? (n_ - TILE_N) : 0U;
            biasWinStart = Min(n0, biasWinStart);
            const uint32_t biasWinLen = Min(TILE_N, n_ - biasWinStart);
            biasOffset = n0 - biasWinStart;
            DataCopyExtParams biasParams;
            biasParams.blockCount = 1;
            biasParams.blockLen = static_cast<uint16_t>(biasWinLen * sizeof(half));
            biasParams.srcStride = 0;
            biasParams.dstStride = 0;
            DataCopyPad(biasUb, biasGm_[biasWinStart], biasParams, DataCopyPadExtParams<half>());
            SetFlag<HardEvent::MTE2_V>(EVT_NV_BIAS);
            WaitFlag<HardEvent::MTE2_V>(EVT_NV_BIAS);
        }
        __VEC_SCOPE__
        {
            __ubuf__ float* accAddr = reinterpret_cast<__ubuf__ float*>(acc.GetPhyAddr());
            __ubuf__ half* yOutAddr = reinterpret_cast<__ubuf__ half*>(yOut.GetPhyAddr());
            __ubuf__ float* scaleAddr = reinterpret_cast<__ubuf__ float*>(x1ScaleUb.GetPhyAddr());
            __ubuf__ half* biasAddr = reinterpret_cast<__ubuf__ half*>(biasUb.GetPhyAddr());
            uint16_t perRowLoopNum = static_cast<uint16_t>(CeilDivU(nAct, PER_VEG_NUM));
            uint16_t biasOn = static_cast<uint16_t>(hasBias_ ? 1U : 0U);
            uint16_t biasOff = static_cast<uint16_t>(biasOffset);
            MicroAPI::RegTensor<float> bcastReg;
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(bcastReg, scaleAddr);
            uint32_t remaining = nAct;
            for (uint16_t j = 0; j < perRowLoopNum; j++) {
                MicroAPI::MaskReg preg = MicroAPI::UpdateMask<float>(remaining);
                MicroAPI::RegTensor<float> accReg;
                MicroAPI::DataCopy(accReg, accAddr + j * PER_VEG_NUM);
                MicroAPI::Mul<float>(accReg, accReg, bcastReg, preg);
                if (biasOn != 0U) {
                    MicroAPI::RegTensor<half> biasH;
                    MicroAPI::DataCopy<half, MicroAPI::LoadDist::DIST_UNPACK_B16>(biasH,
                                                                                  biasAddr + biasOff + j * PER_VEG_NUM);
                    MicroAPI::RegTensor<float> biasF;
                    MicroAPI::Cast<float, half, castF16F32_>(biasF, biasH, preg);
                    MicroAPI::Add<float>(accReg, accReg, biasF, preg);
                }
                MicroAPI::RegTensor<half> yReg;
                MicroAPI::Cast<half, float, castF32F16_>(yReg, accReg, preg);
                MicroAPI::DataCopy<half, MicroAPI::StoreDist::DIST_PACK_B32>(yOutAddr + j * PER_VEG_NUM, yReg, preg);
            }
        }
        if (hasBias_) {
            SetFlag<HardEvent::V_MTE2>(EVT_NV_BIAS);
        }
        SetFlag<HardEvent::V_MTE3>(yEvt);
        WaitFlag<HardEvent::V_MTE3>(yEvt);

        DataCopyExtParams outParams;
        outParams.blockCount = 1;
        outParams.blockLen = static_cast<uint16_t>(nAct * sizeof(half));
        outParams.srcStride = 0;
        outParams.dstStride = 0;
        DataCopyPad(yGm_[n0], yOut, outParams);
        SetFlag<HardEvent::MTE3_V>(yEvt);
    }

    TBuf<TPosition::A1> aBufL1_;
    TBuf<TPosition::A1> sBufL1_;
    TBuf<TPosition::C1> outBufL1_;
    TBuf<TPosition::A2> aBufL0_;
    TBuf<TPosition::B2> bBufL0_;
    TBuf<TPosition::CO1> cBufL0_;
    TBuf<TPosition::VECCALC> accUb_;
    TBuf<TPosition::VECCALC> partialUb_;
    TBuf<TPosition::VECCALC> stagingFp32Ub_;
    TBuf<TPosition::VECCALC> yOutUb_;
    TBuf<TPosition::VECCALC> x1ScaleUb_;
    TBuf<TPosition::VECCALC> biasUb_;
    TBuf<TPosition::VECCALC> deqCompactUb_;

    uint32_t nt_ = 0;
    uint32_t k1t_ = 0;
    uint32_t l0Loops_ = 0;
    uint32_t l1kEff_ = L1_K_M1;
    bool kOverL0a_ = false;
};

extern "C" __global__ __aicore__ void quant_batch_matmul_v4(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR x1Scale,
                                                            GM_ADDR x2Scale, GM_ADDR yScale, GM_ADDR x1Offset,
                                                            GM_ADDR x2Offset, GM_ADDR yOffset, GM_ADDR x2Table,
                                                            GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(QuantBatchMatmulV4Arch31TilingData);
    GET_TILING_DATA_WITH_STRUCT(QuantBatchMatmulV4Arch31TilingData, tilingData, tiling);
    if (tilingData.m <= 0 || tilingData.n <= 0 || tilingData.k <= 0 || tilingData.groupSize <= 0) {
        return;
    }
    if (TILING_KEY_IS(1U)) {
        QuantBatchMatmulV4M1Kernel op;
        op.Init(x1, x2, bias, x1Scale, x2Scale, y, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(0U)) {
        GM_ADDR userWorkspace = GetUserWorkspace(workspace);
        QuantBatchMatmulV4Arch31Kernel op;
        op.Init(x1, x2, bias, x1Scale, x2Scale, y, userWorkspace, tilingData);
        op.Process();
    }
}
