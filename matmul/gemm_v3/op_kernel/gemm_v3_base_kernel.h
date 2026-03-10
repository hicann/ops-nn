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
 * \file gemm_v3_base_kernel.h
 * \brief
 */

#ifndef GEMM_V3_BASE_KERNEL_H
#define GEMM_V3_BASE_KERNEL_H

#ifdef __CCE_KT_TEST__
#include "stub_def.h"
#include "stub_fun.h"
#else
#define __aicore__ [aicore]
#endif

#include "../transpose_batch_mat_mul/utils/common.h"
#include "../transpose_batch_mat_mul/utils/iterator.h"
#include "../transpose_batch_mat_mul/utils/mem.h"
#include "../transpose_batch_mat_mul/utils/utils.h"
#include "../transpose_batch_mat_mul/pp_matmul_common.h"
#include "gemm_v3_tiling_data.h"
#include "kernel_tensor.h"

namespace PpMatMulNS {

using AscendC::SetFlag;
using AscendC::WaitFlag;
using AscendC::PipeBarrier;
using AscendC::HardEvent;

template <uint32_t swizzleDirect,
          bool transA,
          bool transB,
          typename InDtype,
          typename OutDtype,
          typename AccumDtype,
          DataFormat formatA = DataFormat::ND,
          DataFormat formatB = DataFormat::ND>
class GemmV3BaseKernel {
    static constexpr uint8_t UNIT_FLAG_MODE_2 = 2;
    static constexpr uint8_t UNIT_FLAG_MODE_3 = 3;
    static constexpr uint8_t NUM_BUFFER = 2;
    static constexpr uint8_t C_NOTIFY_V = 0;
    static constexpr uint8_t V_NOTIFY_C = 2;
    static constexpr uint8_t INTRA_BLOCK_SYNC = 2;
    static constexpr uint32_t L0_PINGPONG_BUFFER_LEN = 16384;
    static constexpr uint32_t L1_PINGPONG_BUFFER_LEN = 131072;
    static constexpr uint32_t CONST_16 = 16;
    static constexpr uint32_t CONST_32 = 32;
    static constexpr uint32_t CONST_256 = 256;
    static constexpr uint32_t VEC_ITER_REPEAT = 128;
    static constexpr uint32_t VEC_ITER_NUMEL = VEC_ITER_REPEAT * 64;

    using OcBuffer = OnChipBuffer<ArchType::ASCEND_V220>;
    template <typename Dtype>
    using CopyGmToUbufAlign = gm_to_ub_align<ArchType::ASCEND_V220, Dtype>;
    using CopyUbufToGmAlign = ub_to_gm_align<ArchType::ASCEND_V220, OutDtype>;
    template <DataFormat srcFormat, DataFormat dstFormat>
    using CopyGmToCbuf = gm_to_l1<ArchType::ASCEND_V220, InDtype, srcFormat, dstFormat>;
    using LoadCbufToCaVec = l1_to_l0_a<ArchType::ASCEND_V220, InDtype, false, DataFormat::VECTOR, DataFormat::VECTOR>;
    using LoadCbufToCa = l1_to_l0_a<ArchType::ASCEND_V220, InDtype, transA, DataFormat::ZN, DataFormat::ZZ>;
    using LoadCbufToCb = l1_to_l0_b<ArchType::ASCEND_V220, InDtype, transB, DataFormat::ZN, DataFormat::NZ>;
    using CopyCcToGm = l0c_to_gm<ArchType::ASCEND_V220, DataFormat::ND, AccumDtype, AccumDtype>;

public:
    __aicore__ explicit GemmV3BaseKernel(){};
    __aicore__ FORCE_INLINE void Init(__gm__ uint8_t* __restrict__ a,
                                      __gm__ uint8_t* __restrict__ b,
                                      __gm__ uint8_t* __restrict__ c,
                                      __gm__ uint8_t* __restrict__ y,
                                      __gm__ uint8_t* __restrict__ workspace,
                                      const GemmV3TilingData& tilingData);
    __aicore__ FORCE_INLINE void GetBlockIdx(const uint64_t index, MatCoord& tidx);
    __aicore__ FORCE_INLINE void RunCube();
    __aicore__ FORCE_INLINE void RunVector();

private:
    __aicore__ FORCE_INLINE void InitBufferCube(const OcBuffer& buf);
    __aicore__ FORCE_INLINE void InitBufferVector(const OcBuffer& buf);

    __aicore__ FORCE_INLINE uint64_t GetOffsetA(const uint64_t batchIdx,
                                                const uint64_t mTileIdx,
                                                const uint64_t kTileIdx)
    {
        if constexpr (formatA == DataFormat::ND) {
            if constexpr (transA) {
                return batchIdx * m_ * k_ + kTileIdx * kTile_ * m_ + mTileIdx * mTile_;
            } else {
                return batchIdx * m_ * k_ + mTileIdx * mTile_ * k_ + kTileIdx * kTile_;
            }
        } else {
            if constexpr (transA) {
                return batchIdx * kAlign_ * mAlign_ + mTileIdx * mTile_ * kAlign_ + kTileIdx * kTile_ * CONST_16;
            } else {
                return batchIdx * mAlign_ * kAlign_ + kTileIdx * kTile_ * mAlign_ + mTileIdx * mTile_ * CONST_16;
            }
        }
    }

    __aicore__ FORCE_INLINE uint64_t GetOffsetB(const uint64_t batchIdx,
                                                const uint64_t kTileIdx,
                                                const uint64_t nTileIdx)
    {
        if constexpr (formatB == DataFormat::ND) {
            if constexpr (transB) {
                return batchIdx * k_ * n_ + nTileIdx * nTile_ * k_ + kTileIdx * kTile_;
            } else {
                return batchIdx * k_ * n_ + kTileIdx * kTile_ * n_ + nTileIdx * nTile_;
            }
        } else {
            if constexpr (transB) {
                return batchIdx * nAlign_ * kAlign_ + kTileIdx * kTile_ * nAlign_ + nTileIdx * nTile_ * CONST_16;
            } else {
                return batchIdx * kAlign_ * nAlign_ + nTileIdx * nTile_ * kAlign_ + kTileIdx * kTile_ * CONST_16;
            }
        }
    }

    __aicore__ FORCE_INLINE void CopyTileA(const AscendC::LocalTensor<InDtype>& dst,
                                           const AscendC::GlobalTensor<InDtype>& src,
                                           const uint64_t mTileActual,
                                           const uint64_t mTileAlign,
                                           const uint64_t kTileActual,
                                           const uint64_t kTileAlign)
    {
        if constexpr (formatA == DataFormat::ND) {
            if ((m_ == 1) || (mTileActual == 1 && !transA)) {
                CopyGmToCbuf<formatA, DataFormat::ND>(
                    dst, src, mTileActual, mTileAlign, m_, kTileActual, kTileAlign, k_);
            } else {
                if constexpr (transA) {
                    CopyGmToCbuf<formatA, DataFormat::NZ>(
                        dst, src, kTileActual, kTileAlign, k_, mTileActual, mTileAlign, m_);
                } else {
                    CopyGmToCbuf<formatA, DataFormat::NZ>(
                        dst, src, mTileActual, mTileAlign, m_, kTileActual, kTileAlign, k_);
                }
            }
        } else {
            if constexpr (transA) {
                CopyGmToCbuf<formatA, DataFormat::NZ>(
                    dst, src, kTileActual, kTileAlign, kAlign_, mTileActual, mTileAlign, mAlign_);
            } else {
                CopyGmToCbuf<formatA, DataFormat::NZ>(
                    dst, src, mTileActual, mTileAlign, mAlign_, kTileActual, kTileAlign, kAlign_);
            }
        }
    }

    __aicore__ FORCE_INLINE void CopyTileB(const AscendC::LocalTensor<InDtype>& dst,
                                           const AscendC::GlobalTensor<InDtype>& src,
                                           const uint64_t kTileActual,
                                           const uint64_t kTileAlign,
                                           const uint64_t nTileActual,
                                           const uint64_t nTileAlign)
    {
        if constexpr (formatB == DataFormat::ND) {
            if constexpr (transB) {
                CopyGmToCbuf<formatB, DataFormat::NZ>(
                    dst, src, nTileActual, nTileAlign, n_, kTileActual, kTileAlign, k_);
            } else {
                CopyGmToCbuf<formatB, DataFormat::NZ>(
                    dst, src, kTileActual, kTileAlign, k_, nTileActual, nTileAlign, n_);
            }
        } else {
            if constexpr (transB) {
                CopyGmToCbuf<formatB, DataFormat::NZ>(
                    dst, src, nTileActual, nTileAlign, nAlign_, kTileActual, kTileAlign, kAlign_);
            } else {
                CopyGmToCbuf<formatB, DataFormat::NZ>(
                    dst, src, kTileActual, kTileAlign, kAlign_, nTileActual, nTileAlign, nAlign_);
            }
        }
    }

private:
    AscendC::GlobalTensor<InDtype> gmA_;
    AscendC::GlobalTensor<InDtype> gmB_;
    AscendC::GlobalTensor<OutDtype> gmC_;
    AscendC::GlobalTensor<OutDtype> gmY_;
    AscendC::GlobalTensor<AccumDtype> gmWorkspace_;
    AscendC::LocalTensor<InDtype> l1BaseA_;
    AscendC::LocalTensor<InDtype> l1BaseB_;
    AscendC::LocalTensor<InDtype> l0BaseA_;
    AscendC::LocalTensor<InDtype> l0BaseB_;
    AscendC::LocalTensor<AccumDtype> l0BaseC_;
    AscendC::LocalTensor<AccumDtype> ubSum_;
    AscendC::LocalTensor<OutDtype> ubC_;
    AscendC::LocalTensor<OutDtype> ubY_;
    AscendC::LocalTensor<AccumDtype> ubProd_;

    uint32_t numCore_{0};
    uint32_t numBatchA_{0};
    uint32_t numBatchB_{0};
    uint32_t m_{0};
    uint32_t k_{0};
    uint32_t n_{0};
    uint32_t mTile_{0};
    uint32_t kTile_{0};
    uint32_t nTile_{0};
    uint32_t mAlign_{0};
    uint32_t kAlign_{0};
    uint32_t nAlign_{0};
    MatCoord tileDim_{0};
    MatCoord fragDim_{0};
    uint32_t coreLoop_{0};
    uint32_t swizzleCount_{1};
    uint32_t coreIdx_{0};
    uint32_t pingPongFlag_{0};
    float alpha_{0.0f};
    float beta_{0.0f};
    bool enShuffleK_{false};
};

template <uint32_t swizzleDirect,
          bool transA,
          bool transB,
          typename InDtype,
          typename OutDtype,
          typename AccumDtype,
          DataFormat formatA,
          DataFormat formatB>
__aicore__ FORCE_INLINE void
GemmV3BaseKernel<swizzleDirect, transA, transB, InDtype, OutDtype, AccumDtype, formatA, formatB>::Init(
    __gm__ uint8_t* __restrict__ a,
    __gm__ uint8_t* __restrict__ b,
    __gm__ uint8_t* __restrict__ c,
    __gm__ uint8_t* __restrict__ y,
    __gm__ uint8_t* __restrict__ workspace,
    const GemmV3TilingData& tilingData)
{
    numBatchA_ = tilingData.numBatchA;
    numBatchB_ = tilingData.numBatchB;
    m_ = tilingData.m;
    k_ = tilingData.k;
    n_ = tilingData.n;
    mTile_ = tilingData.m0;
    kTile_ = tilingData.k0;
    nTile_ = tilingData.n0;
    tileDim_.m = tilingData.mLoop;
    tileDim_.k = tilingData.kLoop;
    tileDim_.n = tilingData.nLoop;
    coreLoop_ = tilingData.coreLoop;
    swizzleCount_ = tilingData.swizzleCount;
    numCore_ = tilingData.blockDim;
    mAlign_ = RoundUp<CONST_16>(m_);
    kAlign_ = RoundUp<CONST_16>(k_);
    nAlign_ = RoundUp<CONST_16>(n_);
    alpha_ = tilingData.alpha;
    beta_ = tilingData.beta;
#ifdef __DAV_C220_CUBE__
    coreIdx_ = AscendC::GetBlockIdx();
#endif
#ifdef __DAV_C220_VEC__
    coreIdx_ = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
#endif
    pingPongFlag_ = 1;
    enShuffleK_ = tilingData.enShuffleK;

    gmA_.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype*>(a));
    gmB_.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype*>(b));
    gmC_.SetGlobalBuffer(reinterpret_cast<__gm__ OutDtype*>(c));
    gmY_.SetGlobalBuffer(reinterpret_cast<__gm__ OutDtype*>(y));
    gmWorkspace_.SetGlobalBuffer(reinterpret_cast<__gm__ AccumDtype*>(workspace) +
                                 coreIdx_ * mTile_ * nTile_ * NUM_BUFFER);

    OcBuffer buf;
    InitBufferCube(buf);
    InitBufferVector(buf);
}

template <uint32_t swizzleDirect,
          bool transA,
          bool transB,
          typename InDtype,
          typename OutDtype,
          typename AccumDtype,
          DataFormat formatA,
          DataFormat formatB>
__aicore__ FORCE_INLINE void
GemmV3BaseKernel<swizzleDirect, transA, transB, InDtype, OutDtype, AccumDtype, formatA, formatB>::GetBlockIdx(
    const uint64_t index, MatCoord& tidx)
{
    uint64_t in_batch_idx = index % (tileDim_.m * tileDim_.n);
    if constexpr (swizzleDirect == 0) { // Zn
        uint64_t tile_block_loop = (tileDim_.m + swizzleCount_ - 1) / swizzleCount_;
        uint64_t tile_block_idx = in_batch_idx / (swizzleCount_ * tileDim_.n);
        uint64_t in_tile_block_idx = in_batch_idx % (swizzleCount_ * tileDim_.n);

        uint64_t n_row = swizzleCount_;
        if (tile_block_idx == tile_block_loop - 1) {
            n_row = tileDim_.m - swizzleCount_ * tile_block_idx;
        }
        tidx.m = tile_block_idx * swizzleCount_ + in_tile_block_idx % n_row;
        tidx.n = in_tile_block_idx / n_row;
        if ((tile_block_idx & 0b1) != 0) {
            tidx.n = tileDim_.n - tidx.n - 1;
        }
    } else if constexpr (swizzleDirect == 1) { // Nz
        uint64_t tile_block_loop = (tileDim_.n + swizzleCount_ - 1) / swizzleCount_;
        uint64_t tile_block_idx = in_batch_idx / (swizzleCount_ * tileDim_.m);
        uint64_t in_tile_block_idx = in_batch_idx % (swizzleCount_ * tileDim_.m);

        uint64_t n_col = swizzleCount_;
        if (tile_block_idx == tile_block_loop - 1) {
            n_col = tileDim_.n - swizzleCount_ * tile_block_idx;
        }
        tidx.m = in_tile_block_idx / n_col;
        tidx.n = tile_block_idx * swizzleCount_ + in_tile_block_idx % n_col;
        if ((tile_block_idx & 0b1) != 0) {
            tidx.m = tileDim_.m - tidx.m - 1;
        }
    }
}

template <uint32_t swizzleDirect,
          bool transA,
          bool transB,
          typename InDtype,
          typename OutDtype,
          typename AccumDtype,
          DataFormat formatA,
          DataFormat formatB>
__aicore__ FORCE_INLINE void
GemmV3BaseKernel<swizzleDirect, transA, transB, InDtype, OutDtype, AccumDtype, formatA, formatB>::RunCube()
{
#ifdef __DAV_C220_CUBE__
    using LocalTensor = AscendC::LocalTensor<InDtype>;

    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID1);

    uint16_t cvPingPongFlag = 0;
    for (uint64_t loopIdx = coreIdx_; loopIdx < coreLoop_; loopIdx += numCore_) {
        uint64_t batchIdx = loopIdx / (tileDim_.m * tileDim_.n);
        uint64_t batchIdxA = numBatchA_ == 1 ? 0 : batchIdx;
        uint64_t batchIdxB = numBatchB_ == 1 ? 0 : batchIdx;
        MatCoord tileIdx{0};
        GetBlockIdx(loopIdx, tileIdx);
        uint64_t shuffleK = enShuffleK_ ? (coreIdx_ % tileDim_.k) : 0;
        uint64_t mTileActual = (tileIdx.m == (tileDim_.m - 1)) ? (m_ - tileIdx.m * mTile_) : mTile_;
        uint64_t nTileActual = (tileIdx.n == (tileDim_.n - 1)) ? (n_ - tileIdx.n * nTile_) : nTile_;
        uint64_t kTileActual = (shuffleK == tileDim_.k - 1) ? k_ - shuffleK * kTile_ : kTile_;
        uint64_t mTileRound = RoundUp<CONST_16>(mTileActual);
        uint64_t nTileRound = RoundUp<CONST_16>(nTileActual);
        uint64_t kTileRound = RoundUp<CONST_16>(kTileActual);

        if (loopIdx == coreIdx_) {
            event_t eventTileA = pingPongFlag_ ? EVENT_ID0 : EVENT_ID1;
            uint64_t offsetA = GetOffsetA(batchIdxA, tileIdx.m, tileIdx.k);
            LocalTensor l1BufA = pingPongFlag_ ? l1BaseA_ : l1BaseA_[L1_PINGPONG_BUFFER_LEN];
            WaitFlag<HardEvent::MTE1_MTE2>(eventTileA);
            CopyTileA(l1BufA, gmA_[offsetA], mTileActual, mTileRound, kTileActual, kTileRound);
            SetFlag<HardEvent::MTE2_MTE1>(eventTileA);

            event_t eventTileB = pingPongFlag_ ? EVENT_ID2 : EVENT_ID3;
            uint64_t offsetB = GetOffsetB(batchIdxB, tileIdx.k, tileIdx.n);
            LocalTensor l1BufB = pingPongFlag_ ? l1BaseB_ : l1BaseB_[L1_PINGPONG_BUFFER_LEN];
            WaitFlag<HardEvent::MTE1_MTE2>(eventTileB);
            CopyTileB(l1BufB, gmB_[offsetB], kTileActual, kTileRound, nTileActual, nTileRound);
            SetFlag<HardEvent::MTE2_MTE1>(eventTileB);
        }

        uint64_t maxTileRound = mTileRound > nTileRound ? mTileRound : nTileRound;
        uint64_t kFrag = L0_PINGPONG_BUFFER_LEN / maxTileRound / CONST_16 * CONST_16;
        for (tileIdx.k = 0; tileIdx.k < tileDim_.k; ++tileIdx.k) {
            if (tileIdx.k < tileDim_.k - 1) {
                uint64_t shuffleK = enShuffleK_ ? (coreIdx_ + tileIdx.k + 1) % tileDim_.k : (tileIdx.k + 1);
                uint64_t kTileActual = (shuffleK == (tileDim_.k - 1)) ? (k_ - shuffleK * kTile_) : kTile_;
                uint64_t kTileRound = RoundUp<CONST_16>(kTileActual);

                event_t eventTileA = (1 - pingPongFlag_) ? EVENT_ID0 : EVENT_ID1;
                uint64_t offsetA = GetOffsetA(batchIdxA, tileIdx.m, shuffleK);
                LocalTensor l1BufA = (1 - pingPongFlag_) ? l1BaseA_ : l1BaseA_[L1_PINGPONG_BUFFER_LEN];
                WaitFlag<HardEvent::MTE1_MTE2>(eventTileA);
                CopyTileA(l1BufA, gmA_[offsetA], mTileActual, mTileRound, kTileActual, kTileRound);
                SetFlag<HardEvent::MTE2_MTE1>(eventTileA);

                event_t eventTileB = (1 - pingPongFlag_) ? EVENT_ID2 : EVENT_ID3;
                uint64_t offsetB = GetOffsetB(batchIdxB, shuffleK, tileIdx.n);
                LocalTensor l1BufB = (1 - pingPongFlag_) ? l1BaseB_ : l1BaseB_[L1_PINGPONG_BUFFER_LEN];
                WaitFlag<HardEvent::MTE1_MTE2>(eventTileB);
                CopyTileB(l1BufB, gmB_[offsetB], kTileActual, kTileRound, nTileActual, nTileRound);
                SetFlag<HardEvent::MTE2_MTE1>(eventTileB);
            }

            if (tileIdx.k == tileDim_.k - 1 && loopIdx + numCore_ < coreLoop_) {
                uint64_t batchIdx = (loopIdx + numCore_) / (tileDim_.m * tileDim_.n);
                uint64_t batchIdxA = numBatchA_ == 1 ? 0 : batchIdx;
                uint64_t batchIdxB = numBatchB_ == 1 ? 0 : batchIdx;
                MatCoord tileIdx{0};
                GetBlockIdx(loopIdx + numCore_, tileIdx);
                uint64_t shuffleK = enShuffleK_ ? (coreIdx_ % tileDim_.k) : 0;
                uint64_t mTileActual = (tileIdx.m == (tileDim_.m - 1)) ? (m_ - tileIdx.m * mTile_) : mTile_;
                uint64_t nTileActual = (tileIdx.n == (tileDim_.n - 1)) ? (n_ - tileIdx.n * nTile_) : nTile_;
                uint64_t kTileActual = (shuffleK == (tileDim_.k - 1)) ? (k_ - shuffleK * kTile_) : kTile_;
                uint64_t mTileRound = RoundUp<CONST_16>(mTileActual);
                uint64_t nTileRound = RoundUp<CONST_16>(nTileActual);
                uint64_t kTileRound = RoundUp<CONST_16>(kTileActual);

                event_t eventTileA = (1 - pingPongFlag_) ? EVENT_ID0 : EVENT_ID1;
                uint64_t offsetA = GetOffsetA(batchIdxA, tileIdx.m, shuffleK);
                LocalTensor l1BufA = (1 - pingPongFlag_) ? l1BaseA_ : l1BaseA_[L1_PINGPONG_BUFFER_LEN];
                WaitFlag<HardEvent::MTE1_MTE2>(eventTileA);
                CopyTileA(l1BufA, gmA_[offsetA], mTileActual, mTileRound, kTileActual, kTileRound);
                SetFlag<HardEvent::MTE2_MTE1>(eventTileA);

                event_t eventTileB = (1 - pingPongFlag_) ? EVENT_ID2 : EVENT_ID3;
                uint64_t offsetB = GetOffsetB(batchIdxB, shuffleK, tileIdx.n);
                LocalTensor l1BufB = (1 - pingPongFlag_) ? l1BaseB_ : l1BaseB_[L1_PINGPONG_BUFFER_LEN];
                WaitFlag<HardEvent::MTE1_MTE2>(eventTileB);
                CopyTileB(l1BufB, gmB_[offsetB], kTileActual, kTileRound, nTileActual, nTileRound);
                SetFlag<HardEvent::MTE2_MTE1>(eventTileB);
            }

            shuffleK = enShuffleK_ ? (tileIdx.k + coreIdx_) % tileDim_.k : tileIdx.k;
            uint64_t kTileActual = (shuffleK == (tileDim_.k - 1)) ? (k_ - shuffleK * kTile_) : kTile_;
            uint64_t kTileRound = RoundUp<CONST_16>(kTileActual);
            fragDim_.k = (kTileActual + kFrag - 1) / kFrag;

            LocalTensor l1BufA = pingPongFlag_ ? l1BaseA_ : l1BaseA_[L1_PINGPONG_BUFFER_LEN];
            LocalTensor l1BufB = pingPongFlag_ ? l1BaseB_ : l1BaseB_[L1_PINGPONG_BUFFER_LEN];
            event_t eventTileA = pingPongFlag_ ? EVENT_ID0 : EVENT_ID1;
            event_t eventTileB = pingPongFlag_ ? EVENT_ID2 : EVENT_ID3;

            MatCoord fragIdx{0};
            for (fragIdx.k = 0; fragIdx.k < fragDim_.k; ++fragIdx.k) {
                uint32_t kFragActual = (fragIdx.k < fragDim_.k - 1) ? kFrag : kTileActual - fragIdx.k * kFrag;
                uint32_t kFragRound = (fragIdx.k < fragDim_.k - 1) ? kFrag : kTileRound - fragIdx.k * kFrag;

                event_t eventFrag = (1ULL - (fragIdx.k & 0b1)) ? EVENT_ID0 : EVENT_ID1;
                LocalTensor l0BufA = l0BaseA_[(fragIdx.k & 0b1) * L0_PINGPONG_BUFFER_LEN];
                LocalTensor l0BufB = l0BaseB_[(fragIdx.k & 0b1) * L0_PINGPONG_BUFFER_LEN];

                // *** load matrix A from L1 to L0A
                if (fragIdx.k == 0) {
                    WaitFlag<HardEvent::MTE2_MTE1>(eventTileA);
                }
                WaitFlag<HardEvent::M_MTE1>(eventFrag);
                if ((m_ == 1) || (mTileActual == 1 && !transA)) {
                    LoadCbufToCaVec(l0BufA,
                                    l1BufA[fragIdx.k * kFrag],
                                    0,                              // mTileCeil
                                    CeilDiv<CONST_256>(kFragRound), // kPartCeil
                                    0,                              // mSrcStride
                                    1,                              // kSrcStride
                                    0,                              // mDstStride
                                    0);                             // kDstStride
                } else {
                    if constexpr (transA) {
                        LoadCbufToCa(l0BufA,                               // l0Tensor
                                     l1BufA[fragIdx.k * kFrag * CONST_16], // l1Tensor
                                     mTileRound,                           // mTileCeil
                                     kFragRound,                           // kPartCeil
                                     kTileRound / CONST_16,                // mSrcStride
                                     1,                                    // kSrcStride
                                     kFragRound / CONST_16,                // mDstStride
                                     1);                                   // kDstStride
                    } else {
                        LoadCbufToCa(l0BufA,                                 // l0Tensor
                                     l1BufA[fragIdx.k * kFrag * mTileRound], // l1Tensor
                                     mTileRound,                             // mTileCeil
                                     kFragRound,                             // kPartCeil
                                     1,                                      // mSrcStride
                                     mTileRound / CONST_16,                  // kSrcStride
                                     kFragRound / CONST_16,                  // mDstStride
                                     1);                                     // kDstStride
                    }
                }
                if (fragIdx.k == fragDim_.k - 1) {
                    SetFlag<HardEvent::MTE1_MTE2>(eventTileA);
                }

                // *** load matrix B from L1 to L0B
                if (fragIdx.k == 0) {
                    WaitFlag<HardEvent::MTE2_MTE1>(eventTileB);
                }
                if constexpr (transB) {
                    LoadCbufToCb(l0BufB,                                 // l0Tensor
                                 l1BufB[fragIdx.k * kFrag * nTileRound], // l1Tensor
                                 nTileRound,                             // nTileCeil
                                 kFragRound,                             // kPartCeil
                                 1,                                      // nSrcStride
                                 nTileRound / CONST_16,                  // kSrcStride
                                 1,                                      // nDstStride
                                 kFragRound / CONST_16);                 // kDstStride
                } else {
                    LoadCbufToCb(l0BufB,                               // l0Tensor
                                 l1BufB[fragIdx.k * kFrag * CONST_16], // l1Tensor
                                 nTileRound,                           // nTileCeil
                                 kFragRound,                           // kPartCeil
                                 kTileRound / CONST_16,                // nSrcStride
                                 1,                                    // kSrcStride
                                 1,                                    // nDstStride
                                 nTileRound / CONST_16);               // kDstStride
                }
                if (fragIdx.k == fragDim_.k - 1) {
                    SetFlag<HardEvent::MTE1_MTE2>(eventTileB);
                }

                SetFlag<HardEvent::MTE1_M>(eventFrag);
                WaitFlag<HardEvent::MTE1_M>(eventFrag);

                bool initC = (tileIdx.k == 0 && fragIdx.k == 0);
                uint8_t unitFlag = (fragIdx.k == (fragDim_.k - 1) && tileIdx.k == (tileDim_.k - 1)) ? UNIT_FLAG_MODE_3
                                                                                                    : UNIT_FLAG_MODE_2;

                if (m_ != 1 && mTileActual == 1 && transA) {
                    AscendC::Mmad(l0BaseC_,                        // C
                                  l0BufA,                          // A
                                  l0BufB,                          // B
                                  AscendC::MmadParams(CONST_16,    // m
                                                      nTileActual, // n
                                                      kFragActual, // k
                                                      unitFlag,    // unitFlag
                                                      false,       // cmatrixSource
                                                      initC));     // cmatrixInitVal
                } else {
                    AscendC::Mmad(l0BaseC_,                        // C
                                  l0BufA,                          // A
                                  l0BufB,                          // B
                                  AscendC::MmadParams(mTileActual, // m
                                                      nTileActual, // n
                                                      kFragActual, // k
                                                      unitFlag,    // unitFlag
                                                      false,       // cmatrixSource
                                                      initC));     // cmatrixInitVal
                }

                PipeBarrier<PIPE_M>();
                SetFlag<HardEvent::M_MTE1>(eventFrag);
            }

            pingPongFlag_ = 1 - pingPongFlag_;
        }

        // copy from L0C to gm
        AscendC::CrossCoreWaitFlag(V_NOTIFY_C + cvPingPongFlag);
        uint64_t offsetWorkspace = static_cast<uint64_t>(mTile_) * nTile_ * cvPingPongFlag;
        CopyCcToGm(gmWorkspace_[offsetWorkspace], // dst
                   l0BaseC_,                      // src
                   mTileActual,                   // mTileActual
                   nTileActual,                   // nTileActual
                   mTileRound,                    // srcStride
                   nTileActual,                   // dstStride
                   UNIT_FLAG_MODE_3);             // unitFlag
        AscendC::CrossCoreSetFlag<INTRA_BLOCK_SYNC, PIPE_FIX>(C_NOTIFY_V + cvPingPongFlag);
        cvPingPongFlag = (cvPingPongFlag + 1) & 0b1;
    }

    WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
#endif
}

template <uint32_t swizzleDirect,
          bool transA,
          bool transB,
          typename InDtype,
          typename OutDtype,
          typename AccumDtype,
          DataFormat formatA,
          DataFormat formatB>
__aicore__ FORCE_INLINE void
GemmV3BaseKernel<swizzleDirect, transA, transB, InDtype, OutDtype, AccumDtype, formatA, formatB>::RunVector()
{
#ifdef __DAV_C220_VEC__
    using namespace AscendC;
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    uint16_t cvPingPongFlag = 0;
    for (uint64_t loopIdx = coreIdx_; loopIdx < coreLoop_; loopIdx += numCore_) {
        uint64_t batchIdx = loopIdx / (tileDim_.m * tileDim_.n);
        MatCoord tileIdx{0};
        GetBlockIdx(loopIdx, tileIdx);
        uint64_t mTileActual = (tileIdx.m == (tileDim_.m - 1)) ? (m_ - tileIdx.m * mTile_) : mTile_;
        uint64_t nTileActual = (tileIdx.n == (tileDim_.n - 1)) ? (n_ - tileIdx.n * nTile_) : nTile_;
        uint64_t nRound = RoundUp<CONST_16>(nTileActual);
        uint32_t prodDstGap = (nRound - nTileActual) * sizeof(AccumDtype) >= CONST_32 ? 1 : 0;
        uint64_t mTileHalf = (mTileActual + 1) / 2;
        uint64_t mTileHalfActual = (AscendC::GetSubBlockIdx() == 0) ? mTileHalf : (mTileActual - mTileHalf);
        uint64_t offsetCy = batchIdx * m_ * n_ + tileIdx.m * mTile_ * n_ + tileIdx.n * nTile_ +
                            AscendC::GetSubBlockIdx() * mTileHalf * n_;
        uint64_t numel = mTileHalfActual * nRound;
        uint64_t count = CeilDiv<VEC_ITER_NUMEL>(numel);
        if (mTileHalfActual != 0) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            CopyGmToUbufAlign<OutDtype>(ubC_,                                  // dst
                                        gmC_[offsetCy],                        // src
                                        0,                                     // sid
                                        mTileHalfActual,                       // nBurst
                                        nTileActual * sizeof(OutDtype),        // lenBurst
                                        0,                                     // leftPaddingNum
                                        0,                                     // rightPaddingNum
                                        (n_ - nTileActual) * sizeof(OutDtype), // srcGap
                                        0);                                    // dstGap
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            for (uint32_t i = 0; i < count; ++i) {
                AscendC::Cast<AccumDtype, OutDtype, false>(ubSum_[i * VEC_ITER_NUMEL],
                                                           ubC_[i * VEC_ITER_NUMEL],
                                                           AscendC::RoundMode::CAST_NONE,
                                                           (uint64_t)0,
                                                           VEC_ITER_REPEAT,
                                                           AscendC::UnaryRepeatParams(1, 1, 8, 4));
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Muls<AccumDtype, false>(ubSum_[i * VEC_ITER_NUMEL], // dst
                                                 ubSum_[i * VEC_ITER_NUMEL], // src
                                                 beta_,                      // scalar
                                                 (uint64_t)0,                // mask (disabled)
                                                 (uint8_t)VEC_ITER_REPEAT,   // repeat
                                                 AscendC::UnaryRepeatParams(1, 1, 8, 8));
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
        AscendC::CrossCoreWaitFlag(C_NOTIFY_V + cvPingPongFlag);
        if (mTileHalfActual != 0) {
            uint64_t offsetWorkspace = static_cast<uint64_t>(mTile_) * nTile_ * cvPingPongFlag +
                                       AscendC::GetSubBlockIdx() * mTileHalf * nTileActual;
            CopyGmToUbufAlign<AccumDtype>(ubProd_,                          // dst
                                          gmWorkspace_[offsetWorkspace],    // src
                                          0,                                // sid
                                          mTileHalfActual,                  // nBurst
                                          nTileActual * sizeof(AccumDtype), // lenBurst
                                          0,                                // leftPaddingNum
                                          0,                                // rightPaddingNum
                                          0,                                // srcGap
                                          prodDstGap);                      // dstGap
            AscendC::CrossCoreSetFlag<INTRA_BLOCK_SYNC, PIPE_MTE2>(V_NOTIFY_C + cvPingPongFlag);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
            for (uint32_t i = 0; i < count; ++i) {
                AscendC::Axpy<AccumDtype, AccumDtype, false>(ubSum_[i * VEC_ITER_NUMEL],  // dst
                                                             ubProd_[i * VEC_ITER_NUMEL], // src0
                                                             alpha_,                      // src1
                                                             (uint64_t)0,                 // mask (disabled)
                                                             VEC_ITER_REPEAT,             // repeat
                                                             AscendC::UnaryRepeatParams(1, 1, 8, 8));
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast<OutDtype, AccumDtype, false>(ubY_[i * VEC_ITER_NUMEL],      // dst
                                                           ubSum_[i * VEC_ITER_NUMEL],    // src
                                                           AscendC::RoundMode::CAST_RINT, // mode
                                                           (uint64_t)0,                   // mask (disabled)
                                                           VEC_ITER_REPEAT,               // repeat
                                                           AscendC::UnaryRepeatParams(1, 1, 4, 8));
                AscendC::PipeBarrier<PIPE_V>();
            }
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            CopyUbufToGmAlign(gmY_[offsetCy],                         // dst
                              ubY_,                                   // src
                              0,                                      // sid
                              mTileHalfActual,                        // nBurst
                              nTileActual * sizeof(OutDtype),         // lenBurst
                              0,                                      // leftPaddingNum
                              0,                                      // rightPaddingNum
                              0,                                      // srcGap
                              (n_ - nTileActual) * sizeof(OutDtype)); // dstGap
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        } else {
            AscendC::CrossCoreSetFlag<INTRA_BLOCK_SYNC, PIPE_MTE2>(V_NOTIFY_C + cvPingPongFlag);
        }
        cvPingPongFlag = (cvPingPongFlag + 1) & 0b1;
    }
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
#endif
}

template <uint32_t swizzleDirect,
          bool transA,
          bool transB,
          typename InDtype,
          typename OutDtype,
          typename AccumDtype,
          DataFormat formatA,
          DataFormat formatB>
__aicore__ FORCE_INLINE void
GemmV3BaseKernel<swizzleDirect, transA, transB, InDtype, OutDtype, AccumDtype, formatA, formatB>::InitBufferCube(
    const OcBuffer& buf)
{
#ifdef __DAV_C220_CUBE__
    l1BaseA_ = buf.template GetBuffer<BufferType::ASCEND_CB, InDtype>(0);
    l1BaseB_ = buf.template GetBuffer<BufferType::ASCEND_CB, InDtype>(RoundUp<256>(mTile_ * kTile_ * sizeof(InDtype)));
    l0BaseA_ = buf.template GetBuffer<BufferType::ASCEND_L0A, InDtype>(0);
    l0BaseB_ = buf.template GetBuffer<BufferType::ASCEND_L0B, InDtype>(0);
#endif
}

template <uint32_t swizzleDirect,
          bool transA,
          bool transB,
          typename InDtype,
          typename OutDtype,
          typename AccumDtype,
          DataFormat formatA,
          DataFormat formatB>
__aicore__ FORCE_INLINE void
GemmV3BaseKernel<swizzleDirect, transA, transB, InDtype, OutDtype, AccumDtype, formatA, formatB>::InitBufferVector(
    const OcBuffer& buf)
{
#ifdef __DAV_C220_VEC__
    uint64_t sizeUbProd = 65536;
    uint64_t sizeUbSum = 65536;
    uint64_t sizeUbC = 32768;
    ubProd_ = buf.template GetBuffer<BufferType::ASCEND_UB, AccumDtype>(0);
    ubSum_ = buf.template GetBuffer<BufferType::ASCEND_UB, AccumDtype>(sizeUbProd);
    ubC_ = buf.template GetBuffer<BufferType::ASCEND_UB, OutDtype>(sizeUbProd + sizeUbSum);
    ubY_ = buf.template GetBuffer<BufferType::ASCEND_UB, OutDtype>(sizeUbProd + sizeUbSum + sizeUbC);
#endif
}
} // namespace PpMatMulNS
#endif // GEMM_V3_BASE_KERNEL_H