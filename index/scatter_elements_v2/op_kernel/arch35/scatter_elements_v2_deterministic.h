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
 * \file scatter_elements_v2_deterministic.h
 * \brief scatter_elements
 */
#ifndef ASCENDC_SCATTER_ELEMENTS_V2_DETERMINISTIC_H_
#define ASCENDC_SCATTER_ELEMENTS_V2_DETERMINISTIC_H_

#include "kernel_operator.h"

#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "scatter_elements_v2_multi.h"
#include "simt_api/device_warp_functions.h"
#include "simt_api/device_sync_functions.h"

namespace ScatterElements {
using namespace AscendC;

constexpr int64_t BLOCK_SIZE = platform::GetUbBlockSize();
constexpr int64_t INDICES_DB_BUFFER = 2;
constexpr uint16_t PATTERN_SA = 0;
constexpr uint16_t PATTERN_AS = 1;
constexpr uint16_t PATTERN_ASA = 2;
constexpr uint32_t USED_THREAD_DETERM = 256;
constexpr uint32_t USED_THREAD_DETERM_WARP = 1024;
constexpr uint32_t DETERM_WARP_SIZE = 32;
constexpr uint32_t WARP_ADD_MIN_PROCESS_S = 32;
constexpr uint32_t PARAM_NUM = 14;
constexpr uint32_t ADAPTIVE_GROUP_SHIFT_SLOT = PARAM_NUM;
constexpr uint32_t ADAPTIVE_SAMPLE_COUNT = 64;
constexpr uint32_t ADAPTIVE_UPDATES_PER_THREAD = 4;
constexpr uint32_t MAX_GROUP_SHIFT = 5;
static_assert(ADAPTIVE_GROUP_SHIFT_SLOT < PARAM_UB_NUM, "adaptive selection requires a reserved parameter slot");
constexpr uint16_t ONE = 1;

template <typename COMP_T>
struct ScatterElementsQuickDivParam {
    COMP_T m0{1};
    COMP_T m1{1};
    COMP_T m2{1};
    COMP_T m3{1};
    COMP_T m4{1};
    COMP_T m5{1};
    COMP_T m6{1};
    COMP_T shift0{1};
    COMP_T shift1{1};
    COMP_T shift2{1};
    COMP_T shift3{1};
    COMP_T shift4{1};
    COMP_T shift5{1};
    COMP_T shift6{1};
};

template <typename DATA_T, typename IDX_T, typename COMP_T, const uint32_t REDU, const uint32_t TEMPLATE_V2 = 0,
          const bool USE_WARP_ADD = false>
class KernelScatterElementsDeterm {
public:
    __aicore__ inline KernelScatterElementsDeterm(const ScatterElementsV2AscTilingData* tiling, TPipe* pipe)
        : tilingData_(tiling), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR indices, GM_ADDR updates, GM_ADDR y);
    __aicore__ inline void CopyToY(int64_t offset, int64_t dataLen);
    __aicore__ inline void CopyDataToY();
    __aicore__ inline void ProcessPatternSA();
    __aicore__ inline void ProcessPatternAS();
    __aicore__ inline void ProcessPatternASA();
    __aicore__ inline void SortAndUpdate();
    __aicore__ inline void EditSimtParam();
    __aicore__ inline void Process();
    template <const uint16_t PATTERN>
    __aicore__ inline void SimtComputeShell(LocalTensor<uint32_t> sortedIdxLocal, LocalTensor<IDX_T> sortedKeyLocal,
                                            uint32_t processS, uint32_t processA0, COMP_T offset);
    __aicore__ inline void SimtComputeShellASA(LocalTensor<uint32_t> sortedIdxLocal, LocalTensor<IDX_T> sortedKeyLocal,
                                               uint32_t processA1, uint32_t processS, uint32_t processA0,
                                               COMP_T offset);

private:
    template <const uint16_t RANK, const uint16_t DIM, const uint16_t PATTERN>
    __aicore__ inline void LaunchSimtCompute(LocalTensor<uint32_t> sortedIdxLocal, LocalTensor<IDX_T> sortedKeyLocal,
                                             uint32_t processA1, uint32_t processS, uint32_t processA0, COMP_T offset);
    template <const uint16_t RANK, const uint16_t DIM, const uint16_t PATTERN, const bool SAME_SHAPE>
    __aicore__ inline void LaunchSimtComputeImpl(LocalTensor<uint32_t> sortedIdxLocal,
                                                 LocalTensor<IDX_T> sortedKeyLocal, uint32_t processA1,
                                                 uint32_t processS, uint32_t processA0, COMP_T offset);

    GlobalTensor<DATA_T> y_;
    GlobalTensor<DATA_T> x_;
    GlobalTensor<DATA_T> updates_;
    GlobalTensor<IDX_T> indices_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, DB_BUFFER> dataQueue_;
    TQue<QuePosition::VECIN, INDICES_DB_BUFFER> indicesQue_;
    TBuf<QuePosition::VECCALC> sortedIdxBuf_;
    TBuf<QuePosition::VECCALC> sortedKeyBuf_;
    TBuf<QuePosition::VECCALC> sharedTmpBuf_;
    TBuf<TPosition::VECCALC> tilingDataUint64Buf_;
    TBuf<TPosition::VECCALC> paramBuf_;
    TPipe* pipe_ = nullptr;
    const ScatterElementsV2AscTilingData* tilingData_;

    int64_t blockIdx_;
    int64_t blockNum_;
    int64_t curCoreAAxis_{0};
    int64_t midAxis_{1};
    int64_t afterAxis_{1};
    uint32_t warpProcessS_{0};
    uint32_t warpProcessA0_{0};
    uint32_t warpMagicS_{1};
    uint32_t warpShiftS_{0};
    uint32_t warpMagicA0_{1};
    uint32_t warpShiftA0_{0};
};

template <typename COMP_T, const uint16_t RANK, const uint16_t DIM, const bool SAME_SHAPE>
__simt_callee__ __aicore__ inline void CalcOffset(COMP_T origIndicesOffset, COMP_T sValue, COMP_T& yOffset,
                                                  COMP_T& updatesOffset, __ubuf__ uint64_t* TilingUint64Ub,
                                                  __ubuf__ COMP_T* params);

template <typename DATA_T, typename IDX_T, typename COMP_T, const uint32_t REDU, const uint16_t RANK,
          const uint16_t DIM, const uint16_t PATTERN, const bool SAME_SHAPE>
__simt_vf__ __aicore__ LAUNCH_BOUND(USED_THREAD_DETERM) inline void SimtCompute(
    __ubuf__ IDX_T* sortedKey, __ubuf__ uint32_t* sortedIdx, __gm__ DATA_T* updates, __gm__ DATA_T* y,
    __ubuf__ uint64_t* TilingUint64Ub, uint32_t processS, uint32_t processA0, COMP_T offset, __ubuf__ COMP_T* params,
    int64_t midAxis, int64_t afterAxis);

template <typename DATA_T, typename IDX_T, typename COMP_T, const uint32_t REDU, const uint16_t RANK,
          const uint16_t DIM, const bool SAME_SHAPE>
__simt_vf__ __aicore__ LAUNCH_BOUND(USED_THREAD_DETERM) inline void SimtComputeASA(
    __ubuf__ IDX_T* sortedKey, __ubuf__ uint32_t* sortedIdx, __gm__ DATA_T* updates, __gm__ DATA_T* y,
    __ubuf__ uint64_t* TilingUint64Ub, uint32_t processA1, uint32_t processS, uint32_t processA0, COMP_T offset,
    __ubuf__ COMP_T* params, int64_t midAxis, int64_t afterAxis);

template <typename DATA_T, typename IDX_T, typename COMP_T, const uint16_t RANK, const uint16_t DIM,
          const uint16_t PATTERN, const bool SAME_SHAPE>
__simt_vf__ __aicore__ LAUNCH_BOUND(USED_THREAD_DETERM_WARP) inline void SimtComputeWarpAdd(
    __ubuf__ IDX_T* sortedKey, __ubuf__ uint32_t* sortedIdx, __gm__ DATA_T* updates, __gm__ DATA_T* y,
    __ubuf__ uint64_t* tilingUint64Ub, uint32_t processA1, uint32_t processS, uint32_t processA0, COMP_T offset,
    __ubuf__ COMP_T* params, int64_t midAxis, int64_t afterAxis, uint32_t magicS, uint32_t shiftS, uint32_t magicA0,
    uint32_t shiftA0);

template <typename DATA_T, typename IDX_T, typename COMP_T, const uint32_t REDU, const uint32_t TEMPLATE_V2,
          const bool USE_WARP_ADD>
template <const uint16_t RANK, const uint16_t DIM, const uint16_t PATTERN>
__aicore__ inline void
KernelScatterElementsDeterm<DATA_T, IDX_T, COMP_T, REDU, TEMPLATE_V2, USE_WARP_ADD>::LaunchSimtCompute(
    LocalTensor<uint32_t> sortedIdxLocal, LocalTensor<IDX_T> sortedKeyLocal, uint32_t processA1, uint32_t processS,
    uint32_t processA0, COMP_T offset)
{
    // Tiling compares the shapes once; each VF has a compile-time address path.
    if (tilingData_->indicesUpdatesSameShape != 0) {
        LaunchSimtComputeImpl<RANK, DIM, PATTERN, true>(sortedIdxLocal, sortedKeyLocal, processA1, processS, processA0,
                                                        offset);
    } else {
        LaunchSimtComputeImpl<RANK, DIM, PATTERN, false>(sortedIdxLocal, sortedKeyLocal, processA1, processS, processA0,
                                                         offset);
    }
}

template <typename DATA_T, typename IDX_T, typename COMP_T, const uint32_t REDU, const uint32_t TEMPLATE_V2,
          const bool USE_WARP_ADD>
template <const uint16_t RANK, const uint16_t DIM, const uint16_t PATTERN, const bool SAME_SHAPE>
__aicore__ inline void
KernelScatterElementsDeterm<DATA_T, IDX_T, COMP_T, REDU, TEMPLATE_V2, USE_WARP_ADD>::LaunchSimtComputeImpl(
    LocalTensor<uint32_t> sortedIdxLocal, LocalTensor<IDX_T> sortedKeyLocal, uint32_t processA1, uint32_t processS,
    uint32_t processA0, COMP_T offset)
{
    LocalTensor<uint64_t> tilingUint64Ub = tilingDataUint64Buf_.Get<uint64_t>();
    LocalTensor<COMP_T> paramUb = paramBuf_.Get<COMP_T>();
    if constexpr (USE_WARP_ADD) {
        // Admission is decided by tiling and dispatched once at the entry.
        // Preserve the original scalar path for short S tails.
        if (processS >= WARP_ADD_MIN_PROCESS_S) {
            // Cache the current divisors, including S and A0 tails, outside the VF.
            if (warpProcessS_ != processS) {
                GetUintDivMagicAndShift(warpMagicS_, warpShiftS_, processS);
                warpProcessS_ = processS;
            }
            if constexpr (PATTERN == PATTERN_ASA) {
                if (warpProcessA0_ != processA0) {
                    GetUintDivMagicAndShift(warpMagicA0_, warpShiftA0_, processA0);
                    warpProcessA0_ = processA0;
                }
            }
            asc_vf_call<SimtComputeWarpAdd<DATA_T, IDX_T, COMP_T, RANK, DIM, PATTERN, SAME_SHAPE>>(
                dim3(USED_THREAD_DETERM_WARP), (__ubuf__ IDX_T*)(sortedKeyLocal.GetPhyAddr()),
                (__ubuf__ uint32_t*)(sortedIdxLocal.GetPhyAddr()), (__gm__ DATA_T*)(updates_.GetPhyAddr()),
                (__gm__ DATA_T*)(y_.GetPhyAddr()), (__ubuf__ uint64_t*)(tilingUint64Ub.GetPhyAddr()), processA1,
                processS, processA0, offset, (__ubuf__ COMP_T*)(paramUb.GetPhyAddr()), midAxis_, afterAxis_,
                warpMagicS_, warpShiftS_, warpMagicA0_, warpShiftA0_);
            return;
        }
    }
    if constexpr (PATTERN == PATTERN_ASA) {
        asc_vf_call<SimtComputeASA<DATA_T, IDX_T, COMP_T, REDU, RANK, DIM, SAME_SHAPE>>(
            dim3(USED_THREAD_DETERM), (__ubuf__ IDX_T*)(sortedKeyLocal.GetPhyAddr()),
            (__ubuf__ uint32_t*)(sortedIdxLocal.GetPhyAddr()), (__gm__ DATA_T*)(updates_.GetPhyAddr()),
            (__gm__ DATA_T*)(y_.GetPhyAddr()), (__ubuf__ uint64_t*)(tilingUint64Ub.GetPhyAddr()), processA1, processS,
            processA0, offset, (__ubuf__ COMP_T*)(paramUb.GetPhyAddr()), midAxis_, afterAxis_);
    } else {
        asc_vf_call<SimtCompute<DATA_T, IDX_T, COMP_T, REDU, RANK, DIM, PATTERN, SAME_SHAPE>>(
            dim3(USED_THREAD_DETERM), (__ubuf__ IDX_T*)(sortedKeyLocal.GetPhyAddr()),
            (__ubuf__ uint32_t*)(sortedIdxLocal.GetPhyAddr()), (__gm__ DATA_T*)(updates_.GetPhyAddr()),
            (__gm__ DATA_T*)(y_.GetPhyAddr()), (__ubuf__ uint64_t*)(tilingUint64Ub.GetPhyAddr()), processS, processA0,
            offset, (__ubuf__ COMP_T*)(paramUb.GetPhyAddr()), midAxis_, afterAxis_);
    }
}

template <typename DATA_T, typename IDX_T, typename COMP_T, const uint32_t REDU, const uint32_t TEMPLATE_V2,
          const bool USE_WARP_ADD>
__aicore__ inline void KernelScatterElementsDeterm<DATA_T, IDX_T, COMP_T, REDU, TEMPLATE_V2, USE_WARP_ADD>::Init(
    GM_ADDR x, GM_ADDR indices, GM_ADDR updates, GM_ADDR y)
{
    x_.SetGlobalBuffer((__gm__ DATA_T*)(x));
    y_.SetGlobalBuffer((__gm__ DATA_T*)(y));
    updates_.SetGlobalBuffer((__gm__ DATA_T*)(updates));
    indices_.SetGlobalBuffer((__gm__ IDX_T*)(indices));

    blockIdx_ = GetBlockIdx();
    blockNum_ = GetBlockNum();

    curCoreAAxis_ = blockIdx_ != (tilingData_->indicesUsedCoreNum - 1) ? tilingData_->indicesNormBlockData :
                                                                         tilingData_->indicesTailBlockData;
    midAxis_ = tilingData_->midAxis;
    afterAxis_ = tilingData_->afterAxis;

    int64_t sortDim = tilingData_->baseS * tilingData_->baseA;
    if constexpr (TEMPLATE_V2 == 0) {
        pipe_->InitBuffer(dataQueue_, DB_BUFFER, tilingData_->loopLength * sizeof(DATA_T));
    }
    pipe_->InitBuffer(indicesQue_, INDICES_DB_BUFFER,
                      ops::Aligned(static_cast<int64_t>(sortDim * sizeof(IDX_T)), BLOCK_SIZE));
    pipe_->InitBuffer(sortedKeyBuf_, ops::Aligned(static_cast<int64_t>(sortDim * sizeof(IDX_T)), BLOCK_SIZE));
    pipe_->InitBuffer(sortedIdxBuf_, ops::Aligned(static_cast<int64_t>(sortDim * sizeof(uint32_t)), BLOCK_SIZE));
    pipe_->InitBuffer(sharedTmpBuf_, ops::Aligned(static_cast<int64_t>(tilingData_->sortSharedBufSize), BLOCK_SIZE));
    pipe_->InitBuffer(tilingDataUint64Buf_, TILING_DATA_UB_NUM * sizeof(uint64_t));
    pipe_->InitBuffer(paramBuf_, PARAM_UB_NUM * sizeof(COMP_T));
}

template <typename DATA_T, typename IDX_T, typename COMP_T, const uint32_t REDU, const uint32_t TEMPLATE_V2,
          const bool USE_WARP_ADD>
__aicore__ inline void KernelScatterElementsDeterm<DATA_T, IDX_T, COMP_T, REDU, TEMPLATE_V2, USE_WARP_ADD>::CopyToY(
    int64_t offset, int64_t dataLen)
{
    DataCopyExtParams copyParams = {static_cast<uint16_t>(1), static_cast<uint32_t>(dataLen * sizeof(DATA_T)),
                                    static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPadExtParams<DATA_T> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                              static_cast<DATA_T>(0)};
    LocalTensor<DATA_T> xLocal = dataQueue_.AllocTensor<DATA_T>();
    DataCopyPad(xLocal, x_[offset], copyParams, padParams);
    dataQueue_.EnQue(xLocal);

    LocalTensor<DATA_T> yLocal = dataQueue_.DeQue<DATA_T>();
    DataCopyPad(y_[offset], yLocal, copyParams);
    dataQueue_.FreeTensor(yLocal);
}

template <typename DATA_T, typename IDX_T, typename COMP_T, const uint32_t REDU, const uint32_t TEMPLATE_V2,
          const bool USE_WARP_ADD>
__aicore__ inline void
KernelScatterElementsDeterm<DATA_T, IDX_T, COMP_T, REDU, TEMPLATE_V2, USE_WARP_ADD>::CopyDataToY()
{
    int64_t normBlockData = ops::CeilDiv(tilingData_->dataAxis, blockNum_);
    int64_t usedCoreNum = ops::CeilDiv(tilingData_->dataAxis, normBlockData);
    int64_t tailBlockData = tilingData_->dataAxis - (usedCoreNum - 1) * normBlockData;
    int64_t curCoreData = blockIdx_ != (usedCoreNum - 1) ? normBlockData : tailBlockData;
    int64_t loopNum = curCoreData / tilingData_->loopLength;
    int64_t tailLoopLength = curCoreData - loopNum * tilingData_->loopLength;

    if (blockIdx_ < usedCoreNum) {
        int64_t offset = 0;
        for (int64_t idx = 0; idx < loopNum; idx++) {
            offset = blockIdx_ * normBlockData + idx * tilingData_->loopLength;
            CopyToY(offset, tilingData_->loopLength);
        }

        if (tailLoopLength > 0) {
            offset = blockIdx_ * normBlockData + loopNum * tilingData_->loopLength;
            CopyToY(offset, tailLoopLength);
        }
    }
}

template <typename DATA_T, typename IDX_T, typename COMP_T, const uint32_t REDU, const uint32_t TEMPLATE_V2,
          const bool USE_WARP_ADD>
template <const uint16_t PATTERN>
__aicore__ inline void KernelScatterElementsDeterm<DATA_T, IDX_T, COMP_T, REDU, TEMPLATE_V2,
                                                   USE_WARP_ADD>::SimtComputeShell(LocalTensor<uint32_t> sortedIdxLocal,
                                                                                   LocalTensor<IDX_T> sortedKeyLocal,
                                                                                   uint32_t processS,
                                                                                   uint32_t processA0, COMP_T offset)
{
    if (tilingData_->rank == DIM_1) {
        LaunchSimtCompute<DIM_1, 0, PATTERN>(sortedIdxLocal, sortedKeyLocal, 1, processS, processA0, offset);
    } else if (tilingData_->rank == DIM_2) {
        if (tilingData_->dim == 0) {
            LaunchSimtCompute<DIM_2, 0, PATTERN>(sortedIdxLocal, sortedKeyLocal, 1, processS, processA0, offset);
        } else {
            LaunchSimtCompute<DIM_2, DIM_1, PATTERN>(sortedIdxLocal, sortedKeyLocal, 1, processS, processA0, offset);
        }
    } else if (tilingData_->rank == DIM_3) {
        if (tilingData_->dim == 0) {
            LaunchSimtCompute<DIM_3, 0, PATTERN>(sortedIdxLocal, sortedKeyLocal, 1, processS, processA0, offset);
        } else if (tilingData_->dim == DIM_2) {
            LaunchSimtCompute<DIM_3, DIM_2, PATTERN>(sortedIdxLocal, sortedKeyLocal, 1, processS, processA0, offset);
        } else {
            LaunchSimtCompute<DIM_3, DIM_1, PATTERN>(sortedIdxLocal, sortedKeyLocal, 1, processS, processA0, offset);
        }
    } else if (tilingData_->rank == DIM_4) {
        if (tilingData_->dim == 0) {
            LaunchSimtCompute<DIM_4, 0, PATTERN>(sortedIdxLocal, sortedKeyLocal, 1, processS, processA0, offset);
        } else if (tilingData_->dim == DIM_3) {
            LaunchSimtCompute<DIM_4, DIM_3, PATTERN>(sortedIdxLocal, sortedKeyLocal, 1, processS, processA0, offset);
        } else if (tilingData_->dim == DIM_1) {
            LaunchSimtCompute<DIM_4, DIM_1, PATTERN>(sortedIdxLocal, sortedKeyLocal, 1, processS, processA0, offset);
        } else {
            LaunchSimtCompute<DIM_4, DIM_2, PATTERN>(sortedIdxLocal, sortedKeyLocal, 1, processS, processA0, offset);
        }
    } else if (tilingData_->rank == DIM_5) {
        if (tilingData_->dim == 0) {
            LaunchSimtCompute<DIM_5, 0, PATTERN>(sortedIdxLocal, sortedKeyLocal, 1, processS, processA0, offset);
        } else if (tilingData_->dim == DIM_4) {
            LaunchSimtCompute<DIM_5, DIM_4, PATTERN>(sortedIdxLocal, sortedKeyLocal, 1, processS, processA0, offset);
        } else if (tilingData_->dim == DIM_1) {
            LaunchSimtCompute<DIM_5, DIM_1, PATTERN>(sortedIdxLocal, sortedKeyLocal, 1, processS, processA0, offset);
        } else if (tilingData_->dim == DIM_2) {
            LaunchSimtCompute<DIM_5, DIM_2, PATTERN>(sortedIdxLocal, sortedKeyLocal, 1, processS, processA0, offset);
        } else {
            LaunchSimtCompute<DIM_5, DIM_3, PATTERN>(sortedIdxLocal, sortedKeyLocal, 1, processS, processA0, offset);
        }
    } else if (tilingData_->rank == DIM_6) {
        if (tilingData_->dim == 0) {
            LaunchSimtCompute<DIM_6, 0, PATTERN>(sortedIdxLocal, sortedKeyLocal, 1, processS, processA0, offset);
        } else if (tilingData_->dim == DIM_5) {
            LaunchSimtCompute<DIM_6, DIM_5, PATTERN>(sortedIdxLocal, sortedKeyLocal, 1, processS, processA0, offset);
        } else if (tilingData_->dim == DIM_1) {
            LaunchSimtCompute<DIM_6, DIM_1, PATTERN>(sortedIdxLocal, sortedKeyLocal, 1, processS, processA0, offset);
        } else if (tilingData_->dim == DIM_2) {
            LaunchSimtCompute<DIM_6, DIM_2, PATTERN>(sortedIdxLocal, sortedKeyLocal, 1, processS, processA0, offset);
        } else if (tilingData_->dim == DIM_3) {
            LaunchSimtCompute<DIM_6, DIM_3, PATTERN>(sortedIdxLocal, sortedKeyLocal, 1, processS, processA0, offset);
        } else {
            LaunchSimtCompute<DIM_6, DIM_4, PATTERN>(sortedIdxLocal, sortedKeyLocal, 1, processS, processA0, offset);
        }
    } else if (tilingData_->rank == DIM_7) {
        if (tilingData_->dim == 0) {
            LaunchSimtCompute<DIM_7, 0, PATTERN>(sortedIdxLocal, sortedKeyLocal, 1, processS, processA0, offset);
        } else if (tilingData_->dim == DIM_6) {
            LaunchSimtCompute<DIM_7, DIM_6, PATTERN>(sortedIdxLocal, sortedKeyLocal, 1, processS, processA0, offset);
        } else if (tilingData_->dim == DIM_1) {
            LaunchSimtCompute<DIM_7, DIM_1, PATTERN>(sortedIdxLocal, sortedKeyLocal, 1, processS, processA0, offset);
        } else if (tilingData_->dim == DIM_2) {
            LaunchSimtCompute<DIM_7, DIM_2, PATTERN>(sortedIdxLocal, sortedKeyLocal, 1, processS, processA0, offset);
        } else if (tilingData_->dim == DIM_3) {
            LaunchSimtCompute<DIM_7, DIM_3, PATTERN>(sortedIdxLocal, sortedKeyLocal, 1, processS, processA0, offset);
        } else if (tilingData_->dim == DIM_4) {
            LaunchSimtCompute<DIM_7, DIM_4, PATTERN>(sortedIdxLocal, sortedKeyLocal, 1, processS, processA0, offset);
        } else {
            LaunchSimtCompute<DIM_7, DIM_5, PATTERN>(sortedIdxLocal, sortedKeyLocal, 1, processS, processA0, offset);
        }
    } else if (tilingData_->rank == DIM_8) {
        if (tilingData_->dim == 0) {
            LaunchSimtCompute<DIM_8, 0, PATTERN>(sortedIdxLocal, sortedKeyLocal, 1, processS, processA0, offset);
        } else if (tilingData_->dim == DIM_7) {
            LaunchSimtCompute<DIM_8, DIM_7, PATTERN>(sortedIdxLocal, sortedKeyLocal, 1, processS, processA0, offset);
        } else if (tilingData_->dim == DIM_1) {
            LaunchSimtCompute<DIM_8, DIM_1, PATTERN>(sortedIdxLocal, sortedKeyLocal, 1, processS, processA0, offset);
        } else if (tilingData_->dim == DIM_2) {
            LaunchSimtCompute<DIM_8, DIM_2, PATTERN>(sortedIdxLocal, sortedKeyLocal, 1, processS, processA0, offset);
        } else if (tilingData_->dim == DIM_3) {
            LaunchSimtCompute<DIM_8, DIM_3, PATTERN>(sortedIdxLocal, sortedKeyLocal, 1, processS, processA0, offset);
        } else if (tilingData_->dim == DIM_4) {
            LaunchSimtCompute<DIM_8, DIM_4, PATTERN>(sortedIdxLocal, sortedKeyLocal, 1, processS, processA0, offset);
        } else if (tilingData_->dim == DIM_5) {
            LaunchSimtCompute<DIM_8, DIM_5, PATTERN>(sortedIdxLocal, sortedKeyLocal, 1, processS, processA0, offset);
        } else {
            LaunchSimtCompute<DIM_8, DIM_6, PATTERN>(sortedIdxLocal, sortedKeyLocal, 1, processS, processA0, offset);
        }
    }
}

template <typename DATA_T, typename IDX_T, typename COMP_T, const uint32_t REDU, const uint32_t TEMPLATE_V2,
          const bool USE_WARP_ADD>
__aicore__ inline void
KernelScatterElementsDeterm<DATA_T, IDX_T, COMP_T, REDU, TEMPLATE_V2, USE_WARP_ADD>::ProcessPatternSA()
{
    // s:tilingData_->midAxis, a0:tilingData_->afterAxis, per core:curCoreAAxis_
    int64_t blockOffset = blockIdx_ * tilingData_->indicesNormBlockData;
    int64_t sLoopNum = ops::CeilDiv(tilingData_->midAxis, tilingData_->baseS);
    int64_t a0LoopNum = ops::CeilDiv(curCoreAAxis_, tilingData_->baseA);
    int64_t sTail = tilingData_->midAxis - (sLoopNum - 1) * tilingData_->baseS;
    int64_t a0Tail = curCoreAAxis_ - (a0LoopNum - 1) * tilingData_->baseA;
    int64_t processA0 = tilingData_->baseA;
    for (int64_t a0Idx = 0; a0Idx < a0LoopNum; a0Idx++) {
        if (a0Idx == (a0LoopNum - 1)) {
            processA0 = a0Tail;
        }
        int64_t processS = tilingData_->baseS;
        for (int64_t sIdx = 0; sIdx < sLoopNum; sIdx++) {
            int64_t offset = blockOffset + sIdx * tilingData_->baseS * tilingData_->afterAxis +
                             a0Idx * tilingData_->baseA;
            if (sIdx == (sLoopNum - 1)) {
                processS = sTail;
            }
            LocalTensor<IDX_T> indicesLocal = indicesQue_.AllocTensor<IDX_T>();
            static constexpr AscendC::NdDmaConfig config = {false};
            AscendC::NdDmaLoopInfo<DIM_2> loopInfo;

            loopInfo.loopSize[0] = processA0;
            loopInfo.loopSize[DIM_1] = processS;
            loopInfo.loopSrcStride[0] = 1;
            loopInfo.loopSrcStride[DIM_1] = tilingData_->afterAxis;
            loopInfo.loopDstStride[0] = processS;
            loopInfo.loopDstStride[DIM_1] = 1;
            loopInfo.loopLpSize[0] = 0;
            loopInfo.loopLpSize[DIM_1] = 0;
            loopInfo.loopRpSize[0] = 0;
            loopInfo.loopRpSize[DIM_1] = 0;

            IDX_T constValue = 0;
            AscendC::NdDmaParams<IDX_T, DIM_2> paramsMain = {loopInfo, constValue};
            AscendC::DataCopy<IDX_T, DIM_2, config>(indicesLocal, indices_[offset], paramsMain);
            indicesQue_.EnQue(indicesLocal);

            indicesLocal = indicesQue_.DeQue<IDX_T>();
            LocalTensor<uint32_t> sortedIdxLocal = sortedIdxBuf_.Get<uint32_t>();
            LocalTensor<IDX_T> sortedKeyLocal = sortedKeyBuf_.Get<IDX_T>();
            LocalTensor<uint8_t> sharedTmpBuffer = sharedTmpBuf_.Get<uint8_t>();
            static constexpr SortConfig sortConfig{SortType::RADIX_SORT, false};
            AscendC::Sort<IDX_T, false, sortConfig>(sortedKeyLocal, sortedIdxLocal, indicesLocal, sharedTmpBuffer,
                                                    static_cast<uint32_t>(processS * processA0));
            indicesQue_.FreeTensor(indicesLocal);
            SimtComputeShell<PATTERN_SA>(sortedIdxLocal, sortedKeyLocal, static_cast<uint32_t>(processS),
                                         static_cast<uint32_t>(processA0), static_cast<COMP_T>(offset));
        }
    }
}

template <typename DATA_T, typename IDX_T, typename COMP_T, const uint32_t REDU, const uint32_t TEMPLATE_V2,
          const bool USE_WARP_ADD>
__aicore__ inline void
KernelScatterElementsDeterm<DATA_T, IDX_T, COMP_T, REDU, TEMPLATE_V2, USE_WARP_ADD>::ProcessPatternAS()
{
    // a0:tilingData_->preAxis, per core:curCoreAAxis_, s:tilingData_->midAxis
    int64_t blockOffset = blockIdx_ * tilingData_->indicesNormBlockData * tilingData_->midAxis;
    int64_t sLoopNum = ops::CeilDiv(tilingData_->midAxis, tilingData_->baseS);
    int64_t a0LoopNum = ops::CeilDiv(curCoreAAxis_, tilingData_->baseA);
    int64_t sTail = tilingData_->midAxis - (sLoopNum - 1) * tilingData_->baseS;
    int64_t a0Tail = curCoreAAxis_ - (a0LoopNum - 1) * tilingData_->baseA;
    int64_t processA0 = tilingData_->baseA;
    for (int64_t a0Idx = 0; a0Idx < a0LoopNum; a0Idx++) {
        if (a0Idx == (a0LoopNum - 1)) {
            processA0 = a0Tail;
        }
        int64_t processS = tilingData_->baseS;
        for (int64_t sIdx = 0; sIdx < sLoopNum; sIdx++) {
            int64_t offset = blockOffset + a0Idx * tilingData_->baseA * tilingData_->midAxis +
                             sIdx * tilingData_->baseS;
            if (sIdx == (sLoopNum - 1)) {
                processS = sTail;
            }
            LocalTensor<IDX_T> indicesLocal = indicesQue_.AllocTensor<IDX_T>();
            DataCopyExtParams copyParams;
            copyParams.blockCount = static_cast<uint16_t>(processA0);
            copyParams.blockLen = static_cast<uint32_t>(processS * sizeof(IDX_T));
            copyParams.srcStride = static_cast<uint32_t>((tilingData_->midAxis - processS) * sizeof(IDX_T));
            copyParams.dstStride = static_cast<uint32_t>(0);
            DataCopyPadExtParams<IDX_T> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                                     static_cast<IDX_T>(0)};
            DataCopyPad<IDX_T, PaddingMode::Compact>(indicesLocal, indices_[offset], copyParams, padParams);
            indicesQue_.EnQue(indicesLocal);

            indicesLocal = indicesQue_.DeQue<IDX_T>();
            LocalTensor<uint32_t> sortedIdxLocal = sortedIdxBuf_.Get<uint32_t>();
            LocalTensor<IDX_T> sortedKeyLocal = sortedKeyBuf_.Get<IDX_T>();
            LocalTensor<uint8_t> sharedTmpBuffer = sharedTmpBuf_.Get<uint8_t>();
            static constexpr SortConfig sortConfig{SortType::RADIX_SORT, false};
            AscendC::Sort<IDX_T, false, sortConfig>(sortedKeyLocal, sortedIdxLocal, indicesLocal, sharedTmpBuffer,
                                                    static_cast<uint32_t>(processS * processA0));
            indicesQue_.FreeTensor(indicesLocal);
            SimtComputeShell<PATTERN_AS>(sortedIdxLocal, sortedKeyLocal, static_cast<uint32_t>(processS),
                                         static_cast<uint32_t>(processA0), static_cast<COMP_T>(offset));
        }
    }
}

template <typename DATA_T, typename IDX_T, typename COMP_T, const uint32_t REDU, const uint32_t TEMPLATE_V2,
          const bool USE_WARP_ADD>
__aicore__ inline void
KernelScatterElementsDeterm<DATA_T, IDX_T, COMP_T, REDU, TEMPLATE_V2, USE_WARP_ADD>::SimtComputeShellASA(
    LocalTensor<uint32_t> sortedIdxLocal, LocalTensor<IDX_T> sortedKeyLocal, uint32_t processA1, uint32_t processS,
    uint32_t processA0, COMP_T offset)
{
    if (tilingData_->rank == DIM_3) {
        LaunchSimtCompute<DIM_3, DIM_1, PATTERN_ASA>(sortedIdxLocal, sortedKeyLocal, processA1, processS, processA0,
                                                     offset);
    } else if (tilingData_->rank == DIM_4) {
        if (tilingData_->dim == DIM_1) {
            LaunchSimtCompute<DIM_4, DIM_1, PATTERN_ASA>(sortedIdxLocal, sortedKeyLocal, processA1, processS, processA0,
                                                         offset);
        } else {
            LaunchSimtCompute<DIM_4, DIM_2, PATTERN_ASA>(sortedIdxLocal, sortedKeyLocal, processA1, processS, processA0,
                                                         offset);
        }
    } else if (tilingData_->rank == DIM_5) {
        if (tilingData_->dim == DIM_1) {
            LaunchSimtCompute<DIM_5, DIM_1, PATTERN_ASA>(sortedIdxLocal, sortedKeyLocal, processA1, processS, processA0,
                                                         offset);
        } else if (tilingData_->dim == DIM_2) {
            LaunchSimtCompute<DIM_5, DIM_2, PATTERN_ASA>(sortedIdxLocal, sortedKeyLocal, processA1, processS, processA0,
                                                         offset);
        } else {
            LaunchSimtCompute<DIM_5, DIM_3, PATTERN_ASA>(sortedIdxLocal, sortedKeyLocal, processA1, processS, processA0,
                                                         offset);
        }
    } else if (tilingData_->rank == DIM_6) {
        if (tilingData_->dim == DIM_1) {
            LaunchSimtCompute<DIM_6, DIM_1, PATTERN_ASA>(sortedIdxLocal, sortedKeyLocal, processA1, processS, processA0,
                                                         offset);
        } else if (tilingData_->dim == DIM_2) {
            LaunchSimtCompute<DIM_6, DIM_2, PATTERN_ASA>(sortedIdxLocal, sortedKeyLocal, processA1, processS, processA0,
                                                         offset);
        } else if (tilingData_->dim == DIM_3) {
            LaunchSimtCompute<DIM_6, DIM_3, PATTERN_ASA>(sortedIdxLocal, sortedKeyLocal, processA1, processS, processA0,
                                                         offset);
        } else {
            LaunchSimtCompute<DIM_6, DIM_4, PATTERN_ASA>(sortedIdxLocal, sortedKeyLocal, processA1, processS, processA0,
                                                         offset);
        }
    } else if (tilingData_->rank == DIM_7) {
        if (tilingData_->dim == DIM_1) {
            LaunchSimtCompute<DIM_7, DIM_1, PATTERN_ASA>(sortedIdxLocal, sortedKeyLocal, processA1, processS, processA0,
                                                         offset);
        } else if (tilingData_->dim == DIM_2) {
            LaunchSimtCompute<DIM_7, DIM_2, PATTERN_ASA>(sortedIdxLocal, sortedKeyLocal, processA1, processS, processA0,
                                                         offset);
        } else if (tilingData_->dim == DIM_3) {
            LaunchSimtCompute<DIM_7, DIM_3, PATTERN_ASA>(sortedIdxLocal, sortedKeyLocal, processA1, processS, processA0,
                                                         offset);
        } else if (tilingData_->dim == DIM_4) {
            LaunchSimtCompute<DIM_7, DIM_4, PATTERN_ASA>(sortedIdxLocal, sortedKeyLocal, processA1, processS, processA0,
                                                         offset);
        } else {
            LaunchSimtCompute<DIM_7, DIM_5, PATTERN_ASA>(sortedIdxLocal, sortedKeyLocal, processA1, processS, processA0,
                                                         offset);
        }
    } else if (tilingData_->rank == DIM_8) {
        if (tilingData_->dim == DIM_1) {
            LaunchSimtCompute<DIM_8, DIM_1, PATTERN_ASA>(sortedIdxLocal, sortedKeyLocal, processA1, processS, processA0,
                                                         offset);
        } else if (tilingData_->dim == DIM_2) {
            LaunchSimtCompute<DIM_8, DIM_2, PATTERN_ASA>(sortedIdxLocal, sortedKeyLocal, processA1, processS, processA0,
                                                         offset);
        } else if (tilingData_->dim == DIM_3) {
            LaunchSimtCompute<DIM_8, DIM_3, PATTERN_ASA>(sortedIdxLocal, sortedKeyLocal, processA1, processS, processA0,
                                                         offset);
        } else if (tilingData_->dim == DIM_4) {
            LaunchSimtCompute<DIM_8, DIM_4, PATTERN_ASA>(sortedIdxLocal, sortedKeyLocal, processA1, processS, processA0,
                                                         offset);
        } else if (tilingData_->dim == DIM_5) {
            LaunchSimtCompute<DIM_8, DIM_5, PATTERN_ASA>(sortedIdxLocal, sortedKeyLocal, processA1, processS, processA0,
                                                         offset);
        } else {
            LaunchSimtCompute<DIM_8, DIM_6, PATTERN_ASA>(sortedIdxLocal, sortedKeyLocal, processA1, processS, processA0,
                                                         offset);
        }
    }
}

template <typename DATA_T, typename IDX_T, typename COMP_T, const uint32_t REDU, const uint32_t TEMPLATE_V2,
          const bool USE_WARP_ADD>
__aicore__ inline void
KernelScatterElementsDeterm<DATA_T, IDX_T, COMP_T, REDU, TEMPLATE_V2, USE_WARP_ADD>::EditSimtParam()
{
    ScatterElementsQuickDivParam<COMP_T> params;
    static_assert(sizeof(params) == PARAM_NUM * sizeof(COMP_T), "quick-div parameter layout changed");
    LocalTensor<uint64_t> TilingUint64Ub = tilingDataUint64Buf_.Get<uint64_t>();
    LocalTensor<COMP_T> ParamUb = paramBuf_.Get<COMP_T>();
    const uint64_t* tilingUint64 = reinterpret_cast<const uint64_t*>(tilingData_);
    for (uint32_t i = 0; i < TILING_DATA_UINT64_NUM; i++) {
        TilingUint64Ub.SetValue(i, tilingUint64[i]);
    }
    if (tilingData_->rank == DIM_2) {
        GetUintDivMagicAndShift(params.m0, params.shift0, static_cast<COMP_T>(tilingData_->indicesStride[0]));
    } else if (tilingData_->rank == DIM_3) {
        GetUintDivMagicAndShift(params.m0, params.shift0, static_cast<COMP_T>(tilingData_->indicesStride[0]));
        GetUintDivMagicAndShift(params.m1, params.shift1, static_cast<COMP_T>(tilingData_->indicesStride[DIM_1]));
    } else if (tilingData_->rank == DIM_4) {
        GetUintDivMagicAndShift(params.m0, params.shift0, static_cast<COMP_T>(tilingData_->indicesStride[0]));
        GetUintDivMagicAndShift(params.m1, params.shift1, static_cast<COMP_T>(tilingData_->indicesStride[DIM_1]));
        GetUintDivMagicAndShift(params.m2, params.shift2, static_cast<COMP_T>(tilingData_->indicesStride[DIM_2]));
    } else if (tilingData_->rank == DIM_5) {
        GetUintDivMagicAndShift(params.m0, params.shift0, static_cast<COMP_T>(tilingData_->indicesStride[0]));
        GetUintDivMagicAndShift(params.m1, params.shift1, static_cast<COMP_T>(tilingData_->indicesStride[DIM_1]));
        GetUintDivMagicAndShift(params.m2, params.shift2, static_cast<COMP_T>(tilingData_->indicesStride[DIM_2]));
        GetUintDivMagicAndShift(params.m3, params.shift3, static_cast<COMP_T>(tilingData_->indicesStride[DIM_3]));
    } else if (tilingData_->rank == DIM_6) {
        GetUintDivMagicAndShift(params.m0, params.shift0, static_cast<COMP_T>(tilingData_->indicesStride[0]));
        GetUintDivMagicAndShift(params.m1, params.shift1, static_cast<COMP_T>(tilingData_->indicesStride[DIM_1]));
        GetUintDivMagicAndShift(params.m2, params.shift2, static_cast<COMP_T>(tilingData_->indicesStride[DIM_2]));
        GetUintDivMagicAndShift(params.m3, params.shift3, static_cast<COMP_T>(tilingData_->indicesStride[DIM_3]));
        GetUintDivMagicAndShift(params.m4, params.shift4, static_cast<COMP_T>(tilingData_->indicesStride[DIM_4]));
    } else if (tilingData_->rank == DIM_7) {
        GetUintDivMagicAndShift(params.m0, params.shift0, static_cast<COMP_T>(tilingData_->indicesStride[0]));
        GetUintDivMagicAndShift(params.m1, params.shift1, static_cast<COMP_T>(tilingData_->indicesStride[DIM_1]));
        GetUintDivMagicAndShift(params.m2, params.shift2, static_cast<COMP_T>(tilingData_->indicesStride[DIM_2]));
        GetUintDivMagicAndShift(params.m3, params.shift3, static_cast<COMP_T>(tilingData_->indicesStride[DIM_3]));
        GetUintDivMagicAndShift(params.m4, params.shift4, static_cast<COMP_T>(tilingData_->indicesStride[DIM_4]));
        GetUintDivMagicAndShift(params.m5, params.shift5, static_cast<COMP_T>(tilingData_->indicesStride[DIM_5]));
    } else if (tilingData_->rank == DIM_8) {
        GetUintDivMagicAndShift(params.m0, params.shift0, static_cast<COMP_T>(tilingData_->indicesStride[0]));
        GetUintDivMagicAndShift(params.m1, params.shift1, static_cast<COMP_T>(tilingData_->indicesStride[DIM_1]));
        GetUintDivMagicAndShift(params.m2, params.shift2, static_cast<COMP_T>(tilingData_->indicesStride[DIM_2]));
        GetUintDivMagicAndShift(params.m3, params.shift3, static_cast<COMP_T>(tilingData_->indicesStride[DIM_3]));
        GetUintDivMagicAndShift(params.m4, params.shift4, static_cast<COMP_T>(tilingData_->indicesStride[DIM_4]));
        GetUintDivMagicAndShift(params.m5, params.shift5, static_cast<COMP_T>(tilingData_->indicesStride[DIM_5]));
        GetUintDivMagicAndShift(params.m6, params.shift6, static_cast<COMP_T>(tilingData_->indicesStride[DIM_6]));
    }
    const COMP_T* simtParams = reinterpret_cast<const COMP_T*>(&params);
    for (uint32_t i = 0; i < PARAM_NUM; i++) {
        ParamUb.SetValue(i, simtParams[i]);
    }
}

template <typename DATA_T, typename IDX_T, typename COMP_T, const uint32_t REDU, const uint32_t TEMPLATE_V2,
          const bool USE_WARP_ADD>
__aicore__ inline void
KernelScatterElementsDeterm<DATA_T, IDX_T, COMP_T, REDU, TEMPLATE_V2, USE_WARP_ADD>::ProcessPatternASA()
{
    //  a1:tilingData_->preAxis, s:tilingData_->midAxis, a0:tilingData_->afterAxis, per core:curCoreAAxis_,
    // split a0
    int64_t a0Total = curCoreAAxis_;
    int64_t a1Total = tilingData_->preAxis;
    int64_t blockOffset = blockIdx_ * tilingData_->indicesNormBlockData;
    if (tilingData_->preAxis > tilingData_->afterAxis) { // split a1
        a0Total = tilingData_->afterAxis;
        a1Total = curCoreAAxis_;
        blockOffset = blockIdx_ * tilingData_->indicesNormBlockData * tilingData_->midAxis * tilingData_->afterAxis;
    }
    int64_t a0Factor = tilingData_->baseA > a0Total ? a0Total : tilingData_->baseA;
    int64_t a1Factor = tilingData_->baseA / a0Factor;
    int64_t sLoopNum = ops::CeilDiv(tilingData_->midAxis, tilingData_->baseS);
    int64_t a0LoopNum = ops::CeilDiv(a0Total, a0Factor);
    int64_t a1LoopNum = ops::CeilDiv(a1Total, a1Factor);
    int64_t sTail = tilingData_->midAxis - (sLoopNum - 1) * tilingData_->baseS;
    int64_t a0Tail = a0Total - (a0LoopNum - 1) * a0Factor;
    int64_t a1Tail = a1Total - (a1LoopNum - 1) * a1Factor;
    int64_t processA1 = a1Factor;
    for (int64_t a1Idx = 0; a1Idx < a1LoopNum; a1Idx++) {
        if (a1Idx == (a1LoopNum - 1)) {
            processA1 = a1Tail;
        }
        int64_t processA0 = a0Factor;
        for (int64_t a0Idx = 0; a0Idx < a0LoopNum; a0Idx++) {
            if (a0Idx == (a0LoopNum - 1)) {
                processA0 = a0Tail;
            }
            int64_t processS = tilingData_->baseS;
            for (int64_t sIdx = 0; sIdx < sLoopNum; sIdx++) {
                int64_t offset = blockOffset + a1Idx * a1Factor * tilingData_->midAxis * tilingData_->afterAxis +
                                 sIdx * tilingData_->baseS * tilingData_->afterAxis + a0Idx * a0Factor;
                if (sIdx == (sLoopNum - 1)) {
                    processS = sTail;
                }
                LocalTensor<IDX_T> indicesLocal = indicesQue_.AllocTensor<IDX_T>();
                static constexpr AscendC::NdDmaConfig config = {false};
                AscendC::NdDmaLoopInfo<DIM_3> loopInfo;

                loopInfo.loopSize[0] = processA1;
                loopInfo.loopSize[DIM_1] = processA0;
                loopInfo.loopSize[DIM_2] = processS;
                loopInfo.loopSrcStride[0] = tilingData_->midAxis * tilingData_->afterAxis;
                loopInfo.loopSrcStride[DIM_1] = 1;
                loopInfo.loopSrcStride[DIM_2] = tilingData_->afterAxis;
                loopInfo.loopDstStride[0] = processS * processA0;
                loopInfo.loopDstStride[DIM_1] = processS;
                loopInfo.loopDstStride[DIM_2] = 1;
                loopInfo.loopLpSize[0] = 0;
                loopInfo.loopLpSize[DIM_1] = 0;
                loopInfo.loopLpSize[DIM_2] = 0;
                loopInfo.loopRpSize[0] = 0;
                loopInfo.loopRpSize[DIM_1] = 0;
                loopInfo.loopRpSize[DIM_2] = 0;

                IDX_T constValue = 0;
                AscendC::NdDmaParams<IDX_T, DIM_3> paramsMain = {loopInfo, constValue};
                AscendC::DataCopy<IDX_T, DIM_3, config>(indicesLocal, indices_[offset], paramsMain);
                indicesQue_.EnQue(indicesLocal);

                indicesLocal = indicesQue_.DeQue<IDX_T>();
                LocalTensor<uint32_t> sortedIdxLocal = sortedIdxBuf_.Get<uint32_t>();
                LocalTensor<IDX_T> sortedKeyLocal = sortedKeyBuf_.Get<IDX_T>();
                LocalTensor<uint8_t> sharedTmpBuffer = sharedTmpBuf_.Get<uint8_t>();
                static constexpr SortConfig sortConfig{SortType::RADIX_SORT, false};
                AscendC::Sort<IDX_T, false, sortConfig>(sortedKeyLocal, sortedIdxLocal, indicesLocal, sharedTmpBuffer,
                                                        static_cast<uint32_t>(processA1 * processS * processA0));
                indicesQue_.FreeTensor(indicesLocal);
                SimtComputeShellASA(sortedIdxLocal, sortedKeyLocal, static_cast<uint32_t>(processA1),
                                    static_cast<uint32_t>(processS), static_cast<uint32_t>(processA0),
                                    static_cast<COMP_T>(offset));
            }
        }
    }
}

template <typename COMP_T, const uint16_t RANK, const uint16_t DIM, const bool SAME_SHAPE>
__simt_callee__ __aicore__ inline void CalcOffset(COMP_T origIndicesOffset, COMP_T sValue, COMP_T& yOffset,
                                                  COMP_T& updatesOffset, __ubuf__ uint64_t* TilingUint64Ub,
                                                  __ubuf__ COMP_T* params)
{
    uint64_t dataStride[TILING_ARRAY_LEN] = {};
    uint64_t indicesStride[TILING_ARRAY_LEN] = {};
    uint64_t updatesStride[TILING_ARRAY_LEN] = {};
    for (uint32_t i = 0; i < TILING_DATA_UINT64_NUM; i++) {
        if (i < TILING_ARRAY_LEN) {
            dataStride[i] = TilingUint64Ub[i];
        } else if (i < TWO_TILING_ARRAY_LEN) {
            indicesStride[i - TILING_ARRAY_LEN] = TilingUint64Ub[i];
        } else if (i < THREE_TILING_ARRAY_LEN) {
            if constexpr (!SAME_SHAPE) {
                updatesStride[i - TWO_TILING_ARRAY_LEN] = TilingUint64Ub[i];
            }
        }
    }

    COMP_T m0 = params[0];
    COMP_T m1 = params[1];
    COMP_T m2 = params[2];
    COMP_T m3 = params[3];
    COMP_T m4 = params[4];
    COMP_T m5 = params[5];
    COMP_T m6 = params[6];
    COMP_T shift0 = params[7];
    COMP_T shift1 = params[8];
    COMP_T shift2 = params[9];
    COMP_T shift3 = params[10];
    COMP_T shift4 = params[11];
    COMP_T shift5 = params[12];
    COMP_T shift6 = params[13];

    if constexpr (RANK == DIM_1) {
        yOffset = sValue;
        updatesOffset = origIndicesOffset;
    } else if constexpr (RANK == DIM_2) {
        COMP_T dim0Idx = Simt::UintDiv(origIndicesOffset, m0, shift0);
        COMP_T dim1Idx = origIndicesOffset - dim0Idx * indicesStride[0];
        if constexpr (DIM == 0) {
            yOffset = sValue * dataStride[0] + dim1Idx;
        } else {
            yOffset = dim0Idx * dataStride[0] + sValue;
        }
        if constexpr (!SAME_SHAPE) {
            updatesOffset = dim0Idx * updatesStride[0] + dim1Idx;
        }
    } else if constexpr (RANK == DIM_3) {
        COMP_T dim0Idx = Simt::UintDiv(origIndicesOffset, m0, shift0);
        COMP_T dim0Rem = origIndicesOffset - dim0Idx * indicesStride[0];

        COMP_T dim1Idx = Simt::UintDiv(dim0Rem, m1, shift1);
        COMP_T dim2Idx = dim0Rem - dim1Idx * indicesStride[DIM_1];
        if constexpr (DIM == 0) {
            yOffset = sValue * dataStride[0] + dim1Idx * dataStride[DIM_1] + dim2Idx;
        } else if constexpr (DIM == 1) {
            yOffset = dim0Idx * dataStride[0] + sValue * dataStride[DIM_1] + dim2Idx;
        } else {
            yOffset = dim0Idx * dataStride[0] + dim1Idx * dataStride[DIM_1] + sValue;
        }
        if constexpr (!SAME_SHAPE) {
            updatesOffset = dim0Idx * updatesStride[0] + dim1Idx * updatesStride[DIM_1] + dim2Idx;
        }
    } else if constexpr (RANK == DIM_4) {
        COMP_T dim0Idx = Simt::UintDiv(origIndicesOffset, m0, shift0);
        COMP_T dim0Rem = origIndicesOffset - dim0Idx * indicesStride[0];

        COMP_T dim1Idx = Simt::UintDiv(dim0Rem, m1, shift1);
        COMP_T dim1Rem = dim0Rem - dim1Idx * indicesStride[DIM_1];

        COMP_T dim2Idx = Simt::UintDiv(dim1Rem, m2, shift2);
        COMP_T dim3Idx = dim1Rem - dim2Idx * indicesStride[DIM_2];
        if constexpr (DIM == 0) {
            yOffset = sValue * dataStride[0] + dim1Idx * dataStride[DIM_1] + dim2Idx * dataStride[DIM_2] + dim3Idx;
        } else if constexpr (DIM == 1) {
            yOffset = dim0Idx * dataStride[0] + sValue * dataStride[DIM_1] + dim2Idx * dataStride[DIM_2] + dim3Idx;
        } else if constexpr (DIM == DIM_2) {
            yOffset = dim0Idx * dataStride[0] + dim1Idx * dataStride[DIM_1] + sValue * dataStride[DIM_2] + dim3Idx;
        } else {
            yOffset = dim0Idx * dataStride[0] + dim1Idx * dataStride[DIM_1] + dim2Idx * dataStride[DIM_2] + sValue;
        }
        if constexpr (!SAME_SHAPE) {
            updatesOffset = dim0Idx * updatesStride[0] + dim1Idx * updatesStride[DIM_1] +
                            dim2Idx * updatesStride[DIM_2] + dim3Idx;
        }
    } else if constexpr (RANK == DIM_5) {
        COMP_T dim0Idx = Simt::UintDiv(origIndicesOffset, m0, shift0);
        COMP_T dim0Rem = origIndicesOffset - dim0Idx * indicesStride[0];

        COMP_T dim1Idx = Simt::UintDiv(dim0Rem, m1, shift1);
        COMP_T dim1Rem = dim0Rem - dim1Idx * indicesStride[DIM_1];

        COMP_T dim2Idx = Simt::UintDiv(dim1Rem, m2, shift2);
        COMP_T dim2Rem = dim1Rem - dim2Idx * indicesStride[DIM_2];

        COMP_T dim3Idx = Simt::UintDiv(dim2Rem, m3, shift3);
        COMP_T dim4Idx = dim2Rem - dim3Idx * indicesStride[DIM_3];
        if constexpr (DIM == 0) {
            yOffset = sValue * dataStride[0] + dim1Idx * dataStride[DIM_1] + dim2Idx * dataStride[DIM_2] +
                      dim3Idx * dataStride[DIM_3] + dim4Idx;
        } else if constexpr (DIM == 1) {
            yOffset = dim0Idx * dataStride[0] + sValue * dataStride[DIM_1] + dim2Idx * dataStride[DIM_2] +
                      dim3Idx * dataStride[DIM_3] + dim4Idx;
        } else if constexpr (DIM == DIM_2) {
            yOffset = dim0Idx * dataStride[0] + dim1Idx * dataStride[DIM_1] + sValue * dataStride[DIM_2] +
                      dim3Idx * dataStride[DIM_3] + dim4Idx;
        } else if constexpr (DIM == DIM_3) {
            yOffset = dim0Idx * dataStride[0] + dim1Idx * dataStride[DIM_1] + dim2Idx * dataStride[DIM_2] +
                      sValue * dataStride[DIM_3] + dim4Idx;
        } else {
            yOffset = dim0Idx * dataStride[0] + dim1Idx * dataStride[DIM_1] + dim2Idx * dataStride[DIM_2] +
                      dim3Idx * dataStride[DIM_3] + sValue;
        }
        if constexpr (!SAME_SHAPE) {
            updatesOffset = dim0Idx * updatesStride[0] + dim1Idx * updatesStride[DIM_1] +
                            dim2Idx * updatesStride[DIM_2] + dim3Idx * updatesStride[DIM_3] + dim4Idx;
        }
    } else if constexpr (RANK == DIM_6) {
        COMP_T dim0Idx = Simt::UintDiv(origIndicesOffset, m0, shift0);
        COMP_T dim0Rem = origIndicesOffset - dim0Idx * indicesStride[0];

        COMP_T dim1Idx = Simt::UintDiv(dim0Rem, m1, shift1);
        COMP_T dim1Rem = dim0Rem - dim1Idx * indicesStride[DIM_1];

        COMP_T dim2Idx = Simt::UintDiv(dim1Rem, m2, shift2);
        COMP_T dim2Rem = dim1Rem - dim2Idx * indicesStride[DIM_2];

        COMP_T dim3Idx = Simt::UintDiv(dim2Rem, m3, shift3);
        COMP_T dim3Rem = dim2Rem - dim3Idx * indicesStride[DIM_3];

        COMP_T dim4Idx = Simt::UintDiv(dim3Rem, m4, shift4);
        COMP_T dim5Idx = dim3Rem - dim4Idx * indicesStride[DIM_4];

        if constexpr (DIM == 0) {
            yOffset = sValue * dataStride[0] + dim1Idx * dataStride[DIM_1] + dim2Idx * dataStride[DIM_2] +
                      dim3Idx * dataStride[DIM_3] + dim4Idx * dataStride[DIM_4] + dim5Idx;
        } else if constexpr (DIM == 1) {
            yOffset = dim0Idx * dataStride[0] + sValue * dataStride[DIM_1] + dim2Idx * dataStride[DIM_2] +
                      dim3Idx * dataStride[DIM_3] + dim4Idx * dataStride[DIM_4] + dim5Idx;
        } else if constexpr (DIM == DIM_2) {
            yOffset = dim0Idx * dataStride[0] + dim1Idx * dataStride[DIM_1] + sValue * dataStride[DIM_2] +
                      dim3Idx * dataStride[DIM_3] + dim4Idx * dataStride[DIM_4] + dim5Idx;
        } else if constexpr (DIM == DIM_3) {
            yOffset = dim0Idx * dataStride[0] + dim1Idx * dataStride[DIM_1] + dim2Idx * dataStride[DIM_2] +
                      sValue * dataStride[DIM_3] + dim4Idx * dataStride[DIM_4] + dim5Idx;
        } else if constexpr (DIM == DIM_4) {
            yOffset = dim0Idx * dataStride[0] + dim1Idx * dataStride[DIM_1] + dim2Idx * dataStride[DIM_2] +
                      dim3Idx * dataStride[DIM_3] + sValue * dataStride[DIM_4] + dim5Idx;
        } else {
            yOffset = dim0Idx * dataStride[0] + dim1Idx * dataStride[DIM_1] + dim2Idx * dataStride[DIM_2] +
                      dim3Idx * dataStride[DIM_3] + dim4Idx * dataStride[DIM_4] + sValue;
        }
        if constexpr (!SAME_SHAPE) {
            updatesOffset = dim0Idx * updatesStride[0] + dim1Idx * updatesStride[DIM_1] +
                            dim2Idx * updatesStride[DIM_2] + dim3Idx * updatesStride[DIM_3] +
                            dim4Idx * updatesStride[DIM_4] + dim5Idx;
        }
    } else if constexpr (RANK == DIM_7) {
        COMP_T dim0Idx = Simt::UintDiv(origIndicesOffset, m0, shift0);
        COMP_T dim0Rem = origIndicesOffset - dim0Idx * indicesStride[0];

        COMP_T dim1Idx = Simt::UintDiv(dim0Rem, m1, shift1);
        COMP_T dim1Rem = dim0Rem - dim1Idx * indicesStride[DIM_1];

        COMP_T dim2Idx = Simt::UintDiv(dim1Rem, m2, shift2);
        COMP_T dim2Rem = dim1Rem - dim2Idx * indicesStride[DIM_2];

        COMP_T dim3Idx = Simt::UintDiv(dim2Rem, m3, shift3);
        COMP_T dim3Rem = dim2Rem - dim3Idx * indicesStride[DIM_3];

        COMP_T dim4Idx = Simt::UintDiv(dim3Rem, m4, shift4);
        COMP_T dim4Rem = dim3Rem - dim4Idx * indicesStride[DIM_4];

        COMP_T dim5Idx = Simt::UintDiv(dim4Rem, m5, shift5);
        COMP_T dim6Idx = dim4Rem - dim5Idx * indicesStride[DIM_5];

        if constexpr (DIM == 0) {
            yOffset = sValue * dataStride[0] + dim1Idx * dataStride[DIM_1] + dim2Idx * dataStride[DIM_2] +
                      dim3Idx * dataStride[DIM_3] + dim4Idx * dataStride[DIM_4] + dim5Idx * dataStride[DIM_5] + dim6Idx;
        } else if constexpr (DIM == 1) {
            yOffset = dim0Idx * dataStride[0] + sValue * dataStride[DIM_1] + dim2Idx * dataStride[DIM_2] +
                      dim3Idx * dataStride[DIM_3] + dim4Idx * dataStride[DIM_4] + dim5Idx * dataStride[DIM_5] + dim6Idx;
        } else if constexpr (DIM == DIM_2) {
            yOffset = dim0Idx * dataStride[0] + dim1Idx * dataStride[DIM_1] + sValue * dataStride[DIM_2] +
                      dim3Idx * dataStride[DIM_3] + dim4Idx * dataStride[DIM_4] + dim5Idx * dataStride[DIM_5] + dim6Idx;
        } else if constexpr (DIM == DIM_3) {
            yOffset = dim0Idx * dataStride[0] + dim1Idx * dataStride[DIM_1] + dim2Idx * dataStride[DIM_2] +
                      sValue * dataStride[DIM_3] + dim4Idx * dataStride[DIM_4] + dim5Idx * dataStride[DIM_5] + dim6Idx;
        } else if constexpr (DIM == DIM_4) {
            yOffset = dim0Idx * dataStride[0] + dim1Idx * dataStride[DIM_1] + dim2Idx * dataStride[DIM_2] +
                      dim3Idx * dataStride[DIM_3] + sValue * dataStride[DIM_4] + dim5Idx * dataStride[DIM_5] + dim6Idx;
        } else if constexpr (DIM == DIM_5) {
            yOffset = dim0Idx * dataStride[0] + dim1Idx * dataStride[DIM_1] + dim2Idx * dataStride[DIM_2] +
                      dim3Idx * dataStride[DIM_3] + dim4Idx * dataStride[DIM_4] + sValue * dataStride[DIM_5] + dim6Idx;
        } else {
            yOffset = dim0Idx * dataStride[0] + dim1Idx * dataStride[DIM_1] + dim2Idx * dataStride[DIM_2] +
                      dim3Idx * dataStride[DIM_3] + dim4Idx * dataStride[DIM_4] + dim5Idx * dataStride[DIM_5] + sValue;
        }
        if constexpr (!SAME_SHAPE) {
            updatesOffset = dim0Idx * updatesStride[0] + dim1Idx * updatesStride[DIM_1] +
                            dim2Idx * updatesStride[DIM_2] + dim3Idx * updatesStride[DIM_3] +
                            dim4Idx * updatesStride[DIM_4] + dim5Idx * updatesStride[DIM_5] + dim6Idx;
        }
    } else if constexpr (RANK == DIM_8) {
        COMP_T dim0Idx = Simt::UintDiv(origIndicesOffset, m0, shift0);
        COMP_T dim0Rem = origIndicesOffset - dim0Idx * indicesStride[0];

        COMP_T dim1Idx = Simt::UintDiv(dim0Rem, m1, shift1);
        COMP_T dim1Rem = dim0Rem - dim1Idx * indicesStride[DIM_1];

        COMP_T dim2Idx = Simt::UintDiv(dim1Rem, m2, shift2);
        COMP_T dim2Rem = dim1Rem - dim2Idx * indicesStride[DIM_2];

        COMP_T dim3Idx = Simt::UintDiv(dim2Rem, m3, shift3);
        COMP_T dim3Rem = dim2Rem - dim3Idx * indicesStride[DIM_3];

        COMP_T dim4Idx = Simt::UintDiv(dim3Rem, m4, shift4);
        COMP_T dim4Rem = dim3Rem - dim4Idx * indicesStride[DIM_4];

        COMP_T dim5Idx = Simt::UintDiv(dim4Rem, m5, shift5);
        COMP_T dim5Rem = dim4Rem - dim5Idx * indicesStride[DIM_5];

        COMP_T dim6Idx = Simt::UintDiv(dim5Rem, m6, shift6);
        COMP_T dim7Idx = dim5Rem - dim6Idx * indicesStride[DIM_6];

        if constexpr (DIM == 0) {
            yOffset = sValue * dataStride[0] + dim1Idx * dataStride[DIM_1] + dim2Idx * dataStride[DIM_2] +
                      dim3Idx * dataStride[DIM_3] + dim4Idx * dataStride[DIM_4] + dim5Idx * dataStride[DIM_5] +
                      dim6Idx * dataStride[DIM_6] + dim7Idx;
        } else if constexpr (DIM == 1) {
            yOffset = dim0Idx * dataStride[0] + sValue * dataStride[DIM_1] + dim2Idx * dataStride[DIM_2] +
                      dim3Idx * dataStride[DIM_3] + dim4Idx * dataStride[DIM_4] + dim5Idx * dataStride[DIM_5] +
                      dim6Idx * dataStride[DIM_6] + dim7Idx;
        } else if constexpr (DIM == DIM_2) {
            yOffset = dim0Idx * dataStride[0] + dim1Idx * dataStride[DIM_1] + sValue * dataStride[DIM_2] +
                      dim3Idx * dataStride[DIM_3] + dim4Idx * dataStride[DIM_4] + dim5Idx * dataStride[DIM_5] +
                      dim6Idx * dataStride[DIM_6] + dim7Idx;
        } else if constexpr (DIM == DIM_3) {
            yOffset = dim0Idx * dataStride[0] + dim1Idx * dataStride[DIM_1] + dim2Idx * dataStride[DIM_2] +
                      sValue * dataStride[DIM_3] + dim4Idx * dataStride[DIM_4] + dim5Idx * dataStride[DIM_5] +
                      dim6Idx * dataStride[DIM_6] + dim7Idx;
        } else if constexpr (DIM == DIM_4) {
            yOffset = dim0Idx * dataStride[0] + dim1Idx * dataStride[DIM_1] + dim2Idx * dataStride[DIM_2] +
                      dim3Idx * dataStride[DIM_3] + sValue * dataStride[DIM_4] + dim5Idx * dataStride[DIM_5] +
                      dim6Idx * dataStride[DIM_6] + dim7Idx;
        } else if constexpr (DIM == DIM_5) {
            yOffset = dim0Idx * dataStride[0] + dim1Idx * dataStride[DIM_1] + dim2Idx * dataStride[DIM_2] +
                      dim3Idx * dataStride[DIM_3] + dim4Idx * dataStride[DIM_4] + sValue * dataStride[DIM_5] +
                      dim6Idx * dataStride[DIM_6] + dim7Idx;
        } else if constexpr (DIM == DIM_6) {
            yOffset = dim0Idx * dataStride[0] + dim1Idx * dataStride[DIM_1] + dim2Idx * dataStride[DIM_2] +
                      dim3Idx * dataStride[DIM_3] + dim4Idx * dataStride[DIM_4] + dim5Idx * dataStride[DIM_5] +
                      sValue * dataStride[DIM_6] + dim7Idx;
        } else {
            yOffset = dim0Idx * dataStride[0] + dim1Idx * dataStride[DIM_1] + dim2Idx * dataStride[DIM_2] +
                      dim3Idx * dataStride[DIM_3] + dim4Idx * dataStride[DIM_4] + dim5Idx * dataStride[DIM_5] +
                      dim6Idx * dataStride[DIM_6] + sValue;
        }
        if constexpr (!SAME_SHAPE) {
            updatesOffset = dim0Idx * updatesStride[0] + dim1Idx * updatesStride[DIM_1] +
                            dim2Idx * updatesStride[DIM_2] + dim3Idx * updatesStride[DIM_3] +
                            dim4Idx * updatesStride[DIM_4] + dim5Idx * updatesStride[DIM_5] +
                            dim6Idx * updatesStride[DIM_6] + dim7Idx;
        }
    }
    if constexpr (SAME_SHAPE) {
        updatesOffset = origIndicesOffset;
    }
}

template <typename COMP_T, const uint16_t PATTERN>
__simt_callee__ __aicore__ inline COMP_T WarpAddIndicesOffset(uint32_t a, uint32_t s, uint32_t processA0, COMP_T offset,
                                                              int64_t midAxis, int64_t afterAxis, uint32_t magicA0,
                                                              uint32_t shiftA0)
{
    if constexpr (PATTERN == PATTERN_SA) {
        return offset + static_cast<COMP_T>(s) * afterAxis + a;
    } else if constexpr (PATTERN == PATTERN_AS) {
        return offset + static_cast<COMP_T>(a) * midAxis + s;
    } else {
        uint32_t a1 = Simt::UintDiv(a, magicA0, shiftA0);
        uint32_t a0 = a - a1 * processA0;
        return offset + static_cast<COMP_T>(a1) * midAxis * afterAxis + static_cast<COMP_T>(s) * afterAxis + a0;
    }
}

// Every thread calls this before any per-group branch or early exit. Only
// warp 0 samples; one block barrier publishes the choice to all 1024 threads.
template <typename IDX_T, typename COMP_T>
__simt_callee__ __aicore__ inline uint32_t SelectAdaptiveGroupShift(__ubuf__ IDX_T* sortedKey,
                                                                    __ubuf__ uint32_t* sortedIdx,
                                                                    __ubuf__ COMP_T* params, uint32_t ubProcess,
                                                                    uint32_t magicS, uint32_t shiftS)
{
    if (threadIdx.x < DETERM_WARP_SIZE) {
        const uint32_t adjacentCount = ubProcess > 0 ? ubProcess - 1 : 0;
        const uint32_t sampleCount = adjacentCount < ADAPTIVE_SAMPLE_COUNT ? adjacentCount : ADAPTIVE_SAMPLE_COUNT;
        uint32_t heads = 0;
        for (uint32_t sample = threadIdx.x; sample < sampleCount; sample += DETERM_WARP_SIZE) {
            const uint32_t begin = sample * adjacentCount / sampleCount;
            const uint32_t end = (sample + 1) * adjacentCount / sampleCount;
            // Fixed, disjoint strata with deterministic jitter avoid repeatedly
            // sampling the same offset of periodic duplicate groups.
            uint32_t hash = (sample + 1) * 0x9E3779B9U;
            hash ^= hash >> 16;
            const uint32_t pos = 1 + begin + hash % (end - begin);
            heads += sortedKey[pos] != sortedKey[pos - 1] ||
                     Simt::UintDiv(sortedIdx[pos], magicS, shiftS) != Simt::UintDiv(sortedIdx[pos - 1], magicS, shiftS);
        }
        // All 32 lanes participate, including lanes with no valid sample.
        const uint32_t sampledHeads = asc_reduce_add(heads);
        if (threadIdx.x == 0) {
            uint32_t groupShift = 0;
            // Target about four updates per thread. No sampled boundary
            // selects 32 lanes; an empty sample selects one lane.
            while (groupShift < MAX_GROUP_SHIFT &&
                   sampledHeads * (ADAPTIVE_UPDATES_PER_THREAD << groupShift) < sampleCount) {
                ++groupShift;
            }
            params[ADAPTIVE_GROUP_SHIFT_SLOT] = static_cast<COMP_T>(groupShift);
        }
    }
    asc_syncthreads();
    return static_cast<uint32_t>(params[ADAPTIVE_GROUP_SHIFT_SLOT]);
}

__simt_callee__ __aicore__ inline float ReduceAdaptiveGroup(float value, uint32_t groupShift)
{
    const uint32_t groupSize = 1U << groupShift;
    if (groupSize == DETERM_WARP_SIZE) {
        return asc_reduce_add(value);
    }
    const uint32_t lane = threadIdx.x & (groupSize - 1);
    for (uint32_t delta = groupSize >> 1; delta > 0; delta >>= 1) {
        // Every source lane participates, including zero-contribution lanes.
        const float other = asc_shfl_down(value, delta, groupSize);
        if (lane < delta) {
            value += other;
        }
    }
    return value;
}

template <typename DATA_T, typename IDX_T, typename COMP_T, const uint16_t RANK, const uint16_t DIM,
          const uint16_t PATTERN, const bool SAME_SHAPE>
__simt_vf__ __aicore__ LAUNCH_BOUND(USED_THREAD_DETERM_WARP) inline void SimtComputeWarpAdd(
    __ubuf__ IDX_T* sortedKey, __ubuf__ uint32_t* sortedIdx, __gm__ DATA_T* updates, __gm__ DATA_T* y,
    __ubuf__ uint64_t* tilingUint64Ub, uint32_t processA1, uint32_t processS, uint32_t processA0, COMP_T offset,
    __ubuf__ COMP_T* params, int64_t midAxis, int64_t afterAxis, uint32_t magicS, uint32_t shiftS, uint32_t magicA0,
    uint32_t shiftA0)
{
    const uint32_t ubProcess = processA1 * processS * processA0;
    const uint32_t groupShift = SelectAdaptiveGroupShift(sortedKey, sortedIdx, params, ubProcess, magicS, shiftS);
    const uint32_t groupSize = 1U << groupShift;
    const uint32_t lane = threadIdx.x & (groupSize - 1);
    const uint32_t group = threadIdx.x >> groupShift;
    const uint32_t groupCount = blockDim.x >> groupShift;
    for (uint32_t i = group; i < ubProcess; i += groupCount) {
        const IDX_T sValue = sortedKey[i];
        const uint32_t originalIdx = sortedIdx[i];
        const uint32_t a = Simt::UintDiv(originalIdx, magicS, shiftS);
        // Stable radix sort keeps each (key, flattened A) group contiguous.
        // Every lane in this subgroup makes the same group-head decision.
        if (i > 0 && sortedKey[i - 1] == sValue && Simt::UintDiv(sortedIdx[i - 1], magicS, shiftS) == a) {
            continue;
        }

        float partial = 0.0f;
        for (uint32_t idx = i + lane; idx < ubProcess; idx += groupSize) {
            const uint32_t updateIdx = sortedIdx[idx];
            const uint32_t updateA = Simt::UintDiv(updateIdx, magicS, shiftS);
            if (sortedKey[idx] != sValue || updateA != a) {
                break;
            }
            const uint32_t s = updateIdx - updateA * processS;
            const COMP_T origIndicesOffset = WarpAddIndicesOffset<COMP_T, PATTERN>(
                updateA, s, processA0, offset, midAxis, afterAxis, magicA0, shiftA0);
            // sortedIdx is tile-local; only the restored GM element offset can
            // be shared by indices and updates when their shapes match.
            COMP_T updatesOffset = origIndicesOffset;
            if constexpr (!SAME_SHAPE) {
                COMP_T yOffset = 0;
                CalcOffset<COMP_T, RANK, DIM, SAME_SHAPE>(origIndicesOffset, static_cast<COMP_T>(sValue), yOffset,
                                                          updatesOffset, tilingUint64Ub, params);
            }
            partial += static_cast<float>(updates[updatesOffset]);
        }
        // Explicitly select zero for lanes without a matching update; all
        // lanes still participate in the collective.
        const uint32_t firstIdx = i + lane;
        bool laneHasUpdate = false;
        if (firstIdx < ubProcess) {
            laneHasUpdate = sortedKey[firstIdx] == sValue && Simt::UintDiv(sortedIdx[firstIdx], magicS, shiftS) == a;
        }
        const float groupSum = ReduceAdaptiveGroup(laneHasUpdate ? partial : 0.0f, groupShift);
        if (lane == 0) {
            const uint32_t s = originalIdx - a * processS;
            const COMP_T origIndicesOffset = WarpAddIndicesOffset<COMP_T, PATTERN>(a, s, processA0, offset, midAxis,
                                                                                   afterAxis, magicA0, shiftA0);
            COMP_T yOffset = 0;
            COMP_T updatesOffset = 0;
            CalcOffset<COMP_T, RANK, DIM, SAME_SHAPE>(origIndicesOffset, static_cast<COMP_T>(sValue), yOffset,
                                                      updatesOffset, tilingUint64Ub, params);
            // One conversion/store per group in this S tile, never per-lane partial stores.
            y[yOffset] = static_cast<DATA_T>(static_cast<float>(y[yOffset]) + groupSum);
        }
    }
}

template <typename DATA_T, typename IDX_T, typename COMP_T, const uint32_t REDU, const uint16_t RANK,
          const uint16_t DIM, const uint16_t PATTERN, const bool SAME_SHAPE>
__simt_vf__ __aicore__ LAUNCH_BOUND(USED_THREAD_DETERM) inline void SimtCompute(
    __ubuf__ IDX_T* sortedKey, __ubuf__ uint32_t* sortedIdx, __gm__ DATA_T* updates, __gm__ DATA_T* y,
    __ubuf__ uint64_t* TilingUint64Ub, uint32_t processS, uint32_t processA0, COMP_T offset, __ubuf__ COMP_T* params,
    int64_t midAxis, int64_t afterAxis)
{
    uint32_t ubProcess = static_cast<uint32_t>(processS * processA0);
    for (uint32_t i = threadIdx.x; i < ubProcess; i += blockDim.x) {
        IDX_T sValue = sortedKey[i];
        uint32_t a0Idx = sortedIdx[i] / processS;
        if constexpr (REDU == REDU_ADD) {
            if (i > 0) {
                uint32_t a0PreIdx = sortedIdx[i - 1] / processS;
                if (sValue == sortedKey[i - 1] && a0PreIdx == a0Idx) {
                    continue;
                }
            }
        } else {
            // Radix sort preserves original order for equal keys.
            // Keep the group tail so replacement follows last-writer-wins semantics.
            if (i + 1 < ubProcess) {
                uint32_t a0NextIdx = sortedIdx[i + 1] / processS;
                if (sValue == sortedKey[i + 1] && a0NextIdx == a0Idx) {
                    continue;
                }
            }
        }
        COMP_T yOffset = 0;
        COMP_T updatesOffset = 0;
        COMP_T origIndicesOffset = 0;
        uint32_t sIdx = sortedIdx[i] - a0Idx * processS;
        if constexpr (PATTERN == PATTERN_SA) {
            origIndicesOffset = offset + sIdx * afterAxis + a0Idx;
        } else {
            origIndicesOffset = offset + a0Idx * midAxis + sIdx;
        }
        CalcOffset<COMP_T, RANK, DIM, SAME_SHAPE>(origIndicesOffset, static_cast<COMP_T>(sValue), yOffset,
                                                  updatesOffset, TilingUint64Ub, params);
        float accumulatedValue;
        if constexpr (REDU == REDU_ADD) {
            accumulatedValue = static_cast<float>(y[yOffset]) + static_cast<float>(updates[updatesOffset]);
        } else {
            y[yOffset] = updates[updatesOffset];
            continue;
        }
        for (uint32_t idx = i + 1; idx < ubProcess; idx++) {
            uint32_t a0LoopIdx = sortedIdx[idx] / processS;
            if (sortedKey[idx] == sValue && a0LoopIdx == a0Idx) {
                uint32_t sLoopIdx = sortedIdx[idx] - a0LoopIdx * processS;
                if constexpr (PATTERN == PATTERN_SA) {
                    origIndicesOffset = offset + sLoopIdx * afterAxis + a0LoopIdx;
                } else {
                    origIndicesOffset = offset + a0LoopIdx * midAxis + sLoopIdx;
                }
                if constexpr (SAME_SHAPE) {
                    updatesOffset = origIndicesOffset;
                } else {
                    CalcOffset<COMP_T, RANK, DIM, SAME_SHAPE>(origIndicesOffset, static_cast<COMP_T>(sValue), yOffset,
                                                              updatesOffset, TilingUint64Ub, params);
                }
                accumulatedValue += static_cast<float>(updates[updatesOffset]);
            } else {
                break;
            }
        }
        y[yOffset] = static_cast<DATA_T>(accumulatedValue);
    }
}

template <typename DATA_T, typename IDX_T, typename COMP_T, const uint32_t REDU, const uint16_t RANK,
          const uint16_t DIM, const bool SAME_SHAPE>
__simt_vf__ __aicore__ LAUNCH_BOUND(USED_THREAD_DETERM) inline void SimtComputeASA(
    __ubuf__ IDX_T* sortedKey, __ubuf__ uint32_t* sortedIdx, __gm__ DATA_T* updates, __gm__ DATA_T* y,
    __ubuf__ uint64_t* TilingUint64Ub, uint32_t processA1, uint32_t processS, uint32_t processA0, COMP_T offset,
    __ubuf__ COMP_T* params, int64_t midAxis, int64_t afterAxis)
{
    uint32_t ubProcess = static_cast<uint32_t>(processA1 * processS * processA0);
    for (uint32_t i = threadIdx.x; i < ubProcess; i += blockDim.x) {
        IDX_T sValue = sortedKey[i];
        uint32_t a1Idx = sortedIdx[i] / (processS * processA0);
        uint32_t a1Rem = sortedIdx[i] - a1Idx * (processS * processA0);
        uint32_t a0Idx = a1Rem / processS;
        if constexpr (REDU == REDU_ADD) {
            if (i > 0) {
                uint32_t a1PreIdx = sortedIdx[i - 1] / (processS * processA0);
                uint32_t a1PreRem = sortedIdx[i - 1] - a1PreIdx * (processS * processA0);
                uint32_t a0PreIdx = a1PreRem / processS;
                if (sValue == sortedKey[i - 1] && a0PreIdx == a0Idx && a1PreIdx == a1Idx) {
                    continue;
                }
            }
        } else {
            // Radix sort preserves original order for equal keys.
            // Keep the group tail so replacement follows last-writer-wins semantics.
            if (i + 1 < ubProcess) {
                uint32_t a1NextIdx = sortedIdx[i + 1] / (processS * processA0);
                uint32_t a1NextRem = sortedIdx[i + 1] - a1NextIdx * (processS * processA0);
                uint32_t a0NextIdx = a1NextRem / processS;
                if (sValue == sortedKey[i + 1] && a0NextIdx == a0Idx && a1NextIdx == a1Idx) {
                    continue;
                }
            }
        }
        COMP_T yOffset = 0;
        COMP_T updatesOffset = 0;
        uint32_t sIdx = a1Rem - a0Idx * processS;
        COMP_T origIndicesOffset = offset + a1Idx * midAxis * afterAxis + sIdx * afterAxis + a0Idx;
        CalcOffset<COMP_T, RANK, DIM, SAME_SHAPE>(origIndicesOffset, static_cast<COMP_T>(sValue), yOffset,
                                                  updatesOffset, TilingUint64Ub, params);
        float accumulatedValue;
        if constexpr (REDU == REDU_ADD) {
            accumulatedValue = static_cast<float>(y[yOffset]) + static_cast<float>(updates[updatesOffset]);
        } else {
            y[yOffset] = updates[updatesOffset];
            continue;
        }
        for (uint32_t idx = i + 1; idx < ubProcess; idx++) {
            uint32_t a1LoopIdx = sortedIdx[idx] / (processS * processA0);
            a1Rem = sortedIdx[idx] - a1LoopIdx * (processS * processA0);
            uint32_t a0LoopIdx = a1Rem / processS;
            if (sortedKey[idx] == sValue && a0LoopIdx == a0Idx && a1LoopIdx == a1Idx) {
                uint32_t sLoopIdx = a1Rem - a0LoopIdx * processS;
                origIndicesOffset = offset + a1LoopIdx * midAxis * afterAxis + sLoopIdx * afterAxis + a0LoopIdx;
                if constexpr (SAME_SHAPE) {
                    updatesOffset = origIndicesOffset;
                } else {
                    CalcOffset<COMP_T, RANK, DIM, SAME_SHAPE>(origIndicesOffset, static_cast<COMP_T>(sValue), yOffset,
                                                              updatesOffset, TilingUint64Ub, params);
                }
                accumulatedValue += static_cast<float>(updates[updatesOffset]);
            } else {
                break;
            }
        }
        y[yOffset] = static_cast<DATA_T>(accumulatedValue);
    }
}

template <typename DATA_T, typename IDX_T, typename COMP_T, const uint32_t REDU, const uint32_t TEMPLATE_V2,
          const bool USE_WARP_ADD>
__aicore__ inline void
KernelScatterElementsDeterm<DATA_T, IDX_T, COMP_T, REDU, TEMPLATE_V2, USE_WARP_ADD>::SortAndUpdate()
{
    if (blockIdx_ < tilingData_->indicesUsedCoreNum) {
        EditSimtParam();
        DataSyncBarrier<MemDsbT::UB>();
        if (tilingData_->afterAxis == 1) { // AS(a0,s)
            ProcessPatternAS();
        } else if (tilingData_->preAxis == 1) { // SA(s,a0)
            ProcessPatternSA();
        } else { // ASA(a1,s,a0)
            ProcessPatternASA();
        }
    }
}

template <typename DATA_T, typename IDX_T, typename COMP_T, const uint32_t REDU, const uint32_t TEMPLATE_V2,
          const bool USE_WARP_ADD>
__aicore__ inline void KernelScatterElementsDeterm<DATA_T, IDX_T, COMP_T, REDU, TEMPLATE_V2, USE_WARP_ADD>::Process()
{
    if (tilingData_->allAxis == 0) {
        if constexpr (TEMPLATE_V2 == 0) {
            CopyDataToY();
        }
        return;
    }

    if constexpr (TEMPLATE_V2 == 0) {
        CopyDataToY();
    }
    SyncAll();
    SortAndUpdate();
}
} // namespace ScatterElements

#endif
