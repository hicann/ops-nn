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
 * \file mat_mul_nz2nd_util.h
 * \brief NZ2ND utility functions for vector-based NZ to ND format conversion
 */
#ifndef __OP_KERNEL_MATMUL_V3_NZ2ND_UTIL_H__
#define __OP_KERNEL_MATMUL_V3_NZ2ND_UTIL_H__

#include "mat_mul_v3_common.h"
#include "mat_mul_base_block.h"

using namespace AscendC;
using namespace matmul;

__aicore__ inline void Nz2NdUpdateOffsetC(BaseBlockArgs& params, BlockOffset& offset,
                                          const MatmulTilingData* tilingData, bool isNd)
{
    uint64_t mCntIndex = params.index / params.nCntUse;
    uint64_t nCntIndex = params.index % params.nCntUse;
    if (isNd) {
        offset.offsetC = (nCntIndex * tilingData->matmulTiling.singleCoreN +
                          mCntIndex * tilingData->matmulTiling.singleCoreM * tilingData->matmulTiling.N +
                          (params.mTileAddrOffset * tilingData->matmulTiling.N + params.nTileAddrOffset));
    } else {
        offset.offsetC = (nCntIndex * tilingData->matmulTiling.singleCoreN * tilingData->matmulTiling.M +
                          mCntIndex * tilingData->matmulTiling.singleCoreM * BLOCK_SIZE +
                          (params.mTileAddrOffset * BLOCK_SIZE + params.nTileAddrOffset * tilingData->matmulTiling.M));
    }
}

template <class C_T>
__aicore__ inline void Nz2NdCopyInWithNz(LocalTensor<C_T> tensorNZ, GlobalTensor<C_T> cNzGlobal, uint64_t ubProcessM,
                                         uint8_t pingPongId, uint64_t alignedN, uint64_t baseM)
{
    size_t NfractualNum = alignedN / BLOCK_SIZE;
    DataCopyParams copyParams;
    copyParams.blockCount = NfractualNum;
    if constexpr (std::is_same_v<C_T, float>) {
        copyParams.blockLen = ubProcessM * NUM_TWO;
        copyParams.srcStride = (baseM - ubProcessM) * NUM_TWO;
    } else {
        copyParams.blockLen = ubProcessM;
        copyParams.srcStride = baseM - ubProcessM;
    }
    copyParams.dstStride = 0;
    DataCopy(tensorNZ, cNzGlobal, copyParams);
    SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(pingPongId));
    WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(pingPongId));
}

template <class C_T>
__aicore__ inline void Nz2NdMulsAndGatherMask(LocalTensor<C_T> tensorND, LocalTensor<C_T> tensorNZ, uint64_t ubProcessM,
                                              uint8_t pingPongId, uint64_t alignedN, uint64_t c0Size, uint64_t nBlocks,
                                              GatherMaskParams& params)
{
    size_t NfractalNum = alignedN / BLOCK_SIZE;
    uint8_t repeatTimes = MMV3DivCeil(ubProcessM, 8);
    uint64_t mask[2] = {UINT64_MAX, UINT64_MAX};
    UnaryRepeatParams mulsRepeatParams;

    if constexpr (std::is_same_v<C_T, float>) {
        mulsRepeatParams.srcBlkStride = 2;
        mulsRepeatParams.dstBlkStride = alignedN / c0Size;
        mulsRepeatParams.dstRepStride = alignedN;
        mulsRepeatParams.srcRepStride = 16;
        for (size_t inLoop = 0; inLoop < NfractalNum; inLoop++) {
            Muls(tensorND[inLoop * BLOCK_SIZE], tensorNZ[inLoop * ubProcessM * BLOCK_SIZE], static_cast<C_T>(1.0), mask,
                 repeatTimes, mulsRepeatParams);
            Muls(tensorND[inLoop * BLOCK_SIZE + c0Size], tensorNZ[inLoop * ubProcessM * BLOCK_SIZE + c0Size],
                 static_cast<C_T>(1.0), mask, repeatTimes, mulsRepeatParams);
        }
    } else {
        CopyRepeatParams copyParams;
        copyParams.srcStride = 1;
        copyParams.dstStride = alignedN / c0Size;
        copyParams.dstRepeatSize = alignedN / NUM_TWO;
        copyParams.srcRepeatSize = 8;
        for (size_t inLoop = 0; inLoop < NfractalNum; inLoop++) {
            Copy(tensorND[inLoop * BLOCK_SIZE], tensorNZ[inLoop * ubProcessM * BLOCK_SIZE], mask, repeatTimes,
                 copyParams);
        }
    }

    PipeBarrier<PIPE_V>();

    if (alignedN != nBlocks) {
        uint64_t rsvdCnt = 0;
        params.src0BlockStride = 1;
        params.src0RepeatStride = alignedN / c0Size;
        params.src1RepeatStride = 0;
        params.repeatTimes = ubProcessM;
        GatherMask(tensorND, tensorND, 7, true, nBlocks, params, rsvdCnt);
    }

    SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(pingPongId));
    WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(pingPongId));
}

#endif
