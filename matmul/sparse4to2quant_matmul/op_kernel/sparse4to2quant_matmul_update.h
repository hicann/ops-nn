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
 * \file sparse4to2quant_matmul_update.h
 * \brief
 */
#ifndef SPARSE4TO2_QUANT_MATMUL_UPDATE_H
#define SPARSE4TO2_QUANT_MATMUL_UPDATE_H

#include "sparse4to2quant_matmul_base.h"

namespace AscendC {

struct SparseUpdateInfo {
    uint64_t mBaseTail; // 只量化mm用
    uint64_t nBaseTail; // 只量化mm用
    uint64_t alignedKaSize;
    uint64_t alignedKbSize;
};

constexpr uint32_t K0_INT8 = 32;
constexpr uint32_t K0_INDEX = 8;

class Sparse4to2QuantMatmulUpdate {
public:
    __aicore__ inline Sparse4to2QuantMatmulUpdate()
    {}
    template <CubeFormat x1Format, CubeFormat x2Format, bool aTrans, bool bTrans>
    __aicore__ inline void Init(const TCubeTiling* mmTiling, const SparseBaseBlockArgs& params);
    template <CubeFormat x1Format, CubeFormat x2Format, bool aTrans, bool bTrans>
    __aicore__ inline void UpdateBlockParamsAndCalcGmOffset(
        SparseBaseBlockArgs& params, SparseBlockOffset& offset, uint64_t mTileIndex, uint64_t nTileIndex);
    __aicore__ inline void UpdateBlockParams(SparseBaseBlockArgs& params, uint64_t mTileIndex, uint64_t nTileIndex);
    template <CubeFormat x1Format, CubeFormat x2Format, bool aTrans, bool bTrans>
    __aicore__ inline void CalcGMOffset(SparseBaseBlockArgs& params, SparseBlockOffset& offset);

private:
    SparseUpdateInfo info_;
    const TCubeTiling* mmTiling_;
};

template <CubeFormat x1Format, CubeFormat x2Format, bool aTrans, bool bTrans>
__aicore__ inline void Sparse4to2QuantMatmulUpdate::Init(const TCubeTiling* mmTiling, const SparseBaseBlockArgs& params)
{
    mmTiling_ = mmTiling;
    info_.nBaseTail =
        static_cast<uint64_t>(mmTiling_->N) - (params.nTotalCnt - 1) * mmTiling_->singleCoreN; // n方向上的base尾块
    info_.mBaseTail =
        static_cast<uint64_t>(mmTiling_->M) - (params.mTotalCnt - 1) * mmTiling_->singleCoreM; // m方向上的base尾块
    // (m, k)
    info_.alignedKaSize = SparseQmm::Align(mmTiling_->Ka, K0_INT8);
    info_.alignedKbSize = SparseQmm::Align(mmTiling_->Kb, K0_INT8);
}

template <CubeFormat x1Format, CubeFormat x2Format, bool aTrans, bool bTrans>
__aicore__ inline void Sparse4to2QuantMatmulUpdate::UpdateBlockParamsAndCalcGmOffset(
    SparseBaseBlockArgs& params, SparseBlockOffset& offset, uint64_t mTileIndex, uint64_t nTileIndex)
{
    UpdateBlockParams(params, mTileIndex, nTileIndex);
    CalcGMOffset<x1Format, x2Format, aTrans, bTrans>(params, offset);
}

__aicore__ inline void Sparse4to2QuantMatmulUpdate::UpdateBlockParams(
    SparseBaseBlockArgs& params, uint64_t mTileIndex, uint64_t nTileIndex)
{
    if ((mTileIndex == (params.mTileCntL2 - 1)) && (nTileIndex == (params.nTileCntL2 - 1)) &&
        (params.index == (params.totalTileCnt - 1))) {
        params.singleCoreM = info_.mBaseTail;
        params.singleCoreN = info_.nBaseTail;
    } else if ((mTileIndex == (params.mTileCntL2 - 1)) && (params.index >= (params.mCntUse - 1) * params.nCntUse)) {
        params.singleCoreM = info_.mBaseTail;
        params.singleCoreN = mmTiling_->baseN;
    } else if ((nTileIndex == (params.nTileCntL2 - 1)) && ((params.index + 1) % params.nCntUse == 0)) {
        params.singleCoreM = mmTiling_->baseM;
        params.singleCoreN = info_.nBaseTail;
    } else {
        params.singleCoreM = mmTiling_->baseM;
        params.singleCoreN = mmTiling_->baseN;
    }
}

template <CubeFormat x1Format, CubeFormat x2Format, bool aTrans, bool bTrans>
__aicore__ inline void Sparse4to2QuantMatmulUpdate::CalcGMOffset(SparseBaseBlockArgs& params, SparseBlockOffset& offset)
{
    uint64_t mCntIndex = params.index / params.nCntUse;
    uint64_t nCntIndex = params.index - mCntIndex * params.nCntUse;
    // tiling已保证baseM/K/N低轴16/32对齐
    // 前面的m都能保证是singleCoreM(baseM)的倍数，m尾块只会出现在m轴最后
    uint64_t mOffset = mCntIndex * mmTiling_->singleCoreM + params.mTileAddrOffset;
    if constexpr (x1Format == CubeFormat::ND) {
        offset.offsetA = mOffset * mmTiling_->Ka;
    } else if constexpr (x1Format == CubeFormat::NZ) {
        // (k1, m1, m0, k0)
        offset.offsetA = mOffset * K0_INT8;
    }
    // 前面的n都能保证是singleCoreN(baseN)的倍数，n尾块只会出现在n轴最后
    uint64_t nOffset = nCntIndex * mmTiling_->singleCoreN + params.nTileAddrOffset;
    offset.offsetB = nOffset * K0_INT8;
    offset.offsetSparseIndex = nOffset * K0_INDEX;
    // 输出只支持ND
    offset.offsetC = mOffset * mmTiling_->N + nOffset;
    // scale的计算当perchannel统一处理，实际参与mm/ub计算时会区分perchannel/pertensor
    offset.offsetScale = nOffset;
    // 当前基本块算法无batch轴，所以只考虑单维度，可选输入当存在时计算
    offset.offsetBias = nOffset;
    // 即使是纯cube场景，也可以计算pertoken偏移，这样mm输入输出的update接口可以共用
    offset.offsetPertoken = mOffset;
}
} // namespace AscendC
#endif // SPARSE4TO2_QUANT_MATMUL_UPDATE_H