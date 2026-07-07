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
* \file mat_mul_sc_splitk_al1_fullload_block.h
* \brief Single core split K with AL1 full load block parameter management
*/
#ifndef __OP_KERNEL_MATMUL_V3_SC_SPLITK_AL1_FULLLOAD_BLOCK_H__
#define __OP_KERNEL_MATMUL_V3_SC_SPLITK_AL1_FULLLOAD_BLOCK_H__

#include <cstdint>
#include "mat_mul_v3_common.h"
#include "kernel_operator.h"
#include "mat_mul_base_block.h"

using namespace AscendC;
using namespace matmul;

struct SingleCoreSplitKAL1FullLoadBlockArgs {
    bool isTransposeA;
    bool isTransposeB;
    uint32_t isHf32;

    /// 沿 N 方向以 baseN 为步长的基本块个数 ceil(N / baseN)
    uint64_t nCnt;
    uint64_t curCoreN;

    uint64_t kAL1LoopNum;           // A 矩阵在 L1 上按 K 条带重复载入次数
    uint64_t kAL1LoopTail;          // 最后一圈 K 条带大小（元素个数，沿 K）
    uint64_t kAL1FullLoadSize;      // stepKa * baseK，单条带满载 K 尺寸
    /// stepKa * baseK >= Ka：A 一次搬入 L1，无 K 向累加，可直接写 C 输出（无需 fp32 workspace + cast）
    bool isSingleKStrip;

    uint64_t currentKBlockStart;    // 当前 k 循环在 K 轴上的起始偏移（元素）
    uint64_t currentKBlockSize;     // 当前 k 条带实际大小（与高阶 API SetSingleShape 的 K 一致）

    uint64_t innerBlockN;
    uint64_t innerLoopN;
    uint64_t innerSingleCoreN;
    bool atomicAddFlag;

    /// N 方向 baseN 条带的全局编号，范围 [0, nCnt)；每完成一条带上的全部 K 累加后 +1
    uint64_t nBaseTileGlobalIdx;
    /// ceil(nCnt / usedCoreNum)，单核至多要处理的 N 条带条数（用于头/尾核分配）
    uint64_t nBaseTileCeilPerCore;
    /// 当前核实际要循环的 N 条带条数（头核 = ceil，其余尾核可能少 1）
    uint64_t nBaseTileItersThisCore;
    /// 当前核负责的首个 N 条带全局编号（K 外循环每轮需重置）
    uint64_t nBaseTileStartIdx;
    uint64_t preCoreNum;

    uint64_t outNAlign;
    uint64_t c0Size;
    uint64_t alignedOriM;
    uint64_t alignedOriN;
    uint64_t alignedKaSize;
    uint64_t alignedKbSize;

    uint64_t coreOffsetN;
    uint64_t coreIdx;
};

class MatmulSingleCoreSplitKAL1FullLoadBlock {
public:
    __aicore__ inline MatmulSingleCoreSplitKAL1FullLoadBlock() {}
    template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
    __aicore__ inline void Init(const MatmulTilingData *matmulTilingData);
    __aicore__ inline void InitBlockIndex();
    __aicore__ inline void UpdateBlockParamsK(uint64_t kIndex);
    template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
    __aicore__ inline void CalcGMOffset(uint64_t kIndex);

public:
    BlockOffset offset_;
    SingleCoreSplitKAL1FullLoadBlockArgs params_;
    const MatmulTilingData *matmulTilingData_;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void MatmulSingleCoreSplitKAL1FullLoadBlock::Init(
    const MatmulTilingData *matmulTilingData)
{
    matmulTilingData_ = matmulTilingData;
    params_.isTransposeA = matmulTilingData_->matmulRunInfo.transA;
    params_.isTransposeB = matmulTilingData_->matmulRunInfo.transB;
    params_.isHf32 = matmulTilingData_->matmulRunInfo.isHf32;

    const uint64_t baseN = static_cast<uint64_t>(matmulTilingData_->matmulTiling.baseN);
    const uint64_t fullN = static_cast<uint64_t>(matmulTilingData_->matmulTiling.N);

    params_.nCnt = MMV3DivCeil(fullN, baseN);

    params_.curCoreN = baseN;

    params_.kAL1FullLoadSize = static_cast<uint64_t>(matmulTilingData_->matmulTiling.stepKa) *
        static_cast<uint64_t>(matmulTilingData_->matmulTiling.baseK);
    params_.isSingleKStrip = params_.kAL1FullLoadSize >=
        static_cast<uint64_t>(matmulTilingData_->matmulTiling.Ka);
    params_.kAL1LoopNum = MMV3DivCeil(matmulTilingData_->matmulTiling.Ka, params_.kAL1FullLoadSize);
    params_.kAL1LoopTail = static_cast<uint64_t>(matmulTilingData_->matmulTiling.Ka) -
        (params_.kAL1LoopNum - 1UL) * params_.kAL1FullLoadSize;

    params_.innerBlockN = static_cast<uint64_t>(matmulTilingData_->matmulTiling.stepN) *
        matmulTilingData_->matmulTiling.baseN;
    params_.atomicAddFlag = false;
    params_.nBaseTileGlobalIdx = 0;
    params_.nBaseTileCeilPerCore =
        MMV3DivCeil(params_.nCnt, matmulTilingData_->matmulTiling.usedCoreNum);
    params_.nBaseTileItersThisCore = 0;
    params_.preCoreNum = params_.nCnt % matmulTilingData_->matmulTiling.usedCoreNum;
    if (params_.preCoreNum == 0) {
        params_.preCoreNum = matmulTilingData_->matmulTiling.usedCoreNum;
    }
    params_.coreIdx = GetCurrentBlockIdx();

    uint64_t cTypeSize = 64;
    params_.outNAlign = MMV3DivCeil(matmulTilingData_->matmulTiling.N, cTypeSize) * cTypeSize;
    using A_T = typename A_TYPE::T;
    if constexpr (sizeof(A_T) == sizeof(float)) {
        params_.outNAlign = matmulTilingData_->matmulTiling.N;
    }

    GetSizeC0<A_T>(params_.c0Size);
    params_.alignedOriM = MMV3DivCeil(matmulTilingData_->matmulTiling.M, ALIGNED_H) * ALIGNED_H;
    params_.alignedOriN = MMV3DivCeil(matmulTilingData_->matmulTiling.N, params_.c0Size) * params_.c0Size;
    params_.alignedKaSize = MMV3DivCeil(matmulTilingData_->matmulTiling.Ka, params_.c0Size) * params_.c0Size;
    params_.alignedKbSize = MMV3DivCeil(matmulTilingData_->matmulTiling.Kb, ALIGNED_H) * ALIGNED_H;

    params_.currentKBlockStart = 0;
    params_.currentKBlockSize = params_.kAL1FullLoadSize;
}

__aicore__ inline void MatmulSingleCoreSplitKAL1FullLoadBlock::InitBlockIndex()
{
    if (params_.coreIdx < params_.preCoreNum) {
        params_.nBaseTileGlobalIdx = params_.coreIdx * params_.nBaseTileCeilPerCore;
        params_.nBaseTileItersThisCore = params_.nBaseTileCeilPerCore;
    } else {
        params_.nBaseTileGlobalIdx =
            params_.coreIdx * (params_.nBaseTileCeilPerCore - 1UL) + params_.preCoreNum;
        params_.nBaseTileItersThisCore = params_.nBaseTileCeilPerCore - 1UL;
    }
    params_.nBaseTileStartIdx = params_.nBaseTileGlobalIdx;

    const uint64_t baseN = static_cast<uint64_t>(matmulTilingData_->matmulTiling.baseN);
    const uint64_t fullN = static_cast<uint64_t>(matmulTilingData_->matmulTiling.N);
    const uint64_t tileNum = params_.nBaseTileItersThisCore;
    const uint64_t lastTileIdx = params_.nBaseTileStartIdx + tileNum - 1UL;

    params_.coreOffsetN = params_.nBaseTileStartIdx * baseN;
    if (lastTileIdx == params_.nCnt - 1UL) {
        params_.curCoreN = (tileNum - 1UL) * baseN + (fullN - (params_.nCnt - 1UL) * baseN);
    } else {
        params_.curCoreN = tileNum * baseN;
    }
}

__aicore__ inline void MatmulSingleCoreSplitKAL1FullLoadBlock::UpdateBlockParamsK(uint64_t kIndex)
{
    params_.currentKBlockStart = kIndex * params_.kAL1FullLoadSize;
    if (kIndex == params_.kAL1LoopNum - 1) {
        params_.currentKBlockSize = params_.kAL1LoopTail;
    } else {
        params_.currentKBlockSize = params_.kAL1FullLoadSize;
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void MatmulSingleCoreSplitKAL1FullLoadBlock::CalcGMOffset(uint64_t kIndex)
{
    (void)kIndex;
    const uint64_t nColOffset = params_.coreOffsetN;

    if constexpr (B_TYPE::format == CubeFormat::ND) {
        if (params_.isTransposeB) {
            offset_.offsetB =
                params_.currentKBlockStart + nColOffset * matmulTilingData_->matmulTiling.Kb;
        } else {
            offset_.offsetB =
                params_.currentKBlockStart * matmulTilingData_->matmulTiling.N + nColOffset;
        }
    } else if constexpr (B_TYPE::format == CubeFormat::NZ) {
        if (params_.isTransposeB) {
            offset_.offsetB =
                params_.currentKBlockStart * params_.alignedOriN + nColOffset * params_.c0Size;
        } else {
            offset_.offsetB =
                params_.currentKBlockStart * params_.c0Size + nColOffset * params_.alignedKbSize;
        }
    }
    offset_.offsetC = nColOffset;
    offset_.offsetBias = nColOffset;
}

#endif // __OP_KERNEL_MATMUL_V3_SC_SPLITK_AL1_FULLLOAD_BLOCK_H__