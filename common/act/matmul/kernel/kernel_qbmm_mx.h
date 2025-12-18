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
 * \file kernel_qbmm_mx.h
 * \brief
 */

#ifndef MATMUL_KERNEL_KERNEL_QBMM_MX_H
#define MATMUL_KERNEL_KERNEL_QBMM_MX_H
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "../../utils/common_utils.h"
#include "../../utils/fill_utils.h"
#include "../../utils/grouped_matmul_constant.h"
#include "../../utils/layout_utils.h"
#include "../../utils/tuple_utils.h"
#include "../../utils/coord_utils.h"
#include "../../utils/tensor_utils.h"
#include "../block/block_scheduler_qbmm.h"

namespace Act {
namespace Gemm {
namespace Kernel {
#define QBMM_MX_KERNEL_CLASS_TEM_PARAMS \
    template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
#define QBMM_MX_KERNEL_FUN_TEM_PARAMS ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler

using namespace Act;
using namespace Act::Gemm;
using namespace AscendC;

namespace {
constexpr uint64_t IDX_A_OFFSET = 0UL;
constexpr uint64_t IDX_B_OFFSET = 1UL;
constexpr uint64_t IDX_X1SCALE_OFFSET = 2UL;
constexpr uint64_t IDX_X2SCALE_OFFSET = 3UL;
constexpr uint64_t IDX_BIAS_OFFSET = 4UL;
constexpr uint64_t IDX_C_OFFSET = 5UL;
constexpr uint64_t IDX_M_TILEIDX = 0UL;
constexpr uint64_t IDX_N_TILEIDX = 1UL;
constexpr uint64_t IDX_M_TAIL_SPLIT_TILEIDX = 2UL;
constexpr uint64_t IDX_N_TAIL_SPLIT_TILEIDX = 3UL;
} // namespace

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
class QuantMmBatchMX {
public:
    __aicore__ inline QuantMmBatchMX()
    {}
    __aicore__ inline ~QuantMmBatchMX()
    {}

    static constexpr bool transA = BlockMmad::transA;
    static constexpr bool transB = BlockMmad::transB;

    using BlockSchedulerOp = typename Block::BlockSchedulerSelector<
        ProblemShape, typename BlockMmad::L1TileShape, typename BlockMmad::L0TileShape, BlockScheduler, transA,
        transB>::SchedulerOp;

    using BlockMmadParams = typename BlockMmad::Params;
    using BlockMmadInitArgs = typename BlockMmad::Arguments;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using BiasType = typename BlockMmad::BiasType;
    using LayoutB = typename BlockMmad::LayoutB;
    static constexpr CubeFormat FormatB = TagToFormat<LayoutB>::format;

    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;
    // x1,x2,x1Scale,x2Scale,bias,y
    using BlockOffset = AscendC::Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    using TupleL1L0Shape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    using CoordClass = Coordinate<transA, transB, CubeFormat::ND, FormatB, CubeFormat::ND>;
    using BlockSchedulerParams = typename BlockSchedulerOp::Params;

    struct QBMMTiling {
        uint64_t batchA1;
        uint64_t batchA2;
        uint64_t batchA3;
        uint64_t batchA4;
        uint64_t batchB1;
        uint64_t batchB2;
        uint64_t batchB3;
        uint64_t batchB4;
        uint64_t batchC1;
        uint64_t batchC2;
        uint64_t batchC3;
        uint64_t batchC4;
        uint64_t m;
        uint64_t n;
        uint64_t k;
        uint64_t baseM;
        uint64_t baseN;
        uint64_t baseK;
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        BlockMmadInitArgs mmadArgs;
        BlockSchedulerParams schParams;
        QBMMTiling qbmmParams;
    };

public:
    __aicore__ inline void Init(const Params& params);
    __aicore__ inline void Run(const Params& params);
    __aicore__ inline void operator()(const Params& params)
    {
        Run(params);
    }

private:
    __aicore__ inline void ProcessSingleBatch(const Params& params, BlockSchedulerOp& bs);
    __aicore__ inline void UpdateBatchOffset(uint64_t batchIdx);
    __aicore__ inline void UpdateMMGlobalAddr();
    __aicore__ inline void Iterate(int64_t singleCoreM, int64_t singleCoreN);

private:
    BlockMmad mmadOp_;
    TupleShape problemShape_{};
    BlockOffset batchOffset_{0, 0, 0, 0, 0, 0};
    BlockOffset blockOffset_{0, 0, 0, 0, 0, 0};
    AscendC::GlobalTensor<AType> aGlobal_;
    AscendC::GlobalTensor<BType> bGlobal_;
    AscendC::GlobalTensor<CType> cGlobal_;
    AscendC::GlobalTensor<BiasType> biasGlobal_;
    AscendC::GlobalTensor<fp8_e8m0_t> x1scaleGlobal_;
    AscendC::GlobalTensor<fp8_e8m0_t> x2scaleGlobal_;
    GM_ADDR aTensorPtr_;
    GM_ADDR bTensorPtr_;
    GM_ADDR cTensorPtr_;
    GM_ADDR biasTensorPtr_;
    GM_ADDR x1scaleTensorPtr_;
    GM_ADDR x2scaleTensorPtr_;
    AscendC::TPipe* pipe_;
    uint64_t blockIdx_;
    uint64_t batchA1_;
    uint64_t batchA2_;
    uint64_t batchA3_;
    uint64_t batchA4_;
    uint64_t batchB1_;
    uint64_t batchB2_;
    uint64_t batchB3_;
    uint64_t batchB4_;
    uint64_t batchC1_;
    uint64_t batchC2_;
    uint64_t batchC3_;
    uint64_t batchC4_;
};

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMmBatchMX<QBMM_MX_KERNEL_FUN_TEM_PARAMS>::Run(const Params& params)
{
    Init(params);
    uint64_t batchNum_ = batchC1_ * batchC2_ * batchC3_ * batchC4_;
    BlockSchedulerOp bs(params.problemShape, params.schParams);
    for (uint64_t batchIdx = 0; batchIdx < batchNum_; ++batchIdx) {
        UpdateBatchOffset(batchIdx);
        if (Get<MNK_M>(problemShape_) <= 0 || Get<MNK_N>(problemShape_) <= 0) {
            continue;
        }
        bs.ResetAddrOffsets();
        ProcessSingleBatch(params, bs);
    }
}

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMmBatchMX<QBMM_MX_KERNEL_FUN_TEM_PARAMS>::Init(const Params& params)
{
    if ASCEND_IS_AIV {
        return;
    }
    aTensorPtr_ = params.mmadParams.aGmAddr;
    bTensorPtr_ = params.mmadParams.bGmAddr;
    cTensorPtr_ = params.mmadParams.cGmAddr;
    biasTensorPtr_ = params.mmadParams.biasGmAddr;
    x1scaleTensorPtr_ = params.mmadParams.pertokenScaleGmAddr;
    x2scaleTensorPtr_ = params.mmadParams.scaleGmAddr;
    aGlobal_.SetGlobalBuffer((__gm__ AType*)aTensorPtr_);
    bGlobal_.SetGlobalBuffer((__gm__ BType*)bTensorPtr_);
    cGlobal_.SetGlobalBuffer((__gm__ CType*)cTensorPtr_);
    if (biasTensorPtr_ != nullptr) {
        biasGlobal_.SetGlobalBuffer((__gm__ BiasType*)biasTensorPtr_);
    }
    x1scaleGlobal_.SetGlobalBuffer((__gm__ fp8_e8m0_t*)x1scaleTensorPtr_);
    x2scaleGlobal_.SetGlobalBuffer((__gm__ fp8_e8m0_t*)x2scaleTensorPtr_);

    batchA1_ = params.qbmmParams.batchA1;
    batchA2_ = params.qbmmParams.batchA2;
    batchA3_ = params.qbmmParams.batchA3;
    batchA4_ = params.qbmmParams.batchA4;
    batchB1_ = params.qbmmParams.batchB1;
    batchB2_ = params.qbmmParams.batchB2;
    batchB3_ = params.qbmmParams.batchB3;
    batchB4_ = params.qbmmParams.batchB4;
    batchC1_ = params.qbmmParams.batchC1;
    batchC2_ = params.qbmmParams.batchC2;
    batchC3_ = params.qbmmParams.batchC3;
    batchC4_ = params.qbmmParams.batchC4;

    mmadOp_.Init(params.mmadArgs);
}

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMmBatchMX<QBMM_MX_KERNEL_FUN_TEM_PARAMS>::UpdateBatchOffset(uint64_t batchIdx)
{
    // baseOffset is 0 when batchIdx = 0
    if (batchIdx == 0) {
        return;
    }
    int64_t m = Get<MNK_M>(problemShape_);
    int64_t n = Get<MNK_N>(problemShape_);
    int64_t k = Get<MNK_K>(problemShape_);

    // batchIdx(即batchIdxC)->idxC1-4->batchIdxA batchIdxB
    int64_t idxC1 = batchIdx / (batchC2_ * batchC3_ * batchC4_);
    int64_t modulus1 = batchIdx % (batchC2_ * batchC3_ * batchC4_);
    int64_t idxC2 = modulus1 / (batchC3_ * batchC4_);
    int64_t modulus2 = modulus1 % (batchC3_ * batchC4_);
    int64_t idxC3 = modulus2 / batchC4_;
    int64_t idxC4 = modulus2 % batchC4_;
    int64_t batchIdxA = idxC1 * batchA2_ * batchA3_ * batchA4_ * (batchA1_ / batchC1_) +
                        idxC2 * batchA3_ * batchA4_ * (batchA2_ / batchC2_) + idxC3 * batchA4_ * (batchA3_ / batchC3_) +
                        idxC4 * (batchA4_ / batchC4_);
    int64_t batchIdxB = idxC1 * batchB2_ * batchB3_ * batchB4_ * (batchB1_ / batchC1_) +
                        idxC2 * batchB3_ * batchB4_ * (batchB2_ / batchC2_) + idxC3 * batchB4_ * (batchB3_ / batchC3_) +
                        idxC4 * (batchB4_ / batchC4_);
    // aBatchOffset = batchIdxA * m * k
    Get<IDX_A_OFFSET>(batchOffset_) = batchIdxA * m * k;
    // bBatchOffset = batchIdxB * n * k
    Get<IDX_B_OFFSET>(batchOffset_) = batchIdxB * n * k;
    // cBatchOffset = batchIdxC * m * n
    Get<IDX_C_OFFSET>(batchOffset_) = batchIdx * m * n;
}

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMmBatchMX<QBMM_MX_KERNEL_FUN_TEM_PARAMS>::ProcessSingleBatch(
    const Params& params, BlockSchedulerOp& bs)
{
    CoordClass coord(
        Get<MNK_M>(problemShape_), Get<MNK_N>(problemShape_), Get<MNK_K>(problemShape_), params.qbmmParams.baseM,
        params.qbmmParams.baseN, params.qbmmParams.baseK);
    BlockCoord blockIdx;
    while (bs.GetTileIdx(blockIdx)) {
        BlockShape singleShape = bs.GetBlockShape(blockIdx);
        if (Get<MNK_M>(singleShape) <= 0 || Get<MNK_N>(singleShape) <= 0) {
            return;
        }
        blockOffset_ = coord.template GetQuantOffset<true, false>(
            Get<IDX_M_TILEIDX>(blockIdx), Get<IDX_N_TILEIDX>(blockIdx), Get<IDX_M_TAIL_SPLIT_TILEIDX>(singleShape),
            Get<IDX_N_TAIL_SPLIT_TILEIDX>(singleShape));

        Get<IDX_A_OFFSET>(blockOffset_) += Get<IDX_A_OFFSET>(batchOffset_);
        Get<IDX_B_OFFSET>(blockOffset_) += Get<IDX_B_OFFSET>(batchOffset_);
        Get<IDX_C_OFFSET>(blockOffset_) += Get<IDX_C_OFFSET>(batchOffset_);

        TupleL1L0Shape tileShape{
            static_cast<int64_t>(params.qbmmParams.m),     static_cast<int64_t>(params.qbmmParams.n),
            static_cast<int64_t>(params.qbmmParams.k),     static_cast<int64_t>(params.qbmmParams.baseM),
            static_cast<int64_t>(params.qbmmParams.baseN), static_cast<int64_t>(params.qbmmParams.baseK),
        };
        bs.UpdateNextBatchBlockRoundParams();
    }
}

} // namespace Kernel
} // namespace Gemm
} // namespace Act

#endif
