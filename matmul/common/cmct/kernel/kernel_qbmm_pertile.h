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
 * \file kernel_qbmm_pertile.h
 * \brief
 */

#ifndef MATMUL_KERNEL_KERNEL_QBMM_PERTILE_H
#define MATMUL_KERNEL_KERNEL_QBMM_PERTILE_H

#include "kernel_operator.h"
#include "kernel_operator_intf.h"

#include "../utils/common_utils.h"
#include "../utils/fill_utils.h"
#include "../utils/grouped_matmul_constant.h"
#include "../utils/layout_utils.h"
#include "../utils/tuple_utils.h"
#include "../utils/coord_utils.h"
#include "../utils/tensor_utils.h"

namespace Cmct {
namespace Gemm {
namespace Kernel {


#define QBMM_PERTILE_KERNEL_CLASS_TEM_PARAMS \
    template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
#define QBMM_PERTILE_KERNEL_FUN_TEM_PARAMS ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler

using namespace Cmct::Gemm::GroupedMatmul;

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
constexpr uint32_t PER_BLOCK_SIZE = 128;
}

QBMM_PERTILE_KERNEL_CLASS_TEM_PARAMS
class QuantMmBatchPertile {
public:
    __aicore__ inline QuantMmBatchPertile()
    {}
    __aicore__ inline ~QuantMmBatchPertile()
    {}

    static constexpr bool transA = BlockMmad::transA;
    static constexpr bool transB = BlockMmad::transB;

    using BlockSchedulerOp = typename Block::BlockSchedulerSelector<
        ProblemShape, typename BlockMmad::L1TileShape, typename BlockMmad::L0TileShape, BlockScheduler, transA,
        transB>::SchedulerOp;

    using BlockMmadParams = typename BlockMmad::Params;
    using BlockEpilogueParams = typename BlockEpilogue::Params;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using BiasType = typename BlockMmad::BiasType;
    using YType = typename BlockEpilogue::YType;
    using LayoutB = typename BlockMmad::LayoutB;

    static constexpr CubeFormat FormatB = TagToFormat<LayoutB>::format;

    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;
    using BlockOffset = AscendC::Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    using CoordClass = Coordinate<transA, transB, CubeFormat::ND, FormatB, CubeFormat::ND>;

    struct QBMMTiling {
        uint32_t batchC;
        uint32_t batchA1;
        uint32_t batchA2;
        uint32_t batchA3;
        uint32_t batchA4;
        uint32_t batchB1;
        uint32_t batchB2;
        uint32_t batchB3;
        uint32_t batchB4;
        uint32_t batchC1;
        uint32_t batchC2;
        uint32_t batchC3;
        uint32_t batchC4;
        int32_t m;
        int32_t n;
        int32_t k;
        int32_t baseM;
        int32_t baseN;
        int32_t baseK;
        int32_t stepM;
        int32_t stepN;
        int32_t stepKa;
        int32_t stepKb;

        uint32_t mTailTile;
        uint32_t nTailTile;
        uint32_t mBaseTailSplitCnt;
        uint32_t mTailMain;

        uint32_t groupSizeM;
        uint32_t groupSizeN;
        uint32_t groupSizeK;

        __aicore__ QBMMTiling()
        {}
        __aicore__ QBMMTiling(
            uint32_t batchC_, uint32_t batchA1_, uint32_t batchA2_, uint32_t batchA3_, uint32_t batchA4_,
            uint32_t batchB1_, uint32_t batchB2_, uint32_t batchB3_, uint32_t batchB4_, uint32_t batchC1_,
            uint32_t batchC2_, uint32_t batchC3_, uint32_t batchC4_, int32_t m_, int32_t n_, int32_t k_, int32_t baseM_,
            int32_t baseN_, int32_t baseK_, int32_t stepM_, int32_t stepN_, int32_t stepKa_, int32_t stepKb_,
            uint32_t mTailTile_, uint32_t nTailTile_, uint32_t mBaseTailSplitCnt_, uint32_t mTailMain_,
            uint32_t groupSizeM_, uint32_t groupSizeN_, uint32_t groupSizeK_)
            : batchC(batchC_),
              batchA1(batchA1_),
              batchA2(batchA2_),
              batchA3(batchA3_),
              batchA4(batchA4_),
              batchB1(batchB1_),
              batchB2(batchB2_),
              batchB3(batchB3_),
              batchB4(batchB4_),
              batchC1(batchC1_),
              batchC2(batchC2_),
              batchC3(batchC3_),
              batchC4(batchC4_),
              m(m_),
              n(n_),
              k(k_),
              baseM(baseM_),
              baseN(baseN_),
              baseK(baseK_),
              stepM(stepM_),
              stepN(stepN_),
              stepKa(stepKa_),
              stepKb(stepKb_),
              mTailTile(mTailTile_),
              nTailTile(nTailTile_),
              mBaseTailSplitCnt(mBaseTailSplitCnt_),
              mTailMain(mTailMain_),
              groupSizeM(groupSizeM_),
              groupSizeN(groupSizeN_),
              groupSizeK(groupSizeK_)
        {}
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        BlockEpilogueParams epilogueParams;
        QBMMTiling qbmmParams;
        Params() = default;
    };

public:
    __aicore__ inline void Init(const Params& params);
    __aicore__ inline void Run(const Params& params);
    __aicore__ inline void operator()(const Params& params)
    {
        Run(params);
    }

private:
    __aicore__ inline void ProcessWithoutBatch(const Params& params, BlockSchedulerOp& bs, bool isLastBatch);
    __aicore__ inline void ProcessWithBatch(const Params& params, BlockSchedulerOp& bs);
    __aicore__ inline void UpdateOffset(uint64_t batchA4Offset, uint64_t batchB4Offset, uint64_t batchC4Offset);
    __aicore__ inline void UpdateMMGlobalAddr();
    __aicore__ inline void Iterate(int64_t singleCoreM, int64_t singleCoreN);
    __aicore__ inline void End();

private:
    BlockMmad mmadOp_;
    BlockEpilogue epilogueOp_;
    TupleShape problemShape_{};
    BlockOffset baseOffset_{0, 0, 0, 0, 0, 0};
    BlockOffset blockOffset_{0, 0, 0, 0, 0, 0};

    AscendC::GlobalTensor<AType> aGlobal_;
    AscendC::GlobalTensor<BType> bGlobal_;
    AscendC::LocalTensor<CType> mmResPing_;
    AscendC::LocalTensor<CType> mmResPong_;

    GM_ADDR xTensorPtr_;
    GM_ADDR wTensorPtr_;
    GM_ADDR yTensorPtr_;

    uint32_t blockIdx_;
    int32_t preOffset_ = 0;

    uint32_t mTailTile_;
    uint32_t nTailTile_;
    uint32_t mTailMain_;
    uint32_t mTailLast_;
    uint32_t mBaseNormCnt_;
    uint32_t nTailMain_;
    uint32_t nBaseNormCnt_;
    uint32_t groupSizeM_;
    uint32_t groupSizeN_;
    uint32_t groupSizeK_;

    bool isPertile_;
};

QBMM_PERTILE_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMmBatchPertile<QBMM_PERTILE_KERNEL_FUN_TEM_PARAMS>::Run(const Params& params)
{
    Init(params);
    bool isKZeroInit = false;
    BlockSchedulerOp bs(params.qbmmParams.baseM, params.qbmmParams.baseN, params.qbmmParams.baseK);
    if constexpr (!transA) {
        bs.SetLoadBalanceParam(mBaseNormCnt_, mTailMain_, mTailLast_);
    }

    if (params.qbmmParams.batchC == 1UL) {
        ProcessWithoutBatch(params, bs, true);
        End();
        return;
    }

    ProcessWithBatch(params, bs);
    End();
}

QBMM_PERTILE_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMmBatchPertile<QBMM_PERTILE_KERNEL_FUN_TEM_PARAMS>::Init(const Params& params)
{
    xTensorPtr_ = params.mmadParams.aGmAddr;
    wTensorPtr_ = params.mmadParams.bGmAddr;
    yTensorPtr_ = params.mmadParams.cGmAddr;

    blockIdx_ = AscendC::GetBlockIdx();
    if ASCEND_IS_AIV {
        blockIdx_ = blockIdx_ / AscendC::GetTaskRation();
    }

    TupleShape l0Shape{
        static_cast<int64_t>(params.qbmmParams.baseM), static_cast<int64_t>(params.qbmmParams.baseN),
        static_cast<int64_t>(params.qbmmParams.baseK)};
    BlockShape tileL12L0{
        static_cast<int64_t>(params.qbmmParams.stepM), static_cast<int64_t>(params.qbmmParams.stepN),
        static_cast<int64_t>(params.qbmmParams.stepKa), static_cast<int64_t>(params.qbmmParams.stepKb)};

    auto mmResPing_ = epilogueOp_.GetL0c2UbPingTensor();
    auto mmResPong_ = epilogueOp_.GetL0c2UbPongTensor();
    mmadOp_.Init(l0Shape, tileL12L0, &mmResPing_, &mmResPong_);
    epilogueOp_.Init(&params.epilogueParams);

    Get<MNK_M>(problemShape_) = params.qbmmParams.m;
    Get<MNK_N>(problemShape_) = params.qbmmParams.n;
    Get<MNK_K>(problemShape_) = params.qbmmParams.k;
    mTailTile_ = params.qbmmParams.mTailTile;
    nTailTile_ = params.qbmmParams.nTailTile;
    if constexpr (!transA) {
        int32_t mBaseTailSplitCnt_ = params.qbmmParams.mBaseTailSplitCnt;
        mBaseNormCnt_ =
            CeilDiv(static_cast<int64_t>(params.qbmmParams.m), static_cast<int64_t>(params.qbmmParams.baseM)) -
            mBaseTailSplitCnt_;
        nBaseNormCnt_ =
            CeilDiv(static_cast<int64_t>(params.qbmmParams.n), static_cast<int64_t>(params.qbmmParams.baseN)) - 1;
        int32_t mergeSize = params.qbmmParams.m - mBaseNormCnt_ * params.qbmmParams.baseM;
        mTailMain_ = mBaseTailSplitCnt_ == 1 ? mergeSize : params.qbmmParams.mTailMain;
        nTailMain_ = params.qbmmParams.n - nBaseNormCnt_ * params.qbmmParams.baseN;
        mTailLast_ = mergeSize - (mBaseTailSplitCnt_ - 1) * mTailMain_;
    }

    isPertile_ = (params.qbmmParams.groupSizeM == 1);
    if ASCEND_IS_AIC {
        mmadOp_.UpdateParamsForNextProblem(problemShape_);
    }
}

QBMM_PERTILE_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMmBatchPertile<QBMM_PERTILE_KERNEL_FUN_TEM_PARAMS>::UpdateOffset(
    uint64_t batchA4Offset, uint64_t batchB4Offset, uint64_t batchC4Offset)
{
    Get<IDX_A_OFFSET>(baseOffset_) = batchA4Offset * Get<MNK_M>(problemShape_) * Get<MNK_K>(problemShape_);
    Get<IDX_B_OFFSET>(baseOffset_) = batchB4Offset * Get<MNK_N>(problemShape_) * Get<MNK_K>(problemShape_);
    Get<IDX_C_OFFSET>(baseOffset_) = batchC4Offset * Get<MNK_M>(problemShape_) * Get<MNK_N>(problemShape_);

    if (isPertile_) {
        Get<IDX_X1SCALE_OFFSET>(baseOffset_) = batchA4Offset * static_cast<uint64_t>(Get<MNK_M>(problemShape_)) *
                                               CeilDiv(Get<MNK_K>(problemShape_), PER_BLOCK_SIZE);
    } else {
        Get<IDX_X1SCALE_OFFSET>(baseOffset_) = batchA4Offset * CeilDiv(Get<MNK_M>(problemShape_), PER_BLOCK_SIZE) *
                                               CeilDiv(Get<MNK_K>(problemShape_), PER_BLOCK_SIZE);
    }

    Get<IDX_X2SCALE_OFFSET>(baseOffset_) = batchB4Offset * CeilDiv(Get<MNK_K>(problemShape_), PER_BLOCK_SIZE) *
                                           CeilDiv(Get<MNK_N>(problemShape_), PER_BLOCK_SIZE);
}

QBMM_PERTILE_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMmBatchPertile<QBMM_PERTILE_KERNEL_FUN_TEM_PARAMS>::ProcessWithBatch(
    const Params& params, BlockSchedulerOp& bs)
{
    const auto& p = params.qbmmParams;

    const uint64_t batchC3C4 = static_cast<uint64_t>(p.batchC3) * p.batchC4;
    const uint64_t batchC2C3C4 = static_cast<uint64_t>(p.batchC2) * batchC3C4;
    const uint64_t batchB3B4 = static_cast<uint64_t>(p.batchB3) * p.batchB4;
    const uint64_t batchB2B3B4 = static_cast<uint64_t>(p.batchB2) * batchB3B4;
    const uint64_t batchA3A4 = static_cast<uint64_t>(p.batchA3) * p.batchA4;
    const uint64_t batchA2A3A4 = static_cast<uint64_t>(p.batchA2) * batchA3A4;

    uint32_t multiA1C1 = p.batchA1 / p.batchC1;
    uint32_t multiA2C2 = p.batchA2 / p.batchC2;
    uint32_t multiA3C3 = p.batchA3 / p.batchC3;
    uint32_t multiA4C4 = p.batchA4 / p.batchC4;
    uint32_t multiB1C1 = p.batchB1 / p.batchC1;
    uint32_t multiB2C2 = p.batchB2 / p.batchC2;
    uint32_t multiB3C3 = p.batchB3 / p.batchC3;
    uint32_t multiB4C4 = p.batchB4 / p.batchC4;

    uint64_t batchC1Offset = 0;
    uint64_t batchA1Offset = 0;
    uint64_t batchB1Offset = 0;

    for (uint64_t b1Index = 0; b1Index < p.batchC1; ++b1Index) {
        uint64_t batchC2Offset = batchC1Offset;
        uint64_t batchA2Offset = batchA1Offset;
        uint64_t batchB2Offset = batchB1Offset;

        for (uint64_t b2Index = 0; b2Index < p.batchC2; ++b2Index) {
            uint64_t batchC3Offset = batchC2Offset;
            uint64_t batchA3Offset = batchA2Offset;
            uint64_t batchB3Offset = batchB2Offset;

            for (uint64_t b3Index = 0; b3Index < p.batchC3; ++b3Index) {
                uint64_t batchC4Offset = batchC3Offset;
                uint64_t batchA4Offset = batchA3Offset;
                uint64_t batchB4Offset = batchB3Offset;

                for (uint64_t b4Index = 0; b4Index < p.batchC4; ++b4Index) {
                    UpdateOffset(batchA4Offset, batchB4Offset, batchC4Offset);
                    bool isLastBatch = (b1Index == p.batchC1 - 1) && (b2Index == p.batchC2 - 1) &&
                                       (b3Index == p.batchC3 - 1) && (b4Index == p.batchC4 - 1);
                    ProcessWithoutBatch(params, bs, isLastBatch);

                    batchC4Offset += 1;
                    batchA4Offset += multiA4C4;
                    batchB4Offset += multiB4C4;
                }
                batchC3Offset += p.batchC4;
                batchA3Offset += p.batchA4 * static_cast<uint64_t>(multiA3C3);
                batchB3Offset += p.batchB4 * static_cast<uint64_t>(multiB3C3);
            }
            batchC2Offset += batchC3C4;
            batchA2Offset += batchA3A4 * multiA2C2;
            batchB2Offset += batchB3B4 * multiB2C2;
        }
        batchC1Offset += batchC2C3C4;
        batchA1Offset += batchA2A3A4 * multiA1C1;
        batchB1Offset += batchB2B3B4 * multiB1C1;
    }
}

QBMM_PERTILE_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMmBatchPertile<QBMM_PERTILE_KERNEL_FUN_TEM_PARAMS>::ProcessWithoutBatch(
    const Params& params, BlockSchedulerOp& bs, bool isLastBatch)
{
    if ASCEND_IS_AIV {
        epilogueOp_.UpdateParamsForNextProblem(problemShape_);
    }

    bs.UpdateNextProblem(AscendC::Std::tuple<int64_t, int64_t, int64_t, int64_t>{
                         Get<MNK_M>(problemShape_), Get<MNK_N>(problemShape_), Get<MNK_K>(problemShape_), 0L});

    if (isLastBatch && (bs.GetEndBlockIdx() + 1) * mTailTile_ * nTailTile_ <= AscendC::GetBlockNum()) {
        bs.UpdateTailTile(mTailTile_, nTailTile_);
    }

    UpdateMMGlobalAddr();

    CoordClass coord(
        Get<MNK_M>(problemShape_), Get<MNK_N>(problemShape_), Get<MNK_K>(problemShape_), params.qbmmParams.baseM,
        params.qbmmParams.baseN, params.qbmmParams.baseK);
    BlockCoord tileIdx;

    while (bs.GetTileIdx(tileIdx)) {
        BlockShape singleShape = bs.GetBlockShape(tileIdx, true);

        if (Get<MNK_M>(singleShape) <= 0 || Get<MNK_N>(singleShape) <= 0) {
            return;
        }

        if (isPertile_) {
            if constexpr (!transA) {
                blockOffset_ = coord.template GetQuantOffset<GroupedMatmul::QuantMode::PERGROUP_MODE, true>(
                    Get<IDX_M_TILEIDX>(tileIdx), Get<IDX_N_TILEIDX>(tileIdx),
                    Get<IDX_M_TAIL_SPLIT_TILEIDX>(singleShape), Get<IDX_N_TAIL_SPLIT_TILEIDX>(singleShape),
                    AscendC::Std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>{
                        mBaseNormCnt_, mTailMain_, nBaseNormCnt_, nTailMain_});
            } else {
                blockOffset_ = coord.template GetQuantOffset<GroupedMatmul::QuantMode::PERGROUP_MODE>(
                    Get<IDX_M_TILEIDX>(tileIdx), Get<IDX_N_TILEIDX>(tileIdx),
                    Get<IDX_M_TAIL_SPLIT_TILEIDX>(singleShape), Get<IDX_N_TAIL_SPLIT_TILEIDX>(singleShape));
            }
        } else {
            blockOffset_ = coord.template GetQuantOffset<GroupedMatmul::QuantMode::PERBLOCK_MODE>(
                Get<IDX_M_TILEIDX>(tileIdx), Get<IDX_N_TILEIDX>(tileIdx), Get<IDX_M_TAIL_SPLIT_TILEIDX>(singleShape),
                Get<IDX_N_TAIL_SPLIT_TILEIDX>(singleShape));
        }

        Iterate(Get<MNK_M>(singleShape), Get<MNK_N>(singleShape));
    }
}

QBMM_PERTILE_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMmBatchPertile<QBMM_PERTILE_KERNEL_FUN_TEM_PARAMS>::Iterate(
    int64_t singleCoreM, int64_t singleCoreN)
{
    AscendC::Std::tuple<int64_t, int64_t, int64_t> blockShape{
        singleCoreM, singleCoreN, static_cast<int64_t>(Get<MNK_K>(problemShape_))};
    if ASCEND_IS_AIC {
        mmadOp_(blockShape, aGlobal_[Get<IDX_A_OFFSET>(blockOffset_)], bGlobal_[Get<IDX_B_OFFSET>(blockOffset_)]);
    }
    if ASCEND_IS_AIV {
        AscendC::Std::tuple<int64_t, int64_t, int64_t, int64_t> blockCoord{
            static_cast<int64_t>(Get<IDX_C_OFFSET>(blockOffset_)),
            static_cast<int64_t>(Get<IDX_X2SCALE_OFFSET>(blockOffset_)),
            static_cast<int64_t>(Get<IDX_X1SCALE_OFFSET>(blockOffset_)), 0L};
        epilogueOp_(blockShape, blockCoord);
    }
}

QBMM_PERTILE_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMmBatchPertile<QBMM_PERTILE_KERNEL_FUN_TEM_PARAMS>::UpdateMMGlobalAddr()
{
    if ASCEND_IS_AIC {
        aGlobal_.SetGlobalBuffer((__gm__ AType*)xTensorPtr_ + Get<IDX_A_OFFSET>(baseOffset_));
        bGlobal_.SetGlobalBuffer((__gm__ BType*)wTensorPtr_ + Get<IDX_B_OFFSET>(baseOffset_));
    }

    if ASCEND_IS_AIV {
        AscendC::Std::tuple<int64_t, int64_t, int64_t, int64_t> baseOffset{
            static_cast<int64_t>(Get<IDX_C_OFFSET>(baseOffset_)),
            static_cast<int64_t>(Get<IDX_X2SCALE_OFFSET>(baseOffset_)),
            static_cast<int64_t>(Get<IDX_X1SCALE_OFFSET>(baseOffset_)), 0L};
        epilogueOp_.UpdateGlobalAddr(baseOffset);
    }
}

QBMM_PERTILE_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMmBatchPertile<QBMM_PERTILE_KERNEL_FUN_TEM_PARAMS>::End()
{
    if ASCEND_IS_AIC {
        mmadOp_.End();
    }
}

} // namespace Kernel
} // namespace Gemm
} // namespace Cmct

#endif