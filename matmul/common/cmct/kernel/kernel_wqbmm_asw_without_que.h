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
 * \file kernel_wqbmm_asw_without_que.h
 * \brief wqbmmv2 ASW 路径的 kernel 层：初始化 GlobalTensor，实例化 scheduler + mmad，
 *        按 tile 循环驱动计算。
 */
#pragma once

#define ASCENDC_CUBE_ONLY
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "cmct/block/block_mmad_a16w8_fixpipe_antiquant.h"
#include "cmct/block/block_scheduler_wqbmm_asw.h"
#include "cmct/policy/dispatch_policy.h"
#include "cmct/utils/common_utils.h"
#include "cmct/utils/coord_utils.h"
#include "cmct/utils/device_utils.h"
#include "cmct/utils/tuple_utils.h"

namespace Cmct::Gemm::Kernel {

template <class ProblemShape_, class BlockMmad_, class BlockScheduler_, class L1TileShape_, class L0TileShape_>
class WqbmmKernelMatmul {
public:
    using ProblemShape = ProblemShape_;
    using BlockMmadOp = BlockMmad_;
    using BlockScheduler = BlockScheduler_;

    static constexpr bool transA = BlockMmadOp::transA;
    static constexpr bool transB = BlockMmadOp::transB;
    // 经公共 Selector 两段式挂钩解析调度器本体
    using BlockSchedulerOp = typename Block::BlockSchedulerSelector<ProblemShape, L1TileShape_, L0TileShape_,
                                                                    BlockScheduler, transA, transB>::SchedulerOp;
    using AType = typename BlockMmadOp::A_T;
    using BType = typename BlockMmadOp::B_T;
    using CType = typename BlockMmadOp::C_T;
    using BiasType = typename BlockMmadOp::Bias_T;

    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using TupleL1L0Shape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;

    // GM 地址参数
    struct Arguments {
        GM_ADDR aGM{nullptr};
        GM_ADDR bGM{nullptr};
        GM_ADDR cGM{nullptr};
        GM_ADDR biasGM{nullptr};
        GM_ADDR scaleGM{nullptr}; // antiquantScale：per-tensor 指向 8 字节标量，per-channel 指向 uint64 向量
    };

    // TilingData 由调用方（算子入口）指定，scheduler 经模板 Params 以 duck typing 读取字段
    template <class TilingData>
    struct Params {
        ProblemShape problemShape;
        Arguments mmadParams;
        typename BlockSchedulerOp::template Params<TilingData> schParams;
    };

    template <class TilingData>
    __aicore__ inline void operator()(const Params<TilingData>& params)
    {
        if ASCEND_IS_AIV {
            return;
        }
        // 空 shape 直接返回，避免 scheduler 构造时除零
        if (params.problemShape.m == 0 || params.problemShape.n == 0 || params.problemShape.k == 0) {
            return;
        }
        int64_t curBlockIdx = AscendC::GetBlockIdx();
        int64_t blockNum = AscendC::GetBlockNum();
        // Init GlobalTensor
        problemShape_ = {params.problemShape.m, params.problemShape.n, params.problemShape.k, params.problemShape.b};
        aGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ AType*>(params.mmadParams.aGM));
        bGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ BType*>(params.mmadParams.bGM));
        cGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ CType*>(params.mmadParams.cGM));
        scaleGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(params.mmadParams.scaleGM));
        isBias_ = (params.mmadParams.biasGM != nullptr);
        if (isBias_) {
            biasGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ BiasType*>(params.mmadParams.biasGM));
        }

        BlockMmadOp blockMmadOp;
        BlockSchedulerOp bs(params.problemShape, curBlockIdx, blockNum, params.schParams);
        SetL2CacheHint(bs.GetAL2CacheDisable(), bs.GetBL2CacheDisable(), aGlobal_, bGlobal_);
        int64_t tileNum = bs.GetTileNum();
        TupleShape tileL1 = bs.GetTileL1Shape();
        TupleShape tileL0 = bs.GetTileL0Shape();
        int64_t realBlockNum = bs.GetBlockNum(blockNum);
        if (curBlockIdx >= realBlockNum) {
            return;
        }

        AscendC::SetMMLayoutTransform(true);
        // shiftValue/l1BufferNum/l0cDB 是 block mmad 内部参数，直接从 tiling 读取，不经 scheduler 转发
        const TilingData* tilingData = params.schParams.tilingData;
        blockMmadOp.Init(problemShape_, tileL1, tileL0, isBias_, tilingData->l1BufferNum, tilingData->l0cDB > 1,
                         tilingData->shiftValue);
        if constexpr (BlockMmadOp::antiQuantMode == WqbmmAntiQuantMode::PER_TENSOR) {
            blockMmadOp.CacheQuantScalar(LoadQuantScalarFromGm(params.mmadParams.scaleGM));
        }
        for (int64_t tileIdx = curBlockIdx; tileIdx < tileNum; tileIdx += blockNum) {
            // ASW stepM/stepN 恒为 1，L1 tile 内不再迭代 m/n
            TupleL1L0Shape blockShape = bs.GetBlockShape(tileIdx, 0, 0);
            if (Get<0>(blockShape) <= 0 || Get<1>(blockShape) <= 0) {
                AscendC::SetMMLayoutTransform(false);
                return;
            }
            BlockCoord blockCoord = bs.GetBlockCoord(tileIdx);
            // {0, 1, 1} = sliceM 0 / srcNdStride 1 / innerBatch 1，本路径不存在非连续 ND 与 splitK
            auto blockOffset = GetOffsetForNDLayout(blockCoord, problemShape_, transA, transB, isBias_, {0, 1, 1},
                                                    blockShape, false);
            blockMmadOp(cGlobal_[Get<2>(blockOffset)], aGlobal_[Get<0>(blockOffset)], bGlobal_[Get<1>(blockOffset)],
                        biasGlobal_[Get<3>(blockOffset)], scaleGlobal_[Get<1>(blockCoord)], blockShape,
                        tileIdx == curBlockIdx);
        }
        AscendC::SetMMLayoutTransform(false);
    }

private:
    AscendC::GlobalTensor<AType> aGlobal_;
    AscendC::GlobalTensor<BType> bGlobal_;
    AscendC::GlobalTensor<CType> cGlobal_;
    AscendC::GlobalTensor<BiasType> biasGlobal_;
    AscendC::GlobalTensor<uint64_t> scaleGlobal_;
    TupleShape problemShape_{};
    bool isBias_ = false;
};

} // namespace Cmct::Gemm::Kernel
