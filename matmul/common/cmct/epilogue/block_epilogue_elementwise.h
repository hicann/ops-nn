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
 * \file block_epilogue_elementwise.h
 * \brief
 */

#pragma once
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "../utils/common_utils.h"
#include "../utils/device_utils.h"
#include "fusion/default_fusion_op.h"
#include "fusion/fusion_add.h"
#include "fusion/fusion_mul.h"
#include "fusion/fusion_gelu.h"
#include "../utils/status_utils.h"

namespace Cmct {
namespace Gemm {
namespace Block {

template <typename L0TileShape_, typename DataTypeOut_, typename DataTypeIn_, typename FusionOp_>
class BlockEpilogueElementwise {
public:
    using FusionArguments = typename FusionOp_::Arguments;
    using FusionParams = typename FusionOp_::Params;

    __aicore__ inline BlockEpilogueElementwise() {}

    struct Arguments {
        GM_ADDR outGmAddr{nullptr};
        GM_ADDR workspaceGmAddr{nullptr};
        FusionArguments fusionArgs{};
    };

    struct Params {
        GM_ADDR outGmAddr{nullptr};
        GM_ADDR workspaceGmAddr{nullptr};
        FusionParams fusionParams{};
    };

    using DataTypeOut = DataTypeOut_;
    using DataTypeIn = DataTypeIn_;
    using FusionOp = FusionOp_;
    static constexpr bool useWorkspaceForC =
        !AscendC::IsSameType<DataTypeIn, DataTypeOut>::value && FusionOp::kSupportsWorkspaceCopy;
    static constexpr int64_t WORKSPACE_N_REM_THRESHOLD = 8;
    static constexpr uint16_t ZERO_FLAG = 0;
    static constexpr int64_t l0M = GetIntegralConstant<MNK_M, L0TileShape_>();
    static constexpr int64_t l0N = GetIntegralConstant<MNK_N, L0TileShape_>();
    // shape
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;

    // GM ADDR
    AscendC::LocalTensor<DataTypeIn> cLocal_{AscendC::TPosition::VECIN, 0, AscendC::TOTAL_UB_SIZE / sizeof(DataTypeIn)};
    AscendC::LocalTensor<DataTypeIn> cLocalTmp_{AscendC::TPosition::VECIN, 0,
                                                AscendC::TOTAL_UB_SIZE / sizeof(DataTypeIn)};
    AscendC::GlobalTensor<DataTypeOut> outputGlobal_;
    AscendC::GlobalTensor<DataTypeIn> workspaceGlobal_;
    // vector核一次最多计算多少个元素
    int64_t stageSize_ = 0;
    // attribute
    FusionOp fusionOp_;
    ProblemShape problemShape_;
    int64_t workspaceSlotM_{0};
    int64_t workspaceSlotN_{0};

    __aicore__ inline void Init(Params const& params, int64_t l1M, int64_t l1N, ProblemShape& problemShape,
                                int64_t workspaceSlotM = 0, int64_t workspaceSlotN = 0)
    {
        outputGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ DataTypeOut*>(params.outGmAddr));
        problemShape_ = problemShape;
        workspaceSlotM_ = workspaceSlotM > 0 ? workspaceSlotM : l1M * AscendC::GetTaskRation();
        workspaceSlotN_ = workspaceSlotN > 0 ? workspaceSlotN : l1N;
        int64_t l1NAlign = AlignBlock<DataTypeOut>(l1N);
        int64_t ubOffset = l1M * l1NAlign;
        if constexpr (useWorkspaceForC) {
            workspaceGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ DataTypeIn*>(params.workspaceGmAddr));
        }
        fusionOp_.Init(params.fusionParams, cLocal_, l1M, l1NAlign, ubOffset, stageSize_);
        cLocalTmp_ = cLocal_[ubOffset].template ReinterpretCast<DataTypeIn>();
    }

    __aicore__ inline void LoadCFromWorkspace(int64_t wsBase, int64_t copyElems)
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(ZERO_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(ZERO_FLAG);
        AscendC::DataCopyExtParams loadParams{1, static_cast<uint32_t>(copyElems * sizeof(DataTypeIn)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<DataTypeIn> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(cLocal_, workspaceGlobal_[wsBase], loadParams, padParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ZERO_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ZERO_FLAG);
    }

    __aicore__ inline void LoadCFromWorkspacePadded(
        int64_t wsBase, int64_t rowCount, int64_t rowLen, int64_t workspaceRowStride, int64_t blockShapeNAlign)
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(ZERO_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(ZERO_FLAG);
        int64_t validRowBytes = rowLen * static_cast<int64_t>(sizeof(DataTypeIn));
        int64_t alignedValidRowBytes = CeilAlign(validRowBytes, static_cast<int64_t>(UB_ALIGN_SIZE));
        int64_t paddedRowBytes = blockShapeNAlign * static_cast<int64_t>(sizeof(DataTypeIn));
        uint32_t dstStride = static_cast<uint32_t>(
            paddedRowBytes > alignedValidRowBytes ? (paddedRowBytes - alignedValidRowBytes) / UB_ALIGN_SIZE : 0);
        AscendC::DataCopyExtParams loadParams{
            static_cast<uint16_t>(rowCount), static_cast<uint32_t>(rowLen * sizeof(DataTypeIn)),
            static_cast<uint32_t>((workspaceRowStride - rowLen) * sizeof(DataTypeIn)), dstStride, 0};
        AscendC::DataCopyPadExtParams<DataTypeIn> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(cLocal_, workspaceGlobal_[wsBase], loadParams, padParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ZERO_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ZERO_FLAG);
    }

    __aicore__ inline void RunFromWorkspaceContiguous(
        int64_t wsBase, int64_t baseOffset, int64_t blockShapeM, int64_t blockShapeN, int64_t N)
    {
        int64_t inputSize = blockShapeM * blockShapeN;
        int64_t stageSize = AscendC::Std::min(stageSize_, inputSize) / blockShapeN * blockShapeN;
        ASCENDC_ASSERT(stageSize > 0, {
            KERNEL_LOG(KERNEL_EORROR, "RunFromWorkspaceContiguous stageSize limit %ld, %ld, %ld!",
                       stageSize_, blockShapeM, blockShapeN);
        });
        int64_t stageOffset = 0;
        while (stageOffset < inputSize) {
            stageSize = AscendC::Std::min(stageSize, inputSize - stageOffset);
            int64_t offset = baseOffset + stageOffset / blockShapeN * N;
            LoadCFromWorkspace(wsBase + stageOffset, stageSize);
            fusionOp_(cLocal_, cLocalTmp_, offset, blockShapeM, blockShapeN, N, stageSize, true);
            AscendC::LocalTensor<DataTypeOut> outputLocal = cLocal_.template ReinterpretCast<DataTypeOut>();
            if constexpr (!AscendC::IsSameType<DataTypeOut, DataTypeIn>::value) {
                Cast(outputLocal, cLocalTmp_, AscendC::RoundMode::CAST_RINT, stageSize);
                AscendC::PipeBarrier<PIPE_V>();
            } else {
                outputLocal = cLocalTmp_;
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ZERO_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ZERO_FLAG);
            AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(stageSize * sizeof(DataTypeOut)), 0, 0, 0};
            AscendC::DataCopyPad<DataTypeOut>(outputGlobal_[offset], outputLocal, copyParams);
            stageOffset += stageSize;
        }
    }

    __aicore__ inline void RunFromWorkspacePadded(
        int64_t wsBase, int64_t baseOffset, int64_t blockShapeM, int64_t blockShapeN, int64_t N,
        int64_t workspaceRowStride)
    {
        int64_t blockShapeNAlign = AlignBlock<DataTypeOut>(blockShapeN); // 对齐16
        int64_t inputSize = blockShapeM * blockShapeNAlign;
        int64_t stageSize = AscendC::Std::min(stageSize_, inputSize) / blockShapeNAlign * blockShapeNAlign;
        ASCENDC_ASSERT(stageSize > 0, {
            KERNEL_LOG(KERNEL_EORROR, "RunFromWorkspacePadded stageSize limit %ld, %ld, %ld!",
                       stageSize_, blockShapeM, blockShapeN);
        });
        int64_t stageOffset = 0;
        while (stageOffset < inputSize) {
            stageSize = AscendC::Std::min(stageSize, inputSize - stageOffset);
            int64_t offset = baseOffset + stageOffset / blockShapeNAlign * N;
            int64_t rowOffset = stageOffset / blockShapeNAlign;
            int64_t rowCount = stageSize / blockShapeNAlign;
            LoadCFromWorkspacePadded(
                wsBase + rowOffset * workspaceRowStride, rowCount, blockShapeN, workspaceRowStride, blockShapeNAlign);
            fusionOp_(cLocal_, cLocalTmp_, offset, blockShapeM, blockShapeN, N, stageSize);
            AscendC::LocalTensor<DataTypeOut> outputLocal = cLocal_.template ReinterpretCast<DataTypeOut>();
            if constexpr (!AscendC::IsSameType<DataTypeOut, DataTypeIn>::value) {
                Cast(outputLocal, cLocalTmp_, AscendC::RoundMode::CAST_RINT, stageSize);
                AscendC::PipeBarrier<PIPE_V>();
            } else {
                outputLocal = cLocalTmp_;
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ZERO_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ZERO_FLAG);
            AscendC::DataCopyExtParams copyParams{
                static_cast<uint16_t>(stageSize / blockShapeNAlign),
                static_cast<uint32_t>(blockShapeN * sizeof(DataTypeOut)), 0,
                static_cast<uint32_t>((N - blockShapeN) * sizeof(DataTypeOut)), 0};
            AscendC::DataCopyPad<DataTypeOut>(outputGlobal_[offset], outputLocal, copyParams);
            stageOffset += stageSize;
        }
    }

    __aicore__ inline void RunFromWorkspace(BlockShape const& blockShape, int64_t blockShapeM,
                                            int64_t halfBlockShapeM, int64_t blockShapeN, int64_t dstOffset,
                                            int64_t N, int64_t flagId)
    {
        int64_t workspaceRowStride = workspaceSlotN_;
        int64_t wsSlotBase = (AscendC::GetBlockIdx() / AscendC::GetTaskRation()) * workspaceSlotM_ * workspaceSlotN_;
        int64_t wsBase = wsSlotBase + AscendC::GetSubBlockIdx() * halfBlockShapeM * workspaceRowStride;
        int64_t baseOffset = dstOffset + AscendC::GetSubBlockIdx() * halfBlockShapeM * N;
        AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE2>(0);
        if (N <= blockShapeN && workspaceRowStride == blockShapeN) {
            RunFromWorkspaceContiguous(wsBase, baseOffset, blockShapeM, blockShapeN, N);
        } else {
            RunFromWorkspacePadded(wsBase, baseOffset, blockShapeM, blockShapeN, N, workspaceRowStride);
        }
        AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(flagId);
    }

    __aicore__ inline AscendC::LocalTensor<DataTypeOut> DoFusionAndCast(int64_t stageOffset, int64_t offset,
                                                                        int64_t blockShapeM, int64_t blockShapeN,
                                                                        int64_t N, int64_t stageSize)
    {
        // Do fusionOp{add, mul, gelu} in ub:  (cLocal_[stageOffset], x3 or None) -> cLocal_
        fusionOp_(cLocal_[stageOffset], cLocalTmp_, offset, blockShapeM, blockShapeN, N, stageSize);
        AscendC::LocalTensor<DataTypeOut> outputLocal = cLocal_[stageOffset].template ReinterpretCast<DataTypeOut>();
        if constexpr (AscendC::IsSameType<DataTypeOut, DataTypeIn>::value) {
            outputLocal = cLocalTmp_;
        } else {
            Cast(outputLocal, cLocalTmp_, AscendC::RoundMode::CAST_RINT, stageSize);
            AscendC::PipeBarrier<PIPE_V>();
        }
        return outputLocal;
    }

    __aicore__ inline void Run(int64_t blockShapeM, int64_t halfBlockShapeM, int64_t blockShapeN,
                               int64_t N, int64_t dstOffset, int64_t flagId = 5)
    {
        int64_t blockShapeNAlign = AlignBlock<DataTypeOut>(blockShapeN); // 对齐16
        int64_t inputSize = blockShapeM * blockShapeNAlign;

        // 一次计算最多取Min(baseM/2 * baseN, stageSize_)
        int64_t stageSize = AscendC::Std::min(stageSize_, inputSize) / blockShapeNAlign * blockShapeNAlign;
        ASCENDC_ASSERT(stageSize > 0, {
            KERNEL_LOG(KERNEL_EORROR, "stageSize size limit %ld, %ld, %ld!", stageSize_, blockShapeM, blockShapeN);
        });
        int64_t loop = 0;
        int64_t stageOffset = 0;
        while (stageOffset < inputSize) {
            int64_t offset = dstOffset + loop * stageSize / blockShapeNAlign * N;
            // Aiv1需要多偏移aiv0所处理的数据
            offset += AscendC::GetSubBlockIdx() * halfBlockShapeM * N;
            stageSize = AscendC::Std::min(stageSize, inputSize - stageOffset);
            AscendC::LocalTensor<DataTypeOut> outputLocal = DoFusionAndCast(stageOffset, offset, blockShapeM,
                                                                            blockShapeN, N, stageSize);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ZERO_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ZERO_FLAG);
            // copy result from ub to gm
            AscendC::DataCopyExtParams copyParams{static_cast<uint16_t>(stageSize / blockShapeNAlign),
                                                  static_cast<uint32_t>(blockShapeN * sizeof(DataTypeOut)), 0,
                                                  static_cast<uint32_t>((N - blockShapeN) * sizeof(DataTypeOut)), 0};
            AscendC::DataCopyPad<DataTypeOut>(outputGlobal_[offset], outputLocal, copyParams);
            stageOffset += stageSize;
            loop++;
        }
        AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(flagId);
    }

    // GetTensor from ub from current AIV
    __aicore__ inline auto GetTensor() { return cLocal_; }

    __aicore__ inline void operator()(BlockShape const& blockShape, int64_t dstOffset = 0, int64_t flagId = 5)
    {
        // 默认1-2不再基于splitM区分, aiv 0~1分别搬运blockShapeM/2
        int64_t blockShapeM = Get<0>(blockShape);
        int64_t halfBlockShapeM = Cmct::Gemm::CeilDiv(blockShapeM, AscendC::GetTaskRation());
        blockShapeM = ((static_cast<uint64_t>(blockShapeM) & 1UL) > 0UL) ?
                          (halfBlockShapeM - AscendC::GetSubBlockIdx()) :
                          halfBlockShapeM;
        int64_t blockShapeN = Get<1>(blockShape);
        int64_t N = Get<MNK_N>(problemShape_);
        if constexpr (useWorkspaceForC) {
            if (((blockShapeN & 0xF) > 0) && ((blockShapeN & 0xF) <= WORKSPACE_N_REM_THRESHOLD)) {
                RunFromWorkspace(blockShape, blockShapeM, halfBlockShapeM, blockShapeN, dstOffset, N, flagId);
                return;
            }
        }
        Run(blockShapeM, halfBlockShapeM, blockShapeN, N, dstOffset, flagId);
        return;
    }

    // static init
    __host_aicore__ static Params InitParams(Arguments const& args, GM_ADDR x3Gm)
    {
        FusionParams fusionParams = FusionOp::InitParams(args.fusionArgs, x3Gm);
        Params params = {args.outGmAddr, args.workspaceGmAddr, fusionParams};
        return params;
    }

    __host_aicore__ static size_t GetWorkspaceSize(int64_t blockNum, int64_t l1M, int64_t l1N)
    {
        // only quant kernel need workspace
        return 0;
    }

    __host_aicore__ static Status CanImplement(Arguments const& args)
    {
        if (l0M * l0N * sizeof(DataTypeIn_) > AscendC::TOTAL_UB_SIZE) {
            return Status::l1L0ErrorExceedsLimit;
        }
        return Status::success;
    }
};

} // namespace Block
} // namespace Gemm
} // namespace Cmct
#endif // EPILOGUE_BLOCK_EPILOGUE_H
