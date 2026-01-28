/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file batch_norm_grad_tiling_infer.cpp
 * \brief
 */

#include <cstdint>
#include "batch_norm_grad_tiling.h"
#include "tiling_base/tiling_templates_registry.h"
#include "batch_norm_grad_tiling_infer_base.h"

using namespace ge;

namespace optiling
{
static constexpr int64_t TWO = 2;


static constexpr uint64_t BNG_RAR_RECOMPUTE_SPLIT_R1_INFER = 1;
static constexpr uint64_t BNG_RAR_RECOMPUTE_SPLIT_R0_INFER = 2;
static constexpr int ULONG_BIT_LEN = 64;
static constexpr int64_t BNG_PER_CORE_PROCESS_MIN_UB_SIZE = 1024;


class BatchNormGradInferTiling : public BatchNormGradInferBase
{
public:
    explicit BatchNormGradInferTiling(gert::TilingContext* context) : BatchNormGradInferBase(context)
    {
    }
    ~BatchNormGradInferTiling() override = default;

protected:
    bool IsCapable() override
    {
        // R1AR0, R0 == 1时走RA模板
        OP_CHECK_IF(
            r0Dim == 1,
            OP_LOGE(context_, "BatchNormGradInferTiling RAR template is not capable, fused shape: (%ld, %ld, %ld)",
                    r1Dim, aDim, r0Dim),
                    return false);

        CalcBasicInfo();

        OP_LOGD(context_, "BatchNormGradInferTiling RAR template is capable, fused shape: (%ld, %ld, %ld)", r1Dim,
                aDim, r0Dim);
        return true;
    }

    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus PostTiling() override;
private:
    ge::graphStatus DoOpTilingForStage0();
    ge::graphStatus DoOpTilingForStage1();
    void DoRecomputeTilingSplitR1();
    void DoRecomputeTilingSplitR0();
    void DoBinaryAddTiling(BatchNormGradBinaryAddTilingData &tilingData, int64_t quotient);

    int64_t onceProcUbNeed{ 0 }; // A轴上执行一次，需要占用的ub大小
    uint64_t binaryAddBufSize{ 0 };
    uint64_t subTilingKey{ 0 };
    int64_t r1Factor{ 0 };
    int64_t r0Factor{ 0 };
    int64_t ubRDimLoopNum{ 0 };
    int64_t ubRDimFactor{ 0 };
    int64_t ubRDimFactorAlign{ 0 };
    int64_t ubRDimTailFactor{ 0 };
    int64_t ubRDimTailFactorAlign{ 0 };
    int64_t ubRDimTail{ 0 };
    int64_t ubRDimTailLoopNum{ 0 };
    int64_t ubRDimTailTail{ 0 };
    int64_t ubRDimTailTailFactor{ 0 };
    int64_t ubRDimTailTailFactorAlign{ 0 };
    int64_t ubRDimTailTailLoopNum{ 0 };
    BatchNormGradInferTilingData tilingData_;
};

ge::graphStatus BatchNormGradInferTiling::DoOpTilingForStage0() {
    // 切分A、B基本块， （B0,A,B1） -- >(r1Outer*aOuter*r0Outer, B0inner*Ainner*B1innerA(TileBase))
    int64_t aInner = 1;
    int64_t ubBufferSize =
        (aicoreParams_.ubSize / DOUBLE_BUFFER - (bytesPerWeight_ + bytesPerRunningVar_) * aInner * aTileBase_) /
        bytesPerDy_ / INPUT_OUTPUT_NUM;

    // 先按照B切分，再切A
    // UB可载入最大tile块数
    int64_t factorMax = ubBufferSize / aTileBase_;

    // 默认策略: 先按照B0, B1把UB切满
    int64_t b1FactorMax = Ops::Base::CeilDiv(r0Dim, aTileBase_);    // 处理 b1 需要多少 vf
    int64_t b1Inner = factorMax <= b1FactorMax ? factorMax : b1FactorMax;   // b1 vf 内循环处理多少个次
    int64_t r0Outer = Ops::Base::CeilDiv(r0Dim, b1Inner * aTileBase_);  // b1 外循环

    factorMax = factorMax / b1Inner;
    int64_t b0FactorMax = r1Dim;
    int64_t r1Inner = factorMax <= b0FactorMax ? factorMax : b0FactorMax;
    int64_t r1Outer = Ops::Base::CeilDiv(r1Dim, r1Inner);

    factorMax = factorMax / r1Inner;
    int64_t aFactorMax = aDim;
    aInner = factorMax <= aFactorMax ? factorMax : aFactorMax;
    int64_t aOuter = Ops::Base::CeilDiv(aDim, aInner);

    int64_t totalTiles = r1Outer * aOuter * r0Outer;
    int64_t tilesPerCore = Ops::Base::CeilDiv(totalTiles, static_cast<int64_t>(aicoreParams_.blockDim));
    usedCoreNums_ = Ops::Base::CeilDiv(totalTiles, tilesPerCore);

    int64_t tileBlockR1Tail = r1Dim - r1Inner * (r1Outer - 1);
    int64_t tileBlockATail = aDim - aInner * (aOuter - 1);
    int64_t tileBlockR0Tail = r0Dim - b1Inner * aTileBase_ * (r0Outer - 1);

    tilingData_.baseTilingData.set_totalTiles(totalTiles);
    tilingData_.baseTilingData.set_tilesPerCore(tilesPerCore);
    tilingData_.baseTilingData.set_usedCoreNums(usedCoreNums_);
    tilingData_.baseTilingData.set_r1Dim(r1Dim);
    tilingData_.baseTilingData.set_aDim(aDim);
    tilingData_.baseTilingData.set_r0Dim(r0Dim);
    tilingData_.baseTilingData.set_r1Outer(r1Outer);
    tilingData_.baseTilingData.set_aOuter(aOuter);
    tilingData_.baseTilingData.set_r0Outer(r0Outer);
    tilingData_.baseTilingData.set_tileBlockR1Len(r1Inner);
    tilingData_.baseTilingData.set_tileBlockR1Tail(tileBlockR1Tail);
    tilingData_.baseTilingData.set_tileBlockALen(aInner);
    tilingData_.baseTilingData.set_tileBlockATail(tileBlockATail);
    tilingData_.baseTilingData.set_tileBlockR0Len(b1Inner * aTileBase_);
    tilingData_.baseTilingData.set_tileBlockR0Tail(tileBlockR0Tail);
    tilingData_.baseTilingData.set_tileBlockAPaddingNum(0);
    tilingData_.baseTilingData.set_epsilon(epsilon_);
    return ge::GRAPH_SUCCESS;
}

void BatchNormGradInferTiling::DoRecomputeTilingSplitR1()
{
    if (Ops::Base::CeilDiv(static_cast<int64_t>(r1Dim * r0Dim * sizeof(float)), blockSize_) * blockSize_ <= static_cast<int64_t>(binaryAddBufSize)) {
        ubRDimFactor = r1Dim * r0Dim;
        ubRDimFactorAlign = Ops::Base::CeilDiv(static_cast<int64_t>(ubRDimFactor * sizeof(float)), blockSize_) * blockSize_ / sizeof(float);
        tilingData_.set_ubRDimFactor(ubRDimFactor);
        tilingData_.set_ubRDimFactorAlign(ubRDimFactorAlign);
        tilingData_.set_ubRDimLoopNum(1);
        tilingData_.set_ubRDimTail(0);
        tilingData_.set_ubRDimTailFactor(0);
        tilingData_.set_ubRDimTailFactorAlign(0);
        tilingData_.set_ubRDimTailLoopNum(0);
        tilingData_.set_ubRDimTailTail(0);
        tilingData_.set_ubRDimTailTailFactor(0);
        tilingData_.set_ubRDimTailTailFactorAlign(0);
        tilingData_.set_ubRDimTailTailLoopNum(0);
        return;
    }
    r1Factor = (binaryAddBufSize / (r0Dim * sizeof(float))) & (~1L);                    // 计算 ub 最大能装多少 r1          (r1, A, r0) -> (r1Dim / r1Factor, A, r1Factor * r0)
    ubRDimLoopNum = (1L << (ULONG_BIT_LEN - 1 - __builtin_clzl(r1Dim / r1Factor)));     // 小于等于r1Dim / r1Factor 的最大2次幂
    ubRDimFactor = r1Factor * r0Dim;
    ubRDimFactorAlign = Ops::Base::CeilDiv(static_cast<int64_t>(ubRDimFactor * sizeof(float)), blockSize_) * blockSize_ / sizeof(float);
    ubRDimTail = r1Dim * r0Dim - ubRDimFactor * ubRDimLoopNum;
    ubRDimTailFactor = ubRDimFactorAlign * sizeof(float) > (binaryAddBufSize / TWO) ? ubRDimFactor / TWO : ubRDimFactor;
    ubRDimTailFactorAlign = Ops::Base::CeilDiv(static_cast<int64_t>(ubRDimTailFactor * sizeof(float)), blockSize_) * blockSize_ / sizeof(float);
    ubRDimTailLoopNum = ubRDimFactorAlign * sizeof(float) > (binaryAddBufSize / TWO) ? ubRDimTail / ubRDimFactor * TWO :
                                                                                       ubRDimTail / ubRDimFactor;
    ubRDimTailTail = ubRDimTail - ubRDimTailLoopNum * ubRDimTailFactor;
    ubRDimTailTailFactor = ubRDimTailFactor;
    ubRDimTailTailFactorAlign = ubRDimTailFactorAlign;
    ubRDimTailTailLoopNum = ubRDimFactorAlign * sizeof(float) > (binaryAddBufSize / TWO) ? TWO : 1;

    tilingData_.set_ubRDimFactor(ubRDimFactor);
    tilingData_.set_ubRDimFactorAlign(ubRDimFactorAlign);
    tilingData_.set_ubRDimLoopNum(ubRDimLoopNum);
    tilingData_.set_ubRDimTail(ubRDimTail);
    tilingData_.set_ubRDimTailFactor(ubRDimTailFactor);
    tilingData_.set_ubRDimTailFactorAlign(ubRDimTailFactorAlign);
    tilingData_.set_ubRDimTailLoopNum(ubRDimTailLoopNum);
    tilingData_.set_ubRDimTailTail(ubRDimTailTail);
    tilingData_.set_ubRDimTailTailFactor(ubRDimTailTailFactor);
    tilingData_.set_ubRDimTailTailFactorAlign(ubRDimTailTailFactorAlign);
    tilingData_.set_ubRDimTailTailLoopNum(ubRDimTailTailLoopNum);
}

void BatchNormGradInferTiling::DoRecomputeTilingSplitR0()
{
    r0Factor = r0Dim * sizeof(float) / binaryAddBufSize;
    ubRDimLoopNum = r0Factor <= 0 ? 1L : std::max(1L, (1L << (ULONG_BIT_LEN - 1 - __builtin_clzl(r0Factor))));
    ubRDimFactor = std::min(static_cast<uint64_t>(r0Dim), binaryAddBufSize / sizeof(float));
    ubRDimFactorAlign = Ops::Base::CeilDiv(static_cast<int64_t>(ubRDimFactor * sizeof(float)), blockSize_) * blockSize_ / sizeof(float);
    ubRDimTail = r0Dim - ubRDimFactor * ubRDimLoopNum;
    // ubRDimFactor / 2 * 2 may be less than ubRDimFactor
    ubRDimTailFactor = ubRDimFactorAlign * sizeof(float) > (binaryAddBufSize / TWO) ? (ubRDimFactor + TWO - 1) / TWO :
                       ubRDimFactor;
    ubRDimTailFactorAlign = Ops::Base::CeilDiv(static_cast<int64_t>(ubRDimTailFactor * sizeof(float)), blockSize_) * blockSize_ / sizeof(float);
    ubRDimTailLoopNum = ubRDimFactorAlign * sizeof(float) > (binaryAddBufSize / TWO) ? ubRDimTail / ubRDimFactor * TWO :
                                                                                       ubRDimTail / ubRDimFactor;
    ubRDimTailTail = ubRDimTail - ubRDimTailLoopNum * ubRDimTailFactor;
    ubRDimTailTailFactor = ubRDimTailFactor;
    ubRDimTailTailFactorAlign = ubRDimTailFactorAlign;
    ubRDimTailTailLoopNum = ubRDimFactorAlign * sizeof(float) > (binaryAddBufSize / TWO) ? TWO : 1;

    tilingData_.set_ubRDimFactor(ubRDimFactor);
    tilingData_.set_ubRDimFactorAlign(ubRDimFactorAlign);
    tilingData_.set_ubRDimLoopNum(ubRDimLoopNum);
    tilingData_.set_ubRDimTail(ubRDimTail);
    tilingData_.set_ubRDimTailFactor(ubRDimTailFactor);
    tilingData_.set_ubRDimTailFactorAlign(ubRDimTailFactorAlign);
    tilingData_.set_ubRDimTailLoopNum(ubRDimTailLoopNum);
    tilingData_.set_ubRDimTailTail(ubRDimTailTail);
    tilingData_.set_ubRDimTailTailFactor(ubRDimTailTailFactor);
    tilingData_.set_ubRDimTailTailFactorAlign(ubRDimTailTailFactorAlign);
    tilingData_.set_ubRDimTailTailLoopNum(ubRDimTailTailLoopNum);
}

void BatchNormGradInferTiling::DoBinaryAddTiling(BatchNormGradBinaryAddTilingData &tilingData, int64_t quotient)
{
    tilingData.set_binaryAddQuotient(quotient);
    int64_t vcaddNum = quotient / vlFp32_;
    if (vcaddNum <= vlFp32_) {
        tilingData.set_binaryAddk(0);
        tilingData.set_binaryAddLastNum(vcaddNum);
    } else {
        int64_t binaryAddNum = vcaddNum / vlFp32_;
        tilingData.set_binaryAddk(__builtin_ctzl(binaryAddNum));
        tilingData.set_binaryAddLastNum(vlFp32_);
    }
}


ge::graphStatus BatchNormGradInferTiling::DoOpTilingForStage1() {
    // UB内二分累加buffer按32KB的float数算，占512字节
    // dgamma和dbeta的UB间二分累加buffer各占3KB
    // mean rstd gamma各占128字节，共512字节
    // 总共7KB
    constexpr uint64_t extraUbSize = 7 * 1024 + 512;
    binaryAddBufSize = (aicoreParams_.ubSize - extraUbSize) / TWO / DOUBLE_BUFFER;

    // R0不可全载，核内切R0
    subTilingKey = BNG_RAR_RECOMPUTE_SPLIT_R0_INFER;
    if (Ops::Base::CeilDiv(static_cast<int64_t>(r0Dim * sizeof(float) * TWO), blockSize_) * blockSize_ <= static_cast<int64_t>(binaryAddBufSize)) {
        // R0可全载，核内切R1
        subTilingKey = BNG_RAR_RECOMPUTE_SPLIT_R1_INFER;
    }

    onceProcUbNeed = INT64_MAX;
    int64_t minADimPerCore = std::max(1L, Ops::Base::CeilDiv(BNG_PER_CORE_PROCESS_MIN_UB_SIZE, onceProcUbNeed));
    int64_t blockNum = aicoreParams_.blockDim;
    if (static_cast<int64_t>(minADimPerCore * aicoreParams_.blockDim) > aDim) {
        // 每个核最少要处理 BNG_PER_CORE_PROCESS_MIN_UB_SIZE, minADimPerCore向上取整不可能是0
        blockNum = aDim / minADimPerCore;
        if (blockNum == 0) {
            blockNum = 1;
        }
    }
    OP_CHECK_IF(blockNum == 0, OP_LOGE(context_, "block num is 0, failed."), return ge::GRAPH_FAILED);

    int64_t tailBlockNum = aDim % blockNum;
    int64_t formerBlockDim = aDim / blockNum;
    int64_t tailBlockDim = formerBlockDim + 1;

    tilingData_.set_blockNum(blockNum);
    tilingData_.set_tailBlockNum(tailBlockNum);
    tilingData_.set_formerBlockDim(formerBlockDim);
    tilingData_.set_tailBlockDim(tailBlockDim);

    if (subTilingKey == BNG_RAR_RECOMPUTE_SPLIT_R1_INFER) {
        DoRecomputeTilingSplitR1();
    } else {
        DoRecomputeTilingSplitR0();
    }
    int64_t tailBinAddQuotient = ubRDimTailFactor == 0 ? 1 : (1L << (ULONG_BIT_LEN - 1 - __builtin_clzl(ubRDimTailFactor)));
    int64_t generalBinAddQuotient = ubRDimFactor == 0 ? 1 : (1L << (ULONG_BIT_LEN - 1 - __builtin_clzl(ubRDimFactor)));

    DoBinaryAddTiling(tilingData_.generalBinAddTilingData, generalBinAddQuotient);
    DoBinaryAddTiling(tilingData_.tailBinAddTilingData, tailBinAddQuotient);

    tilingData_.set_enableDx(static_cast<int32_t>(enableDx));
    tilingData_.set_enableDgamma(static_cast<int32_t>(enableDgamma));
    tilingData_.set_enableDbeta(static_cast<int32_t>(enableDbeta));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormGradInferTiling::DoOpTiling()
{
    DoOpTilingForStage0();
    DoOpTilingForStage1();
    return ge::GRAPH_SUCCESS;
}

uint64_t BatchNormGradInferTiling::GetTilingKey() const
{
    return TILINGKEY_INFER_BASE + subTilingKey;
}

ge::graphStatus BatchNormGradInferTiling::PostTiling()
{
    context_->SetBlockDim(std::max(usedCoreNums_, tilingData_.get_blockNum()));
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = workspaceSize_;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(BatchNormGrad, BatchNormGradInferTiling, 91000);
}  // namespace optiling
