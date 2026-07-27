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
 * \file batch_norm_grad_v3_tiling_ra_split_r.cpp
 * \brief
 */

#include "op_host/tiling_templates_registry.h"
#include "batch_norm_grad_v3_tiling_ra_split_r.h"

using namespace AscendC;

namespace optiling {

constexpr uint64_t STAGE0_R_ELEM_NUM = 4; // mainX, foldX, mainDy, foldDy
constexpr uint64_t STAGE0_A_ELEM_NUM = 4; // mean, rstd, dbeta, dgamma
constexpr uint64_t STAGE2_A_ELEM_NUM = 5; // mean, rstd, dbeta, dgamma, gamma
constexpr uint64_t STAGE2_R_ELEM_NUM = 3; // dy, x, dx
constexpr uint64_t DOUBLE_BUFF = 2;
constexpr uint64_t WORKSPACE_NUM = 2;
constexpr uint64_t ULONG_BIT_LEN = 64;
constexpr int64_t R_LOOP_FACTOR = 64;
// R轴足够长(>LONG_R_FACTOR*coreNum)时即便切A的备选模板能填满核，切R仍更优。
// 16 来自Ascend950实测标定:R更短时切A那路的固定开销摊薄后两者基本持平，更长时切R稳定胜出。
constexpr int64_t LONG_R_FACTOR = 16;
constexpr uint64_t FLOAT_BYTE_SIZE = sizeof(float);
constexpr uint64_t BNG_V3_RA_SPLIT_R_TK_BASE = 11000000;
constexpr uint64_t BNG_V3_RA_SPLIT_R_TILING_KEY = 50000000;
// fused子变体:stage0载入的dy/x在stage2复用，省第二遍GM重读。次高位区分(对齐RAR家族31/32惯例)。
constexpr uint64_t BNG_V3_RA_SPLIT_R_FUSED_SINGLE_TILING_KEY = 51000000;
constexpr uint64_t BNG_V3_RA_SPLIT_R_FUSED_PAIR_TILING_KEY = 52000000;
constexpr size_t BNG_WORKSPACE_RESERVED = 16 * 1024 * 1024;
constexpr int64_t CONST_ONE = 1;
// fused kernel InitBuffer 的 input queue 深度(与 kernel 内 BUFFER_NUM / C128_FUSED_INPUT_BUFFER 一致):
// single 双缓冲驻留 1 个 tile,pair 需 4 buffer 驻留 2 个 tile。
constexpr uint64_t FUSED_SINGLE_INBUF_NUM = 2;
constexpr uint64_t FUSED_PAIR_INBUF_NUM = 4;
constexpr int64_t MIN_RIVAL_CORES = 2; // 切A的备选模板至少能用 2 核时才与切R竞争(仅1核=单核全载,小/中R反更快)
constexpr int64_t UNIFORM_BINARY_BLOCK_CNT = 2; // uniformFold 判据:二分归约恰 2 块
constexpr int64_t A_LOOP_TIMES_PAIR = 2;        // fused-pair:A 轴两次循环(双 tile)
constexpr uint64_t PAIRED_BUFFER_NUM = 2;       // 成对张量各占一份缓冲(dy/x、dbeta/dgamma、mean/rstd 段)
constexpr uint64_t MERGED_ADIM_QUEUE_NUM = 5; // fused aDim 队列合计:meanInQue(×2)+gammaInQue(×1)+dbetaWsOutQue(×2)

int64_t BatchNormGradV3TilingRASplitR::CalcBlockFactor() const
{
    return std::max(R_LOOP_FACTOR, Ops::Base::CeilDiv(r1Dim, static_cast<int64_t>(coreNum)));
}

int64_t BatchNormGradV3TilingRASplitR::CalcUsedCoreNum() const { return Ops::Base::CeilDiv(r1Dim, CalcBlockFactor()); }

bool BatchNormGradV3TilingRASplitR::IsCapable()
{
    if (r0Dim != 1 || r1Dim < R_LOOP_FACTOR || r1Dim < aDim) {
        return false;
    }
    // 估一下:若本切R模板不接管,会按调度优先级接手同一 shape 的另一路模板(RA_RECOMPUTE / RA_ALL_LOAD)
    // 能用多少核 —— 它们只沿 A(通道)轴切核。下面用与它们相同的公式算出其可用核数 rivalCores,
    // 再与切R自己能用的核数比较,据此决定该不该把这个 shape 调度到切R模板。
    int64_t dtypeSize = (dyDtype == ge::DataType::DT_FLOAT) ? static_cast<int64_t>(FLOAT_BYTE_SIZE) :
                                                              static_cast<int64_t>(sizeof(uint16_t));
    // aFactor恒>=blockSize/dtypeSize>=1，无需再判零
    int64_t aFactor = std::max(Ops::Base::CeilDiv(aDim, static_cast<int64_t>(coreNum)),
                               static_cast<int64_t>(blockSize) / dtypeSize);
    int64_t rivalCores = Ops::Base::CeilDiv(aDim, aFactor);
    // 【重要】必须同时比自己能用几个核:blockFactor 有 R_LOOP_FACTOR 下限,r1Dim < R_LOOP_FACTOR*coreNum 时
    // 切R自己也填不满核。只判"切A那路填不满(rivalCores<coreNum)"会把 ownCores < rivalCores 的存量 shape
    // 抢过来反而更慢(如 NCHW(256,256,1,1):切A 32 核、切R 仅 4 核)。
    int64_t ownCores = CalcUsedCoreNum();
    // 自己比切A那路能用更多核 且 rivalCores 至少 2 → 调度到切R模板; 或 R足够长(只切A不切R必慢)。
    // rivalCores>=2 这条是上板标定加的:rivalCores 仅 1 时是单核全载,小/中 R 下它反而更快
    // (切R 的 workspace 往返 + 2×SyncAll 固定开销吃光多核收益;实测 C=1/C=8 小 R +15~52% 回退),
    // 此时只靠第二分支(R 足够长)接管,不在第一分支抢。
    if ((rivalCores >= MIN_RIVAL_CORES && rivalCores < static_cast<int64_t>(coreNum) && ownCores > rivalCores) ||
        r1Dim > LONG_R_FACTOR * static_cast<int64_t>(coreNum)) {
        return true;
    }
    return false;
}

ge::graphStatus BatchNormGradV3TilingRASplitR::DoOpTiling()
{
    dyTypeSize_ = ge::GetSizeByDataType(dyDtype);
    blockFactor_ = CalcBlockFactor();
    usedCoreNum_ = CalcUsedCoreNum();
    tailBlockFactor_ = (r1Dim % blockFactor_ == 0) ? blockFactor_ : r1Dim % blockFactor_;
    aFactor_ = std::min(Ops::Base::GetVRegSize(context_) / dyTypeSize_, aDim);
    aFactorAlign_ = Ops::Base::CeilAlign(aFactor_, static_cast<int64_t>(blockSize / dyTypeSize_));
    aLoopTimes_ = Ops::Base::CeilDiv(aDim, aFactorAlign_);
    aFactorTail_ = (aDim % aFactorAlign_ == 0) ? aFactorAlign_ : aDim % aFactorAlign_;

    OP_TILING_CHECK(ge::GRAPH_SUCCESS != Stage0Stage1UbTiling(),
                    VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "failed Stage0Stage1UbTiling."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(ge::GRAPH_SUCCESS != Stage2UbTiling(),
                    VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "failed Stage2UbTiling."),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormGradV3TilingRASplitR::Stage0Stage1UbTiling()
{
    // 一次计算一个tiling块
    rLoopFactor_ = std::min(R_LOOP_FACTOR, blockFactor_);
    binaryBlockCnt_ = Ops::Base::CeilDiv(blockFactor_, rLoopFactor_);
    binaryFoldPoint_ = (binaryBlockCnt_ <= 1) ? 1 : 1L << (ULONG_BIT_LEN - 1 - __builtin_clzl(binaryBlockCnt_ - 1));
    cacheBuffCnt_ = ULONG_BIT_LEN - __builtin_clzl(binaryBlockCnt_);
    binaryBlockTail_ = (blockFactor_ % rLoopFactor_) == 0 ? rLoopFactor_ : blockFactor_ % rLoopFactor_;
    lastCoreBlockCnt_ = Ops::Base::CeilDiv(tailBlockFactor_, rLoopFactor_);
    lastCoreFoldPoint_ = (lastCoreBlockCnt_ <= 1) ? 1 :
                                                    1L << (ULONG_BIT_LEN - 1 - __builtin_clzl(lastCoreBlockCnt_ - 1));
    lastCoreLoopTail_ = (tailBlockFactor_ % rLoopFactor_) == 0 ? rLoopFactor_ : tailBlockFactor_ % rLoopFactor_;

    // 校验UB是否越界
    uint64_t rElemUbSize = Ops::Base::CeilAlign(
        aFactorAlign_ * rLoopFactor_ * STAGE0_R_ELEM_NUM * dyTypeSize_ * DOUBLE_BUFF, blockSize / dyTypeSize_);
    // kernel Stage0InitBuffer 开了 dbetaCacheBuffer_ + dgammaCacheBuffer_ 两份,此处按 2 份计
    uint64_t cacheBuffSize = PAIRED_BUFFER_NUM * Ops::Base::CeilAlign(cacheBuffCnt_ * aFactorAlign_ * FLOAT_BYTE_SIZE,
                                                                      blockSize / FLOAT_BYTE_SIZE);
    uint64_t aElemUbSize = Ops::Base::CeilAlign(aFactorAlign_ * STAGE0_A_ELEM_NUM * DOUBLE_BUFF * FLOAT_BYTE_SIZE,
                                                blockSize / FLOAT_BYTE_SIZE);
    uint64_t oneStepUbSize = rElemUbSize + cacheBuffSize + aElemUbSize;
    OP_TILING_CHECK(ubSize < oneStepUbSize,
                    VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "ubSize %ld less than oneStepUbSize: %ld.",
                                                    ubSize, oneStepUbSize),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormGradV3TilingRASplitR::Stage2UbTiling()
{
    uint64_t aElemUbSize = Ops::Base::CeilAlign(aFactorAlign_ * STAGE2_A_ELEM_NUM * FLOAT_BYTE_SIZE * DOUBLE_BUFF,
                                                blockSize / FLOAT_BYTE_SIZE);
    uint64_t rElemUbSize = Ops::Base::CeilAlign(aFactorAlign_ * STAGE2_R_ELEM_NUM * dyTypeSize_ * DOUBLE_BUFF,
                                                blockSize / dyTypeSize_);
    OP_TILING_CHECK(ubSize < aElemUbSize + rElemUbSize,
                    VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "ubSize %ld less than oneTileUbSize: %ld.",
                                                    ubSize, aElemUbSize + rElemUbSize),
                    return ge::GRAPH_FAILED);
    int64_t dxLoopFactor = Ops::Base::FloorDiv(ubSize - aElemUbSize, rElemUbSize);
    dxLoopFactor_ = std::min(blockFactor_, dxLoopFactor);
    dxLoopTimes_ = Ops::Base::CeilDiv(blockFactor_, dxLoopFactor_),
    dxLoopTail_ = (blockFactor_ % dxLoopFactor_ == 0) ? dxLoopFactor_ : blockFactor_ % dxLoopFactor_;
    dxLastCoreFactor_ = std::min(tailBlockFactor_, dxLoopFactor);
    dxLastCoreTimes_ = Ops::Base::CeilDiv(tailBlockFactor_, dxLastCoreFactor_),
    dxLastCoreTail_ = (tailBlockFactor_ % dxLastCoreFactor_ == 0) ? dxLastCoreFactor_ :
                                                                    tailBlockFactor_ % dxLastCoreFactor_;
    return ge::GRAPH_SUCCESS;
}

uint64_t BatchNormGradV3TilingRASplitR::GetTilingKey() const
{
    // fused快速路径触发条件(各核行数一致 binaryBlockCnt==2)：
    //  - single(aLoopTimes==1, C≤aFactor): dtype无关,fp32/fp16/bf16均可(只驻留1个tile,UB~224KB放得下)；
    //  - pair(aLoopTimes==2): 仅fp32。fp16/bf16的aFactor=128→aFactorAlign=128使所有buffer翻倍,
    //    pair驻留2tile+float tmp+stage1归约缓冲≈288KB>248KB UB,会NO_OUTPUT崩;故fp16/bf16的pair退generic。
    //    fp32 pair含部分尾块(C∈65~127),kernel按aFactorTail处理。
    // aLoopTimes>2不fuse:跨核SyncAll翻倍开销>省下的(L2吸收的)GM重读,实测反比generic慢16~27%。
    bool isFp32 = (dyDtype == ge::DataType::DT_FLOAT) && (weightDtype == ge::DataType::DT_FLOAT);
    // binaryBlockCnt_==2 已蕴含 binaryFoldPoint_==1(见 :113 的计算:cnt==2 时 1L<<(63-__builtin_clzl(1))==1),
    // 故此处不再重复判 binaryFoldPoint_。
    bool uniformFold = (blockFactor_ == tailBlockFactor_) && (binaryBlockCnt_ == UNIFORM_BINARY_BLOCK_CNT);
    // fused 合计 UB 兜底:fused InitBuffer 把 dy/x 驻留 + float tmp + cache + stage1 全核归约缓冲 + dxOut
    // 同时常驻(generic 是分段复用,host 已分别校验过;fused 不复用,必须按合计校验)。0 号核最坏:
    // 含全核归约缓冲(usedCoreNum×aFactorAlign)与 dbeta/dgamma 输出。放不下则退 generic,避免 NO_OUTPUT。
    // 【重要】下面每一项与 kernel BatchNormGradV3RASplitR::FusedInitBuffer() 的 InitBuffer 逐条一一对应,
    // 二者是同一份 UB 布局的两处实现;改动 kernel 侧 buffer 时必须同步此处,否则兜底失真会放过 NO_OUTPUT。
    auto alignBlk = [this](uint64_t bytes) { return Ops::Base::CeilAlign(bytes, static_cast<uint64_t>(blockSize)); };
    uint64_t rDimSize = static_cast<uint64_t>(rLoopFactor_) * aFactorAlign_;
    uint64_t aDimSize = static_cast<uint64_t>(aFactorAlign_) * FLOAT_BYTE_SIZE;
    uint64_t coreReduceSize = static_cast<uint64_t>(usedCoreNum_) * aFactorAlign_ * FLOAT_BYTE_SIZE;
    // dbetaWsInQue_ 里 dbeta/dgamma 两段各按 VL 取整预留(kernel Stage1VlAlignedSpan()):
    // ReduceSum<RA> 原地折叠时末次迭代按 fullMask 写满一个 VL,不足 VL 的尾部也要有可写空间。
    uint64_t vlBytes = static_cast<uint64_t>(Ops::Base::GetVRegSize(context_));
    uint64_t stage1InSize = PAIRED_BUFFER_NUM * Ops::Base::CeilAlign(coreReduceSize, vlBytes);
    uint64_t inBuf = alignBlk(rDimSize * dyTypeSize_);
    // 合并队列按 2×aDimSize 计:meanInQue_(mean+rstd)、dbetaWsOutQue_(dbeta+dgamma) 各占 2×;gamma 占 1×
    uint64_t fusedCommonUb = PAIRED_BUFFER_NUM * alignBlk(rDimSize * FLOAT_BYTE_SIZE) + // dyTmpQue_ + xTmpQue_ (float)
                             MERGED_ADIM_QUEUE_NUM *
                                 alignBlk(aDimSize) + // meanInQue_(×2) + gammaInQue_(×1) + dbetaWsOutQue_(×2)
                             PAIRED_BUFFER_NUM *
                                 alignBlk(aDimSize * cacheBuffCnt_) + // dbetaCacheBuffer_ + dgammaCacheBuffer_
                             alignBlk(stage1InSize) + // dbetaWsInQue_(dbeta+dgamma,0 号核全核归约)
                             PAIRED_BUFFER_NUM * alignBlk(aDimSize) + // dbetaOutQue_(dbeta+dgamma 输出,0 号核)
                             alignBlk(static_cast<uint64_t>(blockFactor_) * aFactorAlign_ * dyTypeSize_); // dxOutQue_
    uint64_t fusedSingleUb = fusedCommonUb +
                             PAIRED_BUFFER_NUM * FUSED_SINGLE_INBUF_NUM * inBuf; // dyInQue_+xInQue_ 各 BUFFER_NUM 个
    uint64_t fusedPairUb = fusedCommonUb + PAIRED_BUFFER_NUM * FUSED_PAIR_INBUF_NUM *
                                               inBuf; // dyInQue_+xInQue_ 各 C128_FUSED_INPUT_BUFFER 个
    if (uniformFold) {
        if (aLoopTimes_ == 1 && fusedSingleUb <= ubSize) {
            return BNG_V3_RA_SPLIT_R_FUSED_SINGLE_TILING_KEY;
        }
        if (aLoopTimes_ == A_LOOP_TIMES_PAIR && isFp32 && fusedPairUb <= ubSize) {
            return BNG_V3_RA_SPLIT_R_FUSED_PAIR_TILING_KEY;
        }
    }
    return BNG_V3_RA_SPLIT_R_TILING_KEY;
}

ge::graphStatus BatchNormGradV3TilingRASplitR::GetWorkspaceSize()
{
    workspaceSize_ = BNG_WORKSPACE_RESERVED + usedCoreNum_ * aDim * FLOAT_BYTE_SIZE * WORKSPACE_NUM;
    OP_LOGI(context_->GetNodeName(), "Workspace size: %ld", workspaceSize_);
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

void BatchNormGradV3TilingRASplitR::PrintTilingData()
{
    OP_LOGI(context_->GetNodeName(),
            "BatchNormGradV3TilingRASplitR tilingData: useCoreNum is %ld, rDim is %ld, aDim is %ld, "
            "blockFactor is %ld, tailBlockFactor %ld, rLoopFactor is %ld, binaryBlockCnt is %ld, "
            "binaryFoldPoint is %ld, binaryBlockTail is %ld, lastCoreBlockCnt is %ld, lastCoreFoldPoint is %ld, "
            "lastCoreLoopTail is %ld, aFactor %ld, aFactorAlign is %ld, aFactorTail is %ld, aLoopTimes is %ld, "
            "dxLoopFactor %ld, dxLoopTail is %ld, dxLoopTimes %ld, dxLastCoreFactor %ld, dxLastCoreTail is %ld, "
            "dxLastCoreTimes %ld, cacheBuffCnt is %ld, tilingKey is %ld",
            usedCoreNum_, tilingData_.get_rDim(), tilingData_.get_aDim(), tilingData_.get_blockFactor(),
            tilingData_.get_tailBlockFactor(), tilingData_.get_rLoopFactor(), tilingData_.get_binaryBlockCnt(),
            tilingData_.get_binaryFoldPoint(), tilingData_.get_binaryBlockTail(), tilingData_.get_lastCoreBlockCnt(),
            tilingData_.get_lastCoreFoldPoint(), tilingData_.get_lastCoreLoopTail(), tilingData_.get_aFactor(),
            tilingData_.get_aFactorAlign(), tilingData_.get_aFactorTail(), tilingData_.get_aLoopTimes(),
            tilingData_.get_dxLoopFactor(), tilingData_.get_dxLoopTail(), tilingData_.get_dxLoopTimes(),
            tilingData_.get_dxLastCoreFactor(), tilingData_.get_dxLastCoreTail(), tilingData_.get_dxLastCoreTimes(),
            tilingData_.get_cacheBuffCnt(), GetTilingKey());
    return;
}

ge::graphStatus BatchNormGradV3TilingRASplitR::PostTiling()
{
    tilingData_.set_usedCoreNum(usedCoreNum_);
    tilingData_.set_rDim(r1Dim);
    tilingData_.set_aDim(aDim);
    tilingData_.set_blockFactor(blockFactor_);
    tilingData_.set_tailBlockFactor(tailBlockFactor_);
    tilingData_.set_rLoopFactor(rLoopFactor_);
    tilingData_.set_binaryBlockCnt(binaryBlockCnt_);
    tilingData_.set_binaryFoldPoint(binaryFoldPoint_);
    tilingData_.set_binaryBlockTail(binaryBlockTail_);
    tilingData_.set_lastCoreBlockCnt(lastCoreBlockCnt_);
    tilingData_.set_lastCoreFoldPoint(lastCoreFoldPoint_);
    tilingData_.set_lastCoreLoopTail(lastCoreLoopTail_);
    tilingData_.set_aFactor(aFactor_);
    tilingData_.set_aFactorAlign(aFactorAlign_);
    tilingData_.set_aFactorTail(aFactorTail_);
    tilingData_.set_aLoopTimes(aLoopTimes_);
    tilingData_.set_dxLoopFactor(dxLoopFactor_);
    tilingData_.set_dxLoopTail(dxLoopTail_);
    tilingData_.set_dxLoopTimes(dxLoopTimes_);
    tilingData_.set_dxLastCoreFactor(dxLastCoreFactor_);
    tilingData_.set_dxLastCoreTail(dxLastCoreTail_);
    tilingData_.set_dxLastCoreTimes(dxLastCoreTimes_);
    tilingData_.set_cacheBuffCnt(cacheBuffCnt_);

    uint64_t tilingKey = GetTilingKey();
    context_->SetTilingKey(tilingKey);
    context_->SetBlockDim(usedCoreNum_);
    context_->SetScheduleMode(CONST_ONE);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("BatchNormGradV3", BatchNormGradV3TilingRASplitR, BNG_V3_RA_SPLIT_R_TK_BASE);

} // namespace optiling
