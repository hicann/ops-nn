/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file layer_norm_grad_v3_transpose_regbase_tiling.cpp
 * \brief
 */

#include <iostream>
#include <vector>
#include "layer_norm_grad_v3_tiling.h"
#include "tiling/tiling_api.h"

namespace optiling {
constexpr static int64_t M_TILE_MIN_SIZE = 16;           // M最小块大小
constexpr static int64_t TRANSPOSE_COL_BYTES_LIMIT = 32; // 模板路由字节数上限
constexpr static int64_t TRANSPOSE_M_THRESHOLD = 256;    // 模板路由行数下限
constexpr static int64_t GAMMA_BETA_M_ALIGN_MAX = 512;   // gamma_beta mFactorAlign上限(预留)
constexpr static int64_t GAMMA_BETA_M_ALIGN_MIN = 64;    // gamma_beta mFactorAlign下限(预留)
constexpr static int64_t TRANSPOSE_ALIGN_LIMIT = 16;     // TransdataTo5HD对齐粒度约束(16元素)
constexpr static int64_t M_FACTOR_ALIGN_MAX = 2048;      // gamma_beta mFactorAlign搜索上界(2的幂)
// UB公式合并同类项后的系数(buffer份数, 与kernel侧InitBuffer写死的buffer数对应)
constexpr static int64_t NUM_TWO = 2;            // DOUBLE_BUFFER份数
constexpr static int64_t NUM_SIX = 6;            // M维buffer总份数
constexpr static int64_t NUM_SEVEN = 7;          // MN维buffer总份数
constexpr static int64_t UB_RESERVE_SIZE = 1024; // 切分UB保留1K的UB空间

bool LayerNormGradV3TransposeRegBaseTiling::IsCapable()
{
    if (!commonParams.isRegBase) {
        return false;
    }
    uint64_t dtypeSize = (commonParams.dyDtype == ge::DT_FLOAT) ? FLOAT_SIZE : HALF_SIZE;
    uint64_t colBytes = commonParams.colSize * dtypeSize;
    if (colBytes > TRANSPOSE_COL_BYTES_LIMIT) {
        return false;
    }
    int64_t row = static_cast<int64_t>(commonParams.rowSize);
    if (row <= TRANSPOSE_M_THRESHOLD) {
        return false;
    }
    return true;
}

// 给定mFactorAlign候选，计算UB空间限制下的MMax
int64_t LayerNormGradV3TransposeRegBaseTiling::CalcGammaBetaMMax(int64_t mFactorAlign, int64_t mPerCore, int64_t nAlign,
                                                                 int64_t ubSize)
{
    // 主核按mFactorAlign切mPerCore，主核总迭代轮数
    int64_t totalRounds = ops::CeilDiv(mPerCore, mFactorAlign);
    // mainFactorNum为totalRounds向下取最近的2的幂，得到主块的数量
    int64_t mainFactorNum = FindNearestPower2(totalRounds);
    int64_t cacheBufferCount = 1;
    if (mainFactorNum != 0) {
        cacheBufferCount = ULONG_BIT_LEN - static_cast<int64_t>(__builtin_clzl(static_cast<uint64_t>(mainFactorNum)));
    }

    // UseUB = ((3*dy + 3*x + 1*tmp)*MN + (3*mean + 3*rstd)*M + (2*dgamma + 2*dbeta + 2*reduceSum +
    // 2*cnt*cache)*N)*dtypeSize + UB_RESERVE_SIZE 移项后得到: M <= (ubSize - UB_RESERVE_SIZE - (NUM_SIX +
    // NUM_TWO*cnt)*N*dtypeSize) / ((NUM_SEVEN*N + NUM_SIX)*dtypeSize)
    constexpr int64_t dtypeSize = static_cast<int64_t>(FLOAT_SIZE);
    return (ubSize - UB_RESERVE_SIZE - (NUM_SIX + NUM_TWO * cacheBufferCount) * nAlign * dtypeSize) /
           ((NUM_SEVEN * nAlign + NUM_SIX) * dtypeSize);
}

ge::graphStatus LayerNormGradV3TransposeRegBaseTiling::GammaBetaKernelTiling()
{
    int64_t row = static_cast<int64_t>(commonParams.rowSize);
    int64_t col = static_cast<int64_t>(commonParams.colSize);
    int64_t coreNum = static_cast<int64_t>(commonParams.coreNum);
    int64_t ubSize = static_cast<int64_t>(commonParams.ubSizePlatForm);
    int64_t vlFp32 = static_cast<int64_t>(commonParams.vlFp32);

    // 分核计算每个核多少M
    int64_t mPerCore = ops::CeilDiv(row, coreNum);
    mPerCore = std::max(mPerCore, vlFp32); // 限制单个核最少64
    // 计算使用多少个核
    int64_t usedCoreNum = ops::CeilDiv(row, mPerCore);
    // 计算尾核大小
    int64_t mTailCore = row - mPerCore * (usedCoreNum - 1);

    // Step1: N轴对齐，对齐粒度 = blockSize/dtypeSize
    int64_t dtypeSize = (commonParams.dyDtype == ge::DT_FLOAT16 || commonParams.dyDtype == ge::DT_BF16) ?
                            static_cast<int64_t>(HALF_SIZE) :
                            static_cast<int64_t>(FLOAT_SIZE);
    int64_t nAlign = ops::CeilAlign(col, commonParams.blockSize / dtypeSize);

    // Step2: 从搜索上界开始按2的幂递减寻找最大可行mFactorAlign
    // mFactorAlign下界 = vlFp32，上界 = 人为设定2048
    int64_t mFactorAlignMin = vlFp32;
    int64_t mFactorAlignMax = M_FACTOR_ALIGN_MAX;
    int64_t mFactorAlign = 0;
    for (int64_t candidate = mFactorAlignMax; candidate >= mFactorAlignMin; candidate /= 2) {
        int64_t mFactorMax = CalcGammaBetaMMax(candidate, mPerCore, nAlign, ubSize);
        if (mFactorMax >= candidate) {
            mFactorAlign = candidate;
            break;
        }
    }

    // mFactorAlign为0表示连最小mFactorAlign都无法满足，UB空间不足
    if (mFactorAlign == 0) {
        OP_LOGI(context_->GetNodeName(), "Transpose RegBase gamma_beta is not capable. col: %ld, ubSize: %luB", col,
                commonParams.ubSizePlatForm);
        return ge::GRAPH_PARAM_INVALID;
    }

    // Step3: 分别计算主核和尾核的buffercount和resultid
    // 将每核M行切成mFactorAlign大小的块, 余数为tail
    int64_t mainCoreLoop = mPerCore / mFactorAlign;                 // 主核满块数
    int64_t mainCoreTail = mPerCore - mainCoreLoop * mFactorAlign;  // 主核尾块
    int64_t tailCoreLoop = mTailCore / mFactorAlign;                // 尾核满块数
    int64_t tailCoreTail = mTailCore - tailCoreLoop * mFactorAlign; // 尾核尾块

    // 总轮数 = 满块数 + (有尾块则+1)
    int64_t MainTotalRound = mainCoreLoop + (mainCoreTail > 0 ? 1 : 0);
    int64_t TailTotalRound = tailCoreLoop + (tailCoreTail > 0 ? 1 : 0);

    // 主核
    int64_t cacheBufferCount = 0;
    int64_t mainResultCacheID = 0;
    // 计算二分累加折叠点
    int64_t mainCoreBasicBlock = FindNearestPower2(MainTotalRound);
    if (mainCoreBasicBlock != 0) {
        // cacheBufferCount = log2(basicBlock)+1, 即二分树需要的cache层数
        cacheBufferCount = ULONG_BIT_LEN -
                           static_cast<int64_t>(__builtin_clzl(static_cast<uint64_t>(mainCoreBasicBlock)));
        // 结果cacheID = 二分树最后一层对应的ID(basicBlock-1轮的cacheID)
        mainResultCacheID = GetCacheID(mainCoreBasicBlock - 1);
    } else {
        cacheBufferCount = 1; // 仅1轮时cache只需1层
        mainResultCacheID = 0;
    }

    int64_t tailResultCacheID = 0;
    int64_t tailCoreBasicBlock = FindNearestPower2(TailTotalRound);
    if (tailCoreBasicBlock != 0) {
        tailResultCacheID = GetCacheID(tailCoreBasicBlock - 1);
    } else {
        tailResultCacheID = 0;
    }

    td_.set_gammaBetaNAlign(nAlign);
    td_.set_gammaBetaMAlign(mFactorAlign);
    td_.set_gammaBetaMPerCore(mPerCore);
    td_.set_gammaBetaUsedCoreNum(usedCoreNum);
    td_.set_gammaBetaMTailCore(mTailCore);
    td_.set_gammaBetaCacheBufferCount(cacheBufferCount); // cacheBufferCount统一按主核的数量分配
    td_.set_gammaBetaMainCoreBasicBlock(mainCoreBasicBlock);
    td_.set_gammaBetaTailCoreBasicBlock(tailCoreBasicBlock);
    td_.set_gammaBetaMainResultCacheID(mainResultCacheID);
    td_.set_gammaBetaTailResultCacheID(tailResultCacheID);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormGradV3TransposeRegBaseTiling::BackwardKernelTiling()
{
    int64_t row = static_cast<int64_t>(commonParams.rowSize);
    int64_t col = static_cast<int64_t>(commonParams.colSize);
    int64_t coreNum = static_cast<int64_t>(commonParams.coreNum);
    int64_t ubSize = static_cast<int64_t>(commonParams.ubSizePlatForm);
    int64_t vlFp32 = static_cast<int64_t>(commonParams.vlFp32);
    int64_t blockSize = static_cast<int64_t>(commonParams.blockSize);

    // Step1: N轴对齐，对齐粒度 = blockSize/dtypeSize
    int64_t dtypeSize = (commonParams.dyDtype == ge::DT_FLOAT16 || commonParams.dyDtype == ge::DT_BF16) ?
                            static_cast<int64_t>(HALF_SIZE) :
                            static_cast<int64_t>(FLOAT_SIZE);
    int64_t nAlign = ops::CeilAlign(col, blockSize / dtypeSize);

    // UseUB = ((2*dy + 2*x + 2*dx + 1*xNorm)*MN + (2*mean + 2*rstd + 2*sum)*M + 1*gamma*N)*dtypeSize + UB_RESERVE_SIZE
    // 移项后得到: M <= (ubSize - UB_RESERVE_SIZE - N*dtypeSize) / ((NUM_SEVEN*N + NUM_SIX)*dtypeSize)
    constexpr int64_t dtypeFloatSize = static_cast<int64_t>(FLOAT_SIZE);
    int64_t mFactorMax = (ubSize - UB_RESERVE_SIZE - nAlign * dtypeFloatSize) /
                         ((NUM_SEVEN * nAlign + NUM_SIX) * dtypeFloatSize);
    if (mFactorMax < TRANSPOSE_ALIGN_LIMIT) {
        OP_LOGI(context_->GetNodeName(),
                "Transpose RegBase backward is not capable. mFactorMax: %ld < %ld, col: %ld, ubSize: %luB", mFactorMax,
                TRANSPOSE_ALIGN_LIMIT, col, ubSize);
        return ge::GRAPH_PARAM_INVALID;
    }

    // mFactorAlign向下对齐到16, 保证TransDataTo5HD的C0=16对齐
    int64_t mFactorAlign = ops::FloorAlign(mFactorMax, TRANSPOSE_ALIGN_LIMIT);
    if (mFactorAlign < TRANSPOSE_ALIGN_LIMIT) {
        mFactorAlign = TRANSPOSE_ALIGN_LIMIT; // 兜底至少16
    }

    // 多核切分: 每核mPerCore行, 尾核mTailCore行
    int64_t mPerCore = ops::CeilDiv(row, coreNum);
    mPerCore = std::max(mPerCore, vlFp32); // 限制单个核最少vlFp32
    int64_t usedCoreNum = ops::CeilDiv(row, mPerCore);
    int64_t mTailCore = row - (usedCoreNum - 1) * mPerCore;

    td_.set_backwardMAlign(mFactorAlign);
    td_.set_backwardNAlign(nAlign);
    td_.set_backwardUsedCoreNum(usedCoreNum);
    td_.set_backwardMPerCore(mPerCore);
    td_.set_backwardMTailCore(mTailCore);

    // ReduceSum tmp buffer check: 比较API要求的maxTmpSize与kernel实际可用的dxOut空间
    std::vector<int64_t> reduceDims = {static_cast<int64_t>(col), mFactorAlign};
    ge::Shape reduceShape(reduceDims);
    uint32_t maxTmpSize = 0;
    uint32_t minTmpSize = 0;
    AscendC::GetReduceSumMaxMinTmpSize(reduceShape, ge::DT_FLOAT, AscendC::ReducePattern::RA, false, false, maxTmpSize,
                                       minTmpSize);
    int64_t actualTmpSize = nAlign * mFactorAlign * static_cast<int64_t>(FLOAT_SIZE);
    if (static_cast<int64_t>(maxTmpSize) > actualTmpSize) { // API要求超过实际空间则不达标
        OP_LOGI(context_->GetNodeName(),
                "Transpose RegBase backward tmp buffer insufficient. maxRequired: %u, actual: %ld, col: %ld, "
                "mFactorAlign: %ld",
                maxTmpSize, actualTmpSize, col, mFactorAlign);
        return ge::GRAPH_PARAM_INVALID;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormGradV3TransposeRegBaseTiling::DoOpTiling()
{
    td_.set_row(static_cast<int64_t>(commonParams.rowSize));
    td_.set_col(static_cast<int64_t>(commonParams.colSize));
    td_.set_pdxIsRequire(static_cast<int32_t>(commonParams.pdxIsRequire));
    td_.set_pdgammaIsRequire(static_cast<int32_t>(commonParams.pdgammaIsRequire));
    td_.set_pdbetaIsRequire(static_cast<int32_t>(commonParams.pdbetaIsRequire));

    // 两段无条件执行, 保证tilingdata字段始终完整初始化, 防止部分计算时出现异常初始值;
    ge::graphStatus statusGammaBeta = GammaBetaKernelTiling();
    OP_TILING_CHECK(statusGammaBeta != ge::GRAPH_SUCCESS, , return statusGammaBeta);

    ge::graphStatus statusBackward = BackwardKernelTiling();
    OP_TILING_CHECK(statusBackward != ge::GRAPH_SUCCESS, , return statusBackward);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormGradV3TransposeRegBaseTiling::GetWorkspaceSize()
{
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    // 用户workspace: dbeta/dgamma两段, 每核各占NAlign个float
    int64_t gammaBetaUsedCoreNum = td_.get_gammaBetaUsedCoreNum();
    int64_t gammaBetaNAlign = td_.get_gammaBetaNAlign();
    size_t usrWorkspaceSize = static_cast<size_t>(gammaBetaUsedCoreNum * gammaBetaNAlign * NUM_TWO *
                                                  static_cast<int64_t>(FLOAT_SIZE));
    size_t sysWorkSpaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    currentWorkspace[0] = usrWorkspaceSize + sysWorkSpaceSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormGradV3TransposeRegBaseTiling::PostTiling()
{
    // blockDim取gamma_beta/backward较大核数下发
    int64_t gammaBetaUsedCoreNum = td_.get_gammaBetaUsedCoreNum();
    int64_t backwardUsedCoreNum = td_.get_backwardUsedCoreNum();
    int64_t numBlocks = std::max(gammaBetaUsedCoreNum, backwardUsedCoreNum);
    context_->SetBlockDim(numBlocks);
    context_->SetScheduleMode(1);
    td_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(td_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

uint64_t LayerNormGradV3TransposeRegBaseTiling::GetTilingKey() const
{
    constexpr uint64_t LNG_TRANSPOSE_REGBASE_TILINGKEY = 800;
    return LNG_TRANSPOSE_REGBASE_TILINGKEY;
}

REGISTER_TILING_TEMPLATE("LayerNormGradV3", LayerNormGradV3TransposeRegBaseTiling, 500);
} // namespace optiling
