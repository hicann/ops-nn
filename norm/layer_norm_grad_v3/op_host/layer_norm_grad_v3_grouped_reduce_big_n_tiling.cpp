/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file layer_norm_grad_v3_grouped_reduce_big_n_tiling.cc
 * \brief
 */

#include "layer_norm_grad_v3_tiling.h"

namespace optiling {
constexpr static int64_t CONST_ZERO = 0;
constexpr static int64_t CONST_ONE = 1;
constexpr static int64_t CONST_TWO = 2;
constexpr static int64_t CONST_FOUR = 4;
constexpr static int64_t CONST_EIGHT = 8;

constexpr int64_t DEFAULT_GAMMA_BETA_M_FACTOR = 128;
#ifdef DAVID_FPGA
constexpr int64_t DEFAULT_GAMMA_BETA_N_FACTOR = 16;
#else
constexpr int64_t DEFAULT_GAMMA_BETA_N_FACTOR = 64;
#endif

constexpr int64_t DEFAULT_BACKWARD_M_FACTOR = 1;
#ifdef DAVID_FPGA
constexpr int64_t DEFAULT_BACKWARD_N_FACTOR = 1024;
#else
constexpr int64_t DEFAULT_BACKWARD_N_FACTOR = 1024 * 6;
#endif

bool LayerNormGradV3GroupedReduceBigNTiling::IsCapable()
{
    constexpr static int64_t ROW_THRESHOLD_MIN = 512;
    constexpr static int64_t COL_THRESHOLD_MAX = 8192;
    if (commonParams.rowSize < ROW_THRESHOLD_MIN && commonParams.colSize > COL_THRESHOLD_MAX) {
        return true;
    }
    return false;
}

ge::graphStatus LayerNormGradV3GroupedReduceBigNTiling::GammaBetaKernelTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormGradV3GroupedReduceBigNTiling::BackwardKernelTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormGradV3GroupedReduceBigNTiling::DoOpTiling()
{
    td_.set_row(static_cast<int64_t>(commonParams.rowSize));
    td_.set_col(static_cast<int64_t>(commonParams.colSize));
    td_.set_pdxIsRequire(static_cast<int32_t>(commonParams.pdxIsRequire));
    td_.set_pdgammaIsRequire(static_cast<int32_t>(commonParams.pdgammaIsRequire));
    td_.set_pdbetaIsRequire(static_cast<int32_t>(commonParams.pdbetaIsRequire));
    ge::graphStatus statusGammaBeta = GammaBetaKernelTiling();
    ge::graphStatus statusBackward = BackwardKernelTiling();
    return (statusGammaBeta && statusBackward);
}

ge::graphStatus LayerNormGradV3GroupedReduceBigNTiling::GetWorkspaceSize()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    int64_t ascendcWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();

    constexpr int64_t DEFAULT_WORKSPACE_SIZE = 32 * 1024 * 1024;
    constexpr int64_t NUM_THREE = 3;
    int64_t wsSize = commonParams.coreNum * commonParams.rowSize * NUM_THREE + DEFAULT_WORKSPACE_SIZE;
    wsSize += ascendcWorkspaceSize;
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    if (workspaces == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspaces[0] = wsSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormGradV3GroupedReduceBigNTiling::PostTiling()
{
    constexpr static int64_t MAX_CORE_NUM = 64;
    int64_t blockDim = commonParams.coreNum > MAX_CORE_NUM ? MAX_CORE_NUM : commonParams.coreNum;
    context_->SetBlockDim(blockDim);
    context_->SetScheduleMode(1); // Set to batch mode, all cores start simultaneously
    td_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(td_.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

uint64_t LayerNormGradV3GroupedReduceBigNTiling::GetTilingKey() const
{
    uint64_t templateKey = static_cast<uint64_t>(LNGTemplateKey::GROUPED_REDUCE_BIG_N);
    return templateKey * LNG_TEMPLATE_KEY_WEIGHT + static_cast<uint64_t>(commonParams.dtypeKey);
}

REGISTER_TILING_TEMPLATE("LayerNormGradV3", LayerNormGradV3GroupedReduceBigNTiling, 6000);
} // namespace optiling
