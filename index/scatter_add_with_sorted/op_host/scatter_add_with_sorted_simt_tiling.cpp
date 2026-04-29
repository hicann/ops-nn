/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scatter_add_with_sorted_simt_tiling.cpp
 * \brief scatter_add_with_sorted_simt_tiling
 */

#include "scatter_add_with_sorted_simt_tiling.h"

namespace optiling {

static constexpr int64_t SIMT_INNER_THRES = 512;
static constexpr int64_t DOUBLE = 2;

bool ScatterAddWithSortedSimtTiling::IsCapable()
{
    bool isFloat = (varDtype_ == ge::DT_FLOAT || varDtype_ == ge::DT_FLOAT16 || varDtype_ == ge::DT_BF16);
    if (context_->GetDeterministic() && !isUpdateScalar_ && isFloat) {
        isDeterminTemplate_ = 1;
    }
    bool isSimt = varShape_[1] * updatesDtypeSize_ < SIMT_INNER_THRES;
    return isSimt;
}

void ScatterAddWithSortedSimtTiling::SetTilingData()
{
    tilingData_ = context_->GetTilingData<ScatterAddWithSortedSimtTilingData>();
    tilingData_->varShape[0] = varShape_[0];
    tilingData_->varShape[1] = varShape_[1];
    tilingData_->indicesNum = indicesNum_;
    tilingData_->normBlockIndices = normBlockIndices_;
    tilingData_->tailBlockIndices = tailBlockIndices_;
    tilingData_->usedCoreNum = usedCoreNum_;
    tilingData_->withPos = hasPos_;
    tilingData_->tilingKey = GetTilingKey();

    return;
}

ge::graphStatus ScatterAddWithSortedSimtTiling::DoOpTiling()
{
    if (varShape_[0] * varShape_[1] * indicesNum_ == 0) {
        usedCoreNum_ = 1;
        SetTilingData();
        return ge::GRAPH_SUCCESS;
    }
    normBlockIndices_ = Ops::Base::CeilDiv(indicesNum_, totalCoreNum_);
    usedCoreNum_ = Ops::Base::CeilDiv(static_cast<int64_t>(indicesNum_), normBlockIndices_);
    tailBlockIndices_ = indicesNum_ - (usedCoreNum_ - 1) * normBlockIndices_;
    SetTilingData();

    return ge::GRAPH_SUCCESS;
}

uint64_t ScatterAddWithSortedSimtTiling::GetTilingKey() const
{
    if (varShape_[0] * varShape_[1] * indicesNum_ == 0) {
        return GET_TPL_TILING_KEY(TPL_MODE_EMPTY, TPL_SCALAR_FALSE, TPL_DETERM_FALSE, TPL_ADDR_B32);
    }

    uint64_t isScalar = isUpdateScalar_ ? TPL_SCALAR_TRUE : TPL_SCALAR_FALSE;
    uint64_t isDeterm = isDeterminTemplate_ ? TPL_DETERM_TRUE : TPL_DETERM_FALSE;
    uint64_t addrType =
        ((varShape_[1] * indicesNum_ > INT32_MAX) || (varSize_ > INT32_MAX)) ? TPL_ADDR_B64 : TPL_ADDR_B32;
    return GET_TPL_TILING_KEY(TPL_MODE_SIMT, isScalar, isDeterm, addrType);
}

ge::graphStatus ScatterAddWithSortedSimtTiling::GetWorkspaceSize()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    workspacesSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    if (isDeterminTemplate_) {
        uint32_t userWorkspacesSize =
            (usedCoreNum_ * DOUBLE * varShape_[1]) * varTypeSize_ + usedCoreNum_ * DOUBLE * indicesDtypeSize_;
        userWorkspacesSize = Ops::Base::CeilAlign(userWorkspacesSize, static_cast<uint32_t>(indicesDtypeSize_));
        workspacesSize_ += userWorkspacesSize;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterAddWithSortedSimtTiling::PostTiling()
{
    OP_LOGD(context_->GetNodeName(), "ScatterAddWithSortedSimtTiling simt PostTiling enter.");
    auto workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspacesSize_;

    context_->SetBlockDim(usedCoreNum_);
    context_->SetScheduleMode(1);

    return ge::GRAPH_SUCCESS;
}

void ScatterAddWithSortedSimtTiling::DumpTilingInfo()
{
    std::ostringstream info;
    info << "tilingKey: " << GetTilingKey();
    info << ", UB Size: " << ubSize_;
    info << ", usedCoreNum: " << tilingData_->usedCoreNum;
    info << ", varShape[0]: " << tilingData_->varShape[0];
    info << ", varShape[1]: " << tilingData_->varShape[1];
    info << ", normBlockIndices: " << tilingData_->normBlockIndices;
    info << ", tailBlockIndices: " << tilingData_->tailBlockIndices;

    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

REGISTER_TILING_TEMPLATE("ScatterAddWithSorted", ScatterAddWithSortedSimtTiling, 1);
} // namespace optiling