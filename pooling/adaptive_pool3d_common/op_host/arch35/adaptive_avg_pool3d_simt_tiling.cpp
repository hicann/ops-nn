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
 * \file adaptive_avg_pool3d_simt_tiling.cpp
 * \brief
 */

#include <cctype>
#include <algorithm>
#include "log/log.h"
#include "util/math_util.h"
#include "error_util.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "adaptive_avg_pool3d_simt_tiling.h"
#include "op_common/op_host/util/platform_util.h"
#include "platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"

using namespace ge;
using namespace AdaptivePool3DTiling;
namespace optiling{
    
bool AdaptiveAvgPool3DTilingSimt::IsCapable()
{
    return true;
}

void AdaptiveAvgPool3DTilingSimt::SetTilingData()
{
    if (input_.dataFormat == ge::Format::FORMAT_NDHWC) {
        tilingData_->cDim = input_.wIn;
        tilingData_->dInDim = input_.cIn;
        tilingData_->hInDim = input_.dIn;
        tilingData_->wInDim = input_.hIn;
    } else {
        tilingData_->cDim = input_.cIn;
        tilingData_->dInDim = input_.dIn;
        tilingData_->hInDim = input_.hIn;
        tilingData_->wInDim = input_.wIn;
    }
    tilingData_->nDim = input_.nIn;
    tilingData_->dOutDim = input_.dOut;
    tilingData_->hOutDim = input_.hOut;
    tilingData_->wOutDim = input_.wOut;
}

ge::graphStatus AdaptiveAvgPool3DTilingSimt::DoOpTiling()
{
    OP_TILING_CHECK(
        GetAndCheckDataFormat() != ge::GRAPH_SUCCESS,
        VECTOR_INNER_ERR_REPORT_TILIING(context_, "GetDataFormatAttrInfo fail."),
        return ge::GRAPH_FAILED);
    SetTilingData();

    int64_t outputSize = tilingData_->nDim * tilingData_->cDim * tilingData_->dOutDim * tilingData_->hOutDim * tilingData_->wOutDim;
    maxDivUseNum_ = std::max({tilingData_->dInDim * tilingData_->dOutDim, tilingData_->hInDim * tilingData_->hOutDim, tilingData_->wInDim * tilingData_->wOutDim, outputSize});
    maxDivUseNum_ = std::max({maxDivUseNum_, tilingData_->nDim * tilingData_->cDim * tilingData_->dInDim * tilingData_->hInDim * tilingData_->wInDim});
    int64_t threads = std::min(outputSize, MAX_THREAD_NUM);
    int64_t blockNum = Ops::Base::CeilDiv(outputSize, threads);
    blockNum = std::min(blockNum, static_cast<int64_t>(input_.coreNum));
    context_->SetBlockDim(blockNum);
    return ge::GRAPH_SUCCESS;
}

uint64_t AdaptiveAvgPool3DTilingSimt::GetTilingKey() const
{
    uint64_t formatMode = input_.dataFormat == ge::Format::FORMAT_NDHWC ? TPL_DATA_FORMAT_MODE_0 : TPL_DATA_FORMAT_MODE_1;
    uint64_t divMode = static_cast<uint64_t>(maxDivUseNum_) < MAX_INT32 ? TPL_INT32_UINT32 : TPL_INT64_UINT64;
    return GET_TPL_TILING_KEY(TPL_MODE_2, divMode, TPL_MULTI_MODE_0, formatMode);
}

ge::graphStatus AdaptiveAvgPool3DTilingSimt::PostTiling()
{
    int64_t ubSize = input_.ubSize - DCACHE_SIZE;
    auto res = context_->SetLocalMemorySize(ubSize);
    OP_TILING_CHECK((res != ge::GRAPH_SUCCESS),
                    VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "SetLocalMemorySize ubSize = %ld failed.", ubSize),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveAvgPool3DTilingSimt::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

void AdaptiveAvgPool3DTilingSimt::DumpTilingInfo()
{
    std::string str;
    str += " nDim:" + std::to_string(tilingData_->nDim);
    str += ", cDim:" + std::to_string(tilingData_->cDim);
    str += ", dInDim:" + std::to_string(tilingData_->dInDim);
    str += ", hInDim:" + std::to_string(tilingData_->hInDim);
    str += ", wInDim:" + std::to_string(tilingData_->wInDim);
    str += ", dOutDim:" + std::to_string(tilingData_->dOutDim);
    str += ", hOutDim:" + std::to_string(tilingData_->hOutDim);
    str += ", wOutDim:" + std::to_string(tilingData_->wOutDim);
    OP_LOGI(context_, "%s.", str.c_str());
}
REGISTER_OPS_TILING_TEMPLATE(AdaptiveAvgPool3d, AdaptiveAvgPool3DTilingSimt, 2);
}  // namespace optiling