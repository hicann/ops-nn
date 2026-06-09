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
 * \file adaptive_avg_pool2d_simt_tiling.cpp
 * \brief
 */

#include <cctype>
#include <algorithm>
#include "log/log.h"
#include "util/math_util.h"
#include "error_util.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "log/log.h"
#include "adaptive_avg_pool2d_simt_tiling.h"
#include "op_common/op_host/util/platform_util.h"
#include "platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include <iostream>

namespace optiling {
using namespace ge;
using namespace AdaptiveAvgPool2dOp;
constexpr uint64_t KERNEL_SIZE_RATIO = 8;
constexpr uint64_t KERNEL_SIZE_THRESHOLD = 64;

bool AdaptiveAvgPool2DTilingSimt::IsCapable() { return true; }

void AdaptiveAvgPool2DTilingSimt::SetTilingData()
{
    tilingData_->nDim = input_.nIn;
    tilingData_->cDim = input_.cIn;
    tilingData_->hInDim = input_.hIn;
    tilingData_->wInDim = input_.wIn;
    tilingData_->hOutDim = input_.hOut;
    tilingData_->wOutDim = input_.wOut;
}

ge::graphStatus AdaptiveAvgPool2DTilingSimt::DoOpTiling()
{
    const char* opName_ = "AdaptiveAvgPool2d";
    if (GetAndCheckDataFormat() != ge::GRAPH_SUCCESS) {
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(opName_, "x", "unknown_format", "GetDataFormatAttrInfo fail.");
        return ge::GRAPH_FAILED;
    }
    SetTilingData();

    int64_t outputSize = tilingData_->nDim * tilingData_->cDim * tilingData_->hOutDim * tilingData_->wOutDim;
    maxDivUseNum_ =
        std::max({tilingData_->hInDim * tilingData_->hOutDim, tilingData_->wInDim * tilingData_->wOutDim, outputSize});
    maxDivUseNum_ =
        std::max({maxDivUseNum_, tilingData_->nDim * tilingData_->cDim * tilingData_->hInDim * tilingData_->wInDim});
    threads_ = std::min(outputSize, MIN_THREAD);
    uint64_t kernelHMax = CalKernelSizeOneDimMax(input_.hIn, input_.hOut);
    uint64_t kernelWMax = CalKernelSizeOneDimMax(input_.wIn, input_.wOut);
    if (Ops::Base::FloorDiv(kernelHMax, kernelWMax) >= KERNEL_SIZE_RATIO &&
        kernelHMax * kernelWMax >= KERNEL_SIZE_THRESHOLD &&
        outputSize > MAX_THREAD * static_cast<int64_t>(input_.coreNum)) {
        threads_ = MAX_THREAD;
    }
    int64_t blockNum = Ops::Base::CeilDiv(outputSize, threads_);
    blockNum = std::min(blockNum, static_cast<int64_t>(input_.coreNum));
    context_->SetBlockDim(blockNum);
    tilingData_->threads = threads_;
    return ge::GRAPH_SUCCESS;
}

uint64_t AdaptiveAvgPool2DTilingSimt::GetTilingKey() const
{
    uint64_t divMode = static_cast<uint64_t>(maxDivUseNum_) < MAX_INT32 ? TPL_INT32_UINT32 : TPL_INT64_UINT64;
    return GET_TPL_TILING_KEY(TPL_SIMT_KERNEL, divMode, TPL_NC_FACTOR_64, TPL_BIG_KERNEL_NDDMA);
}

ge::graphStatus AdaptiveAvgPool2DTilingSimt::PostTiling()
{
    const char* opName_ = "AdaptiveAvgPool2d";
    int64_t ubSize = input_.ubSize - DCACHE_SIZE;
    auto res = context_->SetLocalMemorySize(ubSize);
    if (res != ge::GRAPH_SUCCESS) {
        OP_LOGE_FOR_INVALID_CONFIG_WITH_REASON(
            opName_, std::to_string(ubSize).c_str(), "ubSize", "AdaptiveAvgPool2d", "SetLocalMemorySize failed.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveAvgPool2DTilingSimt::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

void AdaptiveAvgPool2DTilingSimt::DumpTilingInfo()
{
    std::string str;
    str += " nDim:" + std::to_string(tilingData_->nDim);
    str += ", cDim:" + std::to_string(tilingData_->cDim);
    str += ", hInDim:" + std::to_string(tilingData_->hInDim);
    str += ", wInDim:" + std::to_string(tilingData_->wInDim);
    str += ", hOutDim:" + std::to_string(tilingData_->hOutDim);
    str += ", wOutDim:" + std::to_string(tilingData_->wOutDim);
    str += ", threads:" + std::to_string(tilingData_->threads);
    OP_LOGI(context_, "%s.", str.c_str());
}
REGISTER_OPS_TILING_TEMPLATE(AdaptiveAvgPool2d, AdaptiveAvgPool2DTilingSimt, 100);
} // namespace optiling