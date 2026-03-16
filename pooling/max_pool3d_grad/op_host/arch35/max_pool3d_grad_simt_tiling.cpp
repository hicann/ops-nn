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
 * \file max_pool3d_grad_simt_tiling.cpp
 * \brief
 */

#include <cctype>
#include <algorithm>
#include "log/log.h"
#include "util/math_util.h"
#include "error_util.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "max_pool3d_grad_tiling.h"
#include "op_common/op_host/util/platform_util.h"
#include "platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "platform/platform_info.h"
#include "op_host/tiling_util.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace optiling {

const int INPUT_IDX_X = 0;
static const gert::Shape g_vec_1_shape = {1};
static constexpr uint64_t DCACHE_SIZE = 128 *1024UL;
static constexpr int64_t DIGIT_TWO = 2;

static const gert::Shape &EnsureNotScalar(const gert::Shape &inShape) {
    if (inShape.IsScalar()) {
        return g_vec_1_shape;
    }
    return inShape;
}

bool MaxPool3DGradSimtTiling::IsCapable()
{
    return true;
}

ge::graphStatus MaxPool3DGradSimtTiling::GetPlatformInfo()
{
    auto platformPtr = context_->GetPlatformInfo();
    if (platformPtr == nullptr) {
        auto compileInfoPtr =
            reinterpret_cast<const Tiling4Pool3DGradCompileInfo*>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_->GetNodeName(), "compile info is null"),
                    return ge::GRAPH_FAILED);
        coreNum = compileInfoPtr->totalCoreNum;
        ubSize = compileInfoPtr->maxUbSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformPtr);
        coreNum = ascendcPlatform.GetCoreNumAiv();

        uint64_t ubSizePlatform;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
        ubSize = static_cast<int64_t>(ubSizePlatform);
    }

    OP_CHECK_IF(coreNum == 0, OP_LOGE(context_->GetNodeName(), "coreNum is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPool3DGradSimtTiling::GetShapeAttrsInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    if (!Ops::NN::OpTiling::IsRegbaseSocVersion(context_)) {
        return ge::GRAPH_PARAM_INVALID;
    }
    auto runtimeAttrs = context_->GetAttrs();
    const char* data_format = runtimeAttrs->GetAttrPointer<char>(FORMAT_POS);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, data_format);
    inputData.data_format = data_format;
    std::transform(inputData.data_format.begin(), inputData.data_format.end(), inputData.data_format.begin(),
                    [](unsigned char c) { return std::tolower(c); });
    OP_CHECK_IF(!(inputData.data_format == "ndhwc" || inputData.data_format == "ncdhw"),
                OP_LOGE(context_, "ATTR data_format is %s, expect [NDHWC] or [NCDHW].", data_format),
                return ge::GRAPH_FAILED);
    auto inputX = context_->GetInputShape(FIRPOS);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, inputX);
    auto inputShape = EnsureNotScalar(inputX->GetStorageShape());

    auto inputGrad = context_->GetInputShape(THIPOS);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, inputGrad);
    auto gradShape = EnsureNotScalar(inputGrad->GetStorageShape());

    auto outX = context_->GetOutputShape(FIRPOS);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, outX);
    auto outShape = EnsureNotScalar(outX->GetStorageShape());
    outShapeSize = outShape.GetShapeSize();

    if (inputShape.GetDimNum() != NCDHW_DIMS) {
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                        "MaxPool3DGrad: input shape dim = %zu, should be equal 5",
                                        inputShape.GetDimNum());
        return ge::GRAPH_FAILED;
    }

    if (inputData.data_format == "ndhwc") {
        nDimPos = 0;
        cDimPos = 4;
        dDimPos = 1;
        hDimPos = 2;
        wDimPos = 3;
    }

    inputData.inputShape = 
        std::array<uint64_t, NCDHW_DIMS>{uint64_t(inputShape.GetDim(nDimPos)), uint64_t(inputShape.GetDim(cDimPos)),
                                        uint64_t(inputShape.GetDim(dDimPos)), uint64_t(inputShape.GetDim(hDimPos)), uint64_t(inputShape.GetDim(wDimPos))};
    inputData.outShape = 
        std::array<uint64_t, NCDHW_DIMS>{uint64_t(inputShape.GetDim(nDimPos)), uint64_t(inputShape.GetDim(cDimPos)),
                                        uint64_t(outShape.GetDim(dDimPos)), uint64_t(outShape.GetDim(hDimPos)), uint64_t(outShape.GetDim(wDimPos))};
    inputData.gradShape = 
        std::array<uint64_t, NCDHW_DIMS>{uint64_t(gradShape.GetDim(nDimPos)), uint64_t(gradShape.GetDim(cDimPos)),
                                        uint64_t(gradShape.GetDim(dDimPos)), uint64_t(gradShape.GetDim(hDimPos)), uint64_t(gradShape.GetDim(wDimPos))};
    auto inputDesc = context_->GetInputDesc(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
    dtype = inputDesc->GetDataType();
    if (dtype != ge::DataType::DT_BF16 && dtype != ge::DataType::DT_FLOAT16 && dtype != ge::DataType::DT_FLOAT) {
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "MaxPool3DGrad: invalid dtype %s, should be BFloat16、Float16 or Float32", Ops::Base::ToString(dtype).c_str());
        return ge::GRAPH_FAILED;
    }
    OPS_CHECK_NULL_WITH_CONTEXT(context_, runtimeAttrs);
    const gert::TypedContinuousVector<int64_t>* kernelSize = runtimeAttrs->GetListInt(KERNEL_POS);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, kernelSize);
    int32_t kSizeD = kernelSize->GetData()[dDimPos];
    int32_t kSizeH = kernelSize->GetData()[hDimPos];
    int32_t kSizeW = kernelSize->GetData()[wDimPos];
    inputData.kernelSize = std::array<uint64_t, DHW_DIMS>{uint64_t(kSizeD), uint64_t(kSizeH), uint64_t(kSizeW)};
    const gert::TypedContinuousVector<int64_t>* stride = runtimeAttrs->GetListInt(STRIDE_POS);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, stride);
    int32_t strideD = stride->GetData()[dDimPos];
    int32_t strideH = stride->GetData()[hDimPos];
    int32_t strideW = stride->GetData()[wDimPos];
    inputData.stride = std::array<uint64_t, DHW_DIMS>{strideD, strideH, strideW};
    const char* padMode = runtimeAttrs->GetAttrPointer<char>(PADDING_MODE_POS);
    std::string padModeStr = padMode;
    uint32_t padsD = 0;
    uint32_t padsH = 0;
    uint32_t padsW = 0;
    if (padModeStr == "SAME") {
        int64_t dPadNeed = std::max(int64_t{0}, int64_t((gradShape.GetDim(dDimPos) - 1) * strideD +
                                                        kSizeD - inputShape.GetDim(dDimPos)));
        padsD = dPadNeed / DIGIT_TWO;
        int64_t hPadNeed = std::max(int64_t{0}, int64_t((gradShape.GetDim(hDimPos) - 1) * strideH +
                                                        kSizeH - inputShape.GetDim(hDimPos)));
        padsH = hPadNeed / DIGIT_TWO;
        int64_t wPadNeed = std::max(int64_t{0}, int64_t((gradShape.GetDim(wDimPos) - 1) * strideW +
                                                        kSizeW - inputShape.GetDim(wDimPos)));
        padsW = wPadNeed / DIGIT_TWO;
    }
    OP_CHECK_IF((padsD * DOUB > kSizeD || padsH * DOUB > kSizeH || padsW * DOUB > kSizeW),
                OP_LOGE(context_, "pad should be smaller than or equal to half of kernel size, pad shape is [%d, %d, %d], kernel shape is [%d, %d, %d]",
                        padsD, padsH, padsW, kSizeD, kSizeH, kSizeW),
                return ge::GRAPH_FAILED);
    inputData.pad = std::array<uint64_t, DHW_DIMS>{uint64_t(padsD), uint64_t(padsH), uint64_t(padsW)};
    inputData.dilation = std::array<uint64_t, DHW_DIMS>{1, 1, 1};
    inputData.ceilMode = false;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPool3DGradSimtTiling::DoOpTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPool3DGradSimtTiling::DoLibApiTiling()
{
    tilingData_->nDim = inputData.inputShape[N_DIM_];
    tilingData_->cDim = inputData.inputShape[C_DIM_];
    tilingData_->dInDim = inputData.inputShape[D_DIM_];
    tilingData_->hInDim = inputData.inputShape[H_DIM_];
    tilingData_->wInDim = inputData.inputShape[W_DIM_];
    tilingData_->dOutDim = inputData.gradShape[D_DIM_];
    tilingData_->hOutDim = inputData.gradShape[H_DIM_];
    tilingData_->wOutDim = inputData.gradShape[W_DIM_];
    tilingData_->kSizeD = inputData.kernelSize[D_IDX_];
    tilingData_->kSizeH = inputData.kernelSize[H_IDX_];
    tilingData_->kSizeW = inputData.kernelSize[W_IDX_];
    tilingData_->stridesD = inputData.stride[D_IDX_];
    tilingData_->stridesH = inputData.stride[H_IDX_];
    tilingData_->stridesW = inputData.stride[W_IDX_];
    tilingData_->padD = inputData.pad[D_IDX_];
    tilingData_->padH = inputData.pad[H_IDX_];
    tilingData_->padW = inputData.pad[W_IDX_];
    tilingData_->dilationD = inputData.dilation[D_IDX_];
    tilingData_->dilationH = inputData.dilation[H_IDX_];
    tilingData_->dilationW = inputData.dilation[W_IDX_];
    tilingData_->ceilMode = inputData.ceilMode;
    int64_t outputDataCount = tilingData_->nDim * tilingData_->cDim * tilingData_->dOutDim * tilingData_->hOutDim * tilingData_->wOutDim;
    int64_t threads = std::min(outputDataCount, MAX_THREAD_NUM);
    int64_t blockNum = Ops::Base::CeilDiv(outputDataCount, threads);
    blockNum = std::min(blockNum, static_cast<int64_t>(coreNum));
    context_->SetBlockDim(blockNum);
    tilingData_->threadNums = threads;
    tilingData_->blockNums = blockNum;
    return ge::GRAPH_SUCCESS;
}

uint64_t MaxPool3DGradSimtTiling::GetTilingKey() const
{
    int64_t outDataCountGrad = inputData.inputShape[N_DIM_] * inputData.inputShape[C_DIM_] * inputData.inputShape[D_DIM_] *
                           inputData.inputShape[H_DIM_] * inputData.inputShape[W_DIM_];
    uint32_t idxDtype = outDataCountGrad <= static_cast<int64_t>(MAX_INT32) ? TPL_INT32 : TPL_INT64;
    uint32_t isChannelLast = (inputData.data_format == "ncdhw") ? 0 : 1;
    uint32_t isSimt = 1;
    uint32_t isCheckRange = 0;
    int64_t outDataCountPos = inputData.inputShape[N_DIM_] * inputData.inputShape[C_DIM_] * inputData.gradShape[D_DIM_] *
                           inputData.gradShape[H_DIM_] * inputData.gradShape[W_DIM_];
    uint32_t useINT64Index = outDataCountPos <= MAX_INT32 ? 0 : 1;
    return GET_TPL_TILING_KEY(idxDtype, isSimt, isChannelLast, isCheckRange, useINT64Index);
}

ge::graphStatus MaxPool3DGradSimtTiling::GetWorkspaceSize()
{
    auto sys_workspace = WS_SYS_SIZE;
    size_t argmaxSize = outShapeSize * 8;
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = argmaxSize + sys_workspace;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPool3DGradSimtTiling::PostTiling()
{
    context_->SetLocalMemorySize(ubSize - DCACHE_SIZE);
    return ge::GRAPH_SUCCESS;
}

void MaxPool3DGradSimtTiling::DumpTilingInfo()
{
    std::string str;
    str += "threadNums:" + std::to_string(tilingData_->threadNums);
    str += "blockNums:" + std::to_string(tilingData_->blockNums);
    str += "nDim:" + std::to_string(tilingData_->nDim);
    str += "cDim:" + std::to_string(tilingData_->cDim);
    str += "dInDim:" + std::to_string(tilingData_->dInDim);
    str += "hInDim:" + std::to_string(tilingData_->hInDim);
    str += "wInDim:" + std::to_string(tilingData_->wInDim);
    str += "dOutDim:" + std::to_string(tilingData_->dOutDim);
    str += "hOutDim:" + std::to_string(tilingData_->hOutDim);
    str += "wOutDim:" + std::to_string(tilingData_->wOutDim);
    str += "kSizeD:" + std::to_string(tilingData_->kSizeD);
    str += "kSizeH:" + std::to_string(tilingData_->kSizeH);
    str += "kSizeW:" + std::to_string(tilingData_->kSizeW);
    str += "stridesD:" + std::to_string(tilingData_->stridesD);
    str += "stridesH:" + std::to_string(tilingData_->stridesH);
    str += "stridesW:" + std::to_string(tilingData_->stridesW);
    str += "padD:" + std::to_string(tilingData_->padD);
    str += "padH:" + std::to_string(tilingData_->padH);
    str += "padW:" + std::to_string(tilingData_->padW);
    str += "dilationD:" + std::to_string(tilingData_->dilationD);
    str += "dilationH:" + std::to_string(tilingData_->dilationH);
    str += "dilationW:" + std::to_string(tilingData_->dilationW);
    str += "ceilMode:" + std::to_string(tilingData_->ceilMode);
    OP_LOGI(context_, "%s", str.c_str());
}

REGISTER_TILING_TEMPLATE("MaxPool3DGrad", MaxPool3DGradSimtTiling, 6);

} // namespace optiling
