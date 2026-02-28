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
 * \file max_pool_with_argmax_v3_simt_tiling.cpp
 * \brief
 */
#include <cctype>
#include <algorithm>
#include "op_host/tiling_templates_registry.h"
#include "max_pool_with_argmax_v3_simt_tiling.h"

using namespace AscendC;
using namespace ge;

namespace optiling {

ge::graphStatus MaxPoolWithArgmaxV3TilingSIMT::GetShapeAttrsInfo()
{
    auto runtimeAttrs = context_->GetAttrs();
    const char* data_format = runtimeAttrs->GetAttrPointer<char>(FORMAT_POS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, data_format);
    inputData.data_format = data_format;
    std::transform(
        inputData.data_format.begin(), inputData.data_format.end(), inputData.data_format.begin(),
        [](unsigned char c) { return std::tolower(c); });
    OP_CHECK_IF(
        !(inputData.data_format == "nchw" || inputData.data_format == "nhwc"),
        OP_LOGE(context_, "ATTR data_format is %s ,expect [NCHW] or [NHWC].", data_format), return ge::GRAPH_FAILED);
    auto inputX = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputX);
    auto inputShape = Ops::Base::EnsureNotScalar(inputX->GetStorageShape());

    auto outX = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outX);
    auto outShape = Ops::Base::EnsureNotScalar(outX->GetStorageShape());

    auto indicesX = context_->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indicesX);
    auto indicesShape = Ops::Base::EnsureNotScalar(indicesX->GetStorageShape());

    if (inputShape.GetDimNum() != NCHW_DIMS) {
        OP_LOGE(
            context_->GetNodeName(), "MaxPoolWithArgmaxV3: input shape dim = %zu, should be equal 4",
            inputShape.GetDimNum());
        return ge::GRAPH_FAILED;
    }
    if (inputData.data_format == "nhwc") {
        nDimPos = 0;
        cDimPos = 3;
        hDimPos = 1;
        wDimPos = 2;
    }
    inputData.inputShape = array<uint64_t, NCHW_DIMS>{
        uint64_t(inputShape.GetDim(nDimPos)), uint64_t(inputShape.GetDim(cDimPos)),
        uint64_t(inputShape.GetDim(hDimPos)), uint64_t(inputShape.GetDim(wDimPos))};
    inputData.outShape = array<uint64_t, NCHW_DIMS>{
        uint64_t(inputShape.GetDim(nDimPos)), uint64_t(inputShape.GetDim(cDimPos)), uint64_t(outShape.GetDim(hDimPos)),
        uint64_t(outShape.GetDim(wDimPos))};
    auto inputDesc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
    dtype = inputDesc->GetDataType();
    if (dtype != ge::DataType::DT_BF16 && dtype != ge::DataType::DT_FLOAT16 && dtype != ge::DataType::DT_FLOAT) {
        OP_LOGE(context_->GetNodeName(), "MaxPoolWithArgmaxV3: invalid dtype");
        return ge::GRAPH_FAILED;
    }
    if (indicesShape != outShape) {
        OP_LOGE(context_->GetNodeName(), "MaxPoolWithArgmaxV3: indices shape and values shape is different");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_NULL_WITH_CONTEXT(context_, runtimeAttrs);
    const gert::TypedContinuousVector<int64_t>* kernelSize = runtimeAttrs->GetListInt(KERNEL_POS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, kernelSize);
    inputData.kernelSize =
        array<uint64_t, HW_DIMS>{uint64_t(*(kernelSize->GetData())), uint64_t(*(kernelSize->GetData() + 1))};
    const gert::TypedContinuousVector<int64_t>* stride = runtimeAttrs->GetListInt(STRIDE_POS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, stride);
    inputData.stride = array<uint64_t, HW_DIMS>{uint64_t(*(stride->GetData())), uint64_t(*(stride->GetData() + 1))};
    const gert::TypedContinuousVector<int64_t>* padding = runtimeAttrs->GetListInt(PADDING_POS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, padding);
    inputData.pad = array<uint64_t, HW_DIMS>{uint64_t(*(padding->GetData())), uint64_t(*(padding->GetData() + 1))};
    const gert::TypedContinuousVector<int64_t>* dilation = runtimeAttrs->GetListInt(DILATION_POS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dilation);
    inputData.dilation =
        array<uint64_t, HW_DIMS>{uint64_t(*(dilation->GetData())), uint64_t(*(dilation->GetData() + 1))};
    inputData.ceilMode = *runtimeAttrs->GetAttrPointer<bool>(CEIL_POS);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPoolWithArgmaxV3TilingSIMT::DoOpTiling()
{
    tiling.set_nDim(inputData.inputShape[N_DIM_]);
    tiling.set_cDim(inputData.inputShape[C_DIM_]);
    tiling.set_hInDim(inputData.inputShape[H_DIM_]);
    tiling.set_wInDim(inputData.inputShape[W_DIM_]);
    tiling.set_hOutDim(inputData.outShape[H_DIM_]);
    tiling.set_wOutDim(inputData.outShape[W_DIM_]);
    tiling.set_kSizeH(inputData.kernelSize[H_IDX_]);
    tiling.set_kSizeW(inputData.kernelSize[W_IDX_]);
    tiling.set_stridesH(inputData.stride[H_IDX_]);
    tiling.set_stridesW(inputData.stride[W_IDX_]);
    tiling.set_padH(inputData.pad[H_IDX_]);
    tiling.set_padW(inputData.pad[W_IDX_]);
    tiling.set_dilationH(inputData.dilation[H_IDX_]);
    tiling.set_dilationW(inputData.dilation[W_IDX_]);
    tiling.set_ceilMode(inputData.ceilMode);
    outputDataCount = tiling.get_nDim() * tiling.get_cDim() * tiling.get_hOutDim() * tiling.get_wOutDim();
    int64_t threads = std::min(outputDataCount, MAX_THREAD_NUM);
    int64_t blockNum = Ops::Base::CeilDiv(outputDataCount, threads);
    blockNum = std::min(blockNum, static_cast<int64_t>(coreNum));
    context_->SetBlockDim(blockNum);
    tiling.set_threadNums(threads);
    tiling.set_blockNums(blockNum);
    OP_LOGI(context_->GetNodeName(), "%s", ToString(tiling).c_str());
    return ge::GRAPH_SUCCESS;
}

uint64_t MaxPoolWithArgmaxV3TilingSIMT::GetTilingKey() const
{
    if (inputData.data_format == "nchw" && outputDataCount <= MAX_INT32) {
        return SIMT_NCHW_TILING_KEY_INT32;
    } else if (inputData.data_format == "nhwc" && outputDataCount <= MAX_INT32) {
        return SIMT_NHWC_TILING_KEY_INT32;
    } else if (inputData.data_format == "nchw" && outputDataCount > MAX_INT32) {
        return SIMT_NCHW_TILING_KEY_INT64;
    } else if (inputData.data_format == "nhwc" && outputDataCount > MAX_INT32) {
        return SIMT_NHWC_TILING_KEY_INT64;
    }
    return SIMT_NCHW_TILING_KEY_INT32;
}

ge::graphStatus MaxPoolWithArgmaxV3TilingSIMT::PostTiling()
{
    OP_CHECK_IF(
        context_->GetRawTilingData()->GetCapacity() < tiling.GetDataSize(),
        OP_LOGE(
            context_, "tiling data's[%zu] is larger than capacity[%zu].", tiling.GetDataSize(),
            context_->GetRawTilingData()->GetCapacity()),
        return ge::GRAPH_FAILED);
    tiling.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

std::string MaxPoolWithArgmaxV3TilingSIMT::ToString(MaxPoolWithArgmaxV3SimtTilingData& tiling)
{
    std::string str;
    str += " threadNums:" + std::to_string(tiling.get_threadNums());
    str += " blockNums:" + std::to_string(tiling.get_blockNums());
    str += " nDim:" + std::to_string(tiling.get_nDim());
    str += " cDim:" + std::to_string(tiling.get_cDim());
    str += " hInDim:" + std::to_string(tiling.get_hInDim());
    str += " wInDim:" + std::to_string(tiling.get_wInDim());
    str += " hOutDim:" + std::to_string(tiling.get_hOutDim());
    str += " wOutDim:" + std::to_string(tiling.get_wOutDim());
    str += " kSizeH:" + std::to_string(tiling.get_kSizeH());
    str += " kSizeW:" + std::to_string(tiling.get_kSizeW());
    str += " stridesH:" + std::to_string(tiling.get_stridesH());
    str += " stridesW:" + std::to_string(tiling.get_stridesW());
    str += " padH:" + std::to_string(tiling.get_padH());
    str += " padW:" + std::to_string(tiling.get_padW());
    str += " dilationH:" + std::to_string(tiling.get_dilationH());
    str += " dilationW:" + std::to_string(tiling.get_dilationW());
    str += " ceilMode:" + std::to_string(tiling.get_ceilMode());
    return str;
}
REGISTER_OPS_TILING_TEMPLATE(MaxPoolWithArgmaxV3, MaxPoolWithArgmaxV3TilingSIMT, 100);
} // namespace optiling