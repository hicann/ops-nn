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
 * \file max_pool3d_with_argmax_v2_tiling_big_kernel_regbase.cpp
 * \brief big kernel imply for max_pool3d_with_argmax_v2
 */

#include "tiling_base/tiling_util.h"
#include "max_pool3d_with_argmax_v2_tiling_big_kernel_regbase.h"
#include "error_util.h"

using namespace std;
static const gert::Shape g_vec_1_shape = {1};

namespace optiling
{
static const gert::Shape &EnsureNotScalar(const gert::Shape &inShape) {
  if (inShape.IsScalar()) {
    return g_vec_1_shape;
  }
  return inShape;
}

ge::graphStatus MaxPool3DWithArgmaxV2BigKernelRegbaseTiling::GetShapeAttrsInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    if (!Ops::NN::OpTiling::IsRegbaseSocVersion(context_)){
        return ge::GRAPH_PARAM_INVALID;
    }
    auto inputX = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputX);
    auto inputShape = EnsureNotScalar(inputX->GetStorageShape());

    OP_CHECK_IF(inputShape.GetDimNum() != NCDHW_DIMS,
                    OP_LOGE(context_->GetNodeName(),
                                                    "MaxPool3DWithArgmaxV2: input shape dim = %zu, should be equal 5",
                                                    inputShape.GetDimNum()),
                    return ge::GRAPH_FAILED);
    OP_CHECK_IF(inputShape.GetShapeSize() <= 0,
                    OP_LOGE(context_->GetNodeName(),
                                                    "MaxPool3DWithArgmaxV2: input shape size %ld less than zero failed",
                                                    inputShape.GetShapeSize()),
                    return ge::GRAPH_FAILED);
    auto inputDesc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
    dtype = inputDesc->GetDataType();
    if (dtype != ge::DataType::DT_BF16 && dtype != ge::DataType::DT_FLOAT16 && dtype != ge::DataType::DT_FLOAT) {
        OP_LOGE(context_->GetNodeName(), "MaxPool3DWithArgmaxV2: invalid dtype");
        return ge::GRAPH_FAILED;
    }

    auto outX = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outX);
    auto outShape = EnsureNotScalar(outX->GetStorageShape());
    auto indicesX = context_->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indicesX);
    auto indicesShape = EnsureNotScalar(indicesX->GetStorageShape());
    if (indicesShape != outShape) {
        OP_LOGE(context_->GetNodeName(),
                                        "MaxPool3DWithArgmaxV2: indices shape and values shape is different");
        return ge::GRAPH_FAILED;
    }
    auto runtimeAttrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, runtimeAttrs);

    std::string inputFormatStr("NCDHW");
    const char* inputFormat = runtimeAttrs->GetAttrPointer<char>(FORMAT_POS);
    if (inputFormat != nullptr) {
        inputFormatStr = inputFormat;
    }
    int d_dim = MP_MAX_3D_DIM_TWO, h_dim = MP_MAX_3D_DIM_THREE, w_dim = MP_MAX_3D_DIM_FOUR;
    if (inputFormatStr == "NCDHW") {
        inputData.inputFormat = ge::Format::FORMAT_NCDHW;
        inputData.batches = inputShape.GetDim(MP_MAX_3D_DIM_ZERO) * inputShape.GetDim(MP_MAX_3D_DIM_ONE);
        inputData.nInput = inputShape.GetDim(MP_MAX_3D_DIM_ZERO);
        inputData.cInput = inputShape.GetDim(MP_MAX_3D_DIM_ONE);
    } else {
        return ge::GRAPH_PARAM_INVALID;
    }

    OP_CHECK_IF(outShape.GetDim(d_dim) < 1 || outShape.GetDim(h_dim) < 1 || outShape.GetDim(w_dim) < 1,
                    OP_LOGE(context_->GetNodeName(),
                                                    "MaxPool3DWithArgmaxV2: output shape [%ld, %ld, %ld] not support",
                                                    outShape.GetDim(d_dim), outShape.GetDim(h_dim), outShape.GetDim(w_dim)),
                    return ge::GRAPH_FAILED);

    inputData.inputShape =
        array<uint64_t, DHW_DIMS>{uint64_t(inputShape.GetDim(d_dim)), uint64_t(inputShape.GetDim(h_dim)), uint64_t(inputShape.GetDim(w_dim))};
    inputData.outShape = array<uint64_t, DHW_DIMS>{uint64_t(outShape.GetDim(d_dim)), uint64_t(outShape.GetDim(h_dim)), uint64_t(outShape.GetDim(w_dim))};

    int32_t dValue = 0;
    int32_t hValue = 0;
    int32_t wValue = 0;
    const gert::TypedContinuousVector<int64_t>* kernelSize = runtimeAttrs->GetListInt(KERNEL_POS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, kernelSize);
    dValue = *(kernelSize->GetData());
    hValue = *(kernelSize->GetData() + 1);
    wValue = *(kernelSize->GetData() + 2);
    inputData.kernelSize = array<uint64_t, DHW_DIMS>{uint64_t(dValue), uint64_t(hValue), uint64_t(wValue)};
    OP_CHECK_IF(
        dValue <= 0 || hValue <= 0 || wValue <= 0,
        OP_LOGE(context_->GetNodeName(),
                                        "MaxPool3DWithArgmaxV2: not support kernel shape [%d, %d, %d]", dValue, hValue, wValue),
        return ge::GRAPH_FAILED);

    int32_t kdValue = dValue;
    int32_t khValue = hValue;
    int32_t kwValue = wValue;
    const gert::TypedContinuousVector<int64_t>* stride = runtimeAttrs->GetListInt(STRIDE_POS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, stride);
    dValue = *(stride->GetData());
    hValue = *(stride->GetData() + 1);
    wValue = *(stride->GetData() + 2);
    inputData.stride = array<uint64_t, DHW_DIMS>{uint64_t(dValue), uint64_t(hValue), uint64_t(wValue)};
    OP_CHECK_IF(
        dValue <= 0 || hValue <= 0 || wValue <= 0,
        OP_LOGE(context_->GetNodeName(),
                                        "MaxPool3DWithArgmaxV2: not support stride shape [%d, %d, %d]", dValue, hValue, wValue),
        return ge::GRAPH_FAILED);

    const gert::TypedContinuousVector<int64_t>* padding = runtimeAttrs->GetListInt(PADDING_POS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, padding);
    dValue = *(padding->GetData());
    hValue = *(padding->GetData() + 1);
    wValue = *(padding->GetData() + 2);
    inputData.pad = array<uint64_t, DHW_DIMS>{uint64_t(dValue), uint64_t(hValue), uint64_t(wValue)};
    OP_CHECK_IF(
        dValue > kdValue / 2 || hValue > khValue / 2 || wValue > kwValue / 2,
        OP_LOGE(context_->GetNodeName(),
                                        "MaxPool3DWithArgmaxV2: not support pad shape [%d, %d, %d] kernel shape [%d, %d, %d]",
                                        dValue, hValue, wValue, kdValue, khValue, kwValue),
        return ge::GRAPH_FAILED);

    inputData.dilation = array<uint64_t, DHW_DIMS>{1, 1, 1};
    dValue = 1;
    hValue = 1;
    wValue = 1;
    const gert::TypedContinuousVector<int64_t>* dilation = runtimeAttrs->GetListInt(DILATION_POS);
    if (dilation != nullptr) {
        dValue = *(dilation->GetData());
        hValue = *(dilation->GetData() + 1);
        wValue = *(dilation->GetData() + 2);
        inputData.dilation = array<uint64_t, DHW_DIMS>{uint64_t(dValue), uint64_t(hValue), uint64_t(wValue)};
        OP_CHECK_IF(
            dValue <= 0 || hValue <= 0 || wValue <= 0,
            OP_LOGE(context_->GetNodeName(),
                                            "MaxPool3DWithArgmaxV2: not support dilation shape [%d, %d, %d]", dValue, hValue, wValue),
            return ge::GRAPH_FAILED);
    }

    inputData.ceilMode = false;
    const bool* ceilModePtr = runtimeAttrs->GetAttrPointer<bool>(CEIL_POS);
    if (ceilModePtr != nullptr) {
        inputData.ceilMode = *ceilModePtr;
    }

    int indexDtype = THREE;
    const int* indexDtypePtr = runtimeAttrs->GetAttrPointer<int>(DTYPE_POS);
    if (indexDtypePtr != nullptr) {
        indexDtype = *indexDtypePtr;
    }
    switch (indexDtype) {
        case MP_MAX_3D_TYPE_INT32:
            inputData.indexDtype = ge::DataType::DT_INT32;
            break;
        case MP_MAX_3D_TYPE_INT64:
            inputData.indexDtype = ge::DataType::DT_INT64;
            break;
        default:
            inputData.indexDtype = ge::DataType::DT_INT32;
            break;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPool3DWithArgmaxV2BigKernelRegbaseTiling::GetPlatformInfo()
{
    auto platformPtr = context_->GetPlatformInfo();
    if (platformPtr == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const MaxPool3DWithArgmaxV2CompileInfo*>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, CUBE_INNER_ERR_REPORT(context_, "compile info is null"),
                        return ge::GRAPH_FAILED);
        coreNum = compileInfoPtr->coreNum;
        ubSize = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformPtr);
        coreNum = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatform;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
        ubSize = static_cast<int64_t>(ubSizePlatform);
    }
    OP_CHECK_IF(coreNum == 0, CUBE_INNER_ERR_REPORT(context_, "coreNum is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

bool MaxPool3DWithArgmaxV2BigKernelRegbaseTiling::IsCapable()
{
    int64_t ubAvailable = ubSize - (BYTES_FOUR + BYTES_EIGHT) * OUT_BUFFER_LEN;
    maxCount_ = ubAvailable / BUFFER_NUM;
    int64_t vRegSize = Ops::Base::GetVRegSize(context_);
    maxCount_ = Ops::Base::FloorAlign(maxCount_, vRegSize);
    int64_t dtypeSize = ge::GetSizeByDataType(dtype);
    OP_CHECK_IF(
        dtypeSize <= 0,
        OP_LOGE(context_, "dtypeSize must be greater than 0, dtypeSize: %ld", dtypeSize),
        return false);
    if (dtypeSize != 0) {
        maxCount_ = maxCount_ / dtypeSize;
    }
    if (inputData.dilation[D_DIM] == 1 && inputData.dilation[H_DIM] == 1 && inputData.dilation[W_DIM] == 1 && maxCount_ > MIN_COUNT &&
        inputData.inputFormat == ge::Format::FORMAT_NCDHW && inputData.kernelSize[W_DIM] * dtypeSize >= KW_THRESHOLD) {
        return true;
    }
    return false;     
}

void MaxPool3DWithArgmaxV2BigKernelRegbaseTiling::DoUBTiling()
{
    totalIdx_ = inputData.batches * inputData.outShape[D_DIM] * inputData.outShape[H_DIM] * inputData.outShape[W_DIM];
    blockFactor_ = totalIdx_ / coreNum;
    blockTail_ = totalIdx_ % coreNum;
    if (blockFactor_ == 0) {
        coreNums_ = totalIdx_;
    } else {
        coreNums_ = coreNum;
    }
    isSigOut_ = (inputData.outShape[D_DIM] == 1 && inputData.outShape[H_DIM] == 1 && inputData.outShape[W_DIM] == 1) ? 1 : 0;
}

void MaxPool3DWithArgmaxV2BigKernelRegbaseTiling::SetTilingData()
{
    tilingData_->dOutDim = inputData.outShape[D_DIM];
    tilingData_->hOutDim = inputData.outShape[H_DIM];
    tilingData_->wOutDim = inputData.outShape[W_DIM];
    tilingData_->dInDim = inputData.inputShape[D_DIM];
    tilingData_->hInDim = inputData.inputShape[H_DIM];
    tilingData_->wInDim = inputData.inputShape[W_DIM];
    tilingData_->kD = inputData.kernelSize[D_DIM];
    tilingData_->kH = inputData.kernelSize[H_DIM];
    tilingData_->kW = inputData.kernelSize[W_DIM];
    tilingData_->sD = inputData.stride[D_DIM];
    tilingData_->sH = inputData.stride[H_DIM];
    tilingData_->sW = inputData.stride[W_DIM];
    tilingData_->pD = inputData.pad[D_DIM];
    tilingData_->pH = inputData.pad[H_DIM];
    tilingData_->pW = inputData.pad[W_DIM];
    tilingData_->dD = inputData.dilation[D_DIM];
    tilingData_->dH = inputData.dilation[H_DIM];
    tilingData_->dW = inputData.dilation[W_DIM];
    tilingData_->blockFactor = blockFactor_;
    tilingData_->blockTail = blockTail_;
    tilingData_->totalIdx = totalIdx_;
    tilingData_->coreNums = coreNums_;
    tilingData_->maxCount = maxCount_;
    tilingData_->isSigOut = isSigOut_;
}

ge::graphStatus MaxPool3DWithArgmaxV2BigKernelRegbaseTiling::DoOpTiling()
{
    DoUBTiling();
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPool3DWithArgmaxV2BigKernelRegbaseTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPool3DWithArgmaxV2BigKernelRegbaseTiling::GetWorkspaceSize()
{
    auto sys_workspace = WS_SYS_SIZE;
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = sys_workspace;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPool3DWithArgmaxV2BigKernelRegbaseTiling::PostTiling()
{
    context_->SetBlockDim(coreNums_);
    return ge::GRAPH_SUCCESS;
}

uint64_t MaxPool3DWithArgmaxV2BigKernelRegbaseTiling::GetTilingKey() const
{
    return MAX_POOL_WITH_ARGMAX_V2_TILING_KEY_BIG_KERNEL_REGBASE_NCDHW;
}

void MaxPool3DWithArgmaxV2BigKernelRegbaseTiling::DumpTilingInfo()
{
    std::string str;
    str += " dInDim:" + std::to_string(tilingData_->dInDim);
    str += " hInDim:" + std::to_string(tilingData_->hInDim);
    str += " wInDim:" + std::to_string(tilingData_->wInDim);
    str += " dOutDim:" + std::to_string(tilingData_->dOutDim);
    str += " hOutDim:" + std::to_string(tilingData_->hOutDim);
    str += " wOutDim:" + std::to_string(tilingData_->wOutDim);
    str += " kD:" + std::to_string(tilingData_->kD);
    str += " kH:" + std::to_string(tilingData_->kH);
    str += " kW:" + std::to_string(tilingData_->kW);
    str += " sD:" + std::to_string(tilingData_->sD);
    str += " sH:" + std::to_string(tilingData_->sH);
    str += " sW:" + std::to_string(tilingData_->sW);
    str += " pD:" + std::to_string(tilingData_->pD);
    str += " pH:" + std::to_string(tilingData_->pH);
    str += " pW:" + std::to_string(tilingData_->pW);
    str += " blockFactor:" + std::to_string(tilingData_->blockFactor);
    str += " blockTail:" + std::to_string(tilingData_->blockTail);
    str += " totalIdx:" + std::to_string(tilingData_->totalIdx);
    str += " coreNums:" + std::to_string(tilingData_->coreNums);
    str += " maxCount:" + std::to_string(tilingData_->maxCount);
    str += " isSigOut:" + std::to_string(tilingData_->isSigOut);
    OP_LOGI(context_, "%s", str.c_str());
}

REGISTER_TILING_TEMPLATE("MaxPool3DWithArgmaxV2", MaxPool3DWithArgmaxV2BigKernelRegbaseTiling, 1);

}