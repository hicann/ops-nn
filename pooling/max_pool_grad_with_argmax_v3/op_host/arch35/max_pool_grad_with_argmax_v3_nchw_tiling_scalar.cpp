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
 * \file max_pool_grad_with_argmax_v3_nchw_tiling_scalar.cpp
 * \brief
 */
#include "max_pool_grad_with_argmax_v3_nchw_tiling_scalar.h"
#include "platform/platform_info.h"
#include "tiling_base/tiling_templates_registry.h"
namespace optiling {
static constexpr int64_t HALF = 2;
static constexpr int64_t DOUBLE_BUFFER = 2;
static constexpr int64_t CHECK_RANGE_TILING_KEY_NCHW_SCALAR = 301;
static constexpr int64_t FLOAT32_SIZE = 4;
bool MaxPoolGradWithArgmaxV3NCHWScalarTiling::IsCapable()
{
    if (inputData.inputFormat != ge::Format::FORMAT_NCHW) {
        return false;
    }
    return true;
}
ge::graphStatus MaxPoolGradWithArgmaxV3NCHWScalarTiling::DoOpTiling()
{
    CalcBase();
    CalcParamsEachCore();
    ge::graphStatus result = CalcGradArgmax();
    if (result != ge::GRAPH_SUCCESS) {
        return result;
    }
    SetTilingData();
    PrintData();
    return ge::GRAPH_SUCCESS;
}
void MaxPoolGradWithArgmaxV3NCHWScalarTiling::CalcBase()
{
    int64_t ncTotal = inputData.nX * inputData.cX;
    int64_t ncSizePerCore = Ops::Base::CeilDiv(ncTotal, hardwareData.coreNum);
    int64_t hwSize = inputData.hX * inputData.wX * FLOAT32_SIZE;
    int64_t inputUbSize = hardwareData.ubSize / DOUBLE_BUFFER / HALF;
    scalarTilingData_.outputBufferSize = inputUbSize;
    if (ncSizePerCore * hwSize <= inputUbSize) {
        scalarTilingData_.highAxisInner = ncSizePerCore;
        scalarTilingData_.hOutputInner = inputData.hX;
        scalarTilingData_.wOutputInner = inputData.wX;
        return;
    }

    if (hwSize <= inputUbSize) {
        scalarTilingData_.highAxisInner = inputUbSize / hwSize;
        scalarTilingData_.hOutputInner = inputData.hX;
        scalarTilingData_.wOutputInner = inputData.wX;
        return;
    }

    int64_t perHSize = 1 * inputData.wX * FLOAT32_SIZE;
    if (perHSize <= inputUbSize) {
        scalarTilingData_.highAxisInner = 1;
        scalarTilingData_.hOutputInner = inputUbSize / perHSize;
        scalarTilingData_.wOutputInner = inputData.wX;
        return;
    }

    int64_t perWSize = 1 * FLOAT32_SIZE;
    if (perWSize <= inputUbSize) {
        scalarTilingData_.highAxisInner = 1;
        scalarTilingData_.hOutputInner = 1;
        scalarTilingData_.wOutputInner = inputUbSize / perWSize;
        return;
    }
}
void MaxPoolGradWithArgmaxV3NCHWScalarTiling::CalcParamsEachCore()
{
    int64_t ncTotal = inputData.nX * inputData.cX;
    scalarTilingData_.highAxisOuter = Ops::Base::CeilDiv(ncTotal, scalarTilingData_.highAxisInner);
    scalarTilingData_.hOutputOuter = Ops::Base::CeilDiv(inputData.hX, scalarTilingData_.hOutputInner);
    scalarTilingData_.wOutputOuter = Ops::Base::CeilDiv(inputData.wX, scalarTilingData_.wOutputInner);
    scalarTilingData_.highAxisTail = ncTotal - (scalarTilingData_.highAxisOuter - 1) * scalarTilingData_.highAxisInner;
    scalarTilingData_.hOutputTail =
        inputData.hX - (scalarTilingData_.hOutputOuter - 1) * scalarTilingData_.hOutputInner;
    scalarTilingData_.wOutputTail =
        inputData.wX - (scalarTilingData_.wOutputOuter - 1) * scalarTilingData_.wOutputInner;
    int64_t totalCount =
        scalarTilingData_.highAxisOuter * scalarTilingData_.hOutputOuter * scalarTilingData_.wOutputOuter;
    scalarTilingData_.normalCoreProcessNum = Ops::Base::CeilDiv(totalCount, hardwareData.coreNum);
    scalarTilingData_.usedCoreNum = Ops::Base::CeilDiv(totalCount, scalarTilingData_.normalCoreProcessNum);
    scalarTilingData_.tailCoreProcessNum =
        totalCount - (scalarTilingData_.usedCoreNum - 1) * scalarTilingData_.normalCoreProcessNum;
    return;
}
ge::graphStatus MaxPoolGradWithArgmaxV3NCHWScalarTiling::CalcGradArgmaxInner(int64_t argmaxCountInUB)
{
    int64_t hInputInner = Ops::Base::CeilDiv(scalarTilingData_.hOutputInner + inputData.hKernel - 1, inputData.hStride);
    int64_t wInputInner = Ops::Base::CeilDiv(scalarTilingData_.wOutputInner + inputData.wKernel - 1, inputData.wStride);
    hInputInner = std::min(hInputInner, inputData.hGrad);
    wInputInner = std::min(wInputInner, inputData.wGrad);
    if (hInputInner == 0 || wInputInner == 0) {
        return ge::GRAPH_FAILED;
    }
    int64_t inputPlaneSize = hInputInner * wInputInner;
    if (scalarTilingData_.highAxisInner * hInputInner * wInputInner <= argmaxCountInUB) {
        scalarTilingData_.argmaxNcInner = scalarTilingData_.highAxisInner;
        scalarTilingData_.argmaxHInner = hInputInner;
        scalarTilingData_.argmaxWInner = wInputInner;
    } else if (inputPlaneSize <= argmaxCountInUB) {
        scalarTilingData_.argmaxNcInner = argmaxCountInUB / inputPlaneSize;
        scalarTilingData_.argmaxHInner = hInputInner;
        scalarTilingData_.argmaxWInner = wInputInner;
    } else if (wInputInner <= argmaxCountInUB) {
        scalarTilingData_.argmaxNcInner = 1;
        scalarTilingData_.argmaxHInner = argmaxCountInUB / wInputInner;
        scalarTilingData_.argmaxWInner = wInputInner;
    } else {
        scalarTilingData_.argmaxNcInner = 1;
        scalarTilingData_.argmaxHInner = 1;
        scalarTilingData_.argmaxWInner = argmaxCountInUB;
    }

    scalarTilingData_.argmaxNcOuter =
        Ops::Base::CeilDiv(scalarTilingData_.highAxisInner, scalarTilingData_.argmaxNcInner);
    scalarTilingData_.argmaxHOuter = Ops::Base::CeilDiv(hInputInner, scalarTilingData_.argmaxHInner);
    scalarTilingData_.argmaxWOuter = Ops::Base::CeilDiv(wInputInner, scalarTilingData_.argmaxWInner);
    scalarTilingData_.argmaxNcTail =
        scalarTilingData_.highAxisInner - (scalarTilingData_.argmaxNcOuter - 1) * scalarTilingData_.argmaxNcInner;
    scalarTilingData_.argmaxHTail = hInputInner - (scalarTilingData_.argmaxHOuter - 1) * scalarTilingData_.argmaxHInner;
    scalarTilingData_.argmaxWTail = wInputInner - (scalarTilingData_.argmaxWOuter - 1) * scalarTilingData_.argmaxWInner;
    scalarTilingData_.argmaxInnerLoop =
        scalarTilingData_.argmaxNcOuter * scalarTilingData_.argmaxHOuter * scalarTilingData_.argmaxWOuter;
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus MaxPoolGradWithArgmaxV3NCHWScalarTiling::CalcGradArgmax()
{
    int64_t totalGradArgmaxUBSize = hardwareData.ubSize / DOUBLE_BUFFER / HALF;
    int64_t inputDtypeSize = ge::GetSizeByDataType(inputData.inputDtype);
    int64_t indexDtypeSize = ge::GetSizeByDataType(inputData.indexDtype);
    int64_t alignTypeSize = (inputDtypeSize < indexDtypeSize ? inputDtypeSize : indexDtypeSize);
    int64_t ubBlockSize = Ops::Base::GetUbBlockSize(context_);
    int64_t baseAlignedCount = ubBlockSize / alignTypeSize;
    int64_t argmaxCountInUB = totalGradArgmaxUBSize / (ge::GetSizeByDataType(inputData.inputDtype) +
                                                       ge::GetSizeByDataType(inputData.indexDtype));
    argmaxCountInUB = argmaxCountInUB / baseAlignedCount * baseAlignedCount;
    scalarTilingData_.gradBufferSize = argmaxCountInUB * ge::GetSizeByDataType(inputData.inputDtype);
    scalarTilingData_.argmaxBufferSize = argmaxCountInUB * ge::GetSizeByDataType(inputData.indexDtype);
    return CalcGradArgmaxInner(argmaxCountInUB);
}
uint64_t MaxPoolGradWithArgmaxV3NCHWScalarTiling::GetTilingKey() const
{
    return CHECK_RANGE_TILING_KEY_NCHW_SCALAR;
}

void MaxPoolGradWithArgmaxV3NCHWScalarTiling::SetTilingData()
{
    tilingData_.set_hArgmax(inputData.hGrad);
    tilingData_.set_wArgmax(inputData.wGrad);
    tilingData_.set_hOutput(inputData.hX);
    tilingData_.set_wOutput(inputData.wX);
    tilingData_.set_hKernel(inputData.hKernel);
    tilingData_.set_wKernel(inputData.wKernel);
    tilingData_.set_hStride(inputData.hStride);
    tilingData_.set_wStride(inputData.wStride);
    tilingData_.set_padH(inputData.hPad);
    tilingData_.set_padW(inputData.wPad);
    tilingData_.set_dilationH(inputData.hDilation);
    tilingData_.set_dilationW(inputData.wDilation);
    tilingData_.set_highAxisInner(scalarTilingData_.highAxisInner);
    tilingData_.set_highAxisTail(scalarTilingData_.highAxisTail);
    tilingData_.set_highAxisOuter(scalarTilingData_.highAxisOuter);
    tilingData_.set_hOutputInner(scalarTilingData_.hOutputInner);
    tilingData_.set_hOutputTail(scalarTilingData_.hOutputTail);
    tilingData_.set_hOutputOuter(scalarTilingData_.hOutputOuter);
    tilingData_.set_wOutputInner(scalarTilingData_.wOutputInner);
    tilingData_.set_wOutputTail(scalarTilingData_.wOutputTail);
    tilingData_.set_wOutputOuter(scalarTilingData_.wOutputOuter);
    tilingData_.set_normalCoreProcessNum(scalarTilingData_.normalCoreProcessNum);
    tilingData_.set_tailCoreProcessNum(scalarTilingData_.tailCoreProcessNum);
    tilingData_.set_usedCoreNum(scalarTilingData_.usedCoreNum);
    tilingData_.set_outputBufferSize(scalarTilingData_.outputBufferSize);
    tilingData_.set_gradBufferSize(scalarTilingData_.gradBufferSize);
    tilingData_.set_argmaxBufferSize(scalarTilingData_.argmaxBufferSize);
    tilingData_.set_argmaxNcInner(scalarTilingData_.argmaxNcInner);
    tilingData_.set_argmaxNcOuter(scalarTilingData_.argmaxNcOuter);
    tilingData_.set_argmaxNcTail(scalarTilingData_.argmaxNcTail);
    tilingData_.set_argmaxHInner(scalarTilingData_.argmaxHInner);
    tilingData_.set_argmaxHOuter(scalarTilingData_.argmaxHOuter);
    tilingData_.set_argmaxHTail(scalarTilingData_.argmaxHTail);
    tilingData_.set_argmaxWInner(scalarTilingData_.argmaxWInner);
    tilingData_.set_argmaxWOuter(scalarTilingData_.argmaxWOuter);
    tilingData_.set_argmaxWTail(scalarTilingData_.argmaxWTail);
    tilingData_.set_argmaxInnerLoop(scalarTilingData_.argmaxInnerLoop);
    return;
}
void MaxPoolGradWithArgmaxV3NCHWScalarTiling::PrintData() const
{
    OP_LOGI("PrintData", "%s", scalarTilingData_.ToString().c_str());
    return;
}
ge::graphStatus MaxPoolGradWithArgmaxV3NCHWScalarTiling::PostTiling()
{
    context_->SetBlockDim(tilingData_.get_usedCoreNum());
    if (tilingData_.GetDataSize() > context_->GetRawTilingData()->GetCapacity()) {
        return ge::GRAPH_FAILED;
    }

    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
REGISTER_OPS_TILING_TEMPLATE(MaxPoolGradWithArgmaxV3, MaxPoolGradWithArgmaxV3NCHWScalarTiling, 10);
} // namespace optiling