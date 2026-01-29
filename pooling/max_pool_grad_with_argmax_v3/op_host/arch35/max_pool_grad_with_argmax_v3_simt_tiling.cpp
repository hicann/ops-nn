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
 * \file max_pool_grad_with_argmax_v3_ksize_one_tiling.cpp
 * \brief
 */

#include "max_pool_grad_with_argmax_v3_simt_tiling.h"
#include "platform/platform_info.h"
#include "tiling_base/tiling_templates_registry.h"

namespace optiling
{
static constexpr int64_t FLOAT16_SIZE = 2;
static constexpr int64_t FLOAT32_SIZE = 4;
static constexpr int64_t INT32_SIZE = 4;
static constexpr int64_t INT64_SIZE = 8;
static constexpr int64_t UB_RESVERVED_SIZE = 1024;
static constexpr int64_t EXTRA_BUFFER_SIZE = 256;
static constexpr int64_t SIMT_NCHW_INT32_TILING_KEY = 900;
static constexpr int64_t SIMT_NCHW_INT64_TILING_KEY = 901;
static constexpr int64_t T3_INT64 = 10;
static constexpr int64_t CACHE_LINE_SIZE = 128;
static constexpr int64_t MIN_DATA_SIZE   = 1024;
static constexpr int64_t LOCAL_MEMORY_SIZE = 16384;
static constexpr int64_t SIMT_OUT_THRESHOLD = 256;

bool MaxPoolGradWithArgmaxV3SimtTiling::IsCapable()
{
    if (inputData.inputFormat != ge::Format::FORMAT_NCHW) {
        return false;
    }

    if ((inputData.hKernel * inputData.wKernel < SIMT_OUT_THRESHOLD) &&
        (inputData.hGrad * inputData.wGrad < SIMT_OUT_THRESHOLD)) {
        return true;
    }

    return true;
}

uint64_t MaxPoolGradWithArgmaxV3SimtTiling::GetTilingKey() const
{
    int64_t totalData = inputData.nX * inputData.cX * inputData.hX * inputData.wX;
    uint64_t tilingKey = SIMT_NCHW_INT32_TILING_KEY;
    if (totalData > INT32_MAX) {
        tilingKey = SIMT_NCHW_INT64_TILING_KEY;
    }
    return tilingKey;
}

void MaxPoolGradWithArgmaxV3SimtTiling::PrintTilingData()
{
    OP_LOGD("MaxPoolGradWithArgmaxV3Simt", "[MaxPoolGradWithArgmaxV3SimtTiling] PrintTilingData start running");

    std::ostringstream info;
    info << "nDim: " << tilingData_.get_nDim() << std::endl;
    info << "cDim: " << tilingData_.get_cDim() << std::endl;
    info << "hInDim: " << tilingData_.get_hInDim() << std::endl;
    info << "wInDim: " << tilingData_.get_wInDim() << std::endl;
    info << "hOutDim: " << tilingData_.get_hOutDim() << std::endl;
    info << "wOutDim: " << tilingData_.get_wOutDim() << std::endl;
    info << "kSizeH: " << tilingData_.get_kSizeH() << std::endl;
    info << "kSizeW: " << tilingData_.get_kSizeW() << std::endl;
    info << "stridesH: " << tilingData_.get_stridesH() << std::endl;
    info << "stridesW: " << tilingData_.get_stridesW() << std::endl;
    info << "padH: " << tilingData_.get_padH() << std::endl;
    info << "padW: " << tilingData_.get_padW() << std::endl;
    info << "dilationH: " << tilingData_.get_dilationH() << std::endl;
    info << "dilationW: " << tilingData_.get_dilationW() << std::endl;
    info << "ceilMode: " << tilingData_.get_ceilMode() << std::endl;
    info << "coreNum: " << hardwareData.coreNum << std::endl;

    OP_LOGI("MaxPoolGradWithArgmaxV3Simt", "%s", info.str().c_str());
}

void MaxPoolGradWithArgmaxV3SimtTiling::SetTilingData()
{
    tilingData_.set_nDim(inputData.nX);
    tilingData_.set_cDim(inputData.cX);
    tilingData_.set_hInDim(inputData.hX);
    tilingData_.set_wInDim(inputData.wX);
    tilingData_.set_hOutDim(inputData.hGrad);
    tilingData_.set_wOutDim(inputData.wGrad);
    tilingData_.set_kSizeH(inputData.hKernel);
    tilingData_.set_kSizeW(inputData.wKernel);
    tilingData_.set_stridesH(inputData.hStride);
    tilingData_.set_stridesW(inputData.wStride);
    tilingData_.set_padH(inputData.hPad);
    tilingData_.set_padW(inputData.wPad);
    tilingData_.set_dilationH(inputData.hDilation);
    tilingData_.set_dilationW(inputData.wDilation);
    tilingData_.set_ceilMode(inputData.ceilMode);
    context_->SetBlockDim(hardwareData.coreNum);
}

ge::graphStatus MaxPoolGradWithArgmaxV3SimtTiling::DoOpTiling()
{
    SetTilingData();
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPoolGradWithArgmaxV3SimtTiling::PostTiling()
{
    context_->SetBlockDim(hardwareData.coreNum);
    if (tilingData_.GetDataSize() > context_->GetRawTilingData()->GetCapacity()) {
        return ge::GRAPH_FAILED;
    }

    context_->SetLocalMemorySize(static_cast<uint32_t>(LOCAL_MEMORY_SIZE));
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(MaxPoolGradWithArgmaxV3, MaxPoolGradWithArgmaxV3SimtTiling, 1);
}  // namespace optiling
