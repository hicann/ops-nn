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
 * \file max_pool_grad_with_argmax_simt_tiling_common.cpp
 * \brief
 */
#include "max_pool_grad_with_argmax_simt_tiling_common.h"
#include "op_common/op_host/util/platform_util.h"
#include <iostream>

namespace optiling
{

void MaxPoolGradWithArgmaxSIMTTilingCommon::SetTilingData(gert::TilingContext* context)
{
    MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxSimtTilingCommonData* tilingData =
        context->GetTilingData<MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxSimtTilingCommonData>();
    tilingData->nDim = inputData->nX;
    tilingData->cDim = inputData->cX;
    tilingData->hInDim = inputData->hX;
    tilingData->wInDim = inputData->wX;
    tilingData->hOutDim = inputData->hGrad;
    tilingData->wOutDim = inputData->wGrad;
    tilingData->kSizeH = inputData->hKernel;
    tilingData->kSizeW = inputData->wKernel;
    tilingData->stridesH = inputData->hStride;
    tilingData->stridesW = inputData->wStride;
    tilingData->padH = inputData->hPad;
    tilingData->padW = inputData->wPad;
    tilingData->dilationH = inputData->hDilation;
    tilingData->dilationW = inputData->wDilation;
    tilingData->ceilMode = inputData->ceilMode;
    OP_LOGD("MaxPoolGradWithArgmax", "MaxPoolGradWithArgmaxSIMTTilingCommon::SetTilingData");
}

ge::graphStatus MaxPoolGradWithArgmaxSIMTTilingCommon::PostTiling(gert::TilingContext* context_, MaxPoolGradWithArgmaxHardwareInfo hwinfo)
{
    context_->SetBlockDim(hwinfo.coreNum);
    OP_LOGD("MaxPoolGradWithArgmax", "MaxPoolGradWithArgmaxSIMTTilingCommon::PostTiling");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPoolGradWithArgmaxSIMTTilingCommon::DoOpTiling(gert::TilingContext* context_){
    SetTilingData(context_);
    return ge::GRAPH_SUCCESS;
}
}