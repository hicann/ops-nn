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
 * \file max_pool3d_grad_grad_tf_plugin.cpp
 * \brief
 */
#include "register/register.h"
#include "graph/operator.h"

#include "log/log.h"

namespace domi {
// The target op expresses padding as a pads list: all-zero means VALID, any non-zero entry
// means SAME (the tiling recomputes the real SAME pads from ksize/strides, the values are
// not used). TF carries a padding string instead, so it must be mapped here; unconditionally
// writing all-zero pads would silently execute TF padding=SAME nodes as VALID.
static Status ParseParamsMaxPool3DGradGrad(const ge::Operator& op_src, ge::Operator& op)
{
    // Convert original tf graph maxpool3dgradgrad attrs to GE graph attrs
    if (AutoMappingByOpFn(op_src, op) != SUCCESS) {
        return FAILED;
    }
    std::string padding = "VALID";
    if (op_src.GetAttr("padding", padding) != ge::GRAPH_SUCCESS) {
        padding = "VALID";
    }
    std::vector<int32_t> padList = {0, 0, 0, 0, 0, 0};
    if (padding == "SAME") {
        padList = {1, 1, 1, 1, 1, 1};
    } else if (padding != "VALID") {
        OP_LOGE("MaxPool3DGradGrad", "unsupported TF padding [%s], only SAME and VALID are defined.", padding.c_str());
        return FAILED;
    }
    (void)op.SetAttr("pads", padList);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("MaxPool3DGradGrad")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("MaxPool3DGradGrad")
    .ParseParamsByOperatorFn(ParseParamsMaxPool3DGradGrad)
    .ImplyType(ImplyType::TVM);
} // namespace domi
