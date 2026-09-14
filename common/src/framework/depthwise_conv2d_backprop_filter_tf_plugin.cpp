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
 * \file depthwise_conv2d_backprop_filter_tf_plugin.cpp
 * \brief
 */
#include <vector>
#include "register/register.h"
#include "framework/plugin_util.h"
#include "graph/operator.h"
#include "error_util.h"

#include "log/log.h"

namespace domi {
namespace {
const int32_t kOutputFilterGrad = 0;
}

/*!
 * @brief Replace GE ParseParams fuction to process graph depthwise_conv2d_backprop_filter node attrs
 * @param opSrc the source op info from tf.
 * @param op the dest GE op.
 * @return status whether this operation success.
 */
static Status ParseParamsDepthwiseConv2DBackpropFilter(const ge::Operator& opSrc, ge::Operator& op)
{
    if (AutoMappingByOpFn(opSrc, op) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "auto mapping failed.");
        return FAILED;
    }

    ge::TensorDesc orgTensorW = op.GetOutputDesc(kOutputFilterGrad);
    orgTensorW.SetOriginFormat(ge::FORMAT_HWCN);
    orgTensorW.SetFormat(ge::FORMAT_HWCN);
    auto ret = op.UpdateOutputDesc(kOutputFilterGrad, orgTensorW);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "update output format failed.");
        return FAILED;
    }
    OP_LOGD(GetOpName(op).c_str(), "update output format success.");

    std::vector<int32_t> padList = {0, 0, 0, 0};
    op.SetAttr("pads", padList);
    OP_LOGD(GetOpName(op).c_str(), "update pads success.");

    return SUCCESS;
}

REGISTER_CUSTOM_OP("DepthwiseConv2DBackpropFilter")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("DepthwiseConv2dNativeBackpropFilter")
    .ParseParamsByOperatorFn(ParseParamsDepthwiseConv2DBackpropFilter)
    .ImplyType(ImplyType::TVM);
} // namespace domi
