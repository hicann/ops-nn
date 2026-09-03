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
 * \file avg_pool3d_tf_plugin.cpp
 * \brief
 */
#include "register/register.h"
#include "framework/plugin_util.h"
#include "error_util.h"
#include "log/log.h"

namespace domi {
// Replace ge ParseParams fuction to process graph conv2d node attrs
static Status ParseParamsAvgPool3D(const ge::Operator& op_src, ge::Operator& op)
{
    ge::AscendString op_name;
    OP_LOGE_IF(op_src.GetName(op_name) != ge::GRAPH_SUCCESS, FAILED, "", "failed to get op_name");

    // Convert original tf graph avg_pool3d attrs to GE graph attrs
    if (AutoMappingByOpFn(op_src, op) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "AutoMappingFn failed.");
        return FAILED;
    }

    // Escape GE require attr [pads] check here
    std::vector<int32_t> padList = {0, 0, 0, 0, 0, 0};
    (void)op.SetAttr("pads", padList);
    (void)op.SetAttr("count_include_pad", false);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("AvgPool3D")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("AvgPool3D")
    .ParseParamsByOperatorFn(ParseParamsAvgPool3D)
    .ImplyType(ImplyType::TVM);
} // namespace domi
