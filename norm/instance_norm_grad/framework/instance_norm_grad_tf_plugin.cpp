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
 * \file instance_norm_grad_tf_plugin.cpp
 * \brief
 */
#include "register/register.h"
#include "framework/plugin_util.h"
#include "log/log.h"

namespace domi {
static Status InstanceNormGradParserParams(const std::vector<const google::protobuf::Message*> inside_nodes,
                                           ge::Operator& op)
{
    OP_LOGI(GetOpName(op).c_str(), "Enter instance norm grad fusion parser.");

    return SUCCESS;
}

REGISTER_CUSTOM_OP("InstanceNormGrad")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("InstanceNormGrad")
    .FusionParseParamsFn(InstanceNormGradParserParams)
    .ImplyType(ImplyType::TVM);
} // namespace domi
