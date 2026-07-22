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
 * \file ascend_quant_tf_plugin.cpp
 * \brief
 */
#include "register/register.h"
#include "graph/types.h"
#include "framework/plugin_util.h"

namespace domi {
// TensorFlow AscendQuant uses string attribute `dst_type` (e.g. "INT4"/"INT16"),
// while the GE op expects int dtype. Also tags the op with `tf_tag="tf"`.
static Status ParseParamsAscendQuant(const google::protobuf::Message* op_src, ge::Operator& op)
{
    if (AutoMappingFn(op_src, op) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "auto mapping failed.");
        return FAILED;
    }
    std::string dst_type_str;
    int dst_type = ge::DT_INT8;
    if (op.GetAttr("dst_type", dst_type_str) == ge::GRAPH_SUCCESS) {
        if (dst_type_str == "INT4") {
            dst_type = ge::DT_INT4;
        } else if (dst_type_str == "INT16") {
            dst_type = ge::DT_INT16;
        }
    }
    (void)op.SetAttr("dst_type", dst_type);
    (void)op.SetAttr("tf_tag", "tf");
    return SUCCESS;
}

REGISTER_CUSTOM_OP("AscendQuant")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("AscendQuant")
    .ParseParamsFn(ParseParamsAscendQuant)
    .ImplyType(ImplyType::TVM);
} // namespace domi
