/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "onnx_common.h"
#include "op_nn_proto_extend.h"

namespace domi {
static Status ParseParamsSparseToDense(const Message* op_src, ge::Operator& op_dest)
{
    const ge::onnx::NodeProto* node = reinterpret_cast<const ge::onnx::NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }

    int validate_indices_int = 0;
    bool validate_indices = false;
    for (const auto& attr : node->attribute()) {
        if (attr.name() == "validate_indices") {
            validate_indices_int = attr.i();
            validate_indices = validate_indices_int != 0;
        }
    }

    op_dest.SetAttr("validate_indices", validate_indices);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("SparseToDense")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::8::SparseToDense", "ai.onnx::9::SparseToDense", "ai.onnx::10::SparseToDense",
                   "ai.onnx::11::SparseToDense", "ai.onnx::12::SparseToDense", "ai.onnx::13::SparseToDense",
                   "ai.onnx::14::SparseToDense", "ai.onnx::15::SparseToDense", "ai.onnx::16::SparseToDense",
                   "ai.onnx::17::SparseToDense", "ai.onnx::18::SparseToDense"})
    .ParseParamsFn(ParseParamsSparseToDense)
    .ImplyType(ImplyType::TVM);
} // namespace domi
