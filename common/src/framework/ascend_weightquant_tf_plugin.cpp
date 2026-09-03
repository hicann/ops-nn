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
 * \file ascend_weightquant_tf_plugin.cpp
 * \brief
 */
#include <string>

#include "graph/types.h"
#include "log/log.h"
#include "register/register.h"

namespace domi {
static Status ParseParamsAscendWeightQuant(const ge::Operator& op_src, ge::Operator& op)
{
    AutoMappingByOpFn(op_src, op);

    std::string dst_type_str;
    if (op.GetAttr("dst_type", dst_type_str) == ge::GRAPH_SUCCESS) {
        int dst_type = ge::DT_INT8;
        if (dst_type_str == "INT4") {
            dst_type = ge::DT_INT4;
        }
        op.SetAttr("dst_type", dst_type);
    }

    OP_LOGI("AscendWeightQuant", "op[AscendWeightQuant] tensowflow plugin parser [AutoMapping] success.");
    return SUCCESS;
}

REGISTER_CUSTOM_OP("AscendWeightQuant")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("AscendWeightQuant")
    .ParseParamsByOperatorFn(ParseParamsAscendWeightQuant)
    .ImplyType(ImplyType::TVM);
} // namespace domi
