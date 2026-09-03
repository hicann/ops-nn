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
 * \file matmul_tf_plugin.cpp
 * \brief
 */
#include "register/register.h"
#include "framework/plugin_util.h"
#include "error_util.h"
#include "log/log.h"

namespace domi {
static Status AutoMappingFnMatMulV2(const ge::Operator& op_src, ge::Operator& op)
{
    Status ret = AutoMappingByOpFn(op_src, op);
    if (ret != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "Tensorflow plugin parser failed. auto mapping failed.");
        return FAILED;
    }
    bool transpose_a = false;
    if (op.GetAttr("transpose_a", transpose_a) != ge::GRAPH_SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "GetAttr transpose_a failed");
        return FAILED;
    }
    bool transpose_b = false;
    if (op.GetAttr("transpose_b", transpose_b) != ge::GRAPH_SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "GetAttr transpose_b failed");
        return FAILED;
    }

    op.SetAttr("transpose_x1", transpose_a);
    op.SetAttr("transpose_x2", transpose_b);
    OP_LOGD(GetOpName(op).c_str(), "Tensorflow plugin parser[AutoMapping] success.");
    return SUCCESS;
}

REGISTER_CUSTOM_OP("MatMulV2")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("MatMul")
    .ParseParamsByOperatorFn(AutoMappingFnMatMulV2)
    .ImplyType(ImplyType::TVM);
} // namespace domi
