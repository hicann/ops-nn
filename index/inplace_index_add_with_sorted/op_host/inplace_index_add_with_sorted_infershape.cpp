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
 * \file inplace_index_add_with_sorted_infershape.cpp
 * \brief InplaceIndexAddWithSorted 形状推导：单输出 var
 *
 * Output shape identical to input var (in-place update).
 * InferDataType 移至 op_graph/inplace_index_add_with_sorted_graph_infer.cpp。
 */

#include <string>
#include <vector>
#include "base/err_msg.h"
#include "register/op_impl_registry.h"
#include "op_common/log/log.h"

using namespace ge;

namespace ops {

static constexpr size_t VAR_INDEX = 0;

static ge::graphStatus InferShape4InplaceIndexAddWithSorted(gert::InferShapeContext* context)
{
    const gert::Shape* varShape = context->GetInputShape(VAR_INDEX);
    OP_CHECK_IF(varShape == nullptr, OP_LOGE_WITH_INVALID_INPUT("InplaceIndexAddWithSorted", "var"),
                return ge::GRAPH_FAILED);

    gert::Shape* varOutShape = context->GetOutputShape(VAR_INDEX);
    if (varOutShape == nullptr) {
        const std::string reason = std::string(__func__) + ": " +
                                   ("framework-provided varOutShape is nullptr; check context construction");
        const std::vector<const char*> keys = {"opName", "reason"};
        const std::vector<const char*> values = {"InplaceIndexAddWithSorted", reason.c_str()};
        OP_LOGE_WITHOUT_REPORT("InplaceIndexAddWithSorted", "%s", reason.c_str());
        REPORT_PREDEFINED_ERR_MSG("EZ1014", keys, values);
        return ge::GRAPH_FAILED;
    }
    *varOutShape = *varShape;

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(InplaceIndexAddWithSorted).InferShape(InferShape4InplaceIndexAddWithSorted);

} // namespace ops
