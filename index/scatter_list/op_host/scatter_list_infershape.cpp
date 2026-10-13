/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file scatter_list_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {
static graphStatus InferDataType4ScatterList(gert::InferDataTypeContext* context)
{
    auto var_dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, var_dtype);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4ScatterList(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do Infershape of ScatterList.");
    auto output_num = context->GetComputeNodeOutputNum();
    for (size_t i = 0U; i < output_num; ++i) {
        auto x_shape = context->GetInputShape(i);
        auto y_shape = context->GetOutputShape(i);
        if ((x_shape == nullptr) || (y_shape == nullptr)) {
            return ge::GRAPH_FAILED;
        }
        *y_shape = *x_shape;
    }
    OP_LOGD(context->GetNodeName(), "End to do Infershape of ScatterList.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ScatterList).InferShape(InferShape4ScatterList).InferDataType(InferDataType4ScatterList);
} // namespace ops
