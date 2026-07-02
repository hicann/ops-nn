/*
 * Copyright (c) 2026 联通（广东）产业互联网有限公司.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "graph/operator.h"

namespace ops {

namespace {
constexpr size_t REQUIRED_DIM_NUM = 2;
constexpr size_t BIAS_DIM_NUM = 1;
}

ge::graphStatus MatmulAddInferShape(const ge::Operator& op, ge::Operator& out_op)
{
    auto a_desc = op.GetInputDescByName("a");
    auto b_desc = op.GetInputDescByName("b");
    auto bias_desc = op.GetInputDescByName("bias");

    auto a_shape = a_desc.GetShape();
    auto b_shape = b_desc.GetShape();
    auto bias_shape = bias_desc.GetShape();

    if (a_shape.GetDimNum() != REQUIRED_DIM_NUM ||
        b_shape.GetDimNum() != REQUIRED_DIM_NUM) {
        return ge::GRAPH_FAILED;
    }

    int64_t M = a_shape.GetDim(0);
    int64_t K = a_shape.GetDim(1);
    int64_t b_k = b_shape.GetDim(0);
    int64_t N = b_shape.GetDim(1);
    if (K != b_k) {
        return ge::GRAPH_FAILED;
    }

    if (bias_shape.GetDimNum() > 0) {
        if (bias_shape.GetDimNum() != BIAS_DIM_NUM ||
            bias_shape.GetDim(0) != N) {
            return ge::GRAPH_FAILED;
        }
    }

    ge::Shape out_shape({M, N});
    auto y_desc = out_op.GetOutputDescByName("y");
    y_desc.SetShape(out_shape);
    y_desc.SetDataType(a_desc.GetDataType());
    out_op.UpdateOutputDesc("y", y_desc);
    return ge::GRAPH_SUCCESS;
}

} // namespace ops
