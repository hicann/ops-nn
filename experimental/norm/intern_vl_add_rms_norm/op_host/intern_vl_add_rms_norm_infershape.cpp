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

ge::graphStatus InternVlAddRmsNormInferShape(const ge::Operator& op, ge::Operator& out_op) {
    auto x_shape = op.GetInputDescByName("x").GetShape();
    out_op.GetOutputDescByName("y").SetShape(x_shape);
    out_op.GetOutputDescByName("residual_out").SetShape(x_shape);
    return ge::GRAPH_SUCCESS;
}

} // namespace ops
