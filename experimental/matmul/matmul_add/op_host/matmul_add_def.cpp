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
#include "register/op_def_registry.h"

namespace ops {
namespace {
constexpr const char* ASCEND910B_CONFIG = "ascend910b";
constexpr const char* ASCEND910_93_CONFIG = "ascend910_93";

enum class TensorPresence {
    Required,
    Optional
};

template <typename TensorBuilder>
void ConfigurePlainNdTensor(TensorBuilder&& tensor, TensorPresence presence)
{
    tensor.ParamType(presence == TensorPresence::Optional ? OPTIONAL : REQUIRED);
    tensor.DataType({ge::DT_FLOAT16, ge::DT_BF16});
    tensor.Format({ge::FORMAT_ND, ge::FORMAT_ND});
    tensor.UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    tensor.AutoContiguous();
}
} // namespace

class MatmulAdd : public OpDef {
public:
    explicit MatmulAdd(const char* name) : OpDef(name)
    {
        ConfigurePlainNdTensor(this->Input("a"), TensorPresence::Required);
        ConfigurePlainNdTensor(this->Input("b"), TensorPresence::Required);
        ConfigurePlainNdTensor(this->Input("bias"), TensorPresence::Optional);
        ConfigurePlainNdTensor(this->Output("y"), TensorPresence::Required);

        this->AICore().AddConfig(ASCEND910B_CONFIG);
        this->AICore().AddConfig(ASCEND910_93_CONFIG);
    }
};

OP_ADD(MatmulAdd);
} // namespace ops
