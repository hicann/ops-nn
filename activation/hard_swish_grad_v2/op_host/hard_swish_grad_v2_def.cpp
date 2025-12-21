/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file hard_swish_grad_v2_def.cpp
 * \brief
 */

#include "register/op_def_registry.h"

namespace ops {

class HardSwishGradV2 : public OpDef {
public:
  explicit HardSwishGradV2(const char* name) : OpDef(name)
  {
    this->Input("gradOutput")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();

    this->Input("self")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();

    this->Output("out")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();

    this->AICore().AddConfig("ascend910b");
    this->AICore().AddConfig("ascend910_93");

    OpAICoreConfig config910A;
    config910A.Input("gradOutput")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();

    config910A.Input("self")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();

    config910A.Output("out")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();

    config910A.DynamicCompileStaticFlag(true)
        .DynamicRankSupportFlag(true)
        .DynamicShapeSupportFlag(true);

    this->AICore().AddConfig("ascend910", config910A);
  }
};

OP_ADD(HardSwishGradV2);
}  // namespace ops