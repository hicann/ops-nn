/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_def_registry.h"

namespace ops {

static const std::vector<ge::DataType> kDtypes = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};
// BF16 Matmul uses an FP32 bias table on Ascend 910B.
static const std::vector<ge::DataType> kBiasDtypes = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT};
static const std::vector<ge::Format> kFormats = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

class FusedPatchMlp : public OpDef {
public:
    explicit FusedPatchMlp(const char* name) : OpDef(name)
    {
        // The generated Ascend C kernel receives this input dtype through the compile-time DTYPE_X macro.
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(kDtypes)
            .Format(kFormats)
            .UnknownShapeFormat(kFormats)
            .AutoContiguous();
        this->Input("weights")
            .ParamType(REQUIRED)
            .DataType(kDtypes)
            .Format(kFormats)
            .UnknownShapeFormat(kFormats)
            .AutoContiguous();
        this->Input("biases")
            .ParamType(REQUIRED)
            .DataType(kBiasDtypes)
            .Format(kFormats)
            .UnknownShapeFormat(kFormats)
            .AutoContiguous();
        this->Output("y").ParamType(REQUIRED).DataType(kDtypes).Format(kFormats).UnknownShapeFormat(kFormats);
        this->Attr("num_layers").AttrType(REQUIRED).Int();
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(FusedPatchMlp);

} // namespace ops
