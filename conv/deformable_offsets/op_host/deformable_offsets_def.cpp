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
 * \file deformable_offsets_def.cpp
 * \brief deformable_offsets_def op_host
 */

#include <cstdint>
#include "register/op_def_registry.h"

namespace ops {
class DeformableOffsets : public OpDef {
public:
    explicit DeformableOffsets(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC})
            .UnknownShapeFormat({ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC});
        this->Input("offsets")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC})
            .UnknownShapeFormat({ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC})
            .UnknownShapeFormat({ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC});
        this->Attr("strides").AttrType(REQUIRED).ListInt();
        this->Attr("pads").AttrType(REQUIRED).ListInt();
        this->Attr("ksize").AttrType(REQUIRED).ListInt();
        this->Attr("dilations").AttrType(OPTIONAL).ListInt({1, 1, 1, 1});
        this->Attr("data_format").AttrType(OPTIONAL).String("NHWC");
        this->Attr("deformable_groups").AttrType(OPTIONAL).Int(1);
        this->Attr("modulated").AttrType(OPTIONAL).Bool(true);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .ExtendCfgInfo("opFile.value", "deformable_offsets_apt");
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};
OP_ADD(DeformableOffsets);
} // namespace ops