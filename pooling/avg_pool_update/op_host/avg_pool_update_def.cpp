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
 * \file avg_pool_update_def.cpp
 * \brief Operator definition for avg_pool_update operator
 */

#include "register/op_def_registry.h"

namespace ops {
class AvgPoolUpdate : public OpDef {
public:
    explicit AvgPoolUpdate(const char* name) : OpDef(name)
    {
        // 全笛卡尔积：x1∈{FP16,FP32} × x2∈{INT4,INT8,FP16,FP32} = 8 组
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        // x2 dtype 独立于 x1（x2 仅用于获取输入空间尺寸）
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT4, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT4, ge::DT_INT8, ge::DT_FLOAT16,
                       ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // 属性注册（顺序与 REG_OP 一致，Tiling 侧按索引 0-6 获取）
        this->Attr("ksize").AttrType(REQUIRED).ListInt();
        this->Attr("strides").AttrType(REQUIRED).ListInt();
        this->Attr("padding_mode").AttrType(OPTIONAL).String("CALCULATED");
        this->Attr("pads").AttrType(OPTIONAL).ListInt({0, 0, 0, 0});
        this->Attr("data_format").AttrType(OPTIONAL).String("NHWC");
        this->Attr("ceil_mode").AttrType(OPTIONAL).Bool(false);
        this->Attr("exclusive").AttrType(OPTIONAL).Bool(true);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "avg_pool_update");
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};
OP_ADD(AvgPoolUpdate);
} // namespace ops
