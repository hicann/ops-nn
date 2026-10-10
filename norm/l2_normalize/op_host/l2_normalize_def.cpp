/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// =============================================================================
// norm/l2_normalize/op_host/l2_normalize_def.cpp
// =============================================================================
//
// ROLE: Operator definition (OpDef) for L2Normalize (Ascend 950 / arch35 only).
//   公开原型：
//   - Input  x : REQUIRED, {DT_FLOAT16, DT_FLOAT}, FORMAT_ND
//   - Output y : REQUIRED, {DT_FLOAT16, DT_FLOAT}, FORMAT_ND
//   - Attr   axis : ListInt, 默认 {}
//   - Attr   eps  : Float,   默认 1e-4f
//   - AICore config: 仅 ascend950，PrecisionReduceFlag(false)（fp16 内部 fp32 提升）
// =============================================================================

#include "register/op_def_registry.h"

namespace ops {

class L2Normalize : public OpDef {
public:
    explicit L2Normalize(const char* name) : OpDef(name)
    {
        // AutoContiguous：非连续 view 输入/输出由框架自动物化为连续布局（x 拷入连续
        // 缓冲、y 经连续临时缓冲回写 view），kernel 侧恒按连续 ND 布局消费/产出
        // 非连续 view 属支持范围；不设置时 executor 无法保证输入物化与输出 view 回写。
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        this->Attr("axis").AttrType(OPTIONAL).ListInt({});
        this->Attr("eps").AttrType(OPTIONAL).Float(1e-4f);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(false)
            .ExtendCfgInfo("opFile.value", "l2_normalize");
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(L2Normalize);
} // namespace ops
