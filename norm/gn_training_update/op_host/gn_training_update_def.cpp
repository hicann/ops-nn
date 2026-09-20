/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// =============================================================================
// gn_training_update_package/op_host/gn_training_update_def.cpp
// =============================================================================
//
// ROLE: Operator definition (OpDef) for GnTrainingUpdate.
//   This file defines the operator's complete interface to the CANN AICore
//   compilation pipeline. It specifies:
//   - Input/output tensor names, dtypes, formats, and whether they are REQUIRED or OPTIONAL
//   - Operator attributes (num_groups, epsilon) with default values
//   - AICore configuration: dynamic shape/rank support, precision reduction, kernel file link
//
//   The OpDef class is used by op_build to generate stubs and by the tiling
//   infrastructure to understand the operator interface at compile time.
//
// CONTENTS:
//   - class GNTrainingUpdate : public OpDef — operator definition
//   - OP_ADD(GNTrainingUpdate) — registration macro
//
// OPERATOR NAME VARIANTS:
//   PascalCase   : GnTrainingUpdate   — class name, OP_ADD argument
//   snake_case   : gn_training_update  — filename
//   UPPER_SNAKE  : GN_TRAINING_UPDATE  — (not directly used)
//
// NAME REPLACEMENT RULES (to create FooBar operator):
//   GnTrainingUpdate → FooBar                           (class name, OP_ADD)
//   gn_training_update_def → foo_bar_def                 (filename)
//   gn_training_update → foo_bar                       (kernel file reference)
//   ascend950 → appropriate compute unit for target hardware
//
// =============================================================================

#include "register/op_def_registry.h" // OpDef base class, OP_ADD macro

namespace ops {

class GNTrainingUpdate : public OpDef {
public:
    explicit GNTrainingUpdate(const char* name) : OpDef(name)
    {
        // —— Input 声明：ParamType + DataType（按「算子规格」组合列展开）+ Format + UnknownShapeFormat，所有 IO 等长 ——
        // x 的 format 组合同时覆盖 y=ND（aclnn 通路 L0 自分配输出恒为 ND）与
        // y=同标签（图模式引擎按标签配对）两种运行时键：(x,y) ∈
        // {(ND,ND),(NCHW,ND),(NHWC,ND),(NCHW,NCHW),(NHWC,NHWC)} × {fp16,fp32}
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_ND,
                     ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NHWC})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NHWC,
                                 ge::FORMAT_ND, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NHWC});
        this->Input("sum")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("square_sum")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("scale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("offset")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("mean")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("variance")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // —— Output 声明：同 Input 结构 ——
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_NCHW, ge::FORMAT_NHWC})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_NCHW, ge::FORMAT_NHWC,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_NCHW, ge::FORMAT_NHWC});
        this->Output("batch_mean")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("batch_variance")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        // —— Attr 声明（官方 IR 默认值：num_groups=2、epsilon=0.0001）——
        this->Attr("num_groups").AttrType(OPTIONAL).Int(2);
        this->Attr("epsilon").AttrType(OPTIONAL).Float(0.0001);

        // —— AICore 配置 ——
        OpAICoreConfig aicoreConfig;
        aicoreConfig
            .DynamicCompileStaticFlag(true)                       // 静态 shape 统一 kernel，避免 per-shape JIT
            .DynamicFormatFlag(false)                             // 不支持动态格式转换（固定 ND）
            .DynamicRankSupportFlag(true)                         // 支持 1D–8D 可变 rank
            .DynamicShapeSupportFlag(true)                        // 支持可变 shape
            .NeedCheckSupportFlag(false)                          // 跳过框架 support check（由 L2 aclnn 自检）
            .PrecisionReduceFlag(true)                            // 允许编译器安全降精度（fp32→fp16）
            .ExtendCfgInfo("opFile.value", "gn_training_update"); // 指向 kernel 入口文件 gn_training_update.cpp
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

// 双注册：GNTrainingUpdate 为官方 IR 图类型名（GE 图模式引擎查找键）；
// GnTrainingUpdate 为 aclnn/GEIR 测试通路使用的 op 类型名（冻结 GEIR baseline
// 的 op_type），回调函数体共用。
class GnTrainingUpdate : public GNTrainingUpdate {
public:
    using GNTrainingUpdate::GNTrainingUpdate;
};
OP_ADD(GNTrainingUpdate);
OP_ADD(GnTrainingUpdate);
} // namespace ops
