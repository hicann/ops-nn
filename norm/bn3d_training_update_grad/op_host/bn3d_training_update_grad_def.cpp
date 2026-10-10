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
 * \file bn3d_training_update_grad_def.cpp
 * \brief BN3DTrainingUpdateGrad 算子定义，声明输入输出、属性与算子配置
 */

#include <vector> // std::vector (dtype/format slot lists)
#include "register/op_def_registry.h"

namespace ops {

// ---------------------------------------------------------------------------
// class BN3DTrainingUpdateGrad : public OpDef
//
// Operator definition class. The constructor chain-calls methods to define
// inputs, outputs, and attributes using a builder pattern.
//
// Constructor parameter: name — operator instance name (for graph construction)
//   This is NOT the operator type name; it's the instance name like "bn3d_training_update_grad_1".
//
// KEY METHODS USED:
//   this->Input("name")   — begins defining a named input tensor
//     .ParamType(REQUIRED/OPTIONAL)  — whether the input must be provided
//     .DataType({...})              — list of supported data types
//     .Format({...})                — list of supported tensor formats per dtype
//     .UnknownShapeFormat({...})    — format for dynamic (unknown) shapes
//     .AutoContiguous()             — auto-convert to contiguous format if needed
//
//   this->Output("name")  — begins defining a named output tensor (same API as Input)
//
//   this->Attr("name")    — begins defining an operator attribute
//     .Int(default)        — integer attribute with default value
//     .Bool(default)       — boolean attribute with default value
//
//   OpAICoreConfig        — configuration for AICore kernel compilation
//     .DynamicCompileStaticFlag(true)   — enable dynamic compile for static shapes
//     .DynamicFormatFlag(false)         — disable dynamic format selection
//     .DynamicRankSupportFlag(true)     — enable dynamic rank support
//     .DynamicShapeSupportFlag(true)    — enable dynamic shape support
//     .NeedCheckSupportFlag(false)      — skip support check at compile time
//     .PrecisionReduceFlag(true)        — enable precision reduction (FP32→FP16)
//     .ExtendCfgInfo(key, value)        — set extended configuration
//
//   this->AICore().AddConfig("chipname", config) — associate AICore config with a chip
// ---------------------------------------------------------------------------
class BN3DTrainingUpdateGrad : public OpDef {
public:
    explicit BN3DTrainingUpdateGrad(const char* name) : OpDef(name)
    {
        // grads/x：3 dtype(fp16/fp32/bf16) × 3 format(NCDHW/NCHW/NHWC) = 9 槽。
        // 显式声明具名 format（不用 FORMAT_ND）——通道轴由 format 唯一确定（NCDHW/NCHW→轴1、
        // NHWC→末轴），ND 通道轴不可判、不受支持。
        const std::vector<ge::DataType> gxDtypes = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16,  // NCDHW × 3 dtype
                                                    ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16,  // NCHW  × 3 dtype
                                                    ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16}; // NHWC  × 3 dtype
        const std::vector<ge::Format> gxFormats = {ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW,
                                                   ge::FORMAT_NCHW,  ge::FORMAT_NCHW,  ge::FORMAT_NCHW,
                                                   ge::FORMAT_NHWC,  ge::FORMAT_NHWC,  ge::FORMAT_NHWC};
        // 统计量/输出：恒 fp32、ND（通道向量 [C]，无空间语义），9 槽与 grads/x 对齐
        const std::vector<ge::DataType> f32Dtypes(9, ge::DT_FLOAT);
        const std::vector<ge::Format> ndFormats(9, ge::FORMAT_ND);

        // —— Input 声明 ——
        this->Input("grads").ParamType(REQUIRED).DataType(gxDtypes).Format(gxFormats).UnknownShapeFormat(gxFormats);
        this->Input("x").ParamType(REQUIRED).DataType(gxDtypes).Format(gxFormats).UnknownShapeFormat(gxFormats);
        this->Input("batch_mean")
            .ParamType(REQUIRED)
            .DataType(f32Dtypes)
            .Format(ndFormats)
            .UnknownShapeFormat(ndFormats);
        this->Input("batch_variance")
            .ParamType(REQUIRED)
            .DataType(f32Dtypes)
            .Format(ndFormats)
            .UnknownShapeFormat(ndFormats);

        // —— Output 声明：恒 fp32、ND ——
        this->Output("diff_scale")
            .ParamType(REQUIRED)
            .DataType(f32Dtypes)
            .Format(ndFormats)
            .UnknownShapeFormat(ndFormats);
        this->Output("diff_offset")
            .ParamType(REQUIRED)
            .DataType(f32Dtypes)
            .Format(ndFormats)
            .UnknownShapeFormat(ndFormats);

        // —— Attr 声明 ——
        this->Attr("epsilon").AttrType(OPTIONAL).Float(0.0001);

        // —— AICore 配置（ascend950）——
        OpAICoreConfig aiCoreConfig;
        aiCoreConfig
            .DynamicCompileStaticFlag(true) // 静态 shape 统一 kernel，避免 per-shape JIT
            .DynamicFormatFlag(false)       // 固定 format 面（NCDHW/NCHW/NHWC），不做动态格式转换
            .DynamicRankSupportFlag(true)   // 支持可变 rank
            .DynamicShapeSupportFlag(true)  // 支持可变 shape
            .NeedCheckSupportFlag(false)    // 沿用 4D 骨架，跳过框架 support check
            .PrecisionReduceFlag(true)      // 允许编译器安全降精度
            // OpType BN3D... 自动 snake 化会误切为 bn3_d_...，须显式覆盖 opFile/opInterface 对齐 kernel 入口
            .ExtendCfgInfo("opFile.value", "bn3d_training_update_grad")
            .ExtendCfgInfo("opInterface.value", "bn3d_training_update_grad");
        this->AICore().AddConfig("ascend950", aiCoreConfig);
    }
};
OP_ADD(BN3DTrainingUpdateGrad);
} // namespace ops
