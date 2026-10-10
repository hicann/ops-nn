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
// LpNormReduce_package/op_host/lp_norm_reduce_def.cpp
// =============================================================================
//
// ROLE: Operator definition (OpDef) for LpNormReduce.
//   OpDef 算子定义，原型与
//   docs/LpNormReduce/develop/proto.md 的「OpDef 类定义」逐项一致——
//   - Input "x"（REQUIRED，fp16/fp32/bf16，ND）
//   - Output "y"（REQUIRED，fp16/fp32/bf16，ND）
//   - Attr "p"（OPTIONAL，Int，默认 2）/ "axes"（OPTIONAL，ListInt，默认 {}）/
//     "keepdim"（OPTIONAL，Bool，默认 false）/ "epsilon"（OPTIONAL，Float，默认 1e-12）
//   - AICore 仅 ascend950（不动 A2/A3），PrecisionReduceFlag(false) 禁止降精度
//   注册名 LpNormReduce 与 canndev 参考源 REG_OP(LpNormReduce)（math_ops.h:1215）
//   字面一致；dtype 列与公共原型保持一致，为 fp16/fp32/bf16 三档。
//
// CONTENTS:
//   - class LpNormReduce : public OpDef — operator definition（与 proto.md §1 一致）
//   - OP_ADD(LpNormReduce) — registration macro
//
// =============================================================================

#include "register/op_def_registry.h" // OpDef base class, OP_ADD macro

namespace ops {

// ---------------------------------------------------------------------------
// class LpNormReduce : public OpDef
//
// OpDef 类定义：与 docs/LpNormReduce/develop/proto.md §1 逐项一致。
// 构造参数 name 为算子实例名（图构建用），非算子类型名。
//
// —— Input 声明：DataType 按受支持组合列展开（fp16 / fp32 / bf16），Format / UnknownShapeFormat 全 ND ——
// —— Output 声明：与 Input 等长（输出恒与输入同 dtype）——
// —— Attr 声明：四属性与 canndev 内置 IR 逐字一致，均带默认值 → OPTIONAL ——
// ---------------------------------------------------------------------------
class LpNormReduce : public OpDef {
public:
    explicit LpNormReduce(const char* name) : OpDef(name)
    {
        // --- Input: x（唯一数据输入，ND 布局、rank ∈ [0,8] 的任意维张量）---
        // AutoContiguous: ND 连续化——归约按连续平铺切分
        // dtype 三档对齐 canndev math_ops.h:1215 REG_OP（批 b-6：补 DT_BF16，
        // bf16 输入 → fp32 计算 → bf16 输出，与 fp16 同款升精度累加模式）
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // --- Output: y（y.dtype = x.dtype，promotion = same_as_first_input）---
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // --- Attr: p（范数阶数；±inf 走整型哨兵 2147483647 / -2147483648）---
        this->Attr("p").AttrType(OPTIONAL).Int(2);

        // --- Attr: axes（规约轴；空列表 = 全部维度参与归约，负轴按 +rank(x) 归一）---
        this->Attr("axes").AttrType(OPTIONAL).ListInt({});

        // --- Attr: keepdim（false 时被归约维删除，true 时以 size=1 保留）---
        this->Attr("keepdim").AttrType(OPTIONAL).Bool(false);

        // --- Attr: epsilon（reduce 段不参与计算）---
        this->Attr("epsilon").AttrType(OPTIONAL).Float(1.0e-12f);

        // --- AICore 配置：仅 950（不动 A2/A3 既有行为），与 proto.md §1 一致 ---
        OpAICoreConfig aicoreConfig;
        aicoreConfig
            .DynamicCompileStaticFlag(true) // 静态 shape 统一 kernel，避免 per-shape JIT
            .DynamicFormatFlag(false)       // 固定 ND，无动态格式转换
            .DynamicRankSupportFlag(true)   // 支持 0D–8D 可变 rank
            .DynamicShapeSupportFlag(true)  // 支持可变 shape（信息库 [-2] ND 动态口径）
            .NeedCheckSupportFlag(false) // 跳过框架 support check（host tiling 入口已全量校验 rank/dtype/shape/p/axes）
            .PrecisionReduceFlag(false) // 禁止降精度：fp16 输入须 fp32 累加（accumulator_dtype）
            .ExtendCfgInfo("opFile.value", "lp_norm_reduce"); // kernel 文件绑定：op_kernel/lp_norm_reduce.cpp
        // 覆盖 Ascend950PR / Ascend950DT（spec.yaml supported_chips）
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

// OP_ADD(LpNormReduce):
//   Registers the LpNormReduce OpDef class with the CANN operator definition registry.
//   注册名与 canndev 参考源 REG_OP(LpNormReduce) 字面一致（proto.md「参考源接口裁定」）。
OP_ADD(LpNormReduce);
} // namespace ops
