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
 * \file masked_scatter_v2_def.cpp
 * \brief MaskedScatterV2 ophost（Ascend 950PR 增强版）
 *
 * 算子语义：y 为 x 的逐元素条件搬移 —— mask=true 处取 updates 的前缀序值，否则取 x。
 * v2 相对 A2 版的两点增强：
 *   1) mask 支持右对齐广播（kernel 原生分段直读，准入条件见 README「约束说明」）；
 *   2) dtype 白名单对齐 950PR kernel 已验证集合（fp32/fp16/bf16/int32）。
 *
 * [opbuild 约束] DataType/Format 链必须内联、且每个端口的列表长度等于 dtype 变体数
 * （4），opbuild 据此静态解析；请勿改回 helper 封装（会导致 The dtype size ... is 0）。
 */
#include "register/op_def_registry.h"

namespace ops {
class MaskedScatterV2 : public OpDef {
public:
    explicit MaskedScatterV2(const char* name) : OpDef(name)
    {
        // ---- x：值张量，shape/dtype 与 out 保持一致（tiling 侧校验） ----
        this->Input("x")
            .AutoContiguous()
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        // ---- mask：布尔掩码；BOOL 按 4 变体对齐 opbuild 静态解析 ----
        this->Input("mask")
            .AutoContiguous()
            .ParamType(REQUIRED)
            .DataType({ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        // ---- updates：按 mask=true 扁平顺序依次消耗 ----
        this->Input("updates")
            .AutoContiguous()
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        // ---- y：shape/dtype 跟随 x（infershape 保证） ----
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        // ---- 950PR AICore 配置：动态 shape/rank；位等语义算子关闭 need-check ----
        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true);
        aicoreConfig.DynamicFormatFlag(false);
        aicoreConfig.DynamicRankSupportFlag(true);
        aicoreConfig.DynamicShapeSupportFlag(true);
        aicoreConfig.NeedCheckSupportFlag(false);
        aicoreConfig.PrecisionReduceFlag(true);
        aicoreConfig.ExtendCfgInfo("opFile.value", "masked_scatter_v2");
        this->AICore().AddConfig("ascend950pr", aicoreConfig);
    }
};

OP_ADD(MaskedScatterV2);
} // namespace ops
