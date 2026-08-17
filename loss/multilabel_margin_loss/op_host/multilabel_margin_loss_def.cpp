/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */
#include "register/op_def_registry.h"

namespace ops {
class MultilabelMarginLoss : public OpDef {
public:
    explicit MultilabelMarginLoss(const char* name) : OpDef(name)
    {
        // op 级原型 = ascend910b/ascend910_93 基线, is_target 保持 IR 的 INT32(3 组: x/y ∈ fp32/fp16/bf16)。
        // regbase(ascend950) 扩展了 is_target(aclnn 路径跟随 self),原型有差异,故在 ascend950 的 regbaseConfig
        // 里【独立重新定义】输入输出(6 组),不改动基线原型。(参照 activation/clipped_swiglu 独立 config 写法)
        static const std::vector<ge::DataType> xyDtypeBase = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
        static const std::vector<ge::DataType> tgtDtypeBase = {ge::DT_INT32, ge::DT_INT32, ge::DT_INT32};
        static const std::vector<ge::Format> fmtBase = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};
        this->Input("x").ParamType(REQUIRED).DataType(xyDtypeBase).Format(fmtBase).UnknownShapeFormat(fmtBase);
        this->Input("target").ParamType(REQUIRED).DataType(tgtDtypeBase).Format(fmtBase).UnknownShapeFormat(fmtBase);
        this->Output("y").ParamType(REQUIRED).DataType(xyDtypeBase).Format(fmtBase).UnknownShapeFormat(fmtBase);
        this->Output("is_target")
            .ParamType(REQUIRED)
            .DataType(tgtDtypeBase)
            .Format(fmtBase)
            .UnknownShapeFormat(fmtBase);
        this->Attr("reduction").AttrType(OPTIONAL).String("mean");

        // ascend910b/ascend910_93: 继承 op 级基线原型(is_target INT32), 仅带原有 flags。基线原型不变。
        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "multilabel_margin_loss");

        // regbase(ascend950): 独立重定义输入输出——is_target 扩展为 6 组(前 3 GE 保 int32, 后 3 aclnn 跟随 self)。
        static const std::vector<ge::DataType> xy6 = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,
                                                      ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
        static const std::vector<ge::DataType> tgt6 = {ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
                                                       ge::DT_INT32, ge::DT_INT32, ge::DT_INT32};
        static const std::vector<ge::DataType> ist6 = {ge::DT_INT32, ge::DT_INT32,   ge::DT_INT32,
                                                       ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
        static const std::vector<ge::Format> fmt6 = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};
        OpAICoreConfig regbaseConfig;
        regbaseConfig.Input("x").ParamType(REQUIRED).DataType(xy6).Format(fmt6).UnknownShapeFormat(fmt6);
        regbaseConfig.Input("target").ParamType(REQUIRED).DataType(tgt6).Format(fmt6).UnknownShapeFormat(fmt6);
        regbaseConfig.Output("y").ParamType(REQUIRED).DataType(xy6).Format(fmt6).UnknownShapeFormat(fmt6);
        regbaseConfig.Output("is_target").ParamType(REQUIRED).DataType(ist6).Format(fmt6).UnknownShapeFormat(fmt6);
        regbaseConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "multilabel_margin_loss");
        this->AICore().AddConfig("ascend950", regbaseConfig);
    }
};
OP_ADD(MultilabelMarginLoss);
} // namespace ops
