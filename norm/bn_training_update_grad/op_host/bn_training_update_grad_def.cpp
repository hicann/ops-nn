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
 * \file bn_training_update_grad_def.cpp
 * \brief BNTrainingUpdateGrad op def (ascend950 vendor 注册)
 *        built-in proto 已有完整定义（op_graph/bn_training_update_grad_proto.h 逐字照抄 canndev）：
 *          grads/x fp16/fp32/bf16（二者同型）；batch_mean/batch_variance 恒 fp32（910b TBE para_check 契约）
 *          epsilon 为 OPTIONAL float 属性，缺省 0.0001（与 proto `.ATTR(epsilon, Float, 0.0001)` 一致）
 *          diff_scale/diff_offset 恒 fp32
 *        格式：grads/x 双格式 6 槽（ND×3 dtype + NHWC×3 dtype，ND 布局 dim0=N/dim1=C/后导维 R、
 *        NHWC 布局 C=最后一维，A2 同款支持面）；统计量/输出恒 ND（元素数=C 校验，[C] 推荐）
 */

#include "register/op_def_registry.h"

namespace ops {
class BNTrainingUpdateGrad : public OpDef {
public:
    explicit BNTrainingUpdateGrad(const char* name) : OpDef(name)
    {
        // 共享注册面:grads/x 三 dtype × 双格式（ND/NHWC）= 6 槽笛卡尔（batch_norm_def 先例）；
        // 统计量/输出恒 fp32 ND（元素数=C 校验，槽位对齐 grads 逐槽配 ND）
        const std::vector<ge::DataType> xDtypes = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16,
                                                   ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};
        const std::vector<ge::DataType> f32Dtypes(6, ge::DT_FLOAT);
        const std::vector<ge::Format> gxFormats = {ge::FORMAT_ND,   ge::FORMAT_ND,   ge::FORMAT_ND,
                                                   ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC};
        const std::vector<ge::Format> ndFormats(6, ge::FORMAT_ND);
        this->Input("grads").ParamType(REQUIRED).DataType(xDtypes).Format(gxFormats).UnknownShapeFormat(gxFormats);
        this->Input("x").ParamType(REQUIRED).DataType(xDtypes).Format(gxFormats).UnknownShapeFormat(gxFormats);
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

        this->Attr("epsilon").AttrType(OPTIONAL).Float(0.0001);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "bn_training_update_grad");
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(BNTrainingUpdateGrad);
} // namespace ops
