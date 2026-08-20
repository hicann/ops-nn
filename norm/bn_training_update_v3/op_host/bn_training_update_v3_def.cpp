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
 * \file bn_training_update_v3_def.cpp
 * \brief BNTrainingUpdateV3 op def (ascend950 vendor 注册)
 *        built-in proto 已有完整定义（op_graph/bn_training_update_v3_proto.h 逐字照抄 canndev）：
 *          x fp16/fp32/bf16；sum/square_sum/scale/offset 恒 fp32（910b TBE para_check 契约）
 *          epsilon 为 REQUIRED float 属性（与 proto 一致）
 *          y 同 x；batch_mean/batch_variance/reserve_1/reserve_2 fp32
 *        格式：仅 ND（dim0=N、dim1=C、后导维为归一化轴 R，统计量 [C]）
 */

#include "register/op_def_registry.h"

namespace ops {
class BNTrainingUpdateV3 : public OpDef {
public:
    explicit BNTrainingUpdateV3(const char* name) : OpDef(name)
    {
        // 共享注册面:x/y 三路 dtype,统计量恒 fp32,全部 ND（压缩行数用具名 vector,声明逐条显式）
        const std::vector<ge::DataType> xDtypes = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};
        const std::vector<ge::DataType> f32Dtypes = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
        const std::vector<ge::Format> ndFormats = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};
        this->Input("x").ParamType(REQUIRED).DataType(xDtypes).Format(ndFormats).UnknownShapeFormat(ndFormats);
        this->Input("sum").ParamType(REQUIRED).DataType(f32Dtypes).Format(ndFormats).UnknownShapeFormat(ndFormats);
        this->Input("square_sum")
            .ParamType(REQUIRED)
            .DataType(f32Dtypes)
            .Format(ndFormats)
            .UnknownShapeFormat(ndFormats);
        this->Input("scale").ParamType(REQUIRED).DataType(f32Dtypes).Format(ndFormats).UnknownShapeFormat(ndFormats);
        this->Input("offset").ParamType(REQUIRED).DataType(f32Dtypes).Format(ndFormats).UnknownShapeFormat(ndFormats);
        this->Output("y").ParamType(REQUIRED).DataType(xDtypes).Format(ndFormats).UnknownShapeFormat(ndFormats);
        this->Output("batch_mean")
            .ParamType(REQUIRED)
            .DataType(f32Dtypes)
            .Format(ndFormats)
            .UnknownShapeFormat(ndFormats);
        this->Output("batch_variance")
            .ParamType(REQUIRED)
            .DataType(f32Dtypes)
            .Format(ndFormats)
            .UnknownShapeFormat(ndFormats);
        this->Output("reserve_1")
            .ParamType(REQUIRED)
            .DataType(f32Dtypes)
            .Format(ndFormats)
            .UnknownShapeFormat(ndFormats);
        this->Output("reserve_2")
            .ParamType(REQUIRED)
            .DataType(f32Dtypes)
            .Format(ndFormats)
            .UnknownShapeFormat(ndFormats);

        this->Attr("epsilon").AttrType(REQUIRED).Float();

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "bn_training_update_v3");
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(BNTrainingUpdateV3);
} // namespace ops
