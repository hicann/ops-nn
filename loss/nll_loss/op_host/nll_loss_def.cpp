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
 * \file nll_loss_def.cpp
 * \brief nll_loss def
 */

#include <cstdint>
#include "register/op_def_registry.h"

namespace ops {
static const std::vector<ge::Format> format = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                               ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                               ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};
static const std::vector<ge::DataType> selfDataType = {ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16,
                                                       ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16,
                                                       ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16};
static const std::vector<ge::DataType> targetDataType = {ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
                                                         ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                                                         ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8};
static constexpr int32_t DEFAULT_IGNORE_IDX = -100;
class NLLLoss : public OpDef {
public:
    explicit NLLLoss(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType(selfDataType).Format(format).UnknownShapeFormat(format);
        this->Input("target").ParamType(REQUIRED).DataType(targetDataType).Format(format).UnknownShapeFormat(format);
        this->Input("weight").ParamType(OPTIONAL).DataType(selfDataType).Format(format).UnknownShapeFormat(format);
        this->Output("y").ParamType(REQUIRED).DataType(selfDataType).Format(format).UnknownShapeFormat(format);
        this->Output("total_weight")
            .ParamType(REQUIRED)
            .DataType(selfDataType)
            .Format(format)
            .UnknownShapeFormat(format);
        this->Attr("reduction").AttrType(OPTIONAL).String("mean");
        this->Attr("ignore_index").AttrType(OPTIONAL).Int(DEFAULT_IGNORE_IDX);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .ExtendCfgInfo("opFile.value", "nll_loss_apt");
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(NLLLoss);
} // namespace ops