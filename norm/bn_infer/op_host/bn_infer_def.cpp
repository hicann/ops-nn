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
 * \file bn_infer_def.cpp
 * \brief BNInfer op definition for Ascend 950.
 */
#include <vector>

#include "register/op_def_registry.h"

namespace ops {
namespace {
const std::vector<ge::DataType> X_DTYPE_LIST = {ge::DT_FLOAT16, ge::DT_FLOAT,   ge::DT_BF16,    ge::DT_FLOAT16,
                                                ge::DT_FLOAT,   ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT,
                                                ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT,   ge::DT_BF16,
                                                ge::DT_FLOAT16, ge::DT_FLOAT,   ge::DT_BF16};
const std::vector<ge::Format> X_FORMAT_LIST = {ge::FORMAT_ND,    ge::FORMAT_ND,    ge::FORMAT_ND,    ge::FORMAT_NCHW,
                                               ge::FORMAT_NCHW,  ge::FORMAT_NCHW,  ge::FORMAT_NCDHW, ge::FORMAT_NCDHW,
                                               ge::FORMAT_NCDHW, ge::FORMAT_NHWC,  ge::FORMAT_NHWC,  ge::FORMAT_NHWC,
                                               ge::FORMAT_NDHWC, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC};
const std::vector<ge::Format> X_UNKNOWN_FORMAT_LIST = {
    ge::FORMAT_ND,   ge::FORMAT_ND,    ge::FORMAT_ND,    ge::FORMAT_NCHW,  ge::FORMAT_NCHW,
    ge::FORMAT_NCHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NHWC,
    ge::FORMAT_NHWC, ge::FORMAT_NHWC,  ge::FORMAT_NDHWC, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC};
const std::vector<ge::DataType> PARAM_DTYPE_LIST(15, ge::DT_FLOAT);
const std::vector<ge::Format> PARAM_FORMAT_LIST(15, ge::FORMAT_ND);
} // namespace

class BNInfer : public OpDef {
public:
    explicit BNInfer(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(X_DTYPE_LIST)
            .Format(X_FORMAT_LIST)
            .UnknownShapeFormat(X_UNKNOWN_FORMAT_LIST);
        this->Input("scale").ParamType(REQUIRED).DataType(PARAM_DTYPE_LIST).Format(PARAM_FORMAT_LIST);
        this->Input("offset").ParamType(REQUIRED).DataType(PARAM_DTYPE_LIST).Format(PARAM_FORMAT_LIST);
        this->Input("mean").ParamType(REQUIRED).DataType(PARAM_DTYPE_LIST).Format(PARAM_FORMAT_LIST);
        this->Input("variance").ParamType(REQUIRED).DataType(PARAM_DTYPE_LIST).Format(PARAM_FORMAT_LIST);
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType(X_DTYPE_LIST)
            .Format(X_FORMAT_LIST)
            .UnknownShapeFormat(X_UNKNOWN_FORMAT_LIST);
        this->Attr("epsilon").AttrType(REQUIRED).Float();

        OpAICoreConfig regbaseConfig;
        regbaseConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "bn_infer");
        this->AICore().AddConfig("ascend950", regbaseConfig);
    }
};

OP_ADD(BNInfer);
} // namespace ops
