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
 * \file thnn_fused_lstm_cell_def.cpp
 * \brief
 */

#include "register/op_def_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

namespace ops {
static const std::vector<ge::DataType> DTYPE_COMMON = {ge::DT_FLOAT, ge::DT_FLOAT16};
static const std::vector<ge::Format> FMT_COMMON = {ge::FORMAT_ND, ge::FORMAT_ND};
static const std::vector<ge::DataType> DTYPE_950 = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
static const std::vector<ge::Format> FMT_950 = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

class ThnnFusedLstmCell : public OpDef {
public:
    explicit ThnnFusedLstmCell(const char* name) : OpDef(name)
    {
        // 全局声明（ascend910b / ascend910_93 生效）：fp32 / fp16
        this->Input("inputGates").ParamType(REQUIRED).DataType(DTYPE_COMMON).Format(FMT_COMMON);

        this->Input("hiddenGates").ParamType(REQUIRED).DataType(DTYPE_COMMON).Format(FMT_COMMON);

        this->Input("cx").ParamType(REQUIRED).DataType(DTYPE_COMMON).Format(FMT_COMMON);

        this->Input("inputBias").ParamType(REQUIRED).DataType(DTYPE_COMMON).Format(FMT_COMMON);

        this->Input("hiddenBias").ParamType(REQUIRED).DataType(DTYPE_COMMON).Format(FMT_COMMON);

        this->Output("hy").ParamType(REQUIRED).DataType(DTYPE_COMMON).Format(FMT_COMMON);

        this->Output("cy").ParamType(REQUIRED).DataType(DTYPE_COMMON).Format(FMT_COMMON);

        this->Output("storage").ParamType(REQUIRED).DataType(DTYPE_COMMON).Format(FMT_COMMON);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true).DynamicRankSupportFlag(true).DynamicShapeSupportFlag(true);
        this->AICore().AddConfig("ascend910b", aicoreConfig);
        this->AICore().AddConfig("ascend910_93", aicoreConfig);

        // ascend950 专属：Input/Output 全量重声明，新增 bfloat16
        OpAICoreConfig aicoreConfig950;
        aicoreConfig950.Input("inputGates").ParamType(REQUIRED).DataType(DTYPE_950).Format(FMT_950);

        aicoreConfig950.Input("hiddenGates").ParamType(REQUIRED).DataType(DTYPE_950).Format(FMT_950);

        aicoreConfig950.Input("cx").ParamType(REQUIRED).DataType(DTYPE_950).Format(FMT_950);

        aicoreConfig950.Input("inputBias").ParamType(REQUIRED).DataType(DTYPE_950).Format(FMT_950);

        aicoreConfig950.Input("hiddenBias").ParamType(REQUIRED).DataType(DTYPE_950).Format(FMT_950);

        aicoreConfig950.Output("hy").ParamType(REQUIRED).DataType(DTYPE_950).Format(FMT_950);

        aicoreConfig950.Output("cy").ParamType(REQUIRED).DataType(DTYPE_950).Format(FMT_950);

        aicoreConfig950.Output("storage").ParamType(REQUIRED).DataType(DTYPE_950).Format(FMT_950);

        aicoreConfig950.DynamicCompileStaticFlag(true).DynamicRankSupportFlag(true).DynamicShapeSupportFlag(true);
        this->AICore().AddConfig("ascend950", aicoreConfig950);
    }
};

OP_ADD(ThnnFusedLstmCell);
} // namespace ops
