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
 * \file erfinv_def.cpp
 * \brief erfinv def
 */

#include <cstdint>
#include "register/op_def_registry.h"

namespace ops {
static const std::vector<ge::DataType> erfinvDataType = {ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT};

static const std::vector<ge::Format> erfinvDataFormat = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

class Erfinv : public OpDef {
public:
    explicit Erfinv(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(erfinvDataType)
            .Format(erfinvDataFormat)
            .UnknownShapeFormat(erfinvDataFormat);
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType(erfinvDataType)
            .Format(erfinvDataFormat)
            .UnknownShapeFormat(erfinvDataFormat);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "erfinv_apt");
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(Erfinv);
} // namespace ops