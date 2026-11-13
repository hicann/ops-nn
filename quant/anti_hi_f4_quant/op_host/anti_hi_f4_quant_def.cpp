/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!\\file anti_hi_f4_quant_def.cpp
 * \brief
 */

#include <cstdint>
#include "register/op_def_registry.h"
#include "../../../common/inc/op_host/op_const_def_nn.h"

namespace ops {
static constexpr int32_t DEFAULT_DST_TYPE = 27; // 27=DT_BF16

static const std::vector<ge::DataType> INPUT_X_DATA_TYPE = {ge::DT_HIFLOAT4, ge::DT_HIFLOAT4};

static const std::vector<ge::DataType> INPUT_SCALE_DATA_TYPE = {Ops::NN::DT_HIFLOAT4_SCALE, Ops::NN::DT_HIFLOAT4_SCALE};

static const std::vector<ge::DataType> OUTPUT_Y_DATA_TYPE = {ge::DT_FLOAT16, ge::DT_BF16};

static const std::vector<ge::Format> FORMAT = {ge::FORMAT_ND, ge::FORMAT_ND};

class AntiHiF4Quant : public OpDef {
public:
    explicit AntiHiF4Quant(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(INPUT_X_DATA_TYPE)
            .Format(FORMAT)
            .UnknownShapeFormat(FORMAT)
            .AutoContiguous();
        this->Input("scale")
            .ParamType(REQUIRED)
            .DataType(INPUT_SCALE_DATA_TYPE)
            .Format(FORMAT)
            .UnknownShapeFormat(FORMAT)
            .AutoContiguous();
        this->Output("y").ParamType(REQUIRED).DataType(OUTPUT_Y_DATA_TYPE).Format(FORMAT).UnknownShapeFormat(FORMAT);
        this->Attr("dst_type").AttrType(OPTIONAL).Int(DEFAULT_DST_TYPE);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
        this->AICore().AddConfig("ascend960dt", aicoreConfig);
        this->AICore().AddConfig("ascend960pr", aicoreConfig);
    }
};

OP_ADD(AntiHiF4Quant);
} // namespace ops
