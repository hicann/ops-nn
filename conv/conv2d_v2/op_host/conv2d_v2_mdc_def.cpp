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
 * \file conv2d_mdc_def.cpp
 * \brief
 */

#include <cstdint>
#include <cstring>
#include "register/op_config_registry.h"

namespace ops {
std::vector<ge::DataType> conv2dv2FmapDataTypeList = {
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16
};
std::vector<ge::DataType> conv2dv2WeightDataTypeList = {
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16
};
std::vector<ge::DataType> conv2dv2BiasDataTypeList = {
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16
};
std::vector<ge::DataType> conv2dv2OffsetWDataTypeList = {
    ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8
};
std::vector<ge::DataType> conv2dv2OutputDataTypeList = {
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16
};
std::vector<ge::Format> conv2dV2FmapFormatList = {
    ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NHWC
};
std::vector<ge::Format> conv2dV2WeightFormatList = {
    ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_Z_C04
};
std::vector<ge::Format> conv2dV2BiasFormatList = {
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND
};
std::vector<ge::Format> conv2dV2OffsetWFormatList = {
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND
};
std::vector<ge::Format> conv2dV2OutputFormatList = {
    ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NHWC
};

REGISTER_OP_AICORE_CONFIG(Conv2DV2, mc62cm12a, []() {
    OpAICoreConfig aicoreConfig("mc62cm12a");
    aicoreConfig.Input("x")
        .ParamType(REQUIRED)
        .DataType(conv2dv2FmapDataTypeList)
        .Format(conv2dV2FmapFormatList)
        .UnknownShapeFormat(conv2dV2FmapFormatList);
    aicoreConfig.Input("filter")
        .ParamType(REQUIRED)
        .DataType(conv2dv2WeightDataTypeList)
        .Format(conv2dV2WeightFormatList)
        .UnknownShapeFormat(conv2dV2WeightFormatList);
    aicoreConfig.Input("bias")
        .ParamType(OPTIONAL)
        .DataType(conv2dv2BiasDataTypeList)
        .Format(conv2dV2BiasFormatList)
        .UnknownShapeFormat(conv2dV2BiasFormatList);
    aicoreConfig.Input("offset_w")
        .ParamType(OPTIONAL)
        .DataType(conv2dv2OffsetWDataTypeList)
        .Format(conv2dV2OffsetWFormatList)
        .UnknownShapeFormat(conv2dV2OffsetWFormatList);
    aicoreConfig.Output("y")
        .ParamType(REQUIRED)
        .DataType(conv2dv2OutputDataTypeList)
        .Format(conv2dV2OutputFormatList)
        .UnknownShapeFormat(conv2dV2OutputFormatList);
    aicoreConfig.DynamicCompileStaticFlag(true)
                .DynamicFormatFlag(true)
                .DynamicRankSupportFlag(true)
                .DynamicShapeSupportFlag(true)
                .NeedCheckSupportFlag(false)
                .PrecisionReduceFlag(true)
                .ExtendCfgInfo("opFile.value", "conv2d_v2")
                .ExtendCfgInfo("opInterface.value", "conv2dv2")
                .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
                .ExtendCfgInfo("jitCompile.flag", "false");

    return aicoreConfig;
});
}