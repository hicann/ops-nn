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
 * \file mish_grad.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
class MishGrad : public OpDef {
public:
    explicit MishGrad(const char* name) : OpDef(name)
    {
        this->Input("grad")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_NZ})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_NZ});
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_NZ})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_NZ});
        this->Input("tanhx")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_NZ})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_NZ});
        this->Output("x_grad")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_NZ})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_NZ});

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "mish_grad");    // 这里制定的值会对应到kernel入口文件名.cpp
        this->AICore().AddConfig("ascend910b", aicoreConfig); // 其他的soc版本补充部分配置项
    }
};
OP_ADD(MishGrad); // 添加算子信息库
} // namespace ops
