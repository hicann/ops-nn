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
 * \file matmul_v2_compress_dequant_def.cpp
 * \brief MatMulV2CompressDequant op definition for ascend310p
 */
#include "register/op_def_registry.h"

namespace ops {
class MatMulV2CompressDequant : public OpDef {
public:
    explicit MatMulV2CompressDequant(const char* name) : OpDef(name)
    {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("compress_index")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("deq_scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("bias")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("offset_w")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("transpose_x1").AttrType(OPTIONAL).Bool(false);
        this->Attr("transpose_x2").AttrType(OPTIONAL).Bool(false);
        this->Attr("compress_info").AttrType(OPTIONAL).ListInt({1, 1, 1, 1, 1});
        this->Attr("offset_x").AttrType(OPTIONAL).Int(0);
        this->Attr("alg").AttrType(OPTIONAL).String("weight_unzip");
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        OpAICoreConfig config_310p;
        config_310p.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .ExtendCfgInfo("opFile.value", "matmul_v2_compress_dequant");
        this->AICore().AddConfig("ascend310p", config_310p);
    }
};

OP_ADD(MatMulV2CompressDequant);
} // namespace ops
