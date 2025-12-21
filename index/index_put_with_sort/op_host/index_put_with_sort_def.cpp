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
 * \file index_put_with_sort_def.cpp
 * \brief
 */

#include <cstdint>
#include "register/op_def_registry.h"

namespace ops {
    class IndexPutWithSort : public OpDef {
    public:
        explicit IndexPutWithSort(const char* name) : OpDef(name)
        {
            this->Input("self")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
                    .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("linear_index")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,})
                    .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("pos_idx")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,})
                    .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("values")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
                    .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("self")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
                    .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Attr("slice_size")
                    .AttrType(REQUIRED)
                    .Int();
            this->Attr("accumulate")
                    .AttrType(OPTIONAL)
                    .Bool(false);
            OpAICoreConfig aicore_config;
            aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true);
            this->AICore().AddConfig("ascend910b");
            this->AICore().AddConfig("ascend910_93");
        }
    };

    OP_ADD(IndexPutWithSort);
}