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
 * \file embedding_hash_table_export_def.cpp
 * \brief embedding_hash_table_export
 */

#include "register/op_def_registry.h"

namespace ops {
class EmbeddingHashTableExport : public OpDef {
 public:
  explicit EmbeddingHashTableExport(const char* name) : OpDef(name) {
    this->Input("table_handles")
        .ParamType(REQUIRED)
        .ValueDepend(OPTIONAL)
        .DataType({ge::DT_INT64} )
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("table_sizes")
        .ParamType(REQUIRED)
        .ValueDepend(OPTIONAL)
        .DataType({ge::DT_INT64} )
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("embedding_dims")
        .ParamType(REQUIRED)
        .ValueDepend(OPTIONAL)
        .DataType({ge::DT_INT64} )
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("bucket_sizes")
        .ParamType(REQUIRED)
        .ValueDepend(OPTIONAL)
        .DataType({ge::DT_INT64} )
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND})
        .AutoContiguous();
    this->Output("keys")
        .ParamType(DYNAMIC)
        .DataType({ge::DT_INT64})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND})
        .AutoContiguous();
    this->Output("counters")
        .ParamType(DYNAMIC)
        .DataType({ge::DT_UINT64})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND})
        .AutoContiguous();
    this->Output("filter_flags")
        .ParamType(DYNAMIC)
        .DataType({ge::DT_UINT8})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND})
        .AutoContiguous();
    this->Output("values")
        .ParamType(DYNAMIC)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND})
        .AutoContiguous();
    this->Attr("export_mode").AttrType(OPTIONAL).String("all");
    this->Attr("filtered_export_flag").AttrType(OPTIONAL).Bool(false);

    OpAICoreConfig aicore_config;
    aicore_config.DynamicCompileStaticFlag(true)
                 .DynamicFormatFlag(false)
                 .DynamicRankSupportFlag(true)
                 .DynamicShapeSupportFlag(true)
                 .NeedCheckSupportFlag(false)
                 .ExtendCfgInfo("opFile.value", "embedding_hash_table_export");
    this->AICore().AddConfig("ascend950", aicore_config);
  }
};

OP_ADD(EmbeddingHashTableExport);
}  // namespace ops