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
 * \file embedding_hash_table_lookup_or_insert_def.cpp
 * \brief embedding_hash_table_lookup_or_insert
 */
#include "register/op_def_registry.h"

namespace ops {
class EmbeddingHashTableLookupOrInsert : public OpDef {
 public:
  explicit EmbeddingHashTableLookupOrInsert(const char* name) : OpDef(name) {
    this->Input("table_handle")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT64})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("keys")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT64})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Output("values")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Attr("bucket_size").AttrType(REQUIRED).Int();
    this->Attr("embedding_dim").AttrType(REQUIRED).Int();
    this->Attr("filter_mode").AttrType(OPTIONAL).String("no_filter");
    this->Attr("filter_freq").AttrType(OPTIONAL).Int(0);
    this->Attr("default_key_or_value").AttrType(OPTIONAL).Bool(false);
    this->Attr("default_key").AttrType(OPTIONAL).Int(0);
    this->Attr("default_value").AttrType(OPTIONAL).Float(0.0);
    this->Attr("filter_key_flag").AttrType(OPTIONAL).Bool(false);
    this->Attr("filter_key").AttrType(OPTIONAL).Int(-1);

    OpAICoreConfig aicore_config;
    aicore_config.DynamicCompileStaticFlag(true)
        .DynamicRankSupportFlag(true)
        .DynamicShapeSupportFlag(true)
        .NeedCheckSupportFlag(false)
        .ExtendCfgInfo("opFile.value", "embedding_hash_table_lookup_or_insert");
    this->AICore().AddConfig("ascend950", aicore_config);
  }
};

OP_ADD(EmbeddingHashTableLookupOrInsert);
}  // namespace ops