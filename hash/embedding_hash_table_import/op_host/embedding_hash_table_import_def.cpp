/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file embedding_hash_table_import_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
class EmbeddingHashTableImport : public OpDef {
public:
    explicit EmbeddingHashTableImport(const char *name) : OpDef(name)
    {
        this->Input("table_handles")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_INT64 })
            .Format({ ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND });
        this->Input("embedding_dims")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_INT64 })
            .Format({ ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND });
        this->Input("bucket_sizes")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_INT64 })
            .Format({ ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND });
        this->Input("keys")
            .ParamType(DYNAMIC)
            .ValueDepend(OPTIONAL)
            .DataType({ ge::DT_INT64 })
            .Format({ ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND });
        this->Input("counters")
            .ParamType(DYNAMIC)
            .DataType({ ge::DT_UINT64 })
            .Format({ ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND });
        this->Input("filter_flags")
            .ParamType(DYNAMIC)
            .DataType({ ge::DT_UINT8 })
            .Format({ ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND });
        this->Input("values")
            .ParamType(DYNAMIC)
            .DataType({ ge::DT_FLOAT })
            .Format({ ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND });

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "embedding_hash_table_import");
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(EmbeddingHashTableImport);
} // namespace ops