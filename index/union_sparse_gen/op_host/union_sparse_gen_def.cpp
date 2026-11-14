/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "register/op_def_registry.h"
#include "union_sparse_gen_contract.h"

namespace Contract = UnionSparseGenContract;

namespace ops {
class UnionSparseGen : public OpDef {
public:
    explicit UnionSparseGen(const char* name) : OpDef(name)
    {
        this->Input("sparse_block_idx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("sparse_block_count")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("cu_sqlen_q")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("cu_sqlen_kv")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("seq_used_q")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("seq_used_kv")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->Output("union_sparse_idx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("union_sparse_count")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("exact_mask")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Attr("block_shape").AttrType(OPTIONAL).ListInt({Contract::kSupportedBlockX, Contract::kSupportedBlockY});
        this->Attr("union_size").AttrType(OPTIONAL).Int(Contract::kSupportedUnionSize);
        this->Attr("mask_type").AttrType(OPTIONAL).Int(Contract::kDefaultMaskType);
        this->Attr("totalq_union_num").AttrType(OPTIONAL).Int(Contract::kUnspecifiedGeometry);
        this->Attr("max_logical_blocks").AttrType(OPTIONAL).Int(Contract::kUnspecifiedGeometry);
        this->Attr("max_q_seqlen").AttrType(OPTIONAL).Int(Contract::kUnspecifiedMaxSeqLen);
        this->Attr("max_kv_seqlen").AttrType(OPTIONAL).Int(Contract::kUnspecifiedMaxSeqLen);
        this->Attr("layout_sparse_pattern").AttrType(OPTIONAL).Int(Contract::kSparsePatternLayout);
        this->Attr("is_input_idx_sorted").AttrType(OPTIONAL).Bool(true);
        this->Attr("residual_block_mode").AttrType(OPTIONAL).Int(Contract::kExplicitBlocks);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
        this->AICore().AddConfig("ascend910b", aicoreConfig);
        this->AICore().AddConfig("ascend910_93", aicoreConfig);
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(UnionSparseGen);
} // namespace ops
