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
#include <vector>

namespace ops {
static const std::vector<ge::DataType> GATHER_V2_X_TYPES = {ge::DT_FLOAT,  ge::DT_FLOAT16, ge::DT_INT8,  ge::DT_INT16,
                                                            ge::DT_INT32,  ge::DT_INT64,   ge::DT_UINT8, ge::DT_UINT16,
                                                            ge::DT_UINT32, ge::DT_UINT64,  ge::DT_BOOL,  ge::DT_DOUBLE};

static const std::vector<ge::DataType> GATHER_V2_INDEX_TYPES = {ge::DT_INT32, ge::DT_INT64};

static std::vector<ge::DataType> BuildGatherV2XTypes()
{
    std::vector<ge::DataType> types;
    for (auto xType : GATHER_V2_X_TYPES) {
        for (size_t indexTypeIdx = 0; indexTypeIdx < GATHER_V2_INDEX_TYPES.size(); ++indexTypeIdx) {
            for (size_t axisTypeIdx = 0; axisTypeIdx < GATHER_V2_INDEX_TYPES.size(); ++axisTypeIdx) {
                types.push_back(xType);
            }
        }
    }
    return types;
}

static std::vector<ge::DataType> BuildGatherV2IndexTypes()
{
    std::vector<ge::DataType> types;
    for (size_t xTypeIdx = 0; xTypeIdx < GATHER_V2_X_TYPES.size(); ++xTypeIdx) {
        for (auto indexType : GATHER_V2_INDEX_TYPES) {
            for (size_t axisTypeIdx = 0; axisTypeIdx < GATHER_V2_INDEX_TYPES.size(); ++axisTypeIdx) {
                types.push_back(indexType);
            }
        }
    }
    return types;
}

static std::vector<ge::DataType> BuildGatherV2AxisTypes()
{
    std::vector<ge::DataType> types;
    for (size_t xTypeIdx = 0; xTypeIdx < GATHER_V2_X_TYPES.size(); ++xTypeIdx) {
        for (size_t indexTypeIdx = 0; indexTypeIdx < GATHER_V2_INDEX_TYPES.size(); ++indexTypeIdx) {
            for (auto axisType : GATHER_V2_INDEX_TYPES) {
                types.push_back(axisType);
            }
        }
    }
    return types;
}

static const std::vector<ge::DataType> GATHER_V2_X_TYPE_COMBINATIONS = BuildGatherV2XTypes();
static const std::vector<ge::DataType> GATHER_V2_INDEX_TYPE_COMBINATIONS = BuildGatherV2IndexTypes();
static const std::vector<ge::DataType> GATHER_V2_AXIS_TYPE_COMBINATIONS = BuildGatherV2AxisTypes();
static const std::vector<ge::Format> GATHER_V2_FORMATS(GATHER_V2_X_TYPE_COMBINATIONS.size(), ge::FORMAT_ND);

class GatherV2 : public OpDef {
public:
    explicit GatherV2(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType(GATHER_V2_X_TYPE_COMBINATIONS).Format(GATHER_V2_FORMATS);
        this->Input("indices")
            .ParamType(REQUIRED)
            .DataType(GATHER_V2_INDEX_TYPE_COMBINATIONS)
            .Format(GATHER_V2_FORMATS);
        this->Input("axis").ParamType(REQUIRED).DataType(GATHER_V2_AXIS_TYPE_COMBINATIONS).Format(GATHER_V2_FORMATS);
        this->Output("y").ParamType(REQUIRED).DataType(GATHER_V2_X_TYPE_COMBINATIONS).Format(GATHER_V2_FORMATS);
        this->Attr("batch_dims").AttrType(OPTIONAL).Int(0);
        this->Attr("negative_index_support").AttrType(OPTIONAL).Bool(false);
        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(GatherV2);
} // namespace ops
