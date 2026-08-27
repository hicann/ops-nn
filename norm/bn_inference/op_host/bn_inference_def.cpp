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
 * \file bn_inference_def.cpp
 * \brief Ascend 950 operator definition for BNInference.
 */
#include <vector>
#include "bn_inference_dtype.h"
#include "register/op_def_registry.h"

namespace ops {
namespace {
struct SupportLists {
    std::vector<ge::DataType> xDtypes;
    std::vector<ge::DataType> statisticDtypes;
    std::vector<ge::DataType> momentumDtypes;
    std::vector<ge::DataType> affineDtypes;
    std::vector<ge::Format> featureFormats;
    std::vector<ge::Format> parameterFormats;
};

SupportLists BuildSupportLists()
{
    SupportLists lists;
    const size_t configCount = BNInferenceSupport::FEATURE_FORMATS.size() *
                               BNInferenceSupport::DTYPE_COMBINATIONS.size();
    lists.xDtypes.reserve(configCount);
    lists.statisticDtypes.reserve(configCount);
    lists.momentumDtypes.reserve(configCount);
    lists.affineDtypes.reserve(configCount);
    lists.featureFormats.reserve(configCount);
    lists.parameterFormats.reserve(configCount);
    for (const ge::Format format : BNInferenceSupport::FEATURE_FORMATS) {
        for (const auto& combination : BNInferenceSupport::DTYPE_COMBINATIONS) {
            lists.xDtypes.push_back(combination.x);
            lists.statisticDtypes.push_back(combination.statistics);
            lists.momentumDtypes.push_back(combination.momentum);
            lists.affineDtypes.push_back(combination.affine);
            lists.featureFormats.push_back(format);
            lists.parameterFormats.push_back(ge::FORMAT_ND);
        }
    }
    return lists;
}
} // namespace

class BNInference : public OpDef {
public:
    explicit BNInference(const char* name) : OpDef(name)
    {
        const SupportLists support = BuildSupportLists();
        ConfigureTensorIo(support);
        ConfigureAttributesAndPlatform();
    }

private:
    void ConfigureTensorIo(const SupportLists& support)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(support.xDtypes)
            .Format(support.featureFormats)
            .UnknownShapeFormat(support.featureFormats)
            .AutoContiguous();
        this->Input("mean")
            .ParamType(REQUIRED)
            .DataType(support.statisticDtypes)
            .Format(support.parameterFormats)
            .UnknownShapeFormat(support.parameterFormats)
            .AutoContiguous();
        this->Input("variance")
            .ParamType(REQUIRED)
            .DataType(support.statisticDtypes)
            .Format(support.parameterFormats)
            .UnknownShapeFormat(support.parameterFormats)
            .AutoContiguous();
        this->Input("momentum")
            .ParamType(REQUIRED)
            .DataType(support.momentumDtypes)
            .Format(support.parameterFormats)
            .UnknownShapeFormat(support.parameterFormats)
            .AutoContiguous();
        this->Input("scale")
            .ParamType(OPTIONAL)
            .DataType(support.affineDtypes)
            .Format(support.parameterFormats)
            .UnknownShapeFormat(support.parameterFormats)
            .AutoContiguous();
        this->Input("offset")
            .ParamType(OPTIONAL)
            .DataType(support.affineDtypes)
            .Format(support.parameterFormats)
            .UnknownShapeFormat(support.parameterFormats)
            .AutoContiguous();
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType(support.xDtypes)
            .Format(support.featureFormats)
            .UnknownShapeFormat(support.featureFormats)
            .AutoContiguous();
    }

    void ConfigureAttributesAndPlatform()
    {
        this->Attr("epsilon").AttrType(OPTIONAL).Float(1e-5f);
        this->Attr("use_global_stats").AttrType(OPTIONAL).Bool(true);
        this->Attr("mode").AttrType(OPTIONAL).Int(1);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "bn_inference");
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(BNInference);
} // namespace ops
