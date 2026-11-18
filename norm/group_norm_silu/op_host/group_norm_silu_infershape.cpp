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
 * \file group_norm_silu_infershape.cpp
 * \brief
 */
#include "op_common/op_host/util/op_const_def.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/shape_util.h"
#include "platform/platform_infos_def.h"
#include "platform/platform_info.h"
#include "platform/soc_spec.h"
#include <set>
#include <string>
#include <vector>
#include "base/err_msg.h"

using namespace ge;
namespace ops {
static constexpr size_t GROUPNORMSILU_IDX_IN_X = 0;
static constexpr size_t GROUPNORMSILU_IDX_OUT_Y = 0;
static constexpr size_t GROUPNORMSILU_IDX_OUT_MEAN = 1;
static constexpr size_t GROUPNORMSILU_IDX_OUT_VAR = 2;
static constexpr size_t NUMGROUPS_IDX = 0;
static constexpr size_t N_IDX = 0;

static ge::graphStatus GroupNormSiluInferShape(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do GroupNormSiluInferShape");

    // get input shapes
    const gert::Shape* x_shape = context->GetInputShape(GROUPNORMSILU_IDX_IN_X);
    OP_CHECK_IF(x_shape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(
                    ((context) == nullptr || (context)->GetNodeName() == nullptr ? "group_norm_silu" :
                                                                                   (context)->GetNodeName()),
                    "x"),
                return ge::GRAPH_FAILED);

    // get output shapes
    gert::Shape* y_shape = context->GetOutputShape(GROUPNORMSILU_IDX_OUT_Y);
    if (y_shape == nullptr) {
        const std::string msgOpName = std::string(((context) == nullptr || (context)->GetNodeName() == nullptr ?
                                                       "group_norm_silu" :
                                                       (context)->GetNodeName()));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s",
                               "InferShape framework object y_shape is unavailable; check the operator registration "
                               "and context construction.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1014", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(), "InferShape framework object y_shape is unavailable; check "
                                                         "the operator registration and context construction."}));
        return ge::GRAPH_FAILED;
    };
    gert::Shape* mean_shape = context->GetOutputShape(GROUPNORMSILU_IDX_OUT_MEAN);
    if (mean_shape == nullptr) {
        const std::string msgOpName = std::string(((context) == nullptr || (context)->GetNodeName() == nullptr ?
                                                       "group_norm_silu" :
                                                       (context)->GetNodeName()));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s",
                               "InferShape framework object mean_shape is unavailable; check the operator registration "
                               "and context construction.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1014", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(), "InferShape framework object mean_shape is unavailable; check "
                                                         "the operator registration and context construction."}));
        return ge::GRAPH_FAILED;
    };
    gert::Shape* var_shape = context->GetOutputShape(GROUPNORMSILU_IDX_OUT_VAR);
    if (var_shape == nullptr) {
        const std::string msgOpName = std::string(((context) == nullptr || (context)->GetNodeName() == nullptr ?
                                                       "group_norm_silu" :
                                                       (context)->GetNodeName()));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s",
                               "InferShape framework object var_shape is unavailable; check the operator registration "
                               "and context construction.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1014", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(), "InferShape framework object var_shape is unavailable; check "
                                                         "the operator registration and context construction."}));
        return ge::GRAPH_FAILED;
    };

    *y_shape = *x_shape;
    mean_shape->SetDimNum(0);

    // process attr
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        const std::string msgOpName = std::string(((context) == nullptr || (context)->GetNodeName() == nullptr ?
                                                       "group_norm_silu" :
                                                       (context)->GetNodeName()));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s",
                               "InferShape framework object attrs is unavailable; check the operator registration and "
                               "context construction.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1014", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(), "InferShape framework object attrs is unavailable; check the "
                                                         "operator registration and context construction."}));
        return ge::GRAPH_FAILED;
    };
    const int64_t* num_groups = attrs->GetAttrPointer<int64_t>(NUMGROUPS_IDX);
    OP_CHECK_IF(num_groups == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "num_groups",
                                                         "Required num_groups attribute is absent."),
                return ge::GRAPH_FAILED);

    // update mean and var shape
    const int64_t n_dim = x_shape->GetDim(N_IDX);
    mean_shape->AppendDim(n_dim);
    mean_shape->AppendDim(*num_groups);
    *var_shape = *mean_shape;

    OP_LOGD(context, "End to do GroupNormSiluInferShape");
    return ge::GRAPH_SUCCESS;
}

static graphStatus GroupNormSiluInferDtype(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "GroupNormSiluInferDtype enter");

    // Get input tout
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        const std::string msgOpName = std::string(((context) == nullptr || (context)->GetNodeName() == nullptr ?
                                                       "group_norm_silu" :
                                                       (context)->GetNodeName()));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s",
                               "InferShape framework object attrs is unavailable; check the operator registration and "
                               "context construction.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1014", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(), "InferShape framework object attrs is unavailable; check the "
                                                         "operator registration and context construction."}));
        return ge::GRAPH_FAILED;
    };
    auto inputDtype = context->GetInputDataType(GROUPNORMSILU_IDX_IN_X);
    context->SetOutputDataType(GROUPNORMSILU_IDX_OUT_Y, inputDtype);

    fe::PlatformInfo platform_info;
    fe::OptionalInfo optional_info;
    if ((fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, optional_info) !=
         ge::GRAPH_SUCCESS)) {
        const std::string msgOpName = std::string(Ops::Base::GetOpInfo(context));
        const std::string msgReason = (std::string(__func__) + ": " + "Cannot get platform info!");
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s", msgReason.c_str());
        REPORT_PREDEFINED_ERR_MSG("EZ1014", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{msgOpName.c_str(), msgReason.c_str()}));
        return ge::GRAPH_FAILED;
    };
    OP_LOGD(context, "soc version is %s", platform_info.str_info.short_soc_version.c_str());
    const int32_t npuArch = platform_info.soc_info.arch_type;
    const std::set<int32_t> supportNpuArchList = {
        static_cast<int32_t>(Ops::Base::DAV_3510), static_cast<int32_t>(Ops::Base::DAV_5102),
        static_cast<int32_t>(Ops::Base::DAV_9201), static_cast<int32_t>(Ops::Base::DAV_9202)};
    if (supportNpuArchList.count(npuArch) != 0) {
        auto gammaDtype = context->GetOptionalInputDataType(GROUPNORMSILU_IDX_OUT_MEAN);
        auto betaDtype = context->GetOptionalInputDataType(GROUPNORMSILU_IDX_OUT_VAR);
        bool validGammaDtype = ((gammaDtype != ge::DT_UNDEFINED) && (gammaDtype != ge::DT_MAX));
        bool validBetaDtype = ((betaDtype != ge::DT_UNDEFINED) && (betaDtype != ge::DT_MAX));
        if (validGammaDtype && validBetaDtype) {
            if (gammaDtype != betaDtype) {
                std::string dtypeMsg = Ops::Base::ToString(gammaDtype) + " and " + Ops::Base::ToString(betaDtype);
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context->GetNodeName(), "gamma and beta", dtypeMsg.c_str(),
                                                       "The datatypes of gamma and beta must be the same");
                return ge::GRAPH_FAILED;
            }
        }
        if (validGammaDtype) {
            context->SetOutputDataType(GROUPNORMSILU_IDX_OUT_MEAN, gammaDtype);
            context->SetOutputDataType(GROUPNORMSILU_IDX_OUT_VAR, gammaDtype);
        } else if (validBetaDtype) {
            context->SetOutputDataType(GROUPNORMSILU_IDX_OUT_MEAN, betaDtype);
            context->SetOutputDataType(GROUPNORMSILU_IDX_OUT_VAR, betaDtype);
        } else {
            context->SetOutputDataType(GROUPNORMSILU_IDX_OUT_MEAN, inputDtype);
            context->SetOutputDataType(GROUPNORMSILU_IDX_OUT_VAR, inputDtype);
        }
    } else {
        context->SetOutputDataType(GROUPNORMSILU_IDX_OUT_MEAN, inputDtype);
        context->SetOutputDataType(GROUPNORMSILU_IDX_OUT_VAR, inputDtype);
    }

    OP_LOGD(context, "GroupNormSiluInferDtype end");

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GroupNormSilu).InferShape(GroupNormSiluInferShape).InferDataType(GroupNormSiluInferDtype);
} // namespace ops
