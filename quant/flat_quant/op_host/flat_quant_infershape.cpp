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
 * \file flat_quant.cc
 * \brief
 */
#include "register/op_impl_registry.h"
#include "error_util.h"
#include "log/log.h"
#include "util/math_util.h"
#include "platform/platform_info.h"

using namespace ge;
namespace ops {
static std::set<std::string> FlatQuantMXFP4DavidSupportSoc = {"Ascend950"};
static const int32_t DTYPE_FLOAT4_E2M1 = 40;
static constexpr size_t FLATQUANT_K_IDX = 0;
static constexpr size_t FLATQUANT_M_IDX = 1;
static constexpr size_t FLATQUANT_N_IDX = 2;
static constexpr size_t FLATQUANT_ATTRS_NUM = 2;
static constexpr size_t FLATQUANT_MX_N_OUT_DIM = 2;
static constexpr size_t FLATQUANT_N_IS_EVEN = 2;
static constexpr size_t FLATQUANT_DOUBLE_CEIL_SIZE = 64;
static const size_t ATTR_INDEX_OF_DST_DTYPE = 1;

static bool IsFlatQuantMxFp4DavidSupport()
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    auto ret = fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo);
    return (ret == GRAPH_SUCCESS && FlatQuantMXFP4DavidSupportSoc.count(platformInfo.str_info.short_soc_version) > 0);
}

static ge::graphStatus InferShape4FlatQuant(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    const gert::Shape* xShape = context->GetInputShape(0);
    gert::Shape* outShape = context->GetOutputShape(0);
    gert::Shape* qScaleShape = context->GetOutputShape(1);
    if ((xShape == nullptr) || (outShape == nullptr) || (qScaleShape == nullptr)) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context, "FlatQuantInferShape begin");
    auto* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return GRAPH_FAILED;
    }

    if (IsFlatQuantMxFp4DavidSupport()) { // Only for 950
        const int32_t* outxDtype = nullptr;
        if (attrs->GetAttrNum() >= FLATQUANT_ATTRS_NUM) {
            outxDtype = attrs->GetAttrPointer<int32_t>(ATTR_INDEX_OF_DST_DTYPE);
        }
        OP_CHECK_IF(
            xShape->GetDim(FLATQUANT_N_IDX) % FLATQUANT_N_IS_EVEN == 1, OP_LOGE(context, "dim N must be even number"),
            return ge::GRAPH_FAILED);

        if (outxDtype != nullptr) {
            int32_t dstDtype = *outxDtype;
            if (dstDtype == DTYPE_FLOAT4_E2M1) { // dst_dtype 为40
                size_t ceilMN = (xShape->GetDim(FLATQUANT_M_IDX) * xShape->GetDim(FLATQUANT_N_IDX) +
                                 FLATQUANT_DOUBLE_CEIL_SIZE - 1) /
                                FLATQUANT_DOUBLE_CEIL_SIZE;
                outShape->SetDimNum(0);
                outShape->AppendDim(xShape->GetDim(FLATQUANT_K_IDX));
                outShape->AppendDim(xShape->GetDim(FLATQUANT_M_IDX) * xShape->GetDim(FLATQUANT_N_IDX) / 2);

                qScaleShape->SetDimNum(0);
                qScaleShape->AppendDim(xShape->GetDim(FLATQUANT_K_IDX));
                qScaleShape->AppendDim(ceilMN);
                qScaleShape->AppendDim(FLATQUANT_MX_N_OUT_DIM);
            } else { // 默认值
                outShape->SetDimNum(0);
                outShape->AppendDim(xShape->GetDim(FLATQUANT_K_IDX));
                outShape->AppendDim(xShape->GetDim(FLATQUANT_M_IDX));
                outShape->AppendDim(xShape->GetDim(FLATQUANT_N_IDX));

                qScaleShape->SetDimNum(0);
                qScaleShape->AppendDim(xShape->GetDim(FLATQUANT_K_IDX));
            }
        } else {
            outShape->SetDimNum(0);
            outShape->AppendDim(xShape->GetDim(FLATQUANT_K_IDX));
            outShape->AppendDim(xShape->GetDim(FLATQUANT_M_IDX));
            outShape->AppendDim(xShape->GetDim(FLATQUANT_N_IDX));

            qScaleShape->SetDimNum(0);
            qScaleShape->AppendDim(xShape->GetDim(FLATQUANT_K_IDX));
        }
    } else { // For 910B
        outShape->SetDimNum(0);
        outShape->AppendDim(xShape->GetDim(FLATQUANT_K_IDX));
        outShape->AppendDim(xShape->GetDim(FLATQUANT_M_IDX));
        outShape->AppendDim(xShape->GetDim(FLATQUANT_N_IDX));
        qScaleShape->SetDimNum(0);
        qScaleShape->AppendDim(xShape->GetDim(FLATQUANT_K_IDX));
    }

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4FlatQuant(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    OP_LOGD(context, "FlatQuantInferDataType begin");
    auto* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return GRAPH_FAILED;
    }

    if (IsFlatQuantMxFp4DavidSupport()) { // For 950
        const int32_t* outxDtype = nullptr;
        if (attrs->GetAttrNum() >= FLATQUANT_ATTRS_NUM) {
            outxDtype = attrs->GetAttrPointer<int32_t>(ATTR_INDEX_OF_DST_DTYPE);
        }
        if (outxDtype != nullptr) {
            int32_t dstDtype = *outxDtype;
            if (dstDtype == DTYPE_FLOAT4_E2M1) {
                context->SetOutputDataType(0, ge::DT_FLOAT4_E2M1);
                context->SetOutputDataType(1, ge::DT_FLOAT8_E8M0);
            } else {
                context->SetOutputDataType(0, ge::DT_INT4);
                context->SetOutputDataType(1, ge::DT_FLOAT);
            }
        } else {
            context->SetOutputDataType(0, ge::DT_INT4);
            context->SetOutputDataType(1, ge::DT_FLOAT);
        }
    } else { // For 910B
        context->SetOutputDataType(0, ge::DT_INT4);
        context->SetOutputDataType(1, ge::DT_FLOAT);
    }

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FlatQuant).InferShape(InferShape4FlatQuant).InferDataType(InferDataType4FlatQuant);
} // namespace ops
