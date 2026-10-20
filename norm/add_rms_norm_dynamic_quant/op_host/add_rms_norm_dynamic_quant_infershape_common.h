/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ADD_RMS_NORM_DYNAMIC_QUANT_INFERSHAPE_COMMON_H
#define ADD_RMS_NORM_DYNAMIC_QUANT_INFERSHAPE_COMMON_H

#include "log/log.h"
#include "util/shape_util.h"
#include "register/op_impl_registry.h"
#include "platform/platform_info.h"

namespace ops::AddRmsNormDynamicQuantCheck {

inline bool IsRegbaseForInfer()
{
    // InferShape/InferDataType must use the GE compilation target, not the
    // current ACL device. Unknown targets keep the original inference path.
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    return fe::PlatformInfoManager::GeInstance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) ==
               ge::GRAPH_SUCCESS &&
           platformInfo.str_info.short_soc_version == "Ascend950";
}

inline bool CompatibleShape(const gert::Shape& lhs, const gert::Shape& rhs)
{
    if (Ops::Base::IsUnknownRank(lhs) || Ops::Base::IsUnknownRank(rhs)) {
        return true;
    }
    if (lhs.GetDimNum() != rhs.GetDimNum()) {
        return false;
    }
    for (size_t i = 0; i < lhs.GetDimNum(); ++i) {
        if (lhs.GetDim(i) >= 0 && rhs.GetDim(i) >= 0 && lhs.GetDim(i) != rhs.GetDim(i)) {
            return false;
        }
    }
    return true;
}

inline ge::graphStatus CheckInputShapes(gert::InferShapeContext* context)
{
    const auto* x1 = context->GetInputShape(0);
    const auto* x2 = context->GetInputShape(1);
    const auto* gamma = context->GetInputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2);
    OP_CHECK_NULL_WITH_CONTEXT(context, gamma);
    OP_CHECK_IF(!CompatibleShape(*x1, *x2), OP_LOGE(context, "x1 and x2 shapes must match."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!Ops::Base::IsUnknownRank(*gamma) && gamma->GetDimNum() != 1,
                OP_LOGE(context, "gamma must be one-dimensional."), return ge::GRAPH_FAILED);
    if (!Ops::Base::IsUnknownRank(*x1)) {
        OP_CHECK_IF(x1->GetDimNum() == 0, OP_LOGE(context, "x1 must have a reduction dimension."),
                    return ge::GRAPH_FAILED);
        if (!Ops::Base::IsUnknownRank(*gamma)) {
            const auto lastDim = x1->GetDim(x1->GetDimNum() - 1);
            const auto gammaDim = gamma->GetDim(0);
            OP_CHECK_IF(lastDim >= 0 && gammaDim >= 0 && lastDim != gammaDim,
                        OP_LOGE(context, "gamma length must match the last dimension of x1."), return ge::GRAPH_FAILED);
        }
    }
    // smooth_scale1, smooth_scale2 and beta share gamma's shape contract.
    for (size_t index = 3; index < 6; ++index) {
        const auto* shape = context->GetOptionalInputShape(index);
        if (shape != nullptr) {
            OP_CHECK_IF(!Ops::Base::IsUnknownRank(*shape) && shape->GetDimNum() != 1,
                        OP_LOGE(context, "Optional input %zu must be one-dimensional.", index),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(!CompatibleShape(*gamma, *shape),
                        OP_LOGE(context, "Optional input %zu must match gamma shape.", index), return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckOutputMask(gert::InferShapeContext* context, size_t expectedSize)
{
    const auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto* mask = attrs->GetAttrPointer<gert::ContinuousVector>(1);
    // Validate before either known-rank or unknown-rank code indexes the mask.
    if (mask == nullptr || mask->GetSize() == 0) {
        OP_CHECK_IF(context->GetOptionalInputShape(3) == nullptr && context->GetOptionalInputShape(4) != nullptr,
                    OP_LOGE(context, "smooth_scale2 requires smooth_scale1 when output_mask is empty."),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(mask->GetSize() != expectedSize,
                OP_LOGE(context, "output_mask must be empty or contain %zu flags.", expectedSize),
                return ge::GRAPH_FAILED);
    const auto* flags = static_cast<const bool*>(mask->GetData());
    OP_CHECK_IF(!flags[0] && !flags[1], OP_LOGE(context, "At least one quantized output must be enabled."),
                return ge::GRAPH_FAILED);
    for (size_t index = 0; index < 2; ++index) {
        OP_CHECK_IF(context->GetOptionalInputShape(3 + index) != nullptr && !flags[index],
                    OP_LOGE(context, "output_mask[%zu] must be true when its smooth scale is present.", index),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckInputDtypes(gert::InferDataTypeContext* context)
{
    const auto dtype = context->GetInputDataType(0);
    OP_CHECK_IF(dtype != ge::DT_FLOAT16 && dtype != ge::DT_BF16,
                OP_LOGE(context, "x1 dtype must be float16 or bfloat16."), return ge::GRAPH_FAILED);
    for (size_t index = 1; index < 6; ++index) {
        const auto inputDtype = context->GetInputDataType(index);
        // All present tensor inputs share this dtype. DT_UNDEFINED is only
        // permitted for absent optional inputs, never for x2 or gamma.
        OP_CHECK_IF((index < 3 || inputDtype != ge::DT_UNDEFINED) && inputDtype != dtype,
                    OP_LOGE(context, "Input %zu dtype must match x1.", index), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace ops::AddRmsNormDynamicQuantCheck
#endif
