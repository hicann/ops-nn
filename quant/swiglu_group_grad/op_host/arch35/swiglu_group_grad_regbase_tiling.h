/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file swiglu_group_grad_tiling.h
 * \brief SwigluGroupGrad tiling header for Ascend950 (DAV_3510) architecture
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_SWIGLU_GROUP_GRAD_ARCH35_OPTILING
#define OPS_BUILD_IN_OP_TILING_RUNTIME_SWIGLU_GROUP_GRAD_ARCH35_OPTILING

#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "swiglu_group_grad_tiling_base.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;
using Ops::Base::FloorDiv;
using Ops::NN::Optiling::TilingBaseClass;

class SwigluGroupGradArch35Tiling : public TilingBaseClass {
public:
    explicit SwigluGroupGradArch35Tiling(gert::TilingContext* context)
        : TilingBaseClass(context), tilingContext(context) {};
    ~SwigluGroupGradArch35Tiling() override {};

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;

private:
    ge::graphStatus CalcDtype();
    ge::graphStatus CheckShape();
    ge::graphStatus ParseOptionalInputs();
    ge::graphStatus ParseAttrs();
    ge::graphStatus CalcTilingStrategy();

    gert::TilingContext* tilingContext = nullptr;
    SwigluGroupGradTilingData* tiling = nullptr;
    ge::DataType gradYDtype = ge::DT_UNDEFINED;
    int64_t totalRows_ = 1;
    int64_t dim2H_ = 1;
    int64_t H_ = 1;
    int64_t isWeight_ = 0;
    int64_t isYOrigin_ = 0;
    int64_t isGroupIndex_ = 0;
    int64_t hasClamp_ = 0;
    float clampLimit_ = 0.0f;
    int64_t coreNumAll_ = 0;
    uint64_t ubSize_ = 0;
    int64_t groupIndexG_ = 0;
    uint64_t schMode_ = TPL_REGBASE_KERNEL;
};

} // namespace optiling
#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_SWIGLU_GROUP_GRAD_ARCH35_OPTILING
