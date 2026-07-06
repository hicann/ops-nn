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
 * \file adaptive_max_pool3d_grad_tiling_arch35.h
 * \brief
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_ADAPTIVE_MAX_POOL3D_GRAD_ARCH35_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_ADAPTIVE_MAX_POOL3D_GRAD_ARCH35_H

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "util/math_util.h"
#include "../op_kernel/arch35/adaptive_max_pool3d_grad_struct.h"

namespace optiling {
using Ops::NN::Optiling::TilingBaseClass;
using namespace AdaptiveMaxPool3dGradOp;

struct AdaptiveMaxPool3dGradInputInfoV35 {
    int64_t nX{1};
    int64_t cX{1};
    int64_t dX{1};
    int64_t hX{1};
    int64_t wX{1};
    int64_t nGrad{1};
    int64_t cGrad{1};
    int64_t dGrad{1};
    int64_t hGrad{1};
    int64_t wGrad{1};
    int64_t gradShapeSize{0};
    ge::DataType inputDtype{ge::DataType::DT_FLOAT};
    ge::DataType argmaxDtype{ge::DataType::DT_INT32};
    ge::Format inputFormat{ge::Format::FORMAT_NCDHW};
    int64_t isInt32Meet{1};
};

struct AdaptiveMaxPool3dGradCompileInfoV35 {
    uint32_t coreNum = 0;
    uint64_t sysWorkspaceSize = 0;
    uint64_t ubSizePlatForm = 0;
};

class AdaptiveMaxPool3dGradTilingBaseV35 : public TilingBaseClass {
public:
    explicit AdaptiveMaxPool3dGradTilingBaseV35(gert::TilingContext* context) : TilingBaseClass(context) {}
    ~AdaptiveMaxPool3dGradTilingBaseV35() override {}

    const std::string nodeName = "AdaptiveMaxPool3DGrad";
    AdaptiveMaxPool3dGradTilingDataV35* tilingData_ = context_->GetTilingData<AdaptiveMaxPool3dGradTilingDataV35>();
    AdaptiveMaxPool3dGradInputInfoV35 inputData;
    int64_t coreNum_{0};
    int64_t ubSize_{0};

    bool CheckInputShape();
    ge::graphStatus CheckInputDtype();
    ge::graphStatus SetInputParams();
    void SetOtherInputParams();

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;
};

class AdaptiveMaxPool3dGradTilingSimt : public AdaptiveMaxPool3dGradTilingBaseV35 {
public:
    explicit AdaptiveMaxPool3dGradTilingSimt(gert::TilingContext* context) : AdaptiveMaxPool3dGradTilingBaseV35(context)
    {}
    ~AdaptiveMaxPool3dGradTilingSimt() override {}

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    uint64_t GetTilingKey() const override;
};

} // namespace optiling

#endif
