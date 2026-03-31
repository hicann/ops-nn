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
 * \file adaptive_avg_pool3d_grad_tiling_arch35.h
 * \brief
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_ADAPTIVE_AVG_POOL3D_GRAD_ARCH35_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_ADAPTIVE_AVG_POOL3D_GRAD_ARCH35_H

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "util/math_util.h"
#include "../op_kernel/arch35/adaptive_avg_pool3d_grad_struct.h"

namespace optiling {
using Ops::NN::Optiling::TilingBaseClass;
using namespace AdaptiveAvgPool3dGradOp;

struct AdaptiveAvgPool3dGradInputInfo {
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
    ge::Format inputFormat{ge::Format::FORMAT_NCDHW};
    int64_t isInt32Meet{1};
};

class AdaptiveAvgPool3dGradTilingBaseV35 : public TilingBaseClass {
public:
    explicit AdaptiveAvgPool3dGradTilingBaseV35(gert::TilingContext* context) : TilingBaseClass(context)
    {}
    ~AdaptiveAvgPool3dGradTilingBaseV35() override
    {}

    const std::string nodeName = "AdaptiveAvgPool3dGrad";
    //71TilingData
    AdaptiveAvgPool3dGradTilingDataV35* tilingData_ = context_->GetTilingData<AdaptiveAvgPool3dGradTilingDataV35>();
    AdaptiveAvgPool3dGradInputInfo inputData;
    int64_t coreNum_{0};
    int64_t ubSize_{0};

    bool CheckInputShape();
    ge::graphStatus CheckInputDtype();
    ge::graphStatus CheckAttrShape();
    ge::graphStatus SetInputParams();
    ge::graphStatus SetAttrParams();
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

class AdaptiveAvgPool3dGradTilingSimt : public AdaptiveAvgPool3dGradTilingBaseV35 {
public:
    explicit AdaptiveAvgPool3dGradTilingSimt(gert::TilingContext* context)
        : AdaptiveAvgPool3dGradTilingBaseV35(context)
    {}
    ~AdaptiveAvgPool3dGradTilingSimt() override
    {}

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;
    bool NeedInt64(int64_t isize, int64_t osize) const;
};

}  // namespace optiling

#endif