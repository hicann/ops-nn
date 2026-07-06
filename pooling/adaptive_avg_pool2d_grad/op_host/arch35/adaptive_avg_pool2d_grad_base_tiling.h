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
 * \file adaptive_avg_pool2d_grad_base_tiling.h
 * \brief
 */

#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_ADAPTIVE_AVG_POOL2D_GRAD_BASE_H_
#define OPS_BUILD_IN_OP_TILING_RUNTIME_ADAPTIVE_AVG_POOL2D_GRAD_BASE_H_
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "util/math_util.h"

namespace optiling {
using Ops::NN::Optiling::TilingBaseClass;

struct AdaptiveAvgPool2dGradCompileInfo {
    uint64_t coreNum;
    uint64_t ubSize;
};

struct AdaptiveAvgPool2dGradInputInfo {
    int64_t nX{1};
    int64_t cX{1};
    int64_t hX{1};
    int64_t wX{1};
    int64_t nGrad{1};
    int64_t cGrad{1};
    int64_t hGrad{1};
    int64_t wGrad{1};
    int64_t gradShapeSize{0};
    ge::DataType inputDtype{ge::DataType::DT_FLOAT};
    int64_t isInt32Meet{1};
};

class AdaptiveAvgPool2dGradTilingBase : public TilingBaseClass {
public:
    explicit AdaptiveAvgPool2dGradTilingBase(gert::TilingContext* context) : TilingBaseClass(context) {}
    ~AdaptiveAvgPool2dGradTilingBase() override {}
    bool CheckInputShape();
    ge::graphStatus CheckInputDtype();
    void SetOtherInputParams();
    ge::graphStatus SetInputParams();

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;
    bool IsGreaterThanInt32Max();

protected:
    int64_t coreNum_{0};
    int64_t ubSize_{0};
    AdaptiveAvgPool2dGradInputInfo inputData_;
};
} // namespace optiling

#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_ADAPTIVE_AVG_POOL2D_GRAD_BASE_H_
