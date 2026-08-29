/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file avg_pool3_d_grad_tiling_base.h
 * \brief 3D average pooling backward shared tiling base (arch35/runtime2.0).
 *        Modeled on max_pool_grad_tiling_base.h: base class implements the common
 *        parse/validate functions; scheme tiling classes inherit it and implement
 *        only the differentiated functions (IsCapable/GetTilingKey/DoOpTiling/...).
 */

#ifndef OP_IMPL_AVG_POOL3_D_GRAD_TILING_BASE_H_
#define OP_IMPL_AVG_POOL3_D_GRAD_TILING_BASE_H_

#include <array>

#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "avg_pool3_d_grad_tiling_common.h"

namespace optiling {
using Ops::NN::Optiling::TilingBaseClass;

struct AvgPool3DGradCompileInfo {
    uint64_t coreNum;
    uint64_t ubSize;
};

struct AvgPool3DCommon {
    int64_t nDim;
    int64_t cDim;
    int64_t dDim;
    int64_t hDim;
    int64_t wDim;
};

// Shared base for all NCDHW/NDHWC/SIMT schemes. Implements the common parse/validate
// functions and provides default empty implementations of the TilingBaseClass virtual
// hooks. Derived scheme classes override only the differentiated functions.
class AvgPool3DGradTilingBase : public TilingBaseClass {
public:
    explicit AvgPool3DGradTilingBase(gert::TilingContext* context) : TilingBaseClass(context) {}
    ~AvgPool3DGradTilingBase() override {}

protected:
    // Common parse/validate, orchestrated by GetShapeAttrsInfo().
    bool CheckInputShape();
    ge::graphStatus CheckInputDtype();
    ge::graphStatus CheckAttrVal();
    ge::graphStatus CheckAttrShape();
    ge::graphStatus CheckGradValid();
    ge::graphStatus SetInputParams();
    ge::graphStatus SetAttrParams();
    void SetOtherInputParams();

    // TilingBaseClass virtual hooks: defaults here, schemes override differences.
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetWorkspaceSize() override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;

public:
    AvgPool3DGradInputInfo inputData;
    uint64_t coreNum = 1;
    uint64_t ubSize = 0;

private:
    // Helpers operating on the member inputData (parsed attrs/shapes).
    bool IsGreaterThanInt32Max() const;
    void SetBatchChannelInfo(const ge::Format format, const bool is5d, const int32_t* shapeValue,
                             const AvgPool3DCommon& origDims, const gert::Shape& gradShape);
    void SetKernelSizeInfo(const gert::RuntimeAttrs* runtimeAttrs, const AvgPool3DCommon& commInfo);
    void SetStrideInfo(const gert::RuntimeAttrs* runtimeAttrs, const AvgPool3DCommon& commInfo);
    void SetPadInfo(const gert::RuntimeAttrs* runtimeAttrs);
    void SetMiscAttrs(const gert::RuntimeAttrs* runtimeAttrs);
    bool IsKernelStrideValid() const;
    bool IsPadValid() const;
    void ComputeExpectedShape(int64_t& expectedD, int64_t& expectedH, int64_t& expectedW) const;
};

ge::graphStatus Tiling4AvgPool3DGrad(gert::TilingContext* context);

ge::graphStatus TilingPrepare4AvgPool3DGrad(gert::TilingParseContext* context);

} // namespace optiling

#endif // OP_IMPL_AVG_POOL3_D_GRAD_TILING_BASE_H_
