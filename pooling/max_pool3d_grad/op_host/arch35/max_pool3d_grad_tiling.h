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
 * \file max_pool3d_grad_tiling.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_MAX_POOL3D_GRAD_TILING_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_MAX_POOL3D_GRAD_TILING_H_

#include "../../../pool_grad_common/op_host/arch35/pool3d_grad_ncdhw_small_kernel_tiling.h"
#include "../../../pool_grad_common/op_kernel/arch35/pool3d_grad_struct_common.h"
#include "../../../pool_grad_common/op_host/arch35/util.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_base.h"
#include "util/math_util.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "op_common/op_host/util/platform_util.h"

namespace optiling {
using Ops::NN::Optiling::TilingBaseClass;
using namespace Pool3DGradNameSpace;

class MaxPool3DGradTilingBase : public TilingBaseClass {
public:
    explicit MaxPool3DGradTilingBase(gert::TilingContext* context) : TilingBaseClass(context)
    {}
    ~MaxPool3DGradTilingBase() override
    {}

    const std::string nodeName = "MaxPool3DGrad";
    Pool3DGradNCDHWTilingData* tilingData_ = context_->GetTilingData<Pool3DGradNCDHWTilingData>();
    Pool3DGradNCDHWInputInfo inputData;
    int64_t coreNum_{0};
    int64_t ubSize_{0};

    bool CheckInputShape();
    ge::graphStatus CheckInputDtype();
    ge::graphStatus CheckAttrShape();
    ge::graphStatus CheckAttrVal();
    ge::graphStatus CheckInputValid();
    ge::graphStatus SetInputParams();
    ge::graphStatus SetAttrParams();
    void SetCntTailTilingParams();
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

class MaxPool3DGradNCDHWSmallKernelTiling : public MaxPool3DGradTilingBase {
public:
    explicit MaxPool3DGradNCDHWSmallKernelTiling(gert::TilingContext* context)
        : MaxPool3DGradTilingBase(context),
        base(new Pool3DGradNCDHWSmallKernelCommonTiling(&inputData))
    {

    }

    ~MaxPool3DGradNCDHWSmallKernelTiling()
    {
        delete base;
    }
private:
    Pool3DGradNCDHWSmallKernelCommonTiling* base;
    uint64_t GetTilingKey() const override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
};

const int64_t NCDHW_DIMS = 5;
const int64_t DHW_DIMS = 3;
const int64_t KERNEL_POS = 0;
const int64_t STRIDE_POS = 1;
const int64_t PADDING_MODE_POS = 2;
const int64_t PADDING_POS = 3;
const int64_t FORMAT_POS = 4;
const int64_t N_DIM_ = 0;
const int64_t C_DIM_ = 1;
const int64_t D_DIM_ = 2;
const int64_t H_DIM_ = 3;
const int64_t W_DIM_ = 4;
const int64_t D_IDX_ = 0;
const int64_t H_IDX_ = 1;
const int64_t W_IDX_ = 2;
const int64_t DOUB = 2;
const int64_t FIRPOS = 0;
const int64_t SECPOS = 1;
const int64_t THIPOS = 2;
constexpr uint64_t SIMT_NCDHW_TILING_KEY_INT32 = 600001;
constexpr uint64_t SIMT_NDHWC_TILING_KEY_INT32 = 600002;
constexpr uint64_t SIMT_NCDHW_TILING_KEY_INT64 = 600011;
constexpr uint64_t SIMT_NDHWC_TILING_KEY_INT64 = 600012;
constexpr int64_t MAX_THREAD_NUM = 256;
constexpr size_t WS_SYS_SIZE = 16 * 1024 * 1024;

struct InputSIMTInfo {
    std::array<uint64_t, NCDHW_DIMS> inputShape;
    std::array<uint64_t, NCDHW_DIMS> gradShape;
    std::array<uint64_t, NCDHW_DIMS> outShape;
    std::array<uint64_t, DHW_DIMS> kernelSize;
    std::array<uint64_t, DHW_DIMS> stride;
    std::array<uint64_t, DHW_DIMS> pad;
    std::array<uint64_t, DHW_DIMS> dilation;
    bool ceilMode;
    std::string data_format;
};

class MaxPool3DGradSimtTiling : public Ops::NN::Optiling::TilingBaseClass
{
public:
    explicit MaxPool3DGradSimtTiling(gert::TilingContext* context) : TilingBaseClass(context)
    {
    }

    ~MaxPool3DGradSimtTiling() override
    {
    }

protected:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override;

private:
    MaxPool3DGradSimtTilingData* tilingData_ = 
        context_->GetTilingData<MaxPool3DGradSimtTilingData>();
    InputSIMTInfo inputData;
    int nDimPos = 0;
    int cDimPos = 1;
    int dDimPos = 2;
    int hDimPos = 3;
    int wDimPos = 4;
    int64_t outShapeSize = 0;
    int64_t coreNum = 0;
    int64_t ubSize = 0;
    ge::DataType dtype = ge::DataType::DT_FLOAT;
};
}  // namespace optiling

#endif