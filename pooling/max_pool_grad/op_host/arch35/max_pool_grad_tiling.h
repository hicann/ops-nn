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
 * \file max_pool_grad_tiling.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_MAX_POOL_GRAD_WITH_AGRMAX_TILING_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_MAX_POOL_GRAD_WITH_AGRMAX_TILING_H_

#include "../../op_kernel/arch35/max_pool_grad_struct.h"
#include "../../../pool_grad_common/op_host/arch35/max_pool_grad_with_argmax_tiling_common.h"
#include "../../../pool_grad_common/op_kernel/arch35/max_pool_grad_with_argmax_struct_common.h"
#include "../../../pool_grad_common/op_host/arch35/max_pool_grad_nchw_tiling_common.h"
#include "../../../pool_grad_common/op_host/arch35/util.h"
#include "platform/platform_info.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "op_common/op_host/util/platform_util.h"
#include "register/op_def_registry.h"

namespace optiling {
using Ops::NN::Optiling::TilingBaseClass;
using namespace MaxPoolGradWithArgmaxNHWCNameSpace;

static constexpr int64_t KERNEL_OFFSET = 1;

constexpr int64_t BIG_FLOAT16_SIZE = 2;
constexpr int64_t BIG_FLOAT32_SIZE = 4;
constexpr int64_t BIG_INT32_SIZE = 4;
constexpr int64_t BIG_INT64_SIZE = 8;
constexpr int64_t BIG_UB_RESERVED_SIZE = 1024;
constexpr int64_t BIG_DOUBLE_BUFFER = 2;

constexpr int64_t BIG_KERNEL_THRESHOLD = 128;
constexpr int64_t BIG_MAX_KERNEL_COUNT = 512;
constexpr int64_t BIG_MERGE_BUF_ALIGN = 32;

class MaxPoolGradTilingBase : public MaxPoolGradWithArgmaxTilingCommon {
public:
    explicit MaxPoolGradTilingBase(gert::TilingContext* context) : MaxPoolGradWithArgmaxTilingCommon(context)
    {}
    ~MaxPoolGradTilingBase() override
    {}

    const std::string nodeName = "MaxPoolGrad";
    MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxSimtTilingCommonData* tilingData_ =
        context_->GetTilingData<MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxSimtTilingCommonData>();
    MaxPoolGradWithArgmaxInputInfoCommon inputData;
    int64_t coreNum_{0};
    int64_t ubSize_{0};

    bool CheckInputShape();
    ge::graphStatus CheckInputDtype();
    ge::graphStatus CheckAttrShape();
    ge::graphStatus CheckAttrVal();
    ge::graphStatus CheckInputValid();
    ge::graphStatus SetInputParams();
    ge::graphStatus SetAttrParams();
    void SetOtherInputParams();

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
};

class MaxPoolGradNCHWTilingHelper : public MaxPoolGradNCHWTilingCommon {
public:
    MaxPoolGradNCHWTilingHelper(MaxPoolGradWithArgmaxInputInfoCommon* input) : MaxPoolGradNCHWTilingCommon(input)
    {}

protected:
    void DoBufferCalculate() override;
};

class MaxPoolGradNCHWTiling : public MaxPoolGradTilingBase {
public:
    explicit MaxPoolGradNCHWTiling(gert::TilingContext* context) : MaxPoolGradTilingBase(context)
    {}

    ~MaxPoolGradNCHWTiling() override
    {}

private:
    MaxPoolGradNCHWTilingHelper commonTiling{&inputData};
    MaxPoolGradWithArgmaxHardwareInfo hwInfo;
    uint64_t GetTilingKey() const override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
};

} // namespace optiling

#endif