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
 * \file max_pool_grad_with_argmax_v3_tiling_base.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_MAX_POOL_GRAD_WITH_AGRMAX_V3_TILING_BASE_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_MAX_POOL_GRAD_WITH_AGRMAX_V3_TILING_BASE_H_

#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling_base/tiling_base.h"
#include "util/math_util.h"
#include "op_common/op_host/util/platform_util.h"

using namespace std;

namespace optiling {

BEGIN_TILING_DATA_DEF(MaxPoolGradWithArgmaxV3TilingData)
TILING_DATA_FIELD_DEF(uint64_t, nc);
TILING_DATA_FIELD_DEF(uint64_t, hx);
TILING_DATA_FIELD_DEF(uint64_t, wx);
TILING_DATA_FIELD_DEF(uint64_t, kh);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MaxPoolGradWithArgmaxV3, MaxPoolGradWithArgmaxV3TilingData);
struct MaxPoolGradWithArgmaxV3InputInfo {
    int64_t hPad{0};
    int64_t wPad{0};
    int64_t hStride{1};
    int64_t wStride{1};
    int64_t hKernel{1};
    int64_t wKernel{1};
    int64_t hDilation{1};
    int64_t wDilation{1};
    int64_t nX{1};
    int64_t cX{1};
    int64_t hX{1};
    int64_t wX{1};
    int64_t nGrad{1};
    int64_t cGrad{1};
    int64_t hGrad{1};
    int64_t wGrad{1};
    bool ceilMode{false};
    int64_t gradShapeSize{0};
    ge::DataType inputDtype{ge::DataType::DT_FLOAT};
    ge::DataType indexDtype{ge::DataType::DT_INT32};
    ge::Format inputFormat{ge::Format::FORMAT_NCHW};
    int64_t isInt32Meet{1};
};

struct MaxPoolGradWithArgmaxV3HardwareInfo {
    int64_t coreNum{0};
    int64_t ubSize{0};
};

struct MaxPoolGradWithArgmaxV3CompileInfo {
    uint64_t coreNum;
    uint64_t ubSize;
};

class MaxPoolGradWithArgmaxV3BaseTiling : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit MaxPoolGradWithArgmaxV3BaseTiling(gert::TilingContext* context) : TilingBaseClass(context)
    {}

    ~MaxPoolGradWithArgmaxV3BaseTiling() override
    {}

protected:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    void PrintInputData() const;

public:
    MaxPoolGradWithArgmaxV3InputInfo inputData;
    MaxPoolGradWithArgmaxV3HardwareInfo hardwareData;
};
} // namespace optiling
#endif