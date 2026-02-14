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
 * \file max_pool_grad_with_argmax_tiling_common.h
 * \brief
 */

#ifndef MAX_POOL_GRAD_WITH_AGRMAX_TILING_COMMON_H_
#define MAX_POOL_GRAD_WITH_AGRMAX_TILING_COMMON_H_

#include "log/log.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling_base/tiling_base.h"
#include "tiling_base/tiling_templates_registry.h"
#include "tiling_base/tiling_util.h"
#include "tiling/tiling_api.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "error_util.h"
#include "platform/platform_infos_def.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_info_def.h"

namespace optiling {
using Ops::NN::Optiling::TilingBaseClass;
static const gert::Shape g_vec_1_shape = {1};
static constexpr size_t WS_SYS_SIZE = static_cast<size_t>(16 * 1024 * 1024);

struct MaxPoolGradWithArgmaxInputInfoCommon {
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
    int64_t isInt32Meet{1};
    ge::Format inputFormat{ge::Format::FORMAT_NHWC};
};

inline static const gert::Shape &EnsureNotScalar(const gert::Shape &inShape) {
 if (inShape.IsScalar()) {
   return g_vec_1_shape;
 }
 return inShape;
}

struct MaxPoolGradWithArgmaxHardwareInfo {
    int64_t coreNum{0};
    int64_t ubSize{0};
};

struct MaxPoolGradWithArgmaxCompileInfo {
    uint64_t coreNum;
    uint64_t ubSize;
};

class MaxPoolGradWithArgmaxTilingCommon : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit MaxPoolGradWithArgmaxTilingCommon(gert::TilingContext* context) : TilingBaseClass(context) {
    }

    ~MaxPoolGradWithArgmaxTilingCommon() override {
    }

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;
    void PrintInputData() const;

public : 
    MaxPoolGradWithArgmaxInputInfoCommon inputData;
    MaxPoolGradWithArgmaxHardwareInfo hardwareData;
};
}  // namespace optiling
#endif