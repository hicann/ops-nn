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
 * \file fused_adamw_tiling.h
 * \brief
 */

#ifndef _OPS_BUILD_IN_OP_TILING_RUNTIME_FUSED_ADAMW_TILING_H_
#define _OPS_BUILD_IN_OP_TILING_RUNTIME_FUSED_ADAMW_TILING_H_

#include "register/tilingdata_base.h"
#include "op_host/tiling_base_util.h"
#include "../op_kernel/fused_adamw_tiling_data.h"

namespace optiling {

struct FusedAdamWCompileInfo {};

class FusedAdamWTiling {
public:
    explicit FusedAdamWTiling(gert::TilingContext* context) : context_(context) {};
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus GetAttrInfo();
    ge::graphStatus GetInputTensorInfo();
    ge::graphStatus CalculateOutputInfo();
    void CheckOptionalInputs();
    void SetTilingData(FusedAdamWTilingData* tilingData);
    std::string TilingDataToString() const;

private:
    gert::TilingContext* context_;
    uint32_t coreNum_{0};
    uint64_t ubSize_{0};
    uint64_t sysWorkspaceSize_{0};
    uint32_t usedCoreNum_{0};
    uint32_t usedRealCoreNum_{0};
    uint32_t lastCoreTensor_{0};
    float lr_{0.001f};
    float beta1_{0.9f};
    float beta2_{0.999f};
    float weightDecay_{0.0f};
    float eps_{1e-8f};
    uint32_t amsgrad_{0};
    uint32_t maximize_{0};
    uint32_t useGradScale_{0};
    uint32_t useFoundInf_{0};
    uint32_t tensorsPerCore_{0};
    uint32_t dtypeSize_{0};
    uint64_t tensorNum_{0};
    uint64_t coreCalcMax_{0};
};
} // namespace optiling
#endif // _OPS_BUILD_IN_OP_TILING_RUNTIME_FUSED_ADAMW_TILING_H_
