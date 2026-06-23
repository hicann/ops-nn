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
 * \file fused_sgd_tiling.h
 * \brief
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_FUSED_SGD_TILING_H_
#define OPS_BUILD_IN_OP_TILING_RUNTIME_FUSED_SGD_TILING_H_


#include "register/tilingdata_base.h"
#include "op_host/tiling_base_util.h"
#include "../op_kernel/fused_sgd_tiling_data.h"

namespace optiling {

struct FusedSgdCompileInfo {
};

class FusedSgdTiling {
public:
    explicit FusedSgdTiling(gert::TilingContext* context) : context_(context) {};
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus GetAttrInfo();
    ge::graphStatus GetInputTensorInfo();
    ge::graphStatus CalculateOutputInfo();
    void CheckOptionalInputs();
    void SetTilingData(FusedSgdTilingData* tilingData);
    std::string TilingDataToString() const;

private:
    gert::TilingContext* context_;
    uint32_t coreNum_{0};
    uint64_t ubSize_{0};
    uint64_t sysWorkspaceSize_{0};
    uint32_t usedCoreNum_{0};
    float weightDecay_{0.0f};
    float momentum_{0.0f};
    float lr_{0.0f};
    float dampening_{0.0f};
    uint32_t nesterov_{0};
    uint32_t maximize_{0};
    uint32_t isFirstStep_{0};
    uint32_t useGradScale_{0};
    uint32_t useMomentum_{0};
    uint32_t tensorsPerCore_{0};
    uint32_t dtypeSize_{0};
    uint64_t tensorNum_{0};
    uint64_t coreCalcMax_{0};
};
} // namespace optiling
#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_FUSED_SGD_TILING_H_