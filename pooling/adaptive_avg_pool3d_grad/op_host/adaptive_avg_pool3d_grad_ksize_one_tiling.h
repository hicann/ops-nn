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
 * \file adaptive_avg_pool3d_grad_ksize_one_tiling.h
 * \brief
 */

#ifndef ADAPTIVE_AVG_POOL3D_GRAD_KSIZE_ONE_TILING_H_
#define ADAPTIVE_AVG_POOL3D_GRAD_KSIZE_ONE_TILING_H_

#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "op_common/op_host/util/platform_util.h"
#include "adaptive_avg_pool3d_grad_tiling_arch35.h"
#include "../op_kernel/arch35/adaptive_avg_pool3d_grad_struct.h"

namespace optiling {

class AdaptiveAvgPool3dGradTilingKsizeOne : public AdaptiveAvgPool3dGradTilingBaseV35 {
public:
    explicit AdaptiveAvgPool3dGradTilingKsizeOne(gert::TilingContext* context)
        : AdaptiveAvgPool3dGradTilingBaseV35(context)
    {}

    ~AdaptiveAvgPool3dGradTilingKsizeOne() override {}

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;

    void DoUBTiling();

    int64_t usedCoreNum_ = 0;
    int64_t blockFactor_ = 0;
    int64_t tailBlockFactor_ = 0;
    int64_t coreLoop_ = 0;
    int64_t tailCoreLoop_ = 0;
    int64_t ubFactor_ = 0;
    int64_t tailUbFactor_ = 0;
    int64_t tailCoreTailUbFactor_ = 0;
};

} // namespace optiling

#endif // ADAPTIVE_AVG_POOL3D_GRAD_KSIZE_ONE_TILING_H_
