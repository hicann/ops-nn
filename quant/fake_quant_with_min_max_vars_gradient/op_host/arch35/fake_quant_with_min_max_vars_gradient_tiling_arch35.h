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
 * \file fake_quant_with_min_max_vars_gradient_tiling_arch35.h
 * \brief Host-side tiling for fake_quant_with_min_max_vars_gradient (arch35)
 *
 * Key constraint: min/max are tensor inputs, NOT attributes.
 * Tiling does NOT read or precompute any tensor values.
 * nudgedMin/nudgedMax are computed on Device side in the Kernel.
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_FAKE_QUANT_WITH_MIN_MAX_VARS_GRADIENT_TILING_ARCH35_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_FAKE_QUANT_WITH_MIN_MAX_VARS_GRADIENT_TILING_ARCH35_H_

#include <cstdint>
#include "graph/types.h"
#include "register/tilingdata_base.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "../../op_kernel/arch35/fake_quant_with_min_max_vars_gradient_tilingdata.h"

namespace optiling {

struct FakeQuantWithMinMaxVarsGradientCompileInfo {
    int32_t vectorCoreNum = 0;
    uint64_t ubSize = 0;
};

class FakeQuantWithMinMaxVarsGradientTiling {
public:
    explicit FakeQuantWithMinMaxVarsGradientTiling(gert::TilingContext* context) : context_(context) {}
    ge::graphStatus DoTiling();

protected:
    ge::graphStatus GetCompileInfo();
    ge::graphStatus GetOpParam();
    ge::graphStatus CalcTiling();
    ge::graphStatus CalcWorkspace();
    ge::graphStatus WriteTilingData();

private:
    gert::TilingContext* context_ = nullptr;
    FakeQuantWithMinMaxVarsGradientTilingData tilingData_{};

    int64_t coreNum_ = 0;
    uint64_t ubSize_ = 0;

    int64_t numBits_ = 8;
    bool narrowRange_ = false;
    int64_t totalLen_ = 0;

    int64_t numCore_ = 0;
    int64_t blockFactor_ = 0;
    int64_t blockTailFactor_ = 0;
    int64_t baseLen_ = 0;
    uint64_t tilingKey_ = 0;
};

} // namespace optiling
#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_FAKE_QUANT_WITH_MIN_MAX_VARS_GRADIENT_TILING_ARCH35_H_
