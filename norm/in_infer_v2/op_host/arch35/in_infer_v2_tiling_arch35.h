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
 * \file in_infer_v2_tiling_arch35.h
 * \brief INInferV2 arch35 tiling（ND-only；GE 可能下发 NCHW 标签）
 */

#ifndef IN_INFER_V2_TILING_ARCH35_H
#define IN_INFER_V2_TILING_ARCH35_H

#include <cstdint>
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../../op_kernel/arch35/in_infer_v2_tiling_data.h"

namespace optiling {

struct INInferV2CompileInfo {
    int64_t coreNum = 0;
    int64_t ubSize = 0;
};

class INInferV2Tiling {
public:
    explicit INInferV2Tiling(gert::TilingContext* context) : context_(context) {}
    ge::graphStatus DoTiling();

private:
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus GetShapeAndDtype();
    ge::graphStatus CalcCoreSplit();
    ge::graphStatus FillTilingData();

    gert::TilingContext* context_ = nullptr;
    int64_t coreNum_ = 0;
    int64_t ubSize_ = 0;

    // shape 信息
    int64_t numN_ = 0;
    int64_t numC_ = 0;      // C（dim1）
    int64_t innerSize_ = 0; // R = prod(d2:)
    int64_t units_ = 0;
    int64_t xDtypeSize_ = 4;
    float epsilon_ = 1e-5f;
    int64_t hasGammaBeta_ = 0;
    int64_t hasBatchMean_ = 0;
    int64_t hasBatchVar_ = 0;

    // 切分结果
    int64_t unitCores_ = 1;
    int64_t formerCoreNum_ = 0;
    int64_t formerUnits_ = 0;
    int64_t latterUnits_ = 0;
    int64_t innerCores_ = 1;
    int64_t innerPerCore_ = 0;
    int64_t ubTileSize_ = 0;
};

} // namespace optiling

#endif // IN_INFER_V2_TILING_ARCH35_H
