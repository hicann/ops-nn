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
 * \file bn_training_update_grad_tiling_arch35.h
 * \brief BNTrainingUpdateGrad arch35 tiling（ND-only；GE 可能下发 NCHW 标签）
 */

#ifndef BN_TRAINING_UPDATE_GRAD_TILING_ARCH35_H
#define BN_TRAINING_UPDATE_GRAD_TILING_ARCH35_H

#include <cstdint>
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../../op_kernel/arch35/bn_training_update_grad_tiling_data.h"

namespace optiling {

struct BNTrainingUpdateGradCompileInfo {
    int64_t coreNum = 0;
    int64_t ubSize = 0;
};

class BNTrainingUpdateGradTiling {
public:
    explicit BNTrainingUpdateGradTiling(gert::TilingContext* context) : context_(context) {}
    ge::graphStatus DoTiling();

private:
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus GetShapeAndDtype();
    ge::graphStatus CheckGradsXDescAndShape();
    ge::graphStatus CheckStatInputs();
    ge::graphStatus CalcCoreSplit();
    ge::graphStatus FillTilingData();

    gert::TilingContext* context_ = nullptr;
    int64_t coreNum_ = 0;
    int64_t ubSize_ = 0;

    // shape 信息
    int64_t numN_ = 0;
    int64_t numC_ = 0;      // C（dim1）
    int64_t innerSize_ = 0; // R = prod(d2:)
    int64_t xDtypeSize_ = 4;
    float epsilon_ = 0.0001f;

    // 切分结果
    int64_t channelCores_ = 1;
    int64_t cFormerCoreNum_ = 0;
    int64_t cFormerLen_ = 0;
    int64_t cLatterLen_ = 0;
    int64_t cLenCap_ = 1;
    int64_t sliceR_ = 1;
    int64_t rowsPerTile_ = 1;
};

} // namespace optiling

#endif // BN_TRAINING_UPDATE_GRAD_TILING_ARCH35_H
