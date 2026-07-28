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
 * \file instance_norm_grad_empty_tiling_arch35.h
 * \brief Empty-tensor (some axis == 0) tiling for InstanceNormGrad (tilingKey 500).
 */
#pragma once

#include "instance_norm_grad_tiling.h"

namespace optiling {
class InstanceNormGradEmptyTiling : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit InstanceNormGradEmptyTiling(gert::TilingContext* context) : TilingBaseClass(context) {}

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

private:
    void CalcUsedCoreNumGamma();
    ge::graphStatus CalcuTilingData();
    uint64_t NearestLowerPowerOfTwo(int32_t tmp);
    void SetTilingData();

private:
    const char* opName = "InstanceNormGrad";
    InstanceNormGradEmptyTilingData tilingData;
    uint32_t aivCoreNum_ = 0;
    uint64_t cols_ = 0;
    uint64_t usedCoreNumDG_ = 0;
    uint64_t colsPerCoreDG_ = 0;
    uint64_t colsPerUBDG_ = 0;
    uint64_t tailUbCols_ = 0;
    uint64_t lastCoreBlockCount_ = 0;
    uint64_t lastCoreTailUbCols_ = 0;
    uint64_t coreUbBlockCount_ = 0;
    uint64_t colsLastCoreDG_ = 0;
    uint64_t ubSize_ = 0;
    uint32_t sysWorkspaceSize_ = 0;
    int64_t workSpaceSize_ = 0;
};
} // namespace optiling
