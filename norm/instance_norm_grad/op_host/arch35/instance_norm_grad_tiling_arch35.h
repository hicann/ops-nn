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
 * \file instance_norm_grad_tiling_arch35.h
 * \brief RegBase (arch35) tiling class for InstanceNormGrad.
 */
#pragma once

#include "instance_norm_grad_tiling.h"
using namespace Ops::NN::Optiling;

namespace optiling {
class InstanceNormGradRegBaseTiling : public TilingBaseClass {
public:
    explicit InstanceNormGradRegBaseTiling(gert::TilingContext* context) : TilingBaseClass(context) {}

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
    ge::graphStatus InputCheck();
    ge::graphStatus ParamsCheck();
    ge::graphStatus CheckTensorDtype(const gert::CompileTimeTensorDesc* desc, const char* name) const;
    ge::graphStatus BlockTiling();
    ge::graphStatus UbTiling();
    void SetTilingData();
    void PrintTilingData() const;
    uint32_t GetTypeSize(ge::DataType dtypeStr) const;

private:
    static constexpr uint32_t DOUBLE_BUFFER = 2;
    static constexpr uint32_t UB_COPIES_3 = 3;      // x, dy, pd_x flowing buffers
    static constexpr uint32_t PARAM_BUFFERS = 8;    // var, mean, gamma, rstd, pdVar, pdMean, accDgamma, accDbeta
    static constexpr uint32_t WORKSPACE_COPIES = 2; // dgamma + dbeta partial sums
    static constexpr uint32_t FLOAT_DTYPE_BYTES = 4;
    static constexpr uint32_t FLOAT16_DTYPE_BYTES = 2;
    static constexpr int64_t MODE_FULL_LOAD = 100;
    static constexpr int64_t MODE_RECOMPUTE = 300;
    static constexpr uint32_t MIN_BLOCK_SIZE = 512;

    const char* opName = "InstanceNormGrad";
    InstanceNormGradTilingData tilingData;

    uint64_t ubSize_ = 0;
    uint32_t coreNum_ = 0;
    uint32_t sysWorkspaceSize_ = 0;
    uint32_t blockSize_ = 0;
    uint32_t vectorLen_ = 0; // fp32 lanes per VL

    ge::DataType dtype_ = ge::DT_UNDEFINED;
    uint32_t tTypeBytes_ = 0;

    int64_t N_ = 0;
    int64_t C_ = 0;
    int64_t M_ = 1;
    int64_t cTile_ = 0;
    int64_t cTileNum_ = 1;
    int64_t taskNum_ = 0;
    uint32_t taskNumPerCore_ = 0;
    uint32_t taskNumPerTailCore_ = 0;
    uint32_t tailCore_ = 0;
    uint32_t stage1CoreUsed_ = 0;
    uint32_t modeKey_ = MODE_FULL_LOAD;
    uint32_t mUbTile_ = 0;
    uint32_t mUbIterNum_ = 1;
    uint32_t mUbTailNum_ = 0;

    int64_t reduceNCnt_ = 0;
    int64_t workSpaceSize_ = 0;
    uint32_t stage2CoreUsed_ = 0;
    int64_t cBlockFactor_ = 0;
    int64_t cTailBlockFactor_ = 0;
};
} // namespace optiling
