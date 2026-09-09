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
 * \file adaptive_avg_pool3d_para_tiling.h
 * \brief
 */

#ifndef ADAPTIVE_AVG_POOL3D_PARA_TILING_H
#define ADAPTIVE_AVG_POOL3D_PARA_TILING_H

#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "adaptive_pool3d_tiling.h"
#include "../op_kernel/arch35/adaptive_pool3d_tiling_struct.h"

namespace optiling {
using namespace std;
using namespace AdaptivePool3DTiling;
using Ops::NN::Optiling::TilingBaseClass;

struct AdaptiveAvgPool3dParaComputeInfo : public AdaptivePool3dComputeInfo {
    uint64_t maxDimOut{0};
};

class AdaptiveAvgPool3dParaPoolTiling : public AdaptivePool3dBaseTiling {
public:
    explicit AdaptiveAvgPool3dParaPoolTiling(gert::TilingContext* context) : AdaptivePool3dBaseTiling(context) {}
    ~AdaptiveAvgPool3dParaPoolTiling() override {}
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus PostTiling() override;

private:
    uint64_t CalOccupySize();
    void CalMaxUbSplitSize();
    void CalUbBlockFactor();
    void BinarySearch(uint64_t& initFactor);
    void SearchOuterSingle(uint64_t& initFactor);
    ge::graphStatus InitUbFactor();
    ge::graphStatus SearchUbFactor();
    ge::graphStatus SearchOuter();
    ge::graphStatus DoTilingForUbFactor();
    void SetTilingData();
    void PrintTilingData() const;
    AdaptiveAvgPool3dParaComputeInfo avgComputeInfo_;
};

} // namespace optiling
#endif // ADAPTIVE_AVG_POOL3D_PARA_TILING_H
