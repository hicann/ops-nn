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
 * \file adaptive_avg_pool3d_gather_trans_tiling.h
 * \brief
 */

#ifndef ADAPTIVE_AVG_POOL3D_GATHER_TRANS_TILING_H
#define ADAPTIVE_AVG_POOL3D_GATHER_TRANS_TILING_H

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

struct GatherTransInfo {
    uint64_t vfLen{0};
    uint64_t availableUbSize{0};
    uint64_t spatialIn{0};
    uint64_t outDHW{0};
    uint64_t inHW{0};
    uint64_t outHW{0};
    uint64_t kernelDMax{0};
    uint64_t kernelHMax{0};
    uint64_t kernelWMax{0};
    uint64_t ncFactor{0};
    uint64_t ncOuter{0};
    uint64_t ncTail{0};
    uint64_t ncBatch{0};
    uint64_t doFactor{0};
    uint64_t doOuter{0};
    uint64_t doTail{0};
    uint64_t maxDInBlock{0};
    uint64_t maxDoBlock{0};
    uint64_t tileNum{0};
    uint64_t blockFactor{0};
    uint64_t blockTail{0};
    uint64_t useCoreNum{0};
};

class AdaptiveAvgPool3dGatherTransTiling : public AdaptivePool3dBaseTiling {
public:
    explicit AdaptiveAvgPool3dGatherTransTiling(gert::TilingContext* context) : AdaptivePool3dBaseTiling(context) {}
    ~AdaptiveAvgPool3dGatherTransTiling() override {}

    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus PostTiling() override;

private:
    uint64_t CalOccupySize(uint64_t ncBatch) const;
    void CalMaxDInBlock(uint64_t doFactor);
    void SetDoBlockInfo(uint64_t doFactor);
    bool IsIndexInRange() const;
    void CalDoFactor();
    void CalBlockFactor();
    void SetTilingData();
    void PrintTilingData() const;

    GatherTransInfo gtInfo_;
};

} // namespace optiling
#endif // ADAPTIVE_AVG_POOL3D_GATHER_TRANS_TILING_H
