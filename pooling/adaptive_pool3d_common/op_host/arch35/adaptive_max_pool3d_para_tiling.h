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
 * \file adaptive_max_pool3d_para_tiling.h
 * \brief
 */

#ifndef ADAPATIVE_MAX_POOL3D_PARA_TILING_H
#define ADAPATIVE_MAX_POOL3D_PARA_TILING_H

#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling_base/tiling_base.h"
#include "util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "adaptive_pool3d_tiling.h"
#include "../op_kernel/arch35/adaptive_pool3d_tiling_struct.h"

using namespace std;
using namespace AdaptivePool3DTiling;

namespace optiling {
using Ops::NN::Optiling::TilingBaseClass;

struct ComputeInfo {
    uint64_t useCoreNum{0};
    uint64_t totalOuter{0};
    uint64_t blockFactor{0};
    uint64_t blockTail{0};
    uint64_t ncFactor{0};
    uint64_t doFactor{0};
    uint64_t hoFactor{0};
    uint64_t woFactor{0};
    uint64_t ncOuter{0};
    uint64_t doOuter{0};
    uint64_t hoOuter{0};
    uint64_t woOuter{0};
    uint64_t ncTail{0};
    uint64_t doTail{0};
    uint64_t hoTail{0};
    uint64_t woTail{0};
    uint64_t kernelDMax{0};
    uint64_t kernelHMax{0};
    uint64_t kernelWMax{0};
    uint64_t vfLen{0};
    uint64_t alignNum{0};
    uint64_t availableUbSize{0};
    uint64_t maxInputSize{0};
};

class AdaptiveMaxPool3dParaPoolTiling : public AdaptivePool3dBaseTiling
{
public:
    explicit AdaptiveMaxPool3dParaPoolTiling(gert::TilingContext* context) : AdaptivePool3dBaseTiling(context)
    {}
    ~AdaptiveMaxPool3dParaPoolTiling() override
    {}
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus PostTiling() override;

private:
    uint64_t CalKernelMax(uint64_t inputSize, uint64_t outputSize);
    uint64_t CalOccupySize();
    void CalMaxUbSplitSize();
    void CalUbBlockFactor();
    void BinarySearch(uint64_t& initFactor);
    void SearchOuterSingle(uint64_t& initFactor);
    ge::graphStatus InitUbFactor();
    ge::graphStatus SearchUbFactor();
    ge::graphStatus SearchOuter();
    void SetTilingData();
    void PrintTilingData() const;
    ComputeInfo computeInfo_;
};


} // namespace optiling
#endif // ADAPTIVE_MAX_POOL3D_PARA_TILING_H