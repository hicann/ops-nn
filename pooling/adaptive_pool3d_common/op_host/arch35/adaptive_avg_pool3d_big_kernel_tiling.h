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
 * \file adaptive_avg_pool3d_big_kernel_tiling.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_ADAPTIVE_AVG_POOL3D_TILING_BIG_KERNEL_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_ADAPTIVE_AVG_POOL3D_TILING_BIG_KERNEL_H_

#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "adaptive_pool3d_tiling.h"
#include "../op_kernel/arch35/adaptive_pool3d_tiling_struct.h"

using namespace std;
using namespace AdaptivePool3DTiling;

namespace optiling {

struct AdaptiveAvgPool3dBigKernelInfo {
    int64_t blockFactor {0};
    int64_t blockTail {0};
    int64_t totalIdx {0};
    int64_t coreNums {0};
    int64_t maxCount {0};
    int64_t kernelMaxDHW {0};
    int64_t batchCount {1};
};

class AdaptiveAvgPool3dBigKernelTiling : public AdaptivePool3dBaseTiling {
public:
    explicit AdaptiveAvgPool3dBigKernelTiling(gert::TilingContext* context) : AdaptivePool3dBaseTiling(context) {}

private:
    ge::graphStatus CheckOutputDtypeInfo();
    void DoBlockTiling();
    void SetTilingData();
    int64_t CalKernelSize(int64_t inSize, int64_t outSize);
    void PrintTilingData() const;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;
    
    AdaptiveAvgPool3dBigKernelInfo avgBigKernelInfo;
};
} // namespace optiling
#endif