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
 * \file max_pool_v3_big_kernel_tiling.h
 * \brief big kernel tiling for max_pool_v3
 */

#ifndef CANN_MAX_POOL_V3_BIG_KERNEL_TILING_H
#define CANN_MAX_POOL_V3_BIG_KERNEL_TILING_H

#include "max_pool_v3_tiling.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(MaxPoolV3BigKernelTilingData)
TILING_DATA_FIELD_DEF(int64_t, hInDim);
TILING_DATA_FIELD_DEF(int64_t, wInDim);
TILING_DATA_FIELD_DEF(int64_t, hOutDim);
TILING_DATA_FIELD_DEF(int64_t, wOutDim);
TILING_DATA_FIELD_DEF(int64_t, kW);
TILING_DATA_FIELD_DEF(int64_t, kH);
TILING_DATA_FIELD_DEF(int64_t, sW);
TILING_DATA_FIELD_DEF(int64_t, sH);
TILING_DATA_FIELD_DEF(int64_t, tPad);
TILING_DATA_FIELD_DEF(int64_t, bPad);
TILING_DATA_FIELD_DEF(int64_t, lPad);
TILING_DATA_FIELD_DEF(int64_t, rPad);
TILING_DATA_FIELD_DEF(int64_t, blockFactor);
TILING_DATA_FIELD_DEF(int64_t, blockTail);
TILING_DATA_FIELD_DEF(int64_t, totalIdx);
TILING_DATA_FIELD_DEF(int64_t, coreNums);
TILING_DATA_FIELD_DEF(int64_t, maxCount);
TILING_DATA_FIELD_DEF(int64_t, isSigOut);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MaxPoolV3_311110, MaxPoolV3BigKernelTilingData);

class MaxPoolV3BigKernelTiling : public MaxPoolV3BaseTiling {
public:
    explicit MaxPoolV3BigKernelTiling(gert::TilingContext* context) : MaxPoolV3BaseTiling(context)
    {}
    ~MaxPoolV3BigKernelTiling() override
    {}

private:
    void DoUBTiling();
    void SetTilingData();
    uint64_t GetTilingKey() const override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override;
    MaxPoolV3BigKernelTilingData tiling;
    int64_t totalIdx_{0};
    int64_t blockFactor_{0};
    int64_t blockTail_{0};
    int64_t maxCount_{0};
    int64_t isSigOut_{0};
    int64_t coreNums_{0};
};

} // namespace optiling
#endif // CANN_MAX_POOL_V3_BIG_KERNEL_TILING_H