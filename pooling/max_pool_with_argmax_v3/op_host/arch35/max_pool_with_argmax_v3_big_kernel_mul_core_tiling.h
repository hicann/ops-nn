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
 * \file max_pool_with_argmax_v3_big_kernel_mul_core_tiling.h
 * \brief big kernel imply for max_pool_with_argmax
 */

#ifndef CANN_MAX_POOL_WITH_ARGMAX_V3_BIG_KERNEL_MUL_CORE_TILING_H
#define CANN_MAX_POOL_WITH_ARGMAX_V3_BIG_KERNEL_MUL_CORE_TILING_H

#include "max_pool_with_argmax_v3_tiling.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(MaxPoolWithArgmaxV3BigKernelMulCoreTilingData)
TILING_DATA_FIELD_DEF(int64_t, hInDim);
TILING_DATA_FIELD_DEF(int64_t, wInDim);
TILING_DATA_FIELD_DEF(int64_t, hOutDim);
TILING_DATA_FIELD_DEF(int64_t, wOutDim);
TILING_DATA_FIELD_DEF(int64_t, kW);
TILING_DATA_FIELD_DEF(int64_t, kH);
TILING_DATA_FIELD_DEF(int64_t, sW);
TILING_DATA_FIELD_DEF(int64_t, sH);
TILING_DATA_FIELD_DEF(int64_t, pW);
TILING_DATA_FIELD_DEF(int64_t, pH);
TILING_DATA_FIELD_DEF(int64_t, dW);
TILING_DATA_FIELD_DEF(int64_t, dH);
TILING_DATA_FIELD_DEF(int64_t, coreNums);
TILING_DATA_FIELD_DEF(int64_t, multiCoreNum);
TILING_DATA_FIELD_DEF(int64_t, kernelBlockFactorH);
TILING_DATA_FIELD_DEF(int64_t, tailKernelBlockFactorH);
TILING_DATA_FIELD_DEF(int64_t, splitW);
TILING_DATA_FIELD_DEF(int64_t, wSplitSize);
TILING_DATA_FIELD_DEF(int64_t, tailWSplitSize);
TILING_DATA_FIELD_DEF(int64_t, splitSlice);
TILING_DATA_FIELD_DEF(int64_t, maxCountLength);
TILING_DATA_FIELD_DEF(int64_t, valueBufferLength);
TILING_DATA_FIELD_DEF(int64_t, indexBufferLength);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MaxPoolWithArgmaxV3_400001, MaxPoolWithArgmaxV3BigKernelMulCoreTilingData);
REGISTER_TILING_DATA_CLASS(MaxPoolWithArgmaxV3_400002, MaxPoolWithArgmaxV3BigKernelMulCoreTilingData);
REGISTER_TILING_DATA_CLASS(MaxPoolWithArgmaxV3_400003, MaxPoolWithArgmaxV3BigKernelMulCoreTilingData);
REGISTER_TILING_DATA_CLASS(MaxPoolWithArgmaxV3_400004, MaxPoolWithArgmaxV3BigKernelMulCoreTilingData);
REGISTER_TILING_DATA_CLASS(MaxPoolWithArgmaxV3_400005, MaxPoolWithArgmaxV3BigKernelMulCoreTilingData);
REGISTER_TILING_DATA_CLASS(MaxPoolWithArgmaxV3_400006, MaxPoolWithArgmaxV3BigKernelMulCoreTilingData);

class MaxPoolWithArgmaxV3BigKernelMulCoreTiling : public MaxPoolWithArgmaxV3BaseTiling {
public:
    explicit MaxPoolWithArgmaxV3BigKernelMulCoreTiling(gert::TilingContext* context)
        : MaxPoolWithArgmaxV3BaseTiling(context)
    {}
    ~MaxPoolWithArgmaxV3BigKernelMulCoreTiling() override
    {}

private:
    void DoUBTiling();
    void SetTilingData();
    uint64_t GetTilingKey() const;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    void DumpTilingInfo() override;
    MaxPoolWithArgmaxV3BigKernelMulCoreTilingData tiling;
    uint32_t totalIdx{0};
    uint32_t coreNums{0};
    int64_t multiCoreNum{0};
    uint64_t kernelBlockFactorH{0};
    int64_t tailKernelBlockFactorH{0};
    int64_t splitW{0};
    uint64_t wSplitSize{0};
    int64_t tailWSplitSize{0};
    uint64_t splitSlice{0};
    int64_t maxCountLength{0};
};

} // namespace optiling
#endif // CANN_MAX_POOL_WITH_ARGMAX_V3_BIG_KERNEL_MUL_CORE_TILING_H