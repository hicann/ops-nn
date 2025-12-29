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
 * \file max_pool_with_argmax_v3_simt_tiling.h
 * \brief simt imply for max_pool_with_argmax
 */

#ifndef CANN_MAX_POOL_WITH_ARGMAX_V3_SIMT_TILING_H
#define CANN_MAX_POOL_WITH_ARGMAX_V3_SIMT_TILING_H

#include "max_pool_with_argmax_v3_tiling.h"

namespace optiling {
const int CHW_DIMS = 3;
const int NCHW_DIMS = 4;
const int KERNEL_POS = 0;
const int STRIDE_POS = 1;
const int PADDING_POS = 2;
const int DILATION_POS = 4;
const int CEIL_POS = 5;
const int FORMAT_POS = 6;
const int N_DIM_ = 0;
const int C_DIM_ = 1;
const int H_DIM_ = 2;
const int W_DIM_ = 3;
const int H_IDX_ = 0;
const int W_IDX_ = 1;
constexpr int64_t MAX_INT32 = 2147483647;
constexpr uint64_t SIMT_NCHW_TILING_KEY_INT32 = 500001;
constexpr uint64_t SIMT_NHWC_TILING_KEY_INT32 = 500002;
constexpr uint64_t SIMT_NCHW_TILING_KEY_INT64 = 500011;
constexpr uint64_t SIMT_NHWC_TILING_KEY_INT64 = 500012;
constexpr int64_t MAX_THREAD_NUM = 256;
constexpr size_t SYS_WORKSPACE_SIZE = 16 * 1024 * 1024;

BEGIN_TILING_DATA_DEF(MaxPoolWithArgmaxV3SimtTilingData)
TILING_DATA_FIELD_DEF(int64_t, threadNums);
TILING_DATA_FIELD_DEF(int64_t, blockNums);
TILING_DATA_FIELD_DEF(int64_t, nDim);
TILING_DATA_FIELD_DEF(int64_t, cDim);
TILING_DATA_FIELD_DEF(int64_t, hInDim);
TILING_DATA_FIELD_DEF(int64_t, wInDim);
TILING_DATA_FIELD_DEF(int64_t, hOutDim);
TILING_DATA_FIELD_DEF(int64_t, wOutDim);
TILING_DATA_FIELD_DEF(int64_t, kSizeH);
TILING_DATA_FIELD_DEF(int64_t, kSizeW);
TILING_DATA_FIELD_DEF(int64_t, stridesH);
TILING_DATA_FIELD_DEF(int64_t, stridesW);
TILING_DATA_FIELD_DEF(int64_t, padH);
TILING_DATA_FIELD_DEF(int64_t, padW);
TILING_DATA_FIELD_DEF(int64_t, dilationH);
TILING_DATA_FIELD_DEF(int64_t, dilationW);
TILING_DATA_FIELD_DEF(int64_t, ceilMode);
END_TILING_DATA_DEF;
// 500001 for NCHW 500002
REGISTER_TILING_DATA_CLASS(MaxPoolWithArgmaxV3_500001, MaxPoolWithArgmaxV3SimtTilingData);
REGISTER_TILING_DATA_CLASS(MaxPoolWithArgmaxV3_500002, MaxPoolWithArgmaxV3SimtTilingData);

struct InputSIMTInfo {
    array<uint64_t, NCHW_DIMS> inputShape;
    array<uint64_t, NCHW_DIMS> outShape;
    array<uint64_t, HW_DIMS> kernelSize;
    array<uint64_t, HW_DIMS> stride;
    array<uint64_t, HW_DIMS> pad;
    array<uint64_t, HW_DIMS> dilation;
    bool ceilMode;
    std::string data_format;
};

class MaxPoolWithArgmaxV3TilingSIMT : public MaxPoolWithArgmaxV3BaseTiling {
public:
    explicit MaxPoolWithArgmaxV3TilingSIMT(gert::TilingContext* context) : MaxPoolWithArgmaxV3BaseTiling(context)
    {}

    ~MaxPoolWithArgmaxV3TilingSIMT() override
    {}

protected:
    // 获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    // 计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 计算TilingKey
    uint64_t GetTilingKey() const;
    // 保存Tiling数据
    ge::graphStatus PostTiling() override;
    // tiling信息打屏
    std::string ToString(MaxPoolWithArgmaxV3SimtTilingData& tilingData);

private:
    uint64_t GenerateTilingKey(uint64_t innerKey);
    MaxPoolWithArgmaxV3SimtTilingData tiling;
    InputSIMTInfo inputData;
    int nDimPos = 0;
    int cDimPos = 1;
    int hDimPos = 2;
    int wDimPos = 3;
    int64_t outputDataCount = 0;
};

} // namespace optiling
#endif // MAX_POOL_WITH_ARGMAX_V3_SIMT_TILING_H