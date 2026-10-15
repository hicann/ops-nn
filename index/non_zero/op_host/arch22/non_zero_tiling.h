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
 * \file non_zero_tiling.h
 * \brief
 */

#ifndef CANN_OPS_INDEX_NON_ZERO_OP_HOST_ARCH22_NON_ZERO_TILING_H_
#define CANN_OPS_INDEX_NON_ZERO_OP_HOST_ARCH22_NON_ZERO_TILING_H_

#include "register/tilingdata_base.h"

namespace optiling {
constexpr int64_t MAX_DIM_NUM = 8;

// Host 侧编译期信息，由 TilingParse 从平台能力读取后填充
struct NonZeroCompileInfo {
    int64_t coreNum = 0;
    int64_t ubSize = 0;
};

BEGIN_TILING_DATA_DEF(NonZeroTilingData)
TILING_DATA_FIELD_DEF(int64_t, numel);        // 输入元素总数
TILING_DATA_FIELD_DEF(int64_t, rank);         // 输入维度数（1~8）
TILING_DATA_FIELD_DEF(int64_t, realCoreNum);  // 实际参与计算的核数
TILING_DATA_FIELD_DEF(int64_t, numPerCore8);  // 前 realCoreNum-1 个核各处理的元素数（8 的倍数）
TILING_DATA_FIELD_DEF(int64_t, tailNum);      // 尾核处理的元素数
TILING_DATA_FIELD_DEF(int64_t, tailCapacity); // 尾核暂存段容量（= tailNum，不 align8）
TILING_DATA_FIELD_DEF(int64_t, transpose);    // 1: 输出 [dim, num]；0: 输出 [num, dim]
TILING_DATA_FIELD_DEF(int64_t, xElemSize); // 输入元素字节数（1/2/4：bool/int8/uint8、fp16/bf16、fp32/int32）
TILING_DATA_FIELD_DEF(int64_t, xIsBf16); // 输入是否为 bfloat16（rank1 向量快路径 Cast 目标类型选择）
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_DIM_NUM, shape);        // 各维大小（不足补 1）
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_DIM_NUM, mulInDimList); // 坐标反解用 stride
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(NonZero, NonZeroTilingData)

class NonZeroAscendCTilingImpl {
public:
    explicit NonZeroAscendCTilingImpl(gert::TilingContext* context) : context_(context) {};
    ge::graphStatus Init();
    ge::graphStatus SetKernelTiling();

private:
    ge::graphStatus GetCompileInfo();
    ge::graphStatus CalcShapeInfo();
    ge::graphStatus CalcWorkspaceSize(size_t* workspaceSize);

    gert::TilingContext* context_ = nullptr;
    NonZeroTilingData tilingData_;

    int64_t coreNum_ = 0;
    int64_t ubSize_ = 0;
    int64_t numel_ = 0;
    int64_t rank_ = 0;
    int64_t realCoreNum_ = 0;
    int64_t numPerCore8_ = 0;
    int64_t tailNum_ = 0;
    int64_t tailCapacity_ = 0;
    int64_t transpose_ = 0;
    int64_t xElemSize_ = 0;
    int64_t xIsBf16_ = 0;
    int64_t shape_[MAX_DIM_NUM] = {1, 1, 1, 1, 1, 1, 1, 1};
    int64_t mulInDimList_[MAX_DIM_NUM] = {1, 1, 1, 1, 1, 1, 1, 1};
};
} // namespace optiling

#endif // CANN_OPS_INDEX_NON_ZERO_OP_HOST_ARCH22_NON_ZERO_TILING_H_
