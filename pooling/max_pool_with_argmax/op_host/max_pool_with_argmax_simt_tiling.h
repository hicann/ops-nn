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
 * \file max_pool_with_argmax_simt_tiling.h
 * \brief simt imply for max_pool_with_argmax
 */

#ifndef CANN_MAX_POOL_WITH_ARGMAX_SIMT_TILING_H
#define CANN_MAX_POOL_WITH_ARGMAX_SIMT_TILING_H

#include "max_pool_with_argmax_tiling.h"
#include "../op_kernel/arch35/max_pool_with_argmax_struct_common.h"

namespace optiling {
const int H_IDX_ = 0;
const int W_IDX_ = 1;
constexpr int64_t MAX_INT32 = 2147483647;
constexpr uint64_t SIMT_NCHW_TILING_KEY_INT32 = 500001;
constexpr uint64_t SIMT_NHWC_TILING_KEY_INT32 = 500002;
constexpr uint64_t SIMT_NCHW_TILING_KEY_INT64 = 500011;
constexpr uint64_t SIMT_NHWC_TILING_KEY_INT64 = 500012;
constexpr int64_t MAX_THREAD_NUM = 256;

class MaxPoolWithArgmaxTilingSIMT : public MaxPoolWithArgmaxBaseTiling
{
public:
    explicit MaxPoolWithArgmaxTilingSIMT(gert::TilingContext* context) : MaxPoolWithArgmaxBaseTiling(context)
    {}

    ~MaxPoolWithArgmaxTilingSIMT() override
    {}

protected:
    // 计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 计算TilingKey
    uint64_t GetTilingKey() const override;
    // 保存Tiling数据
    ge::graphStatus PostTiling() override;
    // tiling信息打屏
    void DumpTilingInfo() override;

private:
    uint64_t GenerateTilingKey(uint64_t innerKey);
    int64_t outputDataCount = 0;
};

} // namespace optiling
#endif // MAX_POOL_WITH_ARGMAX_SIMT_TILING_H