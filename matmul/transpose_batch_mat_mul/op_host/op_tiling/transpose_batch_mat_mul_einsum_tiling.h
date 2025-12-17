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
 * \file transpose_batch_mat_mul_einsum_tiling.h
 * \brief
 */
#ifndef __OP_HOST_TRANSPOSE_BATCH_MAT_MUL_EINSUM_TILING_H__
#define __OP_HOST_TRANSPOSE_BATCH_MAT_MUL_EINSUM_TILING_H__

#include "transpose_batch_mat_mul_tiling.h"
#include "tiling_base/tiling_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "pp_matmul_default.h"


namespace optiling {
namespace transpose_batch_mat_mul {
inline int64_t PermDecode(const int64_t* perm, size_t size)
{
    int64_t trans_ = 0L;
    for (uint32_t i = 0; i < size; i++) {
        trans_ = trans_ * 10 + perm[i] + 1; // 乘10 移位
    }
    return trans_;
}
class TransposeBatchMatMulEinsumTiling : public pp_matmul::PpMatMulDefault
{
public:
    explicit TransposeBatchMatMulEinsumTiling(gert::TilingContext* context)
       : PpMatMulDefault(context) {}
    ~TransposeBatchMatMulEinsumTiling() override = default;

    void DoTiling();
    bool GetMatMulInfo();
    bool GetTilingKey();
    ge::graphStatus PostTiling();

private:
    const gert::ContinuousVector *aPermList_ = nullptr;
    const gert::ContinuousVector *bPermList_ = nullptr;
};
} // namespace transpose_batch_mat_mul
} // namespace optiling
#endif // __OP_HOST_TRANSPOSE_BATCH_MAT_MUL_EINSUM_TILING_H__