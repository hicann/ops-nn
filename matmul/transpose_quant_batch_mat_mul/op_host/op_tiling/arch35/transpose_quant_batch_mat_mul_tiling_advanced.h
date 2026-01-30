/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file transpose_quant_batch_mat_mul_tiling_advanced.h
 * \brief
 */
#ifndef __OP_HOST_TRANSPOSE_QUANT_BATCH_MAT_MUL_TILING_ADVANCED_H__
#define __OP_HOST_TRANSPOSE_QUANT_BATCH_MAT_MUL_TILING_ADVANCED_H__

#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_tiling_advanced.h"
namespace optiling {
namespace transpose_quant_batch_mat_mul_advanced {
using namespace matmul_v3_advanced;
class TransposeQuantBatchMatMulTiling : public MatMulV3Tiling {
public:
    explicit TransposeQuantBatchMatMulTiling(gert::TilingContext* context) : MatMulV3Tiling(context) {};

    ~TransposeQuantBatchMatMulTiling() override = default;

    ge::graphStatus DoTiling() override;

protected:
    ge::graphStatus GetBatchInfo(const gert::TilingContext& context, MatMulV3Args& args,
                                 MatMulV3BatchInfo& batchInfo) const;

    ge::graphStatus GetShapeAttrsInfo() override;

    ge::graphStatus GetArgs() override;

    ge::graphStatus CheckArgs() override;

    ge::graphStatus CheckScale(const int64_t m,const int64_t n) const;
};
} // namespace transpose_quant_batch_mat_mul_advanced
} // namespace optiling
#endif // __OP_HOST_TRANSPOSE_QUANT_BATCH_MAT_MUL_TILING_ADVANCED_H__