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
 * \file batch_mat_mul_v3_tiling_arch30.cpp
 * \brief
 */

#ifndef __OP_HOST_QUANT_BATCH_MAT_MUL_V3_TILING_ARCH20_H__
#define __OP_HOST_QUANT_BATCH_MAT_MUL_V3_TILING_ARCH20_H__

#include "op_host/tiling_base.h"
#include "tiling/tiling_api.h"
#include "../../op_kernel/quant_batch_matmul_v3_kernel_tiling_data.h"

namespace optiling {

bool IsSocVersionArch20Pertoken(gert::TilingContext* context);

class QuantBatchMatmulPertokenArch20 {
public:
    explicit QuantBatchMatmulPertokenArch20(gert::TilingContext* context) : context_(context){}
    ~QuantBatchMatmulPertokenArch20() = default;
    ge::graphStatus DoTiling();
    ge::graphStatus PostTiling();
    QuantMatmulPertokenTilingDataArch20 qbmmTilingDataArch20_{};
    gert::TilingContext *context_ = nullptr;
    uint64_t tilingKey_{0};
};
}
#endif // __OP_HOST_QUANT_BATCH_MAT_MUL_V3_TILING_ARCH20_H__
