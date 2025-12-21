/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file transpose_batch_mat_mul_tiling_key.cc
 * \brief
 */

#include "../../../op_kernel/arch35/transpose_batch_mat_mul_tiling_key.h"
#include "transpose_batch_mat_mul_tiling_key.h"

using namespace optiling::transpose_batch_mat_mul_advanced;

uint64_t TBMMTilingKey::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(static_cast<uint64_t>(permX1_), static_cast<uint64_t>(permX2_),
                              static_cast<uint64_t>(batchSplitMode_));
}

TBMMTilingKey& TBMMTilingKey::SetPermX1(TBMMPermX1 permX1)
{
    permX1_ = permX1;
    return *this;
}

TBMMTilingKey& TBMMTilingKey::SetPermX2(TBMMPermX2 permX2)
{
    permX2_ = permX2;
    return *this;
}

TBMMTilingKey& TBMMTilingKey::SetBatchSplitMode(TBMMBatchSplit batchSplitMode)
{
    batchSplitMode_ = batchSplitMode;
    return *this;
}