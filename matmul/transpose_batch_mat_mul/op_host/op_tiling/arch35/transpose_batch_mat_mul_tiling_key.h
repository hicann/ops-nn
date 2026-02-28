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
 * \file transpose_batch_mat_mul_tiling_key.h
 * \brief
 */

#ifndef __OP_HOST_TBMM_TILING_KEY_H__
#define __OP_HOST_TBMM_TILING_KEY_H__

#include <sstream>
#include "op_host/tiling_key.h"
#include "../../../op_kernel/arch35/transpose_batch_mat_mul_tiling_key_public.h"

namespace optiling {
namespace transpose_batch_mat_mul_advanced {
class TBMMTilingKey {
public:
    uint64_t GetTilingKey() const;
    TBMMTilingKey& SetPermX1(TBMMPermX1 permX1);
    TBMMTilingKey& SetPermX2(TBMMPermX2 permX2);
    TBMMTilingKey& SetBatchSplitMode(TBMMBatchSplit batchSplitMode);

private:
    TBMMPermX1 permX1_ = TBMMPermX1::PERM_X1_0_1_2;
    TBMMPermX2 permX2_ = TBMMPermX2::PERM_X2_0_1_2;
    TBMMBatchSplit batchSplitMode_ = TBMMBatchSplit::BATCH_SPLIT_FALSE;
};
} // namespace transpose_batch_mat_mul_advanced
} // namespace optiling
#endif // __OP_HOST_TBMM_TILING_KEY_H__
