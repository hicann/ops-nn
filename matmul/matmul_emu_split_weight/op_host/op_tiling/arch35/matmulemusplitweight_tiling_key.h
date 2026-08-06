/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "../../../op_kernel/arch35/matmul_emu_split_weight_tiling_key.h"

namespace optiling {
namespace matmul_emu_split_weight {

class MatmulEmuSplitWeightTilingKey {
public:
    MatmulEmuSplitWeightTilingKey& SetTrans(bool aTrans, bool bTrans)
    {
        if (aTrans) {
            aTrans_ = MATMUL_EMU_SPLIT_WEIGHT_TRANS;
        }
        if (bTrans) {
            bTrans_ = MATMUL_EMU_SPLIT_WEIGHT_TRANS;
        }
        return *this;
    }

    uint64_t GetTilingKey() const;

private:
    uint64_t aTrans_{MATMUL_EMU_SPLIT_WEIGHT_NO_TRANS};
    uint64_t bTrans_{MATMUL_EMU_SPLIT_WEIGHT_NO_TRANS};
};

} // namespace matmul_emu_split_weight
} // namespace optiling
