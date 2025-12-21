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
 * \file fused_matmul_tiling_key.cpp
 * \brief
 */

#include "fused_matmul_tiling_key.h"

namespace optiling {
namespace fused_matmul {
uint64_t FusedMatmulTilingKey::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(
        static_cast<uint64_t>(apiLevel_), static_cast<uint64_t>(atrans_) + (static_cast<uint64_t>(btrans_) << 1),
        static_cast<uint64_t>(model_), static_cast<uint64_t>(fullLoad_), static_cast<uint64_t>(out_),
        static_cast<uint64_t>(fusedOpType_));
}

} // namespace fused_matmul
} // namespace optiling
