/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INDEX_PUT_V2_SIMD_DMA_TILING_H
#define INDEX_PUT_V2_SIMD_DMA_TILING_H

#include "index_put_v2_simd_tiling.h"

namespace optiling {
class IndexPutV2SimdDmaTiling : public IndexPutV2SimdTiling {
public:
    explicit IndexPutV2SimdDmaTiling(gert::TilingContext* context) : IndexPutV2SimdTiling(context) {}

protected:
    bool IsCapable() override;
    uint64_t GetTilingKey() const override;
    void DoUBTiling() override;
};
} // namespace optiling
#endif // INDEX_PUT_V2_SIMD_DMA_TILING_H
