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
 * \file cosine_embedding_loss_tiling.h
 * \brief CosineEmbeddingLoss RegBase (arch35) tiling compile info.
 */
#ifndef OPS_LOSS_COSINE_EMBEDDING_LOSS_TILING_H_
#define OPS_LOSS_COSINE_EMBEDDING_LOSS_TILING_H_

#include <cstdint>

namespace optiling {
struct CosineEmbeddingLossCompileInfo {
    int64_t coreNum = 0;
    int64_t ubSize = 0;
};
} // namespace optiling

#endif // OPS_LOSS_COSINE_EMBEDDING_LOSS_TILING_H_
