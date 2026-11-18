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
 * \file foreach_log_tiling_data.h
 * \brief Tiling schema for the ForeachLog arch35 dual-split transport.
 *        Field names of the flat-range part match ForeachFlatTilingData so the
 *        shared TensorListUnaryDispatcher/TypedUnaryMover templates keep working.
 *        The whole struct must stay below the ~4KB kernel dynamic-shared-memory
 *        launch limit (ForeachFlatTilingData at 3652B runs; 4.3KB fails to
 *        launch), hence the 32-bit offsets — the tiling rejects tensors whose
 *        element count or a core's flat range would not fit those fields.
 */

#ifndef FOREACH_LOG_TILING_DATA_H
#define FOREACH_LOG_TILING_DATA_H

#include <cstdint>

constexpr uint16_t FOREACH_LOG_MAX_TENSOR_COUNT = 256;
constexpr uint16_t FOREACH_LOG_MAX_CORE_COUNT = 80;

struct ForeachLogTilingData {
    int32_t tensorCount = 0;
    uint32_t tileElems = 0;
    int64_t tensorDataCountList[FOREACH_LOG_MAX_TENSOR_COUNT] = {0};
    uint16_t tensorStartList[FOREACH_LOG_MAX_CORE_COUNT] = {0};
    uint16_t tensorEndList[FOREACH_LOG_MAX_CORE_COUNT] = {0};
    int32_t tensorStartOffsetList[FOREACH_LOG_MAX_CORE_COUNT] = {0};
    int32_t tensorEndOffsetList[FOREACH_LOG_MAX_CORE_COUNT] = {0};
};

#endif // FOREACH_LOG_TILING_DATA_H
