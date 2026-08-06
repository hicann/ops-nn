/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file embedding_dense_grad_tiling_key.h
 * \brief embedding_dense_grad tiling key declare
 */

#ifndef EMBEDDING_DENSE_GRAD_TILING_KEY_H
#define EMBEDDING_DENSE_GRAD_TILING_KEY_H

#include <cstdint>
#include "ascendc/host_api/tiling/template_argument.h"

#define EMBEDDING_DENSE_GRAD_SCH_MODE_SINGLE_ROW 0
#define EMBEDDING_DENSE_GRAD_SCH_MODE_SEGMENTED 1
#define EMBEDDING_DENSE_GRAD_SCH_MODE_PACKED 2

// Shared constants for embedding_dense_grad kernels
constexpr uint64_t BUFFER_NUM = 1;
constexpr uint64_t BLOCK_SIZE = 32;
constexpr uint64_t WORKSPACE_HEADER_FLOATS = 512;

ASCENDC_TPL_ARGS_DECL(EmbeddingDenseGrad,
                      ASCENDC_TPL_UINT_DECL(schMode, 2, ASCENDC_TPL_UI_LIST, EMBEDDING_DENSE_GRAD_SCH_MODE_SINGLE_ROW,
                                            EMBEDDING_DENSE_GRAD_SCH_MODE_SEGMENTED,
                                            EMBEDDING_DENSE_GRAD_SCH_MODE_PACKED));
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST,
                                                          EMBEDDING_DENSE_GRAD_SCH_MODE_SINGLE_ROW,
                                                          EMBEDDING_DENSE_GRAD_SCH_MODE_SEGMENTED,
                                                          EMBEDDING_DENSE_GRAD_SCH_MODE_PACKED)), );

#endif // EMBEDDING_DENSE_GRAD_TILING_KEY_H
