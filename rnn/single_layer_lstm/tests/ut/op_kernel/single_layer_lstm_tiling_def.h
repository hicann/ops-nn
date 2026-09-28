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
 * \file single_layer_lstm_tiling_def.h
 * \brief CPU-side stand-ins for the two tiling macros the kernel UT does not provide.
 *
 * The real REGISTER_TILING_DEFAULT emits the section entry the runtime needs to register a kernel
 * binary; there is no binary in the CPU UT, so it expands to nothing here. GET_TILING_DATA_WITH_STRUCT
 * becomes a reference to the buffer the test supplies -- which is what it does on device too, minus
 * the copy to the stack.
 *
 * The struct itself is NOT redeclared here. The kernel reads TilingData as a plain struct, so a
 * mirrored field list would be a second source of truth for the layout, free to drift in silence --
 * and a drifted tiling layout gives plausible wrong numbers, not an error. The test includes the
 * production header instead. This includes the appended logicalInputSize/logicalHiddenSize
 * fields: the kernel UT must allocate sizeof(the production struct) and initialize these
 * semantic extents, never append an independent mirrored field list in this macro shim.
 */

#ifndef OPS_RNN_SINGLE_LAYER_LSTM_TEST_TILING_DEF_H
#define OPS_RNN_SINGLE_LAYER_LSTM_TEST_TILING_DEF_H

#define REGISTER_TILING_DEFAULT(tilingStruct)

#define GET_TILING_DATA_WITH_STRUCT(tilingStruct, tilingData, tilingPointer) \
    const tilingStruct& tilingData = *reinterpret_cast<const tilingStruct*>(tilingPointer)

#endif // OPS_RNN_SINGLE_LAYER_LSTM_TEST_TILING_DEF_H
