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
 * \file swiglu_group_quant_with_dual_axis_tiling_def.h
 * \brief
 */

#ifndef SWIGLU_GROUP_QUANT_WITH_DUAL_AXIS_TILING_DEF_H_
#define SWIGLU_GROUP_QUANT_WITH_DUAL_AXIS_TILING_DEF_H_

#include <cstring>
#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

#define __CCE_UT_TEST__

#include "../../../op_kernel/arch35/swiglu_group_quant_with_dual_axis_tiling_data.h"

template <typename T>
inline void InitTilingData(uint8_t* tiling, T* tilingData)
{
    std::memcpy(tilingData, tiling, sizeof(T));
}

#define GET_TILING_DATA(tilingData, tilingArg)         \
    SwigluGroupQuantWithDualAxisTilingData tilingData; \
    InitTilingData(tilingArg, &tilingData)

#define GET_TILING_DATA_WITH_STRUCT(tilingStruct, tilingData, tilingArg) \
    tilingStruct tilingData;                                             \
    InitTilingData(tilingArg, &tilingData)

#endif // SWIGLU_GROUP_QUANT_WITH_DUAL_AXIS_TILING_DEF_H_
