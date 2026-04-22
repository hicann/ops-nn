/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef _ROTATE_QUANT_TILING_DEF_H
#define _ROTATE_QUANT_TILING_DEF_H
#define DTYPE_X bfloat16_t
#define DTYPE_Y int8_t

#include "kernel_tiling/kernel_tiling.h"
#include "rotate_quant_tiling_data.h"
#define CCE_UT_TEST

inline void InitRotateQuantTilingData(RotateQuantOpt::RotateQuantTilingData& tilingData, uint8_t* tilingArg)
{
    tilingData = *reinterpret_cast<const RotateQuantOpt::RotateQuantTilingData*>(tilingArg);
}

// 先取消 SDK 中可能已定义的函数式宏
#ifdef GET_TILING_DATA
#undef GET_TILING_DATA
#endif

#define GET_TILING_DATA RotateQuantOpt::RotateQuantTilingData tilingData; InitRotateQuantTilingData

#endif
