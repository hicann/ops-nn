/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_gru_grad_tiling_def.h
 * \brief gru_grad kernel 测试用 tiling 数据结构。直接复用 op_kernel/gru_grad_tiling_data.h
 *        中定义的普通结构体 GruGradTilingData (GET_TILING_DATA 通过 memcpy 还原),
 *        避免测试侧再维护一份易漂移的镜像。
 */
#ifndef TEST_GRU_GRAD_TILING_DEF_H_
#define TEST_GRU_GRAD_TILING_DEF_H_

#include "kernel_tiling/kernel_tiling.h"
#include "gru_grad_tiling_data.h"

// 直接复用 op_kernel 中定义的普通结构体, 字段顺序与 host/kernel 完全一致
using GruGradTilingDataTest = GruGradTilingData;

inline void InitGruGradTilingDataTest(uint8_t* tiling, GruGradTilingDataTest* data)
{
    memcpy(data, tiling, sizeof(GruGradTilingDataTest));
}

#define GET_TILING_DATA(tiling_data, tiling_arg) \
    GruGradTilingDataTest tiling_data;           \
    InitGruGradTilingDataTest(tiling_arg, &tiling_data)
#endif // TEST_GRU_GRAD_TILING_DEF_H_
