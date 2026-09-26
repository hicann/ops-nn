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
 * \file npu_scatter_add_tiling_def.h
 * \brief kernel UT 使用的 NpuScatterAddTilingData 镜像定义（布局与 op_host/arch22 下定义一致）
 */
#ifndef _TEST_NPU_SCATTER_ADD_TILING_DEF_H_
#define _TEST_NPU_SCATTER_ADD_TILING_DEF_H_

#include <cstdint>
#include <cstring>
#include <securec.h>
#include <kernel_tiling/kernel_tiling.h>

#pragma pack(1)
struct NpuScatterAddTilingData {
    uint32_t totalRows = 0;
    uint32_t hiddenState = 0;
    uint32_t alignHiddenState = 0;
    uint32_t usedCoreNum = 0;
    uint8_t withValid = 0;
};
#pragma pack()

inline void InitNpuScatterAddTilingData(uint8_t* tiling, NpuScatterAddTilingData* constData)
{
    memcpy(constData, tiling, sizeof(NpuScatterAddTilingData));
}

#define GET_TILING_DATA(tiling_data, tiling_arg) \
    NpuScatterAddTilingData tiling_data;         \
    InitNpuScatterAddTilingData(tiling_arg, &tiling_data)
#endif // _TEST_NPU_SCATTER_ADD_TILING_DEF_H_
