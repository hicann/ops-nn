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
 * \file mse_loss_tiling.h
 * \brief MseLoss tiling 数据定义
 */

#ifndef _I_MSELOSS_TILING_H_
#define _I_MSELOSS_TILING_H_

#include <cstdint>
#include <cstring>
#include "../../../op_kernel/mse_loss_tiling_data.h"
#include "tikicpulib.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "graph/c_types.h"
#include "ascendc/host_api/tiling/template_argument.h"

#ifndef __aicore__
#define __aicore__
#endif

#ifndef __gm__
#define __gm__
#endif

#ifndef __ubuf__
#define __ubuf__
#endif

inline void InitTilingData(uint8_t* tiling, MseLossTilingData* constData)
{
    memcpy(constData, tiling, sizeof(MseLossTilingData));
}

#define GET_TILING_DATA_WITH_STRUCT(tilingStruct, tilingData, tilingArg) \
    tilingStruct tilingData;                                             \
    InitTilingData(tilingArg, &tilingData)

#define GET_TILING_DATA(tilingData, tilingArg) \
    MseLossTilingData tilingData;              \
    InitTilingData(tilingArg, &tilingData)

#endif
