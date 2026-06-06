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
 * \file kl_div_loss_grad_tiling_data.h
 * \brief tiling data struct
 */

#ifndef __KL_DIV_LOSS_GRAD_TILLING_DATA_H__
#define __KL_DIV_LOSS_GRAD_TILLING_DATA_H__

struct KlDivLossGradTilingData {
    uint32_t bigCoreDataNum;
    uint32_t smallCoreDataNum;
    uint32_t tileDataNum;
    uint32_t bigCoreNum;
    float coff;
};

#endif
