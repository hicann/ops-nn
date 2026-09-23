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
 * \file swiglu_group_quant_mx_tiling_data.h
 * \brief Dual-axis-owned MX tiling data shared with the single-axis operator.
 */

#ifndef OPS_NN_SWIGLU_GROUP_QUANT_MX_TILING_DATA_H
#define OPS_NN_SWIGLU_GROUP_QUANT_MX_TILING_DATA_H

#include <cstdint>

struct SwigluGroupQuantMxTilingData {
    int64_t usedCoreNum;
    int64_t activateLeft;
    int64_t dimM;
    int64_t dimN;
    int64_t numGroups;
    int64_t dimNBlockNum;
    int64_t dimNTail;
};

#endif // OPS_NN_SWIGLU_GROUP_QUANT_MX_TILING_DATA_H
