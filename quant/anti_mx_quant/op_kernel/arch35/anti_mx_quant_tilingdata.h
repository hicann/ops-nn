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
 * \file anti_mx_quant_tilingdata.h
 * \brief TilingData structure for AntiMxQuant operator.
 */

#ifndef OPS_NN_ANTI_MX_QUANT_TILINGDATA_H
#define OPS_NN_ANTI_MX_QUANT_TILINGDATA_H

#include <cstdint>

struct AntiMxQuantTilingData {
    int64_t ubSize{0};              // UB大小（字节）
    int64_t dstType{0};             // 输出数据类型
    int64_t totalCoreNum{0};        // AIV核总数
    int64_t usedCoreNum{0};         // 实际使用的核数
    int64_t rowTileNum{0};          // 行方向切核数
    int64_t colTileNum{0};          // 列方向切核数
    int64_t rowNum{0};              // 合并后的行大小（除最后一维外的所有维度）
    int64_t colNum{0};              // 合并后的列大小（最后一维）
    int64_t colNormalBlockNum{0};   // 列方向头核处理的块数（按512为单位）
    int64_t colTailLen{0};          // 列方向尾块长度
    int64_t rowNormalBlockNum{0};   // 行方向头核处理的块数
    int64_t rowTailLen{0};          // 行方向尾块长度
    int64_t maxUbBlockNum{0};       // UB能放下的最大1x32块数
};

#endif // OPS_NN_ANTI_MX_QUANT_TILINGDATA_H
