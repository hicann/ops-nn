/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file dynamic_hi_f4_quant_tilingdata.h
 * \brief TilingData structure for DynamicHiF4Quant kernel on ascend960dt/ascend960pr
 */

#ifndef OPS_NN_DYNAMIC_HIF4_QUANT_TILINGDATA_H
#define OPS_NN_DYNAMIC_HIF4_QUANT_TILINGDATA_H

#include <cstdint>

struct DynamicHiF4QuantTilingData {
    int64_t totalCoreNum; // 硬件可用 AIV 总核数（从 PlatformInfo 获取）
    int64_t
        usedCoreNum; // 实际使用的核数（-1 轴: min(totalCoreNum, totalTiles)；-2 轴: min(totalCoreNum, totalTaskNum)）
    int64_t tilingKey;   // 模板选择键：GET_TPL_TILING_KEY(quantAxis)，用于 kernel 实例化路由
    int64_t dim0;        // 合轴后的 batch 维度（所有非 -2/-1 维的乘积，仅 axis=-2 模式使用）
    int64_t dimNeg2;     // 倒数第二维大小
    int64_t dimNeg1;     // 倒数第一维大小
    int64_t blockW;      // UB 单次处理的列方向 block 宽度（固定 128 元素，仅 axis=-2 使用）
    int64_t splitBlockH; // UB 单次处理的行方向高度（= blockSize × groupPerUb，由 UB 容量计算得出）
    int64_t dimNeg2Tail; // -2 轴尾部块的实际行数（dimNeg2 % splitBlockH，余 0 则为 splitBlockH）
    int64_t dimNeg1Tail; // -1 轴尾部块的实际列宽（dimNeg1 % blockW，余 0 则为 blockW）
    int64_t dimNeg2SplitBlockNum; // -2 轴方向需切分的 UB 行块数（CeilDiv(dimNeg2, splitBlockH)）
    int64_t dimNeg1BlockNum;      // -1 轴方向需切分的列 block 数（CeilDiv(dimNeg1, blockW)）
    int64_t blockPerHeadCore; // 满载核（前 headCoreNum 个）分到的任务数（-2: UB block；-1: tile，CeilDiv 后同 -2 公式）
    int64_t blockPerTailCore; // 尾核分到的任务数（-2: UB block；-1: tile，整数除法）
    int64_t headCoreNum;      // 满载核数量（任务数 % 用核数），其余为尾核（-2: block；-1: tile）
    int64_t dimNeg2IsOdd; // -2 轴 scale 行数是否为奇数（CeilDiv(dimNeg2, 64) % 2 != 0），影响 scale 输出对齐
    int64_t dimNeg1IsOdd; // -1 轴 scale 列数是否为奇数（CeilDiv(dimNeg1, 64) % 2 != 0），影响 scale 输出对齐
    int64_t dimNeg1IsPad; // -1 轴原始长度是否非 64 整倍数（dimNeg1 % blockSize != 0），需要 padding
    int64_t blockCountPerBatch;     // 单 batch 的总 block 任务数（= dimNeg2SplitBlockNum × dimNeg1BlockNum）
    int64_t scale1ColCountPerBatch; // 单 batch 的 -1 轴 scale 列数（CeilDiv(dimNeg1, 64)）
    int64_t scale2RowCountPerBatch; // 单 batch 的 -2 轴 scale 行数（CeilDiv(dimNeg2, 64)）
    int64_t groupNum; // -1 轴：全局总 group 数（每 group = 64 个输入元素）；-2 轴不使用
    int64_t alignGroupNum; // -1 轴：单 tile 的 group 数（256 对齐；960DT 为 512，即 CCE 128x256 整块）；-2 轴不使用
    int64_t scaleAlg; // 缩放算法选择：0-默认(阈值为4/2)，1-阈值3.75/1.875
};

#endif // OPS_NN_DYNAMIC_HIF4_QUANT_TILINGDATA_H
