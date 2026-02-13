/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SEGMENT_SUM_STRUCT_H
#define SEGMENT_SUM_STRUCT_H

struct SegmentSumSimtTilingData
{
    uint64_t outerDim{0};
    uint64_t innerDim{0};
    uint64_t initNumPerCore{0};
    uint64_t initNumTailCore{0};
    uint32_t isDeterministic{0};
    uint32_t maxSegIdsInUb{0};
    int32_t loopTimes{0};   // 整核循环次数
    int32_t loopTimesTailCore{0}; // 尾核循环次数
    uint32_t segIdsPerLoop{0};  //整核整循环处理多少id
    uint32_t segIdsPerLoopTailCore{0}; //尾核整循环处理多少id
    uint32_t segIdsTailLoop{0}; // 整核尾循环处理多少id
    uint32_t segIdsTailLoopTailCore{0}; //尾核尾循环处理多少id
};

struct SegmentSumSimdTilingData
{
    int64_t needCoreNum{0};
    int64_t innerDim{0};

    int64_t xBufferSize{0};
    int64_t segmentIdBufferSize{0};
    int64_t yBufferSize{0};

    int64_t usedCoreNumForClear{0}; // 清零使用的核数
    int64_t normalCoreClearNum{0}; // 正常核清零的元素数量
    int64_t tailCoreClearNum{0}; // 尾核清零的元素数量

    int64_t blockNumInRow{0}; // 行切分的核数
    int64_t blockNumInCol{0}; // 列切分的核数

    int64_t normalCoreInnerNum{0}; // 正常列核列上处理的inner数
    int64_t normalCoreOutterNum{0}; // 正常行核行上处理的行数

    int64_t normalCoreRowUbLoop{0}; // 正常行核ub在行上的循环次数
    int64_t normalCoreNormalLoopOutters{0}; // 正常行核ub正常循环一次处理的行数
    int64_t normalCoreTailLoopOutters{0}; // 正常行核ub尾循环一次处理的行数
    int64_t tailCoreRowUbLoop{0}; // 尾行核ub在行上的循环次数
    int64_t tailCoreNormalLoopOutters{0}; // 尾行核ub正常循环一次处理的行数
    int64_t tailCoreTailLoopOutters{0}; // 尾行核ub尾循环一次处理的行数

    int64_t normalCoreColUbLoop{0}; // 正常列核ub在列上的循环次数
    int64_t normalCoreNormalLoopInners{0}; // 正常列核ub正常循环一次处理的列数
    int64_t normalCoreTailLoopInners{0}; // 正常列核ub尾循环一次处理的列数
    int64_t tailCoreColUbLoop{0}; // 尾列核ub在列上的循环次数
    int64_t tailCoreNormalLoopInners{0}; // 尾列核ub正常循环一次处理的列数
    int64_t tailCoreTailLoopInners{0}; // 尾列核ub尾循环一次处理的列数

    int64_t usedCoreNumForMultAdd{0}; // 多核累加使用的核数
    int64_t normalCoreMultAddInners{0}; // 多核累加正常核处理的inner数

    int64_t normalCoreMultAddInnerLoop{0}; // 多核累加正常核列循环次数
    int64_t normalCoreMultAddNormalLoopInners{0}; // 多核累加正常核正常循环处理的inner数
    int64_t normalCoreMultAddTailLoopInners{0}; // 多核累加正常核尾循环处理的inner数
    int64_t tailCoreMultAddInnerLoop{0}; // 多核累加尾核列循环次数
    int64_t tailCoreMultAddNormalLoopInners{0}; // 多核累加尾核正常循环处理的inner数
    int64_t tailCoreMultAddTailLoopInners{0}; // 多核累加尾核尾循环处理的inner数

    int64_t multAddXBufferSize{0};
    int64_t multAddIdsBufferSize{0};
    int64_t multAddYBufferSize{0};
};

struct SegmentSumTilingData
{
    uint64_t outerDim{0};
};

#endif // SEGMENT_SUM_STRUCT_H