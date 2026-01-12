/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file softmax_cross_entropy_with_logits_tiling_data.h
 * \brief softmax_cross_entropy_with_logits_tiling_data
 */


#ifndef SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_TILING_DATA_H
#define SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_TILING_DATA_H

struct SoftmaxCrossEntropyWithLogitsTilingData {
    int64_t realCoreNum;
    int64_t a; //a轴
    int64_t r; //r轴
    int64_t blockFactor; //a轴分核，主核数据量
    int64_t tailBlockFactor; //a轴分核，尾核数据量
    int64_t rUbNumFactor; //R轴切分，一次UB可以放下的数据量，全载模板下等于r，注意32b对齐
    int64_t aUbNumFactor; //A轴切分，一次UB可以放下的数据量，非全载模板下等于1，注意32b对齐
    int64_t aLoopTimes; //主核A方向循环搬移数据的次数
    int64_t aLoopTimesT; //尾核A方向循环搬移数据的次数
    int64_t aLoopTail; //主核A方向尾块的数据量
    int64_t aLoopTailT; //尾核A方向尾块的数据量
    int64_t rLoopTime; //不能全载时，R轴反向的循环次数
    int64_t rLoopTile; //不能全载时，R轴反向的尾块数据量
    int64_t rLoopTileAlign; //不能全载时，R轴反向的尾块数据量
    int64_t kTimesTail; //不能全载时，完全二分累加，存在主尾块相加的次数
    int64_t kTimes; //不能全载时，完全二分累加，2的k次方内循环次数
    int64_t featuresBrcDim;
    int64_t labelsBrcDim;
};

#endif //SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_TILING_DATA_H
