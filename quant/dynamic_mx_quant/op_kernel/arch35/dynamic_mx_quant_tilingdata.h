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
 * \file dynamic_mx_quant_with_dual_axis_tilingdata.h
 * \brief
 */

#ifndef OPS_NN_DYNAMIC_MX_QUANT_H
#define OPS_NN_DYNAMIC_MX_QUANT_H

#include <cstdint>

struct DynamicMxQuantTilingData {
    int64_t totalCoreNum;
    int64_t usedCoreNum;        // 实际使用的核数
    int64_t blockFactor;        // 单核循环次数
    int64_t tailBlockFactor;    // 尾核循环次数
    int64_t ubDim;              // 合轴后，ubfactor所切的轴
    int64_t uo;                 // 切分轴上的循环次数
    int64_t ubFactor;           // 单次循环要处理的数据大小
    int64_t tailUbFactor;       // 尾循环要处理的数据大小
    int64_t roundMode;          // 数据类型转换的模式
    int64_t dstType;            // 输出y的数据类型
    int64_t blockSize;          // 进行微缩的数据块大小
    int64_t scaleAlg;           // scale计算方法
    int64_t blockSizeNumInAxis; // 在axis轴上有多少个blocksize
    int64_t tailBlockSize;      // 指定轴要进行微缩的最后一个数据块大小
    int64_t isPad;              // axis指定的轴是否需要补到blocksize的整数倍
    int64_t isTailAxis;         // 是否为尾轴场景
    int64_t preAxisSize;        // 合轴后axis前面轴的大小
    int64_t postAxisSize;       // 合轴后axis后面轴的大小
    int64_t mxScaleSize;        // scale数据大小
    int64_t tilingKey;
    float dstTypeMax;
    float invDstTypeMax;
};

struct DynamicMxQuant4OptimizeTilingData {
    int64_t totalCoreNum;  // 总核数
    int64_t usedCoreNum;   // 实际使用的核数
    int64_t roundMode;     // 数据类型转换的模式
    int64_t dstType;       // 输出y的数据类型
    int64_t blockSize;     // 进行微缩的数据块大小
    int64_t isPad;         // 量化轴最后一个block无法被blockSize整除时，为True
    int64_t tailBlockSize; // 指定量化轴最后一个blockSize大小
    int64_t scaleAlg;
    int64_t tilingKey;
    int64_t quantAxisSize;    // 优化非尾轴模板量化轴大小
    int64_t preAxisSize;      // 合轴后axis前面轴的大小
    int64_t postAxisSize;     // 合轴后axis后面轴的大小
    int64_t mAlignSize;       // 量化轴对齐blockSize之后元素个数
    int64_t nAlignSize;       // 融合尾轴对齐32,64,128之后元素个数
    int64_t mAlignBlockCount; // 量化轴对齐blockSize之后block的个数，实际上等于blockSizeNumInAxis
    int64_t nAlignBlockCount; // 融合尾轴对齐32,64,128之后block的个数
    int64_t mAlignGroupCount; // 量化轴对齐blockSize*2（一个Group）之后Group的个数
    int64_t quantAxisIsOdd; // 量化轴是否是奇数，如果是奇数，则有些group会有一个全0的dummy block
    int64_t totalGroupNum;  // 当前shape总共需要多少个Group才能计算完
    int64_t groupPerCore;   // 每个核计算多少个Group
    int64_t groupPerTail;   // 尾核计算多少个Group
    int64_t groupPerUb;     // 每个UB可以放下多少个Group
    int64_t totalBlockNum;  // 总共处理的block数量，此处为对齐成group之后的block数量
    int64_t blockNumPerTask; // 每个任务处理多少个blcok
    int64_t totalTaskNum;    // 总任务数量，用总共处理的block数量除以每个任务处理多少个block
    int64_t rowPerHeadCore;
    int64_t rowPerTailCore;
    int64_t needPadPostAxis; // 融合尾轴是否需要对齐
    float dstTypeMax;
    float invDstTypeMax;
};

struct DynamicMxQuantTailAxisTilingData {
    int64_t tilingKey;
    int64_t ubSize;
    int64_t roundMode;
    int64_t blockSize;
    int64_t totalCoreNum;
    int64_t usedCoreNum;
    int64_t rowTileNum;        // row 方向上的切核数
    int64_t colTileNum;        // col 方向上的切核数
    int64_t rowNum;            // 合轴之后 -2 轴大小
    int64_t colNum;            // 合轴之后 -1 轴大小
    int64_t colNormalBlockNum; // 列方向头核处理的块数 (1 x 256)
    int64_t colTailLen;        // 列方向尾块长度
    int64_t rowNormalBlockNum; // 行方向头核处理的块数 (1 行)
    int64_t rowTailLen;        // 行方向尾块长度
    int64_t maxUbBlockNum;     // UB最大能放下的处理块数 (1 x 32) (8 的倍数)
    float dstTypeMax;
    float invDstTypeMax;
};

#endif // OPS_NN_DYNAMIC_MX_QUANT_WITH_DUAL_AXIS_H
