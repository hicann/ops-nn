/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file index_fill_struct.h
 * \brief tiling base data
 */

#ifndef INDEX_FILL_STRUCT_H
#define INDEX_FILL_STRUCT_H

#include <string>

namespace IndexFill {

constexpr uint64_t B8  = 1;
constexpr uint64_t B16 = 2;
constexpr uint64_t B32 = 4;
constexpr uint64_t B64 = 8;

template <typename T>
struct ComputeTypeGet {
    using type = typename std::conditional<
        sizeof(T) == B64, int64_t,
        typename std::conditional<
            sizeof(T) == B32, int32_t,
            typename std::conditional<
                sizeof(T) == B16, int16_t,
                typename std::conditional<sizeof(T) == B8, int8_t, T>::type
            >::type
        >::type
    >::type;
};

class IndexFillTilingData {
public:
    uint32_t coreNum = 0;
    uint64_t N = 0;                          // x在axis上的维度值
    uint64_t indicesNum = 0;                 // 索引tensor长度
    uint64_t indicesProcessMode = 0;         // 索引处理模式
    uint64_t frontCoreNumTaskIndices = 0;
    uint64_t tailCoreNumTaskIndices = 0;
    uint64_t frontCoreDataTaskIndices = 0;
    uint64_t tailCoreDataTaskIndices = 0;
    uint64_t ubSize = 0;
    uint64_t P = 0;
    uint64_t Q = 0;
    uint32_t tilingKey = 0;
};

class IndexFillSimtTilingData {
public:
    int64_t tilingKey;
    int64_t p;
    int64_t n;
    int64_t q;
    int64_t indicesNum;
    int64_t coreNum;
    int64_t usedCoreNum;           // 实际kernel使用的核数，为max(simdUsedCoreNum, simtUsedCoreNum)的取值.
    int64_t simdUsedCoreNum;       // simd搬运tiling侧估算需使用的核数
    int64_t simtUsedCoreNum;       // simt计算tiling侧估算需使用的核数
    int64_t frontCoreNum;          // 前frontCoreNum个核每个多处理一个block分片
    int64_t blockSize;             // 表示切分的一个block分片中有多少个元素
    int64_t tailBlockSize;         // 表示尾块中有多少个元素
    int64_t loopsPerFrontCore;     // 前frontCoreNum个核的单核循环次数
    int64_t loopsPerTailCore;      // 尾部这(usedCoreNum-frontCoreNum)个核的单核循环次数

public:
    std::string ToString() const {
        std::string result;
        result += "tilingKey: " + std::to_string(tilingKey);
        result += ", p: " + std::to_string(p);
        result += ", n: " + std::to_string(n);
        result += ", q: " + std::to_string(q);
        result += ", indicesNum: " + std::to_string(indicesNum);
        result += ", coreNum: " + std::to_string(coreNum);
        result += ", usedCoreNum: " + std::to_string(usedCoreNum);
        result += ", simdUsedCoreNum: " + std::to_string(simdUsedCoreNum);
        result += ", simtUsedCoreNum: " + std::to_string(simtUsedCoreNum);
        result += ", frontCoreNum: " + std::to_string(frontCoreNum);
        result += ", blockSize: " + std::to_string(blockSize);
        result += ", tailBlockSize: " + std::to_string(tailBlockSize);
        result += ", loopsPerFrontCore: " + std::to_string(loopsPerFrontCore);
        result += ", loopsPerTailCore: " + std::to_string(loopsPerTailCore);
        return result;
    }
};

class IndexFillSimdTilingData {
public:
    int64_t p;
    int64_t n;
    int64_t q;
    int64_t indicesNum;
    int64_t splitQ;
    int64_t usedCoreNum;
    int64_t blockFactorPN;
    int64_t usedCoreNumPN;
    int64_t tailBlockNumPN;                // 表示前tailBlockNumPN个核，多处理一个block.
    int64_t usedCoreNumQ;
    int64_t blockFactorQ;                  // AICore核分布可以看成是(corePN, coreQ)两个维度的分布。blockFactorQ表示将Q按照coreQ分核,每个核要处理的大小
    int64_t blockTailQ;
    int64_t blockFactorUbBufferMask;
    int64_t blockFactorTileNumQ;
    int64_t blockFactorUbFactorQ;          // 核内切分，每次Q搬运的个数
    int64_t blockFactorUbTailQ;
    int64_t blockTailUbBufferMask;
    int64_t blockTailTileNumQ;
    int64_t blockTailUbFactorQ;
    int64_t blockTailUbTailQ;
};

} // namespace IndexFill

#endif //INDEX_FILL_STRUCT_H
