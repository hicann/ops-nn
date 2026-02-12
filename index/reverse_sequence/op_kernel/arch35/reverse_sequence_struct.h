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
 * \file reverse_sequence_struct.h
 * \brief tiling base data
 */

#ifndef REVERSE_SEQUENCE_STRUCT_H
#define REVERSE_SEQUENCE_STRUCT_H
namespace ReverseSequence {
class ReverseSequenceSimtTilingData4RegBase {
public:
    int64_t batchDim;
    int64_t seqDim;
    int64_t batchSize;
    int64_t reverseSize;
    int64_t perCoreHandleNums;
    int64_t xUbFactor;
    int64_t usedCoreNums;
    int64_t xUbLoop;
    int64_t xTailUbLoopSize;
    int64_t xTailCoreLoop;
    int64_t xTailCoreTailLoopSize;
};

class ReverseSequenceBSATilingData{
    public:
    int64_t bDim;
    int64_t sDim;
    int64_t aDim;
    int64_t blockFactor;
    int64_t blockTail;
    int64_t ubFactorB;
    int64_t ubFactorS;
    int64_t ubFactorA;
    int64_t bLoop;
    int64_t sLoop;
    int64_t aLoop;
    int64_t inUbSize;
    int64_t usedCoreNum;
    int64_t gatherMode;
    int64_t copyMode;
    int64_t splitMode;
    int64_t gatherUbSize;
    int64_t dtypeSize;
};
}
#endif