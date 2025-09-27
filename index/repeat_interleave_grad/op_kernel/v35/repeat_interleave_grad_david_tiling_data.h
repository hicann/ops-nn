/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file repeat_interleave_grad_david_tiling_data.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_REPEAT_INTERLEAVE_GRAD_DAVID_TILING_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_REPEAT_INTERLEAVE_GRAD_DAVID_TILING_H_

#include "atvoss/reduce/reduce_tiling_data.h"


class UbParaUnit
{
public:
    int32_t ubCount = 0;
    int32_t ubFactor = 0;
    int32_t ubTailFactor = 0;
}; // ubCount = 1, 设置在ubTailFactor中

class RIGUbPara
{
public:
    UbParaUnit mainCoreUbPara;
    UbParaUnit tailCoreUbPara;
};

class RIGBlockPara
{
public:
    int32_t blockCount = 0;
    int64_t blockFactor = 0;
    int64_t blockTailFactor = 0;
}; // 同ubfactor的处理

class RepeatInterleaveGradDavidTilingData
{
public:
    int64_t lenM = 0;
    int64_t lenR = 0;
    int64_t lenN = 0;
    int64_t lenP = 0;
    int64_t lenRepeat = 0;
    int32_t rFactor = 0;
    RIGBlockPara mBlockPara;
    RIGBlockPara repeatBlockPara;
    RIGBlockPara nBlockPara;
    RIGUbPara mUbPara;
    RIGUbPara repeatUbPara; // repeat的ub切分参数
    RIGUbPara nUbPara;
    RIGUbPara repeatLoopPara;  // ub内因为分段取repeat产生的切分，区别于ub切分
    RIGUbPara repeatSumUbPara; // Repeat计算各个核的sum时，产生的Ub切分策略
    int32_t realCoreNum = 0;
    int32_t xBufferSize = 0;
    int32_t yBufferSize = 0;
    int32_t repeatBufferSize = 0;
    int32_t repeatSumBufferSize = 0; // Repeat计算各个核的sum时，所需要的bufferSize
    int32_t repeatCacheSize = 0;
    int32_t basicBlockSize = 0;
    int32_t coreNumPerM = 0;
    Ops::Base::ReduceOpTilingData reduceTiling;
};

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_REPEAT_INTERLEAVE_GRAD_DAVID_TILING_H_