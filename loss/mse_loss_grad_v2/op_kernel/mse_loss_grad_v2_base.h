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
 * \file mse_loss_grad_v2_base.h
 * \brief
 */
#ifndef MSE_LOSS_GRAD_V2_BASE_H
#define MSE_LOSS_GRAD_V2_BASE_H
#pragma once
#include <cstdint>
#include "kernel_operator.h"
using namespace AscendC;

__aicore__ constexpr TQueConfig GetMyTQueConfig(
    bool nd2nzIn, bool nz2ndIn, bool scmBlockGroupIn, uint32_t bufferLenIn, uint32_t bufferNumberIn,
    uint32_t consumerSizeIn, const TPosition consumerIn[])
{
    return {
        nd2nzIn,
        nz2ndIn,
        scmBlockGroupIn,
        bufferLenIn,
        bufferNumberIn,
        consumerSizeIn,
        {consumerIn[1], consumerIn[2], consumerIn[3], consumerIn[4], consumerIn[5], consumerIn[6], consumerIn[7]}};
}

const constexpr TPosition tp[8] = {TPosition::MAX, TPosition::MAX, TPosition::MAX, TPosition::MAX,
                                   TPosition::MAX, TPosition::MAX, TPosition::MAX, TPosition::MAX};
constexpr TQueConfig conf = GetMyTQueConfig(false, false, false, 0, 1, 0, tp);

template <typename inType>
class KernelMseLossGradBase
{
public:
    __aicore__ inline KernelMseLossGradBase()
    {}
    template <typename T1>
    __aicore__ inline T1 CeilAlign(T1 a)
    {
        return (a + 32 - 1) / 32 * 32;
    }

protected:
    float cof;
    bool pad = false;
    uint64_t totalLength;
    uint64_t gmSize;
    uint64_t blockLength;
    uint64_t tileNum;
    uint64_t tileLength;
    uint64_t tileLengthPtr;
    uint64_t tileLengthAlign;
    int32_t bufferNum;
};
#endif // _MSE_LOSS_GRAD_V2_BASE_H