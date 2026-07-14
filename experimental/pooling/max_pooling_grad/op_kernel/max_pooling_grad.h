/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Zhou Jianhua <@LePenseur>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file max_pooling_grad.h
 * \brief
 */

#ifndef MAX_POOLING_GRAD_H
#define MAX_POOLING_GRAD_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "max_pooling_grad_tiling_data.h"

namespace NsMaxPoolingGrad {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;   // CompareScalar/Select 高级 API 场景使用单缓冲
constexpr int32_t BLOCK_SIZE = 256; // CompareScalar/Select API 要求 256B 对齐

// half 类型: CompareScalar 输出 uint8_t 掩码; float 类型: 输出 T 类型掩码
template <typename T>
struct SelectorType {
    using type = T;
};
template <>
struct SelectorType<half> {
    using type = uint8_t;
};

template <typename T>
class KernelMaxPoolingGrad {
    using S = typename SelectorType<T>::type;

public:
    __aicore__ inline KernelMaxPoolingGrad() {}

    /*! 初始化: 设置 GM 指针、TQue pipe buffer、core 数据范围 */
    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR x, GM_ADDR y, GM_ADDR dx,
                                const MaxPoolingGradTilingData* tilingData)
    {
        tiling_ = *tilingData;
        ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");

        // 多核数据划分: 前 tailBlockNum 个 core 为 big core, 其余为 small core
        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint64_t globalOffset = tiling_.bigCoreDataNum * coreIdx;
        if (coreIdx < tiling_.tailBlockNum) {
            coreDataNum_ = tiling_.bigCoreDataNum;
            loopNum_ = tiling_.bigCoreLoopNum;
            tailDataNum_ = tiling_.bigCoreTailDataNum;
        } else {
            coreDataNum_ = tiling_.smallCoreDataNum;
            loopNum_ = tiling_.smallCoreLoopNum;
            tailDataNum_ = tiling_.smallCoreTailDataNum;
            globalOffset -= (tiling_.bigCoreDataNum - tiling_.smallCoreDataNum) * (coreIdx - tiling_.tailBlockNum);
        }

        // #4: 最后一个 core 必须按 lastCoreValidDataNum 钳位，不得处理 256B 对齐膨胀出的尾部
        // padding（否则越界读 dy/x/y、越界写 dx）。host 仅在最后一个 small core 越界时置非 0。
        if (coreIdx == static_cast<uint64_t>(AscendC::GetBlockNum()) - 1 && tiling_.lastCoreValidDataNum != 0) {
            coreDataNum_ = tiling_.lastCoreValidDataNum;
            const uint64_t tile = tiling_.ubPartDataNum;
            loopNum_ = (coreDataNum_ + tile - 1) / tile; // ceil
            tailDataNum_ = (loopNum_ == 0) ? 0 : coreDataNum_ - tile * (loopNum_ - 1);
            if (tailDataNum_ == 0 && loopNum_ > 0) {
                tailDataNum_ = tile;
            }
        }

        dyGm_.SetGlobalBuffer((__gm__ T*)dy + globalOffset, coreDataNum_);
        xGm_.SetGlobalBuffer((__gm__ T*)x + globalOffset, coreDataNum_);
        yGm_.SetGlobalBuffer((__gm__ T*)y + globalOffset, coreDataNum_);
        dxGm_.SetGlobalBuffer((__gm__ T*)dx + globalOffset, coreDataNum_);

        // TQue pipe: 3 输入 + 1 输出
        pipe_.InitBuffer(inQueueDy_, BUFFER_NUM, tiling_.ubPartDataNum * sizeof(T));
        pipe_.InitBuffer(inQueueX_, BUFFER_NUM, tiling_.ubPartDataNum * sizeof(T));
        pipe_.InitBuffer(inQueueY_, BUFFER_NUM, tiling_.ubPartDataNum * sizeof(T));
        pipe_.InitBuffer(outQueueDx_, BUFFER_NUM, tiling_.ubPartDataNum * sizeof(T));
        // TBuf 中间缓冲: diff (T), selector (S), zero (T)
        pipe_.InitBuffer(diffBuf_, tiling_.ubPartDataNum * sizeof(T));
        pipe_.InitBuffer(selectorBuf_, tiling_.ubPartDataNum * sizeof(S));
        pipe_.InitBuffer(zeroBuf_, tiling_.ubPartDataNum * sizeof(T));
    }

    /*! 主循环: 按 tile 分块处理 */
    __aicore__ inline void Process()
    {
        uint64_t processDataNum = tiling_.ubPartDataNum;
        for (uint64_t i = 0; i < loopNum_; i++) {
            if (i == loopNum_ - 1) {
                processDataNum = tailDataNum_;
            }
            uint64_t computeDataNum = AlignToBlock(processDataNum);
            CopyIn(i, processDataNum);
            Compute(processDataNum, computeDataNum);
            CopyOut(i, processDataNum);
        }
    }

private:
    __aicore__ inline uint64_t AlignToBlock(uint64_t dataNum) const
    {
        constexpr uint64_t elementsPerBlock = BLOCK_SIZE / sizeof(T);
        uint64_t aligned = (dataNum + elementsPerBlock - 1) / elementsPerBlock * elementsPerBlock;
        return aligned > tiling_.ubPartDataNum ? tiling_.ubPartDataNum : aligned;
    }

    /*! 从 Global Memory 拷贝 dy/x/y 到 Local Tensor */
    __aicore__ inline void CopyIn(uint64_t prog, uint64_t processDataNum)
    {
        LocalTensor<T> dyLocal = inQueueDy_.AllocTensor<T>();
        LocalTensor<T> xLocal = inQueueX_.AllocTensor<T>();
        LocalTensor<T> yLocal = inQueueY_.AllocTensor<T>();
        DataCopy(dyLocal, dyGm_[prog * tiling_.ubPartDataNum], processDataNum);
        DataCopy(xLocal, xGm_[prog * tiling_.ubPartDataNum], processDataNum);
        DataCopy(yLocal, yGm_[prog * tiling_.ubPartDataNum], processDataNum);
        inQueueDy_.EnQue(dyLocal);
        inQueueX_.EnQue(xLocal);
        inQueueY_.EnQue(yLocal);
    }

    /*! 核心计算: diff = x-y, selector = (diff==0), dx = selector ? dy : 0 */
    __aicore__ inline void Compute(uint64_t processDataNum, uint64_t computeDataNum)
    {
        LocalTensor<T> dyLocal = inQueueDy_.DeQue<T>();
        LocalTensor<T> xLocal = inQueueX_.DeQue<T>();
        LocalTensor<T> yLocal = inQueueY_.DeQue<T>();
        LocalTensor<T> dxLocal = outQueueDx_.AllocTensor<T>();
        LocalTensor<T> diff = diffBuf_.Get<T>();
        LocalTensor<S> selector = selectorBuf_.Get<S>();
        LocalTensor<T> zeroTens = zeroBuf_.Get<T>();

        if (computeDataNum > processDataNum) {
            Duplicate(diff, static_cast<T>(1), computeDataNum);
        }

        // diff = x - y: 当 x 是最大值时 diff == 0
        Sub(diff, xLocal, yLocal, processDataNum);

        // selector = (diff == 0): 标记最大值位置
        T zeroVal = static_cast<T>(0);
        CompareScalar(selector, diff, zeroVal, CMPMODE::EQ, computeDataNum);

        // zeroTens 用于非最大值位置填充
        Duplicate(zeroTens, zeroVal, computeDataNum);

        // dx = selector ? dy : 0
        Select(dxLocal, selector, dyLocal, zeroTens, SELMODE::VSEL_TENSOR_TENSOR_MODE, computeDataNum);

        outQueueDx_.EnQue<T>(dxLocal);
        inQueueDy_.FreeTensor(dyLocal);
        inQueueX_.FreeTensor(xLocal);
        inQueueY_.FreeTensor(yLocal);
    }

    /*! 将计算结果 dx 写回 Global Memory */
    __aicore__ inline void CopyOut(uint64_t prog, uint64_t processDataNum)
    {
        LocalTensor<T> dxLocal = outQueueDx_.DeQue<T>();
        DataCopy(dxGm_[prog * tiling_.ubPartDataNum], dxLocal, processDataNum);
        outQueueDx_.FreeTensor(dxLocal);
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueDy_, inQueueX_, inQueueY_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueDx_;
    TBuf<TPosition::VECCALC> diffBuf_, selectorBuf_, zeroBuf_;
    GlobalTensor<T> dyGm_, xGm_, yGm_, dxGm_;

    uint64_t coreDataNum_ = 0;
    uint64_t loopNum_ = 0;
    uint64_t tailDataNum_ = 0;
    MaxPoolingGradTilingData tiling_;
};

} // namespace NsMaxPoolingGrad
#endif // MAX_POOLING_GRAD_H
