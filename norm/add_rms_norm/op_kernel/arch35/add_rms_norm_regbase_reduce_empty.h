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
 * \file add_rms_norm_regbase_reduce_empty.h
 * \brief 空张量(reduce empty)兜底 kernel: 归一化维 R==0 且外维 A>0 时 rstd=[A,1] 填 NaN,
 *        y/x=[A,0] 空不写。把 rstd 当一维 A 个 fp32, host 多核切分+定每循环元素数,
 *        kernel Duplicate NaN 一次、循环复用 buffer 搬出。
 */

#ifndef ADD_RMS_NORM_REGBASE_REDUCE_EMPTY_H
#define ADD_RMS_NORM_REGBASE_REDUCE_EMPTY_H
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace AddRmsNorm {
using namespace AscendC;

// ReduceEmpty 仅处理 A>0、R==0：y 和 x 的形状均为 [A,0]，没有元素可写；
// rstd 对应空集合的均方根倒数，结果无定义，因此按框架语义把连续的 A 个 FP32 元素填为 quiet NaN。
class KernelAddRmsNormRegBaseReduceEmpty {
public:
    __aicore__ inline explicit KernelAddRmsNormRegBaseReduceEmpty(TPipe* pipe) { pPipe = pipe; }

    __aicore__ inline void Init(GM_ADDR rstd, const AddRMSNormRegbaseReduceEmptyTilingData* tiling)
    {
        // Host 已把一维 rstd 按核切分；该模板不读取 x1/x2/gamma，也不接收 y/x 地址，
        // 从接口上保证空输出不会发生无意义的零长度搬运。
        usedCoreNum_ = tiling->usedCoreNum;
        rowsPerLoop_ = tiling->rowsPerLoop;
        rowsPerCore_ = tiling->rowsPerCore;
        rowsPerTailCore_ = tiling->rowsPerTailCore;
        tailCoreStartIndex_ = tiling->tailCoreStartIndex;
        int64_t coreIdx = GetBlockIdx();
        // 正常启动时 blockDim 等于 usedCoreNum；该判断同时保护 UT 或异常 TilingData 下的多余核。
        if (coreIdx >= static_cast<int64_t>(usedCoreNum_)) {
            return;
        }
        // 除最后一个参与核外，各核长度均为 rowsPerCore；尾核承接向上取整切分后的剩余元素。
        // 因为只有尾核长度不同，所以每个核的 GM 起点都可统一按 coreIdx*rowsPerCore 计算。
        rowsThisCore_ = (coreIdx < static_cast<int64_t>(tailCoreStartIndex_)) ? rowsPerCore_ : rowsPerTailCore_;
        rowOffset_ = coreIdx * static_cast<int64_t>(rowsPerCore_);
        // 核内再按 UB 容量循环，最后一轮使用 tailLen_，避免写过本核负责的 rstd 区间。
        // 用商余式向上取整，避免 rowsThisCore_==INT64_MAX 时先加后除发生溢出。
        loopCount_ = rowsThisCore_ / rowsPerLoop_ + static_cast<int64_t>(rowsThisCore_ % rowsPerLoop_ != 0);
        tailLen_ = rowsThisCore_ % rowsPerLoop_;
        if (tailLen_ == 0) {
            tailLen_ = rowsPerLoop_;
        }
        rstdGm_.SetGlobalBuffer((__gm__ float*)rstd + rowOffset_, rowsThisCore_);
        // 所有输出值完全相同，只需一块单缓冲：先生成一次 NaN，随后在多个搬出循环中反复读取。
        pPipe->InitBuffer(outNanQueue_, 1, rowsPerLoop_ * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        int64_t coreIdx = GetBlockIdx();
        if (coreIdx >= static_cast<int64_t>(usedCoreNum_) || rowsThisCore_ <= 0) {
            return;
        }

        // 先把整块 UB 填成 quiet NaN。VECOUT 队列的 EnQue/DeQue 建立 V 写完成到 MTE3 读取的依赖。
        LocalTensor<float> nanLocal = outNanQueue_.AllocTensor<float>();
        float nanVal = AscendC::NumericLimits<float>::QuietNaN();
        Duplicate(nanLocal, nanVal, static_cast<int32_t>(rowsPerLoop_));
        outNanQueue_.EnQue(nanLocal);
        nanLocal = outNanQueue_.DeQue<float>();

        for (int64_t i = 0; i < loopCount_; i++) {
            // 普通轮次复用完整 NaN 缓冲区，尾轮只搬 tailLen_ 个元素。DataCopyPad 的 blockLen
            // 单位是字节，因此这里乘 sizeof(float)；GM 偏移的单位仍是 FP32 元素。
            int64_t curLen = (i == loopCount_ - 1) ? tailLen_ : static_cast<int64_t>(rowsPerLoop_);
            DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = static_cast<uint32_t>(curLen * sizeof(float));
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopyPad(rstdGm_[i * static_cast<int64_t>(rowsPerLoop_)], nanLocal, copyParams);
        }

        // 所有 MTE3 搬出均已使用完该张量，再归还队列缓冲区。
        outNanQueue_.FreeTensor(nanLocal);
    }

private:
    TPipe* pPipe = nullptr;
    TQue<QuePosition::VECOUT, 1> outNanQueue_;
    GlobalTensor<float> rstdGm_;
    int64_t usedCoreNum_{0};
    int64_t rowsPerLoop_{0};
    int64_t rowsPerCore_{0};
    int64_t rowsPerTailCore_{0};
    int64_t tailCoreStartIndex_{0};
    int64_t rowsThisCore_{0};
    int64_t rowOffset_{0};
    int64_t loopCount_{0};
    int64_t tailLen_{0};
};
} // namespace AddRmsNorm
#endif // ADD_RMS_NORM_REGBASE_REDUCE_EMPTY_H
