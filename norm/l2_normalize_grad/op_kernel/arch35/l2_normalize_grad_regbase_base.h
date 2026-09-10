/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * NOTE: Portions of this code were AI-generated and have been technically reviewed for functional accuracy.
 */

/*!
 * \file l2_normalize_grad_regbase_base.h
 * \brief L2NormalizeGrad arch35 四条 DX 模板的公共基类。
 *
 * 四条模板(full_load / split_d / strided / strided_split)的差异只在**计算与切分策略**,
 * 而"核内任务划分 + GM 绑定 + 四条队列 + 二维搬入搬出"四件事完全相同,统一收在这里。
 * 组织方式对齐同批交付的 instance_norm_grad(它有 instance_norm_grad_base.h)。
 */
#ifndef L2_NORMALIZE_GRAD_REGBASE_BASE_H
#define L2_NORMALIZE_GRAD_REGBASE_BASE_H

#include "l2_normalize_grad_regbase_common.h"

namespace L2NormalizeGrad {

template <typename T_X>
class RegbaseDxBase {
public:
    __aicore__ inline RegbaseDxBase(TPipe* pipe, const L2NormalizeGradTilingData* tilingData)
        : Ppipe_(pipe), tiling_(tilingData)
    {}

protected:
    // 公共 Init:核内任务划分 + GM 绑定 + 四条队列。
    // groupNum = 一个 outer 组的元素数(inner == 1 时即 D;inner > 1 时为 D*inner)。
    // 返回 false 表示本核无任务,调用方直接返回。
    __aicore__ inline bool InitCommon(__gm__ uint8_t* x, __gm__ uint8_t* y, __gm__ uint8_t* dy, __gm__ uint8_t* dx,
                                      int64_t groupNum)
    {
        usedCoreNum_ = tiling_->usedCoreNum;
        coreIdx_ = GetBlockIdx();
        if (coreIdx_ >= usedCoreNum_) {
            return false;
        }
        blockFactor_ = tiling_->blockFactor;
        eps_ = tiling_->eps;

        const int64_t gmOffset = static_cast<int64_t>(coreIdx_) * blockFactor_ * groupNum;
        xGm_.SetGlobalBuffer((__gm__ T_X*)x + gmOffset);
        yGm_.SetGlobalBuffer((__gm__ T_X*)y + gmOffset);
        dyGm_.SetGlobalBuffer((__gm__ T_X*)dy + gmOffset);
        dxGm_.SetGlobalBuffer((__gm__ T_X*)dx + gmOffset);

        return true;
    }

    // 四条队列一律按 host 下发的 qBufBytes 开;队列本身留在派生类(否则依赖名会逼出满篇 .template)。
    __aicore__ inline void InitQueues(TQue<QuePosition::VECIN, DEPTH_TWO>& qx, TQue<QuePosition::VECIN, DEPTH_TWO>& qy,
                                      TQue<QuePosition::VECIN, DEPTH_TWO>& qdy,
                                      TQue<QuePosition::VECOUT, DEPTH_TWO>& qdx)
    {
        Ppipe_->InitBuffer(qx, DB_NUM, tiling_->qBufBytes);
        Ppipe_->InitBuffer(qy, DB_NUM, tiling_->qBufBytes);
        Ppipe_->InitBuffer(qdy, DB_NUM, tiling_->qBufBytes);
        Ppipe_->InitBuffer(qdx, DB_NUM, tiling_->qBufBytes);
    }

    // 本核处理的 outer 组数(末核收尾)。
    __aicore__ inline int64_t CalcOuterNum(int64_t outer) const
    {
        const int64_t blockTail = outer - static_cast<int64_t>(usedCoreNum_ - 1) * blockFactor_;
        return (coreIdx_ == usedCoreNum_ - 1) ? blockTail : blockFactor_;
    }

    // 二维搬入:从 gm[base] 起搬 rows 行 x cols 列,行间在 GM 上隔 srcGap 个元素。
    // rightPadding=true 让每行在 UB 内按 32B 块补零(D 为奇数时避免 VEC_ERROR);
    // dstStride=0 即每行在 UB 内块对齐 —— 四条模板的行距约定都由此而来。
    __aicore__ inline void CopyIn2D(TQue<QuePosition::VECIN, DEPTH_TWO>& que, GlobalTensor<T_X>& gm, int64_t base,
                                    int64_t rows, int64_t cols, int64_t srcGap)
    {
        LocalTensor<T_X> local = que.AllocTensor<T_X>();
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(rows),                 // blockCount
            static_cast<uint32_t>(cols * sizeof(T_X)),   // blockLen (bytes)
            static_cast<uint32_t>(srcGap * sizeof(T_X)), // srcStride (GM 行间空隙)
            0,                                           // dstStride
            0                                            // rsv
        };
        DataCopyPad(local, gm[base], copyParams, {true, 0, 0, 0});
        que.EnQue(local);
    }

    // 二维搬出:与 CopyIn2D 对称,空隙记在 dstStride 上。
    __aicore__ inline void CopyOut2D(TQue<QuePosition::VECOUT, DEPTH_TWO>& que, int64_t base, int64_t rows,
                                     int64_t cols, int64_t dstGap)
    {
        LocalTensor<T_X> dxLocal = que.DeQue<T_X>();
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(rows),                 // blockCount
            static_cast<uint32_t>(cols * sizeof(T_X)),   // blockLen (bytes)
            0,                                           // srcStride
            static_cast<uint32_t>(dstGap * sizeof(T_X)), // dstStride (GM 行间空隙)
            0                                            // rsv
        };
        DataCopyPad(dxGm_[base], dxLocal, copyParams);
        que.FreeTensor(dxLocal);
    }

    TPipe* Ppipe_;
    const L2NormalizeGradTilingData* tiling_;

    GlobalTensor<T_X> xGm_;
    GlobalTensor<T_X> yGm_;
    GlobalTensor<T_X> dyGm_;
    GlobalTensor<T_X> dxGm_;

    uint32_t coreIdx_ = 0;
    uint32_t usedCoreNum_ = 0;
    int64_t blockFactor_ = 0;
    float eps_ = 0.0f;
};

} // namespace L2NormalizeGrad
#endif // L2_NORMALIZE_GRAD_REGBASE_BASE_H
