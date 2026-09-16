/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file update_tensor_desc_kernel.h
 * \brief UpdateTensorDesc kernel 实现（arch35 / Ascend 950，非模板化类）。
 *   计算链 S1→S2→S3 单链（单 tile、单核、无 double buffer）：
 *     S1 CopyIn  : DataCopy y_gm → yUb（1 KB 整块读入，MTE2；先读入再覆写，
 *                  保留槽位原值透传）
 *     Sync1      : SetFlag/WaitFlag<HardEvent::MTE2_S>（S 管线标量消费）
 *     S2 Compute : SetValue 标量覆写 y[3] = rank，y[4+i] = shape[i]
 *     Sync2      : SetFlag/WaitFlag<HardEvent::S_MTE3>
 *     S3 CopyOut : DataCopy yUb → y_gm（1 KB 整块写回，MTE3）
 *   x 为占位输入，kernel 全程不读取；无 GM workspace。
 */

#pragma once
#include "kernel_operator.h"
#include "update_tensor_desc_tiling_data.h"

class UpdateTensorDescKernel {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const UpdateTensorDescTilingData* td, AscendC::TPipe* pipe)
    {
        (void)x; // x 占位输入：不绑定 GlobalTensor、不进 UB
        td_ = td;
        pipe_ = pipe;
        gmY_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(y), kDescSize);
        pipe_->InitBuffer(yBuf_, kDescSize * sizeof(int64_t));
        yUb_ = yBuf_.Get<int64_t>();
    }

    __aicore__ inline void Process()
    {
        auto evMte2S = GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_S);
        auto evSMte3 = GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_MTE3);

        CopyIn();
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(evMte2S);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(evMte2S);
        Compute();
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(evSMte3);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(evSMte3);
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn()
    {
        AscendC::DataCopyParams params;
        params.blockCount = 1;
        params.blockLen = 32; // 32 × 32 B = 1024 B = kDescSize × int64 整块
        params.srcStride = 0;
        params.dstStride = 0;
        AscendC::DataCopy(yUb_, gmY_, params);
    }

    __aicore__ inline void Compute()
    {
        yUb_.SetValue(kDimBaseIdx, td_->rank);
        for (int64_t i = 0; i < td_->rank; ++i) {
            yUb_.SetValue(kDimBaseIdx + 1 + i, td_->shape[i]);
        }
    }

    __aicore__ inline void CopyOut()
    {
        AscendC::DataCopyParams params;
        params.blockCount = 1;
        params.blockLen = 32; // 32 × 32 B = 1024 B 整块
        params.srcStride = 0;
        params.dstStride = 0;
        AscendC::DataCopy(gmY_, yUb_, params);
    }

    const UpdateTensorDescTilingData* td_ = nullptr;
    AscendC::TPipe* pipe_ = nullptr;
    AscendC::GlobalTensor<int64_t> gmY_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> yBuf_;
    AscendC::LocalTensor<int64_t> yUb_;
};
