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
 * \file quant_batch_matmul_v4_common.h
 * \brief
 */

#ifndef QUANT_BATCH_MATMUL_V4_COMMON_H
#define QUANT_BATCH_MATMUL_V4_COMMON_H

#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "lib/matmul_intf.h"
#include "quant_batch_matmul_v4_constant.h"
#include "../quant_batch_matmul_v3/quant_batch_matmul_v3_block.h"
#include "../quant_batch_matmul_v3/quant_batch_matmul_v3_update.h"
#include "quant_batch_matmul_v4_tiling_data.h"

namespace AscendC {

class QuantBatchMatmulV4Common
{
protected:
    QuantBatchMatmulV3BaseBlock block_;
    QuantBatchMatmulV3Update update_;
    QBmmBlockOffset offset_;
    TPipe* pipe_;

    uint32_t usedCoreNum_;
    uint32_t blockIdx_;
    uint32_t subBlockIdx_;
    uint32_t groupSizeM_;
    uint32_t groupSizeN_;
    uint32_t groupSizeK_;
    uint32_t nKgroups_;
    uint32_t stepka_;
    uint32_t stepkb_;
    uint32_t stepKaTail_;
    uint32_t stepKbTail_;
    uint32_t kTail_;
    uint32_t baseM_;
    uint32_t baseN_;
    uint32_t ubCalcM_;
    uint32_t ubCalcN_;

    uint64_t offsetWorkspaceC_ = 0;
    uint64_t offsetWorkspacePong_ = 0;

    TEventID eAL1Ping12_;
    TEventID eAL1Pong12_;
    TEventID eBL1Ping12_;
    TEventID eBL1Pong12_;
    TEventID eAL1Ping21_;
    TEventID eAL1Pong21_;
    TEventID eBL1Ping21_;
    TEventID eBL1Pong21_;

    __aicore__ inline void commonInit(TPipe* tPipe, const TCubeTiling* matmulTiling)
    {
        subBlockIdx_ = GetSubBlockIdx();
        blockIdx_ = GetBlockIdx();
        if ASCEND_IS_AIV {
            if (GetTaskRation() != 0) { blockIdx_ /= GetTaskRation(); }
        }
        usedCoreNum_ = matmulTiling->usedCoreNum;
        pipe_ = tPipe;

        if (blockIdx_ >= usedCoreNum_) { return; }
        baseM_ = matmulTiling->baseM;
        baseN_ = matmulTiling->baseN;
        stepka_ = matmulTiling->stepKa;
        stepkb_ = matmulTiling->stepKb;
        offsetWorkspaceC_ = BUFFER_NUM * blockIdx_ * baseM_ * baseN_;
        offsetWorkspacePong_ = offsetWorkspaceC_ + baseM_ * baseN_;
    }

    __aicore__ inline void releaseEventID()
    {
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_MTE2>(eAL1Ping12_);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_MTE2>(eAL1Pong12_);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_MTE2>(eBL1Ping12_);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_MTE2>(eBL1Pong12_);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_MTE2>(eAL1Ping21_);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_MTE2>(eAL1Pong21_);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_MTE2>(eBL1Ping21_);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_MTE2>(eBL1Ping21_);
    }

    __aicore__ inline void initEventID()
    {
        eAL1Ping12_ = GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>();
        eAL1Pong12_ = GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>();
        eBL1Ping12_ = GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>();
        eBL1Pong12_ = GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>();
        eAL1Ping21_ = GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE1>();
        eAL1Pong21_ = GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE1>();
        eBL1Ping21_ = GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE1>();
        eBL1Pong21_ = GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE1>();
    }
};

} // namespace AscendC
#endif