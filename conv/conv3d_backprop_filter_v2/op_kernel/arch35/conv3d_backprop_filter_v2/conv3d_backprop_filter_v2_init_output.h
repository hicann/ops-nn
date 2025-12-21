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
 * \file conv3d_backprop_filter_v2_init_output.h
 * \brief
 */
#ifndef CONV3D_BACKPROP_FILTER_V2_INIT_OUTPUT_H
#define CONV3D_BACKPROP_FILTER_V2_INIT_OUTPUT_H

namespace AscendC {
constexpr uint8_t SYNC_MODE2 = 2;
constexpr uint8_t VEC_FALG_ID = 9;
constexpr int32_t BEST_FIXPIPE_ELEMENTS = 32;  // 128B/sizeof(float)
template <typename yType>
class Conv3dDwInitOutput {
public:
    __aicore__ inline Conv3dDwInitOutput() {}
    __aicore__ inline void Init(GM_ADDR y, const conv_bp_v2_kernel::Conv3DBackpropFilterV2TilingData *tilingData)
    {
        InitTilingData(tilingData);
        // init global buffer
        yGm_.SetGlobalBuffer((__gm__ yType *)y);
    }

    __aicore__ inline uint64_t Ceil(uint64_t a, uint32_t b)
    {
        return (a + b - 1) / b;
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            OutputClearZeroWithUb2Gm();
        }
        if ASCEND_IS_AIC {
            CrossCoreWaitFlag(VEC_FALG_ID);
        }
        return;
    }

    __aicore__ inline void OutputClearZeroWithUb2Gm()
    {
        uint32_t usedCoreNum = GetBlockNum();
        uint64_t clearSizePerCore = AlignUp(Ceil(outputSize_, usedCoreNum), BEST_FIXPIPE_ELEMENTS);
        usedCoreNum = Ceil(outputSize_, clearSizePerCore);
        if (GetBlockIdx() >= usedCoreNum) {
            SyncAllCores();
            return;
        }
        uint64_t offset = clearSizePerCore * GetBlockIdx();
        uint64_t tail = outputSize_ - offset;
        clearSizePerCore = tail > clearSizePerCore ? clearSizePerCore : tail;
        InitOutput<float>(yGm_[offset], clearSizePerCore, 0.0f);
        SyncAllCores();
    }

    __aicore__ inline void SyncAllCores()
    {
        CrossCoreSetFlag<0, PIPE_MTE3>(SYNC_AIV_FLAG);
        CrossCoreWaitFlag(SYNC_AIV_FLAG);
        CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(VEC_FALG_ID);
    }

    __aicore__ inline void Destroy()
    {
        pipe_.Destroy();
    }

protected:
    uint64_t outputSize_;
    TPipe pipe_;
    GlobalTensor<yType> yGm_;

    __aicore__ inline void InitTilingData(const conv_bp_v2_kernel::Conv3DBackpropFilterV2TilingData *tilingData)
    {
        uint64_t mSize = static_cast<uint64_t>(tilingData->dwTiling.cout);
        uint64_t nSize = static_cast<uint64_t>(tilingData->dwTiling.cin) * tilingData->dwTiling.dk *
            tilingData->dwTiling.hk * tilingData->dwTiling.wk;
        if (tilingData->dwTiling.group > 1) {
            nSize = static_cast<uint64_t>(tilingData->dwTiling.cin / tilingData->dwTiling.group) *
                tilingData->dwTiling.dk * tilingData->dwTiling.hk * tilingData->dwTiling.wk;
        }
        outputSize_ = mSize * nSize;
    }
};
}  // namespace AscendC

#endif  // CONV3D_BACKPROP_FILTER_V2_INIT_OUTPUT_H
