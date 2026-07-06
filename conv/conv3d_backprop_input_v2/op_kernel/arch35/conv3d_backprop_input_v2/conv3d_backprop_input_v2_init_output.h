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
 * \file conv3d_backprop_input_v2_init_output.h
 * \brief
 */
#ifndef CONV3D_BACKPROP_INPUT_V2_INIT_OUTPUT_ADVANCE_H
#define CONV3D_BACKPROP_INPUT_V2_INIT_OUTPUT_ADVANCE_H

#include "conv3d_backprop_input_v2_tiling_data.h"

namespace AscendC {
constexpr uint8_t VEC_FALG_ID = 5;
#if __CCE_AICORE__ == 310 || (__NPU_ARCH__ == 5102)
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 5102)
constexpr int32_t TOTALL0CSIZE = 262144;
#endif
#endif

enum class InitOutputFlag {
    NO_INIT = 0,
    L0_INIT = 1,
    L1_INIT = 2,
};

template <typename yType>
class Conv3dDxInitOutput {
public:
    __aicore__ inline Conv3dDxInitOutput() {}
    __aicore__ inline void Init(GM_ADDR y, const conv_bp_v2_kernel::Conv3DBackpropInputV2TilingData* tilingData)
    {
        InitTilingData(tilingData);
    }

    /** main logical function
     */

    __aicore__ inline uint64_t Ceil(uint64_t a, uint32_t b) { return (a + b - 1) / b; }

    __aicore__ inline void ProcessWithL0(GM_ADDR y)
    {
        // david上数据量为128B对齐搬运效率最高，因此需对齐到128B，由于搬运的是fp32，因此搬运数量为32的倍数
        uint64_t clearSizePerCore = AlignUp(Ceil(outputSize_, GetBlockNum()), ONE_BLK_SIZE);
        if (GetBlockIdx() >= Ceil(outputSize_, clearSizePerCore)) {
            SyncAllCores();
            return;
        }

        uint64_t offset = clearSizePerCore * GetBlockIdx();
        uint64_t realClearSize = outputSize_ - offset;
        realClearSize = realClearSize > clearSizePerCore ? clearSizePerCore : realClearSize;

        if constexpr (IsSameType<yType, hifloat8_t>::value || IsSameType<yType, fp8_e4m3fn_t>::value) {
            GlobalTensor<int8_t> yGm_;
            yGm_.SetGlobalBuffer((__gm__ int8_t*)y);
            InitOutput<int8_t>(yGm_[offset], realClearSize, (int8_t)(0));
        } else {
            GlobalTensor<yType> yGm_;
            yGm_.SetGlobalBuffer((__gm__ yType*)y);
            InitOutput<yType>(yGm_[offset], realClearSize, (yType)(0));
        }

        SyncAllCores();
    }

    __aicore__ inline void Process(GM_ADDR y)
    {
        if ASCEND_IS_AIV_SCALAR {
            ProcessWithL0(y);
        }
        if ASCEND_IS_AIC_SCALAR {
#if (__NPU_ARCH__ == 5102)
            AscendC::TQueSync<PIPE_MTE1, PIPE_MTE3> sync;
            sync.WaitFlag(VEC_FALG_ID);
#else
            CrossCoreWaitFlag(VEC_FALG_ID);
#endif
        }
    }

    __aicore__ inline void SyncAllCores()
    {
#if (__NPU_ARCH__ == 5102)
        AscendC::TQueSync<PIPE_MTE1, PIPE_MTE3> sync;
        sync.SetFlag(VEC_FALG_ID);
#else
        CrossCoreSetFlag<0, PIPE_MTE3>(SYNC_AIV_FLAG);
        CrossCoreWaitFlag(SYNC_AIV_FLAG);
        CrossCoreSetFlag<2, PIPE_MTE3>(VEC_FALG_ID);
#endif
    }

    __aicore__ inline void Destroy() { pipe_.Destroy(); }

protected:
    uint64_t outputSize_;
    TPipe pipe_;
    TBuf<TPosition::CO1> localBuffer_;

    __aicore__ inline void InitTilingData(const conv_bp_v2_kernel::Conv3DBackpropInputV2TilingData* tilingData)
    {
        uint64_t mSize = static_cast<uint64_t>(tilingData->conv3DDxTiling.hi) * tilingData->conv3DDxTiling.wi;
        uint64_t nSize = static_cast<uint64_t>(tilingData->conv3DDxTiling.cin);
        outputSize_ = mSize * nSize * tilingData->conv3DDxTiling.di * tilingData->conv3DDxTiling.batch;
    }
};
} // namespace AscendC

#endif // CONV3D_BACKPROP_INPUT_V2_INIT_OUTPUT_ADVANCE_H