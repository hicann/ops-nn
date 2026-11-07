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
 * \file conv3d_backprop_input_v2_init_output_arch51.h
 * \brief
 */

#ifndef CONV3D_BACKPROP_INPUT_V2_INIT_OUTPUT_ARCH51_H
#define CONV3D_BACKPROP_INPUT_V2_INIT_OUTPUT_ARCH51_H

#include "utils/init_global_memory.h"

namespace AscendC {

enum class InitOutputFlag {
    NO_INIT = 0,
    L0_INIT = 1,
    L1_INIT = 2,
};

template <typename yType>
class Conv3dDxInitOutput {
public:
    __aicore__ inline Conv3dDxInitOutput() {}
    __aicore__ inline void Init(GM_ADDR y, const Conv3DBackpropInputArch35TilingData& tilingData)
    {
        InitTilingData(tilingData);
    }

    __aicore__ inline void Process(GM_ADDR y)
    {
        // 5162 单核：直接清全量输出区
        if constexpr (IsSameType<yType, hifloat8_t>::value || IsSameType<yType, fp8_e4m3fn_t>::value) {
            GlobalTensor<int8_t> yGm_;
            yGm_.SetGlobalBuffer((__gm__ int8_t*)y);
            AscendC::Fill<int8_t>(yGm_, outputSize_, (int8_t)(0));
        } else {
            GlobalTensor<yType> yGm_;
            yGm_.SetGlobalBuffer((__gm__ yType*)y);
            AscendC::Fill<yType>(yGm_, outputSize_, (yType)(0));
        }
    }

    __aicore__ inline void Destroy() { pipe_.Destroy(); }

protected:
    uint64_t outputSize_;
    TPipe pipe_;

    __aicore__ inline void InitTilingData(const Conv3DBackpropInputArch35TilingData& tilingData)
    {
        uint64_t mSize = static_cast<uint64_t>(tilingData.hi) * tilingData.wi;
        uint64_t nSize = static_cast<uint64_t>(tilingData.cin);
        outputSize_ = mSize * nSize * tilingData.di * tilingData.batch;
    }
};
} // namespace AscendC

#endif // CONV3D_BACKPROP_INPUT_V2_INIT_OUTPUT_ARCH51_H
