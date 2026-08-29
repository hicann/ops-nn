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
 * \file device_utils.h
 * \brief
 */

#pragma once
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
namespace Cmct {
namespace Gemm {

template <AscendC::HardEvent event>
__aicore__ inline void TPipeSetWaitFlag()
{
    auto eventID = GetTPipePtr()->FetchEventID(event);
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

// blockAlign 32B
template <typename DataType, int64_t blockSize = 32>
__aicore__ inline int64_t AlignBlock(const int64_t& t)
{
    return AscendC::Align(t, static_cast<int64_t>(blockSize / sizeof(DataType)));
}

// L2 cache 关闭策略（仅 __NPU_ARCH__ == 5102 生效）
template <typename XType, typename WType>
__aicore__ inline void SetL2CacheHint(bool aL2CacheDisable, bool bL2CacheDisable, AscendC::GlobalTensor<XType>& aGlobal,
                                      AscendC::GlobalTensor<WType>& bGlobal)
{
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102)
    if (aL2CacheDisable) {
        aGlobal.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    }
    if (bL2CacheDisable) {
        bGlobal.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    }
#endif
}

} // namespace Gemm
} // namespace Cmct
