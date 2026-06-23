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
 * \file sleep_tiling_data.h
 * \brief tiling data struct for sleep operator
 *
 * cycles: SYS_CNT hardware clock ticks to spin-wait (CUDA clock64() semantics).
 *         Wall time = cycles / SYS_CNT_freq.  910B3 SYS_CNT rate = 1650 MHz.
 */

#ifndef SLEEP_TILING_DATA_H_
#define SLEEP_TILING_DATA_H_

struct SleepTilingData {
    int64_t cycles = 0;
};

#endif
