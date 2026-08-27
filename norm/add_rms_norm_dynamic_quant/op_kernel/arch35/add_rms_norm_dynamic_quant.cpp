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
 * \file add_rms_norm_dynamic_quant.cpp
 * \brief arch35 kernel entry for AddRmsNormDynamicQuant.
 */
#include "add_rms_norm_dynamic_quant_kernel.h"

template <int8_t COMPUTE_MODE, bool Y3_MODE, bool Y4_MODE>
__global__ __aicore__ void add_rms_norm_dynamic_quant(GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR smoothScale1,
                                                      GM_ADDR smoothScale2, GM_ADDR beta, GM_ADDR y1, GM_ADDR y2,
                                                      GM_ADDR x, GM_ADDR scale1, GM_ADDR scale2, GM_ADDR workspace,
                                                      GM_ADDR tiling)
{
    add_rms_norm_dynamic_quant_impl<COMPUTE_MODE, Y3_MODE, Y4_MODE>(x1, x2, gamma, smoothScale1, smoothScale2, beta, y1,
                                                                    y2, nullptr, nullptr, x, scale1, scale2, workspace,
                                                                    tiling);
}
