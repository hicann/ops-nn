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
 * \file add_rms_norm_dynamic_quant_v2.cpp
 * \brief arch35 kernel entry for AddRmsNormDynamicQuantV2.
 */
#include "../../add_rms_norm_dynamic_quant/arch35/add_rms_norm_dynamic_quant_kernel.h"

template <int8_t COMPUTE_MODE, bool Y3_MODE, bool Y4_MODE>
__global__ __aicore__ void add_rms_norm_dynamic_quant_v2(GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR smooth1,
                                                         GM_ADDR smooth2, GM_ADDR beta, GM_ADDR y1, GM_ADDR y2,
                                                         GM_ADDR y3, GM_ADDR y4, GM_ADDR x, GM_ADDR outScale1,
                                                         GM_ADDR outScale2, GM_ADDR workspace, GM_ADDR tiling)
{
    add_rms_norm_dynamic_quant_impl<COMPUTE_MODE, Y3_MODE, Y4_MODE>(x1, x2, gamma, smooth1, smooth2, beta, y1, y2, y3,
                                                                    y4, x, outScale1, outScale2, workspace, tiling);
}
