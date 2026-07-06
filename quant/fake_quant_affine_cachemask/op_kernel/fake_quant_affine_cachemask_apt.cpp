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
 * \file fake_quant_affine_cachemask_apt.cpp
 * \brief Kernel 入口（arch35）
 *
 * 模板参数（与 fake_quant_affine_cachemask_tiling_key.h 一致顺序）：
 *   D_T_X  : x dtype（C_DT_FLOAT / C_DT_FLOAT16 / C_DT_BF16）
 *   D_T_ZP : zp dtype（C_DT_INT32 / C_DT_FLOAT / C_DT_FLOAT16 / C_DT_BF16）
 *   MODE   : 0/1/2/3 (PT / PC / PC_NDDMA / PH)
 *   HAS_ZP : 0/1
 */

#include "arch35/fake_quant_affine_cachemask.h"

template <typename D_T_X, typename D_T_ZP, int MODE, int HAS_ZP>
__global__ __aicore__ void fake_quant_affine_cachemask(GM_ADDR x, GM_ADDR scale, GM_ADDR zp, GM_ADDR y, GM_ADDR mask,
                                                       GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(FakeQuantAffineCachemaskTilingDataArch35);
    GET_TILING_DATA_WITH_STRUCT(FakeQuantAffineCachemaskTilingDataArch35, tilingData, tiling);
    NsFakeQuantAffineCachemask::RunOp<D_T_X, D_T_ZP, MODE, HAS_ZP>(x, scale, zp, y, mask, &tilingData);
}
