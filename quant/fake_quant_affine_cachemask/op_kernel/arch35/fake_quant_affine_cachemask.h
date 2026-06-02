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
 * \file fake_quant_affine_cachemask.h
 * \brief 总入口 —— 按 MODE 派发到 PT / PC / PH 三个独立 kernel 类
 *
 *   MODE = 0          → FakeQuantAffineCachemaskPT
 *   MODE = 1 or 2     → FakeQuantAffineCachemaskPC  (PC / PC_NDDMA 共用同实现)
 *   MODE = 3          → FakeQuantAffineCachemaskPH
 *
 * 三类实现各占一个头文件：
 *   fake_quant_affine_cachemask_pt.h
 *   fake_quant_affine_cachemask_pc.h
 *   fake_quant_affine_cachemask_ph.h
 */
#ifndef FAKE_QUANT_AFFINE_CACHEMASK_H
#define FAKE_QUANT_AFFINE_CACHEMASK_H

#include "fake_quant_affine_cachemask_common.h"
#include "fake_quant_affine_cachemask_pt.h"
#include "fake_quant_affine_cachemask_pc.h"
#include "fake_quant_affine_cachemask_ph.h"

namespace NsFakeQuantAffineCachemask {

template <typename T, typename ZpT, int MODE, int HAS_ZP>
__aicore__ inline void RunOp(GM_ADDR x, GM_ADDR scale, GM_ADDR zp,
                              GM_ADDR y, GM_ADDR mask,
                              const FakeQuantAffineCachemaskTilingDataArch35* td)
{
    if constexpr (MODE == MODE_PT) {
        FakeQuantAffineCachemaskPT<T, ZpT, HAS_ZP> op;
        op.Init(x, scale, zp, y, mask, td);
        op.Process();
    } else if constexpr (MODE == MODE_PC || MODE == MODE_PC_NDDMA) {
        FakeQuantAffineCachemaskPC<T, ZpT, HAS_ZP> op;
        op.Init(x, scale, zp, y, mask, td);
        op.Process();
    } else if constexpr (MODE == MODE_PH) {
        FakeQuantAffineCachemaskPH<T, ZpT, HAS_ZP> op;
        op.Init(x, scale, zp, y, mask, td);
        op.Process();
    }
}

} // namespace NsFakeQuantAffineCachemask

#endif // FAKE_QUANT_AFFINE_CACHEMASK_H
