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
 * \file scatter_list.cpp
 * \brief kernel entry for scatter_list on arch35(A5)
 *
 * A5 以单一 SIMT 实现覆盖 ScatterList 全部形态, 入口无需按场景分发。
 * var 为就地更新(varOut 与 var 同址), workspace 未使用。
 * 其余芯片仍由 op_kernel/scatter_list.cpp 的共享入口承接, 两者按 __NPU_ARCH__ 互斥编译。
 */

#include "scatter_list_simt.h"

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
extern "C" __global__ __aicore__ void scatter_list(GM_ADDR var, GM_ADDR indice, GM_ADDR updates, GM_ADDR mask,
                                                   GM_ADDR varOut, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ScatterListArch35TilingData);
    GET_TILING_DATA_WITH_STRUCT(ScatterListArch35TilingData, tilingData, tiling);

    // A5 只有一个场景键, 这里的运行时判定不做场景分发, 作用是让编译产物保持"通用单实例
    // kernel"形态(符号名即 kernel 名, 无编译期键后缀) —— 省掉它会让编译器按编译期默认键
    // 生成带 _0 后缀的实例, 与 runtime/测试框架按 kernel 名注册符号的约定不符。
    if (TILING_KEY_IS(105)) {
        NsScatterListArch35::Process<DTYPE_VAR, DTYPE_INDICE>(var, indice, updates, mask, &tilingData);
    }
}
#endif
