/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gru_grad.cpp
 * \brief GRU 反向算子 Kernel 唯一入口 —— 按 __NPU_ARCH__ 分派基线 (KFC) 与优化栈 (基础 API) 两套实现。
 *
 * 构建契约：
 *  - 入口源文件名 gru_grad.cpp = optype_snake("GruGrad") + ".cpp"，入口函数名 gru_grad =
 *    OpDef ExtendCfgInfo("opInterface.value")（生成构建脚本 main_func / origin_func_name 与
 *    get_kernel_source 的缺省查找名，三 SoC 统一）；opFile.value = gru_grad（runtime 按 SoC
 *    定位 kernel/<soc>/gru_grad/ 下的二进制目录，按 SoC 构建各自产出）。
 *  - __NPU_ARCH__ 由 dav_3510 kernel 编译期定义（本机 CANN 头文件 asc/include/c_api/sync/sync.h
 *    等大量使用同一判定），== 3510 走优化栈（op_kernel/arch35/，基础 API 裸 Mmad/Fixpipe），
 *    否则走基线（基线 arch22 kernel（KFC 高阶 API），语义不变）。
 *  - KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2) 保留在入口：dav_3510 不自动推导核类型，
 *    缺失即链接期 "auto derivate failed"；基线侧与 KFC 自动推导结果一致（1:2），无害。
 *  - 按 DTYPE_X 实例化 half/float 两份二进制（框架按输入 dtype 分别编译）。
 */

#include "kernel_operator.h"
// TilingData 公用单一声明：op_kernel/gru_grad_tiling_data.h（struct GruGradTilingData），
// 两套实现分别引入；基线 kernel 只消费前段字段，优化栈 kernel 全量消费。
// 两分支 REGISTER_TILING_DEFAULT 指向同一类型，合并为函数体顶部一处（每 SoC 只编一个分支，
// 注册行为不变）。

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
// ---- 优化栈 (dav_3510)：基础 API（裸 Mmad/Fixpipe）路径 --------------------------------------
#include "arch35/gru_grad_arch35.h"

// dav_3510 的 asc 树无 GET_TILING_DATA_WITH_STRUCT：自补实现。
// ⚠ 必须用字节循环——宏展开位于 __global__ __aicore__ 函数体内，glibc memcpy 是 __host__ 函数，
//   会报 "call to __host__ function from __global__"。
template <typename T>
__aicore__ inline void GruGradLoadTilingData(GM_ADDR src, T* dst, int64_t bytes)
{
    const __gm__ uint8_t* s = reinterpret_cast<const __gm__ uint8_t*>(src);
    uint8_t* d = reinterpret_cast<uint8_t*>(dst);
    for (int64_t i = 0; i < bytes; ++i) {
        d[i] = s[i];
    }
}
#else
// ---- 基线 (ascend910b / ascend910_93)：KFC 高阶 API 路径（语义不变）---------------------------
#include "arch22/gru_grad_arch22.h"
#include "lib/matmul_intf.h"
#endif

extern "C" __global__ __aicore__ void gru_grad(GM_ADDR x, GM_ADDR w_input, GM_ADDR w_hidden, GM_ADDR init_h,
                                               GM_ADDR output_h, GM_ADDR reset_gate, GM_ADDR update_gate,
                                               GM_ADDR new_gate, GM_ADDR h_n, GM_ADDR dy, GM_ADDR dh,
                                               GM_ADDR batch_sizes, GM_ADDR dx, GM_ADDR dh_prev, GM_ADDR dw_input,
                                               GM_ADDR dw_hidden, GM_ADDR db_input, GM_ADDR db_hidden,
                                               GM_ADDR workspace, GM_ADDR gruGradTiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(GruGradTilingData);

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    GruGradTilingData tilingData;
    GruGradLoadTilingData(gruGradTiling, &tilingData, static_cast<int64_t>(sizeof(GruGradTilingData)));

    GruGradKernel<DTYPE_X> gruGradOp;
    gruGradOp.Init(x, w_input, w_hidden, init_h, output_h, reset_gate, update_gate, new_gate, h_n, dy, dh, batch_sizes,
                   dx, dh_prev, dw_input, dw_hidden, db_input, db_hidden, &tilingData, workspace);
    gruGradOp.Process();
#else
    GET_TILING_DATA_WITH_STRUCT(GruGradTilingData, tilingData, gruGradTiling);

    const TCubeTiling* dgateMMTiling = &(tilingData.dgateMMParam);
    const TCubeTiling* dwIhMMTiling = &(tilingData.dwIhMMParam);
    const TCubeTiling* dwHhMMTiling = &(tilingData.dwHhMMParam);
    const TCubeTiling* dxMMTiling = &(tilingData.dxMMParam);

#define GRU_GRAD_OP_IMPL(MM_GATE_CFG_VAL, MM_I_CFG_VAL)                                                              \
    do {                                                                                                             \
        GruGradKernel<DTYPE_X, MM_GATE_CFG_VAL, MM_I_CFG_VAL> gruGradOp;                                             \
        REGIST_MATMUL_OBJ(&gruGradOp.pipe, GetSysWorkSpacePtr(), gruGradOp.dgateMM, dgateMMTiling, gruGradOp.dwIhMM, \
                          dwIhMMTiling, gruGradOp.dwHhMM, dwHhMMTiling, gruGradOp.dxMM, dxMMTiling);                 \
        gruGradOp.Init(x, w_input, w_hidden, init_h, output_h, reset_gate, update_gate, new_gate, h_n, dy, dh,       \
                       batch_sizes, dx, dh_prev, dw_input, dw_hidden, db_input, db_hidden, &tilingData, workspace);  \
        gruGradOp.Process();                                                                                         \
    } while (0)

    if (TILING_KEY_IS(0)) {
        GRU_GRAD_OP_IMPL(MM_CFG, MM_CFG);
    } else if (TILING_KEY_IS(1)) {
        // 大内轴: dxMM N 维 = inputSize >= 65535 时,使能 GM→L1 循环搬入保证精度
        GRU_GRAD_OP_IMPL(MM_CFG, MM_HUGE_CFG);
    }
#endif
}
