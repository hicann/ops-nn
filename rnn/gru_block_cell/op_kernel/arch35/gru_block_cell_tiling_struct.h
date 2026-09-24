/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED on an "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdint>

// TilingData 与决策常量（host—kernel 共享 POD；同包原子构建，无跨包 ABI 约束）。
// 决策全部在 host TilingFunc（ComputeLayoutDecision）算出，kernel 只读消费。

// 维度上界：kernel 以 uint32_t 承载 B/I/H，超 2^32 会窄化回绕产生静默错数
// （I 回绕使 split-K 切片区间全错，B 回绕使多数核早退、输出行不写）。
constexpr int64_t GRU_BLOCK_CELL_MAX_DIM = 2147483647LL;
// 声明域 I/H ∈ [1,65535]；H 实际支持域由片上容量 gate 收窄（见 CheckLayoutCapacity）。
constexpr int64_t GRU_BLOCK_CELL_MAX_INPUT = 65535;

constexpr int64_t GRU_TIL_C0F = 8;            // fp32 C0 元素（32B）
constexpr int64_t GRU_TIL_CUBE_BLOCK = 16;    // M/N 分形边长（元素）
constexpr int64_t GRU_TIL_UB_RESERVE = 1024;  // UB 预留对齐余量（字节）
constexpr int64_t GRU_TIL_CAP_L0A_ROWS = 880; // L0A A-tile 行界（LOAD2D 实测表征界，非容量查询值）
constexpr int64_t GRU_TIL_CAP_L0A = 64 * 1024;
constexpr int64_t GRU_TIL_CAP_L0B = 64 * 1024;
// 以下四项仅为平台查询失败时的兜底；正常路径取 GetCoreMemSize 在线值。
// ⚠ L0C 实测存在口径分歧：host GetCoreMemSize(L0_C) 返回 131072，而 kernel 侧
// AscendC::TOTAL_L0C_SIZE（__NPU_ARCH__==3510）为 262144。决策采查询值（更保守：
// nL0c 减半、列片数翻倍，每片足迹减半；实际足迹受 UB 预算耦合恒 ≤64KB，两者均安全）。
constexpr int64_t GRU_TIL_CAP_L0C = 256 * 1024;
constexpr int64_t GRU_TIL_CAP_L1 = 512 * 1024;
constexpr int64_t GRU_TIL_AIV_UB = 253952;   // = AscendC::TOTAL_UB_SIZE（3510）
constexpr int64_t GRU_TIL_C_GROUP_ROWS = 16; // split-K 组宽目标（精度策略）
// UB 静态足迹总宽口径：6 独立平面 + t1/t2 两个 scratch（t3≡rAcc、hp≡t1 等别名不占
// 额外槽）。mChunk 预算与 CheckLayoutCapacity 验算**共用此单一真值**。
// ⚠ 平面宽为 **列片宽 nL0c**（非全幅 Hp）——列片外提为最外层循环后，UB 只需容纳
// 当前列片的 8 个 [rowsMax, nL0c] 平面，这是 mChunk 能脱离 Hp 的关键（见 kernel.h
// DATAFLOW NOTES #5/#7）。
constexpr int64_t GRU_TIL_STATIC_PLANES = 8;
constexpr int64_t GRU_TIL_BITS_PER_BYTE = 8;
// mChunk 下限（M 分形保底；低于此值 cube M 向利用率过低）
constexpr int64_t GRU_TIL_MIN_MCHUNK = 16;
// wRu 合并的门数（r|u 两门 ⇒ wRu 列宽 2H、bRu 元素数 2H）。原为散落的字面量 2，
// 收敛为具名常量（评审：魔鬼数字）。
constexpr int64_t GRU_GATE_NUM = 2;

struct GruBlockCellTilingData {
    int64_t batchSize = 0;  // B
    int64_t inputSize = 0;  // I
    int64_t hiddenSize = 0; // H

    // 多核切分（batch 行切，块间无依赖 → 无跨核栅栏）
    int64_t rowsPerCore = 0; // 满行切=sM / 商余分核=商 q
    int64_t rowsTail = 0;    // 满行切=尾核行数 / 商余分核=余 rem（前 rem 核各 +1 行）
    int64_t coreNumUsed = 0; // 实际启用核数（blockDim，AIC 块为单位）
    int64_t splitMode = 0;   // 0=满行切  1=商余分核（显式编码，kernel 不做算术判别）

    // 片上布局决策（host 唯一计算，kernel 只读）
    int64_t padHidden = 0; // Hp = CeilAlign(H,8)：drain/平面宽
    int64_t nAl = 0;       // CeilAlign(Hp,16)：L0C / B-tile 的 N 分形范围
    int64_t nSlice = 0;    // L0B 单 tile N 宽上限（= min(nL0c, l0b/(4*16))）
    int64_t kc = 0;        // K 分块行数（L0B 预算反推并受组宽钳位，16 倍数）
    int64_t nL0c = 0;      // 列片宽（L0C 片 = UB 平面宽 = drain 会合宽）
    int64_t sliced = 0;    // nAl > nL0c：多列片（bias 走共享槽按片现搬）
    int64_t mChunk = 0;    // 单块行数（UB×L0C 联合预算 + HBM 流量模型择优）
    int64_t cGroups = 0;   // split-K 总组数 = cGroupsX + CeilDiv(H,kgH)
    int64_t cGroupsX = 0;  // x 段组数
    int64_t kgX = 0;       // x 段组宽（16 对齐）
    int64_t kgH = 0;       // h 段组宽（16 对齐）
};
