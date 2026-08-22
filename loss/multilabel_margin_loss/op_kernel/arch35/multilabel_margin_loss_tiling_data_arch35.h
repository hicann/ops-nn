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
 * \file multilabel_margin_loss_tiling_data.h
 * \brief MultilabelMarginLoss arch35 tiling data struct (plain struct, shared by host tiling and kernel)
 *
 * 与 A2 那份(op_kernel/multilabel_margin_loss_tiling_data.h)完全独立:A2 的结构体一个字节不动。
 * ubFactor / wsCoreStride 由 host 侧实算下发,kernel 不再自行推导任何 buffer 尺寸。
 */

#ifndef MULTILABEL_MARGIN_LOSS_ARCH35_TILING_DATA_H
#define MULTILABEL_MARGIN_LOSS_ARCH35_TILING_DATA_H

#include <cstdint>

struct MultilabelMarginLossArch35TilingData {
    uint32_t N = 0;           // batch 行数
    uint32_t C = 0;           // 每行标签数
    uint32_t basePerCore = 0; // 每核基础行数
    uint32_t pivot = 0;       // 前 pivot 个核各多算一行
    uint32_t usedCoreNum = 1; // 实际使用核数(== blockDim)
    int32_t reduction = 0;    // 0=none, 1=mean, 2=sum
    // 每轮 UB 处理的元素数(已按 32B 对齐)。逐行损失的暂存与回读都按它分块,故 UB 占用与 N 解耦。
    // host 由 GetCoreMemSize(UB) 扣除按 C 缩放开销与固定开销后算得:高估开销只会缩小它(安全方向),
    // 低估才会撑爆 UB;UB 不足时 tiling 直接报错,不让 kernel 静默越界。
    uint32_t ubFactor = 0;
    // mean/sum 每核独占的工作区槽位跨步(float 个数,32B 对齐)。相邻核不共享 32B GM 写块,
    // 因此无需原子加 —— 原子加的跨核合并顺序不定,同一输入会跑出不同结果。
    uint32_t wsCoreStride = 0;
    // 每轮 UB 处理的 C 方向元素数。整行(C)装得下 UB 时 == C(kernel 走全行路径, 行为与此前一致);
    // 装不下时 host 解出能装下的分块长度, kernel 走 C 分块路径(语义相同, 只是把一行拆成若干段)。
    // issue #32: 此前没有这一维, C 超过约 8000 就直接 GRAPH_FAILED, 支持面窄于 A2。
    uint32_t cFactor = 0;
};

#endif // MULTILABEL_MARGIN_LOSS_ARCH35_TILING_DATA_H
