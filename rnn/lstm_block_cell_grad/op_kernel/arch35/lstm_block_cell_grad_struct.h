/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * Ascend C template parameter (TPL) declarations for LSTMBlockCellGrad on
 * arch35 — the tilingKey template-parameter entry dispatch declaration.
 *
 * Two TPL parameters drive the inner (tilingKey) enumeration:
 *   - DTYPE (UINT, 2 bit, bit[0..1]): OpDef-registered dtype combo index
 *     (值=下标): 0 = float32, 1 = float16; enum values 2/3 are outside the
 *     table → INVALID_TILING_KEY (reserved encodings).
 *   - USE_PEEPHOLE (BOOL, 1 bit, bit[2]): the use_peephole attr.
 *
 * ASCENDC_TPL_SEL enumerates all 4 legal combinations; the encoded tilingKey
 * of each row is the sub-kernel suffix `_N`:
 *   (FP32, false) → 0    (FP32, true) → 4
 *   (FP16, false) → 1    (FP16, true) → 5
 */

#ifndef LSTM_BLOCK_CELL_GRAD_STRUCT_H_
#define LSTM_BLOCK_CELL_GRAD_STRUCT_H_

#include "ascendc/host_api/tiling/template_argument.h"

// DTYPE 枚举值与 OpDef 注册的 dtype 组合表下标一致（.DataType({DT_FLOAT, DT_FLOAT16})，无 BF16）。
#define LSTM_BLOCK_CELL_GRAD_DTYPE_FP32 0 // ge::DT_FLOAT    （全 21 张量 float32）
#define LSTM_BLOCK_CELL_GRAD_DTYPE_FP16 1 // ge::DT_FLOAT16  （全 21 张量 float16）

// 参数 1 DTYPE：UINT 枚举（2 值，位宽 2 bit 占 bit[0..1]；枚举值 2/3 在表外 →
//               INVALID_TILING_KEY，编码空间保留值）
// 参数 2 USE_PEEPHOLE：BOOL（1 bit，占 bit[2]），窥孔模式
// —— 入口模板参数与 tilingKey 位段一一对应且顺序一致。
ASCENDC_TPL_ARGS_DECL(LSTMBlockCellGrad,
                      ASCENDC_TPL_UINT_DECL(DTYPE, 2, ASCENDC_TPL_UI_LIST, LSTM_BLOCK_CELL_GRAD_DTYPE_FP32,
                                            LSTM_BLOCK_CELL_GRAD_DTYPE_FP16),
                      ASCENDC_TPL_BOOL_DECL(USE_PEEPHOLE, 0, 1));

// 4 个组合全部实例化（2 dtype × 2 peephole，无设计性空位）；
// 每行编码出的 tilingKey：_0 / _4 / _1 / _5（sub-kernel 后缀）。
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(DTYPE, ASCENDC_TPL_UI_LIST, LSTM_BLOCK_CELL_GRAD_DTYPE_FP32),
                                     ASCENDC_TPL_BOOL_SEL(USE_PEEPHOLE, 0)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(DTYPE, ASCENDC_TPL_UI_LIST, LSTM_BLOCK_CELL_GRAD_DTYPE_FP32),
                                     ASCENDC_TPL_BOOL_SEL(USE_PEEPHOLE, 1)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(DTYPE, ASCENDC_TPL_UI_LIST, LSTM_BLOCK_CELL_GRAD_DTYPE_FP16),
                                     ASCENDC_TPL_BOOL_SEL(USE_PEEPHOLE, 0)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(DTYPE, ASCENDC_TPL_UI_LIST, LSTM_BLOCK_CELL_GRAD_DTYPE_FP16),
                                     ASCENDC_TPL_BOOL_SEL(USE_PEEPHOLE, 1)));

#endif // LSTM_BLOCK_CELL_GRAD_STRUCT_H_
