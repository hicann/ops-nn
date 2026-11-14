/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QUANT_MAX_V2_STRUCT_H
#define QUANT_MAX_V2_STRUCT_H

#include "ascendc/host_api/tiling/template_argument.h"

// TilingKey 模板参数取值域（Host 侧 CalcTilingKey 按 roundMode/dstType/输入形态拼出该值，
// Kernel 侧入口据此编译期分发）。三个分量共 4bit，组合出 3×2×2 = 12 个 kernel 变体。
// 舍入模式：由属性 roundMode 与 dstType 匹配推导（HiF8 支持 round/hybrid，FP8 仅 rint）。
#define TPL_ROUND_MODE_RINT 0
#define TPL_ROUND_MODE_ROUND 1
#define TPL_ROUND_MODE_HYBRID 2
// 量化模式：scale 有无决定——传 scale 为静态（单趟融合量化+max），不传为动态（两趟：先 amax 后量化）。
#define TPL_QUANT_MODE_STATIC 0
#define TPL_QUANT_MODE_DYNAMIC 1
// 分组模式：group_list 有无决定——不传为 Per-Tensor（G=1 单段），传为 Per-Group（逐组簿记+跨核归并）。
#define TPL_GROUP_MODE_TENSOR 0
#define TPL_GROUP_MODE_GROUP 1

namespace QuantMaxV2Op {

// 声明算子的模板参数布局：roundMode(2bit)+quantMode(1bit)+hasGroup(1bit) 依次拼进 TilingKey，
// 并登记每个分量的合法取值列表（编译期按取值裁剪无效组合）。Host 侧写 key、Device 侧读 key 均以本布局为准。
ASCENDC_TPL_ARGS_DECL(QuantMaxV2,
                      ASCENDC_TPL_UINT_DECL(roundMode, 2, ASCENDC_TPL_UI_LIST, TPL_ROUND_MODE_ROUND,
                                            TPL_ROUND_MODE_RINT, TPL_ROUND_MODE_HYBRID),
                      ASCENDC_TPL_UINT_DECL(quantMode, 1, ASCENDC_TPL_UI_LIST, TPL_QUANT_MODE_STATIC,
                                            TPL_QUANT_MODE_DYNAMIC),
                      ASCENDC_TPL_UINT_DECL(hasGroup, 1, ASCENDC_TPL_UI_LIST, TPL_GROUP_MODE_TENSOR,
                                            TPL_GROUP_MODE_GROUP));

// 与上方 ARGS_DECL 对应的取值选择器：按同样的分量顺序与合法取值列表，从 TilingKey 反解出
// 各模板实参（kernel 入口的模板形参 RoundMode/QuantMode/HasGroup 即由此选中）。
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(roundMode, ASCENDC_TPL_UI_LIST, TPL_ROUND_MODE_ROUND, TPL_ROUND_MODE_RINT,
                         TPL_ROUND_MODE_HYBRID),
    ASCENDC_TPL_UINT_SEL(quantMode, ASCENDC_TPL_UI_LIST, TPL_QUANT_MODE_STATIC, TPL_QUANT_MODE_DYNAMIC),
    ASCENDC_TPL_UINT_SEL(hasGroup, ASCENDC_TPL_UI_LIST, TPL_GROUP_MODE_TENSOR, TPL_GROUP_MODE_GROUP)));
} // namespace QuantMaxV2Op

#endif // QUANT_MAX_V2_STRUCT_H
