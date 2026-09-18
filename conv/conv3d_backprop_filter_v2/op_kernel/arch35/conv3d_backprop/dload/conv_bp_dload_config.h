/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CONV_BP_DLOAD_CONFIG_H
#define CONV_BP_DLOAD_CONFIG_H

namespace BpDLoad {

struct ShapeAttribute {
    uint16_t hk;
    uint16_t wk;
    uint16_t dk;
    uint16_t hPad;
    uint16_t wPad;
    uint16_t dPad;
    uint32_t batch;
    uint32_t cout;
    uint32_t dout;
    uint32_t hout;
    uint32_t wout;
    uint32_t cin;
    uint32_t din;
    uint32_t hin;
    uint32_t win;
};

struct DLoadTiling {
    // 传进来的基本块c轴要16byte对齐
    uint16_t singleShapeAligned16Cin;
    uint16_t singleShapeAligned16Cout;
    uint16_t kl0HoWo;
    // Mmad HF32 模式开关（Init 设置/End 关闭）
    bool hf32Flag = false;
};

struct DLoadConfig {
    ShapeAttribute shape;
    DLoadTiling tiling;
};

} // namespace BpDLoad
#endif // CONV_BP_DLOAD_CONFIG_H
