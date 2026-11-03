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
 * \file modulate_grad_struct.h
 * \brief modulate_grad regbase tiling data (arch35 / ascend950)
 */

#ifndef MODULATE_GRAD_STRUCT_H
#define MODULATE_GRAD_STRUCT_H

#pragma pack(push, 8)
struct ModulateGradRegbaseTilingData {
    uint64_t B;
    uint64_t L;
    uint64_t D;
    uint32_t coresPerB;
    uint32_t usedCores;
    uint32_t formerNum;
    uint32_t formerLength;
    uint32_t tailNum;
    uint32_t tailLength;
    // 运行时可选存在性(与模板参数一致;作为在线编译变体选择异常时的安全网,
    // 防止误选变体读写缺席可选参数)
    uint32_t has_scale;
    uint32_t has_shift;
    // 运行时切分策略(与模板参数一致,同为安全网)
    uint32_t split_by_d;
};
#pragma pack(pop)

#endif
