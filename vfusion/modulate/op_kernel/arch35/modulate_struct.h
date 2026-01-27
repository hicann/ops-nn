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
 * \file modulate_struct.h
 * \brief modulate tiling data
 */

#ifndef MODULATE_STRUCT_H
#define MODULATE_STRUCT_H

#pragma pack(push, 8)
struct ModulateRegbaseTilingData {
    uint64_t inputB;
    uint64_t inputL;
    uint64_t inputD;
    uint64_t formerCoreNum;
    uint64_t tailCoreNum;
    uint64_t formerDataNum;
    uint64_t tailDataNum;
    uint64_t maxDInUB;
    uint64_t maxCalcNum;
    uint64_t maxCopyRows;
};
#pragma pack(pop)

#endif