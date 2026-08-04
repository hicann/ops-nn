/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SGD_TILING_DATA_H
#define SGD_TILING_DATA_H

/* !
 * \file sgd_tiling_data.h
 * \brief Host / Kernel 共享的 TilingData（普通 C++ struct，不用废弃宏 BEGIN_TILING_DATA_DEF）
 */

#include "atvoss/elewise/elewise_base_struct.h"

struct SgdRegbaseTilingData {
    Ops::Base::EleBaseTilingDataV2 elewiseTiling; // 框架填充：elemNum / ubFormer / blockNum / scheMode ...
    float dampening;                              // 属性注入，仅 hasDampening == 1 时经 sch.SetVar 下发
    float weightDecay;                            // 属性注入，仅 hasWeightDecay == 1 时经 sch.SetVar 下发
};

// 注：nesterov 是纯编译期分支（TilingKey 维度），不占 TilingData 字段。
//     learning_rate / momentum 是 Device 侧 [1] 张量，由 Placeholder::ScalarAttr<true>
//     从 GM 读取，【不得】在 Host 侧读值放进 TilingData。

#endif // SGD_TILING_DATA_H
