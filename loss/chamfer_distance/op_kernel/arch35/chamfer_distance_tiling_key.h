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
 * \file chamfer_distance_tiling_key.h
 * \brief ChamferDistance TilingKey 定义
 *
 * DTYPE_XYZ1 由 chamfer_distance_def.cpp 的输入 dtype profile 驱动, 框架按 dtype 自动注入并实例化,
 * 不进入 TilingKey; 本算子也没有独立于 dtype 的算法分支(单段/多段共用同一段代码),
 * 因此不声明任何 TilingKey 维度, host 侧固定 SetTilingKey(0)。
 */

#ifndef CHAMFER_DISTANCE_TILING_KEY_H
#define CHAMFER_DISTANCE_TILING_KEY_H

#endif // CHAMFER_DISTANCE_TILING_KEY_H
