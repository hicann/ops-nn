/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file squared_relu_tiling_def.h
 * \brief squared_relu_tiling_def
 */
 #ifndef SQUARED_RELU_TILING_DEF_H
 #define SQUARED_RELU_TILING_DEF_H
 
 #include "register/tilingdata_base.h"
 #include "tiling/tiling_api.h"
 
 namespace optiling {
 struct SquaredReluCompileInfo {};
 
 BEGIN_TILING_DATA_DEF(SquaredReluTilingData)
 
 TILING_DATA_FIELD_DEF(int64_t, elementNum);
 TILING_DATA_FIELD_DEF(uint32_t, needCoreNum);
 END_TILING_DATA_DEF;
 
 REGISTER_TILING_DATA_CLASS(SquaredRelu, SquaredReluTilingData)
 }  // namespace optiling
 
 #endif