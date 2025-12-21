/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file modulategrad_tiling.h
 * \brief
 */
#ifndef MODULATE_GRAD_TILING_H
#define MODULATE_GRAD_TILING_H
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling{
    BEGIN_TILING_DATA_DEF(ModulateGradTiling)
        TILING_DATA_FIELD_DEF(uint32_t,B);
        TILING_DATA_FIELD_DEF(uint32_t,L);
        TILING_DATA_FIELD_DEF(uint32_t,D);
        TILING_DATA_FIELD_DEF(uint32_t,dataType);
        TILING_DATA_FIELD_DEF(uint32_t,block_dim);
        TILING_DATA_FIELD_DEF(uint32_t,dataTypeSize);
        TILING_DATA_FIELD_DEF(uint32_t,splitB);
        TILING_DATA_FIELD_DEF(uint32_t,coresPerB);
        TILING_DATA_FIELD_DEF(uint32_t,usedCores);
        TILING_DATA_FIELD_DEF(uint32_t,formerNum);
        TILING_DATA_FIELD_DEF(uint32_t,formerLength);
        TILING_DATA_FIELD_DEF(uint32_t,tailNum);
        TILING_DATA_FIELD_DEF(uint32_t,tailLength);
        TILING_DATA_FIELD_DEF(uint32_t,has_scale);
        TILING_DATA_FIELD_DEF(uint32_t,has_shift);
    
    END_TILING_DATA_DEF;

    REGISTER_TILING_DATA_CLASS(ModulateGrad, ModulateGradTiling)
    struct ModulateGradCompileInfo{
        uint32_t totalCoreNum = 0;
        uint64_t ubSizePlatForm = 0;

    };
 
}
#endif