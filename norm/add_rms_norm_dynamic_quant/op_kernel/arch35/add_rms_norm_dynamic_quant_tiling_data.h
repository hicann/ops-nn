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
 * \file add_rms_norm_dynamic_quant_tiling_data.h
 * \brief
 */
#ifndef ADD_RMS_NORM_DYNAMIC_QUANT_TILING_DATA_H_
#define ADD_RMS_NORM_DYNAMIC_QUANT_TILING_DATA_H_

struct AddRmsNormDynamicQuantRegbaseTilingData {
    uint64_t numM;
    uint64_t numN;
    uint64_t baseM;
    uint64_t baseN;
    uint64_t baseNDtypeAlign;
    uint64_t baseNReduceAlign;
    uint64_t powerSplit;
    uint64_t powerLoop;
    uint64_t mPerCore;
    uint64_t mLastCore;
    float epsilon;
    float avgFactor;
    uint32_t hasSmoothScale1;
    uint32_t hasSmoothScale2;
    uint32_t hasBeta;
    uint32_t outQuant1Flag;
    uint32_t outQuant2Flag;
};

struct AddRmsNormDynamicQuantEmptyTilingData {
    uint64_t numM;
    uint64_t hasSmoothScale1;
    uint64_t hasSmoothScale2;
    uint64_t usedCoreNum;
    uint64_t mPerCore;
    uint64_t mLastCore;
    uint64_t ubSize;
    uint64_t mPerUB;
    uint64_t coreUbBlockCount;
    uint64_t lastCoreBlockCount;
    uint64_t mTailUb;
    uint64_t mlastCoreTailUb;
    uint32_t outQuant1Flag;
    uint32_t outQuant2Flag;
};

#endif
