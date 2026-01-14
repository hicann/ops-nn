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
 * \file conv_api_tiling_algorithm_base.h
 * \brief
 */
#ifndef ASCENDC_TILING_CONV_API_TILING_ALGORITHM_BASE_H
#define ASCENDC_TILING_CONV_API_TILING_ALGORITHM_BASE_H

#include "conv_api_tiling_base.h"

namespace conv_tiling {
enum class L1TilingMode {
    FULL_LOAD_BL1 = 0,
    FULL_LOAD_AL1,
    ALL_FULL_LOAD,
    NONE_FULL_LOAD,
    INVALID
};

struct PBufferParams {
    uint8_t pbAL1 = 1;
    uint8_t pbBL1 = 1;
    uint8_t pbAL0 = 1;
    uint8_t pbBL0 = 1;
    uint8_t pbCL0 = 1;
    uint64_t pBufferFlag = 0;
};

class ConvTilingAlgorithmBase {
public:
    explicit ConvTilingAlgorithmBase(ConvTilingBase *tilingIns);
    virtual ~ConvTilingAlgorithmBase() = default;
    virtual int64_t Process() = 0;

protected:
    virtual void SetL1TilingRes() = 0;
    virtual void SetL0TilingRes() = 0;

    uint64_t CalcAL0Size(uint64_t mL0, uint64_t kL0) const;
    uint64_t CalcBL0Size(uint64_t kL0, uint64_t nL0) const;
    uint64_t CalcCL0Size(uint64_t mL0, uint64_t nL0) const;
    uint64_t CalcBTSize(uint64_t nL0) const;
    uint64_t CalcFBSize(uint64_t nL0) const;
    uint64_t InferHiL1(uint64_t hoL1, int64_t hi) const;
    void SetPBufferRes();
    void PrintRanges(std::vector<uint64_t> inputRanges, std::string rangeName) const;
    bool CheckL0Buffer(uint64_t currmL0, uint64_t currkL0, uint64_t currnL0);
    ConvTilingBase* tilingIns_ = nullptr;
    PBufferParams dbValue;
    uint64_t fMapDTypeSize = 0;
    uint64_t weightDTypeSize = 0;
    uint64_t biasDTypeSize = 0;
    uint64_t scaleDtypeSize = 0;
};
} // namespace conv_tiling

#endif // ASCENDC_TILING_CONV_API_TILING_ALGORITHM_BASE_H