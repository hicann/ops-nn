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
 * \file ifmr_tiling.h
 * \brief
 */

#ifndef IFMR_TILING_H
#define IFMR_TILING_H

#include <cstdint>
#include <string>
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "quant/ifmr/op_kernel/ifmr_tiling_data.h"

namespace optiling {
    constexpr int ATTR_MIN_PERCENTILE_INDEX = 0;
    constexpr int ATTR_MAX_PERCENTILE_INDEX = 1;
    constexpr int ATTR_SEARCH_RANGE_INDEX = 2;
    constexpr int ATTR_SEARCH_STEP_INDEX = 3;
    constexpr int ATTR_WITH_OFFSET_INDEX = 4;
    constexpr int ATTR_QUANT_BITS_INDEX = 5;

    constexpr int ATTR_SEARCH_RANGE_SIZE = 2;

    constexpr float K_PERCENTILE_LOW_BOUND = 0.5f;
    constexpr float K_PERCENTILE_UPPER_BOUND = 1.0f;

    constexpr uint32_t DATA_INPUT_INDEX = 0;
    constexpr uint32_t DATA_MIN_INPUT_INDEX = 1;
    constexpr uint32_t DATA_MAX_INPUT_INDEX = 2;
    constexpr uint32_t CUMSUM_INPUT_INDEX = 3;
    constexpr uint32_t SHAPE_SIZE_LIMIT = 2147483648; //pow(2, 31);
    constexpr uint32_t MAX_CUMSUM_LENGTH = 8192;
    constexpr uint32_t MAX_STEP_NUMS = 4096;
    constexpr uint32_t SUPPORTED_QUANT_BITS[2] = {8, 16};
    constexpr uint32_t SUPPORT_QUANT_BITS_NUM = 2;
    constexpr uint64_t UB_SIZE_RESERVE = 133280;

    struct IfmrAttrs {
        float minPercentile;
        float maxPercentile;
        float searchRange[2];
        float searchStep;
        bool withOffset;
        int quantBits;
        uint32_t dataLength;
        uint32_t cumsumLength;
    };

    class IfmrTiling {
    public:
        explicit IfmrTiling(gert::TilingContext* context) : context_(context), nodeName_(context->GetNodeName()) {}
        ~IfmrTiling() {} 
        ge::graphStatus IfmrTilingFunc(void);

    protected:
        gert::TilingContext* context_ = nullptr;
        const ge::char_t* nodeName_;

    private:
        ge::graphStatus GetIfmrTilingAttrInfo(void);
        ge::graphStatus CheckIfmrTilingAttrs(void);
        ge::graphStatus CheckIfmrTilingInputDtype(void);
        ge::graphStatus CheckIfmrTilingInputDataShape(uint32_t inputIndex, std::string inputName);
        ge::graphStatus GetIfmrTilingInputInfo(void);
        ge::graphStatus GetDataLength(void);
        ge::graphStatus CheckIfmrTilingOutputInfo(void);
        void SetIfmrTiling(void);
        void PostTiling(void);

        IfmrAttrs attrs_;
    };

}  // namespace optiling
#endif  // IFMR_TILING_H