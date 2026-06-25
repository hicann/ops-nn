/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file threshold_v2_tiling_arch35.h
 * \brief
 */

#ifndef OP_HOST_ARCH35_THRESHOLD_H_
#define OP_HOST_ARCH35_THRESHOLD_H_

#include "op_host/tiling_base.h"

namespace optiling {
using namespace Ops::NN::Optiling;

struct ThresholdCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

class ThresholdTiling {
public:
    explicit ThresholdTiling(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus RunTiling();
    ge::graphStatus CalcOutputDtype();
    ge::graphStatus CalcInputDtype();
    ge::graphStatus CheckShape();

private:
    template <typename K, typename V>
    V GetOrDefault(const std::unordered_map<K, V>& map, const K& key, const V& def) {
        auto it = map.find(key);
        return it != map.end() ? it->second : def;
    }
    
    uint64_t dType = 0;
    bool hasValue = true;
    gert::TilingContext *tilingContext;
    ge::DataType outputDtype = ge::DT_UNDEFINED;
    ge::DataType inputDtype = ge::DT_UNDEFINED;
};
} // namespace optiling

#endif // OP_HOST_ARCH35_THRESHOLD_H_
