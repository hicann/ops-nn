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
 * \file cube_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_CUBE_TILING_H_
#define OPS_BUILT_IN_OP_TILING_CUBE_TILING_H_

#include <vector>
#include <string>

#include <nlohmann/json.hpp>
#include "graph/operator.h"

namespace optiling {
/**
 * @brief: The struct that CubeTiling needs to save compile info from the info in json format.
 */
struct CubeTilingCommonParseInfo {
    int32_t fmapC1 = 0;
    bool correctRangeFlag = false;
    std::string tilingType = "";
    std::vector<std::string> varMap;
    std::vector<std::string> tilingKeyList;
    std::vector<std::vector<std::string>> customVarsList;
    std::vector<std::vector<int64_t>> defaultRangeList;
    std::vector<std::vector<int64_t>> tilingRangeList;
    std::vector<int32_t> blockDimList;
    std::vector<std::vector<int32_t>> repoSeedsList;
    std::vector<std::vector<int64_t>> repoRangeList;
    std::vector<std::vector<int64_t>> costRangeList;
};

/**
 * @brief: The class which stores input shape information of op.
 */
class InputShapeInfo {
public:
    InputShapeInfo(const std::vector<int64_t> &inputXShape, const std::string &inputXFormat)
            : xShape(inputXShape), xFormat(inputXFormat){}
    ~InputShapeInfo(){}
    std::vector<int64_t> xShape;
    std::string xFormat = "";
};

}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_CUBE_TILING_H_