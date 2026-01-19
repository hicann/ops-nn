/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file fused_quant_matmul_checker.cpp
 * \brief
 */
#include "fused_quant_matmul_checker.h"

#include <map>

#include "../../../../common/op_host/op_tiling/tiling_type.h"
#include "error_util.h"
#include "log/log.h"
#include "op_util.h"

namespace optiling {

bool FusedQuantMatMulChecker::CheckShape(const std::vector<gert::Shape *> &mandtoryShape,
                                         const std::vector<const gert::StorageShape *> &optionalInputShape,
                                         const std::vector<int64_t> &dimValueOfMKN) const {
    // mandtoryShape = [x1Shape, x2Shape]
    // optionalInputShape = [biasShape, scaleShape]
    OP_TILING_CHECK(mandtoryShape.size() < 2 || optionalInputShape.size() < 2,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "check input is illegal."), return false);

    auto &x1Shape = *mandtoryShape[0]; // using index 0 to get x1Shape
    auto &x2Shape = *mandtoryShape[1]; // using index 1 to get x2Shape

    auto biasShape = optionalInputShape[0];     // using index 0 to get biasShape
    auto scaleShape = optionalInputShape[1];    // using index 1 to get x2ScaleShape
    auto offsetShape = context_->GetOptionalInputShape(GetOffsetIdx());

    if (!CheckShapeInRangeForOptionalInputs(scaleShape, biasShape, offsetShape) ||
        !CheckDimValue(scaleShape, biasShape, offsetShape, dimValueOfMKN) ||
        !CheckShapeInBoundary(x1Shape, GetX1Idx()) || !CheckShapeInBoundary(x2Shape, GetX2Idx())) {
        return false;
    }
    if (!ExtraInputCheck()) {
        return false;
    }
    return true;
}

bool FusedQuantMatMulChecker::CheckDimValue(const gert::StorageShape *scaleShape, const gert::StorageShape *biasShape,
                                            const gert::StorageShape *offsetShape,
                                            const std::vector<int64_t> &dimValueOfMKN) const {
    auto x2Inner = dimValueOfMKN[2]; // using index 2 to get x2Inner
    auto x2Outer = dimValueOfMKN[3]; // using index 3 to get x2Outer
    // 检查kA是否等于kB
    auto kBSize = static_cast<uint64_t>(inputParams_.transB ? x2Inner : x2Outer);
    OP_TILING_CHECK(
        inputParams_.kSize != kBSize,
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "kA[%lu] is not equal to kB[%lu]", inputParams_.kSize, kBSize),
        return false);

    // scaleShapescale shape必须是1或nSize
    if (scaleShape != nullptr) {
        OP_TILING_CHECK(!(scaleShape->GetStorageShape().GetDim(0) == 1 ||
                          static_cast<uint64_t>(scaleShape->GetStorageShape().GetDim(0)) == inputParams_.nSize),
                        CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                              "The scale dimension value must be 1 or n[%lu], but it is %lu.",
                                              inputParams_.nSize, offsetShape->GetStorageShape().GetDim(0)),
                        return false);
    }

    // bias的shape等于nSize
    OP_TILING_CHECK(
        biasShape != nullptr && static_cast<uint64_t>(biasShape->GetStorageShape().GetDim(0)) != inputParams_.nSize,
        CUBE_INNER_ERR_REPORT(inputParams_.opName,
                              "The biasShape of FusedQuantMatMul should equal n, but it is %zu while n is %lu.",
                              biasShape->GetStorageShape().GetDim(0), inputParams_.nSize),
        return false);

    // offsetShape必须是1或nSize
    if (offsetShape != nullptr) {
        OP_TILING_CHECK(!(offsetShape->GetStorageShape().GetDim(0) == 1 ||
                          static_cast<uint64_t>(offsetShape->GetStorageShape().GetDim(0)) == inputParams_.nSize),
                        CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                              "offset dim value must be 1 or n[%lu], but it is %lu", inputParams_.nSize,
                                              offsetShape->GetStorageShape().GetDim(0)),
                        return false);
    }
    return true;
}

bool FusedQuantMatMulChecker::CheckShapeInRangeForOptionalInputs(const gert::StorageShape *scaleShape,
                                                                 const gert::StorageShape *biasShape,
                                                                 const gert::StorageShape *offsetShape) const {
    // scale维数必须是1维
    if (scaleShape != nullptr) {
        auto scaleDimNum = scaleShape->GetStorageShape().GetDimNum();
        OP_TILING_CHECK(
            scaleDimNum != 1,
            CUBE_INNER_ERR_REPORT(inputParams_.opName, "x2Scale's dimension must be 1, actually is : %lu", scaleDimNum),
            return false);
    }

    // bias维数只能是1维
    if (biasShape != nullptr) {
        auto biasDimNum = biasShape->GetStorageShape().GetDimNum();
        OP_TILING_CHECK(biasDimNum != 1,
                        CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                              "The bias dimension should equal to 1, but it is %lu.", biasDimNum),
                        return false);
    }

    if (offsetShape != nullptr) {
        // 当outDtype不为INT8时，offset不存在
        OP_TILING_CHECK(
            inputParams_.cDtype != ge::DT_INT8,
            CUBE_INNER_ERR_REPORT(inputParams_.opName, "When outputDtype is not INT8, x2Offset must be null"),
            return false);
        // offset维数只能是1维
        OP_TILING_CHECK(offsetShape->GetStorageShape().GetDimNum() != 1,
                        CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                              "The x2Offset shape should be 1 dimension, but it is %lu",
                                              offsetShape->GetStorageShape().GetDimNum()),
                        return false);
    }
    return true;
}

} // namespace optiling