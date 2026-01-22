/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_batch_matmul_v3_checker_for_mmads8s4.h
 * \brief
 */
#ifndef QUANT_BATCH_MATMUL_V3_CHECKER_FOR_MMADS8S4_H
#define QUANT_BATCH_MATMUL_V3_CHECKER_FOR_MMADS8S4_H
#include "quant_batch_matmul_v3_checker_base.h"
#include <cstdint>
#include <vector>
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/shape.h"

namespace optiling {

class QuantBatchMatmulV3Checker4MmadS8S4 : public QuantBatchMatmulV3CheckerBase {
public:
    QuantBatchMatmulV3Checker4MmadS8S4(gert::TilingContext *context, const QuantBatchMatmulInfo &inputParams)
        : QuantBatchMatmulV3CheckerBase(context, inputParams)
    {
    }

    ~QuantBatchMatmulV3Checker4MmadS8S4() override = default;
    bool CheckDtype() const override;
    bool CheckShape(const std::vector<gert::Shape *> &mandtoryShape, const gert::StorageShape *biasShape,
                    const gert::StorageShape *pertokenShape, const std::vector<int64_t> &dimValueOfMKN) const override;

protected:
    virtual bool CheckDtypesInRange() const;
    virtual bool CheckABDtypes() const;

    bool CheckShapeInRangeForOptionalInputs(const gert::StorageShape *biasShape,
                                            const gert::StorageShape *offsetShape) const;
    bool CheckBias(const gert::StorageShape* biasShape) const;
    virtual bool CheckOffset(const gert::StorageShape* offsetShape) const;
    bool CheckDimValue(const gert::Shape &scaleShape, const gert::StorageShape *biasShape,
                       const gert::StorageShape *offsetShape, const std::vector<int64_t> &dimValueOfMKN) const;
    bool CheckShapeInBoundary(const gert::Shape &shape, uint32_t shapeIdx) const;
    virtual bool ExtraInputCheck() const;
};
}  // namespace optiling
#endif  // QUANT_BATCH_MATMUL_V3_CHECKER_FOR_MMADS8S4_H