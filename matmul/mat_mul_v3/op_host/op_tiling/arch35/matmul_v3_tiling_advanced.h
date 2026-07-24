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
 * \file matmul_v3_tiling_advanced.h
 * \brief MatMulV3Tiling base class for MatMulV3-family op tiling.
 *        Provides a clean phased interface (Extract / Validate / Detect)
 *        with backward-compatible virtual methods for existing derived classes.
 */
#pragma once

#include <vector>
#include "runtime/tiling_context.h"
#include "matmul_v3_common_advanced.h"
#include "matmul_v3_tiling_key.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace matmul_v3_advanced {
class MatMulV3Tiling {
public:
    explicit MatMulV3Tiling(gert::TilingContext* context) : context_(context) {};
    virtual ~MatMulV3Tiling() = default;
    virtual ge::graphStatus DoTiling();

protected:
    // ====== Phase 1: Context initialization ======
    virtual ge::graphStatus InitContext();

    // ====== Phase 2: Input null-check ======
    virtual ge::graphStatus ValidateInputsNotNull();

    // ====== Phase 3: Optional input detection ======
    virtual void DetectOptionalInputs();

    // ====== Phase 4: Format extraction ======
    virtual void ExtractFormat();

    // ====== Phase 5: Dtype & attr-flag extraction ======
    virtual void ExtractDtype();
    virtual void ExtractAttrFlags();

    // ====== Phase 6: Shape extraction ======
    virtual ge::graphStatus ExtractTranspose();
    virtual ge::graphStatus ExtractMKN();

    // ====== Phase 7: Validation ======
    virtual ge::graphStatus ValidateFormat();
    virtual ge::graphStatus ValidateShape();
    virtual ge::graphStatus ValidateBias();
    virtual ge::graphStatus ValidateDtype();
    virtual ge::graphStatus ValidateOpSpecific();

    // ====== Phase 8: Batch info extraction (optional, base = no-op) ======
    virtual ge::graphStatus ExtractBatchInfo() { return ge::GRAPH_SUCCESS; }

    // ====== Phase 9: Post-batch validation (optional, base = no-op) ======
    virtual ge::graphStatus PostBatchInfoCheck() { return ge::GRAPH_SUCCESS; }

    // ====== Phase 10: Registry delegation hooks ======
    virtual const char* GetRegistryOpType() const { return "MatMulV3"; }
    virtual std::vector<int32_t> GetRegistryPriorities(NpuArch npuArch) const;
    virtual MatMulV3TilingKey* GetTilingKeyObj() { return nullptr; }

    // ====== Dtype support list (subclass provides) ======
    virtual std::vector<std::vector<ge::DataType>> GetDtypeSupportList() const;

    // ====== Old interface (backward compatibility, delegates to new phases) ======
    virtual ge::graphStatus GetShapeAttrsInfo();
    virtual ge::graphStatus GetArgs();
    virtual ge::graphStatus CheckArgs();
    virtual void GetFormat();
    virtual ge::graphStatus GetShape();
    virtual ge::graphStatus BaseOpSpecificCheck();

protected:
    gert::TilingContext* context_ = nullptr;
    MatMulV3Args args_;
    bool isSelfSlice_ = false;
    int64_t kBValue_ = 0; // B 矩阵的 K 维度，ExtractMKN 提取后供 ValidateShape 校验

private:
    // 非连续 shape 提取的内部路由，MatMulV3 专有实现细节，不作为派生类扩展点
    bool ExtractNonContiguousDims(int64_t (&mkDims)[2], int64_t (&knDims)[2]);
    ge::graphStatus ExtractSliceDims(int64_t (&dims)[2]);
    ge::graphStatus ExtractTransposeDims(int64_t (&dims)[2], int64_t idx);
    ge::graphStatus ExtractNormalDims(const gert::Shape& storageShape, const gert::Shape& oriShape, uint64_t dtypeSize,
                                      ge::Format format, int64_t (&dims)[2], const char* paramName);
    NpuArch arch_ = NpuArch::DAV_3510;
};
} // namespace matmul_v3_advanced
} // namespace optiling
