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
 * \file gemmv3_tiling.h
 * \brief GemmV3 tiling, inherits MatMulV3Tiling.
 */
#pragma once

#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_tiling_advanced.h"
#include "gemmv3_tiling_key.h"

namespace optiling {
namespace gemmv3 {
using namespace matmul_v3_advanced;
class GemmV3Tiling : public MatMulV3Tiling {
public:
    explicit GemmV3Tiling(gert::TilingContext* context) : MatMulV3Tiling(context) {};

    ~GemmV3Tiling() override = default;

protected:
    ge::graphStatus ValidateInputsNotNull() override;
    ge::graphStatus DetectOptionalInputs() override;
    void ExtractAttrFlags() override;
    ge::graphStatus ExtractTranspose() override;
    ge::graphStatus ExtractMKN() override;
    ge::graphStatus ValidateShape() override;
    ge::graphStatus ValidateBias() override;
    ge::graphStatus ValidateOpSpecific() override;
    ge::graphStatus ValidateDtype() override;
    std::vector<int32_t> GetRegistryPriorities(NpuArch npuArch) const override;
    MatMulV3TilingKey* GetTilingKeyObj() override;
    std::vector<std::vector<ge::DataType>> GetDtypeSupportList() const override;

private:
    GemmV3TilingKey gemmV3TilingKey_{};
};
} // namespace gemmv3
} // namespace optiling
