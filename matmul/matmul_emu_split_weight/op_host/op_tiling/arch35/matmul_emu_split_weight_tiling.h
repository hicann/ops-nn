/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "op_host/tiling_base.h"
#include "../../../op_kernel/matmul_emu_split_weight_tiling_data.h"

namespace optiling {
namespace matmul_emu_split_weight {

class MatmulEmuSplitWeightTiling : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit MatmulEmuSplitWeightTiling(gert::TilingContext* context) : TilingBaseClass(context) {}
    ~MatmulEmuSplitWeightTiling() override = default;

protected:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

private:
    // Input null-check
    ge::graphStatus ValidateInputsNotNull();

    // Extract phases
    void ExtractDtype();
    void ExtractFormat();
    ge::graphStatus ExtractAttrs();
    ge::graphStatus ExtractShape();

    // Validate phases
    ge::graphStatus ValidateDtype() const;
    ge::graphStatus ValidateFormat() const;
    ge::graphStatus ValidateShape() const;
    ge::graphStatus ValidateAttrs() const;

    // Tiling calculation phases
    void CalcBaseMN();
    void CalcBasicBlock();
    void CalcBaseK();
    void CalcKL1();
    void CalcCoreNum();
    void SetTilingData();

    uint64_t m_{0};
    uint64_t n_{0};
    uint64_t k_{0};
    uint64_t wHighK_{0};
    uint64_t wLowK_{0};
    uint64_t wLowN_{0};
    float scale_{0.00390625f};
    bool transX_{false};
    bool transW_{false};
    int32_t yDtype_{0};
    uint64_t aicNum_{0};
    uint64_t aivNum_{0};
    uint64_t l1Size_{0};
    uint64_t l0aSize_{0};
    uint64_t l0bSize_{0};
    uint64_t l0cSize_{0};
    uint64_t ubSize_{0};
    uint64_t tilingKey_{0};
    uint64_t baseM_{0};
    uint64_t baseN_{0};
    uint64_t baseK_{0};
    uint64_t kL1_{0};
    uint64_t usedCoreNum_{0};
    MatmulEmuSplitWeightTilingData tilingData_{};
};

} // namespace matmul_emu_split_weight
} // namespace optiling
