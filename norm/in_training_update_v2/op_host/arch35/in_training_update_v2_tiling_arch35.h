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
 * \file in_training_update_v2_tiling_arch35.h
 * \brief INTrainingUpdateV2 tiling declarations for DAV_3510.
 */

#ifndef IN_TRAINING_UPDATE_V2_TILING_ARCH35_H
#define IN_TRAINING_UPDATE_V2_TILING_ARCH35_H

#include <array>
#include <cstddef>
#include <cstdint>
#include "register/op_impl_registry.h"
#include "../../op_kernel/arch35/in_training_update_v2_tiling_data.h"

namespace optiling {

struct INTrainingUpdateV2CompileInfo {
    int64_t coreNum = 0;
    int64_t ubSize = 0;
};

class INTrainingUpdateV2Tiling {
public:
    explicit INTrainingUpdateV2Tiling(gert::TilingContext* context) : context_(context) {}
    ge::graphStatus DoTiling();

private:
    static constexpr size_t OPTIONAL_GAMMA_SLOT = 0;
    static constexpr size_t OPTIONAL_BETA_SLOT = 1;
    static constexpr size_t OPTIONAL_MEAN_SLOT = 2;
    static constexpr size_t OPTIONAL_VARIANCE_SLOT = 3;
    static constexpr size_t OPTIONAL_SLOT_COUNT = 4;

    ge::graphStatus GetPlatformInfo();
    ge::graphStatus ParseAndValidate();
    ge::graphStatus ValidateRequiredInputs();
    ge::graphStatus ValidateOptionalInputs();
    ge::graphStatus ValidateOutputs();
    ge::graphStatus CalculateNormalTiling();
    ge::graphStatus FillTilingData();

    bool GetOptionalPresence(size_t index, const char* name, bool& present);
    bool ValidatePublicStatSelf(size_t index, const char* name, bool optional, ge::Format& format,
                                std::array<int64_t, 4>& dims);
    bool ValidateExactStat(size_t index, const char* name, bool optional);
    bool ValidateAffine(size_t index, const char* name, int64_t& batchStride);
    bool ReadShape(const gert::StorageShape* shape, const char* name, std::array<int64_t, 4>& dims) const;
    bool CheckOutput(size_t index, const char* name, ge::DataType dtype, ge::Format format,
                     const std::array<int64_t, 4>& dims) const;

    gert::TilingContext* context_ = nullptr;
    int64_t coreNum_ = 0;
    int64_t ubSize_ = 0;
    int64_t xDtypeSize_ = 0;
    ge::DataType xDtype_ = ge::DT_UNDEFINED;
    int64_t unitCount_ = 0;
    int64_t cTile_ = 0;
    ge::Format xFormat_ = ge::FORMAT_RESERVED;
    std::array<int64_t, 4> xDims_{};
    std::array<int64_t, 4> statDims_{};
    std::array<bool, OPTIONAL_SLOT_COUNT> optionalPresent_{};
    std::array<ge::Format, OPTIONAL_SLOT_COUNT> optionalFormats_{};
    std::array<std::array<int64_t, 4>, OPTIONAL_SLOT_COUNT> optionalDims_{};

    INTrainingUpdateV2TilingData data_{};
    uint64_t tilingKey_ = 0;
};

} // namespace optiling

#endif // IN_TRAINING_UPDATE_V2_TILING_ARCH35_H
