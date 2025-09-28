/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file advance_step_tiling_def.h
 * \brief
 */

#ifndef ADVANCE_STEP_TILING_DEF_H
#define ADVANCE_STEP_TILING_DEF_H

#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AdvanceStepTilingData)
TILING_DATA_FIELD_DEF(int64_t, needCoreNum);
TILING_DATA_FIELD_DEF(int64_t, blockTablesStride); // seq_lens_maxblock
TILING_DATA_FIELD_DEF(int64_t, numSeqs);
TILING_DATA_FIELD_DEF(int64_t, numQueries);
TILING_DATA_FIELD_DEF(int64_t, blockSize);
TILING_DATA_FIELD_DEF(int64_t, perCoreSeqs);
TILING_DATA_FIELD_DEF(int64_t, lastCoreSeqs);
TILING_DATA_FIELD_DEF(int64_t, perLoopMaxSeqs);
TILING_DATA_FIELD_DEF(int64_t, specNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AdvanceStep, AdvanceStepTilingData)

constexpr uint16_t MAX_CORE_COUNT = 48;

struct AdvanceStepCompileInfo {};

enum class UB_TILING_POLICY : uint32_t
{
    ONE_CORE_NO_SPEC = 1,
    FULL_LOAD = 2
};

class AdvanceStepTilingHelper {
public:
    explicit AdvanceStepTilingHelper(gert::TilingContext* context) : context_(context)
    {}

    ~AdvanceStepTilingHelper() = default;
    ge::graphStatus DoTiling();
    ge::graphStatus PostTiling(AdvanceStepTilingData* tilingData) const;

private:
    bool GetBaseInfo();
    bool CheckAttrs();
    bool GetInputsOutputs();
    bool CheckOptionalInputs();
    bool CheckInputOutputShape();
    bool CheckInputOutputDType() const;
    bool Tiling4Seqs();
    ge::graphStatus Tiling4AdvanceStepLegacy();

    gert::TilingContext* context_;

    const gert::Shape* inputTokensShape_ = nullptr;
    const gert::Shape* sampledTokenIdsShape_ = nullptr;
    const gert::Shape* inputPositionsShape_ = nullptr;
    const gert::Shape* seqLensShape_ = nullptr;
    const gert::Shape* slotMappingShape_ = nullptr;
    const gert::Shape* blockTablesShape_ = nullptr;
    const gert::Shape* specTokenShape_ = nullptr;
    const gert::Shape* acceptedNumShape_ = nullptr;

    uint64_t aivNum_{1};
    uint64_t ubSize_{1};

    int64_t numSeqs_{1};
    int64_t numQueries_{1};
    int64_t blockSize_{1};

    int64_t specNum_{1};
    int64_t tokenEachReqs_{1};
    int64_t blockTablesStride_{1};
    int64_t needCoreNum_{1};
    int64_t perCoreSeqs_{1};
    int64_t lastCoreSeqs_{1};
    int64_t perLoopMaxSeqs_{1};

    bool specTokenExist{};
    bool acceptedNumExist{};

    UB_TILING_POLICY ubTilingPolicy_{UB_TILING_POLICY::FULL_LOAD};
};
} // namespace optiling

#endif // ADVANCE_STEP_TILING_H
