/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// mc2编译依赖
//
//
//
//
//
//
//
//
//
//
//
//
//
//
// mc2编译依赖
//
//
// 
// 
//
//
//
//
//
// 
// 
// 
// 
// mc2编译依赖

/*!
 * \file quant_batch_matmul_v4_msd_tiling.h
 * \brief
 */

#ifndef QUANT_BATCH_MATMUL_V4_MSD_TILING_H
#define QUANT_BATCH_MATMUL_V4_MSD_TILING_H

#include <cstdint>
#include <vector>

#include "tiling/tiling_api.h"
#include "quant_batch_matmul_v4_compile_info.h"
#include "quant_batch_matmul_v4_tiling_info.h"
#include "tiling_base/tiling_base.h"

namespace optiling {
using Ops::NN::Optiling::TilingBaseClass;

struct QuantBatchMatmulMsdInfo {
    bool transA;
    bool transB;
    bool weightNz;
    bool hasAntiQuantOffset;
    bool hasBias;
    uint32_t libApiWorkSpaceSize;
    uint64_t groupSize;
    uint64_t mSize;
    uint64_t kSize;
    uint64_t nSize;
    uint32_t useCoreNum;
    ge::DataType aDtype;
    ge::DataType bDtype;
    ge::DataType cDtype;
    ge::DataType x1ScaleDtype;
    ge::DataType x2ScaleDtype;
    ge::DataType yOffsetDtype;
    QuantBatchMatmulV4QuantType antiQuantType;
    const char *opName;
};

BEGIN_TILING_DATA_DEF(QuantBatchMatmulV4MsdTilingData)
TILING_DATA_FIELD_DEF(uint8_t, coreNum);
TILING_DATA_FIELD_DEF(uint32_t, vBaseM);
TILING_DATA_FIELD_DEF(uint32_t, ubRestBytes);
TILING_DATA_FIELD_DEF(uint32_t, parallNum);
TILING_DATA_FIELD_DEF(uint32_t, ubCalSize);
TILING_DATA_FIELD_DEF(uint32_t, mSize);
TILING_DATA_FIELD_DEF(uint32_t, kSize);
TILING_DATA_FIELD_DEF(uint32_t, nSize);
TILING_DATA_FIELD_DEF(uint32_t, groupSize);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, matmulTiling);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(QuantBatchMatmulV4_16, QuantBatchMatmulV4MsdTilingData)

class QuantBatchMatmulV4MsdTiling : public TilingBaseClass {
public:
    explicit QuantBatchMatmulV4MsdTiling(gert::TilingContext *context) : TilingBaseClass(context)
    {
        Reset();
        if (context->GetCompileInfo() == nullptr) {
            InitCompileInfo();
        }
    }

    explicit QuantBatchMatmulV4MsdTiling(gert::TilingContext *context, QuantBatchMatmulV4MsdTilingData * out)
        : TilingBaseClass(context)
    {
        Reset();
        tilingData_ = out;
        InitCompileInfo();
    }

    ~QuantBatchMatmulV4MsdTiling() override = default;

    void Reset(gert::TilingContext * context) override {
        TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

    bool AnalyzeScaleInputs();
    bool AnalyzeYOffsetInputs();
    void Reset();
    void InitCompileInfo();
    void InitPlatformInfo(const QuantBatchMatmulV4CompileInfo* compileInfoPtr_, matmul_tiling::PlatformInfo& platformInfo) const;
    ge::graphStatus CheckContext();
    ge::graphStatus InstantiateTilingData();
    void PrintTilingData() const;
    void PrintMatmulTilingData() const;
    bool AnalyzeAttrs();
    bool AnalyzeDtype();
    bool AnalyzeInputs();
    bool SetMatmulTiling();
    QuantBatchMatmulMsdInfo inputParams_;
    const gert::Shape GetShape(const size_t index) const;
    const gert::Shape GetOptionShape(const size_t index) const;
    std::unique_ptr<QuantBatchMatmulV4MsdTilingData> tilingDataManager_;
    QuantBatchMatmulV4MsdTilingData * tilingData_ = nullptr;
    QuantBatchMatmulV4CompileInfo compileInfo_;
    bool compileInfoInit_ = false;
};
}   // namespace optiling
#endif  // QUANT_BATCH_MATMUL_V4_MSD_TILING_H