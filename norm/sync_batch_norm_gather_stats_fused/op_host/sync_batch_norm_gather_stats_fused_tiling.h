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
 * \file sync_batch_norm_gather_stats_fused_tiling.h
 * \brief
 */

#ifndef SYNC_BATCH_NORM_GATHER_STATS_FUSED_TILING_H
#define SYNC_BATCH_NORM_GATHER_STATS_FUSED_TILING_H

#include "register/tilingdata_base.h"
#include "log/log.h"
#include "error_util.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "op_host/tiling_base.h"
#include "op_common/op_host/util/platform_util.h"
#include "op_host/tiling_templates_registry.h"

namespace optiling {
constexpr uint64_t TEMPLATE_KEY_WEIGHT = 100;
constexpr uint64_t BLOCK_SIZE = 32;
constexpr uint64_t VREG_SIZE = 256;
constexpr uint64_t VL_FP32 = VREG_SIZE / sizeof(float);
constexpr uint64_t FLOAT_SIZE = 4;
constexpr uint64_t HALF_SIZE = 2;

enum class DtypeKey : int { FLOAT32 = 1, FLOAT16 = 2, BFLOAT16 = 3 };

enum class TemplateKey : int { COMMON = 1, WORKSPACE = 2, FIRST_AXIS_COMMON = 3, FIRST_AXIS_WORKSPACE = 4 };

struct ParamsSyncBatchNormGatherStatsFused {
    uint32_t coreNum = 0;
    uint64_t ubSizePlatForm = 0;
    int64_t blockSize = 0;
    int64_t vlFp32 = 0;
    int64_t nLength = 1;
    int64_t cLength = 1;
    ge::DataType dtype;
    DtypeKey dtypeKey;
    float momentum = 0.1f;
    float eps = 1e-5f;
};

struct SyncBatchNormGatherStatsFusedCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSizePlatForm = 0;
    int64_t blockSize = 0;
    int64_t vlFp32 = 0;
    bool isRegBase;
};

BEGIN_TILING_DATA_DEF(SyncBatchNormGatherStatsFusedTilingData)
TILING_DATA_FIELD_DEF(uint32_t, colSize);
TILING_DATA_FIELD_DEF(uint32_t, rowSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SyncBatchNormGatherStatsFused, SyncBatchNormGatherStatsFusedTilingData)

BEGIN_TILING_DATA_DEF(SyncBatchNormGatherStatsFusedTilingDataCommon)
TILING_DATA_FIELD_DEF(int64_t, nLength);
TILING_DATA_FIELD_DEF(int64_t, cLength);
TILING_DATA_FIELD_DEF(int64_t, cFormerAlignV_);
TILING_DATA_FIELD_DEF(int64_t, cFormerAlignM_);
TILING_DATA_FIELD_DEF(int64_t, cTailAlignV_);
TILING_DATA_FIELD_DEF(int64_t, cTailAlignM_);
TILING_DATA_FIELD_DEF(int64_t, blockFormer);
TILING_DATA_FIELD_DEF(int64_t, blockNum);
TILING_DATA_FIELD_DEF(int64_t, blockTail);
TILING_DATA_FIELD_DEF(int64_t, ubFormer);
TILING_DATA_FIELD_DEF(int64_t, ubLoop);
TILING_DATA_FIELD_DEF(int64_t, ubTail);
TILING_DATA_FIELD_DEF(int64_t, wholeBufferByteSize);
TILING_DATA_FIELD_DEF(int64_t, nBufferByteSize);
TILING_DATA_FIELD_DEF(int64_t, cBufferByteSize);
TILING_DATA_FIELD_DEF(int64_t, nBrcbBufferByteSize);
TILING_DATA_FIELD_DEF(int64_t, wholeBufferElemNums);
TILING_DATA_FIELD_DEF(float, momentum);
TILING_DATA_FIELD_DEF(float, eps);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SyncBatchNormGatherStatsFused_101, SyncBatchNormGatherStatsFusedTilingDataCommon)
REGISTER_TILING_DATA_CLASS(SyncBatchNormGatherStatsFused_102, SyncBatchNormGatherStatsFusedTilingDataCommon)
REGISTER_TILING_DATA_CLASS(SyncBatchNormGatherStatsFused_103, SyncBatchNormGatherStatsFusedTilingDataCommon)

BEGIN_TILING_DATA_DEF(SyncBatchNormGatherStatsFusedTilingDataWorkspace)
TILING_DATA_FIELD_DEF(int64_t, nLength);
TILING_DATA_FIELD_DEF(int64_t, cLength);
TILING_DATA_FIELD_DEF(int64_t, blockFormer);
TILING_DATA_FIELD_DEF(int64_t, blockNum);
TILING_DATA_FIELD_DEF(int64_t, blockTail);
TILING_DATA_FIELD_DEF(int64_t, ubFormerOfFormer);
TILING_DATA_FIELD_DEF(int64_t, ubTailOfFormer);
TILING_DATA_FIELD_DEF(int64_t, ubLoopOfFormer);
TILING_DATA_FIELD_DEF(int64_t, ubFormerOfTail);
TILING_DATA_FIELD_DEF(int64_t, ubTailOfTail);
TILING_DATA_FIELD_DEF(int64_t, ubLoopOfTail);
TILING_DATA_FIELD_DEF(float, momentum);
TILING_DATA_FIELD_DEF(float, eps);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SyncBatchNormGatherStatsFused_201, SyncBatchNormGatherStatsFusedTilingDataWorkspace)
REGISTER_TILING_DATA_CLASS(SyncBatchNormGatherStatsFused_202, SyncBatchNormGatherStatsFusedTilingDataWorkspace)
REGISTER_TILING_DATA_CLASS(SyncBatchNormGatherStatsFused_203, SyncBatchNormGatherStatsFusedTilingDataWorkspace)

BEGIN_TILING_DATA_DEF(SyncBatchNormGatherStatsFusedTilingDataFirstAxisCommon)
TILING_DATA_FIELD_DEF(int64_t, nLength);
TILING_DATA_FIELD_DEF(int64_t, cLength);
TILING_DATA_FIELD_DEF(int64_t, cAlignV);
TILING_DATA_FIELD_DEF(int64_t, cAlignM);
TILING_DATA_FIELD_DEF(int64_t, blockFormer);
TILING_DATA_FIELD_DEF(int64_t, blockNum);
TILING_DATA_FIELD_DEF(int64_t, blockTail);
TILING_DATA_FIELD_DEF(int64_t, ubFormer);
TILING_DATA_FIELD_DEF(int64_t, ubLoopOfFormerBlock);
TILING_DATA_FIELD_DEF(int64_t, ubLoopOfTailBlock);
TILING_DATA_FIELD_DEF(int64_t, ubTailOfFormerBlock);
TILING_DATA_FIELD_DEF(int64_t, ubTailOfTailBlock);
TILING_DATA_FIELD_DEF(int64_t, wholeBufferByteSize);
TILING_DATA_FIELD_DEF(int64_t, nBufferByteSize);
TILING_DATA_FIELD_DEF(int64_t, cBufferByteSize);
TILING_DATA_FIELD_DEF(int64_t, nBrcbBufferByteSize);
TILING_DATA_FIELD_DEF(int64_t, wholeBufferElemNums);
TILING_DATA_FIELD_DEF(float, momentum);
TILING_DATA_FIELD_DEF(float, eps);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SyncBatchNormGatherStatsFused_301, SyncBatchNormGatherStatsFusedTilingDataFirstAxisCommon)
REGISTER_TILING_DATA_CLASS(SyncBatchNormGatherStatsFused_302, SyncBatchNormGatherStatsFusedTilingDataFirstAxisCommon)
REGISTER_TILING_DATA_CLASS(SyncBatchNormGatherStatsFused_303, SyncBatchNormGatherStatsFusedTilingDataFirstAxisCommon)

BEGIN_TILING_DATA_DEF(SyncBatchNormGatherStatsFusedTilingDataFirstAxisWorkspace)
TILING_DATA_FIELD_DEF(int64_t, nLength);
TILING_DATA_FIELD_DEF(int64_t, cLength);
TILING_DATA_FIELD_DEF(int64_t, cAlignV);
TILING_DATA_FIELD_DEF(int64_t, blockFormer);
TILING_DATA_FIELD_DEF(int64_t, blockNum);
TILING_DATA_FIELD_DEF(int64_t, blockTail);
TILING_DATA_FIELD_DEF(int64_t, ubFormer);
TILING_DATA_FIELD_DEF(int64_t, ubTail);
TILING_DATA_FIELD_DEF(int64_t, ubLoop);
TILING_DATA_FIELD_DEF(float, momentum);
TILING_DATA_FIELD_DEF(float, eps);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SyncBatchNormGatherStatsFused_401, SyncBatchNormGatherStatsFusedTilingDataFirstAxisWorkspace)
REGISTER_TILING_DATA_CLASS(SyncBatchNormGatherStatsFused_402, SyncBatchNormGatherStatsFusedTilingDataFirstAxisWorkspace)
REGISTER_TILING_DATA_CLASS(SyncBatchNormGatherStatsFused_403, SyncBatchNormGatherStatsFusedTilingDataFirstAxisWorkspace)

class SyncBatchNormGatherStatsFusedTilingBase : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit SyncBatchNormGatherStatsFusedTilingBase(gert::TilingContext* context)
        : Ops::NN::Optiling::TilingBaseClass(context)
    {}
    ~SyncBatchNormGatherStatsFusedTilingBase() override = default;

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;

private:
    ge::graphStatus ValidateInputDtypes();
    ge::graphStatus ValidateInputShapes();
    ge::graphStatus GetAndValidateAttrs();
    void SetDtypeKey();

public:
    ParamsSyncBatchNormGatherStatsFused commonParams;
};

class SyncBatchNormGatherStatsFusedCommonTiling : public SyncBatchNormGatherStatsFusedTilingBase {
public:
    explicit SyncBatchNormGatherStatsFusedCommonTiling(gert::TilingContext* context)
        : SyncBatchNormGatherStatsFusedTilingBase(context)
    {}
    ~SyncBatchNormGatherStatsFusedCommonTiling() override = default;

protected:
    bool IsCapable() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    ge::graphStatus GetWorkspaceSize() override;

private:
    int64_t CalculateubFormer();

    int64_t cFormerAlignV_;
    int64_t cTailAlignV_;
    int64_t cFormerAlignM_;
    int64_t cTailAlignM_;

    SyncBatchNormGatherStatsFusedTilingDataCommon td_;
};

class SyncBatchNormGatherStatsFusedWorkspaceTiling : public SyncBatchNormGatherStatsFusedTilingBase {
public:
    explicit SyncBatchNormGatherStatsFusedWorkspaceTiling(gert::TilingContext* context)
        : SyncBatchNormGatherStatsFusedTilingBase(context)
    {}
    ~SyncBatchNormGatherStatsFusedWorkspaceTiling() override = default;

protected:
    bool IsCapable() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    ge::graphStatus GetWorkspaceSize() override;

private:
    SyncBatchNormGatherStatsFusedTilingDataWorkspace td_;
};

class SyncBatchNormGatherStatsFusedFirstAxisCommonTiling : public SyncBatchNormGatherStatsFusedTilingBase {
public:
    explicit SyncBatchNormGatherStatsFusedFirstAxisCommonTiling(gert::TilingContext* context)
        : SyncBatchNormGatherStatsFusedTilingBase(context)
    {}
    ~SyncBatchNormGatherStatsFusedFirstAxisCommonTiling() override = default;

protected:
    bool IsCapable() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    ge::graphStatus GetWorkspaceSize() override;

private:
    int64_t CalculateubFormer();

    int64_t cAlignV_;
    int64_t cAlignM_;

    SyncBatchNormGatherStatsFusedTilingDataFirstAxisCommon td_;
};

class SyncBatchNormGatherStatsFusedFirstAxisWorkspaceTiling : public SyncBatchNormGatherStatsFusedTilingBase {
public:
    explicit SyncBatchNormGatherStatsFusedFirstAxisWorkspaceTiling(gert::TilingContext* context)
        : SyncBatchNormGatherStatsFusedTilingBase(context)
    {}
    ~SyncBatchNormGatherStatsFusedFirstAxisWorkspaceTiling() override = default;

protected:
    bool IsCapable() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    ge::graphStatus GetWorkspaceSize() override;

private:
    SyncBatchNormGatherStatsFusedTilingDataFirstAxisWorkspace td_;
};

} // namespace optiling

#endif // SYNC_BATCH_NORM_GATHER_STATS_FUSED_TILING_H