/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/**
 * \file apply_centered_rms_prop_tiling.cpp
 * \brief ApplyCenteredRMSProp tiling (arch35).
 *
 * Tiling strategy (DESIGN §5):
 *   1. Multi-core: split total elements evenly across AI vector cores.
 *      blockFactor = CeilAlign(CeilDiv(total, coreNum), ubBlockSize)
 *      so neighbouring cores do not trample each other's GM output.
 *   2. UB: ubFactor = floor((UB_size - reserve) / (N * compute_dtype_size))
 *      where N depends on dtype path (DESIGN §5.2):
 *        - fp32: 8 fp32 buffers
 *        - fp16: 12 mixed buffers (5 fp16 queues x 2B + 7 fp32 tmp x 4B)
 *          -> Conservatively use 12 * 4B = 48B / element (upper bound).
 *      Rounded down to 32-byte granularity.
 *
 * TilingKey (DESIGN §5.4):
 *   dtype fp32 -> TilingKey 1
 *   dtype fp16 -> TilingKey 2
 *   selected via ASCENDC_TPL_SEL_PARAM(context, dTypeVar).
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/apply_centered_rms_prop_tiling_data.h"
#include "../op_kernel/apply_centered_rms_prop_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorAlign;
using Ops::Base::FloorDiv;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
// Per-tile target element count -- matches apply_proximal_adagrad's 2048
// ceiling so the first iteration stays deterministic while leaving UB
// headroom for the fp16 12-buffer layout.
constexpr int64_t TILE_ELEM_NUM_TARGET = 2048;
// UB co-resident buffer accounting (per element, in 4-byte units).
//
// fp16 path (Kernel InitBuffer): 9 queues x 2 DB x 2B + 8 fp32 TBufs x 4B
//          = 36B + 32B = 68B / elem  -> 17 fp32-units / elem.
// fp32 path (Kernel InitBuffer, Suggestion-1 conditional):
//          9 queues x 2 DB x 4B + 3 fp32 TBufs x 4B  (denom/tmp1/tmp2 only;
//          the 5 cast-buffers are NOT allocated on fp32 path)
//          = 72B + 12B = 84B / elem  -> 21 fp32-units / elem.
// We round up to 21 for the fp32 path to match the kernel InitBuffer footprint.
constexpr int64_t UB_BUFFER_COUNT_FP32 = 21;
constexpr int64_t UB_BUFFER_COUNT_FP16 = 17;
constexpr int64_t SIZE_FP32 = 4;
constexpr int64_t SIZE_FP16 = 2;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape)
{
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return in_shape;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context,
                                       uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"),
                return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeInfo(gert::TilingContext* context,
                                    int64_t& totalElements,
                                    ge::DataType& dataType)
{
    // Input 0 = var; its shape is canonical (var/mg/ms/mom/grad must match).
    auto inputVar = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputVar);
    auto varShape = EnsureNotScalar(inputVar->GetStorageShape());
    totalElements = varShape.GetShapeSize();

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();
    OP_CHECK_IF(dataType != ge::DT_FLOAT && dataType != ge::DT_FLOAT16,
                OP_LOGE(context,
                        "ApplyCenteredRMSProp: only float32 / float16 are "
                        "supported, got %d",
                        static_cast<int>(dataType)),
                return ge::GRAPH_FAILED);

    // Tensor-shape consistency: mg/ms/mom/grad must equal var.shape.
    // (Issue/Suggestion-4 closure: shape_mismatch must be rejected at host
    // tiling instead of relying on autogen L2 checker, which only catches
    // dtype mismatches.)
    constexpr size_t kMainTensorIdx[] = {1U, 2U, 3U, 8U};   // mg, ms, mom, grad
    const char* kMainTensorName[] = {"mg", "ms", "mom", "grad"};
    for (size_t i = 0; i < sizeof(kMainTensorIdx) / sizeof(kMainTensorIdx[0]); ++i) {
        auto inShape = context->GetInputShape(kMainTensorIdx[i]);
        OP_CHECK_NULL_WITH_CONTEXT(context, inShape);
        auto s = EnsureNotScalar(inShape->GetStorageShape());
        OP_CHECK_IF(
            s.GetShapeSize() != totalElements,
            OP_LOGE(context,
                    "ApplyCenteredRMSProp: %s.numel=%ld must equal var.numel=%ld",
                    kMainTensorName[i],
                    static_cast<long>(s.GetShapeSize()),
                    static_cast<long>(totalElements)),
            return ge::GRAPH_FAILED);
    }

    // Scalar shape self-defense (Issue-1): lr/rho/momentum/epsilon must be
    // 0-D or single-element 1-D (numel ∈ {0, 1}). Empty scalar is rejected
    // because LoadScalar reads element[0]. (numel == 0 untreated would lead
    // to OOB GetValue in kernel.)
    constexpr size_t kScalarIdx[] = {4U, 5U, 6U, 7U};       // lr, rho, momentum, epsilon
    const char* kScalarName[] = {"lr", "rho", "momentum", "epsilon"};
    for (size_t i = 0; i < sizeof(kScalarIdx) / sizeof(kScalarIdx[0]); ++i) {
        auto inShape = context->GetInputShape(kScalarIdx[i]);
        OP_CHECK_NULL_WITH_CONTEXT(context, inShape);
        const auto& rawShape = inShape->GetStorageShape();
        int64_t numel = (rawShape.GetDimNum() == 0) ? 1 : rawShape.GetShapeSize();
        OP_CHECK_IF(
            numel != 1,
            OP_LOGE(context,
                    "ApplyCenteredRMSProp: scalar %s must be 0-D or 1-element 1-D, "
                    "got numel=%ld",
                    kScalarName[i], static_cast<long>(numel)),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ApplyCenteredRMSPropTilingFunc(gert::TilingContext* context)
{
    // 1. Platform info
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"),
                return ge::GRAPH_FAILED);

    // 2. Shape / dtype info
    int64_t totalElements = 0;
    ge::DataType dataType = ge::DT_FLOAT;
    OP_CHECK_IF(GetShapeInfo(context, totalElements, dataType) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetShapeInfo error"),
                return ge::GRAPH_FAILED);

    // 3. Workspace
    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetWorkspaceSize error"),
                return ge::GRAPH_FAILED);

    // 4. Fill TilingData
    ApplyCenteredRMSPropTilingData* tiling =
        context->GetTilingData<ApplyCenteredRMSPropTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(ApplyCenteredRMSPropTilingData), 0,
                 sizeof(ApplyCenteredRMSPropTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"),
        return ge::GRAPH_FAILED);

    tiling->totalElements = totalElements;

    // Empty tensor: run a single idle core so the launcher still succeeds.
    if (totalElements == 0) {
        tiling->blockFactor = 0;
        tiling->ubFactor = 0;
        context->SetBlockDim(1);
        uint32_t dTypeVar = static_cast<uint32_t>(dataType);
        ASCENDC_TPL_SEL_PARAM(context, dTypeVar);
        return ge::GRAPH_SUCCESS;
    }

    // ubBlockSize = 32B / sizeof(T).
    int64_t ubBlockSize = Ops::Base::GetUbBlockSize(context);
    OP_CHECK_IF(ubBlockSize <= 0,
                OP_LOGE(context, "invalid ubBlockSize=%ld", ubBlockSize),
                return ge::GRAPH_FAILED);

    // Multi-core split: ceil-aligned to DMA granularity.
    int64_t blockFactor = CeilAlign(CeilDiv(totalElements, coreNum), ubBlockSize);
    int64_t usedCoreNum = CeilDiv(totalElements, blockFactor);

    // UB split per dtype path. perElemBytes = sizeof(fp32) for both paths;
    // ubBufCount captures the actual per-element 4-byte-unit footprint
    // (fp32 path: 26 units; fp16 path: 17 units).
    int64_t ubBufCount;
    int64_t perElemBytes = SIZE_FP32;
    if (dataType == ge::DT_FLOAT) {
        ubBufCount = UB_BUFFER_COUNT_FP32;
    } else {
        ubBufCount = UB_BUFFER_COUNT_FP16;
    }

    int64_t ubCapacityElem =
        FloorAlign(FloorDiv(static_cast<int64_t>(ubSize) / perElemBytes,
                            ubBufCount),
                   ubBlockSize);
    OP_CHECK_IF(ubCapacityElem <= 0,
                OP_LOGE(context, "UB too small: ubCapacityElem=%ld",
                        ubCapacityElem),
                return ge::GRAPH_FAILED);

    int64_t ubFactor = (TILE_ELEM_NUM_TARGET < ubCapacityElem)
                           ? TILE_ELEM_NUM_TARGET
                           : ubCapacityElem;

    // Also cap by blockFactor so a single core does not allocate more UB
    // space than it will ever use.
    if (ubFactor > blockFactor) {
        ubFactor = FloorAlign(blockFactor, ubBlockSize);
        if (ubFactor <= 0) {
            ubFactor = ubBlockSize;
        }
    }

    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(usedCoreNum);

    // 5. TilingKey via ASCENDC_TPL_SEL_PARAM (template-argument mechanism).
    //    Key is fully encoded by dtype (fp16=KEY2, fp32=KEY1).
    uint32_t dTypeVar = static_cast<uint32_t>(dataType);
    ASCENDC_TPL_SEL_PARAM(context, dTypeVar);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForApplyCenteredRMSProp(
    [[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ApplyCenteredRMSPropCompileInfo {};

IMPL_OP_OPTILING(ApplyCenteredRMSProp)
    .Tiling(ApplyCenteredRMSPropTilingFunc)
    .TilingParse<ApplyCenteredRMSPropCompileInfo>(TilingParseForApplyCenteredRMSProp);

} // namespace optiling
