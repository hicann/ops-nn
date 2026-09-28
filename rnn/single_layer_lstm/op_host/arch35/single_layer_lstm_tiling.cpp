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
 * \file single_layer_lstm_tiling.cpp
 * \brief SingleLayerLstm tiling (arch35 / dav_3510). Every capacity decision is made here, over the
 *        same layout objects the kernel constructs; the kernel performs no capacity test at all,
 *        because a kernel that returns when a shape does not fit leaves the caller's outputs at
 *        whatever they held. The framework collapses any GRAPH_FAILED from here into one message
 *        (EZ1008 / 561002), so every return path below logs which check failed.
 */

#include "register/op_def_registry.h"
#include <cstdint>
#include <cstring>
#include <string>
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "../../op_kernel/arch35/single_layer_lstm_tiling_data.h"
#include "single_layer_lstm_budget.h"

namespace optiling {
namespace {

constexpr size_t IDX_IN_X = 0;
constexpr size_t IDX_IN_W = 1;
constexpr size_t IDX_IN_B = 2;
constexpr size_t IDX_IN_INIT_H = 3;
constexpr size_t IDX_IN_INIT_C = 4;
constexpr size_t IDX_IN_SEQ_LENGTH = 5;
constexpr size_t IDX_IN_BIAS_HH = 6;

constexpr size_t ATTR_DIRECTION = 0;
constexpr size_t ATTR_GATE_ORDER = 1;
constexpr size_t ATTR_LOGICAL_INPUT_SIZE = 2;
constexpr size_t ATTR_LOGICAL_HIDDEN_SIZE = 3;

constexpr size_t X_RANK = 3;
constexpr size_t STATE_RANK = 2;
constexpr size_t W_RANK = 2;
constexpr size_t B_RANK = 1;

constexpr size_t WORKSPACE_NUM = 1;
/* Fallback only, used when the platform info is absent. The real figure comes from the platform:
 * a constant that is merely >= the true reserve wastes device memory silently, and one that is <
 * it faults. */
constexpr size_t WS_SYS_FALLBACK = 16UL * 1024 * 1024;

/* The only direction and gate order this kernel implements. Both are REFUSED rather than ignored
 * when they differ: a silently ignored non-default value trains a plausible wrong model, whereas a
 * refusal costs the caller one error message. `gate_order` in particular selects the column order
 * of `w` and `b`, and "ifjo" against "ijfo" is a permutation that leaves the result smooth and
 * well conditioned -- nothing downstream would report it. */
constexpr const char* SUPPORTED_DIRECTION = "UNIDIRECTIONAL";
constexpr const char* SUPPORTED_GATE_ORDER = "ifjo";

struct SingleLayerLstmCompileInfo {
    uint64_t sysWorkspaceSize = 0;
};

ge::graphStatus CheckAttrs(gert::TilingContext* context)
{
    const auto* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_SUCCESS; // optional attributes default to the supported values
    }
    const char* direction = attrs->GetAttrPointer<char>(ATTR_DIRECTION);
    if (direction != nullptr && std::strcmp(direction, SUPPORTED_DIRECTION) != 0) {
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(),
                               "direction '%s' is not implemented and is refused rather than ignored; "
                               "only '%s' is supported. Run the reverse pass as a second SingleLayerLstm node "
                               "over a time-reversed input.",
                               direction, SUPPORTED_DIRECTION);
        return ge::GRAPH_FAILED;
    }
    const char* gateOrder = attrs->GetAttrPointer<char>(ATTR_GATE_ORDER);
    if (gateOrder != nullptr && std::strcmp(gateOrder, SUPPORTED_GATE_ORDER) != 0) {
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(),
                               "gate_order '%s' is not implemented and is refused rather than ignored; "
                               "only '%s' is supported. It selects the column order of w and b, and a "
                               "wrong permutation stays numerically plausible.",
                               gateOrder, SUPPORTED_GATE_ORDER);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

/* The element width of the dtype, in bytes, or 0 for one this operator does not implement. */
uint32_t WidthOf(ge::DataType dt)
{
    switch (dt) {
        case ge::DT_FLOAT:
            return 4U;
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
            return 2U;
        default:
            return 0U;
    }
}

const char* NameOf(ge::DataType dt)
{
    switch (dt) {
        case ge::DT_FLOAT:
            return "DT_FLOAT";
        case ge::DT_FLOAT16:
            return "DT_FLOAT16";
        case ge::DT_BF16:
            return "DT_BF16";
        default:
            return "an unsupported dtype";
    }
}

/* Resolves the one width w, init_h, init_c and y share, and writes it to *inBytes. The kernel is
 * built once per dtype combination -- DTYPE_W, from config/ascend950/single_layer_lstm_binary.json
 * -- so a node whose narrow tensors disagree would run a binary compiled for one of them and
 * reinterpret the rest. The def's dtype lists are positional, so the framework has already matched a
 * combination; this catches the case where it matched one this build does not implement. */
ge::graphStatus ResolveDtype(gert::TilingContext* context, uint32_t* inBytes)
{
    *inBytes = 0;
    const auto* wDesc = context->GetInputDesc(IDX_IN_W);
    OP_CHECK_NULL_WITH_CONTEXT(context, wDesc);
    const ge::DataType wDt = wDesc->GetDataType();
    const uint32_t width = WidthOf(wDt);
    if (width == 0) {
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(),
                               "w is dtype %d; this operator implements DT_FLOAT, DT_FLOAT16 and DT_BF16",
                               static_cast<int32_t>(wDt));
        return ge::GRAPH_FAILED;
    }

    const size_t sameAsW[] = {IDX_IN_X, IDX_IN_B, IDX_IN_INIT_H, IDX_IN_INIT_C};
    const char* narrowNames[] = {"x", "b", "init_h", "init_c"};
    for (size_t k = 0; k < sizeof(sameAsW) / sizeof(sameAsW[0]); ++k) {
        const auto* desc = context->GetInputDesc(sameAsW[k]);
        OP_CHECK_NULL_WITH_CONTEXT(context, desc);
        if (desc->GetDataType() != wDt) {
            OP_LOGE_WITHOUT_REPORT(context->GetNodeName(),
                                   "input '%s' is %s but w is %s; they must agree, because the kernel is built "
                                   "once per dtype and reads them all at w's width",
                                   narrowNames[k], NameOf(desc->GetDataType()), NameOf(wDt));
            return ge::GRAPH_FAILED;
        }
    }

    /* THE EIGHT OUTPUTS ARE NOT CHECKED HERE, AND THEY ARE BOUND ALL THE SAME. Entry k of every
     * DataType list in the def describes the k-th supported combination, and the framework has
     * matched one before this function runs -- so an output whose dtype differs is not a
     * combination this operator declares and never reaches tiling. Reading them back would also not
     * work: the tiling context carries compute-node tensor descriptors for INPUTS, and
     * GetOutputDesc faults on a node whose output descriptors were never populated. The declaration
     * in single_layer_lstm_def.cpp is what holds this, and it holds it earlier. */

    const auto* biasHhDesc = context->GetOptionalInputDesc(IDX_IN_BIAS_HH);
    if (biasHhDesc != nullptr && biasHhDesc->GetDataType() != wDt) {
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "bias_hh must have the same dtype as w");
        return ge::GRAPH_FAILED;
    }
    *inBytes = width;
    return ge::GRAPH_SUCCESS;
}

/* seq_length. It reaches tiling only if the caller materialised its VALUE on the host; a CANN input
 * carries no host-memory pinning, so the data may well still be on device. A blind dereference
 * segfaults INSIDE GetWorkspaceSize with an empty log, which reads as a kernel crash and is not
 * one. So: attempt the read, verify the range, and fall back to T. The tail-zeroing path this feeds
 * has not been exercised by a test with a short sequence, and that is stated rather than assumed
 * away. */
uint32_t ResolveSeqLenMax(gert::TilingContext* context, uint32_t timeStep)
{
    /* GetOptionalInputTensor, NOT GetInputTensor. The latter is indexed by INSTANTIATED input, so
     * when a graph omits this trailing optional input the index is out of range and the pointer it
     * hands back is not a tensor -- dereferencing it segfaults inside tiling, which surfaces as a
     * crash with no message rather than as a missing input. The optional accessor is indexed by IR
     * position and returns nullptr both for an invalid index and for an input that was not
     * instantiated. */
    const gert::Tensor* tensor = context->GetOptionalInputTensor(IDX_IN_SEQ_LENGTH);
    if (tensor == nullptr || tensor->GetShapeSize() < 1) {
        return timeStep;
    }
    const int64_t* data = tensor->GetData<int64_t>();
    if (data == nullptr || *data < 0 || *data > static_cast<int64_t>(timeStep)) {
        return timeStep;
    }
    return static_cast<uint32_t>(*data);
}

size_t ResolveSysWorkspace(gert::TilingContext* context)
{
    if (context->GetPlatformInfo() != nullptr) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        return static_cast<size_t>(platform.GetLibApiWorkSpaceSize());
    }
    const auto* compileInfo = context->GetCompileInfo<SingleLayerLstmCompileInfo>();
    if (compileInfo != nullptr && compileInfo->sysWorkspaceSize != 0) {
        return static_cast<size_t>(compileInfo->sysWorkspaceSize);
    }
    return WS_SYS_FALLBACK;
}

} // namespace

static ge::graphStatus SingleLayerLstmTilingFunc(gert::TilingContext* context)
{
    auto* tilingData = context->GetTilingData<SingleLayerLstmTilingData>();
    /* The framework sizes this buffer from the REGISTERED TilingDef, not from the struct this
     * function casts to. With no registration the capacity is a few bytes and GetTilingData<T>()
     * returns nullptr the instant sizeof(T) exceeds it -- with the tiling function never having run
     * a line of its own logic. See SingleLayerLstmTilingDataRaw in single_layer_lstm_def.cpp. */
    OP_CHECK_NULL_WITH_CONTEXT(context, tilingData);
    *tilingData = SingleLayerLstmTilingData{};

    uint32_t inBytes = 0;
    if (CheckAttrs(context) != ge::GRAPH_SUCCESS || ResolveDtype(context, &inBytes) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    const auto* xShape = context->GetInputShape(IDX_IN_X);
    const auto* wShape = context->GetInputShape(IDX_IN_W);
    const auto* bShape = context->GetInputShape(IDX_IN_B);
    const auto* hShape = context->GetInputShape(IDX_IN_INIT_H);
    const auto* cShape = context->GetInputShape(IDX_IN_INIT_C);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, wShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, bShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, hShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, cShape);

    const gert::Shape& x = xShape->GetStorageShape();
    const gert::Shape& w = wShape->GetStorageShape();
    const gert::Shape& bias = bShape->GetStorageShape();
    const gert::Shape& h0 = hShape->GetStorageShape();
    const gert::Shape& c0 = cShape->GetStorageShape();

    if (x.GetDimNum() != X_RANK || w.GetDimNum() != W_RANK || bias.GetDimNum() != B_RANK ||
        h0.GetDimNum() != STATE_RANK || c0.GetDimNum() != STATE_RANK) {
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(),
                               "expected ranks x=3 w=2 b=1 init_h=2 init_c=2; got %zu %zu %zu %zu %zu", x.GetDimNum(),
                               w.GetDimNum(), bias.GetDimNum(), h0.GetDimNum(), c0.GetDimNum());
        return ge::GRAPH_FAILED;
    }

    const int64_t tDim = x.GetDim(0);
    const int64_t bDim = x.GetDim(1);
    const int64_t iDim = x.GetDim(2);
    const int64_t hDim = h0.GetDim(1);
    if (tDim <= 0 || bDim <= 0 || iDim <= 0 || hDim <= 0) {
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(),
                               "T / B / input_size / hidden_size must all be positive; got %ld %ld %ld %ld", tDim, bDim,
                               iDim, hDim);
        return ge::GRAPH_FAILED;
    }

    const uint32_t timeStep = static_cast<uint32_t>(tDim);
    const uint32_t batch = static_cast<uint32_t>(bDim);
    const uint32_t inputSize = static_cast<uint32_t>(iDim);
    const uint32_t hidden = static_cast<uint32_t>(hDim);
    const int64_t gates4H = static_cast<int64_t>(SingleLayerLstmBudget::GATES) * hDim;

    int64_t logicalInput = -1;
    int64_t logicalHidden = -1;
    const auto* attrs = context->GetAttrs();
    if (attrs != nullptr && attrs->GetAttrNum() > ATTR_LOGICAL_INPUT_SIZE) {
        const auto* value = attrs->GetAttrPointer<int64_t>(ATTR_LOGICAL_INPUT_SIZE);
        if (value != nullptr) {
            logicalInput = *value;
        }
    }
    if (attrs != nullptr && attrs->GetAttrNum() > ATTR_LOGICAL_HIDDEN_SIZE) {
        const auto* value = attrs->GetAttrPointer<int64_t>(ATTR_LOGICAL_HIDDEN_SIZE);
        if (value != nullptr) {
            logicalHidden = *value;
        }
    }
    logicalInput = (logicalInput == -1) ? iDim : logicalInput;
    logicalHidden = (logicalHidden == -1) ? hDim : logicalHidden;
    if (logicalInput < 0 || logicalInput > iDim || logicalHidden <= 0 || logicalHidden > hDim) {
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(),
                               "logical_input_size must be in [0, I] and logical_hidden_size in (0, H]; "
                               "-1 selects the physical extent. Got logical I/H %ld/%ld, physical I/H %ld/%ld",
                               logicalInput, logicalHidden, iDim, hDim);
        return ge::GRAPH_FAILED;
    }

    /* w is the FUSED weight [I+H, 4H], with rows [0, I) the input weight and rows [I, I+H) the
     * hidden weight already TRANSPOSED -- which is exactly the operand the recurrence's cube wants,
     * so nothing is copied or transposed on device. Its transpose is a different matrix and would
     * produce plausible wrong numbers, so the shape is checked rather than inferred. */
    if (w.GetDim(0) != iDim + hDim || w.GetDim(1) != gates4H) {
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(),
                               "w must be [input_size + hidden_size, 4 * hidden_size] = [%ld, %ld]; got [%ld, %ld]",
                               iDim + hDim, gates4H, w.GetDim(0), w.GetDim(1));
        return ge::GRAPH_FAILED;
    }
    if (bias.GetDim(0) != gates4H) {
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "b must be [4 * hidden_size] = [%ld]; got [%ld]", gates4H,
                               bias.GetDim(0));
        return ge::GRAPH_FAILED;
    }
    const auto* biasHhShape = context->GetOptionalInputShape(IDX_IN_BIAS_HH);
    if (biasHhShape != nullptr) {
        const auto& shape = biasHhShape->GetStorageShape();
        if (shape.GetDimNum() != B_RANK || shape.GetDim(0) != gates4H) {
            OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "bias_hh must be [4 * hidden_size]");
            return ge::GRAPH_FAILED;
        }
        tilingData->hasBiasHh = 1;
    }
    if (h0.GetDim(0) != bDim || c0.GetDim(0) != bDim || c0.GetDim(1) != hDim) {
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(),
                               "init_h and init_c must both be [batch, hidden_size] = [%ld, %ld]", bDim, hDim);
        return ge::GRAPH_FAILED;
    }

    /* The two extents have different rules, and at fp32 they coincide -- which is why they used to be
     * one check. input_size is phase A's K axis and phase A is the only place the caller's dtype
     * reaches the cube, so with C0 at 32 bytes it is 8 elements at fp32 and 16 at the narrow dtypes.
     * hidden_size is bounded by the fp32 fractal in both cases, because phase B and the whole
     * epilogue are fp32 whatever the caller's width: every strided UB copy there states its row
     * pitch in 32-byte blocks and C0F elements is exactly one. Raising H to the narrow fractal would
     * refuse shapes that run. Batch carries no such constraint. */
    if (inputSize % SingleLayerLstmFwd::C0F != 0) {
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(),
                               "input_size must be a multiple of %u at every dtype (the cube reads fp32 in both "
                               "phases, and C0F elements is one 32-byte fractal); got %u. Pad the weights and "
                               "slice the output instead.",
                               SingleLayerLstmFwd::C0F, inputSize);
        return ge::GRAPH_FAILED;
    }
    if (hidden % SingleLayerLstmFwd::C0F != 0) {
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(),
                               "hidden_size must be a multiple of %u on this SoC; got %u. "
                               "Pad the weights and slice the output instead.",
                               SingleLayerLstmFwd::C0F, hidden);
        return ge::GRAPH_FAILED;
    }

    /* Batch rows are the only axis the recurrence can be partitioned on. The split need not be exact
     * -- each cluster builds its own Layout from its own row count -- and requiring a divisor put
     * the whole batch on one cluster whenever B had no small factor, which is what refused B=4097
     * and B=2049. Nor need it cover the batch in one pass: `rowsPerBlock` is the rows one cluster
     * holds on chip, the grid strides by blockDim * rowsPerBlock, and a cluster runs the whole
     * T-step recurrence once per block. Spreading B over MAX_CLUSTERS alone left 1024 rows per
     * cluster at B=16384 and a 2 MB A tile against 512 KB. Zero means not even one row fits, and the
     * two feasibility tests are then re-run at one row so the refusal names which failed. */
    uint32_t rowsPerBlock = SingleLayerLstmBudget::PickRowChunk(batch, hidden, timeStep, inputSize, inBytes);
    if (rowsPerBlock == 0) {
        rowsPerBlock = 1;
    }

    if (!SingleLayerLstmBudget::RecurrenceFits(batch, hidden, timeStep, rowsPerBlock, inBytes)) {
        /* REACHED ONLY AT ONE ROW ON CHIP, because PickRowChunk already tried every larger block
         * and halved its way down to 1. So this IS a hidden_size ceiling now, and a much higher one
         * than the operator used to have: both axes of the recurrence GEMM are tiled and the
         * weights stream from GM when L1 cannot hold them, leaving h_{t-1} -- [1, H] whole in L1,
         * because every k-chunk reads a column range of it -- and, at fp16 and bf16, the UB chunk
         * that widens W_hh^T. Those two are what a refusal here names. */
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(),
                               "the persistent recurrence does not fit on chip at hidden_size %u with %u-byte "
                               "elements, even with ONE batch row on chip: h_{t-1} is held whole in L1 as "
                               "[1, %u] fp32 and the narrow dtypes stage the weight widening in UB. The batch "
                               "(%u) is not what refuses this -- the grid walks it in blocks.",
                               hidden, inBytes, hidden, batch);
        return ge::GRAPH_FAILED;
    }
    uint32_t tChunk = 0;
    uint32_t kChunk = 0;
    uint32_t projNChunk = 0;
    if (!SingleLayerLstmBudget::PickProjTiling(rowsPerBlock, inputSize, hidden, timeStep, inBytes, &tChunk, &kChunk,
                                               &projNChunk)) {
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(),
                               "the input projection does not fit on chip at (T=%u, B=%u, I=%u, H=%u) with "
                               "%u-byte elements",
                               timeStep, batch, inputSize, hidden, inBytes);
        return ge::GRAPH_FAILED;
    }

    tilingData->batch = batch;
    tilingData->inputSize = inputSize;
    tilingData->hiddenSize = hidden;
    tilingData->logicalInputSize = static_cast<uint32_t>(logicalInput);
    tilingData->logicalHiddenSize = static_cast<uint32_t>(logicalHidden);
    tilingData->timeStep = timeStep;
    tilingData->seqLenMax = ResolveSeqLenMax(context, timeStep);
    tilingData->rowsPerBlock = rowsPerBlock;
    /* CAPPED AT MAX_CLUSTERS, because rowsPerBlock is now an on-chip extent and no longer scales
     * with the batch: at B=16384 with 32 rows on chip the batch needs 512 blocks, and the part has
     * 16 clusters to run them on. Every cluster below the cap gets at least one block, which is
     * what the kernel's stride loop and the widening prologue's band assignment both assume. */
    tilingData->blockDim = SingleLayerLstmCube::CeilDiv(batch, rowsPerBlock);
    if (tilingData->blockDim > SingleLayerLstmBudget::MAX_CLUSTERS) {
        tilingData->blockDim = SingleLayerLstmBudget::MAX_CLUSTERS;
    }
    tilingData->tChunk = tChunk;
    tilingData->kChunk = kChunk;
    tilingData->projNChunk = projNChunk;
    /* Phase A on the vector units where the cube is both wasteful and imprecise. Inert (0) for
     * every other shape; see SingleLayerLstmBudget::PickVecProj. */
    uint32_t projVecKTile = 0;
    tilingData->projVec = SingleLayerLstmBudget::PickVecProj(rowsPerBlock, batch, inputSize, hidden, timeStep,
                                                             &projVecKTile) ?
                              1U :
                              0U;
    tilingData->projVecKTile = projVecKTile;

    /* Workspace, in fp32 elements. igates is phase A's output and phase B's input; hAll and cAll
     * carry slot 0 = init_h / init_c so the backward can take h_prev / c_prev as [:-1] and h / c as
     * [1:] without a copy.
     *
     * Accumulated in uint64 and checked, because the fields are uint32: T*B*4H alone passes 4 G at
     * T=4, B=16384, H=4096, a shape this operator now accepts. In uint32 that wraps, every later
     * offset points inside another buffer, and the kernel writes the caller's outputs from whatever
     * it finds there -- a shape-correct result with wrong numbers. Such a shape is refused. */
    uint64_t offset = 0;
    const uint64_t igElems = static_cast<uint64_t>(timeStep) * batch * SingleLayerLstmBudget::GATES * hidden;
    const uint64_t hcElems = static_cast<uint64_t>(timeStep + 1) * batch * hidden;
    tilingData->offIgates = static_cast<uint32_t>(offset);
    offset += igElems;
    tilingData->offHAll = static_cast<uint32_t>(offset);
    offset += hcElems;
    tilingData->offCAll = static_cast<uint32_t>(offset);
    offset += hcElems;
    tilingData->offStore = static_cast<uint32_t>(offset);
    offset += igElems;
    // FP32 operand images are private workspace, never saved operator outputs.
    tilingData->offXF32 = static_cast<uint32_t>(offset);
    if (inBytes != sizeof(float)) {
        offset += static_cast<uint64_t>(timeStep) * batch * inputSize;
    }
    tilingData->offWF32 = static_cast<uint32_t>(offset);
    offset += SingleLayerLstmBudget::WeightImageFloats(inputSize, hidden, inBytes);
    tilingData->offBF32 = static_cast<uint32_t>(offset);
    if (inBytes != sizeof(float) || tilingData->hasBiasHh != 0) {
        offset += static_cast<uint64_t>(SingleLayerLstmBudget::GATES) * hidden;
    }
    if (offset > static_cast<uint64_t>(UINT32_MAX)) {
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(),
                               "the workspace this shape needs is %lu fp32 elements, past the %u an offset "
                               "field can address. T=%u B=%u H=%u puts %lu elements in igates alone; the "
                               "four intermediates are 2*(T*B*4H) + 2*((T+1)*B*H).",
                               static_cast<unsigned long>(offset), UINT32_MAX, timeStep, batch, hidden,
                               static_cast<unsigned long>(igElems));
        return ge::GRAPH_FAILED;
    }

    context->SetBlockDim(tilingData->blockDim);
    context->SetTilingKey(0);
    /* BATCH SCHEDULING. A cross-core wait only holds if the AIC and its two AIVs are dispatched as
     * a batch; without it the wait can be posted with nobody to answer it. A reference measurement
     * on another recurrent operator reports 3 hangs in 49 runs without it and 0 in 240 with,
     * concentrated on small shapes. Every timestep of this kernel rides that handshake, so it is
     * set. This has not been verified on this path here -- the equivalent measurement available to
     * us was on a raw launch, where 360 runs showed no difference, and that experiment has no power
     * against a rare hang. */
    (void)context->SetScheduleMode(1U);

    /* THE SYSTEM RESERVE IS REQUESTED HERE AND CONSUMED BY THE FRAMEWORK BEFORE THE KERNEL RUNS.
     * GetWorkspaceSizes is the TOTAL the caller must allocate; the framework then hands the kernel
     * a pointer PAST its own reserve, so the kernel treats `workspace` as the base of this
     * operator's area and adds nothing. Getting that backwards -- reserve at the front, kernel skips
     * it -- puts every AIV exactly one reserve past the end of the allocation: 507015, "The DDR
     * address of the MTE instruction is out of range". */
    size_t* workspaces = context->GetWorkspaceSizes(WORKSPACE_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = ResolveSysWorkspace(context) + static_cast<size_t>(offset) * sizeof(float);

    OP_LOGD(context->GetNodeName(),
            "SingleLayerLstm tiling: T=%u B=%u I=%u H=%u inBytes=%u seqLenMax=%u mBlk=%u blockDim=%u tChunk=%u "
            "kChunk=%u ws=%zu",
            timeStep, batch, inputSize, hidden, inBytes, tilingData->seqLenMax, rowsPerBlock, tilingData->blockDim,
            tChunk, kChunk, workspaces[0]);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSingleLayerLstm(gert::TilingParseContext* context)
{
    auto* compileInfo = context->GetCompiledInfo<SingleLayerLstmCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto platform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfo->sysWorkspaceSize = static_cast<uint64_t>(platform.GetLibApiWorkSpaceSize());
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SingleLayerLstm)
    .Tiling(SingleLayerLstmTilingFunc)
    .TilingParse<SingleLayerLstmCompileInfo>(TilingParseForSingleLayerLstm);

} // namespace optiling
