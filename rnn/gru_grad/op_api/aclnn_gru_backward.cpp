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
 * \file aclnn_gru_backward.cpp
 * \brief GRU 反向 Level2 API 实现，支持多参数 TensorList 接口
 */

#include "aclnn_gru_backward.h"
#include "gru_grad.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/slice.h"
#include "aclnn_kernels/transpose.h"
#include "level0/add.h"
#include "level0/concat.h"
#include "index/reverse_v2/op_api/reverse_v2.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

namespace {
constexpr int64_t DIM_ZERO = 0;
constexpr int64_t DIM_ONE = 1;
constexpr int64_t DIM_TWO = 2;
constexpr int64_t DIM_THREE = 3;
constexpr int64_t GATE_COUNT = 3; // GRU: reset, update, new

constexpr int64_t PARAMS_PER_LAYER_NO_BIAS_NO_BIDI = 2; // w_ih, w_hh
constexpr int64_t PARAMS_PER_LAYER_BIAS_OR_BIDI = 4;    // w_ih, w_hh, b_ih, b_hh
constexpr int64_t PARAMS_PER_LAYER_BIAS_AND_BIDI = 8; // w_ih, w_hh, b_ih, b_hh, w_ih_rev, w_hh_rev, b_ih_rev, b_hh_rev

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT,
                                                                       op::DataType::DT_FLOAT16};

static const std::initializer_list<op::Format> FORMAT_SUPPORT_LIST = {op::Format::FORMAT_ND, op::Format::FORMAT_NCL};

static bool IsFormatSupported(const aclTensor* t, const char* name)
{
    if (std::find(FORMAT_SUPPORT_LIST.begin(), FORMAT_SUPPORT_LIST.end(), t->GetStorageFormat()) ==
        FORMAT_SUPPORT_LIST.end()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s format only support ND/NCL, actual %s.", name,
                op::ToString(t->GetStorageFormat()).GetString());
        return false;
    }
    return true;
}

static bool CheckNotNull(const aclTensor* input, const aclTensor* hx, const aclTensorList* params, const aclTensor* dy,
                         const aclTensor* dh, const aclTensorList* r, const aclTensorList* z, const aclTensorList* n,
                         const aclTensorList* hn, const aclTensorList* h, aclTensor* dxOut, aclTensor* dhPrevOut,
                         aclTensorList* dparamsOut)
{
    OP_CHECK_NULL(input, return false);
    OP_CHECK_NULL(hx, return false);
    OP_CHECK_NULL(params, return false);
    OP_CHECK_NULL(dy, return false);
    OP_CHECK_NULL(dh, return false);
    OP_CHECK_NULL(r, return false);
    OP_CHECK_NULL(z, return false);
    OP_CHECK_NULL(n, return false);
    OP_CHECK_NULL(hn, return false);
    OP_CHECK_NULL(h, return false);
    OP_CHECK_NULL(dxOut, return false);
    OP_CHECK_NULL(dhPrevOut, return false);
    OP_CHECK_NULL(dparamsOut, return false);
    return true;
}

static bool CheckDtypeConsistent(const aclTensor* ref, const char* name, ge::DataType baseDtype)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(ref, DTYPE_SUPPORT_LIST, return false);
    if (ref->GetDataType() != baseDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s dtype inconsistent, expected %s, actual %s.", name,
                op::ToString(baseDtype).GetString(), op::ToString(ref->GetDataType()).GetString());
        return false;
    }
    if (!IsFormatSupported(ref, name)) {
        return false;
    }
    return true;
}

static bool CheckTensorListDtype(const aclTensorList* list, const char* name, ge::DataType baseDtype)
{
    OP_CHECK_NULL(list, return false);
    for (uint64_t i = 0; i < list->Size(); ++i) {
        auto* tensor = (*list)[i];
        OP_CHECK_NULL(tensor, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(tensor, DTYPE_SUPPORT_LIST, return false);
        if (tensor->GetDataType() != baseDtype) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s[%llu] dtype inconsistent, expected %s, actual %s.", name,
                    static_cast<unsigned long long>(i), op::ToString(baseDtype).GetString(),
                    op::ToString(tensor->GetDataType()).GetString());
            return false;
        }
        if (!IsFormatSupported(tensor, name)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s[%llu] format invalid", name, static_cast<unsigned long long>(i));
            return false;
        }
    }
    return true;
}

static bool CheckDtypeValid(const aclTensor* input, const aclTensor* hx, const aclTensorList* params,
                            const aclTensor* dy, const aclTensor* dh, const aclTensorList* r, const aclTensorList* z,
                            const aclTensorList* n, const aclTensorList* hn, const aclTensorList* h,
                            const aclTensor* batchSizes, aclTensor* dxOut, aclTensor* dhPrevOut,
                            aclTensorList* dparamsOut)
{
    ge::DataType baseDtype = input->GetDataType();
    if (!CheckDtypeConsistent(input, "input", baseDtype))
        return false;
    if (!CheckDtypeConsistent(hx, "hx", baseDtype))
        return false;
    if (!CheckDtypeConsistent(dy, "dy", baseDtype))
        return false;
    if (!CheckDtypeConsistent(dh, "dh", baseDtype))
        return false;
    if (!CheckDtypeConsistent(dxOut, "dxOut", baseDtype))
        return false;
    if (!CheckDtypeConsistent(dhPrevOut, "dhPrevOut", baseDtype))
        return false;

    if (!CheckTensorListDtype(params, "params", baseDtype))
        return false;
    if (!CheckTensorListDtype(r, "r", baseDtype))
        return false;
    if (!CheckTensorListDtype(z, "z", baseDtype))
        return false;
    if (!CheckTensorListDtype(n, "n", baseDtype))
        return false;
    if (!CheckTensorListDtype(hn, "hn", baseDtype))
        return false;
    if (!CheckTensorListDtype(h, "h", baseDtype))
        return false;
    if (!CheckTensorListDtype(dparamsOut, "dparamsOut", baseDtype))
        return false;

    if (batchSizes != nullptr) {
        if (batchSizes->GetStorageFormat() != Format::FORMAT_ND) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "batchSizes format only support ND");
            return false;
        }
    }
    return true;
}

static bool ValidateShape(const aclTensor* tensor, const std::vector<int64_t>& expected, const char* name)
{
    auto shape = tensor->GetViewShape();
    if (shape.GetDimNum() != static_cast<int64_t>(expected.size())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s has wrong dimension count, expected %zu, actual %u.", name,
                expected.size(), shape.GetDimNum());
        return false;
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        if (expected[i] != shape.GetDim(i)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s dim %zu mismatch, expected %ld, actual %ld.", name, i, expected[i],
                    shape.GetDim(i));
            return false;
        }
    }
    return true;
}

static bool CheckInputHxShape(const aclTensor* input, const aclTensor* hx, const aclTensor* batchSizes, bool batchFirst,
                              int64_t numLayers, bool bidirection, int64_t& T, int64_t& B, int64_t& I, int64_t& H)
{
    auto xShape = input->GetViewShape();
    bool isPacked = (batchSizes != nullptr);
    if (isPacked) {
        OP_CHECK_WRONG_DIMENSION(input, DIM_TWO, return false);
        auto bsShape = batchSizes->GetViewShape();
        T = bsShape.GetDim(DIM_ZERO);
        auto hxShape = hx->GetViewShape();
        B = hxShape.GetDim(DIM_ONE);
        I = xShape.GetDim(DIM_ONE);
    } else {
        OP_CHECK_WRONG_DIMENSION(input, DIM_THREE, return false);
        T = xShape[batchFirst ? DIM_ONE : DIM_ZERO];
        B = xShape[batchFirst ? DIM_ZERO : DIM_ONE];
        I = xShape[DIM_TWO];
    }
    CHECK_RET(T > 0 && B > 0 && I > 0, false);

    int64_t D = bidirection ? 2 : 1;
    auto hxShape = hx->GetViewShape();
    H = hxShape[hxShape.GetDimNum() - 1];
    CHECK_RET(H > 0, false);

    if (isPacked) {
        std::vector<int64_t> expected = {xShape.GetDim(DIM_ZERO), I};
        if (!ValidateShape(input, expected, "input"))
            return false;
    } else if (batchFirst) {
        std::vector<int64_t> expected = {B, T, I};
        if (!ValidateShape(input, expected, "input"))
            return false;
    } else {
        std::vector<int64_t> expected = {T, B, I};
        if (!ValidateShape(input, expected, "input"))
            return false;
    }

    std::vector<int64_t> hxExpected = {D * numLayers, B, H};
    if (!ValidateShape(hx, hxExpected, "hx"))
        return false;
    return true;
}

static bool CheckDyDhShape(const aclTensor* dy, const aclTensor* dh, const aclTensor* batchSizes, bool batchFirst,
                           int64_t numLayers, bool bidirection, int64_t T, int64_t B, int64_t H)
{
    int64_t D = bidirection ? 2 : 1;
    auto xShape = dy->GetViewShape();
    bool isPacked = (batchSizes != nullptr);

    std::vector<int64_t> dyExpected;
    if (isPacked) {
        dyExpected = {xShape.GetDim(DIM_ZERO), H * D};
    } else if (batchFirst) {
        dyExpected = {B, T, H * D};
    } else {
        dyExpected = {T, B, H * D};
    }
    if (!ValidateShape(dy, dyExpected, "dy"))
        return false;

    std::vector<int64_t> dhExpected = {numLayers * D, B, H};
    if (!ValidateShape(dh, dhExpected, "dh"))
        return false;
    return true;
}

static bool CheckGateLists(const aclTensorList* r, const aclTensorList* z, const aclTensorList* n,
                           const aclTensorList* hn, const aclTensorList* h, const aclTensor* batchSizes,
                           int64_t numLayers, bool bidirection, int64_t T, int64_t B, int64_t H, int64_t totalSteps)
{
    int64_t D = bidirection ? 2 : 1;
    bool isPacked = (batchSizes != nullptr);
    uint64_t expectedGateLen = static_cast<uint64_t>(D * numLayers);
    std::vector<int64_t> gateExpected = isPacked ? std::vector<int64_t>{totalSteps, H} : std::vector<int64_t>{T, B, H};
    const char* gateNames[] = {"r", "z", "n", "hn", "h"};
    const aclTensorList* gateLists[] = {r, z, n, hn, h};
    constexpr size_t GATE_LIST_COUNT = sizeof(gateLists) / sizeof(gateLists[0]);
    for (size_t g = 0; g < GATE_LIST_COUNT; ++g) {
        if (gateLists[g]->Size() != expectedGateLen) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s list size mismatch, expected %llu, actual %llu.", gateNames[g],
                    static_cast<unsigned long long>(expectedGateLen),
                    static_cast<unsigned long long>(gateLists[g]->Size()));
            return false;
        }
        for (uint64_t i = 0; i < gateLists[g]->Size(); ++i) {
            if (!ValidateShape((*gateLists[g])[i], gateExpected, gateNames[g]))
                return false;
        }
    }
    return true;
}

static bool CheckOutputsShape(const aclTensor* input, const aclTensorList* params, const aclTensor* dxOut,
                              const aclTensor* dhPrevOut, const aclTensorList* dparamsOut, int64_t numLayers,
                              bool bidirection, bool hasBias, int64_t B, int64_t H)
{
    int64_t D = bidirection ? 2 : 1;
    int64_t paramsPerLayer = hasBias ? (bidirection ? PARAMS_PER_LAYER_BIAS_AND_BIDI : PARAMS_PER_LAYER_BIAS_OR_BIDI) :
                                       (bidirection ? PARAMS_PER_LAYER_BIAS_OR_BIDI : PARAMS_PER_LAYER_NO_BIAS_NO_BIDI);
    uint64_t expectedParamsLen = static_cast<uint64_t>(paramsPerLayer * numLayers);
    if (params->Size() != expectedParamsLen) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "params list size mismatch, expected %llu, actual %llu.",
                static_cast<unsigned long long>(expectedParamsLen), static_cast<unsigned long long>(params->Size()));
        return false;
    }

    auto xShape = input->GetViewShape();
    std::vector<int64_t> xShapeVec(static_cast<size_t>(xShape.GetDimNum()));
    for (size_t i = 0; i < static_cast<size_t>(xShape.GetDimNum()); ++i) {
        xShapeVec[i] = xShape.GetDim(i);
    }
    if (!ValidateShape(dxOut, xShapeVec, "dxOut"))
        return false;

    std::vector<int64_t> dhPrevOutExpected = {D * numLayers, B, H};
    if (!ValidateShape(dhPrevOut, dhPrevOutExpected, "dhPrevOut"))
        return false;

    if (dparamsOut->Size() != expectedParamsLen) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dparamsOut list size mismatch, expected %llu, actual %llu.",
                static_cast<unsigned long long>(expectedParamsLen),
                static_cast<unsigned long long>(dparamsOut->Size()));
        return false;
    }
    return true;
}

static bool CheckShapeValid(const aclTensor* input, const aclTensor* hx, const aclTensorList* params,
                            const aclTensor* dy, const aclTensor* dh, const aclTensorList* r, const aclTensorList* z,
                            const aclTensorList* n, const aclTensorList* hn, const aclTensorList* h,
                            const aclTensor* batchSizes, bool hasBias, int64_t numLayers, bool bidirection,
                            bool batchFirst, aclTensor* dxOut, aclTensor* dhPrevOut, aclTensorList* dparamsOut)
{
    int64_t T = 0, B = 0, I = 0, H = 0;
    if (!CheckInputHxShape(input, hx, batchSizes, batchFirst, numLayers, bidirection, T, B, I, H))
        return false;
    if (!CheckDyDhShape(dy, dh, batchSizes, batchFirst, numLayers, bidirection, T, B, H))
        return false;
    // packed 时 totalSteps = input 的 dim0（2D compact [totalSteps, I]）
    int64_t totalSteps = T;
    if (batchSizes != nullptr) {
        auto xShape = input->GetViewShape();
        totalSteps = xShape.GetDim(DIM_ZERO);
    }
    if (!CheckGateLists(r, z, n, hn, h, batchSizes, numLayers, bidirection, T, B, H, totalSteps))
        return false;
    if (!CheckOutputsShape(input, params, dxOut, dhPrevOut, dparamsOut, numLayers, bidirection, hasBias, B, H))
        return false;
    return true;
}

static aclnnStatus CheckParams(const aclTensor* input, const aclTensor* hx, const aclTensorList* params,
                               const aclTensor* dy, const aclTensor* dh, const aclTensorList* r, const aclTensorList* z,
                               const aclTensorList* n, const aclTensorList* hn, const aclTensorList* h,
                               const aclTensor* batchSizes, bool hasBias, int64_t numLayers, bool bidirection,
                               bool batchFirst, aclTensor* dxOut, aclTensor* dhPrevOut, aclTensorList* dparamsOut)
{
    CHECK_RET(CheckNotNull(input, hx, params, dy, dh, r, z, n, hn, h, dxOut, dhPrevOut, dparamsOut),
              ACLNN_ERR_PARAM_NULLPTR);

    CHECK_RET(CheckDtypeValid(input, hx, params, dy, dh, r, z, n, hn, h, batchSizes, dxOut, dhPrevOut, dparamsOut),
              ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckShapeValid(input, hx, params, dy, dh, r, z, n, hn, h, batchSizes, hasBias, numLayers, bidirection,
                              batchFirst, dxOut, dhPrevOut, dparamsOut),
              ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static const aclTensorList* MakeContiguousList(const aclTensorList* list, aclOpExecutor* executor)
{
    std::vector<const aclTensor*> contiguousTensors;
    for (uint64_t i = 0; i < list->Size(); ++i) {
        auto* c = l0op::Contiguous((*list)[i], executor);
        CHECK_RET(c != nullptr, nullptr);
        contiguousTensors.push_back(c);
    }
    return executor->AllocTensorList(contiguousTensors.data(), contiguousTensors.size());
}

static const aclTensorList* CastListElements(const aclTensorList* list, op::DataType dstDtype, aclOpExecutor* executor)
{
    std::vector<const aclTensor*> castTensors;
    for (uint64_t i = 0; i < list->Size(); ++i) {
        auto* c = l0op::Cast((*list)[i], dstDtype, executor);
        CHECK_RET(c != nullptr, nullptr);
        castTensors.push_back(c);
    }
    return executor->AllocTensorList(castTensors.data(), castTensors.size());
}

struct GruGradLayerOut {
    const aclTensor* dx{nullptr};     // [T, B, inputSize_l]
    const aclTensor* dhPrev{nullptr}; // [1, B, H]
    const aclTensor* dwInput{nullptr};
    const aclTensor* dwHidden{nullptr};
    const aclTensor* dbInput{nullptr};
    const aclTensor* dbHidden{nullptr};
};

static const aclTensor* SliceRowAt(const aclTensor* t, int64_t i, int64_t B, int64_t H, aclOpExecutor* executor)
{
    const int64_t offData[] = {i, 0, 0};
    aclIntArray* offs = executor->AllocIntArray(offData, DIM_THREE);
    const int64_t sizeData[] = {1, B, H};
    aclIntArray* size = executor->AllocIntArray(sizeData, DIM_THREE);
    return l0op::Slice(t, offs, size, executor);
}

static aclnnStatus GruBackwardSingleLayerDirec(int64_t layerIdx, int64_t directIdx, int64_t D, bool hasBias,
                                               const aclTensor* x, const aclTensor* dyDir, const aclTensor* dh,
                                               const aclTensor* hx, const aclTensorList* params, const aclTensorList* r,
                                               const aclTensorList* z, const aclTensorList* n, const aclTensorList* hn,
                                               const aclTensorList* h, const aclTensor* batchSizes,
                                               GruGradLayerOut& out, aclOpExecutor* executor)
{
    int64_t idx = layerIdx * D + directIdx;
    int64_t groupLen = hasBias ? 4 : 2;
    int64_t B = hx->GetViewShape().GetDim(DIM_ONE);
    int64_t H = hx->GetViewShape().GetDim(DIM_TWO);

    auto initH = SliceRowAt(hx, idx, B, H, executor);
    CHECK_RET(initH != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto dhDir = SliceRowAt(dh, idx, B, H, executor);
    CHECK_RET(dhDir != nullptr, ACLNN_ERR_INNER_NULLPTR);

    int64_t pOff = idx * groupLen;
    auto wInput = (*params)[pOff + 0];
    auto wHidden = (*params)[pOff + 1];
    CHECK_RET(wInput != nullptr && wHidden != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    auto resetGate = (*r)[idx];
    auto updateGate = (*z)[idx];
    auto newGate = (*n)[idx];
    auto hiddenNew = (*hn)[idx];
    auto outputH = (*h)[idx];
    CHECK_RET(resetGate != nullptr && updateGate != nullptr && newGate != nullptr && hiddenNew != nullptr &&
                  outputH != nullptr,
              ACLNN_ERR_PARAM_NULLPTR);

    const char* direction = (directIdx == 0) ? "UNIDIRECTIONAL" : "REDIRECTIONAL";
    auto l0out = l0op::GruGrad(x, wInput, wHidden, initH, outputH, resetGate, updateGate, newGate, hiddenNew, dyDir,
                               dhDir, batchSizes, direction, hasBias, false, executor);
    CHECK_RET(l0out[l0op::OUT_DX_INDEX] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0out[l0op::OUT_DH_PREV_INDEX] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0out[l0op::OUT_DW_INPUT_INDEX] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0out[l0op::OUT_DW_HIDDEN_INDEX] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0out[l0op::OUT_DB_INPUT_INDEX] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0out[l0op::OUT_DB_HIDDEN_INDEX] != nullptr, ACLNN_ERR_INNER_NULLPTR);

    out.dx = l0out[l0op::OUT_DX_INDEX];
    out.dhPrev = l0out[l0op::OUT_DH_PREV_INDEX];
    out.dwInput = l0out[l0op::OUT_DW_INPUT_INDEX];
    out.dwHidden = l0out[l0op::OUT_DW_HIDDEN_INDEX];
    out.dbInput = l0out[l0op::OUT_DB_INPUT_INDEX];
    out.dbHidden = l0out[l0op::OUT_DB_HIDDEN_INDEX];
    return ACLNN_SUCCESS;
}

struct GruBackwardPreparedInputs {
    const aclTensor* input{nullptr};
    const aclTensor* hx{nullptr};
    const aclTensorList* params{nullptr};
    const aclTensor* dy{nullptr};
    const aclTensor* dh{nullptr};
    const aclTensorList* r{nullptr};
    const aclTensorList* z{nullptr};
    const aclTensorList* n{nullptr};
    const aclTensorList* hn{nullptr};
    const aclTensorList* h{nullptr};
    const aclTensor* batchSizes{nullptr};
    bool needCast{false};
};

static const aclTensor* TransposeBatchFirstIfNeeded(const aclTensor* t, bool batchFirst, const aclTensor* batchSizes,
                                                    aclOpExecutor* executor)
{
    if (!(batchFirst && batchSizes == nullptr)) {
        return t;
    }
    std::vector<int64_t> perm = {1, 0, 2};
    auto valuePerm = executor->AllocIntArray(perm.data(), 3);
    return l0op::Transpose(t, valuePerm, executor);
}

static aclnnStatus PrepareGruBackwardInputs(const aclTensor* input, const aclTensor* hx, const aclTensorList* params,
                                            const aclTensor* dy, const aclTensor* dh, const aclTensorList* r,
                                            const aclTensorList* z, const aclTensorList* n, const aclTensorList* hn,
                                            const aclTensorList* h, const aclTensor* batchSizes, bool batchFirst,
                                            GruBackwardPreparedInputs& out, aclOpExecutor* executor)
{
    out.input = TransposeBatchFirstIfNeeded(l0op::Contiguous(input, executor), batchFirst, batchSizes, executor);
    CHECK_RET(out.input != nullptr, ACLNN_ERR_INNER_NULLPTR);
    out.hx = l0op::Contiguous(hx, executor);
    CHECK_RET(out.hx != nullptr, ACLNN_ERR_INNER_NULLPTR);
    out.params = MakeContiguousList(params, executor);
    CHECK_RET(out.params != nullptr, ACLNN_ERR_INNER_NULLPTR);
    out.dy = TransposeBatchFirstIfNeeded(l0op::Contiguous(dy, executor), batchFirst, batchSizes, executor);
    CHECK_RET(out.dy != nullptr, ACLNN_ERR_INNER_NULLPTR);
    out.dh = l0op::Contiguous(dh, executor);
    CHECK_RET(out.dh != nullptr, ACLNN_ERR_INNER_NULLPTR);
    out.r = MakeContiguousList(r, executor);
    CHECK_RET(out.r != nullptr, ACLNN_ERR_INNER_NULLPTR);
    out.z = MakeContiguousList(z, executor);
    CHECK_RET(out.z != nullptr, ACLNN_ERR_INNER_NULLPTR);
    out.n = MakeContiguousList(n, executor);
    CHECK_RET(out.n != nullptr, ACLNN_ERR_INNER_NULLPTR);
    out.hn = MakeContiguousList(hn, executor);
    CHECK_RET(out.hn != nullptr, ACLNN_ERR_INNER_NULLPTR);
    out.h = MakeContiguousList(h, executor);
    CHECK_RET(out.h != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (batchSizes != nullptr) {
        out.batchSizes = l0op::Contiguous(batchSizes, executor);
        CHECK_RET(out.batchSizes != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CastInputsToFloat(const aclTensor* input, GruBackwardPreparedInputs& in, aclOpExecutor* executor)
{
    in.needCast = (input->GetDataType() == op::DataType::DT_FLOAT16);
    if (!in.needCast) {
        return ACLNN_SUCCESS;
    }
    in.input = l0op::Cast(in.input, op::DataType::DT_FLOAT, executor);
    CHECK_RET(in.input != nullptr, ACLNN_ERR_INNER_NULLPTR);
    in.hx = l0op::Cast(in.hx, op::DataType::DT_FLOAT, executor);
    CHECK_RET(in.hx != nullptr, ACLNN_ERR_INNER_NULLPTR);
    in.dy = l0op::Cast(in.dy, op::DataType::DT_FLOAT, executor);
    CHECK_RET(in.dy != nullptr, ACLNN_ERR_INNER_NULLPTR);
    in.dh = l0op::Cast(in.dh, op::DataType::DT_FLOAT, executor);
    CHECK_RET(in.dh != nullptr, ACLNN_ERR_INNER_NULLPTR);
    in.params = CastListElements(in.params, op::DataType::DT_FLOAT, executor);
    CHECK_RET(in.params != nullptr, ACLNN_ERR_INNER_NULLPTR);
    in.r = CastListElements(in.r, op::DataType::DT_FLOAT, executor);
    CHECK_RET(in.r != nullptr, ACLNN_ERR_INNER_NULLPTR);
    in.z = CastListElements(in.z, op::DataType::DT_FLOAT, executor);
    CHECK_RET(in.z != nullptr, ACLNN_ERR_INNER_NULLPTR);
    in.n = CastListElements(in.n, op::DataType::DT_FLOAT, executor);
    CHECK_RET(in.n != nullptr, ACLNN_ERR_INNER_NULLPTR);
    in.hn = CastListElements(in.hn, op::DataType::DT_FLOAT, executor);
    CHECK_RET(in.hn != nullptr, ACLNN_ERR_INNER_NULLPTR);
    in.h = CastListElements(in.h, op::DataType::DT_FLOAT, executor);
    CHECK_RET(in.h != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static const aclTensor* GetLayerInput(const aclTensor* inputContiguous, const aclTensorList* h, int64_t l, int64_t D,
                                      const aclTensor* batchSizes, aclOpExecutor* executor)
{
    if (l == 0) {
        return inputContiguous;
    }
    std::vector<const aclTensor*> xv;
    for (int64_t d = 0; d < D; ++d) {
        xv.push_back((*h)[(l - 1) * D + d]);
    }
    if (D == 1) {
        return xv[0];
    }
    auto tl = executor->AllocTensorList(xv.data(), xv.size());
    int64_t concatDim = (batchSizes != nullptr) ? DIM_ONE : DIM_TWO;
    return l0op::ConcatD(tl, concatDim, executor);
}

static const aclTensor* SliceDyDirection(const aclTensor* curDy, int64_t d, int64_t H, int64_t D,
                                         const aclTensor* batchSizes, int64_t B, aclOpExecutor* executor)
{
    if (D == 1) {
        return curDy;
    }
    if (batchSizes != nullptr) {
        // 不定长: dy 2D [totalSteps, H*D], 按特征维切方向
        int64_t TS = curDy->GetViewShape().GetDim(DIM_ZERO);
        const int64_t offData[] = {0, d * H};
        aclIntArray* offs = executor->AllocIntArray(offData, DIM_TWO);
        const int64_t sizeData[] = {TS, H};
        aclIntArray* size = executor->AllocIntArray(sizeData, DIM_TWO);
        return l0op::Slice(curDy, offs, size, executor);
    }
    int64_t T = curDy->GetViewShape().GetDim(DIM_ZERO);
    const int64_t offData[] = {0, 0, d * H};
    aclIntArray* offs = executor->AllocIntArray(offData, DIM_THREE);
    const int64_t sizeData[] = {T, B, H};
    aclIntArray* size = executor->AllocIntArray(sizeData, DIM_THREE);
    return l0op::Slice(curDy, offs, size, executor);
}

static aclnnStatus WriteDparamsOneDirection(const GruGradLayerOut& gout, const aclTensorList* dparamsOut, int64_t pOff,
                                            bool hasBias, bool needCast, aclOpExecutor* executor)
{
    auto dwInputOut = needCast ? l0op::Cast(gout.dwInput, op::DataType::DT_FLOAT16, executor) : gout.dwInput;
    auto dwHiddenOut = needCast ? l0op::Cast(gout.dwHidden, op::DataType::DT_FLOAT16, executor) : gout.dwHidden;
    CHECK_RET(dwInputOut != nullptr && dwHiddenOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(dwInputOut, (*dparamsOut)[pOff + 0], executor) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(dwHiddenOut, (*dparamsOut)[pOff + 1], executor) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (hasBias) {
        auto dbInputOut = needCast ? l0op::Cast(gout.dbInput, op::DataType::DT_FLOAT16, executor) : gout.dbInput;
        auto dbHiddenOut = needCast ? l0op::Cast(gout.dbHidden, op::DataType::DT_FLOAT16, executor) : gout.dbHidden;
        CHECK_RET(dbInputOut != nullptr && dbHiddenOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(l0op::ViewCopy(dbInputOut, (*dparamsOut)[pOff + 2], executor) != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(l0op::ViewCopy(dbHiddenOut, (*dparamsOut)[pOff + 3], executor) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus BackwardAllLayers(const GruBackwardPreparedInputs& in, int64_t D, int64_t H, int64_t B,
                                     int64_t groupLen, bool hasBias, int64_t numLayers, const aclTensorList* dparamsOut,
                                     const aclTensor*& finalDx, std::vector<const aclTensor*>& dhPrevVec,
                                     aclOpExecutor* executor)
{
    const aclTensor* curDy = in.dy;
    for (int64_t l = numLayers - 1; l >= 0; --l) {
        const aclTensor* xL = GetLayerInput(in.input, in.h, l, D, in.batchSizes, executor);
        CHECK_RET(xL != nullptr, ACLNN_ERR_INNER_NULLPTR);
        const aclTensor* dxLayer = nullptr;
        for (int64_t d = 0; d < D; ++d) {
            const aclTensor* dyDir = SliceDyDirection(curDy, d, H, D, in.batchSizes, B, executor);
            CHECK_RET(dyDir != nullptr, ACLNN_ERR_INNER_NULLPTR);
            GruGradLayerOut gout;
            auto ret = GruBackwardSingleLayerDirec(l, d, D, hasBias, xL, dyDir, in.dh, in.hx, in.params, in.r, in.z,
                                                   in.n, in.hn, in.h, in.batchSizes, gout, executor);
            CHECK_RET(ret == ACLNN_SUCCESS, ret);
            dhPrevVec[static_cast<size_t>(l * D + d)] = gout.dhPrev;
            ret = WriteDparamsOneDirection(gout, dparamsOut, (l * D + d) * groupLen, hasBias, in.needCast, executor);
            CHECK_RET(ret == ACLNN_SUCCESS, ret);
            dxLayer = (dxLayer == nullptr) ? gout.dx : l0op::Add(dxLayer, gout.dx, executor);
            CHECK_RET(dxLayer != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
        curDy = dxLayer;
    }
    finalDx = curDy;
    return ACLNN_SUCCESS;
}

static aclnnStatus WriteBackwardResults(const aclTensor* dx, const std::vector<const aclTensor*>& dhPrevVec,
                                        bool needCast, bool batchFirst, const aclTensor* batchSizes, aclTensor* dxOut,
                                        aclTensor* dhPrevOut, aclOpExecutor* executor)
{
    const aclTensor* curDy = TransposeBatchFirstIfNeeded(dx, batchFirst, batchSizes, executor);
    CHECK_RET(curDy != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (needCast) {
        curDy = l0op::Cast(curDy, op::DataType::DT_FLOAT16, executor);
        CHECK_RET(curDy != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    CHECK_RET(l0op::ViewCopy(curDy, dxOut, executor) != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto dhPrevList = executor->AllocTensorList(dhPrevVec.data(), dhPrevVec.size());
    CHECK_RET(dhPrevList != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor* dhPrevConcat = l0op::ConcatD(dhPrevList, DIM_ZERO, executor);
    CHECK_RET(dhPrevConcat != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (needCast) {
        dhPrevConcat = l0op::Cast(dhPrevConcat, op::DataType::DT_FLOAT16, executor);
        CHECK_RET(dhPrevConcat != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    CHECK_RET(l0op::ViewCopy(dhPrevConcat, dhPrevOut, executor) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

} // namespace
aclnnStatus aclnnGRUBackwardGetWorkspaceSize(const aclTensor* input, const aclTensorList* params, const aclTensor* hx,
                                             const aclTensor* dy, const aclTensor* dh, const aclTensorList* r,
                                             const aclTensorList* z, const aclTensorList* n, const aclTensorList* hn,
                                             const aclTensorList* h, const aclTensor* batchSizes, bool hasBias,
                                             int64_t numLayers, bool bidirection, bool batchFirst, aclTensor* dxOut,
                                             aclTensor* dhPrevOut, aclTensorList* dparamsOut, uint64_t* workspaceSize,
                                             aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(
        aclnnGRUBackward,
        DFX_IN(input, params, hx, dy, dh, r, z, n, hn, h, batchSizes, hasBias, numLayers, bidirection, batchFirst),
        DFX_OUT(dxOut, dhPrevOut, dparamsOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(input, hx, params, dy, dh, r, z, n, hn, h, batchSizes, hasBias, numLayers, bidirection,
                           batchFirst, dxOut, dhPrevOut, dparamsOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    GruBackwardPreparedInputs in;
    ret = PrepareGruBackwardInputs(input, hx, params, dy, dh, r, z, n, hn, h, batchSizes, batchFirst, in,
                                   uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    ret = CastInputsToFloat(input, in, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    int64_t D = bidirection ? 2 : 1;
    int64_t H = in.hx->GetViewShape().GetDim(DIM_TWO);
    int64_t B = in.hx->GetViewShape().GetDim(DIM_ONE);
    int64_t groupLen = hasBias ? 4 : 2;
    std::vector<const aclTensor*> dhPrevVec(static_cast<size_t>(D * numLayers), nullptr);

    const aclTensor* dx = nullptr;
    ret = BackwardAllLayers(in, D, H, B, groupLen, hasBias, numLayers, dparamsOut, dx, dhPrevVec, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    ret = WriteBackwardResults(dx, dhPrevVec, in.needCast, batchFirst, in.batchSizes, dxOut, dhPrevOut,
                               uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnGRUBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnGRUBackward);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
