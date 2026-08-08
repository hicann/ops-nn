/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MAX_POOL3_D_H_
#define MAX_POOL3_D_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#if defined(__CCE_AICORE__) && defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 5102))
#include "simt_api/cpp/kernel_simt_intf.h"
#define MAX_POOL3D_ENABLE_SIMT 1
#else
#define MAX_POOL3D_ENABLE_SIMT 0
#endif
#include "max_pool3_d_tiling_data.h"
#include "max_pool3_d_tiling_key.h"
#include "op_kernel/platform_util.h"

#ifdef __CCE_KT_TEST__
#ifndef LAUNCH_BOUND
#define LAUNCH_BOUND(threads)
#endif
#endif

namespace MaxPool3DExp {

using namespace AscendC;

constexpr uint32_t FORMAT_NDHWC_VALUE = 0;
constexpr uint32_t FORMAT_NCDHW_VALUE = 1;
constexpr uint32_t OUTPUT_LAYOUT_NDC1HWC0_VALUE = 1;
constexpr uint32_t INPUT_LAYOUT_NDC1HWC0_VALUE = 1;
constexpr uint32_t OUTPUT_TILE_NUM = 5888;
constexpr uint32_t INPUT_TILE_NUM = 23040;
constexpr uint32_t NCDHW_STRIDE2_TMP_TILE_NUM = 9216;
constexpr uint32_t NDC1HWC0_D3H3_OUTPUT_TILE_NUM = 14336;
constexpr uint32_t NDC1HWC0_SAFE_GATHER_COUNT = 240;
constexpr uint32_t NDHWC_STRIDE2_HBLOCK_ROWS = 3;
constexpr uint32_t NDHWC_STRIDE2_DTHENW_ROWS = 2;
constexpr uint32_t NDHWC_STRIDE2_LARGE_OUTPUT_TILE_NUM = OUTPUT_TILE_NUM;
constexpr uint32_t NDHWC_STRIDE2_LARGE_INPUT_TILE_NUM = INPUT_TILE_NUM;
constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t UB_BLOCK_BYTES = Ops::Base::GetUbBlockSize();
#ifdef __DAV_FPGA__
constexpr uint32_t SIMT_THREAD_NUM = 128;
#else
constexpr uint32_t SIMT_THREAD_NUM = 256;
#endif
constexpr uint32_t SIMT_TILING_FIELD_NUM = 32;
constexpr uint32_t SIMT_PARAM_NUM = 20;
constexpr uint32_t SIMT_REGULAR_MAX_TOTAL_OUT = 8000000;
constexpr uint32_t SIMT_HALF_REGULAR_MAX_TOTAL_OUT = 8000000;

#if MAX_POOL3D_ENABLE_SIMT
struct MaxPool3DSimtLiteTilingData {
    int64_t totalOut = 0;
    int64_t n = 0;
    int64_t inD = 0;
    int64_t inH = 0;
    int64_t inW = 0;
    int64_t c = 0;
    int64_t outD = 0;
    int64_t outH = 0;
    int64_t outW = 0;
    int64_t kD = 0;
    int64_t kH = 0;
    int64_t kW = 0;
    int64_t sD = 0;
    int64_t sH = 0;
    int64_t sW = 0;
    int64_t padFront = 0;
    int64_t padTop = 0;
    int64_t padLeft = 0;
    int64_t dilationD = 0;
    int64_t dilationH = 0;
    int64_t dilationW = 0;
    int64_t dataFormat = 0;
    int64_t outputLayout = 0;
    int64_t outputD = 0;
    int64_t outputH = 0;
    int64_t outputW = 0;
    int64_t outputC1 = 0;
    int64_t outputC0 = 0;
    int64_t outputC0Block = 0;
};

template <typename X_T>
__simt_callee__ __aicore__ inline X_T SimtZeroValue()
{
    if constexpr (AscendC::Std::is_same<X_T, bfloat16_t>::value) {
        constexpr uint16_t zeroBits = 0U;
        return *reinterpret_cast<const bfloat16_t*>(&zeroBits);
    } else if constexpr (AscendC::Std::is_same<X_T, half>::value) {
        constexpr uint16_t zeroBits = 0U;
        return *reinterpret_cast<const half*>(&zeroBits);
    } else {
        return 0.0F;
    }
}

template <typename X_T>
__simt_callee__ __aicore__ inline X_T SimtNegInfValue()
{
    if constexpr (AscendC::Std::is_same<X_T, bfloat16_t>::value) {
        constexpr uint16_t negInfBits = 0xFF80U;
        return *reinterpret_cast<const bfloat16_t*>(&negInfBits);
    } else if constexpr (AscendC::Std::is_same<X_T, half>::value) {
        constexpr uint16_t negInfBits = 0xFC00U;
        return *reinterpret_cast<const half*>(&negInfBits);
    } else {
        constexpr uint32_t negInfBits = 0xFF800000U;
        return *reinterpret_cast<const float*>(&negInfBits);
    }
}

template <typename X_T>
__simt_callee__ __aicore__ inline float SimtValueToFloat(X_T value)
{
    if constexpr (AscendC::Std::is_same<X_T, bfloat16_t>::value) {
        return ToFloat(value);
    } else {
        return static_cast<float>(value);
    }
}

template <typename X_T>
__simt_callee__ __aicore__ inline uint32_t SimtInputOffset(__ubuf__ MaxPool3DSimtLiteTilingData* tiling, int32_t nIdx,
                                                           int32_t dIdx, int32_t hIdx, int32_t wIdx, int32_t cIdx)
{
    if (tiling->dataFormat == FORMAT_NCDHW_VALUE) {
        return (((static_cast<uint32_t>(nIdx) * static_cast<uint32_t>(tiling->c) + static_cast<uint32_t>(cIdx)) *
                     static_cast<uint32_t>(tiling->inD) +
                 static_cast<uint32_t>(dIdx)) *
                    static_cast<uint32_t>(tiling->inH) +
                static_cast<uint32_t>(hIdx)) *
                   static_cast<uint32_t>(tiling->inW) +
               static_cast<uint32_t>(wIdx);
    }
    return (((static_cast<uint32_t>(nIdx) * static_cast<uint32_t>(tiling->inD) + static_cast<uint32_t>(dIdx)) *
                 static_cast<uint32_t>(tiling->inH) +
             static_cast<uint32_t>(hIdx)) *
                static_cast<uint32_t>(tiling->inW) +
            static_cast<uint32_t>(wIdx)) *
               static_cast<uint32_t>(tiling->c) +
           static_cast<uint32_t>(cIdx);
}

template <typename X_T>
__simt_callee__ __aicore__ inline X_T SimtComputeValueAt(__gm__ X_T* x, __ubuf__ MaxPool3DSimtLiteTilingData* tiling,
                                                         int32_t nIdx, int32_t od, int32_t oh, int32_t ow, int32_t cIdx)
{
    X_T maxValue = SimtNegInfValue<X_T>();
    float maxValueFp32 = SimtValueToFloat(maxValue);
    const int32_t inD = static_cast<int32_t>(tiling->inD);
    const int32_t inH = static_cast<int32_t>(tiling->inH);
    const int32_t inW = static_cast<int32_t>(tiling->inW);
    const int32_t kD = static_cast<int32_t>(tiling->kD);
    const int32_t kH = static_cast<int32_t>(tiling->kH);
    const int32_t kW = static_cast<int32_t>(tiling->kW);
    const int32_t sD = static_cast<int32_t>(tiling->sD);
    const int32_t sH = static_cast<int32_t>(tiling->sH);
    const int32_t sW = static_cast<int32_t>(tiling->sW);
    const int32_t padFront = static_cast<int32_t>(tiling->padFront);
    const int32_t padTop = static_cast<int32_t>(tiling->padTop);
    const int32_t padLeft = static_cast<int32_t>(tiling->padLeft);
    const int32_t dilationD = static_cast<int32_t>(tiling->dilationD);
    const int32_t dilationH = static_cast<int32_t>(tiling->dilationH);
    const int32_t dilationW = static_cast<int32_t>(tiling->dilationW);
    int32_t dStart = od * sD - padFront;
    int32_t hStart = oh * sH - padTop;
    int32_t wStart = ow * sW - padLeft;
    const int32_t dEnd = Simt::Min(dStart + (kD - 1) * dilationD + 1, inD);
    const int32_t hEnd = Simt::Min(hStart + (kH - 1) * dilationH + 1, inH);
    const int32_t wEnd = Simt::Min(wStart + (kW - 1) * dilationW + 1, inW);
    while (dStart < 0) {
        dStart += dilationD;
    }
    while (hStart < 0) {
        hStart += dilationH;
    }
    while (wStart < 0) {
        wStart += dilationW;
    }
    for (int32_t id = dStart; id < dEnd; id += dilationD) {
        for (int32_t ih = hStart; ih < hEnd; ih += dilationH) {
            for (int32_t iw = wStart; iw < wEnd; iw += dilationW) {
                const X_T curValue = x[SimtInputOffset<X_T>(tiling, nIdx, id, ih, iw, cIdx)];
                const float cur = SimtValueToFloat(curValue);
                if (cur > maxValueFp32 || Simt::IsNan<float>(cur)) {
                    maxValue = curValue;
                    maxValueFp32 = cur;
                }
            }
        }
    }
    return maxValue;
}

struct SimtOutputIndex {
    int32_t nIdx = 0;
    int32_t od = 0;
    int32_t oh = 0;
    int32_t ow = 0;
    int32_t cIdx = 0;
};

struct SimtNdc1hwc0Storage {
    uint32_t logicalC0;
    uint32_t storageW;
    uint32_t storageH;
    uint32_t storageC1;
    uint32_t storageD;
};

__simt_callee__ __aicore__ inline bool IsCompactSimtStorage(__ubuf__ MaxPool3DSimtLiteTilingData* tiling,
                                                            uint32_t logicalC0, uint32_t validC1, uint32_t outputC0,
                                                            uint32_t outputC1)
{
    const bool packedC0Prefix = logicalC0 > 0U && outputC0 >= logicalC0 && outputC0 % logicalC0 == 0U &&
                                outputC1 * (outputC0 / logicalC0) >= validC1;
    return logicalC0 > 0U && validC1 > 0U && tiling->outputD >= tiling->outD && tiling->outputH >= tiling->outH &&
           tiling->outputW >= tiling->outW && ((outputC0 == logicalC0 && outputC1 >= validC1) || packedC0Prefix);
}

__simt_callee__ __aicore__ inline SimtNdc1hwc0Storage ResolveSimtNdc1hwc0Storage(
    __ubuf__ MaxPool3DSimtLiteTilingData* tiling)
{
    const uint32_t logicalC0 = static_cast<uint32_t>(tiling->outputC0Block > 0 ? tiling->outputC0Block :
                                                                                 tiling->outputC0);
    const uint32_t validC1 = logicalC0 == 0U ? 0U : (static_cast<uint32_t>(tiling->c) + logicalC0 - 1U) / logicalC0;
    const uint32_t outputC0 = static_cast<uint32_t>(tiling->outputC0 > 0 ? tiling->outputC0 : 0);
    const uint32_t outputC1 = static_cast<uint32_t>(tiling->outputC1 > 0 ? tiling->outputC1 : 0);
    const bool compactPrefix = IsCompactSimtStorage(tiling, logicalC0, validC1, outputC0, outputC1);
    const uint32_t storageW = compactPrefix ?
                                  static_cast<uint32_t>(tiling->outW) :
                                  static_cast<uint32_t>(tiling->outputW > 0 ? tiling->outputW : tiling->outW);
    const uint32_t storageH = compactPrefix ?
                                  static_cast<uint32_t>(tiling->outH) :
                                  static_cast<uint32_t>(tiling->outputH > 0 ? tiling->outputH : tiling->outH);
    const uint32_t storageC1 = compactPrefix ? validC1 :
                                               static_cast<uint32_t>(tiling->outputC1 > 0 ? tiling->outputC1 : 1);
    const uint32_t storageD = compactPrefix ?
                                  static_cast<uint32_t>(tiling->outD) :
                                  static_cast<uint32_t>(tiling->outputD > 0 ? tiling->outputD : tiling->outD);
    return {logicalC0, storageW, storageH, storageC1, storageD};
}

__simt_callee__ __aicore__ inline bool IsValidSimtOutputIndex(__ubuf__ MaxPool3DSimtLiteTilingData* tiling,
                                                              const SimtOutputIndex& index, int32_t cBlock)
{
    return index.nIdx < tiling->n && index.od < tiling->outD && index.oh < tiling->outH && index.ow < tiling->outW &&
           index.cIdx < tiling->c && cBlock > 0;
}

__simt_callee__ __aicore__ inline bool DecodeNdc1hwc0SimtIndex(uint32_t linear,
                                                               __ubuf__ MaxPool3DSimtLiteTilingData* tiling,
                                                               __ubuf__ uint32_t* param, SimtOutputIndex& index)
{
    const SimtNdc1hwc0Storage storage = ResolveSimtNdc1hwc0Storage(tiling);
    const uint32_t qC0 = Simt::UintDiv<uint32_t>(linear, param[8], param[9]);
    const uint32_t c0 = linear - qC0 * storage.logicalC0;
    const uint32_t qW = Simt::UintDiv<uint32_t>(qC0, param[10], param[11]);
    index.ow = static_cast<int32_t>(qC0 - qW * storage.storageW);
    const uint32_t qH = Simt::UintDiv<uint32_t>(qW, param[12], param[13]);
    index.oh = static_cast<int32_t>(qW - qH * storage.storageH);
    const uint32_t qC1 = Simt::UintDiv<uint32_t>(qH, param[14], param[15]);
    const uint32_t c1 = qH - qC1 * storage.storageC1;
    const uint32_t qD = Simt::UintDiv<uint32_t>(qC1, param[16], param[17]);
    index.od = static_cast<int32_t>(qC1 - qD * storage.storageD);
    index.nIdx = static_cast<int32_t>(qD);
    const int32_t cBlock = static_cast<int32_t>(storage.logicalC0);
    index.cIdx = static_cast<int32_t>(c1) * cBlock + static_cast<int32_t>(c0);
    return IsValidSimtOutputIndex(tiling, index, cBlock);
}

__simt_callee__ __aicore__ inline void DecodeLogicalSimtIndex(uint32_t linear,
                                                              __ubuf__ MaxPool3DSimtLiteTilingData* tiling,
                                                              __ubuf__ uint32_t* param, SimtOutputIndex& index)
{
    if (tiling->dataFormat == FORMAT_NCDHW_VALUE) {
        const uint32_t qW = Simt::UintDiv<uint32_t>(linear, param[4], param[5]);
        index.ow = static_cast<int32_t>(linear - qW * static_cast<uint32_t>(tiling->outW));
        const uint32_t qH = Simt::UintDiv<uint32_t>(qW, param[2], param[3]);
        index.oh = static_cast<int32_t>(qW - qH * static_cast<uint32_t>(tiling->outH));
        const uint32_t qD = Simt::UintDiv<uint32_t>(qH, param[0], param[1]);
        index.od = static_cast<int32_t>(qH - qD * static_cast<uint32_t>(tiling->outD));
        const uint32_t qC = Simt::UintDiv<uint32_t>(qD, param[6], param[7]);
        index.cIdx = static_cast<int32_t>(qD - qC * static_cast<uint32_t>(tiling->c));
        index.nIdx = static_cast<int32_t>(qC);
        return;
    }
    const uint32_t qC = Simt::UintDiv<uint32_t>(linear, param[6], param[7]);
    index.cIdx = static_cast<int32_t>(linear - qC * static_cast<uint32_t>(tiling->c));
    const uint32_t qW = Simt::UintDiv<uint32_t>(qC, param[4], param[5]);
    index.ow = static_cast<int32_t>(qC - qW * static_cast<uint32_t>(tiling->outW));
    const uint32_t qH = Simt::UintDiv<uint32_t>(qW, param[2], param[3]);
    index.oh = static_cast<int32_t>(qW - qH * static_cast<uint32_t>(tiling->outH));
    const uint32_t qD = Simt::UintDiv<uint32_t>(qH, param[0], param[1]);
    index.od = static_cast<int32_t>(qH - qD * static_cast<uint32_t>(tiling->outD));
    index.nIdx = static_cast<int32_t>(qD);
}

template <typename X_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREAD_NUM) inline void MaxPool3DSimtLiteCompute(
    __gm__ X_T* x, __gm__ X_T* y, __ubuf__ MaxPool3DSimtLiteTilingData* tiling, __ubuf__ uint32_t* param)
{
    for (uint32_t linear = static_cast<uint32_t>(Simt::GetBlockIdx()) * static_cast<uint32_t>(Simt::GetThreadNum()) +
                           static_cast<uint32_t>(Simt::GetThreadIdx());
         linear < static_cast<uint32_t>(tiling->totalOut);
         linear += static_cast<uint32_t>(Simt::GetBlockNum()) * static_cast<uint32_t>(Simt::GetThreadNum())) {
        SimtOutputIndex index;
        bool valid = true;
        if (tiling->outputLayout == OUTPUT_LAYOUT_NDC1HWC0_VALUE) {
            valid = DecodeNdc1hwc0SimtIndex(linear, tiling, param, index);
        } else {
            DecodeLogicalSimtIndex(linear, tiling, param, index);
        }
        if (valid) {
            y[linear] = SimtComputeValueAt<X_T>(x, tiling, index.nIdx, index.od, index.oh, index.ow, index.cIdx);
        } else {
            y[linear] = SimtZeroValue<X_T>();
        }
    }
}

#endif

template <typename T, uint32_t schMode = MAX_POOL3_D_TPL_SCH_MODE_GENERAL>
class MaxPool3DKernel {
public:
    static constexpr bool NDC_FEATURE_SCHEDULE = schMode >= MAX_POOL3_D_TPL_SCH_MODE_TINY_K3 &&
                                                 schMode != MAX_POOL3_D_TPL_SCH_MODE_NCDHW_SMALL_DEPTH_STRIDE2 &&
                                                 schMode != MAX_POOL3_D_TPL_SCH_MODE_NCDHW_POOL2_FEATURE;
    static constexpr bool NDC_K1_FEATURE_SCHEDULE = schMode == MAX_POOL3_D_TPL_SCH_MODE_K1_WIDE_FEATURE ||
                                                    schMode == MAX_POOL3_D_TPL_SCH_MODE_K1_COMPACT_FEATURE ||
                                                    schMode == MAX_POOL3_D_TPL_SCH_MODE_K1_COMPACT_LOGICAL ||
                                                    schMode == MAX_POOL3_D_TPL_SCH_MODE_K1_WIDE_FLOAT_PHYSICAL_FEATURE;
    static constexpr bool NDC_DILATED_W_FEATURE_SCHEDULE = schMode == MAX_POOL3_D_TPL_SCH_MODE_DILATED_W_FEATURE ||
                                                           schMode ==
                                                               MAX_POOL3_D_TPL_SCH_MODE_DILATED_W_FLOAT_NCDHW_FEATURE;
    static constexpr bool NDC_H_ONLY_FEATURE_SCHEDULE = schMode == MAX_POOL3_D_TPL_SCH_MODE_H_ONLY_FEATURE ||
                                                        schMode == MAX_POOL3_D_TPL_SCH_MODE_H_ONLY_HALF_NDHWC_FEATURE;
    static constexpr bool NDC_D3H3_FEATURE_SCHEDULE = schMode == MAX_POOL3_D_TPL_SCH_MODE_D3H3_PHYSICAL_FEATURE;
    static constexpr bool NDC_D2H3W2_FEATURE_SCHEDULE = schMode == MAX_POOL3_D_TPL_SCH_MODE_D2H3W2_PHYSICAL_FEATURE;
    static constexpr bool NDC_D3W3_FEATURE_SCHEDULE = schMode == MAX_POOL3_D_TPL_SCH_MODE_D3W3_FEATURE;
    static constexpr bool NDC_TINY_FEATURE_SCHEDULE = schMode == MAX_POOL3_D_TPL_SCH_MODE_TINY_K3 ||
                                                      schMode == MAX_POOL3_D_TPL_SCH_MODE_TINY_K3_HALF_GROUPED;
    static constexpr bool
        NDC_POOL2_FEATURE_SCHEDULE = schMode == MAX_POOL3_D_TPL_SCH_MODE_POOL2_DEPTH_HEAVY ||
                                     schMode == MAX_POOL3_D_TPL_SCH_MODE_POOL2_MULTI_BATCH_FEATURE ||
                                     schMode == MAX_POOL3_D_TPL_SCH_MODE_POOL2_CHANNEL_HEAVY ||
                                     schMode == MAX_POOL3_D_TPL_SCH_MODE_POOL2_HALF_WIDE_CHANNEL ||
                                     schMode == MAX_POOL3_D_TPL_SCH_MODE_POOL2_BALANCED_CHANNEL ||
                                     schMode == MAX_POOL3_D_TPL_SCH_MODE_POOL2_MULTI_BATCH_HALF_NDHWC_FEATURE;
    static_assert(!NDC_FEATURE_SCHEDULE || NDC_K1_FEATURE_SCHEDULE || NDC_DILATED_W_FEATURE_SCHEDULE ||
                      NDC_H_ONLY_FEATURE_SCHEDULE || NDC_D3H3_FEATURE_SCHEDULE || NDC_D2H3W2_FEATURE_SCHEDULE ||
                      NDC_D3W3_FEATURE_SCHEDULE || NDC_TINY_FEATURE_SCHEDULE || NDC_POOL2_FEATURE_SCHEDULE,
                  "MaxPool3D feature schedule must use a dedicated kernel route");

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const MaxPool3DTilingData* tiling)
    {
        tiling_ = tiling;
        uint64_t inputElements = tiling_->n * tiling_->inD * tiling_->inH * tiling_->inW * tiling_->c;
        if (tiling_->inputLayout == INPUT_LAYOUT_NDC1HWC0_VALUE) {
            const uint64_t inputC0Block = static_cast<uint64_t>(tiling_->inputC0Block > 0 ? tiling_->inputC0Block :
                                                                                            tiling_->inputC0);
            inputElements = tiling_->n * tiling_->inD * tiling_->inputC1 * tiling_->inH * tiling_->inW * inputC0Block;
        }
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), inputElements);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), tiling_->totalOut);
        if constexpr (NDC_FEATURE_SCHEDULE) {
            InitNdc1hwc0FeatureBuffers();
        } else {
            if (TryInitSimtBuffers() || TryInitNdc1hwc0Buffers() || TryInitNcdhwStride2Buffers() ||
                TryInitNdhwcStride2Buffers()) {
                return;
            }
            InitDefaultBuffers();
        }
    }

    __aicore__ inline bool TryInitSimtBuffers()
    {
        if constexpr (NDC_FEATURE_SCHEDULE) {
            return false;
        }
        if (!CanUseSimtFastPath()) {
            return false;
        }
        constexpr uint32_t outBytes = (OUTPUT_TILE_NUM * sizeof(T) + UB_BLOCK_BYTES - 1U) / UB_BLOCK_BYTES *
                                      UB_BLOCK_BYTES;
        pipe_.InitBuffer(simtTilingDataBuf_, SIMT_TILING_FIELD_NUM * sizeof(int64_t));
        pipe_.InitBuffer(simtParamBuf_, SIMT_PARAM_NUM * sizeof(uint32_t));
        pipe_.InitBuffer(yOutQue_, BUFFER_NUM, outBytes);
        return true;
    }

    __aicore__ inline void InitNdc1hwc0FeatureBuffers()
    {
        if constexpr (AscendC::Std::is_same<T, bfloat16_t>::value) {
            InitDefaultBuffers();
        } else {
            if (tiling_->inputLayout == INPUT_LAYOUT_NDC1HWC0_VALUE &&
                tiling_->outputLayout == OUTPUT_LAYOUT_NDC1HWC0_VALUE) {
                if constexpr (NDC_DILATED_W_FEATURE_SCHEDULE) {
                    if (CanUseNdc1hwc0InputOutputDilatedWWholeNPath()) {
                        InitNdc1hwc0InputOutputDilatedWBuffers();
                        return;
                    }
                }
                if constexpr (NDC_D3H3_FEATURE_SCHEDULE) {
                    if (CanUseNdc1hwc0InputOutputD3H3PipelinedPlanePath()) {
                        InitNdc1hwc0InputOutputD3H3PipelinedBuffers();
                        return;
                    }
                }
                if constexpr (NDC_POOL2_FEATURE_SCHEDULE) {
                    if (CanUseNdc1hwc0InputOutputPool2PackedC1PlanePath() ||
                        CanUseNdc1hwc0InputOutputPool2FullDepthPath() ||
                        CanUseNdc1hwc0InputOutputPool2DepthGroupPath() ||
                        CanUseNdc1hwc0InputOutputPool2BatchedRowPath()) {
                        constexpr uint32_t pool2InputTile = AscendC::Std::is_same<T, half>::value ?
                                                                INPUT_TILE_NUM * 2U :
                                                                INPUT_TILE_NUM;
                        pipe_.InitBuffer(calcBuf_, UbBytesForElements(pool2InputTile));
                        pipe_.InitBuffer(tmpBuf_, UbBytesForElements(OUTPUT_TILE_NUM));
                        pipe_.InitBuffer(maskBuf_, UbBytesForElements(OUTPUT_TILE_NUM));
                        return;
                    }
                }
                InitDefaultBuffers();
                return;
            }
            if constexpr (NDC_K1_FEATURE_SCHEDULE) {
                if (CanUseNdc1hwc0NdhwcK1MinimalBufferPath()) {
                    pipe_.InitBuffer(xInQue_, BUFFER_NUM, UbBytesForElements(INPUT_TILE_NUM));
                    return;
                }
            }
            if constexpr (NDC_D3H3_FEATURE_SCHEDULE) {
                if (CanUseNdc1hwc0NdhwcD3H3Dil2GroupPath() || CanUseNdc1hwc0NcdhwD3H3Dil2GroupPath()) {
                    InitNdc1hwc0D3H3Dil2Buffers();
                    return;
                }
            }
            if constexpr (NDC_D2H3W2_FEATURE_SCHEDULE) {
                if (CanUseNdc1hwc0D2H3W2Dil2PlanePath()) {
                    InitNdc1hwc0D2H3W2Dil2Buffers();
                    return;
                }
            }
            InitDefaultBuffers();
        }
    }

    __aicore__ inline bool TryInitNdc1hwc0Buffers()
    {
        if constexpr (schMode == MAX_POOL3_D_TPL_SCH_MODE_NDC1HWC0 || NDC_FEATURE_SCHEDULE) {
            if (CanUseNdc1hwc0NdhwcK1MinimalBufferPath()) {
                pipe_.InitBuffer(xInQue_, BUFFER_NUM, UbBytesForElements(INPUT_TILE_NUM));
                return true;
            }
        }
        if constexpr ((schMode == MAX_POOL3_D_TPL_SCH_MODE_GENERAL && FORMAT_Y == FORMAT_NDC1HWC0 &&
                       AscendC::Std::is_same<T, float>::value) ||
                      schMode == MAX_POOL3_D_TPL_SCH_MODE_NDC1HWC0 || NDC_FEATURE_SCHEDULE) {
            if (tiling_->outputLayout == OUTPUT_LAYOUT_NDC1HWC0_VALUE &&
                (CanUseNdc1hwc0NdhwcD3H3Dil2GroupPath() || CanUseNdc1hwc0NcdhwD3H3Dil2GroupPath())) {
                InitNdc1hwc0D3H3Dil2Buffers();
                return true;
            }
        }
        if constexpr (schMode == MAX_POOL3_D_TPL_SCH_MODE_NDC1HWC0 || NDC_FEATURE_SCHEDULE) {
            if (CanUseNdc1hwc0D2H3W2Dil2PlanePath()) {
                InitNdc1hwc0D2H3W2Dil2Buffers();
                return true;
            }
            if (tiling_->inputLayout == INPUT_LAYOUT_NDC1HWC0_VALUE) {
                InitDefaultBuffers();
                return true;
            }
        }
        return false;
    }

    __aicore__ inline bool TryInitNcdhwStride2Buffers()
    {
        if constexpr (schMode == MAX_POOL3_D_TPL_SCH_MODE_NCDHW_STRIDE2 &&
                      !AscendC::Std::is_same<T, bfloat16_t>::value) {
            const bool useScratch = (tiling_->outW >= 80 || AscendC::Std::is_same<T, half>::value) &&
                                    (CanUseNcdhwStride2MicroHBlockPath() || CanUseNcdhwStride2WholeDDirectPath() ||
                                     CanUseNcdhwStride2DPlaneDirectPath() || CanUseNcdhwStride2RowVectorPath());
            if (useScratch) {
                InitNcdhwStride2ScratchBuffers();
                return true;
            }
        }
        return false;
    }

    __aicore__ inline void InitNcdhwStride2ScratchBuffers()
    {
        constexpr uint32_t inBytes = (INPUT_TILE_NUM * sizeof(T) + UB_BLOCK_BYTES - 1U) / UB_BLOCK_BYTES *
                                     UB_BLOCK_BYTES;
        constexpr uint32_t outBytes = (OUTPUT_TILE_NUM * sizeof(T) + UB_BLOCK_BYTES - 1U) / UB_BLOCK_BYTES *
                                      UB_BLOCK_BYTES;
        constexpr uint32_t tmpBytes = (NCDHW_STRIDE2_TMP_TILE_NUM * sizeof(T) + UB_BLOCK_BYTES - 1U) / UB_BLOCK_BYTES *
                                      UB_BLOCK_BYTES;
        pipe_.InitBuffer(xInQue_, BUFFER_NUM, inBytes);
        pipe_.InitBuffer(calcBuf_, outBytes);
        pipe_.InitBuffer(tmpBuf_, tmpBytes);
        pipe_.InitBuffer(maskBuf_, outBytes);
        pipe_.InitBuffer(yOutQue_, BUFFER_NUM, outBytes);
    }

    template <uint32_t rows>
    __aicore__ inline void InitNdhwcStride2ScratchBuffers()
    {
        const uint32_t outputRowCount = static_cast<uint32_t>(tiling_->outW * tiling_->c);
        const uint32_t alignedInputRowCount = AlignToVector(outputRowCount * 2U);
        const uint32_t inputTile = alignedInputRowCount * 2U * rows;
        const uint32_t outputTile = outputRowCount * rows;
        const uint32_t uncompressedTile = alignedInputRowCount * rows;
        pipe_.InitBuffer(xInQue_, BUFFER_NUM, UbBytesForElements(inputTile));
        pipe_.InitBuffer(calcBuf_, UbBytesForElements(uncompressedTile));
        pipe_.InitBuffer(tmpBuf_, UbBytesForElements(alignedInputRowCount));
        pipe_.InitBuffer(maskBuf_, UbBytesForElements(outputTile));
        pipe_.InitBuffer(yOutQue_, BUFFER_NUM, UB_BLOCK_BYTES);
    }

    __aicore__ inline bool TryInitNdhwcStride2Buffers()
    {
        if constexpr (schMode == MAX_POOL3_D_TPL_SCH_MODE_NDHWC_STRIDE2 &&
                      !AscendC::Std::is_same<T, bfloat16_t>::value) {
            if constexpr (AscendC::Std::is_same<T, float>::value) {
                if (CanUseNdhwcFloatStride2CompactHBlockPath()) {
                    const uint32_t outputRowCount = static_cast<uint32_t>(tiling_->outW * tiling_->c);
                    const uint32_t alignedInputRowCount = AlignToVector(outputRowCount * 2U);
                    const uint32_t inputTile = alignedInputRowCount * 2U * NDHWC_STRIDE2_HBLOCK_ROWS;
                    const uint32_t uncompressedTile = alignedInputRowCount * NDHWC_STRIDE2_HBLOCK_ROWS;
                    pipe_.InitBuffer(xInQue_, BUFFER_NUM, UbBytesForElements(inputTile));
                    pipe_.InitBuffer(calcBuf_, UbBytesForElements(uncompressedTile));
                    pipe_.InitBuffer(tmpBuf_, UbBytesForElements(uncompressedTile));
                    return true;
                }
            }
            if (CanUseNdhwcStride2LargeHBlockBuffers()) {
                InitNdhwcStride2ScratchBuffers<NDHWC_STRIDE2_HBLOCK_ROWS>();
                return true;
            }
            if (CanUseNdhwcStride2TwoRowDThenWPath()) {
                InitNdhwcStride2ScratchBuffers<NDHWC_STRIDE2_DTHENW_ROWS>();
                return true;
            }
        }
        return false;
    }

    __aicore__ inline void Process()
    {
        if constexpr (schMode == MAX_POOL3_D_TPL_SCH_MODE_NCDHW_STRIDE2) {
            ProcessNcdhwStride2Schedule();
        } else if constexpr (schMode == MAX_POOL3_D_TPL_SCH_MODE_NDHWC_STRIDE2) {
            ProcessNdhwcStride2Schedule();
        } else if constexpr (NDC_FEATURE_SCHEDULE) {
            ProcessNdc1hwc0FeatureSchedule();
        } else if constexpr (schMode == MAX_POOL3_D_TPL_SCH_MODE_NDC1HWC0) {
            ProcessNdc1hwc0Mode();
        } else {
            if (TryProcessGeneralNdhwcRoute() || TryProcessGeneralNcdhwRoute()) {
                return;
            }
            ProcessGeneric();
        }
    }

    __aicore__ inline void ProcessNdc1hwc0FeatureSchedule()
    {
        if constexpr (AscendC::Std::is_same<T, bfloat16_t>::value) {
            ProcessAivDirectFast();
        } else {
            if constexpr (NDC_POOL2_FEATURE_SCHEDULE) {
                ProcessNdc1hwc0PhysicalFeatureSchedule();
                return;
            }
            if (tiling_->inputLayout == INPUT_LAYOUT_NDC1HWC0_VALUE) {
                ProcessNdc1hwc0PhysicalFeatureSchedule();
                return;
            }
            if constexpr (NDC_K1_FEATURE_SCHEDULE) {
                if (TryProcessNdc1hwc0K1Route()) {
                    return;
                }
            } else if constexpr (NDC_DILATED_W_FEATURE_SCHEDULE) {
                if (TryProcessNdc1hwc0EarlySpatialRoute()) {
                    return;
                }
            } else if constexpr (NDC_H_ONLY_FEATURE_SCHEDULE || NDC_D3H3_FEATURE_SCHEDULE) {
                if constexpr (NDC_H_ONLY_FEATURE_SCHEDULE) {
                    if (CanUseNdc1hwc0LogicalHOnlyPlanePath()) {
                        ProcessNdc1hwc0LogicalHOnlyPlanes();
                        return;
                    }
                }
                if (TryProcessNdc1hwc0MiddleSpatialRoute()) {
                    return;
                }
            } else if constexpr (NDC_D2H3W2_FEATURE_SCHEDULE || NDC_D3W3_FEATURE_SCHEDULE) {
                if constexpr (NDC_D3W3_FEATURE_SCHEDULE) {
                    if (CanUseNdc1hwc0LogicalD3W3RowGroupPath()) {
                        ProcessNdc1hwc0LogicalD3W3RowGroups();
                        return;
                    }
                }
                if (TryProcessNdc1hwc0LateSpatialRoute()) {
                    return;
                }
            } else if constexpr (NDC_TINY_FEATURE_SCHEDULE) {
                if (CanUseNdc1hwc0LogicalTinyK3WholeNPath()) {
                    ProcessNdc1hwc0LogicalTinyK3WholeN();
                    return;
                }
                if (TryProcessNdc1hwc0NdhwcStrideRoute() || TryProcessNdc1hwc0NcdhwStrideRoute() ||
                    TryProcessNdc1hwc0LateSpatialRoute()) {
                    return;
                }
            }
            ProcessAivDirectFast();
        }
    }

    __aicore__ inline void ProcessNdc1hwc0PhysicalFeatureSchedule()
    {
        if (tiling_->outputLayout != OUTPUT_LAYOUT_NDC1HWC0_VALUE) {
            ProcessAivDirectFast();
            return;
        }
        if constexpr (!AscendC::Std::is_same<T, bfloat16_t>::value) {
            if constexpr (NDC_K1_FEATURE_SCHEDULE) {
                if (CanUseNdc1hwc0InputOutputK1PackedPlanePath()) {
                    ProcessNdc1hwc0InputOutputK1PackedPlane();
                    return;
                }
            }
            if constexpr (NDC_POOL2_FEATURE_SCHEDULE) {
                if (CanUseNdc1hwc0InputOutputPool2PackedC1PlanePath()) {
                    ProcessNdc1hwc0InputOutputPool2PackedC1Planes();
                    return;
                }
                if (CanUseNdc1hwc0InputOutputPool2FullDepthPath()) {
                    ProcessNdc1hwc0InputOutputPool2FullDepth();
                    return;
                }
                if (CanUseNdc1hwc0InputOutputPool2DepthGroupPath()) {
                    ProcessNdc1hwc0InputOutputPool2DepthGroups();
                    return;
                }
                if (CanUseNdc1hwc0InputOutputPool2BatchedRowPath()) {
                    ProcessNdc1hwc0InputOutputPool2BatchedRows();
                    return;
                }
                if (CanUseNdc1hwc0InputOutputPool2PlanePath()) {
                    ProcessNdc1hwc0InputOutputPool2Plane();
                    return;
                }
                if (CanUseNdc1hwc0InputOutputPackedPool2Path()) {
                    ProcessNdc1hwc0InputOutputPackedPool2();
                    return;
                }
            }
            if constexpr (NDC_TINY_FEATURE_SCHEDULE) {
                if (CanUseNdc1hwc0InputOutputTinyK3WholeNPath()) {
                    ProcessNdc1hwc0InputOutputTinyK3WholeN();
                    return;
                }
            }
            if constexpr (NDC_H_ONLY_FEATURE_SCHEDULE) {
                if (CanUseNdc1hwc0InputOutputHOnlyPlanePath()) {
                    ProcessNdc1hwc0InputOutputHOnlyPlanes();
                    return;
                }
            }
            if constexpr (NDC_D3H3_FEATURE_SCHEDULE) {
                if (CanUseNdc1hwc0InputOutputD3H3PipelinedPlanePath()) {
                    ProcessNdc1hwc0InputOutputD3H3PipelinedPlanes();
                    return;
                }
                if (CanUseNdc1hwc0InputOutputD3H3PlanePath()) {
                    ProcessNdc1hwc0InputOutputD3H3Planes();
                    return;
                }
            }
            if constexpr (NDC_D3W3_FEATURE_SCHEDULE) {
                if (CanUseNdc1hwc0InputOutputD3W3RowGroupPath()) {
                    ProcessNdc1hwc0InputOutputD3W3RowGroups();
                    return;
                }
            }
            if constexpr (NDC_D2H3W2_FEATURE_SCHEDULE) {
                if (CanUseNdc1hwc0InputOutputD2H3W2DepthGroupPath()) {
                    ProcessNdc1hwc0InputOutputD2H3W2DepthGroups();
                    return;
                }
            }
            if constexpr (NDC_DILATED_W_FEATURE_SCHEDULE) {
                if (CanUseNdc1hwc0InputOutputDilatedWWholeNPath()) {
                    ProcessNdc1hwc0InputOutputDilatedWWholeN();
                    return;
                }
            }
            if constexpr (!NDC_K1_FEATURE_SCHEDULE) {
                if (CanUseNdc1hwc0InputOutputFeatureRowPath()) {
                    ProcessNdc1hwc0InputOutputFeatureRows();
                    return;
                }
            }
            if (CanUseNdc1hwc0InputOutputK1IdentityCopyPath()) {
                ProcessNdc1hwc0InputOutputK1IdentityCopy();
                return;
            }
            if (CanUseNdc1hwc0InputOutputK1DirectPath()) {
                ProcessNdc1hwc0InputOutputK1Direct();
                return;
            }
            if (CanUseNdc1hwc0InputOutputRowPath()) {
                ProcessNdc1hwc0InputOutputRow();
                return;
            }
        }
        ProcessAivDirectFast();
    }

    __aicore__ inline void ProcessNdc1hwc0Mode()
    {
        if constexpr (!AscendC::Std::is_same<T, bfloat16_t>::value) {
            if (tiling_->inputLayout != INPUT_LAYOUT_NDC1HWC0_VALUE &&
                (TryProcessNdc1hwc0EarlySpatialRouteBoundary() || TryProcessNdc1hwc0K1RouteBoundary() ||
                 TryProcessNdc1hwc0NdhwcStrideRouteBoundary() || TryProcessNdc1hwc0MiddleSpatialRouteBoundary() ||
                 TryProcessNdc1hwc0NcdhwStrideRouteBoundary() || TryProcessNdc1hwc0LateSpatialRouteBoundary())) {
                return;
            }
        }
        ProcessNdc1hwc0Schedule();
    }

    __aicore__ inline bool TryProcessGeneralNdhwcRoute()
    {
        if (CanUseSimtFastPath()) {
            ProcessSimtFast();
            return true;
        }
        if constexpr (schMode == MAX_POOL3_D_TPL_SCH_MODE_GENERAL && FORMAT_Y == FORMAT_NDC1HWC0 &&
                      AscendC::Std::is_same<T, float>::value) {
            if (tiling_->outputLayout == OUTPUT_LAYOUT_NDC1HWC0_VALUE && CanUseNdc1hwc0NdhwcD3H3Dil2GroupPath()) {
                ProcessNdc1hwc0NdhwcD3H3Dil2Group();
                return true;
            }
            if (tiling_->outputLayout == OUTPUT_LAYOUT_NDC1HWC0_VALUE && CanUseNdc1hwc0NcdhwD3H3Dil2GroupPath()) {
                ProcessNdc1hwc0NcdhwD3H3Dil2Group();
                return true;
            }
        }
        if constexpr (AscendC::Std::is_same<T, bfloat16_t>::value) {
            if (CanUseNdhwcBf16SmallCVectorPath()) {
                ProcessNdhwcBf16SmallCVector();
                return true;
            }
        } else {
            if constexpr (AscendC::Std::is_same<T, float>::value) {
                if (CanUseNdhwcFloatStride2CompactHBlockPath()) {
                    ProcessNdhwcFloatStride2CompactHBlockDirect();
                    return true;
                }
            }
            if (TryProcessNdhwcVectorRoute()) {
                return true;
            }
        }
        return false;
    }

    __aicore__ inline bool TryProcessNdhwcVectorRoute()
    {
        if (CanUseNdhwcFloatC3NoPad2x2x2DirectPath()) {
            ProcessNdhwcFloatC3NoPad2x2x2Direct();
        } else if (CanUseNdhwcStride2FullRowDirectPath()) {
            ProcessNdhwcStride2FullRowDirect();
        } else if (CanUseNdhwcStride2TwoRowDThenWPath()) {
            ProcessNdhwcStride2TwoRowDThenW();
        } else if (CanUseNdhwcStride2SingleRowFusedDPath()) {
            ProcessNdhwcStride2SingleRowFusedD();
        } else if (CanUseNdhwcHalfC8Stride2ScalarRowPath()) {
            ProcessNdhwcHalfC8Stride2ScalarRow();
        } else if (CanUseNdhwcStride2WBlockVectorPath()) {
            ProcessNdhwcStride2WBlockVector();
        } else if (CanUseNdhwcVectorPath()) {
            ProcessNdhwcVector();
        } else {
            return false;
        }
        return true;
    }

    __aicore__ inline bool TryProcessGeneralNcdhwRoute()
    {
        if constexpr (AscendC::Std::is_same<T, float>::value) {
            if (CanUseNcdhwFloatStride1RowReusePath()) {
                ProcessNcdhwFloatStride1RowReuse();
                return true;
            }
        }
        if constexpr (!AscendC::Std::is_same<T, bfloat16_t>::value) {
            if (CanUseNcdhwStride2MicroHBlockPath()) {
                ProcessNcdhwStride2MicroHBlock();
                return true;
            }
            if (CanUseNcdhwStride2WholeDDirectPath()) {
                ProcessNcdhwStride2WholeDDirect();
                return true;
            }
            if (CanUseNcdhwStride2DPlaneDirectPath()) {
                ProcessNcdhwStride2DPlaneDirect();
                return true;
            }
            if (CanUseNcdhwStride2RowVectorPath()) {
                ProcessNcdhwStride2RowVector();
                return true;
            }
        }
        if (CanUseNcdhwScalar2x2x2Path()) {
            ProcessNcdhwScalar2x2x2();
        } else if (CanUseNdhwcScalar2x2x2Path()) {
            ProcessNdhwcScalar2x2x2();
        } else if (CanUseAivDirectFastPath()) {
            ProcessAivDirectFast();
        } else {
            return false;
        }
        return true;
    }

private:
    __aicore__ inline void ProcessNcdhwStride2Schedule()
    {
        if constexpr (!AscendC::Std::is_same<T, bfloat16_t>::value) {
            if constexpr (AscendC::Std::is_same<T, half>::value) {
                if (CanUseNcdhwStride2WholeDDirectPath()) {
                    ProcessNcdhwStride2WholeDDirect();
                    return;
                }
                if (CanUseNcdhwStride2DPlaneDirectPath()) {
                    ProcessNcdhwStride2DPlaneDirect();
                    return;
                }
            }
            if (CanUseNcdhwStride2MicroHBlockPath()) {
                ProcessNcdhwStride2MicroHBlock();
                return;
            }
            if constexpr (!AscendC::Std::is_same<T, half>::value) {
                if (CanUseNcdhwStride2WholeDDirectPath()) {
                    ProcessNcdhwStride2WholeDDirect();
                    return;
                }
            }
            if (CanUseNcdhwStride2DPlaneDirectPath()) {
                ProcessNcdhwStride2DPlaneDirect();
                return;
            }
            if (CanUseNcdhwStride2RowVectorPath()) {
                ProcessNcdhwStride2RowVector();
                return;
            }
        }
        if (CanUseNcdhwScalar2x2x2Path()) {
            ProcessNcdhwScalar2x2x2();
            return;
        }
        ProcessGeneric();
    }

    __aicore__ inline void ProcessNdhwcStride2Schedule()
    {
        if constexpr (!AscendC::Std::is_same<T, bfloat16_t>::value) {
            if constexpr (AscendC::Std::is_same<T, float>::value) {
                if (CanUseNdhwcFloatStride2CompactHBlockPath()) {
                    ProcessNdhwcFloatStride2CompactHBlockDirect();
                    return;
                }
            }
            if (CanUseNdhwcStride2FullRowDirectPath()) {
                ProcessNdhwcStride2FullRowDirect();
                return;
            }
            if (CanUseNdhwcStride2TwoRowDThenWPath()) {
                ProcessNdhwcStride2TwoRowDThenW();
                return;
            }
            if (CanUseNdhwcStride2SingleRowFusedDPath()) {
                ProcessNdhwcStride2SingleRowFusedD();
                return;
            }
            if (CanUseNdhwcHalfC8Stride2ScalarRowPath()) {
                ProcessNdhwcHalfC8Stride2ScalarRow();
                return;
            }
            if (CanUseNdhwcStride2WBlockVectorPath()) {
                ProcessNdhwcStride2WBlockVector();
                return;
            }
            if (CanUseNdhwcVectorPath()) {
                ProcessNdhwcVector();
                return;
            }
        }
        if (CanUseNdhwcScalar2x2x2Path()) {
            ProcessNdhwcScalar2x2x2();
            return;
        }
        ProcessGeneric();
    }

    __aicore__ inline void ProcessNdc1hwc0Schedule()
    {
        if (tiling_->inputLayout == INPUT_LAYOUT_NDC1HWC0_VALUE &&
            tiling_->outputLayout == OUTPUT_LAYOUT_NDC1HWC0_VALUE) {
            if constexpr (!AscendC::Std::is_same<T, bfloat16_t>::value) {
                if (CanUseNdc1hwc0InputOutputK1DirectPath()) {
                    ProcessNdc1hwc0InputOutputK1Direct();
                    return;
                }
                if (CanUseNdc1hwc0InputOutputRowPath()) {
                    ProcessNdc1hwc0InputOutputRow();
                    return;
                }
            }
            ProcessAivDirectFast();
            return;
        }
        if (tiling_->inputLayout == INPUT_LAYOUT_NDC1HWC0_VALUE) {
            ProcessAivDirectFast();
            return;
        }
        if (tiling_->outputLayout == OUTPUT_LAYOUT_NDC1HWC0_VALUE) {
            const uint64_t ndcBlock = Ndc1hwc0Block();
            const uint64_t ndcValidC1 = Ndc1hwc0ValidC1(ndcBlock);
            if (!IsNdc1hwc0CompactStorage(ndcBlock, ndcValidC1)) {
                if constexpr (!AscendC::Std::is_same<T, bfloat16_t>::value) {
                    if (CanUseNdc1hwc0StorageRowVectorPath()) {
                        ProcessNdc1hwc0StorageRowVector();
                        return;
                    }
                }
                ProcessAivDirectFast();
                return;
            }
            if constexpr (!AscendC::Std::is_same<T, bfloat16_t>::value) {
                if (TryProcessNdc1hwc0EarlySpatialRouteBoundary() || TryProcessNdc1hwc0K1RouteBoundary() ||
                    TryProcessNdc1hwc0NdhwcStrideRouteBoundary() || TryProcessNdc1hwc0MiddleSpatialRouteBoundary() ||
                    TryProcessNdc1hwc0NcdhwStrideRouteBoundary() || TryProcessNdc1hwc0LateSpatialRouteBoundary()) {
                    return;
                }
            }
            if (CanUseNdc1hwc0NcdhwRowScalarPath()) {
                ProcessNdc1hwc0NcdhwRowScalar();
                return;
            }
            ProcessNdc1hwc0();
            return;
        }
        ProcessGeneric();
    }

    __attribute__((noinline)) __aicore__ bool TryProcessNdc1hwc0K1RouteBoundary()
    {
        return TryProcessNdc1hwc0K1Route();
    }

    __attribute__((noinline)) __aicore__ bool TryProcessNdc1hwc0EarlySpatialRouteBoundary()
    {
        return TryProcessNdc1hwc0EarlySpatialRoute();
    }

    __attribute__((noinline)) __aicore__ bool TryProcessNdc1hwc0NdhwcStrideRouteBoundary()
    {
        return TryProcessNdc1hwc0NdhwcStrideRoute();
    }

    __attribute__((noinline)) __aicore__ bool TryProcessNdc1hwc0NcdhwStrideRouteBoundary()
    {
        return TryProcessNdc1hwc0NcdhwStrideRoute();
    }

    __attribute__((noinline)) __aicore__ bool TryProcessNdc1hwc0MiddleSpatialRouteBoundary()
    {
        return TryProcessNdc1hwc0MiddleSpatialRoute();
    }

    __attribute__((noinline)) __aicore__ bool TryProcessNdc1hwc0LateSpatialRouteBoundary()
    {
        return TryProcessNdc1hwc0LateSpatialRoute();
    }

    __aicore__ inline bool TryProcessNdc1hwc0K1Route()
    {
        if (CanUseNdc1hwc0NcdhwK1FullC1PlanePath()) {
            ProcessNdc1hwc0NcdhwK1FullC1Plane();
        } else if (CanUseNdc1hwc0NcdhwK1DirectGroupPath()) {
            ProcessNdc1hwc0NcdhwK1DirectGroup();
        } else if (CanUseNdc1hwc0NcdhwK1DirectPath()) {
            ProcessNdc1hwc0NcdhwK1Direct();
        } else if (CanUseNdc1hwc0NdhwcK1FullC1PlanePath()) {
            ProcessNdc1hwc0NdhwcK1FullC1Plane();
        } else if (CanUseNdc1hwc0NdhwcK1BalancedDirectPath()) {
            ProcessNdc1hwc0NdhwcK1BalancedDirect();
        } else if (CanUseNdc1hwc0NdhwcK1DirectGroupPath()) {
            ProcessNdc1hwc0NdhwcK1DirectGroup();
        } else {
            return false;
        }
        return true;
    }

    __aicore__ inline bool TryProcessNdc1hwc0EarlySpatialRoute()
    {
        if (CanUseNdc1hwc0NcdhwDilatedWDirectPath()) {
            ProcessNdc1hwc0NcdhwDilatedWDirect();
        } else if (CanUseNdc1hwc0NdhwcDilatedWDirectPath()) {
            ProcessNdc1hwc0NdhwcDilatedWDirect();
        } else {
            return false;
        }
        return true;
    }

    __aicore__ inline bool TryProcessNdc1hwc0NdhwcStrideRoute()
    {
        if (CanUseNdc1hwc0NdhwcStride2DualC1GroupPath()) {
            ProcessNdc1hwc0NdhwcStride2DualC1Group();
        } else if (CanUseNdc1hwc0NdhwcStride2GroupPath()) {
            ProcessNdc1hwc0NdhwcStride2Group();
        } else {
            return false;
        }
        return true;
    }

    __aicore__ inline bool TryProcessNdc1hwc0NcdhwStrideRoute()
    {
        if (CanUseNdc1hwc0NcdhwStride2FullC1PlanePath()) {
            ProcessNdc1hwc0NcdhwStride2FullC1Plane();
        } else if (CanUseNdc1hwc0NcdhwStride2GroupPath()) {
            ProcessNdc1hwc0NcdhwStride2Group();
        } else {
            return false;
        }
        return true;
    }

    __aicore__ inline bool TryProcessNdc1hwc0MiddleSpatialRoute()
    {
        if (CanUseNdc1hwc0HOnlyStride3GroupPath()) {
            ProcessNdc1hwc0HOnlyStride3Group();
        } else if (CanUseNdc1hwc0NdhwcD3H3Dil2GroupPath()) {
            ProcessNdc1hwc0NdhwcD3H3Dil2Group();
        } else if (CanUseNdc1hwc0NcdhwD3H3Dil2GroupPath()) {
            ProcessNdc1hwc0NcdhwD3H3Dil2Group();
        } else if (CanUseNdc1hwc0NdhwcD3H3Dil2ReusePath()) {
            ProcessNdc1hwc0NdhwcD3H3Dil2Reuse();
        } else {
            return false;
        }
        return true;
    }

    __aicore__ inline bool TryProcessNdc1hwc0LateSpatialRoute()
    {
        if (CanUseNdc1hwc0D2H3W2Dil2PlanePath()) {
            ProcessNdc1hwc0D2H3W2Dil2Plane();
        } else if (CanUseNdc1hwc0NcdhwD3W3DilD2GroupPath()) {
            ProcessNdc1hwc0NcdhwD3W3DilD2Group();
        } else if (CanUseNdc1hwc0TinyK3ValidGroupPath()) {
            ProcessNdc1hwc0TinyK3ValidGroup();
        } else if (CanUseNdc1hwc0SmallCGroupPath()) {
            ProcessNdc1hwc0SmallCGroup();
        } else if (CanUseNdc1hwc0RowVectorPath() || CanUseNdc1hwc0NcdhwRowVectorPath()) {
            ProcessNdc1hwc0RowVector();
        } else {
            return false;
        }
        return true;
    }

    __aicore__ inline void InitDefaultBuffers()
    {
        constexpr uint32_t inBytes = (INPUT_TILE_NUM * sizeof(T) + UB_BLOCK_BYTES - 1U) / UB_BLOCK_BYTES *
                                     UB_BLOCK_BYTES;
        constexpr uint32_t outBytes = (OUTPUT_TILE_NUM * sizeof(T) + UB_BLOCK_BYTES - 1U) / UB_BLOCK_BYTES *
                                      UB_BLOCK_BYTES;
        pipe_.InitBuffer(xInQue_, BUFFER_NUM, inBytes);
        pipe_.InitBuffer(calcBuf_, outBytes);
        pipe_.InitBuffer(tmpBuf_, outBytes);
        pipe_.InitBuffer(maskBuf_, outBytes);
        pipe_.InitBuffer(yOutQue_, BUFFER_NUM, outBytes);
    }

    __aicore__ inline uint32_t UbBytesForElements(uint32_t count) const
    {
        return static_cast<uint32_t>((static_cast<uint64_t>(count) * sizeof(T) + UB_BLOCK_BYTES - 1U) / UB_BLOCK_BYTES *
                                     UB_BLOCK_BYTES);
    }

    struct D3H3BufferSizes {
        uint32_t inputTile;
        uint32_t calcTile;
        uint32_t tmpTile;
        uint32_t maskTile;
    };

    __aicore__ inline uint32_t LimitD3H3TileRows(uint32_t compactStride, uint32_t rowElements) const
    {
        uint32_t tileRows = static_cast<uint32_t>(tiling_->outH > 0 ? tiling_->outH : 1);
        const uint32_t rowOffsetNeed = Ndc1hwc0GatherTempOffset(rowElements);
        const uint32_t maxRowsByOutput = rowElements == 0U ? 1U : NDC1HWC0_D3H3_OUTPUT_TILE_NUM / rowElements;
        if (maxRowsByOutput > 0U && tileRows > maxRowsByOutput) {
            tileRows = maxRowsByOutput;
        }
        while (tileRows > 1U &&
               (static_cast<uint64_t>(tileRows) * compactStride > NDC1HWC0_D3H3_OUTPUT_TILE_NUM ||
                AlignToVector(tileRows * compactStride) + rowOffsetNeed > NDC1HWC0_D3H3_OUTPUT_TILE_NUM)) {
            --tileRows;
        }
        return tileRows == 0U ? 1U : tileRows;
    }

    __aicore__ inline D3H3BufferSizes GetNdhwcD3H3BufferSizes(uint32_t block, uint32_t cCount, uint32_t outW,
                                                              uint32_t inH) const
    {
        const uint32_t rowElements = outW * block;
        const uint32_t rowCount = outW * cCount;
        const uint32_t alignedRowCount = outW * AlignToVector(cCount);
        const uint32_t alignedCompactStride = AlignToVector(alignedRowCount + 1U);
        const uint32_t compactStride = AlignToVector(AlignToVector(rowCount) + 1U);
        const uint32_t tileRows = LimitD3H3TileRows(compactStride, rowElements);
        const uint32_t rowOffsetNeed = Ndc1hwc0GatherTempOffset(rowElements);
        const uint32_t fallbackTmpTile = AlignToVector(tileRows * compactStride) + rowOffsetNeed;
        const uint32_t alignedTmpTile = AlignToVector(tileRows * alignedCompactStride) + rowOffsetNeed;
        return {inH * alignedRowCount, inH * alignedRowCount,
                fallbackTmpTile > alignedTmpTile ? fallbackTmpTile : alignedTmpTile, tileRows * rowElements};
    }

    __aicore__ inline D3H3BufferSizes GetNcdhwD3H3BufferSizes(uint32_t block, uint32_t cCount, uint32_t outW,
                                                              uint32_t inH) const
    {
        const uint32_t rowElements = outW * block;
        const uint32_t alignedW = AlignToVector(outW);
        const uint32_t compactCount = cCount * alignedW;
        const uint32_t compactStride = AlignToVector(compactCount + 1U);
        const uint32_t tileRows = LimitD3H3TileRows(compactStride, rowElements);
        const uint32_t rowOffsetNeed = Ndc1hwc0GatherTempOffset(rowElements);
        uint32_t tmpTile = AlignToVector(tileRows * compactStride) + rowOffsetNeed;
        uint32_t maskTile = tileRows * rowElements;
        const uint32_t planeValid = static_cast<uint32_t>(tiling_->outH > 0 ? tiling_->outH : 1) * outW;
        const uint32_t alignedPlane = AlignToVector(planeValid + 1U);
        const uint32_t transTile = alignedPlane * block;
        if (transTile > maskTile) {
            maskTile = transTile;
        }
        const uint32_t planeTmpTile = (cCount + 1U) * alignedPlane;
        if (planeTmpTile > tmpTile) {
            tmpTile = planeTmpTile;
        }
        const uint32_t planeGatherTmpTile = planeTmpTile + Ndc1hwc0GatherTempOffset(planeValid * block);
        if (planeGatherTmpTile > tmpTile) {
            tmpTile = planeGatherTmpTile;
        }
        return {inH * 3U * alignedW, compactCount * inH, tmpTile, maskTile};
    }

    __aicore__ inline void InitNdc1hwc0D3H3Dil2Buffers()
    {
        const uint32_t block = static_cast<uint32_t>(Ndc1hwc0Block());
        const uint32_t cCount = static_cast<uint32_t>(tiling_->c);
        const uint32_t outW = static_cast<uint32_t>(tiling_->outW);
        const uint32_t inH = static_cast<uint32_t>(tiling_->inH);
        const D3H3BufferSizes sizes = tiling_->dataFormat == FORMAT_NDHWC_VALUE ?
                                          GetNdhwcD3H3BufferSizes(block, cCount, outW, inH) :
                                          GetNcdhwD3H3BufferSizes(block, cCount, outW, inH);
        pipe_.InitBuffer(xInQue_, BUFFER_NUM, UbBytesForElements(sizes.inputTile));
        pipe_.InitBuffer(calcBuf_, UbBytesForElements(sizes.calcTile));
        pipe_.InitBuffer(tmpBuf_, UbBytesForElements(sizes.tmpTile));
        pipe_.InitBuffer(maskBuf_, UbBytesForElements(sizes.maskTile));
        pipe_.InitBuffer(yOutQue_, BUFFER_NUM, UB_BLOCK_BYTES);
    }

    __aicore__ inline void InitNdc1hwc0D2H3W2Dil2Buffers()
    {
        const uint32_t block = static_cast<uint32_t>(Ndc1hwc0Block());
        const uint32_t outW = static_cast<uint32_t>(tiling_->outW > 0 ? tiling_->outW : 1);
        const uint32_t outH = static_cast<uint32_t>(tiling_->outH > 0 ? tiling_->outH : 1);
        const uint32_t inH = static_cast<uint32_t>(tiling_->inH > 0 ? tiling_->inH : 1);
        const uint32_t inW = static_cast<uint32_t>(tiling_->inW > 0 ? tiling_->inW : 1);
        const uint32_t rowElements = outW * block;
        const uint32_t planeElements = outH * rowElements;
        uint32_t groupPlanes = planeElements == 0U ? 1U : NDC1HWC0_D3H3_OUTPUT_TILE_NUM / planeElements;
        if (groupPlanes == 0U) {
            groupPlanes = 1U;
        }
        if (tiling_->outD > 0 && groupPlanes > static_cast<uint32_t>(tiling_->outD)) {
            groupPlanes = static_cast<uint32_t>(tiling_->outD);
        }
        const uint32_t calcTile = groupPlanes * planeElements;
        uint32_t inputTile = inH * inW * block;
        uint32_t tmpTile = inH * rowElements;
        uint32_t maskTile = planeElements;
        if (tiling_->dataFormat == FORMAT_NCDHW_VALUE) {
            const uint32_t cCount = static_cast<uint32_t>(tiling_->c > 0 ? tiling_->c : 1);
            const uint32_t alignedInputPlane = AlignToVector(inH * inW);
            const uint32_t offsetNeed = Ndc1hwc0GatherTempOffset(rowElements);
            inputTile = (cCount + 2U) * alignedInputPlane;
            tmpTile = 2U * offsetNeed + inH * rowElements + planeElements;
            maskTile = rowElements;
        }
        pipe_.InitBuffer(xInQue_, BUFFER_NUM, UbBytesForElements(inputTile));
        pipe_.InitBuffer(calcBuf_, UbBytesForElements(calcTile));
        pipe_.InitBuffer(tmpBuf_, UbBytesForElements(tmpTile));
        pipe_.InitBuffer(maskBuf_, UbBytesForElements(maskTile));
        pipe_.InitBuffer(yOutQue_, BUFFER_NUM, UB_BLOCK_BYTES);
    }

    __aicore__ inline bool IsPool2Stride2NoPad() const
    {
        return tiling_->kD == 2 && tiling_->kH == 2 && tiling_->kW == 2 && tiling_->sD == 2 && tiling_->sH == 2 &&
               tiling_->sW == 2 && tiling_->dilationD == 1 && tiling_->dilationH == 1 && tiling_->dilationW == 1 &&
               tiling_->padFront == 0 && tiling_->padTop == 0 && tiling_->padLeft == 0;
    }

    __aicore__ inline bool HasPositiveTensorDims() const
    {
        return tiling_->totalOut > 0U && tiling_->n > 0 && tiling_->inD > 0 && tiling_->inH > 0 && tiling_->inW > 0 &&
               tiling_->c > 0 && tiling_->outD > 0 && tiling_->outH > 0 && tiling_->outW > 0;
    }

    __aicore__ inline bool HasPositivePoolParams() const
    {
        return tiling_->kD > 0 && tiling_->kH > 0 && tiling_->kW > 0 && tiling_->sD > 0 && tiling_->sH > 0 &&
               tiling_->sW > 0 && tiling_->dilationD > 0 && tiling_->dilationH > 0 && tiling_->dilationW > 0;
    }

    __aicore__ inline bool HasValidSimtNdc1hwc0Storage(uint64_t block, uint64_t validC1) const
    {
        if (block == 0U || validC1 == 0U || tiling_->outputC1 <= 0 || tiling_->outputD <= 0 || tiling_->outputH <= 0 ||
            tiling_->outputW <= 0) {
            return false;
        }
        if (!IsNdc1hwc0CompactStorage(block, validC1) && Ndc1hwc0StorageC0() != block) {
            return false;
        }
        return validC1 <= 1U || static_cast<uint64_t>(tiling_->c) % block == 0U;
    }

    __aicore__ inline bool CanUseSimtFastPath() const
    {
#if !MAX_POOL3D_ENABLE_SIMT
        return false;
#else
        if constexpr (AscendC::Std::is_same<T, bfloat16_t>::value) {
            return false;
        }
        if (!HasPositiveTensorDims() || !HasPositivePoolParams()) {
            return false;
        }
        if (tiling_->inputLayout == INPUT_LAYOUT_NDC1HWC0_VALUE) {
            return false;
        }
        if (tiling_->outputLayout == OUTPUT_LAYOUT_NDC1HWC0_VALUE) {
            const uint64_t block = Ndc1hwc0Block();
            const uint64_t validC1 = Ndc1hwc0ValidC1(block);
            if (!HasValidSimtNdc1hwc0Storage(block, validC1)) {
                return false;
            }
            const uint64_t simtOut = SimtEffectiveOut();
            if (simtOut > 120000U) {
                return false;
            }
            if constexpr (AscendC::Std::is_same<T, half>::value) {
                return simtOut <= SIMT_HALF_REGULAR_MAX_TOTAL_OUT;
            }
            return simtOut <= SIMT_REGULAR_MAX_TOTAL_OUT;
        }
        return false;
#endif
    }

    __aicore__ inline bool CanUseAivDirectFastPath() const
    {
        if constexpr (AscendC::Std::is_same<T, bfloat16_t>::value) {
            return false;
        }
        if (!HasPositiveTensorDims() || !HasPositivePoolParams()) {
            return false;
        }
        if (tiling_->outputLayout == OUTPUT_LAYOUT_NDC1HWC0_VALUE) {
            const int64_t c0 = tiling_->outputC0 > 0 ? tiling_->outputC0 : tiling_->outputC0Block;
            const int64_t storageD = tiling_->outputD > 0 ? tiling_->outputD : tiling_->outD;
            const int64_t storageH = tiling_->outputH > 0 ? tiling_->outputH : tiling_->outH;
            const int64_t storageW = tiling_->outputW > 0 ? tiling_->outputW : tiling_->outW;
            return c0 > 0 && tiling_->outputC1 > 0 && storageD > 0 && storageH > 0 && storageW > 0;
        }
        if (!IsPool2Stride2NoPad()) {
            return false;
        }
        if constexpr (AscendC::Std::is_same<T, half>::value) {
            return tiling_->totalOut <= SIMT_HALF_REGULAR_MAX_TOTAL_OUT;
        }
        return tiling_->totalOut <= SIMT_REGULAR_MAX_TOTAL_OUT;
    }

    __aicore__ inline uint64_t PositiveDivisor(int64_t value) const
    {
        return value > 0 ? static_cast<uint64_t>(value) : 1U;
    }

    __aicore__ inline void SetMagicDivisor(LocalTensor<uint32_t> paramLocal, uint32_t idx, uint64_t divisor) const
    {
#if MAX_POOL3D_ENABLE_SIMT
        uint32_t magic = 0U;
        uint32_t shift = 0U;
        GetUintDivMagicAndShift<uint32_t>(magic, shift, static_cast<uint32_t>(divisor == 0U ? 1U : divisor));
        paramLocal.SetValue(idx, magic);
        paramLocal.SetValue(idx + 1U, shift);
#else
        (void)paramLocal;
        (void)idx;
        (void)divisor;
#endif
    }

    __aicore__ inline uint64_t SimtEffectiveOut() const
    {
        if (tiling_->outputLayout != OUTPUT_LAYOUT_NDC1HWC0_VALUE) {
            return tiling_->totalOut;
        }
        const uint64_t block = Ndc1hwc0Block();
        const uint64_t validC1 = Ndc1hwc0ValidC1(block);
        if (!IsNdc1hwc0CompactPrefix(block, validC1)) {
            return tiling_->totalOut;
        }
        const uint64_t validOut = Ndc1hwc0ValidOut(block, validC1);
        return validOut < tiling_->totalOut ? validOut : tiling_->totalOut;
    }

    __aicore__ inline void FillSimtTiling(LocalTensor<int64_t> simtTiling, uint64_t simtOut) const
    {
        simtTiling.SetValue(0, static_cast<int64_t>(simtOut));
        simtTiling.SetValue(1, tiling_->n);
        simtTiling.SetValue(2, tiling_->inD);
        simtTiling.SetValue(3, tiling_->inH);
        simtTiling.SetValue(4, tiling_->inW);
        simtTiling.SetValue(5, tiling_->c);
        simtTiling.SetValue(6, tiling_->outD);
        simtTiling.SetValue(7, tiling_->outH);
        simtTiling.SetValue(8, tiling_->outW);
        simtTiling.SetValue(9, tiling_->kD);
        simtTiling.SetValue(10, tiling_->kH);
        simtTiling.SetValue(11, tiling_->kW);
        simtTiling.SetValue(12, tiling_->sD);
        simtTiling.SetValue(13, tiling_->sH);
        simtTiling.SetValue(14, tiling_->sW);
        simtTiling.SetValue(15, tiling_->padFront);
        simtTiling.SetValue(16, tiling_->padTop);
        simtTiling.SetValue(17, tiling_->padLeft);
        simtTiling.SetValue(18, tiling_->dilationD);
        simtTiling.SetValue(19, tiling_->dilationH);
        simtTiling.SetValue(20, tiling_->dilationW);
        simtTiling.SetValue(21, static_cast<int64_t>(tiling_->dataFormat));
        simtTiling.SetValue(22, static_cast<int64_t>(tiling_->outputLayout));
        simtTiling.SetValue(23, tiling_->outputD);
        simtTiling.SetValue(24, tiling_->outputH);
        simtTiling.SetValue(25, tiling_->outputW);
        simtTiling.SetValue(26, tiling_->outputC1);
        simtTiling.SetValue(27, tiling_->outputC0);
        simtTiling.SetValue(28, tiling_->outputC0Block);
    }

    __aicore__ inline void FillSimtParam(LocalTensor<uint32_t> paramLocal) const
    {
        SetMagicDivisor(paramLocal, 0, PositiveDivisor(tiling_->outD));
        SetMagicDivisor(paramLocal, 2, PositiveDivisor(tiling_->outH));
        SetMagicDivisor(paramLocal, 4, PositiveDivisor(tiling_->outW));
        SetMagicDivisor(paramLocal, 6, PositiveDivisor(tiling_->c));
        const int64_t storageC0 = tiling_->outputLayout == OUTPUT_LAYOUT_NDC1HWC0_VALUE ?
                                      (tiling_->outputC0Block > 0 ? tiling_->outputC0Block : tiling_->outputC0) :
                                      (tiling_->outputC0 > 0 ? tiling_->outputC0 : tiling_->outputC0Block);
        const uint64_t simtBlock = storageC0 > 0 ? static_cast<uint64_t>(storageC0) : 0U;
        const uint64_t simtValidC1 = Ndc1hwc0ValidC1(simtBlock);
        const bool compactPrefix = tiling_->outputLayout == OUTPUT_LAYOUT_NDC1HWC0_VALUE &&
                                   IsNdc1hwc0CompactPrefix(simtBlock, simtValidC1);
        const int64_t storageD = compactPrefix ? tiling_->outD :
                                                 (tiling_->outputD > 0 ? tiling_->outputD : tiling_->outD);
        const int64_t storageH = compactPrefix ? tiling_->outH :
                                                 (tiling_->outputH > 0 ? tiling_->outputH : tiling_->outH);
        const int64_t storageW = compactPrefix ? tiling_->outW :
                                                 (tiling_->outputW > 0 ? tiling_->outputW : tiling_->outW);
        const int64_t storageC1 = compactPrefix ? static_cast<int64_t>(simtValidC1) :
                                                  (tiling_->outputC1 > 0 ? tiling_->outputC1 : 1);
        SetMagicDivisor(paramLocal, 8, PositiveDivisor(storageC0));
        SetMagicDivisor(paramLocal, 10, PositiveDivisor(storageW));
        SetMagicDivisor(paramLocal, 12, PositiveDivisor(storageH));
        SetMagicDivisor(paramLocal, 14, PositiveDivisor(storageC1));
        SetMagicDivisor(paramLocal, 16, PositiveDivisor(storageD));
    }

    __aicore__ inline void ProcessSimtFast()
    {
#if MAX_POOL3D_ENABLE_SIMT
        if constexpr (!AscendC::Std::is_same<T, bfloat16_t>::value) {
            LocalTensor<int64_t> simtTiling = simtTilingDataBuf_.Get<int64_t>();
            LocalTensor<uint32_t> simtParam = simtParamBuf_.Get<uint32_t>();
            const uint64_t simtOut = SimtEffectiveOut();
            FillSimtTiling(simtTiling, simtOut);
            FillSimtParam(simtParam);
            DataSyncBarrier<MemDsbT::UB>();
            Simt::VF_CALL<MaxPool3DSimtLiteCompute<T>>(Simt::Dim3(SIMT_THREAD_NUM), (__gm__ T*)xGm_.GetPhyAddr(),
                                                       (__gm__ T*)yGm_.GetPhyAddr(),
                                                       (__ubuf__ MaxPool3DSimtLiteTilingData*)simtTiling.GetPhyAddr(),
                                                       (__ubuf__ uint32_t*)simtParam.GetPhyAddr());
            if (simtOut < tiling_->totalOut) {
                CopyOutZeroRangeByCore(simtOut, tiling_->totalOut - simtOut);
            }
        }
#endif
    }

    __aicore__ inline void ProcessAivDirectFast()
    {
        const uint64_t outOffset = ValidCoreStartOffset(tiling_->totalOut);
        const uint64_t outCount = ValidCoreElementCount(tiling_->totalOut, outOffset);
        uint64_t processed = 0U;
        while (processed < outCount) {
            const uint64_t remain = outCount - processed;
            const uint32_t curCount = remain > OUTPUT_TILE_NUM ? OUTPUT_TILE_NUM : static_cast<uint32_t>(remain);
            LocalTensor<T> yLocal = yOutQue_.AllocTensor<T>();
            for (uint32_t i = 0; i < curCount; ++i) {
                const uint64_t linear = outOffset + processed + static_cast<uint64_t>(i);
                const T outValue = tiling_->outputLayout == OUTPUT_LAYOUT_NDC1HWC0_VALUE ?
                                       ComputeNdc1hwc0OutputValue(linear) :
                                       ComputeValue(linear);
                yLocal.SetValue(i, outValue);
            }
            yOutQue_.EnQue(yLocal);
            LocalTensor<T> yOut = yOutQue_.DeQue<T>();
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(curCount * sizeof(T)), 0, 0, 0};
            DataCopyPad(yGm_[outOffset + processed], yOut, copyParams);
            yOutQue_.FreeTensor(yOut);
            processed += curCount;
        }
    }

    __aicore__ inline bool CanUseNdhwcVectorPath() const
    {
        if (tiling_->dataFormat != FORMAT_NDHWC_VALUE && tiling_->dataFormat != FORMAT_NCDHW_VALUE) {
            return false;
        }
        if (tiling_->c >= static_cast<int64_t>(VectorAlignNum())) {
            return true;
        }
        return false;
    }

    __aicore__ inline uint32_t VectorAlignNum() const { return static_cast<uint32_t>(UB_BLOCK_BYTES / sizeof(T)); }

    __aicore__ inline void BalancedAtomicRange(uint64_t limit, uint64_t quantum, uint64_t& start, uint64_t& count) const
    {
        if (quantum == 0U) {
            quantum = 1U;
        }
        const uint64_t units = (limit + quantum - 1U) / quantum;
        const uint64_t blockDim = static_cast<uint64_t>(ActiveBlockDim());
        const uint64_t workers = units < blockDim ? units : blockDim;
        const uint64_t blockIdx = static_cast<uint64_t>(GetBlockIdx());
        if (workers == 0U || blockIdx >= workers) {
            start = limit;
            count = 0U;
            return;
        }
        const uint64_t baseUnits = units / workers;
        const uint64_t extraUnits = units % workers;
        const uint64_t myUnits = baseUnits + (blockIdx < extraUnits ? 1U : 0U);
        const uint64_t beginUnit = blockIdx * baseUnits + (blockIdx < extraUnits ? blockIdx : extraUnits);
        start = beginUnit * quantum;
        count = myUnits * quantum;
        if (start >= limit) {
            count = 0U;
        } else if (start + count > limit) {
            count = limit - start;
        }
    }

    __aicore__ inline uint64_t CoreStartOffset() const
    {
        if (tiling_->balancedSplit != 0U && tiling_->splitQuantum > 0U) {
            uint64_t start = 0U;
            uint64_t count = 0U;
            BalancedAtomicRange(tiling_->totalOut, tiling_->splitQuantum, start, count);
            return start;
        }
        const uint64_t blockIdx = static_cast<uint64_t>(GetBlockIdx());
        return tiling_->normalCoreOut * blockIdx;
    }

    __aicore__ inline uint64_t CoreElementCount(uint64_t outOffset) const
    {
        if (tiling_->balancedSplit != 0U && tiling_->splitQuantum > 0U) {
            uint64_t start = 0U;
            uint64_t count = 0U;
            BalancedAtomicRange(tiling_->totalOut, tiling_->splitQuantum, start, count);
            return start == outOffset ? count : 0U;
        }
        if (outOffset >= tiling_->totalOut) {
            return 0;
        }

        uint64_t outCount = tiling_->normalCoreOut;
        if (outOffset + outCount > tiling_->totalOut) {
            outCount = tiling_->totalOut - outOffset;
        }
        return outCount;
    }

    __aicore__ inline uint32_t ActiveBlockDim() const { return tiling_->blockDim == 0U ? 1U : tiling_->blockDim; }

    __aicore__ inline uint64_t ValidCoreStartOffset(uint64_t validOut) const
    {
        const uint64_t blockDim = static_cast<uint64_t>(ActiveBlockDim());
        const uint64_t normalValidOut = (validOut + blockDim - 1U) / blockDim;
        return normalValidOut * static_cast<uint64_t>(GetBlockIdx());
    }

    __aicore__ inline uint64_t ValidCoreElementCount(uint64_t validOut, uint64_t outOffset) const
    {
        if (outOffset >= validOut) {
            return 0U;
        }
        const uint64_t blockDim = static_cast<uint64_t>(ActiveBlockDim());
        const uint64_t normalValidOut = (validOut + blockDim - 1U) / blockDim;
        const uint64_t remain = validOut - outOffset;
        return remain < normalValidOut ? remain : normalValidOut;
    }

    __aicore__ inline uint64_t Ndc1hwc0RowsPerActiveCore(uint64_t validOut, uint64_t rowElements) const
    {
        if (rowElements == 0U) {
            return 0U;
        }
        const uint64_t configuredRows = tiling_->normalCoreOut / rowElements;
        if (configuredRows > 0U) {
            return configuredRows;
        }
        const uint64_t blockDim = static_cast<uint64_t>(ActiveBlockDim());
        const uint64_t totalRows = (validOut + rowElements - 1U) / rowElements;
        uint64_t rows = (totalRows + blockDim - 1U) / blockDim;
        return rows == 0U ? 1U : rows;
    }

    __aicore__ inline uint64_t Ndc1hwc0ValidCoreStartOffset(uint64_t validOut, uint64_t rowElements) const
    {
        if (tiling_->balancedSplit != 0U && tiling_->splitQuantum >= rowElements && rowElements > 0U &&
            tiling_->splitQuantum % rowElements == 0U) {
            uint64_t start = 0U;
            uint64_t count = 0U;
            BalancedAtomicRange(validOut, tiling_->splitQuantum, start, count);
            return start;
        }
        return Ndc1hwc0RowsPerActiveCore(validOut, rowElements) * static_cast<uint64_t>(GetBlockIdx()) * rowElements;
    }

    __aicore__ inline uint64_t Ndc1hwc0ValidCoreElementCount(uint64_t validOut, uint64_t rowElements,
                                                             uint64_t outOffset) const
    {
        if (tiling_->balancedSplit != 0U && tiling_->splitQuantum >= rowElements && rowElements > 0U &&
            tiling_->splitQuantum % rowElements == 0U) {
            uint64_t start = 0U;
            uint64_t count = 0U;
            BalancedAtomicRange(validOut, tiling_->splitQuantum, start, count);
            return start == outOffset ? count : 0U;
        }
        if (outOffset >= validOut || rowElements == 0U) {
            return 0U;
        }
        const uint64_t rowsPerCore = Ndc1hwc0RowsPerActiveCore(validOut, rowElements);
        uint64_t outCount = rowsPerCore * rowElements;
        if (outOffset + outCount > validOut) {
            outCount = validOut - outOffset;
        }
        return outCount;
    }

    __aicore__ inline bool InitNdc1hwc0ValidOutput(uint64_t& block, uint64_t& validC1, uint64_t& validOut) const
    {
        block = Ndc1hwc0Block();
        validC1 = Ndc1hwc0ValidC1(block);
        validOut = Ndc1hwc0ValidOut(block, validC1);
        return block != 0U && validC1 != 0U && validOut != 0U;
    }

    __aicore__ inline void GetNdc1hwc0ValidCoreRange(uint64_t validOut, uint64_t rowElements, uint64_t& outOffset,
                                                     uint64_t& outCount, uint64_t& outEnd) const
    {
        outOffset = Ndc1hwc0ValidCoreStartOffset(validOut, rowElements);
        outCount = Ndc1hwc0ValidCoreElementCount(validOut, rowElements, outOffset);
        outEnd = outOffset + outCount;
    }

    __aicore__ inline bool InitNdc1hwc0GroupRange(uint64_t& block, uint64_t& validC1, uint64_t& validOut,
                                                  uint32_t& outW, uint32_t& rowElements, uint64_t& outOffset,
                                                  uint64_t& outCount, uint64_t& outEnd) const
    {
        if (!InitNdc1hwc0ValidOutput(block, validC1, validOut)) {
            return false;
        }
        outW = static_cast<uint32_t>(tiling_->outW);
        rowElements = static_cast<uint32_t>(static_cast<uint64_t>(outW) * block);
        GetNdc1hwc0ValidCoreRange(validOut, rowElements, outOffset, outCount, outEnd);
        return true;
    }

    __aicore__ inline bool HasNdc1hwc0CoreWork(uint64_t outCount, uint64_t validOut)
    {
        if (outCount > 0U) {
            return true;
        }
        if (validOut < tiling_->totalOut) {
            CopyOutZeroRangeByCore(validOut, tiling_->totalOut - validOut);
        }
        return false;
    }

    __aicore__ inline bool InitNdc1hwc0ActiveGroupRange(uint64_t& block, uint64_t& validC1, uint64_t& validOut,
                                                        uint32_t& outW, uint32_t& rowElements, uint64_t& outOffset,
                                                        uint64_t& outCount, uint64_t& outEnd)
    {
        return InitNdc1hwc0GroupRange(block, validC1, validOut, outW, rowElements, outOffset, outCount, outEnd) &&
               HasNdc1hwc0CoreWork(outCount, validOut);
    }

    __aicore__ inline bool InitNdc1hwc0LinearRange(uint64_t& block, uint64_t& validC1, uint64_t& validOut,
                                                   uint64_t& rowElements, uint64_t& outOffset, uint64_t& outCount,
                                                   uint64_t& outEnd) const
    {
        if (!InitNdc1hwc0ValidOutput(block, validC1, validOut)) {
            return false;
        }
        rowElements = static_cast<uint64_t>(tiling_->outW) * block;
        GetNdc1hwc0ValidCoreRange(validOut, rowElements, outOffset, outCount, outEnd);
        return true;
    }

    struct Ndc1hwc0LinearRangeContext {
        uint64_t block = 0U;
        uint64_t validC1 = 0U;
        uint64_t validOut = 0U;
        uint64_t rowElements = 0U;
        uint64_t outOffset = 0U;
        uint64_t outCount = 0U;
        uint64_t outEnd = 0U;
    };

    __aicore__ inline bool InitNdc1hwc0ActiveLinearRange(Ndc1hwc0LinearRangeContext& context)
    {
        return InitNdc1hwc0LinearRange(context.block, context.validC1, context.validOut, context.rowElements,
                                       context.outOffset, context.outCount, context.outEnd) &&
               HasNdc1hwc0CoreWork(context.outCount, context.validOut);
    }

    __aicore__ inline uint32_t Ndc1hwc0SafeGatherChunk(uint32_t remaining) const
    {
        return remaining > NDC1HWC0_SAFE_GATHER_COUNT ? NDC1HWC0_SAFE_GATHER_COUNT : remaining;
    }

    __aicore__ inline bool InitNdc1hwc0FullPlaneGeometry(uint64_t& block, uint64_t& validC1, uint64_t& validOut,
                                                         uint32_t& outW, uint32_t& outH, uint32_t& rowElements,
                                                         uint32_t& outputCount) const
    {
        if (!InitNdc1hwc0ValidOutput(block, validC1, validOut)) {
            return false;
        }
        outW = static_cast<uint32_t>(tiling_->outW);
        outH = static_cast<uint32_t>(tiling_->outH);
        rowElements = static_cast<uint32_t>(static_cast<uint64_t>(outW) * block);
        outputCount = static_cast<uint32_t>(validC1) * outH * rowElements;
        return rowElements != 0U && outputCount != 0U;
    }

    __aicore__ inline void ProcessGenericRange(uint64_t outOffset, uint64_t outCount)
    {
        uint64_t processed = 0;
        while (processed < outCount) {
            const uint64_t remain = outCount - processed;
            constexpr uint32_t c3TileCount = OUTPUT_TILE_NUM / 3U * 3U;
            const uint32_t curCount = remain > c3TileCount ? c3TileCount : static_cast<uint32_t>(remain);
            LocalTensor<T> yLocal = yOutQue_.AllocTensor<T>();
            for (uint32_t i = 0; i < curCount; ++i) {
                yLocal.SetValue(i, ComputeValue(outOffset + processed + i));
            }
            yOutQue_.EnQue(yLocal);

            LocalTensor<T> yOut = yOutQue_.DeQue<T>();
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(curCount * sizeof(T)), 0, 0, 0};
            DataCopyPad(yGm_[outOffset + processed], yOut, copyParams);
            yOutQue_.FreeTensor(yOut);
            processed += curCount;
        }
    }

    __aicore__ inline void ProcessGeneric()
    {
        const uint64_t outOffset = CoreStartOffset();
        ProcessGenericRange(outOffset, CoreElementCount(outOffset));
    }

    __aicore__ inline T ZeroValue() const
    {
        if constexpr (AscendC::Std::is_same<T, bfloat16_t>::value) {
            constexpr uint16_t zeroBits = 0U;
            return *reinterpret_cast<const bfloat16_t*>(&zeroBits);
        } else if constexpr (AscendC::Std::is_same<T, half>::value) {
            constexpr uint16_t zeroBits = 0U;
            return *reinterpret_cast<const half*>(&zeroBits);
        } else {
            return 0.0F;
        }
    }

    __aicore__ inline T ComputeNdc1hwc0OutputValue(uint64_t linear)
    {
        const uint64_t block = Ndc1hwc0Block();
        const uint64_t validC1 = Ndc1hwc0ValidC1(block);
        const bool compactPrefix = IsNdc1hwc0CompactPrefix(block, validC1);
        const uint64_t storageD = compactPrefix ? static_cast<uint64_t>(tiling_->outD) : Ndc1hwc0StorageD();
        const uint64_t storageH = compactPrefix ? static_cast<uint64_t>(tiling_->outH) : Ndc1hwc0StorageH();
        const uint64_t storageW = compactPrefix ? static_cast<uint64_t>(tiling_->outW) : Ndc1hwc0StorageW();
        const uint64_t storageC1 = compactPrefix ? validC1 : Ndc1hwc0StorageC1();
        const uint64_t storageC0 = compactPrefix ? block : Ndc1hwc0StorageC0();
        if (storageD == 0U || storageH == 0U || storageW == 0U || storageC1 == 0U || storageC0 == 0U) {
            return ZeroValue();
        }
        const int64_t c0Idx = static_cast<int64_t>(linear % storageC0);
        linear /= storageC0;
        const int64_t ow = static_cast<int64_t>(linear % storageW);
        linear /= storageW;
        const int64_t oh = static_cast<int64_t>(linear % storageH);
        linear /= storageH;
        const int64_t c1Idx = static_cast<int64_t>(linear % storageC1);
        linear /= storageC1;
        const int64_t od = static_cast<int64_t>(linear % storageD);
        const int64_t nIdx = static_cast<int64_t>(linear / storageD);
        const int64_t cIdx = c1Idx * static_cast<int64_t>(storageC0) + c0Idx;
        if (nIdx >= tiling_->n || od >= tiling_->outD || oh >= tiling_->outH || ow >= tiling_->outW ||
            cIdx >= tiling_->c) {
            return ZeroValue();
        }
        return ComputeValueAt(nIdx, od, oh, ow, cIdx);
    }

    __aicore__ inline uint64_t Ndc1hwc0Block() const
    {
        return static_cast<uint64_t>(tiling_->outputC0Block > 0 ? tiling_->outputC0Block : tiling_->outputC0);
    }

    __aicore__ inline int64_t DilatedInputD(int64_t od, int64_t kd) const
    {
        return od * tiling_->sD + kd * tiling_->dilationD - tiling_->padFront;
    }

    __aicore__ inline int64_t DilatedInputH(int64_t oh, int64_t kh) const
    {
        return oh * tiling_->sH + kh * tiling_->dilationH - tiling_->padTop;
    }

    __aicore__ inline int64_t DilatedInputW(int64_t ow, int64_t kw) const
    {
        return ow * tiling_->sW + kw * tiling_->dilationW - tiling_->padLeft;
    }

    __aicore__ inline int64_t DilatedInputWFromStart(uint32_t wStart, int64_t kw) const
    {
        return static_cast<int64_t>(wStart) * tiling_->sW + kw * tiling_->dilationW - tiling_->padLeft;
    }

    __aicore__ inline int64_t Pool2InputD(int64_t od, int64_t kd) const { return od * 2 + kd; }

    __aicore__ inline int64_t Pool2InputH(int64_t oh, int64_t kh) const { return oh * 2 + kh; }

    __aicore__ inline int64_t Ndc1hwc0ActiveChannels(int64_t cBase, uint64_t block) const
    {
        int64_t activeChannels = tiling_->c - cBase;
        if (activeChannels < 0) {
            activeChannels = 0;
        }
        if (activeChannels > static_cast<int64_t>(block)) {
            activeChannels = static_cast<int64_t>(block);
        }
        return activeChannels;
    }

    __aicore__ inline bool MatchesPoolSpec(int64_t kD, int64_t kH, int64_t kW, int64_t sD, int64_t sH, int64_t sW,
                                           int64_t dilationD, int64_t dilationH, int64_t dilationW, int64_t padFront,
                                           int64_t padTop, int64_t padLeft) const
    {
        return tiling_->kD == kD && tiling_->kH == kH && tiling_->kW == kW && tiling_->sD == sD && tiling_->sH == sH &&
               tiling_->sW == sW && tiling_->dilationD == dilationD && tiling_->dilationH == dilationH &&
               tiling_->dilationW == dilationW && tiling_->padFront == padFront && tiling_->padTop == padTop &&
               tiling_->padLeft == padLeft;
    }

    __aicore__ inline bool HasNdc1hwc0InputOutputLayout() const
    {
        return tiling_->inputLayout == INPUT_LAYOUT_NDC1HWC0_VALUE &&
               tiling_->outputLayout == OUTPUT_LAYOUT_NDC1HWC0_VALUE;
    }

    __aicore__ inline uint64_t Ndc1hwc0StorageD() const
    {
        return static_cast<uint64_t>(tiling_->outputD > 0 ? tiling_->outputD : tiling_->outD);
    }

    __aicore__ inline uint64_t Ndc1hwc0StorageH() const
    {
        return static_cast<uint64_t>(tiling_->outputH > 0 ? tiling_->outputH : tiling_->outH);
    }

    __aicore__ inline uint64_t Ndc1hwc0StorageW() const
    {
        return static_cast<uint64_t>(tiling_->outputW > 0 ? tiling_->outputW : tiling_->outW);
    }

    __aicore__ inline uint64_t Ndc1hwc0StorageC1() const
    {
        return static_cast<uint64_t>(tiling_->outputC1 > 0 ? tiling_->outputC1 : 1);
    }

    __aicore__ inline uint64_t Ndc1hwc0StorageC0() const
    {
        return static_cast<uint64_t>(tiling_->outputC0 > 0 ? tiling_->outputC0 : Ndc1hwc0Block());
    }

    __aicore__ inline uint64_t InputNdc1hwc0Block() const
    {
        return static_cast<uint64_t>(tiling_->inputC0Block > 0 ? tiling_->inputC0Block :
                                                                 (tiling_->inputC0 > 0 ? tiling_->inputC0 : 16));
    }

    __aicore__ inline bool IsNdc1hwc0CompactPrefix(uint64_t block, uint64_t validC1) const
    {
        if (block == 0U || validC1 == 0U || tiling_->c <= 0 || tiling_->outD <= 0 || tiling_->outH <= 0 ||
            tiling_->outW <= 0 || tiling_->outputC0 <= 0) {
            return false;
        }
        const uint64_t storageC0 = Ndc1hwc0StorageC0();
        const uint64_t storageC1 = Ndc1hwc0StorageC1();
        const bool packedC0Prefix = storageC0 >= block && storageC0 % block == 0U &&
                                    storageC1 * (storageC0 / block) >= validC1;
        return Ndc1hwc0StorageD() >= static_cast<uint64_t>(tiling_->outD) &&
               Ndc1hwc0StorageH() >= static_cast<uint64_t>(tiling_->outH) &&
               Ndc1hwc0StorageW() >= static_cast<uint64_t>(tiling_->outW) &&
               ((storageC0 == block && storageC1 >= validC1) || packedC0Prefix);
    }

    __aicore__ inline bool IsNdc1hwc0CompactStorage(uint64_t block, uint64_t validC1) const
    {
        return IsNdc1hwc0CompactPrefix(block, validC1);
    }

    __aicore__ inline uint64_t Ndc1hwc0ValidC1(uint64_t block) const
    {
        if (block == 0U || tiling_->c <= 0) {
            return 0U;
        }
        return (static_cast<uint64_t>(tiling_->c) + block - 1U) / block;
    }

    __aicore__ inline uint64_t Ndc1hwc0ValidOut(uint64_t block, uint64_t validC1) const
    {
        if (block == 0U || validC1 == 0U || tiling_->n <= 0 || tiling_->outD <= 0 || tiling_->outH <= 0 ||
            tiling_->outW <= 0) {
            return 0U;
        }
        return static_cast<uint64_t>(tiling_->n) * static_cast<uint64_t>(tiling_->outD) * validC1 *
               static_cast<uint64_t>(tiling_->outH) * static_cast<uint64_t>(tiling_->outW) * block;
    }

    __aicore__ inline void DecodeNdc1hwc0Row(uint64_t row, uint64_t validC1, int64_t& nIdx, int64_t& od, int64_t& c1Idx,
                                             int64_t& oh) const
    {
        oh = static_cast<int64_t>(row % static_cast<uint64_t>(tiling_->outH));
        row /= static_cast<uint64_t>(tiling_->outH);
        if (validC1 == 0U) {
            return;
        }
        c1Idx = static_cast<int64_t>(row % validC1);
        row /= validC1;
        od = static_cast<int64_t>(row % static_cast<uint64_t>(tiling_->outD));
        nIdx = static_cast<int64_t>(row / static_cast<uint64_t>(tiling_->outD));
    }

    struct Ndc1hwc0DecodedRow {
        uint64_t row = 0U;
        int64_t nIdx = 0;
        int64_t od = 0;
        int64_t c1Idx = 0;
        int64_t oh = 0;
    };

    __aicore__ inline Ndc1hwc0DecodedRow DecodeNdc1hwc0RowContext(uint64_t row, uint64_t validC1) const
    {
        Ndc1hwc0DecodedRow context{};
        context.row = row;
        DecodeNdc1hwc0Row(row, validC1, context.nIdx, context.od, context.c1Idx, context.oh);
        return context;
    }

    __aicore__ inline Ndc1hwc0DecodedRow DecodeNdc1hwc0GroupRow(uint64_t cur, uint64_t rowElements,
                                                                uint64_t validC1) const
    {
        Ndc1hwc0DecodedRow context{};
        if (rowElements == 0U) {
            return context;
        }
        return DecodeNdc1hwc0RowContext(cur / rowElements, validC1);
    }

    __aicore__ inline void DecodeNdc1hwc0StorageRow(uint64_t row, int64_t& nIdx, int64_t& od, int64_t& c1Idx,
                                                    int64_t& oh) const
    {
        const uint64_t storageH = Ndc1hwc0StorageH();
        const uint64_t storageC1 = Ndc1hwc0StorageC1();
        const uint64_t storageD = Ndc1hwc0StorageD();
        if (storageH == 0U) {
            return;
        }
        oh = static_cast<int64_t>(row % storageH);
        row /= storageH;
        if (storageC1 == 0U) {
            return;
        }
        c1Idx = static_cast<int64_t>(row % storageC1);
        row /= storageC1;
        if (storageD == 0U) {
            return;
        }
        od = static_cast<int64_t>(row % storageD);
        nIdx = static_cast<int64_t>(row / storageD);
    }

    __aicore__ inline void CopyOutZeroRange(uint64_t outOffset, uint64_t outCount)
    {
        uint64_t processed = 0;
        while (processed < outCount) {
            const uint64_t remain = outCount - processed;
            const uint32_t curCount = remain > OUTPUT_TILE_NUM ? OUTPUT_TILE_NUM : static_cast<uint32_t>(remain);
            LocalTensor<T> yLocal = yOutQue_.AllocTensor<T>();
            Duplicate(yLocal, ZeroValue(), curCount);
            PipeBarrier<PIPE_V>();
            yOutQue_.EnQue(yLocal);

            LocalTensor<T> yOut = yOutQue_.DeQue<T>();
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(curCount * sizeof(T)), 0, 0, 0};
            DataCopyPad(yGm_[outOffset + processed], yOut, copyParams);
            yOutQue_.FreeTensor(yOut);
            processed += curCount;
        }
    }

    __aicore__ inline void CopyOutZeroRangeByCore(uint64_t outOffset, uint64_t outCount)
    {
        if (outCount == 0U) {
            return;
        }
        const uint64_t start = ValidCoreStartOffset(outCount);
        const uint64_t count = ValidCoreElementCount(outCount, start);
        if (count == 0U) {
            return;
        }
        CopyOutZeroRange(outOffset + start, count);
    }

    __aicore__ inline bool CanUseNdc1hwc0NdhwcCompactRow(uint32_t cCount, uint32_t block) const
    {
        const uint32_t rowCount = static_cast<uint32_t>(tiling_->outW) * cCount;
        const uint32_t rowElements = static_cast<uint32_t>(tiling_->outW) * block;
        const uint32_t alignedCount = AlignToVector(rowCount);
        if (tiling_->sW <= 0 || tiling_->dilationW <= 0 || tiling_->kW <= 0) {
            return false;
        }
        uint64_t maxSpanCount = 0U;
        uint32_t maxValidCount = 0U;
        for (int64_t kw = 0; kw < tiling_->kW; ++kw) {
            uint32_t wStart = 0U;
            uint32_t wCount = 0U;
            if (!CalcValidWRange(kw, wStart, wCount)) {
                continue;
            }
            const uint64_t spanW = tiling_->sW == 1 ?
                                       static_cast<uint64_t>(wCount) :
                                       static_cast<uint64_t>(wCount - 1U) * static_cast<uint64_t>(tiling_->sW) + 1U;
            const uint64_t spanCount = spanW * static_cast<uint64_t>(cCount);
            if (maxSpanCount < spanCount) {
                maxSpanCount = spanCount;
            }
            const uint32_t validCount = wCount * cCount;
            if (maxValidCount < validCount) {
                maxValidCount = validCount;
            }
        }
        const uint32_t gatherScratchOffset = Ndc1hwc0GatherTempOffset(rowElements);
        const uint32_t stridedScratchOffset = Ndc1hwc0GatherTempOffset(maxValidCount);
        return CanFitNdc1hwc0NdhwcCompactRow(cCount, block, rowCount, rowElements, alignedCount, maxSpanCount,
                                             maxValidCount, gatherScratchOffset, stridedScratchOffset);
    }

    __aicore__ inline bool CanFitNdc1hwc0NdhwcCompactRow(uint32_t cCount, uint32_t block, uint32_t rowCount,
                                                         uint32_t rowElements, uint32_t alignedCount,
                                                         uint64_t maxSpanCount, uint32_t maxValidCount,
                                                         uint32_t gatherScratchOffset,
                                                         uint32_t stridedScratchOffset) const
    {
        return rowCount > 0U && rowElements <= OUTPUT_TILE_NUM && rowCount + 1U <= OUTPUT_TILE_NUM &&
               alignedCount <= OUTPUT_TILE_NUM && maxSpanCount > 0U && maxSpanCount <= INPUT_TILE_NUM &&
               gatherScratchOffset < OUTPUT_TILE_NUM && rowCount <= OUTPUT_TILE_NUM - gatherScratchOffset &&
               (tiling_->sW == 1 || stridedScratchOffset + maxValidCount <= OUTPUT_TILE_NUM) && tiling_->c > 0 &&
               tiling_->c <= static_cast<int64_t>(block) && cCount == static_cast<uint32_t>(tiling_->c) &&
               (cCount == block || alignedCount > rowCount);
    }

    __aicore__ inline void CopyInVectorPadValue(uint64_t inputOffset, uint32_t validCount, uint32_t alignedCount,
                                                T padValue)
    {
        LocalTensor<T> xLocal = xInQue_.AllocTensor<T>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(validCount * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{true, 0, static_cast<uint8_t>(alignedCount - validCount), padValue};
        DataCopyPad(xLocal, xGm_[inputOffset], copyParams, padParams);
        xInQue_.EnQue(xLocal);
    }

    __aicore__ inline void InitNdc1hwc0NdhwcCompactGatherOffsets(LocalTensor<uint32_t> offsetLocal, uint32_t cCount,
                                                                 uint32_t block, uint32_t outW, uint32_t zeroIndex)
    {
        LocalTensor<int32_t> offsetI32 = offsetLocal.template ReinterpretCast<int32_t>();
        const int32_t zeroOffset = static_cast<int32_t>(zeroIndex * sizeof(T));
        for (uint32_t ow = 0; ow < outW; ++ow) {
            const uint32_t outBase = ow * block;
            const uint32_t srcBase = ow * cCount;
            for (uint32_t c0 = 0; c0 < block; ++c0) {
                int32_t srcOffset = zeroOffset;
                if (c0 < cCount) {
                    srcOffset = static_cast<int32_t>((srcBase + c0) * sizeof(T));
                }
                offsetI32.SetValue(outBase + c0, srcOffset);
            }
        }
        PipeBarrier<PIPE_V>();
    }

    struct GatherOffsetIndex {
        uint32_t ow;
        uint32_t c0;
    };

    __aicore__ inline GatherOffsetIndex DecodeGatherOffsetIndex(uint32_t start, uint32_t i, uint32_t block) const
    {
        if (block == 0U) {
            return {0U, 0U};
        }
        const uint32_t outIndex = start + i;
        const uint32_t ow = outIndex / block;
        return {ow, outIndex - ow * block};
    }

    __aicore__ inline void InitNdc1hwc0NdhwcGatherOffsetsChunk(LocalTensor<uint32_t> offsetLocal, uint32_t start,
                                                               uint32_t count, uint32_t cCount, uint32_t block,
                                                               uint32_t outW, uint32_t zeroIndex, uint32_t srcWStride)
    {
        LocalTensor<int32_t> offsetI32 = offsetLocal.template ReinterpretCast<int32_t>();
        const int32_t zeroOffset = static_cast<int32_t>(zeroIndex * sizeof(T));
        if (block == 0U) {
            return;
        }
        for (uint32_t i = 0; i < count; ++i) {
            const GatherOffsetIndex index = DecodeGatherOffsetIndex(start, i, block);
            int32_t srcOffset = zeroOffset;
            if (index.ow < outW && index.c0 < cCount) {
                srcOffset = static_cast<int32_t>((index.ow * srcWStride + index.c0) * sizeof(T));
            }
            offsetI32.SetValue(i, srcOffset);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void InitNdc1hwc0NdhwcCompactGatherOffsetsChunk(LocalTensor<uint32_t> offsetLocal, uint32_t start,
                                                                      uint32_t count, uint32_t cCount, uint32_t block,
                                                                      uint32_t outW, uint32_t zeroIndex)
    {
        InitNdc1hwc0NdhwcGatherOffsetsChunk(offsetLocal, start, count, cCount, block, outW, zeroIndex, cCount);
    }

    __aicore__ inline void InitNdc1hwc0NdhwcCompactStridedOffsets(LocalTensor<uint32_t> offsetLocal, uint32_t wCount,
                                                                  uint32_t cCount)
    {
        LocalTensor<int32_t> offsetI32 = offsetLocal.template ReinterpretCast<int32_t>();
        const uint32_t srcWStride = static_cast<uint32_t>(tiling_->sW) * cCount;
        for (uint32_t ow = 0; ow < wCount; ++ow) {
            const uint32_t srcBase = ow * srcWStride;
            const uint32_t dstBase = ow * cCount;
            for (uint32_t c0 = 0; c0 < cCount; ++c0) {
                offsetI32.SetValue(dstBase + c0, static_cast<int32_t>((srcBase + c0) * sizeof(T)));
            }
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ReduceNdc1hwc0NdhwcCompactW(LocalTensor<T> accLocal, int64_t nIdx, int64_t id, int64_t ih,
                                                       int64_t kw, uint32_t cCount, uint32_t alignedCount)
    {
        uint32_t wStart = 0U;
        uint32_t wCount = 0U;
        if (!CalcValidWRange(kw, wStart, wCount)) {
            return;
        }
        const int64_t iw = DilatedInputWFromStart(wStart, kw);
        const uint32_t validCount = wCount * cCount;
        const uint32_t alignedValidCount = AlignToVector(validCount);
        const uint64_t inputOffset = InputOffset(nIdx, id, ih, iw, 0);
        if (tiling_->sW == 1) {
            CopyInVectorPadValue(inputOffset, validCount, alignedValidCount, NegInfValue());
            LocalTensor<T> xLocal = xInQue_.DeQue<T>();
            const uint32_t dstOffset = wStart * cCount;
            const uint32_t reduceCount = static_cast<uint64_t>(dstOffset) + alignedValidCount <= alignedCount ?
                                             alignedValidCount :
                                             validCount;
            Max(accLocal[static_cast<uint64_t>(wStart) * cCount], accLocal[static_cast<uint64_t>(wStart) * cCount],
                xLocal, reduceCount);
            PipeBarrier<PIPE_V>();
            xInQue_.FreeTensor(xLocal);
            return;
        }
        const uint32_t spanW = (wCount - 1U) * static_cast<uint32_t>(tiling_->sW) + 1U;
        const uint32_t spanCount = spanW * cCount;
        const uint32_t alignedSpanCount = AlignToVector(spanCount);
        CopyInVectorPadValue(inputOffset, spanCount, alignedSpanCount, NegInfValue());
        LocalTensor<T> xLocal = xInQue_.DeQue<T>();
        LocalTensor<T> scratchLocal = tmpBuf_.Get<T>();
        LocalTensor<uint32_t> offsetLocal = scratchLocal.template ReinterpretCast<uint32_t>();
        InitNdc1hwc0NdhwcCompactStridedOffsets(offsetLocal, wCount, cCount);
        LocalTensor<T> gatheredLocal = scratchLocal[Ndc1hwc0GatherTempOffset(validCount)];
        Gather(gatheredLocal, xLocal, offsetLocal, static_cast<uint32_t>(0), validCount);
        PipeBarrier<PIPE_V>();
        Max(accLocal[static_cast<uint64_t>(wStart) * cCount], accLocal[static_cast<uint64_t>(wStart) * cCount],
            gatheredLocal, validCount);
        PipeBarrier<PIPE_V>();
        xInQue_.FreeTensor(xLocal);
    }

    __aicore__ inline void ReduceNdc1hwc0NdhwcCompactDepth(LocalTensor<T> accLocal, int64_t nIdx, int64_t id,
                                                           int64_t oh, uint32_t cCount, uint32_t alignedCount)
    {
        for (int64_t kh = 0; kh < tiling_->kH; ++kh) {
            const int64_t ih = DilatedInputH(oh, kh);
            if (IsOutOfRange(ih, tiling_->inH)) {
                continue;
            }
            for (int64_t kw = 0; kw < tiling_->kW; ++kw) {
                ReduceNdc1hwc0NdhwcCompactW(accLocal, nIdx, id, ih, kw, cCount, alignedCount);
            }
        }
    }

    __aicore__ inline void ReduceNdc1hwc0NdhwcCompactWindow(LocalTensor<T> accLocal, int64_t nIdx, int64_t od,
                                                            int64_t oh, uint32_t cCount, uint32_t alignedCount)
    {
        for (int64_t kd = 0; kd < tiling_->kD; ++kd) {
            const int64_t id = DilatedInputD(od, kd);
            if (!IsOutOfRange(id, tiling_->inD)) {
                ReduceNdc1hwc0NdhwcCompactDepth(accLocal, nIdx, id, oh, cCount, alignedCount);
            }
        }
    }

    __aicore__ inline void ComputeNdc1hwc0NdhwcCompactRowVector(LocalTensor<T> rowLocal, int64_t nIdx, int64_t od,
                                                                int64_t oh, uint32_t cCount, uint32_t block)
    {
        const uint32_t outW = static_cast<uint32_t>(tiling_->outW);
        const uint32_t rowCount = outW * cCount;
        const uint32_t alignedCount = AlignToVector(rowCount);
        if (CanUseNdc1hwc0NdhwcCompactK1Direct()) {
            ComputeNdc1hwc0NdhwcCompactK1Direct(rowLocal, nIdx, od, oh, cCount, block, rowCount);
            return;
        }
        LocalTensor<T> accLocal = calcBuf_.Get<T>();
        Duplicate(accLocal, NegInfValue(), alignedCount);
        PipeBarrier<PIPE_V>();
        ReduceNdc1hwc0NdhwcCompactWindow(accLocal, nIdx, od, oh, cCount, alignedCount);
        LocalTensor<T> resultLocal = accLocal;
        LocalTensor<T> scratchLocal = tmpBuf_.Get<T>();
        if (!ScatterNdc1hwc0NdhwcCompactRowGather(rowLocal, resultLocal, scratchLocal, cCount, block, outW, rowCount)) {
            ScatterNdc1hwc0NdhwcCompactRowScalar(rowLocal, resultLocal, cCount, block, outW);
            return;
        }
    }

    __aicore__ inline bool CanUseNdc1hwc0NdhwcCompactK1Direct() const
    {
        return tiling_->kD == 1 && tiling_->kH == 1 && tiling_->kW == 1 && tiling_->sD == 1 && tiling_->sH == 1 &&
               tiling_->sW == 1 && tiling_->dilationD == 1 && tiling_->dilationH == 1 && tiling_->dilationW == 1 &&
               tiling_->padFront == 0 && tiling_->padTop == 0 && tiling_->padLeft == 0 &&
               tiling_->outD <= tiling_->inD && tiling_->outH <= tiling_->inH && tiling_->outW <= tiling_->inW;
    }

    __aicore__ inline void ComputeNdc1hwc0NdhwcCompactK1Direct(LocalTensor<T> rowLocal, int64_t nIdx, int64_t od,
                                                               int64_t oh, uint32_t cCount, uint32_t block,
                                                               uint32_t rowCount)
    {
        const uint32_t outW = static_cast<uint32_t>(tiling_->outW);
        const uint32_t alignedCount = AlignToVector(rowCount + 1U);
        CopyInVectorPadValue(InputOffset(nIdx, od, oh, 0, 0), rowCount, alignedCount, ZeroValue());
        LocalTensor<T> xLocal = xInQue_.DeQue<T>();
        const uint32_t rowElements = outW * block;
        LocalTensor<T> scratchLocal = tmpBuf_.Get<T>();
        if (!ScatterNdc1hwc0NdhwcCompactRowGather(rowLocal, xLocal, scratchLocal, cCount, block, outW, rowCount)) {
            ScatterNdc1hwc0NdhwcCompactRowScalar(rowLocal, xLocal, cCount, block, outW);
            xInQue_.FreeTensor(xLocal);
            return;
        }
        xInQue_.FreeTensor(xLocal);
    }

    __aicore__ inline bool CanUseNdc1hwc0WideGather(uint32_t count) const
    {
        constexpr uint32_t safeGatherCount = 1024U;
        return count <= safeGatherCount;
    }

    __aicore__ inline void ScatterNdc1hwc0NdhwcCompactRowScalar(LocalTensor<T> rowLocal, LocalTensor<T> srcLocal,
                                                                uint32_t cCount, uint32_t block, uint32_t outW)
    {
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        for (uint32_t ow = 0; ow < outW; ++ow) {
            const uint64_t dstBase = static_cast<uint64_t>(ow) * block;
            const uint64_t srcBase = static_cast<uint64_t>(ow) * cCount;
            for (uint32_t c0 = 0; c0 < cCount; ++c0) {
                rowLocal.SetValue(dstBase + c0, srcLocal.GetValue(srcBase + c0));
            }
        }
    }

    __aicore__ inline void ZeroNdc1hwc0RowTail(LocalTensor<T> rowLocal, uint32_t cCount, uint32_t block, uint32_t outW)
    {
        if (cCount >= block) {
            return;
        }
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        for (uint32_t ow = 0; ow < outW; ++ow) {
            const uint64_t dstBase = static_cast<uint64_t>(ow) * block;
            for (uint32_t c0 = cCount; c0 < block; ++c0) {
                rowLocal.SetValue(dstBase + c0, ZeroValue());
            }
        }
    }

    __aicore__ inline bool ScatterNdc1hwc0NdhwcCompactRowGather(LocalTensor<T> rowLocal, LocalTensor<T> srcLocal,
                                                                LocalTensor<T> scratchLocal, uint32_t cCount,
                                                                uint32_t block, uint32_t outW, uint32_t rowCount)
    {
        if (cCount == 0U || block == 0U || outW == 0U || cCount > block) {
            return false;
        }
        const uint32_t rowElements = outW * block;
        if (rowElements == 0U || rowElements > OUTPUT_TILE_NUM || rowCount + 1U > OUTPUT_TILE_NUM) {
            return false;
        }
        if (cCount == block) {
            CopyLocalTensor(rowLocal, srcLocal, rowElements);
            PipeBarrier<PIPE_V>();
            return true;
        }
        const uint32_t zeroIndex = AlignToVector(rowCount);
        if (zeroIndex >= OUTPUT_TILE_NUM) {
            return false;
        }
        Duplicate(srcLocal[zeroIndex], ZeroValue(), 1);
        PipeBarrier<PIPE_V>();

        LocalTensor<uint32_t> offsetLocal = scratchLocal.template ReinterpretCast<uint32_t>();
        if (CanUseNdc1hwc0WideGather(rowElements)) {
            InitNdc1hwc0NdhwcCompactGatherOffsets(offsetLocal, cCount, block, outW, zeroIndex);
            Gather(rowLocal, srcLocal, offsetLocal, static_cast<uint32_t>(0), rowElements);
            PipeBarrier<PIPE_V>();
            return true;
        }

        uint32_t done = 0U;
        while (done < rowElements) {
            const uint32_t curCount = Ndc1hwc0SafeGatherChunk(rowElements - done);
            InitNdc1hwc0NdhwcCompactGatherOffsetsChunk(offsetLocal, done, curCount, cCount, block, outW, zeroIndex);
            Gather(rowLocal[done], srcLocal, offsetLocal, static_cast<uint32_t>(0), curCount);
            PipeBarrier<PIPE_V>();
            done += curCount;
        }
        return true;
    }

    __aicore__ inline uint32_t Ndc1hwc0MaxCompactTileRows(uint32_t rows, uint32_t compactStride,
                                                          uint32_t rowElements) const
    {
        return Ndc1hwc0MaxCompactTileRowsLimit(rows, compactStride, rowElements, OUTPUT_TILE_NUM);
    }

    __aicore__ inline uint32_t Ndc1hwc0MaxCompactTileRowsLimit(uint32_t rows, uint32_t compactStride,
                                                               uint32_t rowElements, uint32_t tileLimit) const
    {
        uint32_t tileRows = rows;
        if (tileRows == 0U) {
            return 1U;
        }
        const uint32_t maxRowsByOutput = rowElements == 0U ? 1U : tileLimit / rowElements;
        if (maxRowsByOutput > 0U && tileRows > maxRowsByOutput) {
            tileRows = maxRowsByOutput;
        }
        while (tileRows > 1U) {
            const uint32_t compactNeed = tileRows * compactStride;
            const uint32_t offsetBase = AlignToVector(compactNeed + 1U);
            const uint32_t offsetNeed = Ndc1hwc0GatherTempOffset(tileRows * rowElements);
            if (offsetBase + offsetNeed <= tileLimit) {
                break;
            }
            --tileRows;
        }
        return tileRows == 0U ? 1U : tileRows;
    }

    __aicore__ inline void GatherNdc1hwc0CompactTile(LocalTensor<T> outLocal, LocalTensor<T> compactLocal,
                                                     LocalTensor<uint32_t> offsetLocal, uint32_t totalCount)
    {
        constexpr uint32_t safeGatherCount = 1024U;
        uint32_t done = 0U;
        while (done < totalCount) {
            uint32_t curCount = totalCount - done;
            if (curCount > safeGatherCount) {
                curCount = safeGatherCount;
            }
            Gather(outLocal[done], compactLocal, offsetLocal[done], static_cast<uint32_t>(0), curCount);
            PipeBarrier<PIPE_V>();
            done += curCount;
        }
    }

    __aicore__ inline void InitNdc1hwc0CompactTileActiveOffsets(LocalTensor<uint32_t> offsetLocal, uint32_t tileRows,
                                                                uint32_t compactStride, uint32_t cCount, uint32_t block,
                                                                uint32_t outW, uint32_t srcWStep, uint32_t srcCStep,
                                                                uint32_t zeroIndex)
    {
        LocalTensor<int32_t> offsetI32 = offsetLocal.template ReinterpretCast<int32_t>();
        const int32_t zeroOffset = static_cast<int32_t>(zeroIndex * sizeof(T));
        for (uint32_t rowIdx = 0; rowIdx < tileRows; ++rowIdx) {
            const uint32_t outRowBase = rowIdx * outW * block;
            const uint32_t srcRowBase = rowIdx * compactStride;
            for (uint32_t ow = 0; ow < outW; ++ow) {
                const uint32_t outBase = outRowBase + ow * block;
                const uint32_t srcWBase = srcRowBase + ow * srcWStep;
                for (uint32_t c0 = 0; c0 < block; ++c0) {
                    int32_t srcOffset = zeroOffset;
                    if (c0 < cCount) {
                        srcOffset = static_cast<int32_t>((srcWBase + c0 * srcCStep) * sizeof(T));
                    }
                    offsetI32.SetValue(outBase + c0, srcOffset);
                }
            }
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void InitNdc1hwc0CompactRowActiveOffsets(LocalTensor<uint32_t> offsetLocal, uint32_t srcRowBase,
                                                               uint32_t cCount, uint32_t block, uint32_t outW,
                                                               uint32_t srcWStep, uint32_t srcCStep, uint32_t zeroIndex)
    {
        LocalTensor<int32_t> offsetI32 = offsetLocal.template ReinterpretCast<int32_t>();
        const int32_t zeroOffset = static_cast<int32_t>(zeroIndex * sizeof(T));
        for (uint32_t ow = 0; ow < outW; ++ow) {
            const uint32_t outBase = ow * block;
            const uint32_t srcWBase = srcRowBase + ow * srcWStep;
            for (uint32_t c0 = 0; c0 < block; ++c0) {
                int32_t srcOffset = zeroOffset;
                if (c0 < cCount) {
                    srcOffset = static_cast<int32_t>((srcWBase + c0 * srcCStep) * sizeof(T));
                }
                offsetI32.SetValue(outBase + c0, srcOffset);
            }
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline bool ScatterNdc1hwc0CompactTileGather(LocalTensor<T> outLocal, LocalTensor<T> compactLocal,
                                                            LocalTensor<uint32_t> offsetLocal, uint32_t tileRows,
                                                            uint32_t compactStride, uint32_t cCount, uint32_t block,
                                                            uint32_t outW, uint32_t srcWStep, uint32_t srcCStep,
                                                            uint32_t zeroIndex)
    {
        if (tileRows == 0U || compactStride == 0U || cCount == 0U || block == 0U || outW == 0U || srcWStep == 0U ||
            srcCStep == 0U || cCount > block) {
            return false;
        }
        const uint32_t totalElements = tileRows * outW * block;
        if (totalElements == 0U || totalElements > OUTPUT_TILE_NUM) {
            return false;
        }
        Duplicate(compactLocal[zeroIndex], ZeroValue(), 1);
        PipeBarrier<PIPE_V>();
        InitNdc1hwc0CompactTileActiveOffsets(offsetLocal, tileRows, compactStride, cCount, block, outW, srcWStep,
                                             srcCStep, zeroIndex);
        GatherNdc1hwc0CompactTile(outLocal, compactLocal, offsetLocal, totalElements);
        return true;
    }

    __aicore__ inline void ScatterNdc1hwc0CompactRowWithOffsets(LocalTensor<T> rowLocal, LocalTensor<T> compactRow,
                                                                LocalTensor<uint32_t> offsetLocal, uint32_t rowElements,
                                                                uint32_t zeroIndex)
    {
        Duplicate(compactRow[zeroIndex], ZeroValue(), 1);
        PipeBarrier<PIPE_V>();
        GatherNdc1hwc0CompactTile(rowLocal, compactRow, offsetLocal, rowElements);
    }

    __aicore__ inline void ScatterNdc1hwc0CompactRowWithOffsetsNoZero(LocalTensor<T> rowLocal,
                                                                      LocalTensor<T> compactRow,
                                                                      LocalTensor<uint32_t> offsetLocal,
                                                                      uint32_t rowElements)
    {
        GatherNdc1hwc0CompactTile(rowLocal, compactRow, offsetLocal, rowElements);
    }

    __aicore__ inline bool ScatterNdc1hwc0CompactRowsReuseRowOffset(
        LocalTensor<T> outLocal, LocalTensor<T> compactLocal, LocalTensor<uint32_t> offsetLocal, uint32_t tileRows,
        uint32_t compactStride, uint32_t cCount, uint32_t block, uint32_t outW, uint32_t srcWStep, uint32_t srcCStep,
        uint32_t zeroIndex)
    {
        if (tileRows == 0U || compactStride == 0U || cCount == 0U || block == 0U || outW == 0U || srcWStep == 0U ||
            srcCStep == 0U || cCount > block || zeroIndex >= compactStride) {
            return false;
        }
        const uint32_t rowElements = outW * block;
        if (rowElements == 0U || rowElements > OUTPUT_TILE_NUM ||
            Ndc1hwc0GatherTempOffset(rowElements) > OUTPUT_TILE_NUM) {
            return false;
        }
        InitNdc1hwc0CompactRowActiveOffsets(offsetLocal, 0U, cCount, block, outW, srcWStep, srcCStep, zeroIndex);
        for (uint32_t rowIdx = 0; rowIdx < tileRows; ++rowIdx) {
            ScatterNdc1hwc0CompactRowWithOffsets(outLocal[static_cast<uint64_t>(rowIdx) * rowElements],
                                                 compactLocal[static_cast<uint64_t>(rowIdx) * compactStride],
                                                 offsetLocal, rowElements, zeroIndex);
        }
        return true;
    }

    __aicore__ inline bool ScatterNdc1hwc0CompactRowsWithRowOffset(LocalTensor<T> outLocal, LocalTensor<T> compactLocal,
                                                                   LocalTensor<uint32_t> offsetLocal, uint32_t tileRows,
                                                                   uint32_t compactStride, uint32_t block,
                                                                   uint32_t outW, uint32_t zeroIndex)
    {
        if (tileRows == 0U || compactStride == 0U || block == 0U || outW == 0U || zeroIndex >= compactStride) {
            return false;
        }
        const uint32_t rowElements = outW * block;
        if (rowElements == 0U || rowElements > OUTPUT_TILE_NUM ||
            Ndc1hwc0GatherTempOffset(rowElements) > OUTPUT_TILE_NUM) {
            return false;
        }
        for (uint32_t rowIdx = 0; rowIdx < tileRows; ++rowIdx) {
            ScatterNdc1hwc0CompactRowWithOffsets(outLocal[static_cast<uint64_t>(rowIdx) * rowElements],
                                                 compactLocal[static_cast<uint64_t>(rowIdx) * compactStride],
                                                 offsetLocal, rowElements, zeroIndex);
        }
        return true;
    }

    __aicore__ inline bool ScatterNdc1hwc0CompactTileActiveChannels(LocalTensor<T> outLocal,
                                                                    LocalTensor<T> compactLocal, uint32_t tileRows,
                                                                    uint32_t compactStride, uint32_t cCount,
                                                                    uint32_t block, uint32_t outW, uint32_t srcWStep,
                                                                    uint32_t srcCStep)
    {
        if (tileRows == 0U || compactStride == 0U || cCount == 0U || block == 0U || outW == 0U || srcWStep == 0U ||
            srcCStep == 0U || cCount > block || tileRows * outW == 0U) {
            return false;
        }
        const uint32_t rowElements = outW * block;
        const uint32_t totalElements = tileRows * rowElements;
        if (totalElements == 0U || totalElements > OUTPUT_TILE_NUM) {
            return false;
        }
        const uint32_t zeroIndex = tileRows * compactStride;
        const uint32_t offsetBase = AlignToVector(zeroIndex + 1U);
        const uint32_t offsetNeed = Ndc1hwc0GatherTempOffset(totalElements);
        if (offsetBase + offsetNeed > OUTPUT_TILE_NUM) {
            return false;
        }
        LocalTensor<uint32_t> offsetLocal = compactLocal[offsetBase].template ReinterpretCast<uint32_t>();
        return ScatterNdc1hwc0CompactTileGather(outLocal, compactLocal, offsetLocal, tileRows, compactStride, cCount,
                                                block, outW, srcWStep, srcCStep, zeroIndex);
    }

    __aicore__ inline void ScatterNdc1hwc0CompactRowsScalar(LocalTensor<T> outLocal, LocalTensor<T> compactLocal,
                                                            uint32_t tileRows, uint32_t compactStride, uint32_t cCount,
                                                            uint32_t block, uint32_t outW, uint32_t srcWStep,
                                                            uint32_t srcCStep)
    {
        const uint32_t rowElements = outW * block;
        for (uint32_t rowIdx = 0U; rowIdx < tileRows; ++rowIdx) {
            const uint32_t outRowBase = rowIdx * rowElements;
            const uint32_t srcRowBase = rowIdx * compactStride;
            for (uint32_t ow = 0U; ow < outW; ++ow) {
                const uint32_t outBase = outRowBase + ow * block;
                const uint32_t srcWBase = srcRowBase + ow * srcWStep;
                for (uint32_t c0 = 0U; c0 < block; ++c0) {
                    const T value = c0 < cCount ? compactLocal.GetValue(srcWBase + c0 * srcCStep) : ZeroValue();
                    outLocal.SetValue(outBase + c0, value);
                }
            }
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline bool ScatterNdc1hwc0CompactRowsSmallCScalar(LocalTensor<T> outLocal, LocalTensor<T> compactLocal,
                                                                  uint32_t tileRows, uint32_t compactStride,
                                                                  uint32_t cCount, uint32_t block, uint32_t outW,
                                                                  uint32_t srcWStep, uint32_t srcCStep)
    {
        if (tileRows == 0U || compactStride == 0U || cCount == 0U || block == 0U || outW == 0U || srcWStep == 0U ||
            srcCStep == 0U || cCount > block || cCount > 8U || cCount * 2U > block) {
            return false;
        }
        const uint32_t rowElements = outW * block;
        const uint32_t totalElements = tileRows * rowElements;
        if (rowElements == 0U || totalElements == 0U || totalElements > OUTPUT_TILE_NUM) {
            return false;
        }
        Duplicate(outLocal, ZeroValue(), totalElements);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        for (uint32_t rowIdx = 0U; rowIdx < tileRows; ++rowIdx) {
            const uint32_t outRowBase = rowIdx * rowElements;
            const uint32_t srcRowBase = rowIdx * compactStride;
            for (uint32_t ow = 0U; ow < outW; ++ow) {
                const uint32_t outBase = outRowBase + ow * block;
                const uint32_t srcWBase = srcRowBase + ow * srcWStep;
                for (uint32_t c0 = 0U; c0 < cCount; ++c0) {
                    outLocal.SetValue(outBase + c0, compactLocal.GetValue(srcWBase + c0 * srcCStep));
                }
            }
        }
        PipeBarrier<PIPE_V>();
        return true;
    }

    __aicore__ inline void ScatterNdc1hwc0CompactRowsChecked(LocalTensor<T> outLocal, LocalTensor<T> compactLocal,
                                                             LocalTensor<uint32_t> offsetLocal, uint32_t tileRows,
                                                             uint32_t compactStride, uint32_t cCount, uint32_t block,
                                                             uint32_t outW, uint32_t srcWStep, uint32_t srcCStep,
                                                             uint32_t zeroIndex)
    {
        if (ScatterNdc1hwc0CompactTileActiveChannels(outLocal, compactLocal, tileRows, compactStride, cCount, block,
                                                     outW, srcWStep, srcCStep)) {
            return;
        }
        if (ScatterNdc1hwc0CompactRowsReuseRowOffset(outLocal, compactLocal, offsetLocal, tileRows, compactStride,
                                                     cCount, block, outW, srcWStep, srcCStep, zeroIndex)) {
            return;
        }
        if (ScatterNdc1hwc0CompactRowsSmallCScalar(outLocal, compactLocal, tileRows, compactStride, cCount, block, outW,
                                                   srcWStep, srcCStep)) {
            return;
        }
        ScatterNdc1hwc0CompactRowsScalar(outLocal, compactLocal, tileRows, compactStride, cCount, block, outW, srcWStep,
                                         srcCStep);
    }

    __aicore__ inline bool GetNdc1hwc0D3H3ValidRowRange(uint32_t ohStart, uint32_t tileRows, int64_t kh,
                                                        int64_t& beginRow, int64_t& endRow) const
    {
        const int64_t hOffset = kh * 2 - 2;
        beginRow = 0;
        const int64_t firstH = static_cast<int64_t>(ohStart) + hOffset;
        if (firstH < 0) {
            beginRow = -firstH;
        }
        endRow = static_cast<int64_t>(tileRows);
        const int64_t lastValidExclusive = tiling_->inH - (static_cast<int64_t>(ohStart) + hOffset);
        if (endRow > lastValidExclusive) {
            endRow = lastValidExclusive;
        }
        return beginRow < endRow;
    }

    __aicore__ inline void MaxNdc1hwc0NdhwcD3H3CompactRows(LocalTensor<T> compactLocal, LocalTensor<T> dmaxLocal,
                                                           uint32_t ohStart, uint32_t tileRows, uint32_t compactStride)
    {
        const uint32_t totalCount = tileRows * compactStride;
        Duplicate(compactLocal, NegInfValue(), totalCount);
        PipeBarrier<PIPE_V>();
        for (int64_t kh = 0; kh < 3; ++kh) {
            int64_t beginRow = 0;
            int64_t endRow = 0;
            if (!GetNdc1hwc0D3H3ValidRowRange(ohStart, tileRows, kh, beginRow, endRow)) {
                continue;
            }
            const int64_t hOffset = kh * 2 - 2;
            const uint32_t validRows = static_cast<uint32_t>(endRow - beginRow);
            const uint32_t dstOffset = static_cast<uint32_t>(beginRow) * compactStride;
            const uint32_t srcH = static_cast<uint32_t>(static_cast<int64_t>(ohStart) + beginRow + hOffset);
            Max(compactLocal[dstOffset], compactLocal[dstOffset],
                dmaxLocal[static_cast<uint64_t>(srcH) * compactStride], validRows * compactStride);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void MaxNdc1hwc0NcdhwD3H3CompactRowsKh(LocalTensor<T> compactLocal, LocalTensor<T> dmaxLocal,
                                                             uint32_t ohStart, uint32_t tileRows, uint32_t cCount,
                                                             uint32_t alignedW, uint32_t compactStride, int64_t kh,
                                                             const BinaryRepeatParams& params)
    {
        int64_t beginRow = 0;
        int64_t endRow = 0;
        if (!GetNdc1hwc0D3H3ValidRowRange(ohStart, tileRows, kh, beginRow, endRow)) {
            return;
        }
        const int64_t hOffset = kh * 2 - 2;
        const uint8_t repeatTimes = static_cast<uint8_t>(endRow - beginRow);
        const uint32_t dstRowOffset = static_cast<uint32_t>(beginRow) * compactStride;
        const uint32_t srcH = static_cast<uint32_t>(static_cast<int64_t>(ohStart) + beginRow + hOffset);
        const uint32_t inH = static_cast<uint32_t>(tiling_->inH);
        for (uint32_t c0 = 0; c0 < cCount; ++c0) {
            const uint64_t dstBase = static_cast<uint64_t>(dstRowOffset) + static_cast<uint64_t>(c0) * alignedW;
            const uint64_t srcBase = (static_cast<uint64_t>(c0) * inH + srcH) * alignedW;
            Max(compactLocal[dstBase], compactLocal[dstBase], dmaxLocal[srcBase], alignedW, repeatTimes, params);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline bool MaxNdc1hwc0NcdhwD3H3CompactRowsRepeat(LocalTensor<T> compactLocal, LocalTensor<T> dmaxLocal,
                                                                 uint32_t ohStart, uint32_t tileRows, uint32_t cCount,
                                                                 uint32_t alignedW, uint32_t compactStride)
    {
        if constexpr (AscendC::Std::is_same<T, bfloat16_t>::value) {
            return false;
        } else {
            if (alignedW == 0U || compactStride == 0U || tileRows == 0U || tileRows > 255U) {
                return false;
            }
            const uint32_t dstRepStride = static_cast<uint32_t>(static_cast<uint64_t>(compactStride) * sizeof(T) /
                                                                UB_BLOCK_BYTES);
            const uint32_t srcRepStride = static_cast<uint32_t>(static_cast<uint64_t>(alignedW) * sizeof(T) /
                                                                UB_BLOCK_BYTES);
            if (dstRepStride == 0U || srcRepStride == 0U || dstRepStride > 255U || srcRepStride > 255U ||
                (static_cast<uint64_t>(compactStride) * sizeof(T)) % UB_BLOCK_BYTES != 0U ||
                (static_cast<uint64_t>(alignedW) * sizeof(T)) % UB_BLOCK_BYTES != 0U) {
                return false;
            }
            const BinaryRepeatParams params{1U,
                                            1U,
                                            1U,
                                            static_cast<uint8_t>(dstRepStride),
                                            static_cast<uint8_t>(dstRepStride),
                                            static_cast<uint8_t>(srcRepStride)};
            Duplicate(compactLocal, NegInfValue(), tileRows * compactStride);
            PipeBarrier<PIPE_V>();
            for (int64_t kh = 0; kh < 3; ++kh) {
                MaxNdc1hwc0NcdhwD3H3CompactRowsKh(compactLocal, dmaxLocal, ohStart, tileRows, cCount, alignedW,
                                                  compactStride, kh, params);
            }
            return true;
        }
    }

    __aicore__ inline void MaxNdc1hwc0NcdhwD3H3CompactRows(LocalTensor<T> compactLocal, LocalTensor<T> dmaxLocal,
                                                           uint32_t ohStart, uint32_t tileRows, uint32_t cCount,
                                                           uint32_t alignedW, uint32_t compactStride)
    {
        if (MaxNdc1hwc0NcdhwD3H3CompactRowsRepeat(compactLocal, dmaxLocal, ohStart, tileRows, cCount, alignedW,
                                                  compactStride)) {
            return;
        }
        Duplicate(compactLocal, NegInfValue(), tileRows * compactStride);
        PipeBarrier<PIPE_V>();
        for (int64_t kh = 0; kh < 3; ++kh) {
            const int64_t hOffset = kh * 2 - 2;
            for (uint32_t rowIdx = 0; rowIdx < tileRows; ++rowIdx) {
                const int64_t ih = static_cast<int64_t>(ohStart + rowIdx) + hOffset;
                if (IsOutOfRange(ih, tiling_->inH)) {
                    continue;
                }
                LocalTensor<T> compactRowLocal = compactLocal[static_cast<uint64_t>(rowIdx) * compactStride];
                for (uint32_t c0 = 0; c0 < cCount; ++c0) {
                    const uint64_t dmaxOffset = (static_cast<uint64_t>(c0) * static_cast<uint32_t>(tiling_->inH) +
                                                 static_cast<uint64_t>(ih)) *
                                                alignedW;
                    Max(compactRowLocal[static_cast<uint64_t>(c0) * alignedW],
                        compactRowLocal[static_cast<uint64_t>(c0) * alignedW], dmaxLocal[dmaxOffset], alignedW);
                }
                PipeBarrier<PIPE_V>();
            }
        }
    }

    __aicore__ inline void InitNdc1hwc0NdhwcAlignedGatherOffsetsChunk(LocalTensor<uint32_t> offsetLocal, uint32_t start,
                                                                      uint32_t count, uint32_t cCount, uint32_t block,
                                                                      uint32_t alignedCount, uint32_t outW,
                                                                      uint32_t zeroIndex)
    {
        InitNdc1hwc0NdhwcGatherOffsetsChunk(offsetLocal, start, count, cCount, block, outW, zeroIndex, alignedCount);
    }

    __aicore__ inline bool ScatterNdc1hwc0NdhwcAlignedRowGather(LocalTensor<T> rowLocal, LocalTensor<T> resultLocal,
                                                                LocalTensor<T> scratchLocal, uint32_t cCount,
                                                                uint32_t block, uint32_t outW, uint32_t alignedCount)
    {
        if (cCount == 0U || block == 0U || outW == 0U || alignedCount == 0U || cCount > block ||
            cCount > alignedCount) {
            return false;
        }
        const uint32_t rowElements = outW * block;
        const uint32_t inputElements = outW * alignedCount;
        if (rowElements > OUTPUT_TILE_NUM || inputElements > OUTPUT_TILE_NUM) {
            return false;
        }
        if (cCount == block && alignedCount == block) {
            CopyLocalTensor(rowLocal, resultLocal, rowElements);
            PipeBarrier<PIPE_V>();
            return true;
        }
        if (inputElements + 1U > OUTPUT_TILE_NUM) {
            return false;
        }

        Duplicate(resultLocal[inputElements], ZeroValue(), 1);
        PipeBarrier<PIPE_V>();
        LocalTensor<uint32_t> offsetLocal = scratchLocal.template ReinterpretCast<uint32_t>();
        uint32_t done = 0U;
        while (done < rowElements) {
            const uint32_t curCount = Ndc1hwc0SafeGatherChunk(rowElements - done);
            InitNdc1hwc0NdhwcAlignedGatherOffsetsChunk(offsetLocal, done, curCount, cCount, block, alignedCount, outW,
                                                       inputElements);
            Gather(rowLocal[done], resultLocal, offsetLocal, static_cast<uint32_t>(0), curCount);
            PipeBarrier<PIPE_V>();
            done += curCount;
        }
        return true;
    }

    __aicore__ inline void ReduceNdc1hwc0NdhwcRowHeight(LocalTensor<T> accLocal, int64_t nIdx, int64_t id,
                                                        int64_t cBase, int64_t ih, uint32_t cCount,
                                                        uint32_t alignedCount)
    {
        for (int64_t kw = 0; kw < tiling_->kW; ++kw) {
            uint32_t wStart = 0U;
            uint32_t wCount = 0U;
            if (!CalcValidWRange(kw, wStart, wCount)) {
                continue;
            }
            const int64_t iw = DilatedInputWFromStart(wStart, kw);
            const uint64_t inputOffset = InputOffset(nIdx, id, ih, iw, cBase);
            const uint32_t srcStrideElements = static_cast<uint32_t>(
                static_cast<uint64_t>(tiling_->sW) * static_cast<uint64_t>(tiling_->c) - static_cast<uint64_t>(cCount));
            CopyInVectorWBlocksPadStride(inputOffset, wCount, cCount, alignedCount, srcStrideElements);
            LocalTensor<T> xLocal = xInQue_.DeQue<T>();
            Max(accLocal[static_cast<uint64_t>(wStart) * alignedCount],
                accLocal[static_cast<uint64_t>(wStart) * alignedCount], xLocal, wCount * alignedCount);
            PipeBarrier<PIPE_V>();
            xInQue_.FreeTensor(xLocal);
        }
    }

    __aicore__ inline void ReduceNdc1hwc0NdhwcRowDepth(LocalTensor<T> accLocal, int64_t nIdx, int64_t id, int64_t cBase,
                                                       int64_t oh, uint32_t cCount, uint32_t alignedCount)
    {
        for (int64_t kh = 0; kh < tiling_->kH; ++kh) {
            const int64_t ih = DilatedInputH(oh, kh);
            if (IsOutOfRange(ih, tiling_->inH)) {
                continue;
            }
            ReduceNdc1hwc0NdhwcRowHeight(accLocal, nIdx, id, cBase, ih, cCount, alignedCount);
        }
    }

    __aicore__ inline void ReduceNdc1hwc0NdhwcRowWindow(LocalTensor<T> accLocal, int64_t nIdx, int64_t od,
                                                        int64_t cBase, int64_t oh, uint32_t cCount,
                                                        uint32_t alignedCount)
    {
        for (int64_t kd = 0; kd < tiling_->kD; ++kd) {
            const int64_t id = DilatedInputD(od, kd);
            if (!IsOutOfRange(id, tiling_->inD)) {
                ReduceNdc1hwc0NdhwcRowDepth(accLocal, nIdx, id, cBase, oh, cCount, alignedCount);
            }
        }
    }

    __aicore__ inline void ScatterNdc1hwc0NdhwcRowScalar(LocalTensor<T> rowLocal, LocalTensor<T> accLocal,
                                                         uint32_t cCount, uint32_t block, uint32_t outW,
                                                         uint32_t alignedCount)
    {
        for (uint32_t ow = 0; ow < outW; ++ow) {
            const uint64_t dstBase = static_cast<uint64_t>(ow) * block;
            const uint64_t srcBase = static_cast<uint64_t>(ow) * alignedCount;
            for (uint32_t c0 = 0; c0 < cCount; ++c0) {
                rowLocal.SetValue(dstBase + c0, accLocal.GetValue(srcBase + c0));
            }
        }
    }

    __aicore__ inline void ComputeNdc1hwc0NdhwcRowVector(LocalTensor<T> rowLocal, int64_t nIdx, int64_t od,
                                                         int64_t cBase, int64_t oh, uint32_t cCount, uint32_t block)
    {
        const uint32_t outW = static_cast<uint32_t>(tiling_->outW);
        if (CanUseNdc1hwc0NdhwcK1TailDirectRow(cBase, cCount, block)) {
            ComputeNdc1hwc0NdhwcK1TailDirectRow(rowLocal, nIdx, od, cBase, oh, cCount, block, outW);
            return;
        }
        if (CanUseNdc1hwc0NdhwcStride2Pool2Row(cBase, cCount, block)) {
            ComputeNdc1hwc0NdhwcStride2Pool2Row(rowLocal, nIdx, od, cBase, oh, cCount, block, outW);
            return;
        }
        if (cBase == 0 && CanUseNdc1hwc0NdhwcCompactRow(cCount, block)) {
            ComputeNdc1hwc0NdhwcCompactRowVector(rowLocal, nIdx, od, oh, cCount, block);
            return;
        }
        uint32_t alignedCount = AlignToVector(cCount);
        if (alignedCount < cCount) {
            alignedCount = cCount;
        }
        const uint32_t totalCount = outW * alignedCount;
        LocalTensor<T> accLocal = calcBuf_.Get<T>();
        Duplicate(accLocal, NegInfValue(), totalCount);
        PipeBarrier<PIPE_V>();
        ReduceNdc1hwc0NdhwcRowWindow(accLocal, nIdx, od, cBase, oh, cCount, alignedCount);
        LocalTensor<T> scratchLocal = tmpBuf_.Get<T>();
        if (ScatterNdc1hwc0NdhwcAlignedRowGather(rowLocal, accLocal, scratchLocal, cCount, block, outW, alignedCount)) {
            return;
        }
        ScatterNdc1hwc0NdhwcRowScalar(rowLocal, accLocal, cCount, block, outW, alignedCount);
    }

    __aicore__ inline bool CanUseNdc1hwc0NdhwcK1TailDirectRow(int64_t cBase, uint32_t cCount, uint32_t block) const
    {
        if (!HasNdhwcK1TailDirectArgs(cBase, cCount, block) || !HasNdhwcK1UnitPool()) {
            return false;
        }
        return tiling_->outD <= tiling_->inD && tiling_->outH <= tiling_->inH && tiling_->outW <= tiling_->inW &&
               cBase + static_cast<int64_t>(cCount) <= tiling_->c;
    }

    __aicore__ inline bool HasNdhwcK1TailDirectArgs(int64_t cBase, uint32_t cCount, uint32_t block) const
    {
        return tiling_->dataFormat == FORMAT_NDHWC_VALUE && cBase > 0 && cCount > 0U && block > 0U && cCount < block &&
               tiling_->c > 0 && tiling_->outW > 0;
    }

    __aicore__ inline bool HasNdhwcK1UnitPool() const
    {
        return tiling_->kD == 1 && tiling_->kH == 1 && tiling_->kW == 1 && tiling_->sD == 1 && tiling_->sH == 1 &&
               tiling_->sW == 1 && tiling_->dilationD == 1 && tiling_->dilationH == 1 && tiling_->dilationW == 1 &&
               tiling_->padFront == 0 && tiling_->padTop == 0 && tiling_->padLeft == 0;
    }

    __aicore__ inline void ComputeNdc1hwc0NdhwcK1TailDirectRow(LocalTensor<T> rowLocal, int64_t nIdx, int64_t od,
                                                               int64_t cBase, int64_t oh, uint32_t cCount,
                                                               uint32_t block, uint32_t outW)
    {
        for (uint32_t ow = 0; ow < outW; ++ow) {
            const uint64_t inputBase = InputOffset(nIdx, od, oh, static_cast<int64_t>(ow), cBase);
            const uint64_t dstBase = static_cast<uint64_t>(ow) * block;
            for (uint32_t c0 = 0; c0 < cCount; ++c0) {
                rowLocal.SetValue(dstBase + c0, xGm_.GetValue(inputBase + c0));
            }
            for (uint32_t c0 = cCount; c0 < block; ++c0) {
                rowLocal.SetValue(dstBase + c0, ZeroValue());
            }
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline bool CanUseNdc1hwc0NdhwcStride2Pool2Row(int64_t cBase, uint32_t cCount, uint32_t block) const
    {
        if (tiling_->dataFormat != FORMAT_NDHWC_VALUE || cCount == 0U || block == 0U || cCount > block ||
            tiling_->outW <= 0 || tiling_->inW <= 0 || tiling_->c <= 0 || !IsPool2Stride2NoPad()) {
            return false;
        }
        if (cBase < 0 || cBase + static_cast<int64_t>(cCount) > tiling_->c) {
            return false;
        }
        const uint32_t outW = static_cast<uint32_t>(tiling_->outW);
        const uint32_t rowElements = outW * block;
        const uint32_t validInputCount = static_cast<uint32_t>(tiling_->inW) * static_cast<uint32_t>(tiling_->c);
        const uint32_t alignedInputCount = AlignToVector(validInputCount + 1U);
        const uint32_t offsetElements = Ndc1hwc0GatherTempOffset(rowElements);
        const uint32_t scratchNeed = offsetElements * 2U + rowElements;
        return rowElements > 0U && rowElements <= OUTPUT_TILE_NUM && alignedInputCount <= INPUT_TILE_NUM &&
               scratchNeed <= OUTPUT_TILE_NUM;
    }

    __aicore__ inline void InitNdc1hwc0NdhwcStride2GatherOffsets(LocalTensor<uint32_t> evenOffset,
                                                                 LocalTensor<uint32_t> oddOffset, int64_t cBase,
                                                                 uint32_t cCount, uint32_t block, uint32_t outW)
    {
        LocalTensor<int32_t> evenI32 = evenOffset.template ReinterpretCast<int32_t>();
        LocalTensor<int32_t> oddI32 = oddOffset.template ReinterpretCast<int32_t>();
        const uint32_t channels = static_cast<uint32_t>(tiling_->c);
        const uint32_t invalidIndex = static_cast<uint32_t>(tiling_->inW) * channels;
        const int32_t invalidOffset = static_cast<int32_t>(invalidIndex * sizeof(T));
        for (uint32_t ow = 0; ow < outW; ++ow) {
            const uint32_t outBase = ow * block;
            const uint32_t evenW = ow * 2U;
            const uint32_t oddW = evenW + 1U;
            for (uint32_t c0 = 0; c0 < block; ++c0) {
                int32_t even = invalidOffset;
                int32_t odd = invalidOffset;
                if (c0 < cCount) {
                    even = static_cast<int32_t>((evenW * channels + static_cast<uint32_t>(cBase) + c0) * sizeof(T));
                    if (oddW < static_cast<uint32_t>(tiling_->inW)) {
                        odd = static_cast<int32_t>((oddW * channels + static_cast<uint32_t>(cBase) + c0) * sizeof(T));
                    }
                }
                evenI32.SetValue(outBase + c0, even);
                oddI32.SetValue(outBase + c0, odd);
            }
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeNdc1hwc0NdhwcStride2Pool2Row(LocalTensor<T> rowLocal, int64_t nIdx, int64_t od,
                                                               int64_t cBase, int64_t oh, uint32_t cCount,
                                                               uint32_t block, uint32_t outW)
    {
        const uint32_t rowElements = outW * block;
        const uint32_t validInputCount = static_cast<uint32_t>(tiling_->inW) * static_cast<uint32_t>(tiling_->c);
        const uint32_t alignedInputCount = AlignToVector(validInputCount + 1U);
        const uint32_t offsetElements = Ndc1hwc0GatherTempOffset(rowElements);

        LocalTensor<T> evenLocal = calcBuf_.Get<T>();
        LocalTensor<T> scratchLocal = tmpBuf_.Get<T>();
        LocalTensor<uint32_t> evenOffset = scratchLocal.template ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> oddOffset = scratchLocal[offsetElements].template ReinterpretCast<uint32_t>();
        LocalTensor<T> oddLocal = scratchLocal[offsetElements * 2U];
        InitNdc1hwc0NdhwcStride2GatherOffsets(evenOffset, oddOffset, cBase, cCount, block, outW);
        Duplicate(rowLocal, NegInfValue(), rowElements);
        PipeBarrier<PIPE_V>();

        for (int64_t kd = 0; kd < 2; ++kd) {
            const int64_t id = Pool2InputD(od, kd);
            if (!IsOutOfRange(id, tiling_->inD)) {
                for (int64_t kh = 0; kh < 2; ++kh) {
                    const int64_t ih = Pool2InputH(oh, kh);
                    if (IsOutOfRange(ih, tiling_->inH)) {
                        continue;
                    }
                    CopyInVectorPadValue(InputOffset(nIdx, id, ih, 0, 0), validInputCount, alignedInputCount,
                                         NegInfValue());
                    LocalTensor<T> xLocal = xInQue_.DeQue<T>();
                    Gather(evenLocal, xLocal, evenOffset, static_cast<uint32_t>(0), rowElements);
                    Gather(oddLocal, xLocal, oddOffset, static_cast<uint32_t>(0), rowElements);
                    PipeBarrier<PIPE_V>();
                    Max(evenLocal, evenLocal, oddLocal, rowElements);
                    PipeBarrier<PIPE_V>();
                    Max(rowLocal, rowLocal, evenLocal, rowElements);
                    PipeBarrier<PIPE_V>();
                    xInQue_.FreeTensor(xLocal);
                }
            }
        }
        ZeroNdc1hwc0RowTail(rowLocal, cCount, block, outW);
    }

    __aicore__ inline void ReduceNdc1hwc0NcdhwRowWindow(LocalTensor<T> accLocal, int64_t nIdx, int64_t od,
                                                        int64_t cBase, int64_t oh, uint32_t cCount, uint32_t outW,
                                                        uint32_t alignedW)
    {
        for (int64_t kd = 0; kd < tiling_->kD; ++kd) {
            ReduceNdc1hwc0NcdhwRowDepth(accLocal, nIdx, od, kd, cBase, oh, cCount, outW, alignedW);
        }
    }

    __aicore__ inline void ReduceNdc1hwc0NcdhwRowDepth(LocalTensor<T> accLocal, int64_t nIdx, int64_t od, int64_t kd,
                                                       int64_t cBase, int64_t oh, uint32_t cCount, uint32_t outW,
                                                       uint32_t alignedW)
    {
        const int64_t id = DilatedInputD(od, kd);
        if (IsOutOfRange(id, tiling_->inD)) {
            return;
        }
        for (int64_t kh = 0; kh < tiling_->kH; ++kh) {
            const int64_t ih = DilatedInputH(oh, kh);
            if (IsOutOfRange(ih, tiling_->inH)) {
                continue;
            }
            for (int64_t kw = 0; kw < tiling_->kW; ++kw) {
                uint32_t wStart = 0U;
                uint32_t wCount = 0U;
                if (!CalcValidWRange(kw, wStart, wCount)) {
                    continue;
                }
                const int64_t iw = DilatedInputWFromStart(wStart, kw);
                if (tiling_->sW == 1) {
                    ReduceNdc1hwc0NcdhwRowWRange(nIdx, id, ih, iw, cBase, cCount, outW, alignedW, wStart, wCount,
                                                 accLocal);
                } else {
                    ReduceNdc1hwc0NcdhwRowWRangeStrided(nIdx, id, ih, iw, cBase, cCount, alignedW, wStart, wCount,
                                                        accLocal);
                }
            }
        }
    }

    __aicore__ inline void ScatterNdc1hwc0NcdhwRowScalar(LocalTensor<T> rowLocal, LocalTensor<T> accLocal,
                                                         uint32_t cCount, uint32_t block, uint32_t outW,
                                                         uint32_t alignedW)
    {
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        for (uint32_t c0 = 0; c0 < cCount; ++c0) {
            const uint64_t srcBase = static_cast<uint64_t>(c0) * alignedW;
            for (uint32_t ow = 0; ow < outW; ++ow) {
                rowLocal.SetValue(static_cast<uint64_t>(ow) * block + c0, accLocal.GetValue(srcBase + ow));
            }
        }
    }

    __aicore__ inline void ComputeNdc1hwc0NcdhwRowVector(LocalTensor<T> rowLocal, int64_t nIdx, int64_t od,
                                                         int64_t cBase, int64_t oh, uint32_t cCount, uint32_t block)
    {
        const uint32_t outW = static_cast<uint32_t>(tiling_->outW);
        if (CanUseNdc1hwc0NcdhwStride2Pool2Row(cCount, block)) {
            ComputeNdc1hwc0NcdhwStride2Pool2Row(rowLocal, nIdx, od, cBase, oh, cCount, block, outW);
            return;
        }
        const uint32_t alignedW = AlignToVector(outW);
        LocalTensor<T> accLocal = calcBuf_.Get<T>();
        Duplicate(accLocal, NegInfValue(), cCount * alignedW);
        PipeBarrier<PIPE_V>();
        ReduceNdc1hwc0NcdhwRowWindow(accLocal, nIdx, od, cBase, oh, cCount, outW, alignedW);
        LocalTensor<T> scratchLocal = tmpBuf_.Get<T>();
        if (ScatterNdc1hwc0NcdhwRowGather(rowLocal, accLocal, scratchLocal, cCount, block, outW, alignedW)) {
            return;
        }
        ScatterNdc1hwc0NcdhwRowScalar(rowLocal, accLocal, cCount, block, outW, alignedW);
    }

    __aicore__ inline bool CanUseNdc1hwc0NcdhwStride2Pool2Row(uint32_t cCount, uint32_t block) const
    {
        if (tiling_->dataFormat != FORMAT_NCDHW_VALUE || cCount == 0U || block == 0U || cCount > block ||
            tiling_->outW <= 0 || tiling_->inW <= 0) {
            return false;
        }
        if (!MatchesPoolSpec(2, 2, 2, 2, 2, 2, 1, 1, 1, 0, 0, 0)) {
            return false;
        }
        const uint32_t outW = static_cast<uint32_t>(tiling_->outW);
        const uint32_t inputNeedW = outW * 2U;
        const uint32_t alignedInputW = AlignToVector(inputNeedW);
        const uint32_t alignedW = AlignToVector(outW);
        const uint32_t rowElements = outW * block;
        const uint32_t compactCount = cCount * alignedW;
        const uint32_t channelOffset = AlignToVector(compactCount);
        const uint32_t oddOffset = channelOffset + Ndc1hwc0GatherTempOffset(compactCount);
        const uint32_t compactOffset = oddOffset + compactCount;
        const uint32_t scratchNeed = compactOffset + compactCount;
        return rowElements <= OUTPUT_TILE_NUM && compactCount + 1U <= OUTPUT_TILE_NUM &&
               compactCount <= OUTPUT_TILE_NUM && scratchNeed <= OUTPUT_TILE_NUM;
    }

    __aicore__ inline void ReduceNdc1hwc0NcdhwStride2Pool2Row(LocalTensor<T> accLocal, LocalTensor<T> compactLocal,
                                                              LocalTensor<T> oddLocal,
                                                              LocalTensor<uint32_t> offsetLocal, int64_t nIdx,
                                                              int64_t od, int64_t cBase, int64_t oh, uint32_t cCount,
                                                              uint32_t validInputW, uint32_t alignedInputW,
                                                              uint32_t compactCount, uint32_t srcStrideElements)
    {
        for (int64_t kd = 0; kd < 2; ++kd) {
            const int64_t id = Pool2InputD(od, kd);
            if (IsOutOfRange(id, tiling_->inD)) {
                continue;
            }
            for (int64_t kh = 0; kh < 2; ++kh) {
                const int64_t ih = Pool2InputH(oh, kh);
                if (IsOutOfRange(ih, tiling_->inH)) {
                    continue;
                }
                CopyInVectorWBlocksPadStride(InputOffset(nIdx, id, ih, 0, cBase), cCount, validInputW, alignedInputW,
                                             srcStrideElements);
                LocalTensor<T> xLocal = xInQue_.DeQue<T>();
                GatherNdc1hwc0NcdhwStride2ChannelPairs(compactLocal, oddLocal, xLocal, offsetLocal, compactCount);
                PipeBarrier<PIPE_V>();
                Max(compactLocal, compactLocal, oddLocal, compactCount);
                PipeBarrier<PIPE_V>();
                Max(accLocal, accLocal, compactLocal, compactCount);
                PipeBarrier<PIPE_V>();
                xInQue_.FreeTensor(xLocal);
            }
        }
    }

    __aicore__ inline void ComputeNdc1hwc0NcdhwStride2Pool2Row(LocalTensor<T> rowLocal, int64_t nIdx, int64_t od,
                                                               int64_t cBase, int64_t oh, uint32_t cCount,
                                                               uint32_t block, uint32_t outW)
    {
        const uint32_t inputNeedW = outW * 2U;
        const uint32_t validInputW = static_cast<uint32_t>(
            static_cast<int64_t>(inputNeedW) < tiling_->inW ? inputNeedW : tiling_->inW);
        const uint32_t alignedInputW = AlignToVector(inputNeedW);
        const uint32_t alignedW = AlignToVector(outW);
        const uint32_t compactCount = cCount * alignedW;
        const uint64_t channelStride = static_cast<uint64_t>(tiling_->inD) * static_cast<uint64_t>(tiling_->inH) *
                                       static_cast<uint64_t>(tiling_->inW);
        const uint32_t srcStrideElements = static_cast<uint32_t>(channelStride - static_cast<uint64_t>(validInputW));

        LocalTensor<T> accLocal = calcBuf_.Get<T>();
        LocalTensor<T> tmpLocal = tmpBuf_.Get<T>();
        const uint32_t channelOffset = AlignToVector(compactCount);
        const uint32_t oddOffset = channelOffset + Ndc1hwc0GatherTempOffset(compactCount);
        const uint32_t compactOffset = oddOffset + compactCount;
        LocalTensor<uint32_t> offsetLocal = tmpLocal[channelOffset].template ReinterpretCast<uint32_t>();
        LocalTensor<T> oddLocal = tmpLocal[oddOffset];
        LocalTensor<T> compactLocal = tmpLocal[compactOffset];
        InitNdc1hwc0NcdhwStride2ChannelGatherOffsets(offsetLocal, cCount, outW, alignedW, alignedInputW);
        Duplicate(accLocal, NegInfValue(), compactCount);
        PipeBarrier<PIPE_V>();

        ReduceNdc1hwc0NcdhwStride2Pool2Row(accLocal, compactLocal, oddLocal, offsetLocal, nIdx, od, cBase, oh, cCount,
                                           validInputW, alignedInputW, compactCount, srcStrideElements);
        if (!ScatterNdc1hwc0NcdhwRowGather(rowLocal, accLocal, tmpLocal, cCount, block, outW, alignedW)) {
            ScatterNdc1hwc0NcdhwRowScalar(rowLocal, accLocal, cCount, block, outW, alignedW);
        }
    }

    __aicore__ inline void InitNdc1hwc0NcdhwStride2ChannelGatherOffsets(LocalTensor<uint32_t> offsetLocal,
                                                                        uint32_t cCount, uint32_t outW,
                                                                        uint32_t alignedW, uint32_t alignedInputW)
    {
        LocalTensor<int32_t> offsetI32 = offsetLocal.template ReinterpretCast<int32_t>();
        for (uint32_t c0 = 0; c0 < cCount; ++c0) {
            const uint32_t dstBase = c0 * alignedW;
            const uint32_t srcBase = c0 * alignedInputW;
            for (uint32_t ow = 0; ow < alignedW; ++ow) {
                uint32_t srcIndex = srcBase;
                if (ow < outW) {
                    srcIndex = srcBase + ow * 2U;
                }
                offsetI32.SetValue(dstBase + ow, static_cast<int32_t>(srcIndex * sizeof(T)));
            }
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void GatherNdc1hwc0NcdhwStride2ChannelPairs(LocalTensor<T> evenLocal, LocalTensor<T> oddLocal,
                                                                  LocalTensor<T> xLocal,
                                                                  LocalTensor<uint32_t> offsetLocal, uint32_t count)
    {
        uint32_t done = 0U;
        while (done < count) {
            const uint32_t curCount = Ndc1hwc0SafeGatherChunk(count - done);
            Gather(evenLocal[done], xLocal, offsetLocal[done], static_cast<uint32_t>(0), curCount);
            Gather(oddLocal[done], xLocal, offsetLocal[done], static_cast<uint32_t>(sizeof(T)), curCount);
            PipeBarrier<PIPE_V>();
            done += curCount;
        }
    }

    __aicore__ inline bool CalcValidWRange(int64_t kw, uint32_t& wStart, uint32_t& wCount) const
    {
        if (tiling_->outW <= 0 || tiling_->sW <= 0) {
            return false;
        }
        int64_t first = 0;
        while (first < tiling_->outW && first * tiling_->sW + kw * tiling_->dilationW - tiling_->padLeft < 0) {
            ++first;
        }
        int64_t last = tiling_->outW - 1;
        while (last >= first && last * tiling_->sW + kw * tiling_->dilationW - tiling_->padLeft >= tiling_->inW) {
            --last;
        }
        if (last < first) {
            return false;
        }
        wStart = static_cast<uint32_t>(first);
        wCount = static_cast<uint32_t>(last - first + 1);
        return true;
    }

    __aicore__ inline void ReduceNdc1hwc0NcdhwRowWRange(int64_t nIdx, int64_t id, int64_t ih, int64_t iw, int64_t cBase,
                                                        uint32_t cCount, uint32_t outW, uint32_t alignedW,
                                                        uint32_t wStart, uint32_t wCount, LocalTensor<T> accLocal)
    {
        const uint32_t alignedCopyW = AlignToVector(wCount);
        const uint64_t channelStride = static_cast<uint64_t>(tiling_->inD) * static_cast<uint64_t>(tiling_->inH) *
                                       static_cast<uint64_t>(tiling_->inW);
        const uint32_t srcStrideElements = static_cast<uint32_t>(channelStride - static_cast<uint64_t>(wCount));
        CopyInVectorWBlocksPadStride(InputOffset(nIdx, id, ih, iw, cBase), cCount, wCount, alignedCopyW,
                                     srcStrideElements);
        LocalTensor<T> xLocal = xInQue_.DeQue<T>();
        if (wStart == 0U && wCount == outW && alignedCopyW == alignedW) {
            Max(accLocal, accLocal, xLocal, cCount * alignedW);
        } else {
            for (uint32_t c0 = 0; c0 < cCount; ++c0) {
                Max(accLocal[static_cast<uint64_t>(c0) * alignedW + wStart],
                    accLocal[static_cast<uint64_t>(c0) * alignedW + wStart],
                    xLocal[static_cast<uint64_t>(c0) * alignedCopyW], wCount);
            }
        }
        PipeBarrier<PIPE_V>();
        xInQue_.FreeTensor(xLocal);
    }

    __aicore__ inline void ReduceNdc1hwc0NcdhwRowWRangeStrided(int64_t nIdx, int64_t id, int64_t ih, int64_t iw,
                                                               int64_t cBase, uint32_t cCount, uint32_t alignedW,
                                                               uint32_t wStart, uint32_t wCount,
                                                               LocalTensor<T> accLocal)
    {
        const uint32_t spanW = (wCount - 1U) * static_cast<uint32_t>(tiling_->sW) + 1U;
        const uint32_t alignedSpanW = AlignToVector(spanW);
        const uint64_t channelStride = static_cast<uint64_t>(tiling_->inD) * static_cast<uint64_t>(tiling_->inH) *
                                       static_cast<uint64_t>(tiling_->inW);
        const uint32_t srcStrideElements = static_cast<uint32_t>(channelStride - static_cast<uint64_t>(spanW));
        CopyInVectorWBlocksPadStride(InputOffset(nIdx, id, ih, iw, cBase), cCount, spanW, alignedSpanW,
                                     srcStrideElements);
        LocalTensor<T> xLocal = xInQue_.DeQue<T>();
        LocalTensor<T> scratchLocal = tmpBuf_.Get<T>();
        LocalTensor<uint32_t> offsetLocal = scratchLocal.template ReinterpretCast<uint32_t>();
        InitNdc1hwc0NcdhwStridedWOffsets(offsetLocal, wCount);
        const uint32_t gatheredOffset = Ndc1hwc0GatherTempOffset(wCount);
        LocalTensor<T> gatheredLocal = scratchLocal[gatheredOffset];
        for (uint32_t c0 = 0; c0 < cCount; ++c0) {
            Gather(gatheredLocal, xLocal[static_cast<uint64_t>(c0) * alignedSpanW], offsetLocal,
                   static_cast<uint32_t>(0), wCount);
            PipeBarrier<PIPE_V>();
            Max(accLocal[static_cast<uint64_t>(c0) * alignedW + wStart],
                accLocal[static_cast<uint64_t>(c0) * alignedW + wStart], gatheredLocal, wCount);
            PipeBarrier<PIPE_V>();
        }
        xInQue_.FreeTensor(xLocal);
    }

    __aicore__ inline uint32_t Ndc1hwc0GatherTempOffset(uint32_t count) const
    {
        const uint32_t offsetBytes = count * sizeof(uint32_t);
        const uint32_t offsetElements = static_cast<uint32_t>((offsetBytes + sizeof(T) - 1U) / sizeof(T));
        return AlignToVector(offsetElements);
    }

    __aicore__ inline void InitNdc1hwc0NcdhwStridedWOffsets(LocalTensor<uint32_t> offsetLocal, uint32_t wCount)
    {
        LocalTensor<int32_t> offsetI32 = offsetLocal.template ReinterpretCast<int32_t>();
        ArithProgression(offsetI32, static_cast<int32_t>(0),
                         static_cast<int32_t>(static_cast<uint32_t>(tiling_->sW) * sizeof(T)), wCount);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline bool CanUseNdc1hwc0NcdhwRowGather(uint32_t cCount, uint32_t block, uint32_t outW,
                                                        uint32_t alignedW) const
    {
        if (cCount == 0U || block == 0U || outW == 0U || alignedW == 0U || cCount > block) {
            return false;
        }
        const uint32_t rowElements = outW * block;
        const uint32_t inputElements = cCount * alignedW;
        if (inputElements + 1U > OUTPUT_TILE_NUM) {
            return false;
        }
        return rowElements <= OUTPUT_TILE_NUM &&
               static_cast<uint64_t>(CanUseNdc1hwc0WideGather(rowElements) ? rowElements : 255U) * sizeof(uint32_t) <=
                   static_cast<uint64_t>(OUTPUT_TILE_NUM) * sizeof(T);
    }

    __aicore__ inline void InitNdc1hwc0NcdhwRowGatherOffsets(LocalTensor<uint32_t> offsetLocal, uint32_t cCount,
                                                             uint32_t block, uint32_t outW, uint32_t alignedW,
                                                             uint32_t zeroIndex)
    {
        LocalTensor<int32_t> offsetI32 = offsetLocal.template ReinterpretCast<int32_t>();
        const int32_t zeroOffset = static_cast<int32_t>(zeroIndex * sizeof(T));
        for (uint32_t ow = 0; ow < outW; ++ow) {
            const uint32_t outBase = ow * block;
            for (uint32_t c0 = 0; c0 < block; ++c0) {
                int32_t srcOffset = zeroOffset;
                if (c0 < cCount) {
                    srcOffset = static_cast<int32_t>((c0 * alignedW + ow) * sizeof(T));
                }
                offsetI32.SetValue(outBase + c0, srcOffset);
            }
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void InitNdc1hwc0NcdhwRowGatherOffsetsChunk(LocalTensor<uint32_t> offsetLocal, uint32_t start,
                                                                  uint32_t count, uint32_t cCount, uint32_t block,
                                                                  uint32_t alignedW, uint32_t zeroIndex)
    {
        LocalTensor<int32_t> offsetI32 = offsetLocal.template ReinterpretCast<int32_t>();
        const int32_t zeroOffset = static_cast<int32_t>(zeroIndex * sizeof(T));
        if (block == 0U) {
            return;
        }
        for (uint32_t i = 0; i < count; ++i) {
            const GatherOffsetIndex index = DecodeGatherOffsetIndex(start, i, block);
            int32_t srcOffset = zeroOffset;
            if (index.c0 < cCount) {
                srcOffset = static_cast<int32_t>((index.c0 * alignedW + index.ow) * sizeof(T));
            }
            offsetI32.SetValue(i, srcOffset);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline bool ScatterNdc1hwc0NcdhwRowGather(LocalTensor<T> rowLocal, LocalTensor<T> resultLocal,
                                                         LocalTensor<T> scratchLocal, uint32_t cCount, uint32_t block,
                                                         uint32_t outW, uint32_t alignedW)
    {
        if (!CanUseNdc1hwc0NcdhwRowGather(cCount, block, outW, alignedW)) {
            return false;
        }
        const uint32_t rowElements = outW * block;
        const uint32_t zeroIndex = cCount * alignedW;
        Duplicate(resultLocal[zeroIndex], ZeroValue(), 1);
        PipeBarrier<PIPE_V>();
        LocalTensor<uint32_t> offsetLocal = scratchLocal.template ReinterpretCast<uint32_t>();
        if (CanUseNdc1hwc0WideGather(rowElements)) {
            InitNdc1hwc0NcdhwRowGatherOffsets(offsetLocal, cCount, block, outW, alignedW, zeroIndex);
            Gather(rowLocal, resultLocal, offsetLocal, static_cast<uint32_t>(0), rowElements);
            PipeBarrier<PIPE_V>();
            return true;
        }
        uint32_t done = 0U;
        while (done < rowElements) {
            const uint32_t curCount = Ndc1hwc0SafeGatherChunk(rowElements - done);
            InitNdc1hwc0NcdhwRowGatherOffsetsChunk(offsetLocal, done, curCount, cCount, block, alignedW, zeroIndex);
            Gather(rowLocal[done], resultLocal, offsetLocal, static_cast<uint32_t>(0), curCount);
            PipeBarrier<PIPE_V>();
            done += curCount;
        }
        return true;
    }

    __aicore__ inline void ProcessNdc1hwc0RowVectorByRow(uint64_t row, uint64_t outputOffset, uint64_t block,
                                                         uint64_t validC1)
    {
        int64_t nIdx = 0;
        int64_t od = 0;
        int64_t c1Idx = 0;
        int64_t oh = 0;
        DecodeNdc1hwc0Row(row, validC1, nIdx, od, c1Idx, oh);
        const int64_t cBase = c1Idx * static_cast<int64_t>(block);
        const int64_t activeChannels = Ndc1hwc0ActiveChannels(cBase, block);

        const uint32_t rowElements = static_cast<uint32_t>(static_cast<uint64_t>(tiling_->outW) * block);
        LocalTensor<T> rowLocal = maskBuf_.Get<T>();
        Duplicate(rowLocal, ZeroValue(), rowElements);
        PipeBarrier<PIPE_V>();
        if (activeChannels > 0) {
            if (tiling_->dataFormat == FORMAT_NDHWC_VALUE) {
                ComputeNdc1hwc0NdhwcRowVector(rowLocal, nIdx, od, cBase, oh, static_cast<uint32_t>(activeChannels),
                                              static_cast<uint32_t>(block));
            } else {
                ComputeNdc1hwc0NcdhwRowVector(rowLocal, nIdx, od, cBase, oh, static_cast<uint32_t>(activeChannels),
                                              static_cast<uint32_t>(block));
            }
        }
        CopyOutVector(outputOffset, rowLocal, rowElements);
    }

    __aicore__ inline void FillNdc1hwc0RowVectorTile(uint64_t startRow, uint32_t rowCount, uint64_t block,
                                                     uint64_t validC1, LocalTensor<T> yLocal)
    {
        const uint32_t rowElements = static_cast<uint32_t>(static_cast<uint64_t>(tiling_->outW) * block);
        Duplicate(yLocal, ZeroValue(), rowElements * rowCount);
        PipeBarrier<PIPE_V>();
        for (uint32_t rowIdx = 0; rowIdx < rowCount; ++rowIdx) {
            const Ndc1hwc0DecodedRow context = DecodeNdc1hwc0RowContext(startRow + rowIdx, validC1);
            const int64_t cBase = context.c1Idx * static_cast<int64_t>(block);
            const int64_t activeChannels = Ndc1hwc0ActiveChannels(cBase, block);
            if (activeChannels <= 0) {
                continue;
            }
            LocalTensor<T> rowLocal = yLocal[static_cast<uint64_t>(rowIdx) * rowElements];
            if (tiling_->dataFormat == FORMAT_NDHWC_VALUE) {
                ComputeNdc1hwc0NdhwcRowVector(rowLocal, context.nIdx, context.od, cBase, context.oh,
                                              static_cast<uint32_t>(activeChannels), static_cast<uint32_t>(block));
            } else {
                ComputeNdc1hwc0NcdhwRowVector(rowLocal, context.nIdx, context.od, cBase, context.oh,
                                              static_cast<uint32_t>(activeChannels), static_cast<uint32_t>(block));
            }
        }
    }

    __aicore__ inline void ProcessNdc1hwc0RowVectorTile(uint64_t startRow, uint32_t rowCount, uint64_t outputOffset,
                                                        uint64_t block, uint64_t validC1)
    {
        const uint32_t rowElements = static_cast<uint32_t>(static_cast<uint64_t>(tiling_->outW) * block);
        LocalTensor<T> yLocal = maskBuf_.Get<T>();
        FillNdc1hwc0RowVectorTile(startRow, rowCount, block, validC1, yLocal);
        CopyOutVector(outputOffset, yLocal, rowElements * rowCount);
    }

    __aicore__ inline bool CanUseNdc1hwc0NcdhwD3W3DilD2GroupPath() const
    {
        if (!HasNdc1hwc0NcdhwD3W3Shape() || !HasNdc1hwc0NcdhwD3W3PoolSpec()) {
            return false;
        }
        const uint64_t block = Ndc1hwc0Block();
        const uint64_t validC1 = Ndc1hwc0ValidC1(block);
        if (block == 0U || validC1 != 1U || static_cast<uint64_t>(tiling_->c) > block ||
            !IsNdc1hwc0CompactStorage(block, validC1)) {
            return false;
        }
        const uint32_t outW = static_cast<uint32_t>(tiling_->outW);
        const uint32_t alignedW = AlignToVector(outW);
        const uint32_t alignedInputW = AlignToVector(static_cast<uint32_t>(tiling_->inW));
        const uint32_t cCount = static_cast<uint32_t>(tiling_->c);
        const uint32_t rowElements = outW * static_cast<uint32_t>(block);
        const uint32_t rowStride = AlignToVector(cCount * alignedW + 1U);
        const uint32_t compactCount = cCount * alignedW;
        const uint32_t offsetNeed = Ndc1hwc0GatherTempOffset(compactCount);
        return outW > 0U && alignedW > 0U && alignedInputW >= static_cast<uint32_t>(tiling_->inW) && rowElements > 0U &&
               rowElements <= OUTPUT_TILE_NUM && rowStride <= OUTPUT_TILE_NUM && compactCount > 0U &&
               compactCount <= OUTPUT_TILE_NUM && offsetNeed + compactCount <= OUTPUT_TILE_NUM &&
               cCount * alignedInputW <= INPUT_TILE_NUM && Ndc1hwc0GatherTempOffset(rowElements) <= OUTPUT_TILE_NUM;
    }

    __aicore__ inline bool HasNdc1hwc0NcdhwD3W3Shape() const
    {
        return tiling_->dataFormat == FORMAT_NCDHW_VALUE && tiling_->outputLayout == OUTPUT_LAYOUT_NDC1HWC0_VALUE &&
               tiling_->n > 0 && tiling_->c > 0 && tiling_->inD > 0 && tiling_->inH > 0 && tiling_->inW > 0 &&
               tiling_->outD == 1 && tiling_->outH == tiling_->inH && tiling_->outW > 0 &&
               (tiling_->outW - 1) * tiling_->sW + (tiling_->kW - 1) * tiling_->dilationW < tiling_->inW;
    }

    __aicore__ inline bool HasNdc1hwc0NcdhwD3W3PoolSpec() const
    {
        return tiling_->kD == 3 && tiling_->kH == 1 && tiling_->kW == 3 && tiling_->sD == 3 && tiling_->sH == 1 &&
               tiling_->sW == 3 && tiling_->dilationD == 2 && tiling_->dilationH == 1 && tiling_->dilationW == 1 &&
               tiling_->padFront == 0 && tiling_->padTop == 0 && tiling_->padLeft == 0;
    }

    __aicore__ inline bool ProcessNdc1hwc0NcdhwD3W3DilD2GroupStep(uint64_t& cur, uint64_t outEnd, uint64_t block,
                                                                  uint64_t validC1, uint32_t outW, uint32_t rowElements)
    {
        if (rowElements == 0U) {
            cur = outEnd;
            return false;
        }
        const uint64_t rowOffset = cur % rowElements;
        if (rowOffset != 0U || outEnd - cur < rowElements) {
            ProcessNdc1hwc0RowVectorByRow(cur / rowElements, cur, block, validC1);
            cur += rowElements - rowOffset;
            return true;
        }
        const uint64_t row = cur / rowElements;
        int64_t nIdx = 0;
        int64_t od = 0;
        int64_t c1Idx = 0;
        int64_t oh = 0;
        DecodeNdc1hwc0Row(row, validC1, nIdx, od, c1Idx, oh);
        if (c1Idx != 0 || od != 0) {
            ProcessNdc1hwc0RowVectorByRow(row, cur, block, validC1);
            cur += rowElements;
            return true;
        }
        const uint32_t cCount = static_cast<uint32_t>(tiling_->c);
        const uint32_t rowStride = AlignToVector(cCount * AlignToVector(outW) + 1U);
        uint32_t rows = static_cast<uint32_t>(tiling_->outH - oh);
        const uint64_t remainRows = (outEnd - cur) / rowElements;
        if (static_cast<uint64_t>(rows) > remainRows) {
            rows = static_cast<uint32_t>(remainRows);
        }
        const uint32_t maxRowsByOut = OUTPUT_TILE_NUM / rowElements;
        const uint32_t maxRowsByCompact = rowStride == 0U ? 1U : OUTPUT_TILE_NUM / rowStride;
        uint32_t maxRows = maxRowsByOut < maxRowsByCompact ? maxRowsByOut : maxRowsByCompact;
        if (maxRows == 0U) {
            maxRows = 1U;
        }
        if (rows > maxRows) {
            rows = maxRows;
        }
        if (rows == 0U) {
            return false;
        }
        ProcessNdc1hwc0NcdhwD3W3DilD2GroupTile(cur, nIdx, static_cast<uint32_t>(oh), rows, cCount,
                                               static_cast<uint32_t>(block), outW);
        cur += static_cast<uint64_t>(rows) * rowElements;
        return true;
    }

    __aicore__ inline void ProcessNdc1hwc0NcdhwD3W3DilD2Group()
    {
        uint64_t block = 0U, validC1 = 0U, validOut = 0U;
        if (!InitNdc1hwc0ValidOutput(block, validC1, validOut)) {
            return;
        }
        const uint32_t outW = static_cast<uint32_t>(tiling_->outW);
        const uint32_t rowElements = outW * static_cast<uint32_t>(block);
        uint64_t outOffset = 0U, outCount = 0U, outEnd = 0U;
        GetNdc1hwc0ValidCoreRange(validOut, rowElements, outOffset, outCount, outEnd);
        uint64_t cur = outOffset;
        while (cur < outEnd) {
            if (!ProcessNdc1hwc0NcdhwD3W3DilD2GroupStep(cur, outEnd, block, validC1, outW, rowElements)) {
                break;
            }
        }
        if (validOut < tiling_->totalOut) {
            CopyOutZeroRangeByCore(validOut, tiling_->totalOut - validOut);
        }
    }

    __aicore__ inline void ReduceNdc1hwc0NcdhwD3W3Rows(LocalTensor<T> compactLocal, LocalTensor<T> gatheredLocal,
                                                       LocalTensor<uint32_t> gatherOffsetLocal, int64_t nIdx,
                                                       uint32_t ohStart, uint32_t rows, uint32_t cCount,
                                                       uint32_t alignedInputW, uint32_t compactCount,
                                                       uint32_t rowStride, uint32_t srcStrideElements)
    {
        for (uint32_t rowIdx = 0U; rowIdx < rows; ++rowIdx) {
            LocalTensor<T> accRow = compactLocal[static_cast<uint64_t>(rowIdx) * rowStride];
            const int64_t ih = static_cast<int64_t>(ohStart + rowIdx);
            for (int64_t kd = 0; kd < 3; ++kd) {
                const int64_t id = kd * 2;
                CopyInVectorWBlocksPadStrideValue(InputOffset(nIdx, id, ih, 0, 0), cCount,
                                                  static_cast<uint32_t>(tiling_->inW), alignedInputW, srcStrideElements,
                                                  NegInfValue());
                LocalTensor<T> xLocal = xInQue_.DeQue<T>();
                for (uint32_t kw = 0U; kw < 3U; ++kw) {
                    Gather(gatheredLocal, xLocal, gatherOffsetLocal, static_cast<uint32_t>(kw * sizeof(T)),
                           compactCount);
                    PipeBarrier<PIPE_V>();
                    Max(accRow, accRow, gatheredLocal, compactCount);
                    PipeBarrier<PIPE_V>();
                }
                xInQue_.FreeTensor(xLocal);
            }
        }
    }

    __aicore__ inline void ScatterNdc1hwc0NcdhwD3W3Rows(LocalTensor<T> outLocal, LocalTensor<T> compactLocal,
                                                        LocalTensor<T> scratchLocal, uint32_t rows, uint32_t rowStride,
                                                        uint32_t cCount, uint32_t block, uint32_t outW,
                                                        uint32_t alignedW, uint32_t compactCount)
    {
        LocalTensor<uint32_t> scatterOffsetLocal = scratchLocal.template ReinterpretCast<uint32_t>();
        if (!ScatterNdc1hwc0CompactRowsReuseRowOffset(outLocal, compactLocal, scatterOffsetLocal, rows, rowStride,
                                                      cCount, block, outW, 1U, alignedW, compactCount)) {
            ScatterNdc1hwc0CompactRowsChecked(outLocal, compactLocal, scatterOffsetLocal, rows, rowStride, cCount,
                                              block, outW, 1U, alignedW, compactCount);
        }
    }

    __aicore__ inline void ProcessNdc1hwc0NcdhwD3W3DilD2GroupTile(uint64_t outputOffset, int64_t nIdx, uint32_t ohStart,
                                                                  uint32_t rows, uint32_t cCount, uint32_t block,
                                                                  uint32_t outW)
    {
        const uint32_t alignedW = AlignToVector(outW);
        const uint32_t alignedInputW = AlignToVector(static_cast<uint32_t>(tiling_->inW));
        const uint32_t compactCount = cCount * alignedW;
        const uint32_t rowStride = AlignToVector(compactCount + 1U);
        const uint32_t rowElements = outW * block;
        const uint32_t offsetNeed = Ndc1hwc0GatherTempOffset(compactCount);
        if (ProcessNdc1hwc0NcdhwD3W3DilD2CompactTile(outputOffset, nIdx, ohStart, rows, cCount, block, outW)) {
            return;
        }
        if (ProcessNdc1hwc0NcdhwD3W3DilD2SlabTile(outputOffset, nIdx, ohStart, rows, cCount, block, outW)) {
            return;
        }
        LocalTensor<T> compactLocal = calcBuf_.Get<T>();
        LocalTensor<T> outLocal = maskBuf_.Get<T>();
        LocalTensor<T> scratchLocal = tmpBuf_.Get<T>();
        LocalTensor<uint32_t> gatherOffsetLocal = scratchLocal.template ReinterpretCast<uint32_t>();
        LocalTensor<T> gatheredLocal = scratchLocal[offsetNeed];
        InitNdc1hwc0NcdhwD3W3DilD2GatherOffsets(gatherOffsetLocal, cCount, outW, alignedW, alignedInputW);
        Duplicate(compactLocal, NegInfValue(), rows * rowStride);
        PipeBarrier<PIPE_V>();

        const uint64_t channelStride = static_cast<uint64_t>(tiling_->inD) * static_cast<uint64_t>(tiling_->inH) *
                                       static_cast<uint64_t>(tiling_->inW);
        const uint32_t srcStrideElements = static_cast<uint32_t>(channelStride - static_cast<uint64_t>(tiling_->inW));
        ReduceNdc1hwc0NcdhwD3W3Rows(compactLocal, gatheredLocal, gatherOffsetLocal, nIdx, ohStart, rows, cCount,
                                    alignedInputW, compactCount, rowStride, srcStrideElements);

        ScatterNdc1hwc0NcdhwD3W3Rows(outLocal, compactLocal, scratchLocal, rows, rowStride, cCount, block, outW,
                                     alignedW, compactCount);
        CopyOutVector(outputOffset, outLocal, rows * rowElements);
    }

    __aicore__ inline void ReduceNdc1hwc0NcdhwD3W3Slab(LocalTensor<T> compactLocal, LocalTensor<T> gatheredLocal,
                                                       LocalTensor<uint32_t> gatherOffsetLocal, int64_t nIdx,
                                                       uint32_t ohStart, uint32_t rows, uint32_t cCount, uint32_t inW,
                                                       uint32_t slabW, uint32_t alignedSlabW, uint32_t compactCount,
                                                       uint32_t rowStride, uint32_t srcStrideElements)
    {
        for (int64_t kd = 0; kd < 3; ++kd) {
            const int64_t id = kd * 2;
            CopyInVectorWBlocksPadStrideValue(InputOffset(nIdx, id, ohStart, 0, 0), cCount, slabW, alignedSlabW,
                                              srcStrideElements, NegInfValue());
            LocalTensor<T> xLocal = xInQue_.DeQue<T>();
            for (uint32_t rowIdx = 0U; rowIdx < rows; ++rowIdx) {
                LocalTensor<T> accRow = compactLocal[static_cast<uint64_t>(rowIdx) * rowStride];
                const uint32_t rowBaseBytes = rowIdx * inW * sizeof(T);
                for (uint32_t kw = 0U; kw < 3U; ++kw) {
                    Gather(gatheredLocal, xLocal, gatherOffsetLocal,
                           rowBaseBytes + static_cast<uint32_t>(kw * sizeof(T)), compactCount);
                    PipeBarrier<PIPE_V>();
                    Max(accRow, accRow, gatheredLocal, compactCount);
                    PipeBarrier<PIPE_V>();
                }
            }
            xInQue_.FreeTensor(xLocal);
        }
    }

    __aicore__ inline bool ProcessNdc1hwc0NcdhwD3W3DilD2CompactTile(uint64_t outputOffset, int64_t nIdx,
                                                                    uint32_t ohStart, uint32_t rows, uint32_t cCount,
                                                                    uint32_t block, uint32_t outW)
    {
        if (rows <= 1U || cCount == 0U || cCount > block || outW == 0U) {
            return false;
        }
        const uint32_t inW = static_cast<uint32_t>(tiling_->inW);
        const uint32_t slabW = rows * inW;
        const uint32_t alignedSlabW = AlignToVector(slabW);
        const uint32_t compactCount = cCount * outW;
        const uint32_t rowStride = AlignToVector(compactCount + 1U);
        const uint32_t rowElements = outW * block;
        const uint32_t offsetNeed = Ndc1hwc0GatherTempOffset(compactCount);
        if (inW == 0U || slabW == 0U || alignedSlabW == 0U || compactCount == 0U || rowStride == 0U ||
            rowElements == 0U || rows * rowStride > OUTPUT_TILE_NUM || rows * rowElements > OUTPUT_TILE_NUM ||
            cCount * alignedSlabW > INPUT_TILE_NUM || offsetNeed + compactCount > OUTPUT_TILE_NUM) {
            return false;
        }

        LocalTensor<T> compactLocal = calcBuf_.Get<T>();
        LocalTensor<T> outLocal = maskBuf_.Get<T>();
        LocalTensor<T> scratchLocal = tmpBuf_.Get<T>();
        LocalTensor<uint32_t> gatherOffsetLocal = scratchLocal.template ReinterpretCast<uint32_t>();
        LocalTensor<T> gatheredLocal = scratchLocal[offsetNeed];
        InitNdc1hwc0NcdhwD3W3DilD2CompactGatherOffsets(gatherOffsetLocal, cCount, outW, alignedSlabW);
        Duplicate(compactLocal, NegInfValue(), rows * rowStride);
        PipeBarrier<PIPE_V>();

        const uint64_t channelStride = static_cast<uint64_t>(tiling_->inD) * static_cast<uint64_t>(tiling_->inH) *
                                       static_cast<uint64_t>(tiling_->inW);
        const uint32_t srcStrideElements = static_cast<uint32_t>(channelStride - slabW);
        ReduceNdc1hwc0NcdhwD3W3Slab(compactLocal, gatheredLocal, gatherOffsetLocal, nIdx, ohStart, rows, cCount, inW,
                                    slabW, alignedSlabW, compactCount, rowStride, srcStrideElements);

        if (!ScatterNdc1hwc0CompactTileActiveChannels(outLocal, compactLocal, rows, rowStride, cCount, block, outW,
                                                      cCount, 1U)) {
            return false;
        }
        CopyOutVector(outputOffset, outLocal, rows * rowElements);
        return true;
    }

    __aicore__ inline bool ProcessNdc1hwc0NcdhwD3W3DilD2SlabTile(uint64_t outputOffset, int64_t nIdx, uint32_t ohStart,
                                                                 uint32_t rows, uint32_t cCount, uint32_t block,
                                                                 uint32_t outW)
    {
        if (rows <= 1U) {
            return false;
        }
        const uint32_t inW = static_cast<uint32_t>(tiling_->inW);
        const uint32_t slabW = rows * inW;
        const uint32_t alignedSlabW = AlignToVector(slabW);
        const uint32_t alignedW = AlignToVector(outW);
        const uint32_t compactCount = cCount * alignedW;
        const uint32_t rowStride = AlignToVector(compactCount + 1U);
        const uint32_t rowElements = outW * block;
        const uint32_t offsetNeed = Ndc1hwc0GatherTempOffset(compactCount);
        if (inW == 0U || slabW == 0U || alignedSlabW == 0U || compactCount == 0U || rowElements == 0U ||
            rows * rowStride > OUTPUT_TILE_NUM || rows * rowElements > OUTPUT_TILE_NUM ||
            cCount * alignedSlabW > INPUT_TILE_NUM || offsetNeed + compactCount > OUTPUT_TILE_NUM) {
            return false;
        }

        LocalTensor<T> compactLocal = calcBuf_.Get<T>();
        LocalTensor<T> outLocal = maskBuf_.Get<T>();
        LocalTensor<T> scratchLocal = tmpBuf_.Get<T>();
        LocalTensor<uint32_t> gatherOffsetLocal = scratchLocal.template ReinterpretCast<uint32_t>();
        LocalTensor<T> gatheredLocal = scratchLocal[offsetNeed];
        InitNdc1hwc0NcdhwD3W3DilD2SlabGatherOffsets(gatherOffsetLocal, cCount, outW, alignedW, alignedSlabW);
        Duplicate(compactLocal, NegInfValue(), rows * rowStride);
        PipeBarrier<PIPE_V>();

        const uint64_t channelStride = static_cast<uint64_t>(tiling_->inD) * static_cast<uint64_t>(tiling_->inH) *
                                       static_cast<uint64_t>(tiling_->inW);
        const uint32_t srcStrideElements = static_cast<uint32_t>(channelStride - slabW);
        ReduceNdc1hwc0NcdhwD3W3Slab(compactLocal, gatheredLocal, gatherOffsetLocal, nIdx, ohStart, rows, cCount, inW,
                                    slabW, alignedSlabW, compactCount, rowStride, srcStrideElements);

        ScatterNdc1hwc0NcdhwD3W3Rows(outLocal, compactLocal, scratchLocal, rows, rowStride, cCount, block, outW,
                                     alignedW, compactCount);
        CopyOutVector(outputOffset, outLocal, rows * rowElements);
        return true;
    }

    __aicore__ inline void InitNdc1hwc0NcdhwD3W3DilD2GatherOffsets(LocalTensor<uint32_t> offsetLocal, uint32_t cCount,
                                                                   uint32_t outW, uint32_t alignedW,
                                                                   uint32_t alignedInputW)
    {
        InitNdc1hwc0NcdhwDilatedWGatherOffsets(offsetLocal, cCount, outW, alignedW, alignedInputW);
    }

    __aicore__ inline void InitNdc1hwc0NcdhwDilatedWGatherOffsets(LocalTensor<uint32_t> offsetLocal, uint32_t cCount,
                                                                  uint32_t outW, uint32_t alignedW,
                                                                  uint32_t sourceStride)
    {
        LocalTensor<int32_t> offsetI32 = offsetLocal.template ReinterpretCast<int32_t>();
        for (uint32_t c0 = 0U; c0 < cCount; ++c0) {
            const uint32_t dstBase = c0 * alignedW;
            const uint32_t srcBase = c0 * sourceStride;
            for (uint32_t ow = 0U; ow < alignedW; ++ow) {
                uint32_t srcIndex = srcBase;
                if (ow < outW) {
                    srcIndex = srcBase + ow * 3U;
                }
                offsetI32.SetValue(dstBase + ow, static_cast<int32_t>(srcIndex * sizeof(T)));
            }
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void InitNdc1hwc0NcdhwD3W3DilD2CompactGatherOffsets(LocalTensor<uint32_t> offsetLocal,
                                                                          uint32_t cCount, uint32_t outW,
                                                                          uint32_t alignedSlabW)
    {
        LocalTensor<int32_t> offsetI32 = offsetLocal.template ReinterpretCast<int32_t>();
        for (uint32_t ow = 0U; ow < outW; ++ow) {
            const uint32_t dstBase = ow * cCount;
            for (uint32_t c0 = 0U; c0 < cCount; ++c0) {
                const uint32_t srcIndex = c0 * alignedSlabW + ow * 3U;
                offsetI32.SetValue(dstBase + c0, static_cast<int32_t>(srcIndex * sizeof(T)));
            }
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void InitNdc1hwc0NcdhwD3W3DilD2SlabGatherOffsets(LocalTensor<uint32_t> offsetLocal,
                                                                       uint32_t cCount, uint32_t outW,
                                                                       uint32_t alignedW, uint32_t alignedSlabW)
    {
        InitNdc1hwc0NcdhwDilatedWGatherOffsets(offsetLocal, cCount, outW, alignedW, alignedSlabW);
    }

    __aicore__ inline bool InitNdc1hwc0D2H3W2Dil2PlanePath(int64_t dataFormat, bool requireInputW,
                                                           uint64_t& block) const
    {
        if (tiling_->dataFormat != dataFormat || tiling_->outputLayout != OUTPUT_LAYOUT_NDC1HWC0_VALUE ||
            tiling_->outW <= 0 || tiling_->outH <= 0 || tiling_->inH <= 0 || tiling_->c <= 0 ||
            (requireInputW && tiling_->inW <= 0)) {
            return false;
        }
        block = Ndc1hwc0Block();
        const uint64_t validC1 = Ndc1hwc0ValidC1(block);
        return block != 0U && validC1 == 1U && static_cast<uint64_t>(tiling_->c) <= block &&
               IsNdc1hwc0CompactStorage(block, validC1) && HasNdc1hwc0D2H3W2PoolSpec() && HasNdc1hwc0D2H3W2Shape();
    }

    __aicore__ inline bool CanUseNdc1hwc0NdhwcD2H3W2Dil2PlanePath() const
    {
        uint64_t block = 0U;
        if (!InitNdc1hwc0D2H3W2Dil2PlanePath(FORMAT_NDHWC_VALUE, false, block)) {
            return false;
        }
        const uint32_t cCount = static_cast<uint32_t>(tiling_->c);
        const uint32_t alignedC = AlignToVector(cCount);
        const uint32_t outW = static_cast<uint32_t>(tiling_->outW);
        const uint32_t outH = static_cast<uint32_t>(tiling_->outH);
        const uint32_t inW = static_cast<uint32_t>(tiling_->inW);
        const uint32_t blockU32 = static_cast<uint32_t>(block);
        const uint32_t rowElements = outW * blockU32;
        const uint32_t planeElements = outH * rowElements;
        const uint32_t inputSlabElements = static_cast<uint32_t>(tiling_->inH) * inW * blockU32;
        const uint32_t compactPlane = outH * rowElements;
        return HasNdhwcD2H3W2BufferCapacity(alignedC, blockU32, outW, inW, inputSlabElements, planeElements,
                                            compactPlane);
    }

    __aicore__ inline bool CanUseNdc1hwc0NcdhwD2H3W2Dil2PlanePath() const
    {
        uint64_t block = 0U;
        if (!InitNdc1hwc0D2H3W2Dil2PlanePath(FORMAT_NCDHW_VALUE, true, block)) {
            return false;
        }
        const uint32_t outW = static_cast<uint32_t>(tiling_->outW);
        const uint32_t outH = static_cast<uint32_t>(tiling_->outH);
        const uint32_t rowElements = outW * static_cast<uint32_t>(block);
        const uint32_t planeElements = outH * rowElements;
        const uint32_t alignedW = AlignToVector(static_cast<uint32_t>(tiling_->inW));
        const uint32_t inputChannelPlane = alignedW * static_cast<uint32_t>(tiling_->inH);
        return rowElements > 0U && planeElements > 0U && planeElements <= OUTPUT_TILE_NUM && inputChannelPlane > 0U &&
               inputChannelPlane <= INPUT_TILE_NUM;
    }

    __aicore__ inline bool HasNdc1hwc0D2H3W2PoolSpec() const
    {
        return tiling_->kD == 2 && tiling_->kH == 3 && tiling_->kW == 2 && tiling_->sD == 1 && tiling_->sH == 2 &&
               tiling_->sW == 1 && tiling_->dilationD == 2 && tiling_->dilationH == 2 && tiling_->dilationW == 1;
    }

    __aicore__ inline bool HasNdhwcD2H3W2BufferCapacity(uint32_t alignedC, uint32_t block, uint32_t outW, uint32_t inW,
                                                        uint32_t inputSlabElements, uint32_t planeElements,
                                                        uint32_t compactPlane) const
    {
        return alignedC == block && outW > 0U && inW > 0U && inputSlabElements > 0U &&
               inputSlabElements <= INPUT_TILE_NUM && planeElements > 0U && planeElements <= OUTPUT_TILE_NUM &&
               compactPlane > 0U && compactPlane <= OUTPUT_TILE_NUM;
    }

    __aicore__ inline bool HasNdc1hwc0D2H3W2Shape() const
    {
        return tiling_->n > 0 && tiling_->c > 0 && tiling_->inD > 0 && tiling_->inH > 0 && tiling_->inW > 0 &&
               tiling_->outD > 0 && tiling_->outH > 0 && tiling_->outW > 0 && tiling_->padFront >= 0 &&
               tiling_->padTop >= 0 && tiling_->padLeft >= 0;
    }

    __aicore__ inline bool CanUseNdc1hwc0D2H3W2Dil2PlanePath() const
    {
        return CanUseNdc1hwc0NdhwcD2H3W2Dil2PlanePath() || CanUseNdc1hwc0NcdhwD2H3W2Dil2PlanePath();
    }

    __aicore__ inline void ProcessNdc1hwc0D2H3W2Dil2Plane()
    {
        if (tiling_->dataFormat == FORMAT_NCDHW_VALUE) {
            ProcessNdc1hwc0NcdhwD2H3W2Dil2Plane();
        } else {
            ProcessNdc1hwc0NdhwcD2H3W2Dil2Plane();
        }
    }

    template <bool ncdhw>
    __aicore__ inline void ProcessNdc1hwc0D2H3W2Dil2PlaneRange(uint64_t& cur, uint64_t outEnd, uint64_t block,
                                                               uint64_t validC1, uint32_t outH, uint32_t outW,
                                                               uint32_t rowElements, uint32_t planeElements,
                                                               uint32_t maxGroupPlanes)
    {
        while (cur < outEnd) {
            if (cur % planeElements != 0U || outEnd - cur < planeElements) {
                const uint64_t row = cur / rowElements;
                ProcessNdc1hwc0RowVectorByRow(row, cur, block, validC1);
                cur += rowElements - (cur - row * static_cast<uint64_t>(rowElements));
                continue;
            }
            const uint64_t planeIdx = cur / planeElements;
            const int64_t od = static_cast<int64_t>(planeIdx % static_cast<uint64_t>(tiling_->outD));
            const int64_t nIdx = static_cast<int64_t>(planeIdx / static_cast<uint64_t>(tiling_->outD));
            uint32_t groupPlanes = static_cast<uint32_t>((outEnd - cur) / planeElements);
            const uint32_t remainInN = static_cast<uint32_t>(tiling_->outD - od);
            if (groupPlanes > remainInN) {
                groupPlanes = remainInN;
            }
            if (groupPlanes > maxGroupPlanes) {
                groupPlanes = maxGroupPlanes;
            }
            if constexpr (ncdhw) {
                if (groupPlanes == 0U) {
                    break;
                }
                ProcessNdc1hwc0NcdhwD2H3W2Dil2PlaneGroupTile(cur, nIdx, od, groupPlanes, static_cast<uint32_t>(block),
                                                             outH, outW);
                cur += static_cast<uint64_t>(groupPlanes) * planeElements;
            } else if (groupPlanes > 1U && ProcessNdc1hwc0NdhwcD2H3W2Dil2PlaneGroupTile(
                                               cur, nIdx, od, groupPlanes, static_cast<uint32_t>(block), outH, outW)) {
                cur += static_cast<uint64_t>(groupPlanes) * planeElements;
            } else {
                ProcessNdc1hwc0NdhwcD2H3W2Dil2PlaneTile(cur, nIdx, od, static_cast<uint32_t>(block), outH, outW);
                cur += planeElements;
            }
        }
    }

    __aicore__ inline void ProcessNdc1hwc0NdhwcD2H3W2Dil2Plane()
    {
        uint64_t block = 0U, validC1 = 0U, validOut = 0U;
        if (!InitNdc1hwc0ValidOutput(block, validC1, validOut)) {
            return;
        }
        const uint32_t outW = static_cast<uint32_t>(tiling_->outW);
        const uint32_t outH = static_cast<uint32_t>(tiling_->outH);
        const uint32_t rowElements = outW * static_cast<uint32_t>(block);
        const uint32_t planeElements = outH * rowElements;
        const uint32_t maxGroupPlanes = OUTPUT_TILE_NUM / planeElements;
        uint64_t outOffset = 0U, outCount = 0U, outEnd = 0U;
        GetNdc1hwc0ValidCoreRange(validOut, rowElements, outOffset, outCount, outEnd);
        if (!HasNdc1hwc0CoreWork(outCount, validOut)) {
            return;
        }

        uint64_t cur = outOffset;
        ProcessNdc1hwc0D2H3W2Dil2PlaneRange<false>(cur, outEnd, block, validC1, outH, outW, rowElements, planeElements,
                                                   maxGroupPlanes);
        if (validOut < tiling_->totalOut) {
            CopyOutZeroRangeByCore(validOut, tiling_->totalOut - validOut);
        }
    }

    __aicore__ inline void ProcessNdc1hwc0NcdhwD2H3W2Dil2Plane()
    {
        uint64_t block = 0U, validC1 = 0U, validOut = 0U;
        if (!InitNdc1hwc0ValidOutput(block, validC1, validOut)) {
            return;
        }
        const uint32_t outW = static_cast<uint32_t>(tiling_->outW);
        const uint32_t outH = static_cast<uint32_t>(tiling_->outH);
        const uint32_t rowElements = outW * static_cast<uint32_t>(block);
        const uint32_t planeElements = outH * rowElements;
        const uint32_t maxGroupPlanes = NDC1HWC0_D3H3_OUTPUT_TILE_NUM / planeElements;
        uint64_t outOffset = 0U, outCount = 0U, outEnd = 0U;
        GetNdc1hwc0ValidCoreRange(validOut, rowElements, outOffset, outCount, outEnd);
        if (!HasNdc1hwc0CoreWork(outCount, validOut)) {
            return;
        }

        uint64_t cur = outOffset;
        ProcessNdc1hwc0D2H3W2Dil2PlaneRange<true>(cur, outEnd, block, validC1, outH, outW, rowElements, planeElements,
                                                  maxGroupPlanes);
        if (validOut < tiling_->totalOut) {
            CopyOutZeroRangeByCore(validOut, tiling_->totalOut - validOut);
        }
    }

    __aicore__ inline void ProcessNdc1hwc0NcdhwD2H3W2Dil2PlaneGroupTile(uint64_t outputOffset, int64_t nIdx,
                                                                        int64_t odStart, uint32_t groupPlanes,
                                                                        uint32_t block, uint32_t outH, uint32_t outW)
    {
        const uint32_t rowElements = outW * block;
        const uint32_t planeElements = outH * rowElements;
        const uint32_t totalElements = groupPlanes * planeElements;
        const uint32_t offsetNeed = Ndc1hwc0GatherTempOffset(rowElements);
        const uint32_t wReducedOffset = 2U * offsetNeed;
        const uint32_t outPlaneOffset = wReducedOffset + static_cast<uint32_t>(tiling_->inH) * rowElements;
        const uint32_t cCount = static_cast<uint32_t>(tiling_->c);
        const uint32_t inW = static_cast<uint32_t>(tiling_->inW);
        const uint32_t inputPlane = static_cast<uint32_t>(tiling_->inH) * inW;
        const uint32_t alignedInputPlane = AlignToVector(inputPlane);
        const uint32_t zeroIndex = cCount * alignedInputPlane;
        const uint32_t negIndex = zeroIndex + alignedInputPlane;
        LocalTensor<T> scratchLocal = tmpBuf_.Get<T>();
        LocalTensor<uint32_t> evenOffset = scratchLocal.template ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> oddOffset = scratchLocal[offsetNeed].template ReinterpretCast<uint32_t>();
        InitNdc1hwc0NcdhwD2H3W2InputGatherOffsets(evenOffset, oddOffset, cCount, block, outW, alignedInputPlane, inW,
                                                  zeroIndex, negIndex);
        LocalTensor<T> hwLocal = tmpBuf_.Get<T>()[outPlaneOffset];
        LocalTensor<T> accLocal = calcBuf_.Get<T>();
        Duplicate(accLocal, NegInfValue(), totalElements);
        PipeBarrier<PIPE_V>();

        const int64_t idStart = odStart - tiling_->padFront;
        const int64_t idEnd = odStart + static_cast<int64_t>(groupPlanes - 1U) + tiling_->dilationD - tiling_->padFront;
        for (int64_t id = idStart; id <= idEnd; ++id) {
            if (IsOutOfRange(id, tiling_->inD)) {
                continue;
            }
            ComputeNdc1hwc0NcdhwD2H3W2Dil2HwPlaneGather(nIdx, id, hwLocal, evenOffset, oddOffset, block, outH, outW);
            for (int64_t kd = 0; kd < 2; ++kd) {
                const int64_t gp = id - (odStart + kd * tiling_->dilationD - tiling_->padFront);
                if (gp >= 0 && gp < static_cast<int64_t>(groupPlanes)) {
                    Max(accLocal[static_cast<uint64_t>(gp) * planeElements],
                        accLocal[static_cast<uint64_t>(gp) * planeElements], hwLocal, planeElements);
                    PipeBarrier<PIPE_V>();
                }
            }
        }

        CopyOutVector(outputOffset, accLocal, totalElements);
    }

    __aicore__ inline void InitNdc1hwc0NcdhwD2H3W2InputGatherOffsets(LocalTensor<uint32_t> evenOffset,
                                                                     LocalTensor<uint32_t> oddOffset, uint32_t cCount,
                                                                     uint32_t block, uint32_t outW,
                                                                     uint32_t alignedInputPlane, uint32_t inW,
                                                                     uint32_t zeroIndex, uint32_t negIndex)
    {
        LocalTensor<int32_t> evenI32 = evenOffset.template ReinterpretCast<int32_t>();
        LocalTensor<int32_t> oddI32 = oddOffset.template ReinterpretCast<int32_t>();
        const int32_t zeroOffset = static_cast<int32_t>(zeroIndex * sizeof(T));
        const int32_t negOffset = static_cast<int32_t>(negIndex * sizeof(T));
        for (uint32_t ow = 0U; ow < outW; ++ow) {
            const uint32_t outBase = ow * block;
            for (uint32_t c0 = 0U; c0 < block; ++c0) {
                int32_t even = zeroOffset;
                int32_t odd = zeroOffset;
                if (c0 < cCount) {
                    const uint32_t channelBase = c0 * alignedInputPlane;
                    const uint32_t evenW = ow;
                    even = static_cast<int32_t>((channelBase + evenW) * sizeof(T));
                    odd = negOffset;
                    if (evenW + 1U < inW) {
                        odd = static_cast<int32_t>((channelBase + evenW + 1U) * sizeof(T));
                    }
                }
                evenI32.SetValue(outBase + c0, even);
                oddI32.SetValue(outBase + c0, odd);
            }
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeNdc1hwc0NcdhwD2H3W2Dil2HwPlaneGather(int64_t nIdx, int64_t id,
                                                                       LocalTensor<T> outPlaneLocal,
                                                                       LocalTensor<uint32_t> evenOffset,
                                                                       LocalTensor<uint32_t> oddOffset, uint32_t block,
                                                                       uint32_t outH, uint32_t outW)
    {
        const uint32_t cCount = static_cast<uint32_t>(tiling_->c);
        const uint32_t inW = static_cast<uint32_t>(tiling_->inW);
        const uint32_t inputPlane = static_cast<uint32_t>(tiling_->inH) * inW;
        const uint32_t alignedInputPlane = AlignToVector(inputPlane);
        const uint32_t rowElements = outW * block;
        const uint64_t channelStride = static_cast<uint64_t>(tiling_->inD) * static_cast<uint64_t>(tiling_->inH) *
                                       static_cast<uint64_t>(tiling_->inW);
        const uint32_t srcStrideElements = static_cast<uint32_t>(channelStride - inputPlane);
        const uint32_t zeroIndex = cCount * alignedInputPlane;
        CopyInVectorWBlocksPadStrideValue(InputOffset(nIdx, id, 0, 0, 0), cCount, inputPlane, alignedInputPlane,
                                          srcStrideElements, NegInfValue());
        LocalTensor<T> xLocal = xInQue_.DeQue<T>();
        const uint32_t negIndex = zeroIndex + alignedInputPlane;
        Duplicate(xLocal[zeroIndex], ZeroValue(), alignedInputPlane);
        Duplicate(xLocal[negIndex], NegInfValue(), alignedInputPlane);
        PipeBarrier<PIPE_V>();

        LocalTensor<T> scratchLocal = tmpBuf_.Get<T>();
        const uint32_t offsetNeed = Ndc1hwc0GatherTempOffset(rowElements);
        LocalTensor<T> wReducedLocal = scratchLocal[2U * offsetNeed];
        LocalTensor<T> oddLocal = maskBuf_.Get<T>();

        for (uint32_t ih = 0U; ih < static_cast<uint32_t>(tiling_->inH); ++ih) {
            const uint64_t rowSrcBase = static_cast<uint64_t>(ih) * inW;
            Gather(wReducedLocal[static_cast<uint64_t>(ih) * rowElements], xLocal[rowSrcBase], evenOffset,
                   static_cast<uint32_t>(0), rowElements);
            Gather(oddLocal, xLocal[rowSrcBase], oddOffset, static_cast<uint32_t>(0), rowElements);
            PipeBarrier<PIPE_V>();
            Max(wReducedLocal[static_cast<uint64_t>(ih) * rowElements],
                wReducedLocal[static_cast<uint64_t>(ih) * rowElements], oddLocal, rowElements);
            PipeBarrier<PIPE_V>();
        }
        xInQue_.FreeTensor(xLocal);

        Duplicate(outPlaneLocal, NegInfValue(), outH * rowElements);
        PipeBarrier<PIPE_V>();
        for (uint32_t oh = 0U; oh < outH; ++oh) {
            LocalTensor<T> dstRow = outPlaneLocal[static_cast<uint64_t>(oh) * rowElements];
            for (int64_t kh = 0; kh < 3; ++kh) {
                const int64_t ih = static_cast<int64_t>(oh) * tiling_->sH + kh * tiling_->dilationH - tiling_->padTop;
                if (IsOutOfRange(ih, tiling_->inH)) {
                    continue;
                }
                Max(dstRow, dstRow, wReducedLocal[static_cast<uint64_t>(ih) * rowElements], rowElements);
                PipeBarrier<PIPE_V>();
            }
        }
    }

    __aicore__ inline void ProcessNdc1hwc0NdhwcD2H3W2Dil2PlaneTile(uint64_t outputOffset, int64_t nIdx, int64_t od,
                                                                   uint32_t block, uint32_t outH, uint32_t outW)
    {
        const uint32_t cCount = static_cast<uint32_t>(tiling_->c);
        const uint32_t inW = static_cast<uint32_t>(tiling_->inW);
        const uint32_t rowElements = outW * block;
        LocalTensor<T> scratchLocal = tmpBuf_.Get<T>();
        LocalTensor<T> accLocal = calcBuf_.Get<T>();
        Duplicate(accLocal, NegInfValue(), outH * rowElements);
        PipeBarrier<PIPE_V>();

        for (int64_t kd = 0; kd < 2; ++kd) {
            const int64_t id = od + kd * 2 - tiling_->padFront;
            if (IsOutOfRange(id, tiling_->inD)) {
                continue;
            }
            CopyInVectorWBlocksPadStrideValue(InputOffset(nIdx, id, 0, 0, 0), static_cast<uint32_t>(tiling_->inH) * inW,
                                              cCount, block, 0U, ZeroValue());
            LocalTensor<T> xLocal = xInQue_.DeQue<T>();
            AccumulateNdc1hwc0NdhwcD2H3W2Dil2Slab(xLocal, scratchLocal, accLocal, block, outH, outW);
            xInQue_.FreeTensor(xLocal);
        }

        CopyOutVector(outputOffset, accLocal, outH * rowElements);
    }

    __aicore__ inline bool ProcessNdc1hwc0NdhwcD2H3W2Dil2PlaneGroupTile(uint64_t outputOffset, int64_t nIdx,
                                                                        int64_t odStart, uint32_t groupPlanes,
                                                                        uint32_t block, uint32_t outH, uint32_t outW)
    {
        if (groupPlanes <= 1U) {
            return false;
        }
        const uint32_t cCount = static_cast<uint32_t>(tiling_->c);
        const uint32_t inW = static_cast<uint32_t>(tiling_->inW);
        const uint32_t rowElements = outW * block;
        const uint32_t planeElements = outH * rowElements;
        const uint32_t totalElements = groupPlanes * planeElements;
        if (totalElements == 0U || totalElements > OUTPUT_TILE_NUM) {
            return false;
        }

        LocalTensor<T> hwLocal = tmpBuf_.Get<T>();
        LocalTensor<T> accLocal = calcBuf_.Get<T>();
        Duplicate(accLocal, NegInfValue(), totalElements);
        PipeBarrier<PIPE_V>();

        const int64_t idStart = odStart - tiling_->padFront;
        const int64_t idEnd = odStart + static_cast<int64_t>(groupPlanes - 1U) + 2 - tiling_->padFront;
        for (int64_t id = idStart; id <= idEnd; ++id) {
            if (IsOutOfRange(id, tiling_->inD)) {
                continue;
            }
            CopyInVectorWBlocksPadStrideValue(InputOffset(nIdx, id, 0, 0, 0), static_cast<uint32_t>(tiling_->inH) * inW,
                                              cCount, block, 0U, ZeroValue());
            LocalTensor<T> xLocal = xInQue_.DeQue<T>();
            ComputeNdc1hwc0NdhwcD2H3W2Dil2HwPlane(xLocal, hwLocal, block, outW);
            xInQue_.FreeTensor(xLocal);
            const int64_t rel0 = id - (odStart - tiling_->padFront);
            const int64_t gp0 = rel0;
            const int64_t gp1 = rel0 - 2;
            if (gp0 >= 0 && gp0 < static_cast<int64_t>(groupPlanes)) {
                AccumulateNdc1hwc0NdhwcD2H3W2Dil2HwPlane(hwLocal, accLocal[static_cast<uint64_t>(gp0) * planeElements],
                                                         block, outH, outW);
            }
            if (gp1 >= 0 && gp1 < static_cast<int64_t>(groupPlanes)) {
                AccumulateNdc1hwc0NdhwcD2H3W2Dil2HwPlane(hwLocal, accLocal[static_cast<uint64_t>(gp1) * planeElements],
                                                         block, outH, outW);
            }
        }

        CopyOutVector(outputOffset, accLocal, totalElements);
        return true;
    }

    __aicore__ inline void AccumulateNdc1hwc0NdhwcD2H3W2Dil2Slab(LocalTensor<T> xLocal, LocalTensor<T> hwLocal,
                                                                 LocalTensor<T> accLocal, uint32_t block, uint32_t outH,
                                                                 uint32_t outW)
    {
        ComputeNdc1hwc0NdhwcD2H3W2Dil2HwPlane(xLocal, hwLocal, block, outW);
        AccumulateNdc1hwc0NdhwcD2H3W2Dil2HwPlane(hwLocal, accLocal, block, outH, outW);
    }

    __aicore__ inline void ComputeNdc1hwc0NdhwcD2H3W2Dil2HwPlane(LocalTensor<T> xLocal, LocalTensor<T> hwLocal,
                                                                 uint32_t block, uint32_t outW)
    {
        const uint32_t inH = static_cast<uint32_t>(tiling_->inH);
        const uint32_t inW = static_cast<uint32_t>(tiling_->inW);
        const uint32_t inputHStride = inW * block;
        const uint32_t rowElements = outW * block;
        for (uint32_t ih = 0U; ih < inH; ++ih) {
            LocalTensor<T> srcRow = xLocal[static_cast<uint64_t>(ih) * inputHStride];
            LocalTensor<T> hwRow = hwLocal[static_cast<uint64_t>(ih) * rowElements];
            Max(hwRow, srcRow, srcRow[block], 2U * block);
            Max(hwRow[2U * block], srcRow[2U * block], srcRow[2U * block], block);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void AccumulateNdc1hwc0NdhwcD2H3W2Dil2HwPlane(LocalTensor<T> hwLocal, LocalTensor<T> accLocal,
                                                                    uint32_t block, uint32_t outH, uint32_t outW)
    {
        const uint32_t rowElements = outW * block;
        for (uint32_t oh = 0U; oh < outH; ++oh) {
            LocalTensor<T> accRow = accLocal[static_cast<uint64_t>(oh) * rowElements];
            for (int64_t kh = 0; kh < 3; ++kh) {
                const int64_t ih = static_cast<int64_t>(oh) * 2 + kh * 2 - tiling_->padTop;
                if (IsOutOfRange(ih, tiling_->inH)) {
                    continue;
                }
                Max(accRow, accRow, hwLocal[static_cast<uint64_t>(ih) * rowElements], rowElements);
                PipeBarrier<PIPE_V>();
            }
        }
    }

    __aicore__ inline uint32_t ProcessNdc1hwc0PartialValidRange(uint64_t cur, uint64_t outEnd, uint64_t block,
                                                                uint64_t rowElements, uint64_t validC1)
    {
        if (rowElements == 0U) {
            return 0U;
        }
        const uint64_t rowOffset = cur % rowElements;
        uint64_t curCount64 = rowElements - rowOffset;
        if (curCount64 > outEnd - cur) {
            curCount64 = outEnd - cur;
        }
        const uint32_t curCount = curCount64 > OUTPUT_TILE_NUM ? OUTPUT_TILE_NUM : static_cast<uint32_t>(curCount64);
        LocalTensor<T> yLocal = yOutQue_.AllocTensor<T>();
        FillNdc1hwc0ValidTile(yLocal, cur, curCount, block, rowElements, validC1);
        yOutQue_.EnQue(yLocal);
        LocalTensor<T> yOut = yOutQue_.DeQue<T>();
        DataCopyExtParams copyParams{1U, static_cast<uint32_t>(curCount * sizeof(T)), 0U, 0U, 0U};
        DataCopyPad(yGm_[cur], yOut, copyParams);
        yOutQue_.FreeTensor(yOut);
        return curCount;
    }

    __aicore__ inline bool PrepareNdc1hwc0FullRow(uint64_t& cur, uint64_t outEnd, uint64_t block, uint64_t rowElements,
                                                  uint64_t validC1)
    {
        if (rowElements == 0U) {
            cur = outEnd;
            return false;
        }
        const uint64_t rowOffset = cur % rowElements;
        if (rowOffset == 0U && outEnd - cur >= rowElements) {
            return true;
        }
        cur += ProcessNdc1hwc0PartialValidRange(cur, outEnd, block, rowElements, validC1);
        return false;
    }

    __aicore__ inline bool PrepareNdc1hwc0DecodedFullRow(uint64_t& cur, uint64_t outEnd, uint64_t block,
                                                         uint64_t rowElements, uint64_t validC1,
                                                         Ndc1hwc0DecodedRow& context)
    {
        if (!PrepareNdc1hwc0FullRow(cur, outEnd, block, rowElements, validC1)) {
            return false;
        }
        context = DecodeNdc1hwc0GroupRow(cur, rowElements, validC1);
        return true;
    }

    __aicore__ inline bool CanUseNdc1hwc0TinyK3ValidGroupPath() const
    {
        if constexpr (AscendC::Std::is_same<T, bfloat16_t>::value) {
            return false;
        } else {
            if (!HasNdc1hwc0TinyK3ValidShape() || !HasNdc1hwc0TinyK3ValidPoolSpec()) {
                return false;
            }
            const uint64_t block = Ndc1hwc0Block();
            const uint64_t validC1 = Ndc1hwc0ValidC1(block);
            if (block != 16U || validC1 != 1U || static_cast<uint64_t>(tiling_->c) > block ||
                !IsNdc1hwc0CompactStorage(block, validC1)) {
                return false;
            }
            const uint32_t outW = static_cast<uint32_t>(tiling_->outW);
            const uint32_t rowElements = outW * static_cast<uint32_t>(block);
            if (rowElements == 0U || rowElements > OUTPUT_TILE_NUM) {
                return false;
            }
            const uint32_t cCount = static_cast<uint32_t>(tiling_->c);
            const uint32_t inW = static_cast<uint32_t>(tiling_->inW);
            if (tiling_->dataFormat == FORMAT_NDHWC_VALUE) {
                const uint32_t alignedC = AlignToVector(cCount);
                return alignedC <= block && inW * alignedC <= INPUT_TILE_NUM;
            }
            const uint32_t alignedW = AlignToVector(outW);
            const uint32_t alignedInputW = AlignToVector(inW);
            const uint32_t rowStride = AlignToVector(cCount * alignedW + 1U);
            return cCount * alignedInputW <= INPUT_TILE_NUM && rowStride <= OUTPUT_TILE_NUM;
        }
    }

    __aicore__ inline bool CanUseNdc1hwc0LogicalTinyK3WholeNPath() const
    {
        const uint64_t block = Ndc1hwc0Block();
        const uint64_t validC1 = Ndc1hwc0ValidC1(block);
        if constexpr (AscendC::Std::is_same<T, bfloat16_t>::value) {
            return false;
        }
        if (tiling_->inputLayout == INPUT_LAYOUT_NDC1HWC0_VALUE ||
            tiling_->outputLayout != OUTPUT_LAYOUT_NDC1HWC0_VALUE || block != 16U || validC1 != 1U ||
            tiling_->outputC1 != 1 || tiling_->c <= 0 || static_cast<uint64_t>(tiling_->c) > block ||
            !HasNdc1hwc0TinyK3ValidShape() || !HasNdc1hwc0TinyK3ValidPoolSpec()) {
            return false;
        }
        const uint64_t inputSpatial = static_cast<uint64_t>(tiling_->inD) * tiling_->inH * tiling_->inW;
        const uint64_t alignedInputSpatial = AlignToVector(static_cast<uint32_t>(inputSpatial));
        const uint64_t inputRow = static_cast<uint64_t>(tiling_->inW) * block;
        const uint64_t compactRow = static_cast<uint64_t>(tiling_->outW) * block;
        const uint64_t widthCount = static_cast<uint64_t>(tiling_->inD) * tiling_->inH * compactRow;
        const uint64_t heightCount = static_cast<uint64_t>(tiling_->inD) * compactRow;
        const uint64_t outputCount = static_cast<uint64_t>(tiling_->outD) * compactRow;
        const uint64_t inputRowStride = inputRow * sizeof(T) / UB_BLOCK_BYTES;
        const uint64_t compactRowStride = compactRow * sizeof(T) / UB_BLOCK_BYTES;
        return inputSpatial > 0U && alignedInputSpatial >= inputSpatial &&
               (alignedInputSpatial - inputSpatial) <= 255U &&
               (static_cast<uint64_t>(tiling_->c) + 1U) * alignedInputSpatial <= INPUT_TILE_NUM &&
               alignedInputSpatial * block <= OUTPUT_TILE_NUM && widthCount <= OUTPUT_TILE_NUM &&
               heightCount <= OUTPUT_TILE_NUM && outputCount <= OUTPUT_TILE_NUM && inputRowStride > 0U &&
               inputRowStride <= 255U && compactRowStride > 0U && compactRowStride <= 255U &&
               static_cast<uint64_t>(tiling_->inD) * tiling_->inH <= 255U && tiling_->inD <= 255;
    }

    __aicore__ inline void LoadNdc1hwc0LogicalTinyK3Input(LocalTensor<T> inputLocal, LocalTensor<T> blockedLocal,
                                                          uint32_t nIdx, uint32_t block, uint32_t inputSpatial,
                                                          uint32_t alignedInputSpatial)
    {
        if (tiling_->dataFormat == FORMAT_NDHWC_VALUE) {
            const uint32_t inputPoints = static_cast<uint32_t>(tiling_->inD * tiling_->inH * tiling_->inW);
            const DataCopyExtParams inputCopy{static_cast<uint16_t>(inputPoints),
                                              static_cast<uint32_t>(tiling_->c * sizeof(T)), 0U, 0U, 0U};
            const DataCopyPadExtParams<T> inputPad{
                true, 0U, static_cast<uint8_t>(block - static_cast<uint32_t>(tiling_->c)), ZeroValue()};
            DataCopyPad(blockedLocal, xGm_[static_cast<uint64_t>(nIdx) * inputPoints * tiling_->c], inputCopy,
                        inputPad);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            return;
        }
        const DataCopyExtParams inputCopy{static_cast<uint16_t>(tiling_->c),
                                          static_cast<uint32_t>(inputSpatial * sizeof(T)), 0U, 0U, 0U};
        const DataCopyPadExtParams<T> inputPad{true, 0U, static_cast<uint8_t>(alignedInputSpatial - inputSpatial),
                                               ZeroValue()};
        DataCopyPad(inputLocal, xGm_[static_cast<uint64_t>(nIdx) * tiling_->c * inputSpatial], inputCopy, inputPad);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        LocalTensor<T> zeroLocal = inputLocal[static_cast<uint64_t>(tiling_->c) * alignedInputSpatial];
        Duplicate(zeroLocal, ZeroValue(), alignedInputSpatial);
        PipeBarrier<PIPE_V>();
        TransposeNdc1hwc0C0PlaneBlock(blockedLocal, inputLocal, zeroLocal, alignedInputSpatial,
                                      static_cast<uint32_t>(tiling_->c), block);
    }

    __aicore__ inline void ReduceNdc1hwc0LogicalTinyK3Rows(LocalTensor<T> dstLocal, LocalTensor<T> src0Local,
                                                           LocalTensor<T> src1Local, uint32_t rowElements,
                                                           uint8_t repeats, const BinaryRepeatParams& params)
    {
        constexpr uint32_t vectorElements = 256U / sizeof(T);
        for (uint32_t offset = 0U; offset < rowElements; offset += vectorElements) {
            const uint32_t count = rowElements - offset > vectorElements ? vectorElements : rowElements - offset;
            Max(dstLocal[offset], src0Local[offset], src1Local[offset], count, repeats, params);
        }
    }

    __aicore__ inline void ReduceNdc1hwc0LogicalTinyK3(LocalTensor<T> blockedLocal, LocalTensor<T> widthLocal,
                                                       LocalTensor<T> heightLocal, uint32_t block)
    {
        const uint32_t inputRow = static_cast<uint32_t>(tiling_->inW) * block;
        const uint32_t compactRow = static_cast<uint32_t>(tiling_->outW) * block;
        const uint8_t inputRowStride = static_cast<uint8_t>(inputRow * sizeof(T) / UB_BLOCK_BYTES);
        const uint8_t compactRowStride = static_cast<uint8_t>(compactRow * sizeof(T) / UB_BLOCK_BYTES);
        const uint8_t widthRepeats = static_cast<uint8_t>(tiling_->inD * tiling_->inH);
        const BinaryRepeatParams widthParams{1U, 1U, 1U, compactRowStride, inputRowStride, inputRowStride};
        ReduceNdc1hwc0LogicalTinyK3Rows(widthLocal, blockedLocal, blockedLocal[block], compactRow, widthRepeats,
                                        widthParams);
        PipeBarrier<PIPE_V>();
        const BinaryRepeatParams widthFinishParams{1U, 1U, 1U, compactRowStride, compactRowStride, inputRowStride};
        ReduceNdc1hwc0LogicalTinyK3Rows(widthLocal, widthLocal, blockedLocal[2U * block], compactRow, widthRepeats,
                                        widthFinishParams);
        PipeBarrier<PIPE_V>();
        const uint8_t depthRepeats = static_cast<uint8_t>(tiling_->inD);
        const uint8_t heightInputStride = static_cast<uint8_t>(tiling_->inH * compactRowStride);
        const BinaryRepeatParams heightParams{1U, 1U, 1U, compactRowStride, heightInputStride, heightInputStride};
        ReduceNdc1hwc0LogicalTinyK3Rows(heightLocal, widthLocal, widthLocal[compactRow], compactRow, depthRepeats,
                                        heightParams);
        PipeBarrier<PIPE_V>();
        const BinaryRepeatParams heightFinishParams{1U, 1U, 1U, compactRowStride, compactRowStride, heightInputStride};
        ReduceNdc1hwc0LogicalTinyK3Rows(heightLocal, heightLocal, widthLocal[2U * compactRow], compactRow, depthRepeats,
                                        heightFinishParams);
        PipeBarrier<PIPE_V>();
        const uint32_t outputCount = static_cast<uint32_t>(tiling_->outD) * compactRow;
        Max(blockedLocal, heightLocal, heightLocal[compactRow], outputCount);
        PipeBarrier<PIPE_V>();
        Max(blockedLocal, blockedLocal, heightLocal[2U * compactRow], outputCount);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ProcessNdc1hwc0LogicalTinyK3WholeN()
    {
        const uint32_t block = static_cast<uint32_t>(Ndc1hwc0Block());
        const uint32_t inputSpatial = static_cast<uint32_t>(tiling_->inD * tiling_->inH * tiling_->inW);
        const uint32_t alignedInputSpatial = AlignToVector(inputSpatial);
        const uint32_t compactRow = static_cast<uint32_t>(tiling_->outW) * block;
        const uint32_t outputPerN = static_cast<uint32_t>(tiling_->outD) * compactRow;
        const uint32_t worker = GetBlockIdx();
        const uint32_t workerDim = ActiveBlockDim();
        for (uint32_t nIdx = worker; nIdx < static_cast<uint32_t>(tiling_->n); nIdx += workerDim) {
            LocalTensor<T> inputLocal = xInQue_.AllocTensor<T>();
            LocalTensor<T> blockedLocal = calcBuf_.Get<T>();
            LoadNdc1hwc0LogicalTinyK3Input(inputLocal, blockedLocal, nIdx, block, inputSpatial, alignedInputSpatial);
            LocalTensor<T> widthLocal = tmpBuf_.Get<T>();
            LocalTensor<T> heightLocal = maskBuf_.Get<T>();
            ReduceNdc1hwc0LogicalTinyK3(blockedLocal, widthLocal, heightLocal, block);
            CopyOutVector(static_cast<uint64_t>(nIdx) * outputPerN, blockedLocal, outputPerN);
            xInQue_.FreeTensor(inputLocal);
        }
        const uint64_t validOut = static_cast<uint64_t>(tiling_->n) * outputPerN;
        if (validOut < tiling_->totalOut) {
            CopyOutZeroRangeByCore(validOut, tiling_->totalOut - validOut);
        }
    }

    __aicore__ inline bool HasNdc1hwc0TinyK3ValidShape() const
    {
        return tiling_->outputLayout == OUTPUT_LAYOUT_NDC1HWC0_VALUE &&
               (tiling_->dataFormat == FORMAT_NCDHW_VALUE || tiling_->dataFormat == FORMAT_NDHWC_VALUE) &&
               tiling_->outD > 0 && tiling_->outH == 1 && tiling_->outW > 0 && tiling_->c > 0 &&
               tiling_->inD == tiling_->outD + 2 && tiling_->inH == 3 && tiling_->inW == tiling_->outW + 2;
    }

    __aicore__ inline bool HasNdc1hwc0TinyK3ValidPoolSpec() const
    {
        return tiling_->kD == 3 && tiling_->kH == 3 && tiling_->kW == 3 && tiling_->sD == 1 && tiling_->sH == 1 &&
               tiling_->sW == 1 && tiling_->dilationD == 1 && tiling_->dilationH == 1 && tiling_->dilationW == 1 &&
               tiling_->padFront == 0 && tiling_->padTop == 0 && tiling_->padLeft == 0;
    }

    __aicore__ inline void ProcessNdc1hwc0TinyK3ValidGroupStep(uint64_t& cur, uint64_t outEnd, uint64_t block,
                                                               uint64_t validC1, uint32_t cCount, uint32_t outW,
                                                               uint32_t rowElements)
    {
        if (rowElements == 0U) {
            cur = outEnd;
            return;
        }
        if (!PrepareNdc1hwc0FullRow(cur, outEnd, block, rowElements, validC1)) {
            return;
        }
        uint32_t rows = static_cast<uint32_t>((outEnd - cur) / rowElements);
        uint32_t maxRows = OUTPUT_TILE_NUM / rowElements;
        if (tiling_->dataFormat == FORMAT_NCDHW_VALUE) {
            const uint32_t rowStride = AlignToVector(cCount * AlignToVector(outW) + 1U);
            const uint32_t maxCompactRows = rowStride == 0U ? 1U : OUTPUT_TILE_NUM / rowStride;
            if (maxRows > maxCompactRows) {
                maxRows = maxCompactRows;
            }
        }
        if (maxRows == 0U) {
            maxRows = 1U;
        }
        if (rows > maxRows) {
            rows = maxRows;
        }
        const uint64_t startRow = cur / rowElements;
        if (CanUseNdc1hwc0TinyK3WholeNTile(startRow, rows, validC1)) {
            ProcessNdc1hwc0TinyK3ValidNdhwcWholeNTile(cur, startRow, cCount, static_cast<uint32_t>(block), outW);
        } else if (tiling_->dataFormat == FORMAT_NDHWC_VALUE) {
            ProcessNdc1hwc0TinyK3ValidNdhwcTile(cur, startRow, rows, validC1, cCount, static_cast<uint32_t>(block),
                                                outW);
        } else {
            ProcessNdc1hwc0TinyK3ValidNcdhwTile(cur, startRow, rows, validC1, cCount, static_cast<uint32_t>(block),
                                                outW);
        }
        cur += static_cast<uint64_t>(rows) * rowElements;
    }

    __aicore__ inline void ProcessNdc1hwc0TinyK3ValidGroup()
    {
        uint64_t block = 0U, validC1 = 0U, validOut = 0U;
        if (!InitNdc1hwc0ValidOutput(block, validC1, validOut)) {
            return;
        }
        const uint32_t outW = static_cast<uint32_t>(tiling_->outW);
        const uint32_t rowElements = outW * static_cast<uint32_t>(block);
        const uint32_t cCount = static_cast<uint32_t>(tiling_->c);
        uint64_t outOffset = 0U, outCount = 0U, outEnd = 0U;
        GetNdc1hwc0ValidCoreRange(validOut, rowElements, outOffset, outCount, outEnd);
        if (!HasNdc1hwc0CoreWork(outCount, validOut)) {
            return;
        }

        uint64_t cur = outOffset;
        while (cur < outEnd) {
            ProcessNdc1hwc0TinyK3ValidGroupStep(cur, outEnd, block, validC1, cCount, outW, rowElements);
        }
        if (validOut < tiling_->totalOut) {
            CopyOutZeroRangeByCore(validOut, tiling_->totalOut - validOut);
        }
    }

    __aicore__ inline bool CanUseNdc1hwc0TinyK3WholeNTile(uint64_t startRow, uint32_t rows, uint64_t validC1) const
    {
        if (tiling_->dataFormat != FORMAT_NDHWC_VALUE || validC1 != 1U || tiling_->outD <= 0 ||
            rows != static_cast<uint32_t>(tiling_->outD) || startRow % static_cast<uint64_t>(tiling_->outD) != 0U) {
            return false;
        }
        if (tiling_->c <= 0 || tiling_->inD != tiling_->outD + 2 || tiling_->inH != 3 ||
            tiling_->inW != tiling_->outW + 2 || tiling_->outH != 1 || tiling_->outW <= 0) {
            return false;
        }
        const uint32_t inputCount = static_cast<uint32_t>(
            static_cast<uint64_t>(tiling_->c) * static_cast<uint64_t>(tiling_->inD) *
            static_cast<uint64_t>(tiling_->inH) * static_cast<uint64_t>(tiling_->inW));
        const uint32_t compactStride = AlignToVector(static_cast<uint32_t>(tiling_->outW * tiling_->c) + 1U);
        const uint64_t compactNeed = static_cast<uint64_t>(tiling_->outD) * compactStride;
        const uint64_t outputNeed = static_cast<uint64_t>(tiling_->outD) * tiling_->outW * Ndc1hwc0Block();
        return inputCount > 0U && AlignToVector(inputCount) <= INPUT_TILE_NUM && compactNeed <= OUTPUT_TILE_NUM &&
               outputNeed <= OUTPUT_TILE_NUM;
    }

    __aicore__ inline void ProcessNdc1hwc0TinyK3ValidNdhwcWholeNTile(uint64_t outputOffset, uint64_t startRow,
                                                                     uint32_t cCount, uint32_t block, uint32_t outW)
    {
        const int64_t nIdx = static_cast<int64_t>(startRow / static_cast<uint64_t>(tiling_->outD));

        const uint32_t outD = static_cast<uint32_t>(tiling_->outD);
        const uint32_t inH = static_cast<uint32_t>(tiling_->inH);
        const uint32_t inW = static_cast<uint32_t>(tiling_->inW);
        const uint32_t rowElements = outW * block;
        const uint32_t inputCount = static_cast<uint32_t>(
            static_cast<uint64_t>(tiling_->c) * static_cast<uint64_t>(tiling_->inD) *
            static_cast<uint64_t>(tiling_->inH) * static_cast<uint64_t>(tiling_->inW));
        const uint32_t alignedInputCount = AlignToVector(inputCount);
        CopyInVectorPadValue(InputOffset(nIdx, 0, 0, 0, 0), inputCount, alignedInputCount, NegInfValue());
        LocalTensor<T> xLocal = xInQue_.DeQue<T>();

        LocalTensor<T> outLocal = maskBuf_.Get<T>();
        LocalTensor<T> compactLocal = calcBuf_.Get<T>();
        LocalTensor<T> scratchLocal = tmpBuf_.Get<T>();
        LocalTensor<uint32_t> offsetLocal = scratchLocal.template ReinterpretCast<uint32_t>();
        const uint32_t rowCount = outW * cCount;
        const uint32_t compactStride = AlignToVector(rowCount + 1U);
        Duplicate(compactLocal, NegInfValue(), outD * compactStride);
        PipeBarrier<PIPE_V>();
        const uint32_t dStride = inH * inW * cCount;
        const uint32_t hStride = inW * cCount;
        for (uint32_t outDIdx = 0U; outDIdx < outD; ++outDIdx) {
            LocalTensor<T> accRow = compactLocal[static_cast<uint64_t>(outDIdx) * compactStride];
            for (uint32_t kd = 0U; kd < 3U; ++kd) {
                const uint32_t id = outDIdx + kd;
                for (uint32_t kh = 0U; kh < 3U; ++kh) {
                    const uint32_t base = id * dStride + kh * hStride;
                    Max(accRow, accRow, xLocal[base], rowCount);
                    PipeBarrier<PIPE_V>();
                    Max(accRow, accRow, xLocal[base + cCount], rowCount);
                    PipeBarrier<PIPE_V>();
                    Max(accRow, accRow, xLocal[base + 2U * cCount], rowCount);
                    PipeBarrier<PIPE_V>();
                }
            }
        }
        const uint32_t zeroIndex = rowCount;
        if (!ScatterNdc1hwc0CompactRowsReuseRowOffset(outLocal, compactLocal, offsetLocal, outD, compactStride, cCount,
                                                      block, outW, cCount, 1U, zeroIndex)) {
            ScatterNdc1hwc0CompactRowsChecked(outLocal, compactLocal, offsetLocal, outD, compactStride, cCount, block,
                                              outW, cCount, 1U, zeroIndex);
        }
        CopyOutVector(outputOffset, outLocal, outD * rowElements);
        xInQue_.FreeTensor(xLocal);
    }

    __aicore__ inline void ProcessNdc1hwc0TinyK3ValidNdhwcTile(uint64_t outputOffset, uint64_t startRow, uint32_t rows,
                                                               uint64_t validC1, uint32_t cCount, uint32_t block,
                                                               uint32_t outW)
    {
        const uint32_t rowElements = outW * block;
        const uint32_t inW = static_cast<uint32_t>(tiling_->inW);
        LocalTensor<T> outLocal = maskBuf_.Get<T>();
        Duplicate(outLocal, NegInfValue(), rows * rowElements);
        PipeBarrier<PIPE_V>();
        for (uint32_t rowIdx = 0U; rowIdx < rows; ++rowIdx) {
            int64_t nIdx = 0;
            int64_t od = 0;
            int64_t c1Idx = 0;
            int64_t oh = 0;
            DecodeNdc1hwc0Row(startRow + rowIdx, validC1, nIdx, od, c1Idx, oh);
            LocalTensor<T> accRow = outLocal[static_cast<uint64_t>(rowIdx) * rowElements];
            for (int64_t kd = 0; kd < 3; ++kd) {
                const int64_t id = od + kd;
                for (int64_t kh = 0; kh < 3; ++kh) {
                    CopyInVectorWBlocksPadStrideValue(InputOffset(nIdx, id, kh, 0, 0), inW, cCount, block, 0U,
                                                      NegInfValue());
                    LocalTensor<T> xLocal = xInQue_.DeQue<T>();
                    for (uint32_t kw = 0U; kw < 3U; ++kw) {
                        Max(accRow, accRow, xLocal[static_cast<uint64_t>(kw) * block], rowElements);
                        PipeBarrier<PIPE_V>();
                    }
                    xInQue_.FreeTensor(xLocal);
                }
            }
            ZeroNdhwcAlignedCTail(accRow, cCount, block, outW);
        }
        CopyOutVector(outputOffset, outLocal, rows * rowElements);
    }

    __aicore__ inline void ProcessNdc1hwc0TinyK3ValidNcdhwTile(uint64_t outputOffset, uint64_t startRow, uint32_t rows,
                                                               uint64_t validC1, uint32_t cCount, uint32_t block,
                                                               uint32_t outW)
    {
        const uint32_t rowElements = outW * block;
        const uint32_t alignedW = AlignToVector(outW);
        const uint32_t inW = static_cast<uint32_t>(tiling_->inW);
        const uint32_t alignedInputW = AlignToVector(inW);
        const uint32_t rowStride = AlignToVector(cCount * alignedW + 1U);
        const uint64_t channelStride = static_cast<uint64_t>(tiling_->inD) * static_cast<uint64_t>(tiling_->inH) *
                                       static_cast<uint64_t>(tiling_->inW);
        const uint32_t srcStrideElements = static_cast<uint32_t>(channelStride - inW);
        LocalTensor<T> compactLocal = calcBuf_.Get<T>();
        LocalTensor<T> outLocal = maskBuf_.Get<T>();
        LocalTensor<T> scratchLocal = tmpBuf_.Get<T>();
        LocalTensor<uint32_t> offsetLocal = scratchLocal.template ReinterpretCast<uint32_t>();
        Duplicate(compactLocal, NegInfValue(), rows * rowStride);
        PipeBarrier<PIPE_V>();
        for (uint32_t rowIdx = 0U; rowIdx < rows; ++rowIdx) {
            LocalTensor<T> accRow = compactLocal[static_cast<uint64_t>(rowIdx) * rowStride];
            ReduceNdc1hwc0TinyK3ValidNcdhwRow(accRow, startRow + rowIdx, validC1, cCount, outW, alignedW, inW,
                                              alignedInputW, srcStrideElements);
        }
        const uint32_t zeroIndex = cCount * alignedW;
        if (!ScatterNdc1hwc0CompactRowsReuseRowOffset(outLocal, compactLocal, offsetLocal, rows, rowStride, cCount,
                                                      block, outW, 1U, alignedW, zeroIndex)) {
            ScatterNdc1hwc0CompactRowsChecked(outLocal, compactLocal, offsetLocal, rows, rowStride, cCount, block, outW,
                                              1U, alignedW, zeroIndex);
        }
        CopyOutVector(outputOffset, outLocal, rows * rowElements);
    }

    __aicore__ inline void ReduceNdc1hwc0TinyK3ValidNcdhwRow(LocalTensor<T> accRow, uint64_t row, uint64_t validC1,
                                                             uint32_t cCount, uint32_t outW, uint32_t alignedW,
                                                             uint32_t inW, uint32_t alignedInputW,
                                                             uint32_t srcStrideElements)
    {
        int64_t nIdx = 0;
        int64_t od = 0;
        int64_t c1Idx = 0;
        int64_t oh = 0;
        DecodeNdc1hwc0Row(row, validC1, nIdx, od, c1Idx, oh);
        for (int64_t kd = 0; kd < 3; ++kd) {
            const int64_t id = od + kd;
            for (int64_t kh = 0; kh < 3; ++kh) {
                CopyInVectorWBlocksPadStrideValue(InputOffset(nIdx, id, kh, 0, 0), cCount, inW, alignedInputW,
                                                  srcStrideElements, NegInfValue());
                LocalTensor<T> xLocal = xInQue_.DeQue<T>();
                for (uint32_t kw = 0U; kw < 3U; ++kw) {
                    for (uint32_t c0 = 0U; c0 < cCount; ++c0) {
                        Max(accRow[static_cast<uint64_t>(c0) * alignedW], accRow[static_cast<uint64_t>(c0) * alignedW],
                            xLocal[static_cast<uint64_t>(c0) * alignedInputW + kw], outW);
                    }
                    PipeBarrier<PIPE_V>();
                }
                xInQue_.FreeTensor(xLocal);
            }
        }
    }

    __aicore__ inline bool CanUseNdc1hwc0SmallCGroupPath() const
    {
        if (!HasNdc1hwc0SmallCBasicParams()) {
            return false;
        }
        const uint64_t block = Ndc1hwc0Block();
        const uint64_t validC1 = Ndc1hwc0ValidC1(block);
        if (block == 0U || validC1 == 0U || block > 32U || !IsNdc1hwc0CompactStorage(block, validC1)) {
            return false;
        }
        if (validC1 > 1U && static_cast<uint64_t>(tiling_->c) % block != 0U) {
            return false;
        }
        const uint64_t rowElements = static_cast<uint64_t>(tiling_->outW) * block;
        if (rowElements == 0U || rowElements > OUTPUT_TILE_NUM) {
            return false;
        }
        const uint64_t maxSpanW = MaxNdc1hwc0SmallCSpanW();
        if (maxSpanW == 0U) {
            return false;
        }
        const uint64_t cCount = block < static_cast<uint64_t>(tiling_->c) ? block : static_cast<uint64_t>(tiling_->c);
        const uint64_t alignedC = AlignToVector(static_cast<uint32_t>(cCount));
        return maxSpanW * alignedC <= INPUT_TILE_NUM &&
               static_cast<uint64_t>(tiling_->outW) * alignedC + 1U <= OUTPUT_TILE_NUM;
    }

    __aicore__ inline bool HasNdc1hwc0SmallCBasicParams() const
    {
        return tiling_->outputLayout == OUTPUT_LAYOUT_NDC1HWC0_VALUE && tiling_->dataFormat == FORMAT_NDHWC_VALUE &&
               tiling_->outW > 0 && tiling_->outH > 0 && tiling_->c > 0 && tiling_->kD > 0 && tiling_->kH > 0 &&
               tiling_->kW > 0 && tiling_->sW > 0 && tiling_->dilationW > 0;
    }

    __aicore__ inline uint64_t MaxNdc1hwc0SmallCSpanW() const
    {
        uint64_t maxSpanW = 0U;
        for (int64_t kw = 0; kw < tiling_->kW; ++kw) {
            uint32_t wStart = 0U;
            uint32_t wCount = 0U;
            if (!CalcValidWRange(kw, wStart, wCount)) {
                continue;
            }
            const uint64_t spanW = tiling_->sW == 1 ?
                                       static_cast<uint64_t>(wCount) :
                                       static_cast<uint64_t>(wCount - 1U) * static_cast<uint64_t>(tiling_->sW) + 1U;
            if (maxSpanW < spanW) {
                maxSpanW = spanW;
            }
        }
        return maxSpanW;
    }

    __aicore__ inline uint32_t Ndc1hwc0SmallCGroupMaxRows(uint32_t rows, uint32_t rowElements, uint32_t rowStride) const
    {
        uint32_t tileRows = rows;
        const uint32_t maxRowsByOutput = rowElements == 0U ? 1U : OUTPUT_TILE_NUM / rowElements;
        if (maxRowsByOutput > 0U && tileRows > maxRowsByOutput) {
            tileRows = maxRowsByOutput;
        }
        while (tileRows > 1U) {
            const uint32_t compactNeed = tileRows * rowStride;
            const uint32_t offsetNeed = Ndc1hwc0GatherTempOffset(rowElements);
            if (compactNeed <= OUTPUT_TILE_NUM && offsetNeed <= OUTPUT_TILE_NUM) {
                break;
            }
            --tileRows;
        }
        return tileRows == 0U ? 1U : tileRows;
    }

    __aicore__ inline void ProcessNdc1hwc0SmallCGroupStep(uint64_t& cur, uint64_t outEnd, uint64_t block,
                                                          uint64_t validC1, uint32_t outW, uint32_t rowElements)
    {
        Ndc1hwc0DecodedRow context{};
        if (!PrepareNdc1hwc0DecodedFullRow(cur, outEnd, block, rowElements, validC1, context)) {
            return;
        }
        const int64_t cBase = context.c1Idx * static_cast<int64_t>(block);
        const int64_t activeChannels = Ndc1hwc0ActiveChannels(cBase, block);
        if (activeChannels <= 0) {
            ProcessNdc1hwc0RowVectorByRow(context.row, cur, block, validC1);
            cur += rowElements;
            return;
        }
        if (rowElements == 0U) {
            cur = outEnd;
            return;
        }
        uint32_t rows = validC1 == 1U ? static_cast<uint32_t>((outEnd - cur) / rowElements) :
                                        static_cast<uint32_t>(tiling_->outH - context.oh);
        const uint64_t remainRows = (outEnd - cur) / rowElements;
        if (static_cast<uint64_t>(rows) > remainRows) {
            rows = static_cast<uint32_t>(remainRows);
        }
        const uint32_t rowStride = Ndc1hwc0SmallCCompactStride(static_cast<uint32_t>(activeChannels), outW);
        rows = Ndc1hwc0SmallCGroupMaxRows(rows, rowElements, rowStride);
        if (validC1 == 1U) {
            ProcessNdc1hwc0SmallCLinearGroupTile(cur, context.row, rows, validC1, static_cast<uint32_t>(activeChannels),
                                                 static_cast<uint32_t>(block), outW);
        } else {
            ProcessNdc1hwc0SmallCGroupTile(cur, context.nIdx, context.od, cBase, static_cast<uint32_t>(context.oh),
                                           rows, static_cast<uint32_t>(activeChannels), static_cast<uint32_t>(block),
                                           outW);
        }
        cur += static_cast<uint64_t>(rows) * rowElements;
    }

    __aicore__ inline void ProcessNdc1hwc0SmallCGroup()
    {
        uint64_t block = 0U, validC1 = 0U, validOut = 0U;
        uint64_t outOffset = 0U, outCount = 0U, outEnd = 0U;
        uint32_t outW = 0U, rowElements = 0U;
        if (!InitNdc1hwc0GroupRange(block, validC1, validOut, outW, rowElements, outOffset, outCount, outEnd)) {
            return;
        }
        if (!HasNdc1hwc0CoreWork(outCount, validOut)) {
            return;
        }

        uint64_t cur = outOffset;
        while (cur < outEnd) {
            ProcessNdc1hwc0SmallCGroupStep(cur, outEnd, block, validC1, outW, rowElements);
        }
        if (validOut < tiling_->totalOut) {
            CopyOutZeroRangeByCore(validOut, tiling_->totalOut - validOut);
        }
    }

    __aicore__ inline uint32_t Ndc1hwc0SmallCCompactStride(uint32_t cCount, uint32_t outW) const
    {
        if (tiling_->dataFormat == FORMAT_NDHWC_VALUE) {
            return AlignToVector(outW * AlignToVector(cCount) + 1U);
        }
        return AlignToVector(cCount * AlignToVector(outW) + 1U);
    }

    __aicore__ inline void ProcessNdc1hwc0SmallCLinearGroupTile(uint64_t outputOffset, uint64_t startRow, uint32_t rows,
                                                                uint64_t validC1, uint32_t cCount, uint32_t block,
                                                                uint32_t outW)
    {
        const uint32_t rowElements = outW * block;
        const uint32_t rowStride = Ndc1hwc0SmallCCompactStride(cCount, outW);
        LocalTensor<T> compactLocal = calcBuf_.Get<T>();
        LocalTensor<T> outLocal = maskBuf_.Get<T>();
        LocalTensor<T> scratchLocal = tmpBuf_.Get<T>();
        LocalTensor<uint32_t> offsetLocal = scratchLocal.template ReinterpretCast<uint32_t>();
        Duplicate(compactLocal, NegInfValue(), rows * rowStride);
        PipeBarrier<PIPE_V>();

        uint32_t doneRows = 0U;
        while (doneRows < rows) {
            int64_t nIdx = 0;
            int64_t od = 0;
            int64_t c1Idx = 0;
            int64_t oh = 0;
            DecodeNdc1hwc0Row(startRow + doneRows, validC1, nIdx, od, c1Idx, oh);
            const int64_t cBase = c1Idx * static_cast<int64_t>(block);
            uint32_t runRows = static_cast<uint32_t>(tiling_->outH - oh);
            if (runRows > rows - doneRows) {
                runRows = rows - doneRows;
            }
            LocalTensor<T> runLocal = compactLocal[static_cast<uint64_t>(doneRows) * rowStride];
            if (tiling_->dataFormat == FORMAT_NDHWC_VALUE) {
                FillNdc1hwc0SmallCNdhwcGroupRows(runLocal, nIdx, od, cBase, static_cast<uint32_t>(oh), runRows, cCount,
                                                 outW, rowStride);
            } else {
                FillNdc1hwc0SmallCNcdhwGroupRows(runLocal, nIdx, od, cBase, static_cast<uint32_t>(oh), runRows, cCount,
                                                 outW, rowStride);
            }
            doneRows += runRows;
        }

        if (tiling_->dataFormat == FORMAT_NDHWC_VALUE) {
            const uint32_t alignedC = AlignToVector(cCount);
            ScatterNdc1hwc0CompactRowsChecked(outLocal, compactLocal, offsetLocal, rows, rowStride, cCount, block, outW,
                                              alignedC, 1U, outW * alignedC);
        } else {
            ScatterNdc1hwc0SmallCNcdhwRows(outLocal, compactLocal, offsetLocal, rows, rowStride, cCount, block, outW);
        }
        CopyOutVector(outputOffset, outLocal, rows * rowElements);
    }

    __aicore__ inline void ProcessNdc1hwc0SmallCGroupTile(uint64_t outputOffset, int64_t nIdx, int64_t od,
                                                          int64_t cBase, uint32_t ohStart, uint32_t rows,
                                                          uint32_t cCount, uint32_t block, uint32_t outW)
    {
        const uint32_t rowElements = outW * block;
        const uint32_t rowStride = Ndc1hwc0SmallCCompactStride(cCount, outW);
        LocalTensor<T> compactLocal = calcBuf_.Get<T>();
        LocalTensor<T> outLocal = maskBuf_.Get<T>();
        LocalTensor<T> scratchLocal = tmpBuf_.Get<T>();
        LocalTensor<uint32_t> offsetLocal = scratchLocal.template ReinterpretCast<uint32_t>();
        Duplicate(compactLocal, NegInfValue(), rows * rowStride);
        PipeBarrier<PIPE_V>();
        if (tiling_->dataFormat == FORMAT_NDHWC_VALUE) {
            FillNdc1hwc0SmallCNdhwcGroupRows(compactLocal, nIdx, od, cBase, ohStart, rows, cCount, outW, rowStride);
            const uint32_t alignedC = AlignToVector(cCount);
            ScatterNdc1hwc0CompactRowsChecked(outLocal, compactLocal, offsetLocal, rows, rowStride, cCount, block, outW,
                                              alignedC, 1U, outW * alignedC);
        } else {
            FillNdc1hwc0SmallCNcdhwGroupRows(compactLocal, nIdx, od, cBase, ohStart, rows, cCount, outW, rowStride);
            ScatterNdc1hwc0SmallCNcdhwRows(outLocal, compactLocal, offsetLocal, rows, rowStride, cCount, block, outW);
        }
        CopyOutVector(outputOffset, outLocal, rows * rowElements);
    }

    __aicore__ inline void ScatterNdc1hwc0SmallCNcdhwRows(LocalTensor<T> outLocal, LocalTensor<T> compactLocal,
                                                          LocalTensor<uint32_t> offsetLocal, uint32_t rows,
                                                          uint32_t rowStride, uint32_t cCount, uint32_t block,
                                                          uint32_t outW)
    {
        const uint32_t alignedW = AlignToVector(outW);
        const uint32_t zeroIndex = cCount * alignedW;
        if (!ScatterNdc1hwc0CompactRowsReuseRowOffset(outLocal, compactLocal, offsetLocal, rows, rowStride, cCount,
                                                      block, outW, 1U, alignedW, zeroIndex)) {
            ScatterNdc1hwc0CompactRowsChecked(outLocal, compactLocal, offsetLocal, rows, rowStride, cCount, block, outW,
                                              1U, alignedW, zeroIndex);
        }
    }

    __aicore__ inline void FillNdc1hwc0SmallCNdhwcGroupRows(LocalTensor<T> compactLocal, int64_t nIdx, int64_t od,
                                                            int64_t cBase, uint32_t ohStart, uint32_t rows,
                                                            uint32_t cCount, uint32_t outW, uint32_t rowStride)
    {
        const uint32_t alignedC = AlignToVector(cCount);
        for (uint32_t rowIdx = 0; rowIdx < rows; ++rowIdx) {
            LocalTensor<T> accRow = compactLocal[static_cast<uint64_t>(rowIdx) * rowStride];
            const int64_t oh = static_cast<int64_t>(ohStart + rowIdx);
            ReduceNdc1hwc0NdhwcRowWindow(accRow, nIdx, od, cBase, oh, cCount, alignedC);
        }
    }

    __aicore__ inline void FillNdc1hwc0SmallCNcdhwGroupRows(LocalTensor<T> compactLocal, int64_t nIdx, int64_t od,
                                                            int64_t cBase, uint32_t ohStart, uint32_t rows,
                                                            uint32_t cCount, uint32_t outW, uint32_t rowStride)
    {
        const uint32_t alignedW = AlignToVector(outW);
        for (uint32_t rowIdx = 0; rowIdx < rows; ++rowIdx) {
            LocalTensor<T> accRow = compactLocal[static_cast<uint64_t>(rowIdx) * rowStride];
            const int64_t oh = static_cast<int64_t>(ohStart + rowIdx);
            ReduceNdc1hwc0NcdhwRowWindow(accRow, nIdx, od, cBase, oh, cCount, outW, alignedW);
        }
    }

    __aicore__ inline void ProcessNdc1hwc0RowVector()
    {
        Ndc1hwc0LinearRangeContext range{};
        if (!InitNdc1hwc0ActiveLinearRange(range)) {
            return;
        }

        const uint64_t validEnd = range.outEnd;
        uint64_t cur = range.outOffset;
        while (cur < validEnd) {
            const uint64_t rowOffset = cur % range.rowElements;
            if (rowOffset != 0U || validEnd - cur < range.rowElements) {
                cur += ProcessNdc1hwc0PartialValidRange(cur, validEnd, range.block, range.rowElements, range.validC1);
                continue;
            }
            const uint32_t rowElements32 = static_cast<uint32_t>(range.rowElements);
            uint32_t maxTileRows = OUTPUT_TILE_NUM / rowElements32;
            if (maxTileRows == 0U) {
                maxTileRows = 1U;
            }
            const uint64_t remainRows64 = (validEnd - cur) / range.rowElements;
            uint32_t tileRows = remainRows64 > maxTileRows ? maxTileRows : static_cast<uint32_t>(remainRows64);
            if (tileRows == 0U) {
                ProcessNdc1hwc0RowVectorByRow(cur / range.rowElements, cur, range.block, range.validC1);
                cur += range.rowElements;
                continue;
            }
            ProcessNdc1hwc0RowVectorTile(cur / range.rowElements, tileRows, cur, range.block, range.validC1);
            cur += static_cast<uint64_t>(tileRows) * range.rowElements;
        }
        if (range.validOut < tiling_->totalOut) {
            CopyOutZeroRangeByCore(range.validOut, tiling_->totalOut - range.validOut);
        }
    }

    __aicore__ inline bool CanUseNdc1hwc0InputOutputRowPath() const
    {
        const uint64_t inBlock = InputNdc1hwc0Block();
        const uint64_t outBlock = Ndc1hwc0Block();
        const uint64_t validC1 = Ndc1hwc0ValidC1(outBlock);
        if (!HasValidNdc1hwc0InputOutputRows(inBlock, outBlock, validC1)) {
            return false;
        }
        const uint64_t rowElements = static_cast<uint64_t>(tiling_->outW) * outBlock;
        if (rowElements == 0U || rowElements > OUTPUT_TILE_NUM) {
            return false;
        }
        return outBlock <= 32U && tiling_->kD > 0 && tiling_->kH > 0 && tiling_->kW > 0 && tiling_->sD > 0 &&
               tiling_->sH > 0 && tiling_->sW > 0 && tiling_->dilationD > 0 && tiling_->dilationH > 0 &&
               tiling_->dilationW > 0;
    }

    __aicore__ inline bool HasValidNdc1hwc0InputOutputRows(uint64_t inBlock, uint64_t outBlock, uint64_t validC1) const
    {
        return inBlock > 0U && outBlock > 0U && inBlock == outBlock && validC1 > 0U && tiling_->outW > 0 &&
               tiling_->outH > 0 && tiling_->outD > 0 && tiling_->inputC1 > 0 && tiling_->c > 0;
    }

    __aicore__ inline bool CanUseNdc1hwc0InputOutputK1DirectPath() const
    {
        const uint64_t inBlock = InputNdc1hwc0Block();
        const uint64_t outBlock = Ndc1hwc0Block();
        const uint64_t validC1 = Ndc1hwc0ValidC1(outBlock);
        if (!HasNdc1hwc0InputOutputK1Layout(inBlock, outBlock, validC1) || !HasNdhwcK1UnitPool()) {
            return false;
        }
        if (tiling_->outD != tiling_->inD || tiling_->outH != tiling_->inH || tiling_->outW != tiling_->inW ||
            tiling_->outW <= 0) {
            return false;
        }
        const uint64_t rowElements = static_cast<uint64_t>(tiling_->outW) * outBlock;
        return rowElements > 0U && rowElements <= OUTPUT_TILE_NUM && rowElements <= INPUT_TILE_NUM;
    }

    __aicore__ inline bool CanUseNdc1hwc0InputOutputK1IdentityCopyPath() const
    {
        if (!CanUseNdc1hwc0InputOutputK1DirectPath()) {
            return false;
        }
        const uint64_t block = InputNdc1hwc0Block();
        const uint64_t inputElements = static_cast<uint64_t>(tiling_->n) * static_cast<uint64_t>(tiling_->inD) *
                                       static_cast<uint64_t>(tiling_->inputC1) * static_cast<uint64_t>(tiling_->inH) *
                                       static_cast<uint64_t>(tiling_->inW) * block;
        return tiling_->outputD == tiling_->inD && tiling_->outputH == tiling_->inH &&
               tiling_->outputW == tiling_->inW && tiling_->outputC1 == tiling_->inputC1 &&
               tiling_->outputC0 == static_cast<int64_t>(block) && inputElements == tiling_->totalOut;
    }

    __aicore__ inline bool HasMatchingNdc1hwc0InputOutputSpatialStorage() const
    {
        return tiling_->outputD == tiling_->inD && tiling_->outputH == tiling_->inH &&
               tiling_->outputW == tiling_->inW && tiling_->outD == tiling_->inD && tiling_->outH == tiling_->inH &&
               tiling_->outW == tiling_->inW;
    }

    __aicore__ inline bool CanUseNdc1hwc0InputOutputK1PackedPlanePath() const
    {
        if constexpr (!AscendC::Std::is_same<T, float>::value) {
            return false;
        }
        const uint64_t inputBlock = InputNdc1hwc0Block();
        const uint64_t outputBlock = Ndc1hwc0Block();
        const uint64_t inputCapacity = static_cast<uint64_t>(tiling_->inputC1) * inputBlock;
        const uint64_t outputValidC1 = Ndc1hwc0ValidC1(outputBlock);
        if (tiling_->inputLayout != INPUT_LAYOUT_NDC1HWC0_VALUE ||
            tiling_->outputLayout != OUTPUT_LAYOUT_NDC1HWC0_VALUE || !HasNdhwcK1UnitPool() ||
            !HasMatchingNdc1hwc0InputOutputSpatialStorage() || inputBlock * sizeof(T) != UB_BLOCK_BYTES ||
            tiling_->inputC1 <= 0 || outputValidC1 == 0U || outputBlock < inputBlock ||
            outputBlock % inputBlock != 0U || outputValidC1 * outputBlock < inputCapacity ||
            !IsNdc1hwc0CompactPrefix(outputBlock, outputValidC1)) {
            return false;
        }
        const uint64_t spatial = static_cast<uint64_t>(tiling_->inH) * static_cast<uint64_t>(tiling_->inW);
        return spatial > 0U && spatial * inputCapacity <= INPUT_TILE_NUM &&
               spatial * outputValidC1 * outputBlock <= OUTPUT_TILE_NUM;
    }

    __aicore__ inline void ProcessNdc1hwc0InputOutputK1PackedPlane()
    {
        const uint32_t inputBlock = static_cast<uint32_t>(InputNdc1hwc0Block());
        const uint32_t outputBlock = static_cast<uint32_t>(Ndc1hwc0Block());
        const uint32_t outputValidC1 = static_cast<uint32_t>(Ndc1hwc0ValidC1(outputBlock));
        const uint32_t spatial = static_cast<uint32_t>(tiling_->inH * tiling_->inW);
        const uint32_t inputPlaneElements = spatial * static_cast<uint32_t>(tiling_->inputC1) * inputBlock;
        const uint32_t outputPlaneElements = spatial * outputValidC1 * outputBlock;
        const uint32_t totalPlanes = static_cast<uint32_t>(tiling_->n * tiling_->inD);
        const uint32_t worker = GetBlockIdx();
        const uint32_t workerDim = ActiveBlockDim();
        for (uint32_t plane = worker; plane < totalPlanes; plane += workerDim) {
            CopyInVector(static_cast<uint64_t>(plane) * inputPlaneElements, inputPlaneElements);
            LocalTensor<T> xLocal = xInQue_.DeQue<T>();
            LocalTensor<T> outLocal = maskBuf_.Get<T>();
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Duplicate(outLocal, ZeroValue(), outputPlaneElements);
            PipeBarrier<PIPE_V>();
            const uint16_t dstStride = static_cast<uint16_t>((outputBlock - inputBlock) * sizeof(T) / UB_BLOCK_BYTES);
            const DataCopyParams packParams{static_cast<uint16_t>(spatial), 1U, 0U, dstStride};
            for (uint32_t c1 = 0U; c1 < static_cast<uint32_t>(tiling_->inputC1); ++c1) {
                const uint32_t channelBase = c1 * inputBlock;
                const uint32_t outputC1 = channelBase / outputBlock;
                const uint32_t outputC0 = channelBase - outputC1 * outputBlock;
                const uint32_t outputBase = outputC1 * spatial * outputBlock + outputC0;
                DataCopy(outLocal[outputBase], xLocal[static_cast<uint64_t>(c1) * spatial * inputBlock], packParams);
            }
            DataSyncBarrier<MemDsbT::UB>();
            CopyOutVector(static_cast<uint64_t>(plane) * outputPlaneElements, outLocal, outputPlaneElements);
            xInQue_.FreeTensor(xLocal);
        }
        const uint64_t validOut = static_cast<uint64_t>(totalPlanes) * outputPlaneElements;
        if (validOut < tiling_->totalOut) {
            CopyOutZeroRangeByCore(validOut, tiling_->totalOut - validOut);
        }
    }

    __aicore__ inline void ProcessNdc1hwc0InputOutputK1IdentityCopy()
    {
        const uint64_t outOffset = CoreStartOffset();
        const uint64_t outCount = CoreElementCount(outOffset);
        const uint64_t outEnd = outOffset + outCount;
        uint64_t cur = outOffset;
        while (cur < outEnd) {
            const uint64_t remain = outEnd - cur;
            const uint32_t count = remain > INPUT_TILE_NUM ? INPUT_TILE_NUM : static_cast<uint32_t>(remain);
            CopyInVector(cur, count);
            LocalTensor<T> xLocal = xInQue_.DeQue<T>();
            CopyOutVectorFromMte2(cur, xLocal, count);
            xInQue_.FreeTensor(xLocal);
            cur += count;
        }
    }

    __aicore__ inline bool HasNdc1hwc0InputOutputK1Layout(uint64_t inBlock, uint64_t outBlock, uint64_t validC1) const
    {
        return tiling_->inputLayout == INPUT_LAYOUT_NDC1HWC0_VALUE &&
               tiling_->outputLayout == OUTPUT_LAYOUT_NDC1HWC0_VALUE && inBlock > 0U && outBlock > 0U &&
               inBlock == outBlock && validC1 > 0U && static_cast<uint64_t>(tiling_->inputC1) == validC1 &&
               IsNdc1hwc0CompactStorage(outBlock, validC1) && outBlock <= 32U;
    }

    __aicore__ inline void ProcessNdc1hwc0InputOutputK1Direct()
    {
        uint64_t block = 0U, validC1 = 0U, validOut = 0U;
        uint64_t outOffset = 0U, outCount = 0U, outEnd = 0U;
        uint32_t outW = 0U, rowElements = 0U;
        if (!InitNdc1hwc0GroupRange(block, validC1, validOut, outW, rowElements, outOffset, outCount, outEnd)) {
            return;
        }
        uint64_t cur = outOffset;
        while (cur < outEnd) {
            const uint64_t rowOffset = cur % rowElements;
            if (rowOffset != 0U || outEnd - cur < rowElements) {
                cur += ProcessNdc1hwc0PartialValidRange(cur, outEnd, block, rowElements, validC1);
                continue;
            }
            const uint64_t remainRows64 = (outEnd - cur) / rowElements;
            uint32_t tileRows = remainRows64 > static_cast<uint64_t>(OUTPUT_TILE_NUM / rowElements) ?
                                    OUTPUT_TILE_NUM / rowElements :
                                    static_cast<uint32_t>(remainRows64);
            const uint32_t maxInputRows = INPUT_TILE_NUM / rowElements;
            if (tileRows > maxInputRows) {
                tileRows = maxInputRows;
            }
            if (tileRows == 0U) {
                tileRows = 1U;
            }
            ProcessNdc1hwc0InputOutputK1DirectTile(cur / rowElements, cur, tileRows, static_cast<uint32_t>(block),
                                                   validC1, outW);
            cur += static_cast<uint64_t>(tileRows) * rowElements;
        }
        if (validOut < tiling_->totalOut) {
            CopyOutZeroRangeByCore(validOut, tiling_->totalOut - validOut);
        }
    }

    __aicore__ inline void ProcessNdc1hwc0InputOutputK1DirectTile(uint64_t startRow, uint64_t outputOffset,
                                                                  uint32_t tileRows, uint32_t block, uint64_t validC1,
                                                                  uint32_t outW)
    {
        const uint32_t rowElements = outW * block;
        int64_t nIdx = 0;
        int64_t od = 0;
        int64_t c1Idx = 0;
        int64_t oh = 0;
        DecodeNdc1hwc0Row(startRow, validC1, nIdx, od, c1Idx, oh);
        const uint64_t inputOffset = InputOffset(nIdx, od, oh, 0, c1Idx * static_cast<int64_t>(block));
        const uint32_t totalCount = tileRows * rowElements;
        CopyInVector(inputOffset, totalCount);
        LocalTensor<T> xLocal = xInQue_.DeQue<T>();
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        for (uint32_t rowIdx = 0; rowIdx < tileRows; ++rowIdx) {
            const uint64_t row = startRow + rowIdx;
            if (validC1 == 0U) {
                xInQue_.FreeTensor(xLocal);
                return;
            }
            const int64_t rowC1 = static_cast<int64_t>(row / static_cast<uint64_t>(tiling_->outH) % validC1);
            const uint32_t active = Ndc1hwc0InputActiveChannels(rowC1, block);
            ZeroNdc1hwc0Tail(xLocal[static_cast<uint64_t>(rowIdx) * rowElements], outW, block, active);
        }
        CopyOutVector(outputOffset, xLocal, totalCount);
        xInQue_.FreeTensor(xLocal);
    }

    __aicore__ inline void ProcessNdc1hwc0InputOutputRow()
    {
        Ndc1hwc0LinearRangeContext range{};
        if (!InitNdc1hwc0ActiveLinearRange(range)) {
            return;
        }
        uint64_t cur = range.outOffset;
        while (cur < range.outEnd) {
            const uint64_t rowOffset = cur % range.rowElements;
            if (rowOffset != 0U || range.outEnd - cur < range.rowElements) {
                uint64_t curCount64 = range.rowElements - rowOffset;
                if (curCount64 > range.outEnd - cur) {
                    curCount64 = range.outEnd - cur;
                }
                const uint32_t curCount = curCount64 > OUTPUT_TILE_NUM ? OUTPUT_TILE_NUM :
                                                                         static_cast<uint32_t>(curCount64);
                LocalTensor<T> yLocal = yOutQue_.AllocTensor<T>();
                FillNdc1hwc0ValidTile(yLocal, cur, curCount, range.block, range.rowElements, range.validC1);
                yOutQue_.EnQue(yLocal);
                LocalTensor<T> yOut = yOutQue_.DeQue<T>();
                DataCopyExtParams copyParams{1, static_cast<uint32_t>(curCount * sizeof(T)), 0, 0, 0};
                DataCopyPad(yGm_[cur], yOut, copyParams);
                yOutQue_.FreeTensor(yOut);
                cur += curCount;
                continue;
            }
            ProcessNdc1hwc0InputOutputRowByRow(cur / range.rowElements, cur, range.block, range.validC1);
            cur += range.rowElements;
        }
        if (range.validOut < tiling_->totalOut) {
            CopyOutZeroRangeByCore(range.validOut, tiling_->totalOut - range.validOut);
        }
    }

    __aicore__ inline void ProcessNdc1hwc0InputOutputRowByRow(uint64_t row, uint64_t outputOffset, uint64_t block,
                                                              uint64_t validC1)
    {
        int64_t nIdx = 0;
        int64_t od = 0;
        int64_t c1Idx = 0;
        int64_t oh = 0;
        DecodeNdc1hwc0Row(row, validC1, nIdx, od, c1Idx, oh);
        if (CanUseNdc1hwc0InputStride2VectorPath()) {
            ProcessNdc1hwc0InputStride2VectorRow(outputOffset, nIdx, od, c1Idx, oh, static_cast<uint32_t>(block));
            return;
        }
        const int64_t cBase = c1Idx * static_cast<int64_t>(block);
        const uint32_t rowElements = static_cast<uint32_t>(static_cast<uint64_t>(tiling_->outW) * block);
        LocalTensor<T> yLocal = maskBuf_.Get<T>();
        for (uint32_t ow = 0; ow < static_cast<uint32_t>(tiling_->outW); ++ow) {
            for (uint32_t c0 = 0; c0 < static_cast<uint32_t>(block); ++c0) {
                const int64_t cIdx = cBase + static_cast<int64_t>(c0);
                const uint32_t outIdx = ow * static_cast<uint32_t>(block) + c0;
                if (cIdx >= tiling_->c) {
                    yLocal.SetValue(outIdx, ZeroValue());
                } else {
                    yLocal.SetValue(outIdx, ComputeValueAt(nIdx, od, oh, static_cast<int64_t>(ow), cIdx));
                }
            }
        }
        CopyOutVector(outputOffset, yLocal, rowElements);
    }

    __aicore__ inline bool CanUseNdc1hwc0InputStride2VectorPath() const
    {
        const uint64_t block = Ndc1hwc0Block();
        if (block == 0U || block != InputNdc1hwc0Block() || tiling_->outW <= 0 || tiling_->inW <= 0 ||
            tiling_->outH <= 0 || tiling_->inputC1 <= 0) {
            return false;
        }
        if (!IsPool2Stride2NoPad() || tiling_->outD != (tiling_->inD + 1) / 2 ||
            tiling_->outH != (tiling_->inH + 1) / 2 || tiling_->outW != (tiling_->inW + 1) / 2) {
            return false;
        }
        const uint32_t inputRowCount = static_cast<uint32_t>(tiling_->inW) * static_cast<uint32_t>(block);
        const uint32_t paddedInputRowCount = static_cast<uint32_t>(tiling_->outW) * 2U * static_cast<uint32_t>(block);
        const uint32_t outputRowCount = static_cast<uint32_t>(tiling_->outW) * static_cast<uint32_t>(block);
        return inputRowCount > 0U && paddedInputRowCount >= inputRowCount &&
               paddedInputRowCount - inputRowCount <= 255U && outputRowCount > 0U &&
               paddedInputRowCount <= INPUT_TILE_NUM && outputRowCount <= OUTPUT_TILE_NUM;
    }

    __aicore__ inline uint32_t Ndc1hwc0InputActiveChannels(int64_t c1Idx, uint32_t block) const
    {
        const int64_t cBase = c1Idx * static_cast<int64_t>(block);
        const int64_t activeChannels = Ndc1hwc0ActiveChannels(cBase, block);
        return static_cast<uint32_t>(activeChannels);
    }

    __aicore__ inline void ZeroNdc1hwc0Tail(LocalTensor<T> yLocal, uint32_t outW, uint32_t block, uint32_t active)
    {
        if (active >= block) {
            return;
        }
        const uint32_t tail = block - active;
        const uint32_t repeatStride = block * sizeof(T) / UB_BLOCK_BYTES;
        constexpr uint32_t vectorElements = 256U / sizeof(T);
        if (tail > vectorElements || repeatStride == 0U || repeatStride > 255U ||
            active * sizeof(T) % UB_BLOCK_BYTES != 0U) {
            for (uint32_t ow = 0; ow < outW; ++ow) {
                const uint32_t base = ow * block;
                for (uint32_t c0 = active; c0 < block; ++c0) {
                    yLocal.SetValue(base + c0, ZeroValue());
                }
            }
            return;
        }
        uint32_t point = 0U;
        while (point < outW) {
            const uint32_t remaining = outW - point;
            const uint8_t repeats = static_cast<uint8_t>(remaining > 255U ? 255U : remaining);
            Duplicate<T, false>(yLocal[static_cast<uint64_t>(point) * block + active], ZeroValue(), tail, repeats, 1U,
                                static_cast<uint8_t>(repeatStride));
            point += repeats;
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ProcessNdc1hwc0InputStride2VectorRow(uint64_t outputOffset, int64_t nIdx, int64_t od,
                                                                int64_t c1Idx, int64_t oh, uint32_t block)
    {
        const uint32_t outW = static_cast<uint32_t>(tiling_->outW);
        const uint32_t inputRowCount = static_cast<uint32_t>(tiling_->inW) * block;
        const uint32_t paddedInputRowCount = outW * 2U * block;
        const uint32_t outputRowCount = outW * block;
        const int64_t d0 = od * tiling_->sD;
        const int64_t d1 = d0 + 1 < tiling_->inD ? d0 + 1 : d0;
        const int64_t h0 = oh * tiling_->sH;
        const int64_t h1 = h0 + 1 < tiling_->inH ? h0 + 1 : h0;
        const int64_t cBase = c1Idx * static_cast<int64_t>(block);
        LocalTensor<T> accLocal = calcBuf_.Get<T>();
        LocalTensor<T> tmpLocal = tmpBuf_.Get<T>();

        CopyInVectorPad(InputOffset(nIdx, d0, h0, 0, cBase), inputRowCount, paddedInputRowCount);
        LocalTensor<T> inputLocal = xInQue_.DeQue<T>();
        Max(accLocal, inputLocal, inputLocal, paddedInputRowCount);
        PipeBarrier<PIPE_V>();
        xInQue_.FreeTensor(inputLocal);
        AccumulateNdc1hwc0PhysicalPool2Row(accLocal, nIdx, d0, h1, cBase, inputRowCount, paddedInputRowCount);
        AccumulateNdc1hwc0PhysicalPool2Row(accLocal, nIdx, d1, h0, cBase, inputRowCount, paddedInputRowCount);
        AccumulateNdc1hwc0PhysicalPool2Row(accLocal, nIdx, d1, h1, cBase, inputRowCount, paddedInputRowCount);
        CompressNdhwcStride2WPairNoBarrier(tmpLocal, accLocal, outW, block);
        ZeroNdc1hwc0Tail(tmpLocal, outW, block, Ndc1hwc0InputActiveChannels(c1Idx, block));
        CopyOutVector(outputOffset, tmpLocal, outputRowCount);
    }

    __aicore__ inline void AccumulateNdc1hwc0PhysicalPool2Row(LocalTensor<T> accLocal, int64_t nIdx, int64_t dIdx,
                                                              int64_t hIdx, int64_t cBase, uint32_t inputRowCount,
                                                              uint32_t paddedInputRowCount)
    {
        CopyInVectorPad(InputOffset(nIdx, dIdx, hIdx, 0, cBase), inputRowCount, paddedInputRowCount);
        LocalTensor<T> inputLocal = xInQue_.DeQue<T>();
        Max(accLocal, accLocal, inputLocal, paddedInputRowCount);
        PipeBarrier<PIPE_V>();
        xInQue_.FreeTensor(inputLocal);
    }

    __aicore__ inline bool CanUseNdc1hwc0InputOutputPool2PackedC1PlanePath() const
    {
        if constexpr (!AscendC::Std::is_same<T, float>::value) {
            return false;
        }
        const uint64_t inputBlock = InputNdc1hwc0Block();
        const uint64_t outputBlock = Ndc1hwc0Block();
        const uint64_t outputValidC1 = Ndc1hwc0ValidC1(outputBlock);
        const uint64_t inputCapacity = static_cast<uint64_t>(tiling_->inputC1) * inputBlock;
        if (!HasNdc1hwc0InputOutputLayout() || !IsPool2Stride2NoPad() || inputBlock == 0U || outputBlock == 0U ||
            inputBlock * sizeof(T) != UB_BLOCK_BYTES || tiling_->inputC1 <= 0 || outputValidC1 == 0U ||
            outputBlock < inputBlock || outputBlock % inputBlock != 0U || outputValidC1 * outputBlock < inputCapacity ||
            !IsNdc1hwc0CompactPrefix(outputBlock, outputValidC1) || tiling_->outputD != tiling_->outD ||
            tiling_->outputH != tiling_->outH || tiling_->outputW != tiling_->outW || tiling_->inD <= 0 ||
            tiling_->inH <= 0 || tiling_->inW <= 0 || tiling_->outD != (tiling_->inD + 1) / 2 ||
            tiling_->outH != (tiling_->inH + 1) / 2 || tiling_->outW != (tiling_->inW + 1) / 2) {
            return false;
        }
        const uint64_t inputC1PerOutput = outputBlock / inputBlock;
        const uint64_t inputPlane = static_cast<uint64_t>(tiling_->inH) * tiling_->inW * inputBlock;
        const uint64_t selectedDepth = inputC1PerOutput * inputPlane;
        const uint64_t inputRow = static_cast<uint64_t>(tiling_->inW) * inputBlock;
        const uint64_t outputPlane = static_cast<uint64_t>(tiling_->outH) * tiling_->outW * outputBlock;
        return inputC1PerOutput > 1U && selectedDepth * 2U <= INPUT_TILE_NUM && inputRow * 2U <= OUTPUT_TILE_NUM &&
               outputPlane > 0U && outputPlane <= OUTPUT_TILE_NUM && tiling_->inW / 2 <= 255;
    }

    __aicore__ inline void ReduceNdc1hwc0Pool2PackedC1Row(LocalTensor<T> depthLocal, LocalTensor<T> reduceLocal,
                                                          LocalTensor<T> pairLocal, LocalTensor<T> outputRowLocal,
                                                          uint32_t localC1, uint32_t inputPlane, uint32_t inputRow,
                                                          uint32_t inputBlock, uint32_t outputBlock,
                                                          uint32_t inputC1Count, uint32_t oh, uint64_t depth1Base)
    {
        if (localC1 >= inputC1Count) {
            return;
        }
        const uint32_t h0 = oh * 2U;
        const uint32_t h1 = h0 + 1U < static_cast<uint32_t>(tiling_->inH) ? h0 + 1U : h0;
        const uint64_t c1Base = static_cast<uint64_t>(localC1) * inputPlane;
        const uint64_t offset0 = c1Base + static_cast<uint64_t>(h0) * inputRow;
        const uint64_t offset1 = c1Base + static_cast<uint64_t>(h1) * inputRow;
        Max(reduceLocal, depthLocal[offset0], depthLocal[offset1], inputRow);
        Max(pairLocal, depthLocal[depth1Base + offset0], depthLocal[depth1Base + offset1], inputRow);
        PipeBarrier<PIPE_V>();
        Max(reduceLocal, reduceLocal, pairLocal, inputRow);
        PipeBarrier<PIPE_V>();
        const uint32_t pairedWidth = static_cast<uint32_t>(tiling_->inW) / 2U;
        const uint32_t channelOffset = localC1 * inputBlock;
        if (pairedWidth > 0U) {
            const uint8_t dstStride = static_cast<uint8_t>(outputBlock * sizeof(T) / UB_BLOCK_BYTES);
            const uint8_t srcStride = static_cast<uint8_t>(2U * inputBlock * sizeof(T) / UB_BLOCK_BYTES);
            const BinaryRepeatParams widthParams{1U, 1U, 1U, dstStride, srcStride, srcStride};
            Max(outputRowLocal[channelOffset], reduceLocal, reduceLocal[inputBlock], inputBlock,
                static_cast<uint8_t>(pairedWidth), widthParams);
        }
        if ((tiling_->inW & 1) != 0) {
            const uint64_t outputOffset = static_cast<uint64_t>(tiling_->outW - 1) * outputBlock + channelOffset;
            const uint64_t inputOffset = static_cast<uint64_t>(tiling_->inW - 1) * inputBlock;
            Max(outputRowLocal[outputOffset], reduceLocal[inputOffset], reduceLocal[inputOffset], inputBlock);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ProcessNdc1hwc0InputOutputPool2PackedC1Planes()
    {
        const uint32_t inputBlock = static_cast<uint32_t>(InputNdc1hwc0Block());
        const uint32_t outputBlock = static_cast<uint32_t>(Ndc1hwc0Block());
        const uint32_t inputC1PerOutput = outputBlock / inputBlock;
        const uint32_t outputValidC1 = static_cast<uint32_t>(Ndc1hwc0ValidC1(outputBlock));
        const uint32_t inputRow = static_cast<uint32_t>(tiling_->inW) * inputBlock;
        const uint32_t inputPlane = static_cast<uint32_t>(tiling_->inH) * inputRow;
        const uint32_t inputDepth = static_cast<uint32_t>(tiling_->inputC1) * inputPlane;
        const uint32_t outputRow = static_cast<uint32_t>(tiling_->outW) * outputBlock;
        const uint32_t outputPlane = static_cast<uint32_t>(tiling_->outH) * outputRow;
        const uint32_t totalUnits = static_cast<uint32_t>(tiling_->n * tiling_->outD) * outputValidC1;
        const uint32_t worker = GetBlockIdx();
        const uint32_t workerDim = ActiveBlockDim();
        for (uint32_t unit = worker; unit < totalUnits; unit += workerDim) {
            const uint32_t outputC1 = unit % outputValidC1;
            const uint32_t planeUnit = unit / outputValidC1;
            const uint32_t od = planeUnit % static_cast<uint32_t>(tiling_->outD);
            const uint32_t nIdx = planeUnit / static_cast<uint32_t>(tiling_->outD);
            const uint32_t inputC1Start = outputC1 * inputC1PerOutput;
            const uint32_t inputC1Remain = static_cast<uint32_t>(tiling_->inputC1) - inputC1Start;
            const uint32_t inputC1Count = inputC1Remain < inputC1PerOutput ? inputC1Remain : inputC1PerOutput;
            const uint32_t selectedDepth = inputC1Count * inputPlane;
            const uint32_t d0 = od * 2U;
            const uint32_t d1 = d0 + 1U < static_cast<uint32_t>(tiling_->inD) ? d0 + 1U : d0;
            const uint64_t nBase = static_cast<uint64_t>(nIdx) * tiling_->inD * inputDepth;
            const uint64_t c1Base = static_cast<uint64_t>(inputC1Start) * inputPlane;
            LocalTensor<T> depthLocal = calcBuf_.Get<T>();
            DataCopy(depthLocal, xGm_[nBase + static_cast<uint64_t>(d0) * inputDepth + c1Base], selectedDepth);
            uint64_t depth1Base = 0U;
            if (d1 != d0) {
                depth1Base = selectedDepth;
                DataCopy(depthLocal[depth1Base], xGm_[nBase + static_cast<uint64_t>(d1) * inputDepth + c1Base],
                         selectedDepth);
            }
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            LocalTensor<T> reduceLocal = tmpBuf_.Get<T>();
            LocalTensor<T> pairLocal = reduceLocal[inputRow];
            LocalTensor<T> outputLocal = maskBuf_.Get<T>();
            Duplicate(outputLocal, ZeroValue(), outputPlane);
            PipeBarrier<PIPE_V>();
            for (uint32_t oh = 0U; oh < static_cast<uint32_t>(tiling_->outH); ++oh) {
                LocalTensor<T> outputRowLocal = outputLocal[static_cast<uint64_t>(oh) * outputRow];
                for (uint32_t localC1 = 0U; localC1 < inputC1Count; ++localC1) {
                    ReduceNdc1hwc0Pool2PackedC1Row(depthLocal, reduceLocal, pairLocal, outputRowLocal, localC1,
                                                   inputPlane, inputRow, inputBlock, outputBlock, inputC1Count, oh,
                                                   depth1Base);
                }
            }
            const uint64_t outputOffset = ((static_cast<uint64_t>(nIdx) * tiling_->outD + od) * outputValidC1 +
                                           outputC1) *
                                          outputPlane;
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopy(yGm_[outputOffset], outputLocal, outputPlane);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
        const uint64_t validOut = static_cast<uint64_t>(tiling_->n) * tiling_->outD * outputValidC1 * outputPlane;
        if (validOut < tiling_->totalOut) {
            CopyOutZeroRangeByCore(validOut, tiling_->totalOut - validOut);
        }
    }

    __aicore__ inline bool CanUseNdc1hwc0InputOutputPool2FullDepthPath() const
    {
        const uint64_t inputBlock = InputNdc1hwc0Block();
        const uint64_t storageC0 = Ndc1hwc0StorageC0();
        const uint64_t storageC1 = Ndc1hwc0StorageC1();
        if (tiling_->inputLayout != INPUT_LAYOUT_NDC1HWC0_VALUE ||
            tiling_->outputLayout != OUTPUT_LAYOUT_NDC1HWC0_VALUE || !IsPool2Stride2NoPad() || inputBlock == 0U ||
            inputBlock * sizeof(T) != UB_BLOCK_BYTES || storageC0 != inputBlock ||
            storageC1 < static_cast<uint64_t>(tiling_->inputC1) || tiling_->inH % 2 != 0 || tiling_->inW % 2 != 0 ||
            tiling_->outD != (tiling_->inD + 1) / 2 || tiling_->outH != tiling_->inH / 2 ||
            tiling_->outW != tiling_->inW / 2) {
            return false;
        }
        const uint64_t inputPlane = static_cast<uint64_t>(tiling_->inH) * tiling_->inW * inputBlock;
        const uint64_t inputVolume = static_cast<uint64_t>(tiling_->inD) * inputPlane;
        const uint64_t outputVolume = static_cast<uint64_t>(tiling_->outD) * tiling_->outH * tiling_->outW * inputBlock;
        constexpr uint64_t calcCapacity = AscendC::Std::is_same<T, half>::value ? INPUT_TILE_NUM * 2ULL :
                                                                                  INPUT_TILE_NUM;
        return inputVolume > 0U && inputVolume <= calcCapacity && outputVolume > 0U && outputVolume <= OUTPUT_TILE_NUM;
    }

    __aicore__ inline void LoadNdc1hwc0Pool2FullDepth(LocalTensor<T> inputLocal, uint32_t nIdx, uint32_t c1,
                                                      uint32_t inputPlane, uint32_t inputVolume)
    {
        const uint32_t inputC1 = static_cast<uint32_t>(tiling_->inputC1);
        const uint64_t inputDepth = static_cast<uint64_t>(inputC1) * inputPlane;
        const uint64_t inputOffset = static_cast<uint64_t>(nIdx) * tiling_->inD * inputDepth +
                                     static_cast<uint64_t>(c1) * inputPlane;
        const DataCopyExtParams inputCopy{static_cast<uint16_t>(tiling_->inD),
                                          static_cast<uint32_t>(inputPlane * sizeof(T)),
                                          static_cast<uint32_t>((inputC1 - 1U) * inputPlane * sizeof(T)), 0U, 0U};
        const DataCopyPadExtParams<T> noPad{false, 0U, 0U, ZeroValue()};
        DataCopyPad(inputLocal, xGm_[inputOffset], inputCopy, noPad);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }

    __aicore__ inline void ReduceNdc1hwc0Pool2FullDepth(LocalTensor<T> inputLocal, LocalTensor<T> heightLocal,
                                                        LocalTensor<T> outputLocal, uint32_t inputBlock,
                                                        uint32_t inputPlane, uint32_t inputRow, uint32_t outputRow)
    {
        for (uint32_t od = 0U; od < static_cast<uint32_t>(tiling_->outD); ++od) {
            const uint32_t d0 = od * 2U;
            const uint32_t d1 = d0 + 1U < static_cast<uint32_t>(tiling_->inD) ? d0 + 1U : d0;
            Max(inputLocal[static_cast<uint64_t>(od) * inputPlane], inputLocal[static_cast<uint64_t>(d0) * inputPlane],
                inputLocal[static_cast<uint64_t>(d1) * inputPlane], inputPlane);
        }
        PipeBarrier<PIPE_V>();
        for (uint32_t od = 0U; od < static_cast<uint32_t>(tiling_->outD); ++od) {
            LocalTensor<T> depthLocal = inputLocal[static_cast<uint64_t>(od) * inputPlane];
            for (uint32_t oh = 0U; oh < static_cast<uint32_t>(tiling_->outH); ++oh) {
                Max(heightLocal[(static_cast<uint64_t>(od) * tiling_->outH + oh) * inputRow],
                    depthLocal[static_cast<uint64_t>(oh) * 2U * inputRow],
                    depthLocal[(static_cast<uint64_t>(oh) * 2U + 1U) * inputRow], inputRow);
            }
        }
        PipeBarrier<PIPE_V>();
        for (uint32_t od = 0U; od < static_cast<uint32_t>(tiling_->outD); ++od) {
            for (uint32_t oh = 0U; oh < static_cast<uint32_t>(tiling_->outH); ++oh) {
                const uint64_t row = static_cast<uint64_t>(od) * tiling_->outH + oh;
                CompressNdhwcStride2WPairNoBarrier(outputLocal[row * outputRow], heightLocal[row * inputRow],
                                                   static_cast<uint32_t>(tiling_->outW), inputBlock);
            }
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void StoreNdc1hwc0Pool2FullDepth(LocalTensor<T> outputLocal, uint32_t nIdx, uint32_t c1,
                                                       uint32_t outputPlane)
    {
        const uint32_t storageC1 = static_cast<uint32_t>(Ndc1hwc0StorageC1());
        const uint64_t outputOffset = (static_cast<uint64_t>(nIdx) * tiling_->outD * storageC1 + c1) * outputPlane;
        const DataCopyExtParams outputCopy{static_cast<uint16_t>(tiling_->outD),
                                           static_cast<uint32_t>(outputPlane * sizeof(T)), 0U,
                                           static_cast<uint32_t>((storageC1 - 1U) * outputPlane * sizeof(T)), 0U};
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyPad(yGm_[outputOffset], outputLocal, outputCopy);
    }

    __aicore__ inline void ProcessNdc1hwc0InputOutputPool2FullDepth()
    {
        const uint32_t inputBlock = static_cast<uint32_t>(InputNdc1hwc0Block());
        const uint32_t inputRow = static_cast<uint32_t>(tiling_->inW) * inputBlock;
        const uint32_t inputPlane = static_cast<uint32_t>(tiling_->inH) * inputRow;
        const uint32_t inputVolume = static_cast<uint32_t>(tiling_->inD) * inputPlane;
        const uint32_t outputRow = static_cast<uint32_t>(tiling_->outW) * inputBlock;
        const uint32_t outputPlane = static_cast<uint32_t>(tiling_->outH) * outputRow;
        const uint32_t totalUnits = static_cast<uint32_t>(tiling_->n * tiling_->inputC1);
        const uint32_t worker = GetBlockIdx();
        const uint32_t workerDim = ActiveBlockDim();
        LocalTensor<T> inputLocal = calcBuf_.Get<T>();
        LocalTensor<T> heightLocal = tmpBuf_.Get<T>();
        LocalTensor<T> outputLocal = maskBuf_.Get<T>();
        for (uint32_t unit = worker; unit < totalUnits; unit += workerDim) {
            const uint32_t nIdx = unit / static_cast<uint32_t>(tiling_->inputC1);
            const uint32_t c1 = unit - nIdx * static_cast<uint32_t>(tiling_->inputC1);
            LoadNdc1hwc0Pool2FullDepth(inputLocal, nIdx, c1, inputPlane, inputVolume);
            ReduceNdc1hwc0Pool2FullDepth(inputLocal, heightLocal, outputLocal, inputBlock, inputPlane, inputRow,
                                         outputRow);
            StoreNdc1hwc0Pool2FullDepth(outputLocal, nIdx, c1, outputPlane);
        }
    }

    __aicore__ inline bool CanUseNdc1hwc0InputOutputPool2DepthGroupPath() const
    {
        if constexpr (AscendC::Std::is_same<T, float>::value) {
            return false;
        }
        const uint64_t inputBlock = InputNdc1hwc0Block();
        const uint64_t outputBlock = Ndc1hwc0Block();
        const uint64_t validC1 = Ndc1hwc0ValidC1(outputBlock);
        if (tiling_->inputLayout != INPUT_LAYOUT_NDC1HWC0_VALUE ||
            tiling_->outputLayout != OUTPUT_LAYOUT_NDC1HWC0_VALUE || !IsPool2Stride2NoPad() || inputBlock == 0U ||
            inputBlock * sizeof(T) != UB_BLOCK_BYTES || outputBlock < inputBlock || outputBlock % inputBlock != 0U ||
            validC1 == 0U || validC1 * outputBlock < static_cast<uint64_t>(tiling_->inputC1) * inputBlock ||
            tiling_->outD != (tiling_->inD + 1) / 2 || tiling_->outH != (tiling_->inH + 1) / 2 ||
            tiling_->outW != (tiling_->inW + 1) / 2) {
            return false;
        }
        const uint64_t inputDepth = static_cast<uint64_t>(tiling_->inputC1) * tiling_->inH * tiling_->inW * inputBlock;
        const uint64_t outputDepth = validC1 * tiling_->outH * tiling_->outW * outputBlock;
        const uint64_t tempNeed = inputDepth + static_cast<uint64_t>(tiling_->inputC1) * tiling_->outH *
                                                   (static_cast<uint64_t>(tiling_->outW) * 2U) * inputBlock;
        const uint64_t workerDim = ActiveBlockDim();
        if (tiling_->n <= 0 || tiling_->outD <= 0 || workerDim == 0U) {
            return false;
        }
        uint64_t groupsPerN = workerDim / static_cast<uint64_t>(tiling_->n);
        if (groupsPerN == 0U) {
            groupsPerN = 1U;
        }
        if (groupsPerN > static_cast<uint64_t>(tiling_->outD)) {
            groupsPerN = static_cast<uint64_t>(tiling_->outD);
        }
        const uint64_t maxOutputDepthCount = (static_cast<uint64_t>(tiling_->outD) + groupsPerN - 1U) / groupsPerN;
        const uint64_t maxInputDepthCount = maxOutputDepthCount * 2U > static_cast<uint64_t>(tiling_->inD) ?
                                                static_cast<uint64_t>(tiling_->inD) :
                                                maxOutputDepthCount * 2U;
        constexpr uint64_t calcCapacity = AscendC::Std::is_same<T, half>::value ? INPUT_TILE_NUM * 2ULL :
                                                                                  INPUT_TILE_NUM;
        const uint64_t outputCapacity = static_cast<uint64_t>(tiling_->outD) * tiling_->outH * tiling_->outW *
                                                    inputBlock >
                                                OUTPUT_TILE_NUM ?
                                            static_cast<uint64_t>(tiling_->outD) * tiling_->outH * tiling_->outW *
                                                inputBlock :
                                            OUTPUT_TILE_NUM;
        return inputDepth > 0U && maxInputDepthCount * inputDepth <= calcCapacity && outputDepth > 0U &&
               maxOutputDepthCount * outputDepth <= outputCapacity && tempNeed <= outputCapacity;
    }

    __aicore__ inline void ReduceNdc1hwc0Pool2DepthGroupPlane(LocalTensor<T> inputLocal, LocalTensor<T> depthLocal,
                                                              LocalTensor<T> heightLocal, LocalTensor<T> outputLocal,
                                                              uint32_t localOd, uint32_t inputDepthCount,
                                                              uint32_t inputBlock, uint32_t outputBlock,
                                                              uint32_t validC1, uint32_t inputDepthElements)
    {
        const uint32_t inputRow = static_cast<uint32_t>(tiling_->inW) * inputBlock;
        const uint32_t inputPlane = static_cast<uint32_t>(tiling_->inH) * inputRow;
        const uint32_t paddedInputRow = static_cast<uint32_t>(tiling_->outW) * 2U * inputBlock;
        const uint32_t d0 = localOd * 2U;
        const uint32_t d1 = d0 + 1U < inputDepthCount ? d0 + 1U : d0;
        Max(depthLocal, inputLocal[static_cast<uint64_t>(d0) * inputDepthElements],
            inputLocal[static_cast<uint64_t>(d1) * inputDepthElements], inputDepthElements);
        PipeBarrier<PIPE_V>();
        Duplicate(heightLocal, NegInfValue(), static_cast<uint32_t>(tiling_->inputC1) * tiling_->outH * paddedInputRow);
        PipeBarrier<PIPE_V>();
        for (uint32_t c1 = 0U; c1 < static_cast<uint32_t>(tiling_->inputC1); ++c1) {
            for (uint32_t oh = 0U; oh < static_cast<uint32_t>(tiling_->outH); ++oh) {
                const uint32_t h0 = oh * 2U;
                const uint32_t h1 = h0 + 1U < static_cast<uint32_t>(tiling_->inH) ? h0 + 1U : h0;
                Max(heightLocal[(static_cast<uint64_t>(c1) * tiling_->outH + oh) * paddedInputRow],
                    depthLocal[static_cast<uint64_t>(c1) * inputPlane + static_cast<uint64_t>(h0) * inputRow],
                    depthLocal[static_cast<uint64_t>(c1) * inputPlane + static_cast<uint64_t>(h1) * inputRow],
                    inputRow);
            }
        }
        PipeBarrier<PIPE_V>();
        const uint32_t outputRow = static_cast<uint32_t>(tiling_->outW) * outputBlock;
        const uint8_t dstStride = static_cast<uint8_t>(outputBlock * sizeof(T) / UB_BLOCK_BYTES);
        const uint8_t srcStride = static_cast<uint8_t>(2U * inputBlock * sizeof(T) / UB_BLOCK_BYTES);
        const BinaryRepeatParams widthParams{1U, 1U, 1U, dstStride, srcStride, srcStride};
        for (uint32_t c1 = 0U; c1 < static_cast<uint32_t>(tiling_->inputC1); ++c1) {
            const uint32_t channelBase = c1 * inputBlock;
            const uint32_t outputC1 = channelBase / outputBlock;
            const uint32_t outputC0 = channelBase - outputC1 * outputBlock;
            for (uint32_t oh = 0U; oh < static_cast<uint32_t>(tiling_->outH); ++oh) {
                LocalTensor<T>
                    inputRowLocal = heightLocal[(static_cast<uint64_t>(c1) * tiling_->outH + oh) * paddedInputRow];
                LocalTensor<T> outputRowLocal = outputLocal[(static_cast<uint64_t>(localOd) * validC1 + outputC1) *
                                                                tiling_->outH * outputRow +
                                                            static_cast<uint64_t>(oh) * outputRow + outputC0];
                Max(outputRowLocal, inputRowLocal, inputRowLocal[inputBlock], inputBlock,
                    static_cast<uint8_t>(tiling_->outW), widthParams);
            }
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ProcessNdc1hwc0InputOutputPool2DepthGroups()
    {
        const uint32_t inputBlock = static_cast<uint32_t>(InputNdc1hwc0Block());
        const uint32_t outputBlock = static_cast<uint32_t>(Ndc1hwc0Block());
        const uint32_t validC1 = static_cast<uint32_t>(Ndc1hwc0ValidC1(outputBlock));
        const uint32_t inputDepthElements = static_cast<uint32_t>(tiling_->inputC1 * tiling_->inH * tiling_->inW *
                                                                  inputBlock);
        const uint32_t outputDepthElements = static_cast<uint32_t>(validC1 * tiling_->outH * tiling_->outW *
                                                                   outputBlock);
        const uint32_t worker = GetBlockIdx();
        const uint32_t workerDim = ActiveBlockDim();
        uint32_t groupsPerN = workerDim / static_cast<uint32_t>(tiling_->n);
        if (groupsPerN == 0U) {
            groupsPerN = 1U;
        }
        if (groupsPerN > static_cast<uint32_t>(tiling_->outD)) {
            groupsPerN = static_cast<uint32_t>(tiling_->outD);
        }
        const uint32_t totalGroups = static_cast<uint32_t>(tiling_->n) * groupsPerN;
        for (uint32_t unit = worker; unit < totalGroups; unit += workerDim) {
            const uint32_t nIdx = unit / groupsPerN;
            const uint32_t group = unit - nIdx * groupsPerN;
            const uint32_t baseDepths = static_cast<uint32_t>(tiling_->outD) / groupsPerN;
            const uint32_t extraDepths = static_cast<uint32_t>(tiling_->outD) - baseDepths * groupsPerN;
            const uint32_t odCount = baseDepths + (group < extraDepths ? 1U : 0U);
            const uint32_t odStart = group * baseDepths + (group < extraDepths ? group : extraDepths);
            const uint32_t inputDepthStart = odStart * 2U;
            uint32_t inputDepthCount = odCount * 2U;
            if (inputDepthStart + inputDepthCount > static_cast<uint32_t>(tiling_->inD)) {
                inputDepthCount = static_cast<uint32_t>(tiling_->inD) - inputDepthStart;
            }
            const uint64_t inputOffset = (static_cast<uint64_t>(nIdx) * tiling_->inD + inputDepthStart) *
                                         inputDepthElements;
            LocalTensor<T> inputLocal = calcBuf_.Get<T>();
            DataCopy(inputLocal, xGm_[inputOffset], inputDepthCount * inputDepthElements);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            LocalTensor<T> tempLocal = tmpBuf_.Get<T>();
            LocalTensor<T> depthLocal = tempLocal;
            LocalTensor<T> heightLocal = tempLocal[inputDepthElements];
            LocalTensor<T> outputLocal = maskBuf_.Get<T>();
            Duplicate(outputLocal, ZeroValue(), odCount * outputDepthElements);
            PipeBarrier<PIPE_V>();
            for (uint32_t localOd = 0U; localOd < odCount; ++localOd) {
                ReduceNdc1hwc0Pool2DepthGroupPlane(inputLocal, depthLocal, heightLocal, outputLocal, localOd,
                                                   inputDepthCount, inputBlock, outputBlock, validC1,
                                                   inputDepthElements);
            }
            for (uint32_t localOd = 0U; localOd < odCount; ++localOd) {
                for (uint32_t c1 = 0U; c1 < validC1; ++c1) {
                    const uint32_t channelBase = c1 * outputBlock;
                    const uint32_t activeChannels = channelBase >= static_cast<uint32_t>(tiling_->c) ?
                                                        0U :
                                                        (static_cast<uint32_t>(tiling_->c) - channelBase > outputBlock ?
                                                             outputBlock :
                                                             static_cast<uint32_t>(tiling_->c) - channelBase);
                    ZeroNdc1hwc0Tail(outputLocal[(static_cast<uint64_t>(localOd) * validC1 + c1) * tiling_->outH *
                                                 tiling_->outW * outputBlock],
                                     static_cast<uint32_t>(tiling_->outH * tiling_->outW), outputBlock, activeChannels);
                }
            }
            const uint64_t outputOffset = (static_cast<uint64_t>(nIdx) * tiling_->outD + odStart) * outputDepthElements;
            CopyOutVector(outputOffset, outputLocal, odCount * outputDepthElements);
        }
        const uint64_t validOut = static_cast<uint64_t>(tiling_->n) * tiling_->outD * outputDepthElements;
        if (validOut < tiling_->totalOut) {
            CopyOutZeroRangeByCore(validOut, tiling_->totalOut - validOut);
        }
    }

    struct Ndc1hwc0Pool2BatchedContext {
        uint32_t nIdx;
        uint32_t c1Idx;
        uint32_t od;
        uint32_t inD;
        uint32_t outD;
        uint32_t outH;
        uint32_t outW;
        uint32_t inputBlock;
        uint32_t inputRowElements;
        uint32_t inputPlaneElements;
        uint32_t inputDepthElements;
        uint32_t storageC0;
        uint32_t storageC1;
        uint32_t outputC1;
        uint32_t outputC0;
        uint32_t storageBlockRatio;
        uint32_t compactOutputRowElements;
        uint32_t maxComputeRows;
        uint32_t maxStageRows;
    };

    __aicore__ inline bool CanUseNdc1hwc0InputOutputPool2BatchedRowPath() const
    {
        const uint64_t inputBlock = InputNdc1hwc0Block();
        const uint64_t storageC0 = Ndc1hwc0StorageC0();
        const uint64_t storageC1 = Ndc1hwc0StorageC1();
        if constexpr (AscendC::Std::is_same<T, float>::value) {
            if (storageC0 != inputBlock) {
                return false;
            }
        }
        if (tiling_->inputLayout != INPUT_LAYOUT_NDC1HWC0_VALUE ||
            tiling_->outputLayout != OUTPUT_LAYOUT_NDC1HWC0_VALUE || !IsPool2Stride2NoPad() || inputBlock == 0U ||
            inputBlock * sizeof(T) != UB_BLOCK_BYTES || storageC0 < inputBlock || storageC0 % inputBlock != 0U ||
            storageC1 * storageC0 < static_cast<uint64_t>(tiling_->inputC1) * inputBlock || tiling_->inH % 2 != 0 ||
            tiling_->inW % 2 != 0 || tiling_->outD != (tiling_->inD + 1) / 2 || tiling_->outH != tiling_->inH / 2 ||
            tiling_->outW != tiling_->inW / 2) {
            return false;
        }
        const uint64_t inputRow = static_cast<uint64_t>(tiling_->inW) * inputBlock;
        const uint64_t outputRow = static_cast<uint64_t>(tiling_->outW) * inputBlock;
        constexpr uint64_t calcCapacity = AscendC::Std::is_same<T, half>::value ? INPUT_TILE_NUM * 2ULL :
                                                                                  INPUT_TILE_NUM;
        return inputRow > 0U && inputRow * 4U <= calcCapacity && outputRow > 0U && outputRow <= OUTPUT_TILE_NUM;
    }

    __aicore__ inline bool InitNdc1hwc0Pool2BatchedContext(uint32_t unit, Ndc1hwc0Pool2BatchedContext& context) const
    {
        const uint32_t inputBlock = static_cast<uint32_t>(InputNdc1hwc0Block());
        const uint32_t inputC1 = static_cast<uint32_t>(tiling_->inputC1);
        const uint32_t outD = static_cast<uint32_t>(tiling_->outD);
        const uint32_t unitsPerN = inputC1 * outD;
        const uint32_t totalUnits = static_cast<uint32_t>(tiling_->n) * unitsPerN;
        if (unitsPerN == 0U || unit >= totalUnits) {
            return false;
        }
        const uint32_t nIdx = unit / unitsPerN;
        const uint32_t unitInN = unit - nIdx * unitsPerN;
        const uint32_t c1Idx = unitInN / outD;
        const uint32_t od = unitInN - c1Idx * outD;
        const uint32_t inD = static_cast<uint32_t>(tiling_->inD);
        const uint32_t inH = static_cast<uint32_t>(tiling_->inH);
        const uint32_t inW = static_cast<uint32_t>(tiling_->inW);
        const uint32_t outH = static_cast<uint32_t>(tiling_->outH);
        const uint32_t outW = static_cast<uint32_t>(tiling_->outW);
        const uint32_t inputRowElements = inW * inputBlock;
        const uint32_t inputPlaneElements = inH * inputRowElements;
        const uint32_t inputDepthElements = inputC1 * inputPlaneElements;
        const uint32_t storageC0 = static_cast<uint32_t>(Ndc1hwc0StorageC0());
        const uint32_t storageC1 = static_cast<uint32_t>(Ndc1hwc0StorageC1());
        const uint32_t channelBase = c1Idx * inputBlock;
        const uint32_t outputC1 = channelBase / storageC0;
        const uint32_t outputC0 = channelBase - outputC1 * storageC0;
        const uint32_t compactOutputRowElements = outW * inputBlock;
        constexpr uint32_t calcCapacity = AscendC::Std::is_same<T, half>::value ? INPUT_TILE_NUM * 2U : INPUT_TILE_NUM;
        uint32_t maxComputeRows = calcCapacity / (4U * inputRowElements);
        if (maxComputeRows == 0U) {
            maxComputeRows = 1U;
        }
        if (maxComputeRows > outH) {
            maxComputeRows = outH;
        }
        uint32_t maxStageRows = OUTPUT_TILE_NUM / compactOutputRowElements;
        if (maxStageRows == 0U) {
            maxStageRows = 1U;
        }
        if (maxStageRows > outH) {
            maxStageRows = outH;
        }
        context = {nIdx,
                   c1Idx,
                   od,
                   inD,
                   outD,
                   outH,
                   outW,
                   inputBlock,
                   inputRowElements,
                   inputPlaneElements,
                   inputDepthElements,
                   storageC0,
                   storageC1,
                   outputC1,
                   outputC0,
                   storageC0 / inputBlock,
                   compactOutputRowElements,
                   maxComputeRows,
                   maxStageRows};
        return true;
    }

    __aicore__ inline void LoadAndReduceNdc1hwc0Pool2BatchedRows(const Ndc1hwc0Pool2BatchedContext& context,
                                                                 LocalTensor<T> rowsLocal, uint32_t ohBase,
                                                                 uint32_t computeRows)
    {
        const uint64_t nBase = static_cast<uint64_t>(context.nIdx) * context.inD * context.inputDepthElements;
        const uint64_t c1Base = static_cast<uint64_t>(context.c1Idx) * context.inputPlaneElements;
        const uint32_t d0 = context.od * 2U;
        const uint32_t d1 = d0 + 1U < context.inD ? d0 + 1U : d0;
        const uint32_t bankElements = computeRows * 2U * context.inputRowElements;
        const uint64_t rowBase = static_cast<uint64_t>(ohBase) * 2U * context.inputRowElements;
        const uint64_t offset0 = nBase + static_cast<uint64_t>(d0) * context.inputDepthElements + c1Base + rowBase;
        const uint64_t offset1 = nBase + static_cast<uint64_t>(d1) * context.inputDepthElements + c1Base + rowBase;
        DataCopy(rowsLocal, xGm_[offset0], bankElements);
        DataCopy(rowsLocal[bankElements], xGm_[offset1], bankElements);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        for (uint32_t row = 0U; row < computeRows; ++row) {
            const uint32_t compactRow = row * context.inputRowElements;
            const uint32_t sourceRow = row * 2U * context.inputRowElements;
            Max(rowsLocal[compactRow], rowsLocal[sourceRow], rowsLocal[sourceRow + context.inputRowElements],
                context.inputRowElements);
            Max(rowsLocal[bankElements + compactRow], rowsLocal[bankElements + sourceRow],
                rowsLocal[bankElements + sourceRow + context.inputRowElements], context.inputRowElements);
        }
        PipeBarrier<PIPE_V>();
        for (uint32_t row = 0U; row < computeRows; ++row) {
            const uint32_t compactRow = row * context.inputRowElements;
            Max(rowsLocal[compactRow], rowsLocal[compactRow], rowsLocal[bankElements + compactRow],
                context.inputRowElements);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void StoreNdc1hwc0Pool2BatchedRows(const Ndc1hwc0Pool2BatchedContext& context,
                                                         LocalTensor<T> rowsLocal, LocalTensor<T> stageLocal,
                                                         uint32_t ohBase, uint32_t computeRows)
    {
        uint32_t rowBase = 0U;
        while (rowBase < computeRows) {
            uint32_t stageRows = computeRows - rowBase;
            if (stageRows > context.maxStageRows) {
                stageRows = context.maxStageRows;
            }
            for (uint32_t row = 0U; row < stageRows; ++row) {
                CompressNdhwcStride2WPairNoBarrier(
                    stageLocal[static_cast<uint64_t>(row) * context.compactOutputRowElements],
                    rowsLocal[static_cast<uint64_t>(rowBase + row) * context.inputRowElements], context.outW,
                    context.inputBlock);
            }
            PipeBarrier<PIPE_V>();
            const uint64_t outputOffset = (((static_cast<uint64_t>(context.nIdx) * context.outD + context.od) *
                                                context.storageC1 +
                                            context.outputC1) *
                                               context.outH +
                                           ohBase + rowBase) *
                                              context.outW * context.storageC0 +
                                          context.outputC0;
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            if (context.storageBlockRatio == 1U) {
                DataCopy(yGm_[outputOffset], stageLocal, stageRows * context.compactOutputRowElements);
            } else {
                const DataCopyExtParams outputParams{
                    static_cast<uint16_t>(stageRows * context.outW),
                    static_cast<uint32_t>(context.inputBlock * sizeof(T)), 0U,
                    static_cast<uint32_t>((context.storageC0 - context.inputBlock) * sizeof(T)), 0U};
                DataCopyPad(yGm_[outputOffset], stageLocal, outputParams);
            }
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            rowBase += stageRows;
        }
    }

    __aicore__ inline void ProcessNdc1hwc0InputOutputPool2BatchedRows()
    {
        const uint32_t worker = GetBlockIdx();
        const uint32_t workerDim = ActiveBlockDim();
        const uint32_t totalUnits = static_cast<uint32_t>(tiling_->n * tiling_->inputC1 * tiling_->outD);
        LocalTensor<T> rowsLocal = calcBuf_.Get<T>();
        LocalTensor<T> stageLocal = maskBuf_.Get<T>();
        for (uint32_t unit = worker; unit < totalUnits; unit += workerDim) {
            Ndc1hwc0Pool2BatchedContext context{};
            if (!InitNdc1hwc0Pool2BatchedContext(unit, context)) {
                continue;
            }
            uint32_t ohBase = 0U;
            while (ohBase < context.outH) {
                uint32_t computeRows = context.outH - ohBase;
                if (computeRows > context.maxComputeRows) {
                    computeRows = context.maxComputeRows;
                }
                LoadAndReduceNdc1hwc0Pool2BatchedRows(context, rowsLocal, ohBase, computeRows);
                StoreNdc1hwc0Pool2BatchedRows(context, rowsLocal, stageLocal, ohBase, computeRows);
                SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                ohBase += computeRows;
            }
        }
    }

    __aicore__ inline bool CanUseNdc1hwc0InputOutputPool2PlanePath() const
    {
        const uint64_t inputBlock = InputNdc1hwc0Block();
        const uint64_t outputBlock = Ndc1hwc0Block();
        const uint64_t outputValidC1 = Ndc1hwc0ValidC1(outputBlock);
        const uint64_t inputCapacity = static_cast<uint64_t>(tiling_->inputC1) * inputBlock;
        if (tiling_->inputLayout != INPUT_LAYOUT_NDC1HWC0_VALUE ||
            tiling_->outputLayout != OUTPUT_LAYOUT_NDC1HWC0_VALUE || !IsPool2Stride2NoPad() ||
            inputBlock * sizeof(T) != UB_BLOCK_BYTES || tiling_->inputC1 <= 0 || outputValidC1 == 0U ||
            outputBlock < inputBlock || outputBlock % inputBlock != 0U || outputValidC1 * outputBlock < inputCapacity ||
            !IsNdc1hwc0CompactPrefix(outputBlock, outputValidC1) || tiling_->outputD != tiling_->outD ||
            tiling_->outputH != tiling_->outH || tiling_->outputW != tiling_->outW ||
            tiling_->outD != (tiling_->inD + 1) / 2 || tiling_->outH != (tiling_->inH + 1) / 2 ||
            tiling_->outW != (tiling_->inW + 1) / 2) {
            return false;
        }
        const uint64_t inputPlane = static_cast<uint64_t>(tiling_->inH) * static_cast<uint64_t>(tiling_->inW) *
                                    inputBlock;
        const uint64_t paddedRow = static_cast<uint64_t>(tiling_->outW) * 2U * inputBlock;
        const uint64_t reducedPlane = static_cast<uint64_t>(tiling_->outH) * paddedRow;
        const uint64_t outputPlane = static_cast<uint64_t>(tiling_->outH) * static_cast<uint64_t>(tiling_->outW) *
                                     inputBlock;
        return inputPlane > 0U && inputPlane <= INPUT_TILE_NUM &&
               paddedRow >= static_cast<uint64_t>(tiling_->inW) * inputBlock && reducedPlane > 0U &&
               reducedPlane <= OUTPUT_TILE_NUM && outputPlane > 0U && outputPlane <= OUTPUT_TILE_NUM;
    }

    __aicore__ inline void LoadNdc1hwc0Pool2DepthPlane(LocalTensor<T> depthLocal, int64_t nIdx, int64_t dIdx,
                                                       int64_t cBase, uint32_t inputPlaneCount, bool first)
    {
        CopyInVector(InputOffset(nIdx, dIdx, 0, 0, cBase), inputPlaneCount);
        LocalTensor<T> inputLocal = xInQue_.DeQue<T>();
        if (first) {
            Max(depthLocal, inputLocal, inputLocal, inputPlaneCount);
        } else {
            Max(depthLocal, depthLocal, inputLocal, inputPlaneCount);
        }
        PipeBarrier<PIPE_V>();
        xInQue_.FreeTensor(inputLocal);
    }

    __aicore__ inline void ReduceNdc1hwc0Pool2PlaneHeight(LocalTensor<T> depthLocal, LocalTensor<T> heightLocal,
                                                          uint32_t inputRowCount, uint32_t paddedInputRowCount)
    {
        const uint32_t outH = static_cast<uint32_t>(tiling_->outH);
        Duplicate(heightLocal, NegInfValue(), outH * paddedInputRowCount);
        PipeBarrier<PIPE_V>();
        for (uint32_t oh = 0U; oh < outH; ++oh) {
            const uint32_t h0 = oh * 2U;
            const uint32_t h1 = h0 + 1U < static_cast<uint32_t>(tiling_->inH) ? h0 + 1U : h0;
            Max(heightLocal[static_cast<uint64_t>(oh) * paddedInputRowCount],
                depthLocal[static_cast<uint64_t>(h0) * inputRowCount],
                depthLocal[static_cast<uint64_t>(h1) * inputRowCount], inputRowCount);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopyOutNdc1hwc0Pool2PlaneChunk(uint64_t outputOffset, LocalTensor<T> outputLocal,
                                                          uint32_t outputPointCount, uint32_t inputBlock,
                                                          uint32_t outputBlock)
    {
        if (inputBlock == outputBlock) {
            CopyOutVector(outputOffset, outputLocal, outputPointCount * inputBlock);
            return;
        }
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyExtParams copyParams{static_cast<uint16_t>(outputPointCount),
                                     static_cast<uint32_t>(inputBlock * sizeof(T)), 0U,
                                     static_cast<uint32_t>((outputBlock - inputBlock) * sizeof(T)), 0U};
        DataCopyPad(yGm_[outputOffset], outputLocal, copyParams);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }

    __aicore__ inline void ProcessNdc1hwc0InputOutputPool2Plane()
    {
        const uint32_t inputBlock = static_cast<uint32_t>(InputNdc1hwc0Block());
        const uint32_t outputBlock = static_cast<uint32_t>(Ndc1hwc0Block());
        const uint32_t outputValidC1 = static_cast<uint32_t>(Ndc1hwc0ValidC1(outputBlock));
        const uint32_t inputRowCount = static_cast<uint32_t>(tiling_->inW) * inputBlock;
        const uint32_t inputPlaneCount = static_cast<uint32_t>(tiling_->inH) * inputRowCount;
        const uint32_t paddedInputRowCount = static_cast<uint32_t>(tiling_->outW) * 2U * inputBlock;
        const uint32_t outputPointCount = static_cast<uint32_t>(tiling_->outH * tiling_->outW);
        const uint32_t baseTaskCount = static_cast<uint32_t>(tiling_->n * tiling_->inputC1);
        const uint32_t worker = GetBlockIdx();
        const uint32_t workerDim = ActiveBlockDim();
        for (uint32_t baseTask = worker; baseTask < baseTaskCount; baseTask += workerDim) {
            const uint32_t inputC1 = baseTask % static_cast<uint32_t>(tiling_->inputC1);
            const int64_t nIdx = static_cast<int64_t>(baseTask / static_cast<uint32_t>(tiling_->inputC1));
            const uint32_t channelBase = inputC1 * inputBlock;
            const uint32_t outputC1 = channelBase / outputBlock;
            const uint32_t outputC0 = channelBase - outputC1 * outputBlock;
            for (uint32_t od = 0U; od < static_cast<uint32_t>(tiling_->outD); ++od) {
                const int64_t d0 = static_cast<int64_t>(od * 2U);
                const int64_t d1 = d0 + 1 < tiling_->inD ? d0 + 1 : d0;
                LocalTensor<T> depthLocal = calcBuf_.Get<T>();
                LocalTensor<T> heightLocal = tmpBuf_.Get<T>();
                LocalTensor<T> outputLocal = maskBuf_.Get<T>();
                LoadNdc1hwc0Pool2DepthPlane(depthLocal, nIdx, d0, static_cast<int64_t>(channelBase), inputPlaneCount,
                                            true);
                LoadNdc1hwc0Pool2DepthPlane(depthLocal, nIdx, d1, static_cast<int64_t>(channelBase), inputPlaneCount,
                                            false);
                ReduceNdc1hwc0Pool2PlaneHeight(depthLocal, heightLocal, inputRowCount, paddedInputRowCount);
                for (uint32_t oh = 0U; oh < static_cast<uint32_t>(tiling_->outH); ++oh) {
                    CompressNdhwcStride2WPairNoBarrier(
                        outputLocal[static_cast<uint64_t>(oh) * static_cast<uint32_t>(tiling_->outW) * inputBlock],
                        heightLocal[static_cast<uint64_t>(oh) * paddedInputRowCount],
                        static_cast<uint32_t>(tiling_->outW), inputBlock);
                }
                ZeroNdc1hwc0Tail(outputLocal, outputPointCount, inputBlock,
                                 Ndc1hwc0InputActiveChannels(static_cast<int64_t>(inputC1), inputBlock));
                const uint64_t outputOffset = ((((static_cast<uint64_t>(nIdx) * static_cast<uint64_t>(tiling_->outD) +
                                                  od) *
                                                     outputValidC1 +
                                                 outputC1) *
                                                static_cast<uint64_t>(tiling_->outH) *
                                                static_cast<uint64_t>(tiling_->outW)) *
                                               outputBlock) +
                                              outputC0;
                CopyOutNdc1hwc0Pool2PlaneChunk(outputOffset, outputLocal, outputPointCount, inputBlock, outputBlock);
            }
        }
        const uint64_t validOut = static_cast<uint64_t>(tiling_->n) * static_cast<uint64_t>(tiling_->outD) *
                                  outputValidC1 * static_cast<uint64_t>(tiling_->outH) *
                                  static_cast<uint64_t>(tiling_->outW) * outputBlock;
        if (validOut < tiling_->totalOut) {
            CopyOutZeroRangeByCore(validOut, tiling_->totalOut - validOut);
        }
    }

    __aicore__ inline bool CanUseNdc1hwc0InputOutputPackedPool2Path() const
    {
        if constexpr (!AscendC::Std::is_same<T, float>::value) {
            return false;
        }
        const uint64_t inputBlock = InputNdc1hwc0Block();
        const uint64_t outputBlock = Ndc1hwc0Block();
        const uint64_t inputCapacity = static_cast<uint64_t>(tiling_->inputC1) * inputBlock;
        const uint64_t outputValidC1 = Ndc1hwc0ValidC1(outputBlock);
        if (tiling_->inputLayout != INPUT_LAYOUT_NDC1HWC0_VALUE ||
            tiling_->outputLayout != OUTPUT_LAYOUT_NDC1HWC0_VALUE || !IsPool2Stride2NoPad() ||
            inputBlock * sizeof(T) != UB_BLOCK_BYTES || tiling_->inputC1 <= 0 || outputValidC1 == 0U ||
            outputBlock < inputBlock || outputBlock % inputBlock != 0U || outputValidC1 * outputBlock < inputCapacity ||
            !IsNdc1hwc0CompactPrefix(outputBlock, outputValidC1) || tiling_->outputD != tiling_->outD ||
            tiling_->outputH != tiling_->outH || tiling_->outputW != tiling_->outW ||
            tiling_->outD != (tiling_->inD + 1) / 2 || tiling_->outH != (tiling_->inH + 1) / 2 ||
            tiling_->outW != (tiling_->inW + 1) / 2) {
            return false;
        }
        const uint64_t inputRow = static_cast<uint64_t>(tiling_->inW) * inputBlock;
        const uint64_t paddedInputRow = static_cast<uint64_t>(tiling_->outW) * 2U * inputBlock;
        const uint64_t outputRow = static_cast<uint64_t>(tiling_->outW) * outputBlock;
        return paddedInputRow >= inputRow && paddedInputRow - inputRow <= 255U && paddedInputRow <= INPUT_TILE_NUM &&
               outputRow <= OUTPUT_TILE_NUM;
    }

    __aicore__ inline void ProcessNdc1hwc0InputOutputPackedPool2()
    {
        const uint32_t inputBlock = static_cast<uint32_t>(InputNdc1hwc0Block());
        const uint32_t outputBlock = static_cast<uint32_t>(Ndc1hwc0Block());
        const uint32_t outputValidC1 = static_cast<uint32_t>(Ndc1hwc0ValidC1(outputBlock));
        const uint32_t outW = static_cast<uint32_t>(tiling_->outW);
        const uint32_t inputRowCount = static_cast<uint32_t>(tiling_->inW) * inputBlock;
        const uint32_t paddedInputRowCount = outW * 2U * inputBlock;
        const uint32_t outputRowCount = outW * outputBlock;
        const uint32_t taskCount = static_cast<uint32_t>(tiling_->n * tiling_->outD * tiling_->outH);
        const uint32_t worker = GetBlockIdx();
        const uint32_t workerDim = ActiveBlockDim();
        for (uint32_t task = worker; task < taskCount; task += workerDim) {
            uint32_t q = task;
            const int64_t oh = static_cast<int64_t>(q % static_cast<uint32_t>(tiling_->outH));
            q /= static_cast<uint32_t>(tiling_->outH);
            const int64_t od = static_cast<int64_t>(q % static_cast<uint32_t>(tiling_->outD));
            const int64_t nIdx = static_cast<int64_t>(q / static_cast<uint32_t>(tiling_->outD));
            const int64_t d0 = od * tiling_->sD;
            const int64_t d1 = d0 + 1 < tiling_->inD ? d0 + 1 : d0;
            const int64_t h0 = oh * tiling_->sH;
            const int64_t h1 = h0 + 1 < tiling_->inH ? h0 + 1 : h0;
            for (uint32_t outputC1 = 0U; outputC1 < outputValidC1; ++outputC1) {
                LocalTensor<T> outLocal = maskBuf_.Get<T>();
                Duplicate(outLocal, ZeroValue(), outputRowCount);
                PipeBarrier<PIPE_V>();
                for (uint32_t c1 = 0U; c1 < static_cast<uint32_t>(tiling_->inputC1); ++c1) {
                    const uint32_t channelBase = c1 * inputBlock;
                    if (channelBase / outputBlock != outputC1) {
                        continue;
                    }
                    const int64_t cBase = static_cast<int64_t>(channelBase);
                    LocalTensor<T> accLocal = calcBuf_.Get<T>();
                    LocalTensor<T> compressedLocal = tmpBuf_.Get<T>();
                    CopyInVectorPad(InputOffset(nIdx, d0, h0, 0, cBase), inputRowCount, paddedInputRowCount);
                    LocalTensor<T> inputLocal = xInQue_.DeQue<T>();
                    Max(accLocal, inputLocal, inputLocal, paddedInputRowCount);
                    PipeBarrier<PIPE_V>();
                    xInQue_.FreeTensor(inputLocal);
                    AccumulateNdc1hwc0PhysicalPool2Row(accLocal, nIdx, d0, h1, cBase, inputRowCount,
                                                       paddedInputRowCount);
                    AccumulateNdc1hwc0PhysicalPool2Row(accLocal, nIdx, d1, h0, cBase, inputRowCount,
                                                       paddedInputRowCount);
                    AccumulateNdc1hwc0PhysicalPool2Row(accLocal, nIdx, d1, h1, cBase, inputRowCount,
                                                       paddedInputRowCount);
                    CompressNdhwcStride2WPairNoBarrier(compressedLocal, accLocal, outW, inputBlock);
                    const uint32_t outputC0 = channelBase - outputC1 * outputBlock;
                    const uint16_t dstStride = static_cast<uint16_t>((outputBlock - inputBlock) * sizeof(T) /
                                                                     UB_BLOCK_BYTES);
                    const DataCopyParams packParams{static_cast<uint16_t>(outW), 1U, 0U, dstStride};
                    DataCopy(outLocal[outputC0], compressedLocal, packParams);
                    DataSyncBarrier<MemDsbT::UB>();
                }
                const uint64_t outputOffset = ((((static_cast<uint64_t>(nIdx) * static_cast<uint64_t>(tiling_->outD) +
                                                  static_cast<uint64_t>(od)) *
                                                     outputValidC1 +
                                                 outputC1) *
                                                    static_cast<uint64_t>(tiling_->outH) +
                                                static_cast<uint64_t>(oh)) *
                                               outW) *
                                              outputBlock;
                CopyOutVector(outputOffset, outLocal, outputRowCount);
            }
        }
        const uint64_t validOut = static_cast<uint64_t>(tiling_->n) * static_cast<uint64_t>(tiling_->outD) *
                                  outputValidC1 * static_cast<uint64_t>(tiling_->outH) * outW * outputBlock;
        if (validOut < tiling_->totalOut) {
            CopyOutZeroRangeByCore(validOut, tiling_->totalOut - validOut);
        }
    }

    __aicore__ inline bool HasNdc1hwc0InputOutputSingleBlockFeatureLayout(uint32_t& inputBlock,
                                                                          uint32_t& outputBlock) const
    {
        if (!HasNdc1hwc0InputOutputLayout() || tiling_->inputC1 <= 0 || tiling_->outputC1 != 1 || tiling_->c <= 0) {
            return false;
        }
        inputBlock = static_cast<uint32_t>(InputNdc1hwc0Block());
        outputBlock = static_cast<uint32_t>(Ndc1hwc0Block());
        const uint64_t inputCapacity = static_cast<uint64_t>(tiling_->inputC1) * inputBlock;
        return inputBlock > 0U && outputBlock > 0U && inputBlock * sizeof(T) == UB_BLOCK_BYTES &&
               outputBlock >= inputBlock && outputBlock % inputBlock == 0U &&
               inputCapacity >= static_cast<uint64_t>(tiling_->c) && inputCapacity <= outputBlock &&
               IsNdc1hwc0CompactPrefix(outputBlock, 1U);
    }

    __aicore__ inline void PackNdc1hwc0FeatureChannels(LocalTensor<T> outputLocal, LocalTensor<T> compactLocal,
                                                       uint32_t inputC1, uint32_t inputBlock, uint32_t outputBlock,
                                                       uint32_t pointCount)
    {
        const uint16_t blockLen = static_cast<uint16_t>(inputBlock * sizeof(T) / UB_BLOCK_BYTES);
        const uint16_t dstStride = static_cast<uint16_t>((outputBlock - inputBlock) * sizeof(T) / UB_BLOCK_BYTES);
        const DataCopyParams packParams{static_cast<uint16_t>(pointCount), blockLen, 0U, dstStride};
        for (uint32_t c1 = 0U; c1 < inputC1; ++c1) {
            DataCopy(outputLocal[c1 * inputBlock], compactLocal, packParams);
            DataSyncBarrier<MemDsbT::UB>();
        }
    }

    __aicore__ inline void ClearNdc1hwc0FeatureOutputTail(LocalTensor<T> outputLocal, uint32_t pointCount,
                                                          uint32_t outputBlock)
    {
        const uint32_t active = static_cast<uint32_t>(tiling_->c);
        ZeroNdc1hwc0Tail(outputLocal, pointCount, outputBlock, active);
    }

    __aicore__ inline bool CanUseNdc1hwc0InputOutputTinyK3WholeNPath() const
    {
        uint32_t inputBlock = 0U;
        uint32_t outputBlock = 0U;
        if (!HasNdc1hwc0InputOutputSingleBlockFeatureLayout(inputBlock, outputBlock) ||
            !MatchesPoolSpec(3, 3, 3, 1, tiling_->sH, 1, 1, 1, 1, 0, 0, 0) || (tiling_->sH != 1 && tiling_->sH != 3) ||
            tiling_->outH != 1 || tiling_->outD != tiling_->inD - 2 ||
            tiling_->outH != (tiling_->inH - 3) / tiling_->sH + 1 || tiling_->outW != tiling_->inW - 2 ||
            tiling_->outD <= 0 || tiling_->outH <= 0 || tiling_->outW <= 0) {
            return false;
        }
        const uint64_t inputPerN = static_cast<uint64_t>(tiling_->inD) * tiling_->inputC1 * tiling_->inH *
                                   tiling_->inW * inputBlock;
        const uint64_t widthCount = static_cast<uint64_t>(tiling_->inD) * tiling_->inH * tiling_->outW * inputBlock;
        const uint64_t heightCount = static_cast<uint64_t>(tiling_->inD) * tiling_->outH * tiling_->outW * inputBlock;
        const uint64_t pointCount = static_cast<uint64_t>(tiling_->outD) * tiling_->outH * tiling_->outW;
        const uint64_t rowStride = static_cast<uint64_t>(tiling_->outW) * inputBlock * sizeof(T) / UB_BLOCK_BYTES;
        return inputPerN <= INPUT_TILE_NUM && widthCount <= OUTPUT_TILE_NUM && heightCount <= OUTPUT_TILE_NUM &&
               pointCount * outputBlock <= OUTPUT_TILE_NUM && pointCount <= 65535U && rowStride > 0U &&
               rowStride * static_cast<uint64_t>(tiling_->inH) <= 255U && tiling_->inD <= 255 && tiling_->outD <= 255;
    }

    __aicore__ inline void ReduceNdc1hwc0TinyK3Width(LocalTensor<T> inputLocal, LocalTensor<T> widthLocal, uint32_t c1,
                                                     uint32_t inputBlock, uint32_t outW)
    {
        const uint32_t inputRow = static_cast<uint32_t>(tiling_->inW) * inputBlock;
        const uint32_t compactRow = outW * inputBlock;
        const uint32_t inputC1Plane = static_cast<uint32_t>(tiling_->inH) * inputRow;
        const uint32_t inputDepth = static_cast<uint32_t>(tiling_->inputC1) * inputC1Plane;
        for (uint32_t d = 0U; d < static_cast<uint32_t>(tiling_->inD); ++d) {
            for (uint32_t h = 0U; h < static_cast<uint32_t>(tiling_->inH); ++h) {
                const uint32_t src = d * inputDepth + c1 * inputC1Plane + h * inputRow;
                const uint32_t dst = (d * static_cast<uint32_t>(tiling_->inH) + h) * compactRow;
                Max(widthLocal[dst], inputLocal[src], inputLocal[src + inputBlock], compactRow);
            }
        }
        PipeBarrier<PIPE_V>();
        for (uint32_t d = 0U; d < static_cast<uint32_t>(tiling_->inD); ++d) {
            for (uint32_t h = 0U; h < static_cast<uint32_t>(tiling_->inH); ++h) {
                const uint32_t src = d * inputDepth + c1 * inputC1Plane + h * inputRow;
                const uint32_t dst = (d * static_cast<uint32_t>(tiling_->inH) + h) * compactRow;
                Max(widthLocal[dst], widthLocal[dst], inputLocal[src + 2U * inputBlock], compactRow);
            }
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ReduceNdc1hwc0TinyK3HeightDepth(LocalTensor<T> widthLocal, LocalTensor<T> heightLocal,
                                                           uint32_t inputBlock, uint32_t outW)
    {
        const uint32_t compactRow = outW * inputBlock;
        const uint32_t rowStride = compactRow * sizeof(T) / UB_BLOCK_BYTES;
        const BinaryRepeatParams heightFirstParams{1U,
                                                   1U,
                                                   1U,
                                                   static_cast<uint8_t>(rowStride),
                                                   static_cast<uint8_t>(tiling_->inH * rowStride),
                                                   static_cast<uint8_t>(tiling_->inH * rowStride)};
        Max(heightLocal, widthLocal, widthLocal[compactRow], compactRow, static_cast<uint8_t>(tiling_->inD),
            heightFirstParams);
        PipeBarrier<PIPE_V>();
        const BinaryRepeatParams heightFinishParams{1U,
                                                    1U,
                                                    1U,
                                                    static_cast<uint8_t>(rowStride),
                                                    static_cast<uint8_t>(rowStride),
                                                    static_cast<uint8_t>(tiling_->inH * rowStride)};
        Max(heightLocal, heightLocal, widthLocal[2U * compactRow], compactRow, static_cast<uint8_t>(tiling_->inD),
            heightFinishParams);
        PipeBarrier<PIPE_V>();
        const BinaryRepeatParams depthParams{1U,
                                             1U,
                                             1U,
                                             static_cast<uint8_t>(rowStride),
                                             static_cast<uint8_t>(rowStride),
                                             static_cast<uint8_t>(rowStride)};
        Max(widthLocal, heightLocal, heightLocal[compactRow], compactRow, static_cast<uint8_t>(tiling_->outD),
            depthParams);
        PipeBarrier<PIPE_V>();
        Max(widthLocal, widthLocal, heightLocal[2U * compactRow], compactRow, static_cast<uint8_t>(tiling_->outD),
            depthParams);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ProcessNdc1hwc0InputOutputTinyK3WholeN()
    {
        const uint32_t inputBlock = static_cast<uint32_t>(InputNdc1hwc0Block());
        const uint32_t outputBlock = static_cast<uint32_t>(Ndc1hwc0Block());
        const uint32_t outW = static_cast<uint32_t>(tiling_->outW);
        const uint32_t pointCount = static_cast<uint32_t>(tiling_->outD * tiling_->outH * tiling_->outW);
        const uint32_t inputPerN = static_cast<uint32_t>(tiling_->inD * tiling_->inputC1 * tiling_->inH * tiling_->inW *
                                                         inputBlock);
        const uint32_t outputPerN = pointCount * outputBlock;
        const uint32_t worker = GetBlockIdx();
        const uint32_t workerDim = ActiveBlockDim();
        for (uint32_t nIdx = worker; nIdx < static_cast<uint32_t>(tiling_->n); nIdx += workerDim) {
            CopyInVector(static_cast<uint64_t>(nIdx) * inputPerN, inputPerN);
            LocalTensor<T> inputLocal = xInQue_.DeQue<T>();
            LocalTensor<T> widthLocal = calcBuf_.Get<T>();
            LocalTensor<T> heightLocal = tmpBuf_.Get<T>();
            LocalTensor<T> outputLocal = maskBuf_.Get<T>();
            Duplicate(outputLocal, ZeroValue(), outputPerN);
            PipeBarrier<PIPE_V>();
            for (uint32_t c1 = 0U; c1 < static_cast<uint32_t>(tiling_->inputC1); ++c1) {
                ReduceNdc1hwc0TinyK3Width(inputLocal, widthLocal, c1, inputBlock, outW);
                ReduceNdc1hwc0TinyK3HeightDepth(widthLocal, heightLocal, inputBlock, outW);
                PackNdc1hwc0FeatureChannels(outputLocal, widthLocal, 1U, inputBlock, outputBlock, pointCount);
                outputLocal = outputLocal[inputBlock];
            }
            outputLocal = maskBuf_.Get<T>();
            ClearNdc1hwc0FeatureOutputTail(outputLocal, pointCount, outputBlock);
            CopyOutVector(static_cast<uint64_t>(nIdx) * outputPerN, outputLocal, outputPerN);
            xInQue_.FreeTensor(inputLocal);
        }
        const uint64_t validOut = static_cast<uint64_t>(tiling_->n) * outputPerN;
        if (validOut < tiling_->totalOut) {
            CopyOutZeroRangeByCore(validOut, tiling_->totalOut - validOut);
        }
    }

    __aicore__ inline bool CanUseNdc1hwc0InputOutputHOnlyPlanePath() const
    {
        uint32_t inputBlock = 0U;
        uint32_t outputBlock = 0U;
        if (!HasNdc1hwc0InputOutputSingleBlockFeatureLayout(inputBlock, outputBlock) || inputBlock != outputBlock ||
            !MatchesPoolSpec(1, 3, 1, 1, 3, 1, 1, 1, 1, 0, 1, 0) || tiling_->outD != tiling_->inD ||
            tiling_->outW != tiling_->inW || tiling_->outH < 3 || tiling_->inH != tiling_->outH * 3 - 2) {
            return false;
        }
        const uint64_t inputPlane = static_cast<uint64_t>(tiling_->inputC1) * tiling_->inH * tiling_->inW * inputBlock;
        const uint64_t compactPlane = static_cast<uint64_t>(tiling_->outH) * tiling_->outW * inputBlock;
        const uint64_t outputPlane = static_cast<uint64_t>(tiling_->outH) * tiling_->outW * outputBlock;
        const uint64_t rowStride = static_cast<uint64_t>(tiling_->outW) * inputBlock * sizeof(T) / UB_BLOCK_BYTES;
        return inputPlane <= INPUT_TILE_NUM && compactPlane <= OUTPUT_TILE_NUM && outputPlane <= OUTPUT_TILE_NUM &&
               static_cast<uint64_t>(tiling_->outH) * tiling_->outW <= 65535U && rowStride > 0U &&
               rowStride * 3U <= 255U && tiling_->outH <= 255;
    }

    __aicore__ inline void ReduceNdc1hwc0HOnlyRows(LocalTensor<T> dst, LocalTensor<T> src0, LocalTensor<T> src1,
                                                   uint32_t rowElements, uint8_t repeat,
                                                   const BinaryRepeatParams& params)
    {
        constexpr uint32_t vectorElements = 256U / sizeof(T);
        for (uint32_t offset = 0U; offset < rowElements; offset += vectorElements) {
            const uint32_t count = rowElements - offset > vectorElements ? vectorElements : rowElements - offset;
            Max(dst[offset], src0[offset], src1[offset], count, repeat, params);
        }
    }

    __aicore__ inline void ProcessNdc1hwc0InputOutputHOnlyPlanes()
    {
        const uint32_t inputBlock = static_cast<uint32_t>(InputNdc1hwc0Block());
        const uint32_t outputBlock = static_cast<uint32_t>(Ndc1hwc0Block());
        const uint32_t inputRow = static_cast<uint32_t>(tiling_->inW) * inputBlock;
        const uint32_t inputC1Plane = static_cast<uint32_t>(tiling_->inH) * inputRow;
        const uint32_t inputPlane = static_cast<uint32_t>(tiling_->inputC1) * inputC1Plane;
        const uint32_t compactRow = static_cast<uint32_t>(tiling_->outW) * inputBlock;
        const uint32_t compactPlane = static_cast<uint32_t>(tiling_->outH) * compactRow;
        const uint32_t pointCount = static_cast<uint32_t>(tiling_->outH * tiling_->outW);
        const uint32_t outputPlane = pointCount * outputBlock;
        const uint32_t totalPlanes = static_cast<uint32_t>(tiling_->n * tiling_->outD);
        const uint32_t worker = GetBlockIdx();
        const uint32_t workerDim = ActiveBlockDim();
        for (uint32_t plane = worker; plane < totalPlanes; plane += workerDim) {
            CopyInVector(static_cast<uint64_t>(plane) * inputPlane, inputPlane);
            LocalTensor<T> inputLocal = xInQue_.DeQue<T>();
            LocalTensor<T> compactLocal = calcBuf_.Get<T>();
            LocalTensor<T> outputLocal = maskBuf_.Get<T>();
            Duplicate(outputLocal, ZeroValue(), outputPlane);
            PipeBarrier<PIPE_V>();
            for (uint32_t c1 = 0U; c1 < static_cast<uint32_t>(tiling_->inputC1); ++c1) {
                LocalTensor<T> inputC1Local = inputLocal[static_cast<uint64_t>(c1) * inputC1Plane];
                Max(compactLocal, inputC1Local, inputC1Local[inputRow], inputRow);
                Max(compactLocal[static_cast<uint64_t>(tiling_->outH - 1) * compactRow],
                    inputC1Local[static_cast<uint64_t>(tiling_->inH - 2) * inputRow],
                    inputC1Local[static_cast<uint64_t>(tiling_->inH - 1) * inputRow], inputRow);
                const uint32_t rowStride = compactRow * sizeof(T) / UB_BLOCK_BYTES;
                const uint32_t middleRows = static_cast<uint32_t>(tiling_->outH - 2);
                const BinaryRepeatParams firstParams{1U,
                                                     1U,
                                                     1U,
                                                     static_cast<uint8_t>(rowStride),
                                                     static_cast<uint8_t>(3U * rowStride),
                                                     static_cast<uint8_t>(3U * rowStride)};
                ReduceNdc1hwc0HOnlyRows(compactLocal[compactRow], inputC1Local[2U * inputRow],
                                        inputC1Local[3U * inputRow], compactRow, static_cast<uint8_t>(middleRows),
                                        firstParams);
                PipeBarrier<PIPE_V>();
                const BinaryRepeatParams finishParams{1U,
                                                      1U,
                                                      1U,
                                                      static_cast<uint8_t>(rowStride),
                                                      static_cast<uint8_t>(rowStride),
                                                      static_cast<uint8_t>(3U * rowStride)};
                ReduceNdc1hwc0HOnlyRows(compactLocal[compactRow], compactLocal[compactRow], inputC1Local[4U * inputRow],
                                        compactRow, static_cast<uint8_t>(middleRows), finishParams);
                PipeBarrier<PIPE_V>();
                PackNdc1hwc0FeatureChannels(outputLocal, compactLocal, 1U, inputBlock, outputBlock, pointCount);
                outputLocal = outputLocal[inputBlock];
            }
            outputLocal = maskBuf_.Get<T>();
            ClearNdc1hwc0FeatureOutputTail(outputLocal, pointCount, outputBlock);
            CopyOutVector(static_cast<uint64_t>(plane) * outputPlane, outputLocal, outputPlane);
            xInQue_.FreeTensor(inputLocal);
        }
        const uint64_t validOut = static_cast<uint64_t>(totalPlanes) * outputPlane;
        if (validOut < tiling_->totalOut) {
            CopyOutZeroRangeByCore(validOut, tiling_->totalOut - validOut);
        }
    }

    __aicore__ inline bool CanUseNdc1hwc0InputOutputD3H3PipelinedPlanePath() const
    {
        const uint64_t inputBlock = InputNdc1hwc0Block();
        const uint64_t outputBlock = Ndc1hwc0Block();
        const uint64_t inputCapacity = static_cast<uint64_t>(tiling_->inputC1) * inputBlock;
        if (!HasNdc1hwc0InputOutputLayout() || tiling_->inputC1 != 1 || tiling_->outputC1 != 1 ||
            inputBlock * sizeof(T) != UB_BLOCK_BYTES || outputBlock == 0U ||
            inputCapacity < static_cast<uint64_t>(tiling_->c) || inputBlock > outputBlock ||
            !IsNdc1hwc0CompactPrefix(outputBlock, 1U) || !MatchesPoolSpec(3, 3, 1, 3, 1, 1, 1, 2, 1, 0, 2, 0) ||
            tiling_->inD <= 0 || tiling_->outD * tiling_->sD != tiling_->inD || tiling_->outH != tiling_->inH ||
            tiling_->outW != tiling_->inW || tiling_->inH < 5 || tiling_->inW <= 0 || tiling_->inW > 255) {
            return false;
        }
        const uint64_t inputPlane = static_cast<uint64_t>(tiling_->inH) * tiling_->inW * inputBlock;
        const uint64_t outputPlane = static_cast<uint64_t>(tiling_->outH) * tiling_->outW * outputBlock;
        return inputPlane > 0U && inputPlane * 3U <= INPUT_TILE_NUM && inputPlane <= OUTPUT_TILE_NUM &&
               outputPlane > 0U && outputPlane <= OUTPUT_TILE_NUM &&
               static_cast<uint64_t>(tiling_->inH) * tiling_->inW <= UINT16_MAX;
    }

    __aicore__ inline void InitNdc1hwc0InputOutputD3H3PipelinedBuffers()
    {
        const uint32_t inputBlock = static_cast<uint32_t>(InputNdc1hwc0Block());
        const uint32_t outputBlock = static_cast<uint32_t>(Ndc1hwc0Block());
        const uint32_t inputPlane = static_cast<uint32_t>(tiling_->inH * tiling_->inW) * inputBlock;
        const uint32_t outputPlane = static_cast<uint32_t>(tiling_->outH * tiling_->outW) * outputBlock;
        pipe_.InitBuffer(xInQue_, AscendC::Std::is_same<T, half>::value ? 2U : 1U, UbBytesForElements(3U * inputPlane));
        pipe_.InitBuffer(calcBuf_, UbBytesForElements(inputPlane));
        pipe_.InitBuffer(yOutQue_, 2U, UbBytesForElements(outputPlane));
        pipe_.InitBuffer(tmpBuf_, UbBytesForElements(AscendC::Std::is_same<T, float>::value ? outputPlane : 1U));
    }

    __aicore__ inline uint64_t Ndc1hwc0InputOutputD3H3InputOffset(uint32_t unit, uint32_t outD,
                                                                  uint32_t inputPlane) const
    {
        const uint32_t nIdx = unit / outD;
        const uint32_t od = unit - nIdx * outD;
        return (static_cast<uint64_t>(nIdx) * tiling_->inD + static_cast<uint64_t>(od) * tiling_->sD) * inputPlane;
    }

    __aicore__ inline void PrefetchNdc1hwc0InputOutputD3H3First(uint32_t worker, uint32_t totalUnits, uint32_t outD,
                                                                uint32_t inputPlane)
    {
        if constexpr (AscendC::Std::is_same<T, half>::value) {
            if (worker < totalUnits) {
                LocalTensor<T> firstLocal = xInQue_.AllocTensor<T>();
                DataCopy(firstLocal, xGm_[Ndc1hwc0InputOutputD3H3InputOffset(worker, outD, inputPlane)],
                         3U * inputPlane);
                xInQue_.EnQue(firstLocal);
            }
        }
    }

    __aicore__ inline void ReduceNdc1hwc0InputOutputD3H3Depth(LocalTensor<T> depthLocal, uint32_t unit,
                                                              uint32_t workerDim, uint32_t totalUnits, uint32_t outD,
                                                              uint32_t inputPlane)
    {
        LocalTensor<T> inputSlab;
        if constexpr (AscendC::Std::is_same<T, half>::value) {
            inputSlab = xInQue_.DeQue<T>();
        } else {
            LocalTensor<T> inputLocal = xInQue_.AllocTensor<T>();
            DataCopy(inputLocal, xGm_[Ndc1hwc0InputOutputD3H3InputOffset(unit, outD, inputPlane)], 3U * inputPlane);
            xInQue_.EnQue(inputLocal);
            inputSlab = xInQue_.DeQue<T>();
        }
        Max(depthLocal, inputSlab, inputSlab[inputPlane], inputPlane);
        PipeBarrier<PIPE_V>();
        Max(depthLocal, depthLocal, inputSlab[static_cast<uint64_t>(inputPlane) * 2U], inputPlane);
        PipeBarrier<PIPE_V>();
        if constexpr (AscendC::Std::is_same<T, half>::value) {
            const uint32_t nextUnit = unit + workerDim;
            if (nextUnit < totalUnits) {
                LocalTensor<T> nextLocal = xInQue_.AllocTensor<T>();
                DataCopy(nextLocal, xGm_[Ndc1hwc0InputOutputD3H3InputOffset(nextUnit, outD, inputPlane)],
                         3U * inputPlane);
                xInQue_.EnQue(nextLocal);
            }
        }
        xInQue_.FreeTensor(inputSlab);
    }

    __aicore__ inline bool StoreNdc1hwc0InputOutputD3H3FloatInput8(LocalTensor<T> depthLocal, uint64_t outputOffset,
                                                                   uint32_t inputBlock, uint32_t outputBlock,
                                                                   uint32_t inputRow, uint32_t outputPlane,
                                                                   event_t eventIdVToMte3, event_t eventIdMte3ToV)
    {
        if constexpr (AscendC::Std::is_same<T, float>::value) {
            if (inputBlock == UB_BLOCK_BYTES / sizeof(T)) {
                const uint32_t inH = static_cast<uint32_t>(tiling_->inH);
                const uint32_t inW = static_cast<uint32_t>(tiling_->inW);
                LocalTensor<T> outputLocal = tmpBuf_.Get<T>();
                Duplicate(outputLocal, ZeroValue(), outputPlane);
                PipeBarrier<PIPE_V>();
                const uint8_t repeats = static_cast<uint8_t>(inW);
                const uint8_t outputStep = static_cast<uint8_t>(outputBlock * sizeof(T) / UB_BLOCK_BYTES);
                const uint8_t inputStep = static_cast<uint8_t>(inputBlock * sizeof(T) / UB_BLOCK_BYTES);
                const BinaryRepeatParams firstParams{1U, 1U, 1U, outputStep, inputStep, inputStep};
                const BinaryRepeatParams finishParams{1U, 1U, 1U, outputStep, outputStep, inputStep};
                for (uint32_t oh = 0U; oh < inH; ++oh) {
                    const uint32_t firstH = oh >= 2U ? oh - 2U : oh;
                    const uint32_t secondH = oh >= 2U ? oh : oh + 2U;
                    LocalTensor<T> outputRow = outputLocal[static_cast<uint64_t>(oh) * inW * outputBlock];
                    Max(outputRow, depthLocal[static_cast<uint64_t>(firstH) * inputRow],
                        depthLocal[static_cast<uint64_t>(secondH) * inputRow], inputBlock, repeats, firstParams);
                    if (oh >= 2U && oh + 2U < inH) {
                        PipeBarrier<PIPE_V>();
                        Max(outputRow, outputRow, depthLocal[static_cast<uint64_t>(oh + 2U) * inputRow], inputBlock,
                            repeats, finishParams);
                    }
                }
                PipeBarrier<PIPE_V>();
                CopyOutVectorWithQueueSync(outputOffset, outputLocal, outputPlane, eventIdVToMte3, eventIdMte3ToV);
                return true;
            }
        }
        return false;
    }

    __aicore__ inline void StoreNdc1hwc0InputOutputD3H3Reduced(LocalTensor<T> depthLocal, uint64_t outputOffset,
                                                               uint32_t inputBlock, uint32_t outputBlock,
                                                               uint32_t inputRow, uint32_t outputPlane,
                                                               event_t eventIdVToMte3, event_t eventIdMte3ToV)
    {
        const uint32_t inH = static_cast<uint32_t>(tiling_->inH);
        const uint32_t inW = static_cast<uint32_t>(tiling_->inW);
        LocalTensor<T> heightLocal = yOutQue_.AllocTensor<T>();
        Max(heightLocal[2U * inputRow], depthLocal[2U * inputRow], depthLocal, (inH - 2U) * inputRow);
        Max(heightLocal, depthLocal, depthLocal[2U * inputRow], 2U * inputRow);
        PipeBarrier<PIPE_V>();
        Max(heightLocal[2U * inputRow], heightLocal[2U * inputRow], depthLocal[4U * inputRow], (inH - 4U) * inputRow);
        PipeBarrier<PIPE_V>();
        if constexpr (AscendC::Std::is_same<T, half>::value) {
            yOutQue_.EnQue(heightLocal);
            LocalTensor<T> readyLocal = yOutQue_.DeQue<T>();
            DataCopy(yGm_[outputOffset], readyLocal, outputPlane);
            yOutQue_.FreeTensor(readyLocal);
        } else if (inputBlock == outputBlock) {
            yOutQue_.EnQue(heightLocal);
            LocalTensor<T> readyLocal = yOutQue_.DeQue<T>();
            DataCopy(yGm_[outputOffset], readyLocal, outputPlane);
            yOutQue_.FreeTensor(readyLocal);
        } else {
            LocalTensor<T> outputLocal = tmpBuf_.Get<T>();
            Duplicate(outputLocal, ZeroValue(), outputPlane);
            PipeBarrier<PIPE_V>();
            const uint16_t blockCount = static_cast<uint16_t>(inH * inW);
            const uint16_t blockLen = static_cast<uint16_t>(inputBlock * sizeof(T) / UB_BLOCK_BYTES);
            const uint16_t dstStride = static_cast<uint16_t>((outputBlock - inputBlock) * sizeof(T) / UB_BLOCK_BYTES);
            const DataCopyParams packParams{blockCount, blockLen, 0U, dstStride};
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            DataCopy(outputLocal, heightLocal, packParams);
            SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            yOutQue_.FreeTensor(heightLocal);
            CopyOutVectorWithQueueSync(outputOffset, outputLocal, outputPlane, eventIdVToMte3, eventIdMte3ToV);
        }
    }

    __aicore__ inline void ZeroNdc1hwc0InputOutputD3H3FloatTail(uint32_t worker, uint32_t workerDim,
                                                                uint32_t totalUnits, uint32_t outputPlane,
                                                                event_t eventIdVToMte3, event_t eventIdMte3ToV)
    {
        if constexpr (AscendC::Std::is_same<T, float>::value) {
            const uint64_t validOut = static_cast<uint64_t>(totalUnits) * outputPlane;
            if (validOut < tiling_->totalOut) {
                LocalTensor<T> zeroLocal = tmpBuf_.Get<T>();
                Duplicate(zeroLocal, ZeroValue(), outputPlane);
                PipeBarrier<PIPE_V>();
                for (uint32_t unit = worker; unit < totalUnits; unit += workerDim) {
                    CopyOutVectorWithQueueSync(validOut + static_cast<uint64_t>(unit) * outputPlane, zeroLocal,
                                               outputPlane, eventIdVToMte3, eventIdMte3ToV);
                }
            }
        }
    }

    __aicore__ inline void ProcessNdc1hwc0InputOutputD3H3PipelinedPlanes()
    {
        const uint32_t inputBlock = static_cast<uint32_t>(InputNdc1hwc0Block());
        const uint32_t outputBlock = static_cast<uint32_t>(Ndc1hwc0Block());
        const uint32_t outD = static_cast<uint32_t>(tiling_->outD);
        const uint32_t inputRow = static_cast<uint32_t>(tiling_->inW) * inputBlock;
        const uint32_t inputPlane = static_cast<uint32_t>(tiling_->inH) * inputRow;
        const uint32_t outputPlane = static_cast<uint32_t>(tiling_->outH * tiling_->outW) * outputBlock;
        const uint32_t totalUnits = static_cast<uint32_t>(tiling_->n) * outD;
        const uint32_t worker = GetBlockIdx();
        const uint32_t workerDim = ActiveBlockDim();
        event_t eventIdVToMte3 = static_cast<event_t>(EVENT_ID0);
        event_t eventIdMte3ToV = static_cast<event_t>(EVENT_ID0);
        if constexpr (AscendC::Std::is_same<T, float>::value) {
            eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            eventIdMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        }
        LocalTensor<T> depthLocal = calcBuf_.Get<T>();
        PrefetchNdc1hwc0InputOutputD3H3First(worker, totalUnits, outD, inputPlane);
        for (uint32_t unit = worker; unit < totalUnits; unit += workerDim) {
            ReduceNdc1hwc0InputOutputD3H3Depth(depthLocal, unit, workerDim, totalUnits, outD, inputPlane);
            const uint64_t outputOffset = static_cast<uint64_t>(unit) * outputPlane;
            if (!StoreNdc1hwc0InputOutputD3H3FloatInput8(depthLocal, outputOffset, inputBlock, outputBlock, inputRow,
                                                         outputPlane, eventIdVToMte3, eventIdMte3ToV)) {
                StoreNdc1hwc0InputOutputD3H3Reduced(depthLocal, outputOffset, inputBlock, outputBlock, inputRow,
                                                    outputPlane, eventIdVToMte3, eventIdMte3ToV);
            }
        }
        ZeroNdc1hwc0InputOutputD3H3FloatTail(worker, workerDim, totalUnits, outputPlane, eventIdVToMte3,
                                             eventIdMte3ToV);
        if constexpr (AscendC::Std::is_same<T, float>::value) {
            GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(eventIdVToMte3);
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(eventIdMte3ToV);
        }
    }

    __aicore__ inline bool CanUseNdc1hwc0InputOutputD3H3PlanePath() const
    {
        uint32_t inputBlock = 0U;
        uint32_t outputBlock = 0U;
        if (!HasNdc1hwc0InputOutputSingleBlockFeatureLayout(inputBlock, outputBlock) || tiling_->inputC1 != 1 ||
            !MatchesPoolSpec(3, 3, 1, 3, 1, 1, 1, 2, 1, 0, 2, 0) || tiling_->outW != tiling_->inW ||
            tiling_->outD <= 0 || tiling_->outH != tiling_->inH || tiling_->outH < 4) {
            return false;
        }
        const uint64_t inputPlane = static_cast<uint64_t>(tiling_->inH) * tiling_->inW * inputBlock;
        const uint64_t compactPlane = static_cast<uint64_t>(tiling_->outH) * tiling_->outW * inputBlock;
        const uint64_t outputPlane = static_cast<uint64_t>(tiling_->outH) * tiling_->outW * outputBlock;
        return inputPlane * 3U <= INPUT_TILE_NUM && inputPlane <= OUTPUT_TILE_NUM && compactPlane <= OUTPUT_TILE_NUM &&
               outputPlane <= OUTPUT_TILE_NUM && static_cast<uint64_t>(tiling_->outH) * tiling_->outW <= 65535U;
    }

    __aicore__ inline void ProcessNdc1hwc0InputOutputD3H3Planes()
    {
        const uint32_t inputBlock = static_cast<uint32_t>(InputNdc1hwc0Block());
        const uint32_t outputBlock = static_cast<uint32_t>(Ndc1hwc0Block());
        const uint32_t inputRow = static_cast<uint32_t>(tiling_->inW) * inputBlock;
        const uint32_t inputPlane = static_cast<uint32_t>(tiling_->inH) * inputRow;
        const uint32_t compactRow = static_cast<uint32_t>(tiling_->outW) * inputBlock;
        const uint32_t compactPlane = static_cast<uint32_t>(tiling_->outH) * compactRow;
        const uint32_t pointCount = static_cast<uint32_t>(tiling_->outH * tiling_->outW);
        const uint32_t outputPlane = pointCount * outputBlock;
        const uint32_t totalPlanes = static_cast<uint32_t>(tiling_->n * tiling_->outD);
        const uint32_t worker = GetBlockIdx();
        const uint32_t workerDim = ActiveBlockDim();
        for (uint32_t unit = worker; unit < totalPlanes; unit += workerDim) {
            const uint32_t nIdx = unit / static_cast<uint32_t>(tiling_->outD);
            const uint32_t od = unit - nIdx * static_cast<uint32_t>(tiling_->outD);
            const uint64_t inputOffset = (static_cast<uint64_t>(nIdx) * tiling_->inD + static_cast<uint64_t>(od) * 3U) *
                                         inputPlane;
            CopyInVector(inputOffset, 3U * inputPlane);
            LocalTensor<T> inputLocal = xInQue_.DeQue<T>();
            LocalTensor<T> depthLocal = calcBuf_.Get<T>();
            LocalTensor<T> compactLocal = tmpBuf_.Get<T>();
            LocalTensor<T> outputLocal = maskBuf_.Get<T>();
            Max(depthLocal, inputLocal, inputLocal[inputPlane], inputPlane);
            PipeBarrier<PIPE_V>();
            Max(depthLocal, depthLocal, inputLocal[2U * inputPlane], inputPlane);
            PipeBarrier<PIPE_V>();
            Max(compactLocal[2U * compactRow], depthLocal[2U * inputRow], depthLocal,
                static_cast<uint32_t>(tiling_->outH - 2) * compactRow);
            Max(compactLocal, depthLocal, depthLocal[2U * inputRow], 2U * compactRow);
            PipeBarrier<PIPE_V>();
            Max(compactLocal[2U * compactRow], compactLocal[2U * compactRow], depthLocal[4U * inputRow],
                static_cast<uint32_t>(tiling_->outH - 4) * compactRow);
            PipeBarrier<PIPE_V>();
            Duplicate(outputLocal, ZeroValue(), outputPlane);
            PipeBarrier<PIPE_V>();
            PackNdc1hwc0FeatureChannels(outputLocal, compactLocal, 1U, inputBlock, outputBlock, pointCount);
            ClearNdc1hwc0FeatureOutputTail(outputLocal, pointCount, outputBlock);
            CopyOutVector(static_cast<uint64_t>(unit) * outputPlane, outputLocal, outputPlane);
            xInQue_.FreeTensor(inputLocal);
        }
        const uint64_t validOut = static_cast<uint64_t>(totalPlanes) * outputPlane;
        if (validOut < tiling_->totalOut) {
            CopyOutZeroRangeByCore(validOut, tiling_->totalOut - validOut);
        }
    }

    __aicore__ inline bool CanUseNdc1hwc0InputOutputD3W3RowGroupPath() const
    {
        uint32_t inputBlock = 0U;
        uint32_t outputBlock = 0U;
        if (!HasNdc1hwc0InputOutputSingleBlockFeatureLayout(inputBlock, outputBlock) ||
            !MatchesPoolSpec(3, 1, 3, 3, 1, 3, 2, 1, 1, 0, 0, 0) || tiling_->outD != 1 ||
            tiling_->outH != tiling_->inH || tiling_->outW <= 0) {
            return false;
        }
        const uint64_t inputRow = static_cast<uint64_t>(tiling_->inW) * inputBlock;
        const uint64_t compactRow = static_cast<uint64_t>(tiling_->outW) * inputBlock;
        const uint64_t outputRow = static_cast<uint64_t>(tiling_->outW) * outputBlock;
        const uint64_t workerDim = static_cast<uint64_t>(tiling_->blockDim > 0 ? tiling_->blockDim : 1);
        const uint64_t totalRows = static_cast<uint64_t>(tiling_->n) * tiling_->outH;
        const uint64_t maxGroupRows = (totalRows + workerDim - 1U) / workerDim;
        return inputRow * 3U <= INPUT_TILE_NUM && inputRow <= OUTPUT_TILE_NUM && compactRow <= OUTPUT_TILE_NUM &&
               outputRow <= OUTPUT_TILE_NUM && maxGroupRows * inputRow * 3U <= INPUT_TILE_NUM &&
               maxGroupRows * outputRow <= OUTPUT_TILE_NUM && maxGroupRows * tiling_->outW <= 255U;
    }

    __aicore__ inline void ReduceNdc1hwc0D3W3DepthRow(LocalTensor<T> depthLocal, uint32_t nIdx, uint32_t hIdx,
                                                      uint32_t c1, uint32_t inputBlock, uint32_t inputRow)
    {
        const int64_t cBase = static_cast<int64_t>(c1) * inputBlock;
        for (uint32_t kd = 0U; kd < 3U; ++kd) {
            const int64_t id = static_cast<int64_t>(kd) * 2 - tiling_->padFront;
            if (IsOutOfRange(id, tiling_->inD)) {
                continue;
            }
            CopyInVector(InputOffset(nIdx, id, hIdx, 0, cBase), inputRow);
            LocalTensor<T> inputLocal = xInQue_.DeQue<T>();
            if (kd == 0U) {
                Max(depthLocal, inputLocal, inputLocal, inputRow);
            } else {
                Max(depthLocal, depthLocal, inputLocal, inputRow);
            }
            PipeBarrier<PIPE_V>();
            xInQue_.FreeTensor(inputLocal);
        }
    }

    __aicore__ inline void ReduceNdc1hwc0D3W3WidthRow(LocalTensor<T> depthLocal, LocalTensor<T> compactLocal,
                                                      uint32_t inputBlock)
    {
        for (uint32_t ow = 0U; ow < static_cast<uint32_t>(tiling_->outW); ++ow) {
            const int64_t iwBase = static_cast<int64_t>(ow) * tiling_->sW - tiling_->padLeft;
            LocalTensor<T> outBlock = compactLocal[static_cast<uint64_t>(ow) * inputBlock];
            bool hasValue = false;
            for (uint32_t kw = 0U; kw < 3U; ++kw) {
                const int64_t iw = iwBase + static_cast<int64_t>(kw) * tiling_->dilationW;
                if (IsOutOfRange(iw, tiling_->inW)) {
                    continue;
                }
                LocalTensor<T> inBlock = depthLocal[static_cast<uint64_t>(iw) * inputBlock];
                if (!hasValue) {
                    Max(outBlock, inBlock, inBlock, inputBlock);
                    hasValue = true;
                } else {
                    Max(outBlock, outBlock, inBlock, inputBlock);
                }
                PipeBarrier<PIPE_V>();
            }
        }
    }

    __aicore__ inline void ProcessNdc1hwc0InputOutputD3W3RowGroups()
    {
        const uint32_t inputBlock = static_cast<uint32_t>(InputNdc1hwc0Block());
        const uint32_t outputBlock = static_cast<uint32_t>(Ndc1hwc0Block());
        const uint32_t inputRow = static_cast<uint32_t>(tiling_->inW) * inputBlock;
        const uint32_t pointCount = static_cast<uint32_t>(tiling_->outW);
        const uint32_t outputRow = pointCount * outputBlock;
        const uint32_t totalRows = static_cast<uint32_t>(tiling_->n * tiling_->outH);
        const uint32_t worker = GetBlockIdx();
        const uint32_t workerDim = ActiveBlockDim();
        const uint32_t baseRows = totalRows / workerDim;
        const uint32_t extraRows = totalRows - baseRows * workerDim;
        uint32_t row = worker * baseRows + (worker < extraRows ? worker : extraRows);
        uint32_t remainRows = baseRows + (worker < extraRows ? 1U : 0U);
        while (remainRows > 0U) {
            const uint32_t nIdx = row / static_cast<uint32_t>(tiling_->outH);
            const uint32_t hStart = row - nIdx * static_cast<uint32_t>(tiling_->outH);
            uint32_t groupRows = remainRows;
            const uint32_t rowsInN = static_cast<uint32_t>(tiling_->outH) - hStart;
            if (groupRows > rowsInN) {
                groupRows = rowsInN;
            }
            LocalTensor<T> outputLocal = maskBuf_.Get<T>();
            Duplicate(outputLocal, ZeroValue(), groupRows * outputRow);
            PipeBarrier<PIPE_V>();
            for (uint32_t c1 = 0U; c1 < static_cast<uint32_t>(tiling_->inputC1); ++c1) {
                const uint32_t groupElements = groupRows * inputRow;
                LocalTensor<T> inputLocal = xInQue_.AllocTensor<T>();
                const int64_t cBase = static_cast<int64_t>(c1) * inputBlock;
                for (uint32_t kd = 0U; kd < 3U; ++kd) {
                    const int64_t id = static_cast<int64_t>(kd) * tiling_->dilationD - tiling_->padFront;
                    DataCopy(inputLocal[static_cast<uint64_t>(kd) * groupElements],
                             xGm_[InputOffset(nIdx, id, hStart, 0, cBase)], groupElements);
                }
                xInQue_.EnQue(inputLocal);
                inputLocal = xInQue_.DeQue<T>();
                LocalTensor<T> depthLocal = calcBuf_.Get<T>();
                Max(depthLocal, inputLocal, inputLocal[groupElements], groupElements);
                PipeBarrier<PIPE_V>();
                Max(depthLocal, depthLocal, inputLocal[2U * groupElements], groupElements);
                PipeBarrier<PIPE_V>();
                const uint8_t dstStride = static_cast<uint8_t>(outputBlock * sizeof(T) / UB_BLOCK_BYTES);
                const uint8_t srcStride = static_cast<uint8_t>(tiling_->sW * inputBlock * sizeof(T) / UB_BLOCK_BYTES);
                const BinaryRepeatParams firstParams{1U, 1U, 1U, dstStride, srcStride, srcStride};
                const uint8_t repeats = static_cast<uint8_t>(groupRows * pointCount);
                Max(outputLocal[c1 * inputBlock], depthLocal, depthLocal[inputBlock], inputBlock, repeats, firstParams);
                PipeBarrier<PIPE_V>();
                const BinaryRepeatParams finishParams{1U, 1U, 1U, dstStride, dstStride, srcStride};
                Max(outputLocal[c1 * inputBlock], outputLocal[c1 * inputBlock], depthLocal[2U * inputBlock], inputBlock,
                    repeats, finishParams);
                PipeBarrier<PIPE_V>();
                xInQue_.FreeTensor(inputLocal);
            }
            ClearNdc1hwc0FeatureOutputTail(outputLocal, groupRows * pointCount, outputBlock);
            CopyOutVector(static_cast<uint64_t>(row) * outputRow, outputLocal, groupRows * outputRow);
            row += groupRows;
            remainRows -= groupRows;
        }
        const uint64_t validOut = static_cast<uint64_t>(totalRows) * outputRow;
        if (validOut < tiling_->totalOut) {
            CopyOutZeroRangeByCore(validOut, tiling_->totalOut - validOut);
        }
    }

    __aicore__ inline bool CanUseNdc1hwc0InputOutputD2H3W2DepthGroupPath() const
    {
        uint32_t inputBlock = 0U;
        uint32_t outputBlock = 0U;
        if (!HasNdc1hwc0InputOutputSingleBlockFeatureLayout(inputBlock, outputBlock) || inputBlock != outputBlock ||
            tiling_->inputC1 != 1 || !MatchesPoolSpec(2, 3, 2, 1, 2, 1, 2, 2, 1, tiling_->padFront, 2, 0) ||
            tiling_->outD <= 0 || tiling_->outH < 3 || tiling_->outW <= 0 || tiling_->outW != tiling_->inW ||
            tiling_->inH != tiling_->outH * 2 - 1) {
            return false;
        }
        const uint64_t inputPlane = static_cast<uint64_t>(tiling_->inH) * tiling_->inW * inputBlock;
        const uint64_t widthPlane = static_cast<uint64_t>(tiling_->inH) * tiling_->outW * inputBlock;
        const uint64_t outputPlane = static_cast<uint64_t>(tiling_->outH) * tiling_->outW * outputBlock;
        return inputPlane > 0U && inputPlane * 8U <= INPUT_TILE_NUM && widthPlane <= OUTPUT_TILE_NUM &&
               outputPlane > 0U && outputPlane * 6U <= OUTPUT_TILE_NUM && tiling_->inH <= 255 && tiling_->outH <= 255 &&
               static_cast<uint64_t>(tiling_->inW) * inputBlock * sizeof(T) / UB_BLOCK_BYTES <= 255U &&
               static_cast<uint64_t>(tiling_->outW) * outputBlock * sizeof(T) / UB_BLOCK_BYTES <= 255U;
    }

    __aicore__ inline void ReduceNdc1hwc0D2H3W2Plane(LocalTensor<T> inputPlane, LocalTensor<T> widthLocal,
                                                     LocalTensor<T> outputLocal, uint32_t inputBlock)
    {
        const uint32_t inputRow = static_cast<uint32_t>(tiling_->inW) * inputBlock;
        const uint32_t compactRow = static_cast<uint32_t>(tiling_->outW) * inputBlock;
        const uint32_t normalOutW = static_cast<uint32_t>(tiling_->outW - 1);
        const uint32_t inputRowStride = inputRow * sizeof(T) / UB_BLOCK_BYTES;
        const uint32_t compactRowStride = compactRow * sizeof(T) / UB_BLOCK_BYTES;
        const BinaryRepeatParams widthParams{1U,
                                             1U,
                                             1U,
                                             static_cast<uint8_t>(compactRowStride),
                                             static_cast<uint8_t>(inputRowStride),
                                             static_cast<uint8_t>(inputRowStride)};
        Max(widthLocal, inputPlane, inputPlane[inputBlock], normalOutW * inputBlock, static_cast<uint8_t>(tiling_->inH),
            widthParams);
        const BinaryRepeatParams widthTailParams{1U,
                                                 1U,
                                                 1U,
                                                 static_cast<uint8_t>(compactRowStride),
                                                 static_cast<uint8_t>(inputRowStride),
                                                 static_cast<uint8_t>(inputRowStride)};
        Max(widthLocal[static_cast<uint64_t>(normalOutW) * inputBlock],
            inputPlane[static_cast<uint64_t>(tiling_->inW - 1) * inputBlock],
            inputPlane[static_cast<uint64_t>(tiling_->inW - 1) * inputBlock], inputBlock,
            static_cast<uint8_t>(tiling_->inH), widthTailParams);
        PipeBarrier<PIPE_V>();
        Max(outputLocal, widthLocal, widthLocal[2U * compactRow], compactRow);
        Max(outputLocal[static_cast<uint64_t>(tiling_->outH - 1) * compactRow],
            widthLocal[static_cast<uint64_t>(tiling_->inH - 3) * compactRow],
            widthLocal[static_cast<uint64_t>(tiling_->inH - 1) * compactRow], compactRow);
        const uint32_t middleRows = static_cast<uint32_t>(tiling_->outH - 2);
        const BinaryRepeatParams heightFirstParams{1U,
                                                   1U,
                                                   1U,
                                                   static_cast<uint8_t>(compactRowStride),
                                                   static_cast<uint8_t>(2U * compactRowStride),
                                                   static_cast<uint8_t>(2U * compactRowStride)};
        Max(outputLocal[compactRow], widthLocal, widthLocal[2U * compactRow], compactRow,
            static_cast<uint8_t>(middleRows), heightFirstParams);
        PipeBarrier<PIPE_V>();
        const BinaryRepeatParams heightFinishParams{1U,
                                                    1U,
                                                    1U,
                                                    static_cast<uint8_t>(compactRowStride),
                                                    static_cast<uint8_t>(compactRowStride),
                                                    static_cast<uint8_t>(2U * compactRowStride)};
        Max(outputLocal[compactRow], outputLocal[compactRow], widthLocal[4U * compactRow], compactRow,
            static_cast<uint8_t>(middleRows), heightFinishParams);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ProcessNdc1hwc0InputOutputD2H3W2DepthGroups()
    {
        const uint32_t block = static_cast<uint32_t>(InputNdc1hwc0Block());
        const uint32_t inputPlaneElements = static_cast<uint32_t>(tiling_->inH * tiling_->inW * block);
        const uint32_t outputPlaneElements = static_cast<uint32_t>(tiling_->outH * tiling_->outW * block);
        const uint32_t worker = GetBlockIdx();
        const uint32_t workerDim = ActiveBlockDim();
        uint32_t groupsPerN = workerDim / static_cast<uint32_t>(tiling_->n);
        if (groupsPerN == 0U) {
            groupsPerN = 1U;
        }
        if (groupsPerN > static_cast<uint32_t>(tiling_->outD)) {
            groupsPerN = static_cast<uint32_t>(tiling_->outD);
        }
        const uint32_t totalGroups = static_cast<uint32_t>(tiling_->n) * groupsPerN;
        for (uint32_t groupUnit = worker; groupUnit < totalGroups; groupUnit += workerDim) {
            const uint32_t nIdx = groupUnit / groupsPerN;
            const uint32_t group = groupUnit - nIdx * groupsPerN;
            const uint32_t basePlanes = static_cast<uint32_t>(tiling_->outD) / groupsPerN;
            const uint32_t extraPlanes = static_cast<uint32_t>(tiling_->outD) - basePlanes * groupsPerN;
            const uint32_t planeCount = basePlanes + (group < extraPlanes ? 1U : 0U);
            const uint32_t odStart = group * basePlanes + (group < extraPlanes ? group : extraPlanes);
            const int64_t firstInput = static_cast<int64_t>(odStart) * tiling_->sD - tiling_->padFront;
            const int64_t lastInput = static_cast<int64_t>(odStart + planeCount - 1U) * tiling_->sD +
                                      tiling_->dilationD - tiling_->padFront;
            const uint32_t idStart = static_cast<uint32_t>(firstInput > 0 ? firstInput : 0);
            const uint32_t idEnd = static_cast<uint32_t>(lastInput < tiling_->inD - 1 ? lastInput : tiling_->inD - 1);
            const uint32_t inputPlaneCount = idEnd - idStart + 1U;
            const uint64_t inputOffset = (static_cast<uint64_t>(nIdx) * tiling_->inD + idStart) * inputPlaneElements;
            CopyInVector(inputOffset, inputPlaneCount * inputPlaneElements);
            LocalTensor<T> inputLocal = xInQue_.DeQue<T>();
            LocalTensor<T> widthLocal = calcBuf_.Get<T>();
            LocalTensor<T> spatialLocal = tmpBuf_.Get<T>();
            LocalTensor<T> outputLocal = maskBuf_.Get<T>();
            Duplicate(outputLocal, NegInfValue(), planeCount * outputPlaneElements);
            PipeBarrier<PIPE_V>();
            for (uint32_t localId = 0U; localId < inputPlaneCount; ++localId) {
                const uint32_t id = idStart + localId;
                ReduceNdc1hwc0D2H3W2Plane(inputLocal[static_cast<uint64_t>(localId) * inputPlaneElements], widthLocal,
                                          spatialLocal, block);
                for (uint32_t localOd = 0U; localOd < planeCount; ++localOd) {
                    const int64_t od = odStart + localOd;
                    for (uint32_t kd = 0U; kd < 2U; ++kd) {
                        const int64_t sourceDepth = od * tiling_->sD + static_cast<int64_t>(kd) * tiling_->dilationD -
                                                    tiling_->padFront;
                        if (sourceDepth == static_cast<int64_t>(id)) {
                            Max(outputLocal[static_cast<uint64_t>(localOd) * outputPlaneElements],
                                outputLocal[static_cast<uint64_t>(localOd) * outputPlaneElements], spatialLocal,
                                outputPlaneElements);
                        }
                    }
                }
                PipeBarrier<PIPE_V>();
            }
            for (uint32_t localOd = 0U; localOd < planeCount; ++localOd) {
                ClearNdc1hwc0FeatureOutputTail(outputLocal[static_cast<uint64_t>(localOd) * outputPlaneElements],
                                               static_cast<uint32_t>(tiling_->outH * tiling_->outW), block);
            }
            const uint64_t outputOffset = (static_cast<uint64_t>(nIdx) * tiling_->outD + odStart) * outputPlaneElements;
            CopyOutVector(outputOffset, outputLocal, planeCount * outputPlaneElements);
            xInQue_.FreeTensor(inputLocal);
        }
        const uint64_t validOut = static_cast<uint64_t>(tiling_->n) * tiling_->outD * outputPlaneElements;
        if (validOut < tiling_->totalOut) {
            CopyOutZeroRangeByCore(validOut, tiling_->totalOut - validOut);
        }
    }

    __aicore__ inline bool CanUseNdc1hwc0InputOutputDilatedWWholeNPath() const
    {
        uint32_t inputBlock = 0U;
        uint32_t outputBlock = 0U;
        if (!HasNdc1hwc0InputOutputSingleBlockFeatureLayout(inputBlock, outputBlock) ||
            !MatchesPoolSpec(1, 1, 3, 3, 3, 3, 1, 1, 2, tiling_->padFront, tiling_->padTop, tiling_->padLeft) ||
            tiling_->outD <= 0 || tiling_->outH <= 0 || tiling_->outW <= 0) {
            return false;
        }
        const uint64_t inputRow = static_cast<uint64_t>(tiling_->inW) * inputBlock;
        const uint64_t validInputRows = CountNdc1hwc0DilatedWInputRows();
        const uint64_t outputPerN = static_cast<uint64_t>(tiling_->outD) * tiling_->outH * tiling_->outW * outputBlock;
        return validInputRows > 0U && validInputRows * inputRow <= INPUT_TILE_NUM &&
               outputPerN <= NDC1HWC0_D3H3_OUTPUT_TILE_NUM && tiling_->outW <= 255 &&
               static_cast<uint64_t>(outputBlock) * sizeof(T) / UB_BLOCK_BYTES <= 255U &&
               static_cast<uint64_t>(tiling_->sW) * inputBlock * sizeof(T) / UB_BLOCK_BYTES <= 255U;
    }

    __aicore__ inline uint64_t CountNdc1hwc0DilatedWInputRows() const
    {
        uint64_t validDepth = 0U;
        for (int64_t od = 0; od < tiling_->outD; ++od) {
            const int64_t id = od * tiling_->sD - tiling_->padFront;
            validDepth += IsOutOfRange(id, tiling_->inD) ? 0U : 1U;
        }
        uint64_t validHeight = 0U;
        for (int64_t oh = 0; oh < tiling_->outH; ++oh) {
            const int64_t ih = oh * tiling_->sH - tiling_->padTop;
            validHeight += IsOutOfRange(ih, tiling_->inH) ? 0U : 1U;
        }
        return validDepth * validHeight;
    }

    __aicore__ inline void InitNdc1hwc0InputOutputDilatedWBuffers()
    {
        pipe_.InitBuffer(xInQue_, BUFFER_NUM, UbBytesForElements(INPUT_TILE_NUM));
        pipe_.InitBuffer(maskBuf_, UbBytesForElements(NDC1HWC0_D3H3_OUTPUT_TILE_NUM));
        pipe_.InitBuffer(yOutQue_, BUFFER_NUM, UbBytesForElements(OUTPUT_TILE_NUM));
    }

    __aicore__ inline void LoadNdc1hwc0DilatedWInputRows(LocalTensor<T> inputLocal, uint32_t nIdx, uint32_t c1,
                                                         uint32_t inputBlock, uint32_t inputRow)
    {
        uint32_t localRow = 0U;
        const int64_t cBase = static_cast<int64_t>(c1) * inputBlock;
        for (uint32_t od = 0U; od < static_cast<uint32_t>(tiling_->outD); ++od) {
            const int64_t id = static_cast<int64_t>(od) * tiling_->sD - tiling_->padFront;
            if (IsOutOfRange(id, tiling_->inD)) {
                continue;
            }
            for (uint32_t oh = 0U; oh < static_cast<uint32_t>(tiling_->outH); ++oh) {
                const int64_t ih = static_cast<int64_t>(oh) * tiling_->sH - tiling_->padTop;
                if (IsOutOfRange(ih, tiling_->inH)) {
                    continue;
                }
                DataCopy(inputLocal[static_cast<uint64_t>(localRow) * inputRow],
                         xGm_[InputOffset(nIdx, id, ih, 0, cBase)], inputRow);
                ++localRow;
            }
        }
        xInQue_.EnQue(inputLocal);
    }

    __aicore__ inline void ReduceNdc1hwc0DilatedWRows(LocalTensor<T> inputLocal, LocalTensor<T> outputLocal,
                                                      uint32_t c1, uint32_t inputBlock, uint32_t outputBlock,
                                                      uint32_t inputRow)
    {
        for (uint32_t kw = 0U; kw < 3U; ++kw) {
            uint32_t owBegin = 0U;
            while (owBegin < static_cast<uint32_t>(tiling_->outW) &&
                   static_cast<int64_t>(owBegin) * tiling_->sW + static_cast<int64_t>(kw) * tiling_->dilationW -
                           tiling_->padLeft <
                       0) {
                ++owBegin;
            }
            uint32_t owEnd = static_cast<uint32_t>(tiling_->outW);
            while (owEnd > owBegin && static_cast<int64_t>(owEnd - 1U) * tiling_->sW +
                                              static_cast<int64_t>(kw) * tiling_->dilationW - tiling_->padLeft >=
                                          tiling_->inW) {
                --owEnd;
            }
            if (owBegin == owEnd) {
                continue;
            }
            const int64_t iwBegin = static_cast<int64_t>(owBegin) * tiling_->sW +
                                    static_cast<int64_t>(kw) * tiling_->dilationW - tiling_->padLeft;
            const uint8_t dstStride = static_cast<uint8_t>(outputBlock * sizeof(T) / UB_BLOCK_BYTES);
            const uint8_t srcStride = static_cast<uint8_t>(tiling_->sW * inputBlock * sizeof(T) / UB_BLOCK_BYTES);
            const BinaryRepeatParams params{1U, 1U, 1U, dstStride, dstStride, srcStride};
            const uint8_t repeats = static_cast<uint8_t>(owEnd - owBegin);
            uint32_t localRow = 0U;
            for (uint32_t od = 0U; od < static_cast<uint32_t>(tiling_->outD); ++od) {
                const int64_t id = static_cast<int64_t>(od) * tiling_->sD - tiling_->padFront;
                if (IsOutOfRange(id, tiling_->inD)) {
                    continue;
                }
                for (uint32_t oh = 0U; oh < static_cast<uint32_t>(tiling_->outH); ++oh) {
                    const int64_t ih = static_cast<int64_t>(oh) * tiling_->sH - tiling_->padTop;
                    if (IsOutOfRange(ih, tiling_->inH)) {
                        continue;
                    }
                    LocalTensor<T> inputRowLocal = inputLocal[static_cast<uint64_t>(localRow) * inputRow];
                    const uint64_t outputRow = (static_cast<uint64_t>(od) * tiling_->outH + oh) * tiling_->outW *
                                               outputBlock;
                    LocalTensor<T> outputBlockLocal = outputLocal[outputRow +
                                                                  static_cast<uint64_t>(owBegin) * outputBlock +
                                                                  c1 * inputBlock];
                    LocalTensor<T> inputBlockLocal = inputRowLocal[static_cast<uint64_t>(iwBegin) * inputBlock];
                    Max(outputBlockLocal, outputBlockLocal, inputBlockLocal, inputBlock, repeats, params);
                    ++localRow;
                }
            }
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ProcessNdc1hwc0InputOutputDilatedWWholeN()
    {
        const uint32_t inputBlock = static_cast<uint32_t>(InputNdc1hwc0Block());
        const uint32_t outputBlock = static_cast<uint32_t>(Ndc1hwc0Block());
        const uint32_t inputRow = static_cast<uint32_t>(tiling_->inW) * inputBlock;
        const uint32_t outputPointCount = static_cast<uint32_t>(tiling_->outD * tiling_->outH * tiling_->outW);
        const uint32_t outputPerN = outputPointCount * outputBlock;
        const uint32_t worker = GetBlockIdx();
        const uint32_t workerDim = ActiveBlockDim();
        for (uint32_t nIdx = worker; nIdx < static_cast<uint32_t>(tiling_->n); nIdx += workerDim) {
            LocalTensor<T> outputLocal = maskBuf_.Get<T>();
            Duplicate(outputLocal, NegInfValue(), outputPerN);
            PipeBarrier<PIPE_V>();
            for (uint32_t c1 = 0U; c1 < static_cast<uint32_t>(tiling_->inputC1); ++c1) {
                LocalTensor<T> inputLocal = xInQue_.AllocTensor<T>();
                LoadNdc1hwc0DilatedWInputRows(inputLocal, nIdx, c1, inputBlock, inputRow);
                inputLocal = xInQue_.DeQue<T>();
                ReduceNdc1hwc0DilatedWRows(inputLocal, outputLocal, c1, inputBlock, outputBlock, inputRow);
                xInQue_.FreeTensor(inputLocal);
            }
            ClearNdc1hwc0FeatureOutputTail(outputLocal, outputPointCount, outputBlock);
            CopyOutVector(static_cast<uint64_t>(nIdx) * outputPerN, outputLocal, outputPerN);
        }
        const uint64_t validOut = static_cast<uint64_t>(tiling_->n) * outputPerN;
        if (validOut < tiling_->totalOut) {
            CopyOutZeroRangeByCore(validOut, tiling_->totalOut - validOut);
        }
    }

    __aicore__ inline bool CanUseNdc1hwc0InputOutputFeatureRowPath() const
    {
        const uint64_t inputBlock = InputNdc1hwc0Block();
        const uint64_t outputBlock = Ndc1hwc0Block();
        const uint64_t outputValidC1 = Ndc1hwc0ValidC1(outputBlock);
        const uint64_t inputCapacity = static_cast<uint64_t>(tiling_->inputC1) * inputBlock;
        if (tiling_->inputLayout != INPUT_LAYOUT_NDC1HWC0_VALUE ||
            tiling_->outputLayout != OUTPUT_LAYOUT_NDC1HWC0_VALUE || inputBlock * sizeof(T) != UB_BLOCK_BYTES ||
            inputBlock == 0U || outputBlock < inputBlock || outputBlock % inputBlock != 0U || outputValidC1 == 0U ||
            inputCapacity < static_cast<uint64_t>(tiling_->c) || outputValidC1 * outputBlock < inputCapacity ||
            !IsNdc1hwc0CompactPrefix(outputBlock, outputValidC1) || !HasPositivePoolParams() ||
            tiling_->outputD != tiling_->outD || tiling_->outputH != tiling_->outH ||
            tiling_->outputW != tiling_->outW) {
            return false;
        }
        const uint64_t compactRow = static_cast<uint64_t>(tiling_->outW) * inputBlock;
        const uint64_t outputRow = static_cast<uint64_t>(tiling_->outW) * outputBlock;
        const uint64_t inputSpan = static_cast<uint64_t>(tiling_->inW) * inputBlock;
        const uint64_t scratchNeed = Ndc1hwc0GatherTempOffset(static_cast<uint32_t>(compactRow)) + compactRow;
        return compactRow > 0U && compactRow <= OUTPUT_TILE_NUM && outputRow > 0U && outputRow <= OUTPUT_TILE_NUM &&
               inputSpan > 0U && inputSpan <= INPUT_TILE_NUM && scratchNeed <= OUTPUT_TILE_NUM;
    }

    __aicore__ inline void InitNdc1hwc0PhysicalStridedOffsets(LocalTensor<uint32_t> offsetLocal, uint32_t wCount,
                                                              uint32_t inputBlock)
    {
        LocalTensor<int32_t> offsetI32 = offsetLocal.template ReinterpretCast<int32_t>();
        for (uint32_t ow = 0U; ow < wCount; ++ow) {
            for (uint32_t c0 = 0U; c0 < inputBlock; ++c0) {
                offsetI32.SetValue(
                    static_cast<uint64_t>(ow) * inputBlock + c0,
                    static_cast<int32_t>(
                        (static_cast<uint64_t>(ow) * static_cast<uint32_t>(tiling_->sW) * inputBlock + c0) *
                        sizeof(T)));
            }
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ReduceNdc1hwc0PhysicalFeatureW(LocalTensor<T> accLocal, int64_t nIdx, int64_t id, int64_t ih,
                                                          int64_t kw, int64_t cBase, uint32_t inputBlock,
                                                          uint32_t alignedCompactCount)
    {
        uint32_t wStart = 0U;
        uint32_t wCount = 0U;
        if (!CalcValidWRange(kw, wStart, wCount)) {
            return;
        }
        const int64_t iw = DilatedInputWFromStart(wStart, kw);
        const uint32_t validCount = wCount * inputBlock;
        const uint32_t alignedValidCount = AlignToVector(validCount);
        if (tiling_->sW == 1) {
            CopyInVectorPadValue(InputOffset(nIdx, id, ih, iw, cBase), validCount, alignedValidCount, NegInfValue());
            LocalTensor<T> inputLocal = xInQue_.DeQue<T>();
            const uint32_t dstOffset = wStart * inputBlock;
            const uint32_t reduceCount = static_cast<uint64_t>(dstOffset) + alignedValidCount <= alignedCompactCount ?
                                             alignedValidCount :
                                             validCount;
            Max(accLocal[dstOffset], accLocal[dstOffset], inputLocal, reduceCount);
            PipeBarrier<PIPE_V>();
            xInQue_.FreeTensor(inputLocal);
            return;
        }
        const uint32_t spanW = (wCount - 1U) * static_cast<uint32_t>(tiling_->sW) + 1U;
        const uint32_t spanCount = spanW * inputBlock;
        CopyInVectorPadValue(InputOffset(nIdx, id, ih, iw, cBase), spanCount, AlignToVector(spanCount), NegInfValue());
        LocalTensor<T> inputLocal = xInQue_.DeQue<T>();
        LocalTensor<T> scratchLocal = tmpBuf_.Get<T>();
        LocalTensor<uint32_t> offsetLocal = scratchLocal.template ReinterpretCast<uint32_t>();
        InitNdc1hwc0PhysicalStridedOffsets(offsetLocal, wCount, inputBlock);
        LocalTensor<T> gatheredLocal = scratchLocal[Ndc1hwc0GatherTempOffset(validCount)];
        Gather(gatheredLocal, inputLocal, offsetLocal, static_cast<uint32_t>(0), validCount);
        PipeBarrier<PIPE_V>();
        Max(accLocal[static_cast<uint64_t>(wStart) * inputBlock], accLocal[static_cast<uint64_t>(wStart) * inputBlock],
            gatheredLocal, validCount);
        PipeBarrier<PIPE_V>();
        xInQue_.FreeTensor(inputLocal);
    }

    __aicore__ inline void ComputeNdc1hwc0PhysicalFeatureChunk(LocalTensor<T> accLocal, int64_t nIdx, int64_t od,
                                                               int64_t oh, int64_t cBase, uint32_t inputBlock,
                                                               uint32_t alignedCompactCount)
    {
        Duplicate(accLocal, NegInfValue(), alignedCompactCount);
        PipeBarrier<PIPE_V>();
        for (int64_t kd = 0; kd < tiling_->kD; ++kd) {
            const int64_t id = DilatedInputD(od, kd);
            if (IsOutOfRange(id, tiling_->inD)) {
                continue;
            }
            for (int64_t kh = 0; kh < tiling_->kH; ++kh) {
                const int64_t ih = DilatedInputH(oh, kh);
                if (IsOutOfRange(ih, tiling_->inH)) {
                    continue;
                }
                for (int64_t kw = 0; kw < tiling_->kW; ++kw) {
                    ReduceNdc1hwc0PhysicalFeatureW(accLocal, nIdx, id, ih, kw, cBase, inputBlock, alignedCompactCount);
                }
            }
        }
    }

    __aicore__ inline void ProcessNdc1hwc0InputOutputFeatureRows()
    {
        const uint32_t inputBlock = static_cast<uint32_t>(InputNdc1hwc0Block());
        const uint32_t outputBlock = static_cast<uint32_t>(Ndc1hwc0Block());
        const uint32_t outputValidC1 = static_cast<uint32_t>(Ndc1hwc0ValidC1(outputBlock));
        const uint32_t compactCount = static_cast<uint32_t>(tiling_->outW) * inputBlock;
        const uint32_t alignedCompactCount = AlignToVector(compactCount);
        const uint32_t outputRowCount = static_cast<uint32_t>(tiling_->outW) * outputBlock;
        const uint64_t totalRows = static_cast<uint64_t>(tiling_->n) * static_cast<uint64_t>(tiling_->outD) *
                                   outputValidC1 * static_cast<uint64_t>(tiling_->outH);
        const uint64_t worker = GetBlockIdx();
        const uint64_t workerDim = ActiveBlockDim();
        for (uint64_t row = worker; row < totalRows; row += workerDim) {
            uint64_t q = row;
            const int64_t oh = static_cast<int64_t>(q % static_cast<uint64_t>(tiling_->outH));
            q /= static_cast<uint64_t>(tiling_->outH);
            const uint32_t outputC1 = static_cast<uint32_t>(q % outputValidC1);
            q /= outputValidC1;
            const int64_t od = static_cast<int64_t>(q % static_cast<uint64_t>(tiling_->outD));
            const int64_t nIdx = static_cast<int64_t>(q / static_cast<uint64_t>(tiling_->outD));
            LocalTensor<T> outputLocal = maskBuf_.Get<T>();
            Duplicate(outputLocal, ZeroValue(), outputRowCount);
            PipeBarrier<PIPE_V>();
            for (uint32_t inputC1 = 0U; inputC1 < static_cast<uint32_t>(tiling_->inputC1); ++inputC1) {
                const uint32_t channelBase = inputC1 * inputBlock;
                if (channelBase / outputBlock != outputC1) {
                    continue;
                }
                LocalTensor<T> accLocal = calcBuf_.Get<T>();
                ComputeNdc1hwc0PhysicalFeatureChunk(accLocal, nIdx, od, oh, static_cast<int64_t>(channelBase),
                                                    inputBlock, alignedCompactCount);
                ZeroNdc1hwc0Tail(accLocal, static_cast<uint32_t>(tiling_->outW), inputBlock,
                                 Ndc1hwc0InputActiveChannels(static_cast<int64_t>(inputC1), inputBlock));
                const uint32_t outputC0 = channelBase - outputC1 * outputBlock;
                const DataCopyParams packParams{
                    static_cast<uint16_t>(tiling_->outW), 1U, 0U,
                    static_cast<uint16_t>((outputBlock - inputBlock) * sizeof(T) / UB_BLOCK_BYTES)};
                DataCopy(outputLocal[outputC0], accLocal, packParams);
                DataSyncBarrier<MemDsbT::UB>();
            }
            CopyOutVector(row * outputRowCount, outputLocal, outputRowCount);
        }
        const uint64_t validOut = totalRows * outputRowCount;
        if (validOut < tiling_->totalOut) {
            CopyOutZeroRangeByCore(validOut, tiling_->totalOut - validOut);
        }
    }

    __aicore__ inline bool CanUseNdc1hwc0StorageRowVectorPath() const
    {
        if (tiling_->outputLayout != OUTPUT_LAYOUT_NDC1HWC0_VALUE || tiling_->outW <= 0 || tiling_->c <= 0 ||
            tiling_->sW <= 0 || tiling_->dilationW <= 0 || tiling_->kW <= 0) {
            return false;
        }
        const uint64_t storageD = Ndc1hwc0StorageD();
        const uint64_t storageH = Ndc1hwc0StorageH();
        const uint64_t storageW = Ndc1hwc0StorageW();
        const uint64_t storageC1 = Ndc1hwc0StorageC1();
        const uint64_t storageC0 = Ndc1hwc0StorageC0();
        if (storageD == 0U || storageH == 0U || storageW == 0U || storageC1 == 0U || storageC0 == 0U ||
            storageW < static_cast<uint64_t>(tiling_->outW)) {
            return false;
        }
        const uint64_t rowElements = storageW * storageC0;
        const uint64_t validRowElements = static_cast<uint64_t>(tiling_->outW) * storageC0;
        if (rowElements == 0U || rowElements > OUTPUT_TILE_NUM || validRowElements == 0U ||
            validRowElements > OUTPUT_TILE_NUM) {
            return false;
        }
        if (tiling_->dataFormat == FORMAT_NDHWC_VALUE) {
            const uint64_t inputNeed = static_cast<uint64_t>(tiling_->outW) *
                                       static_cast<uint64_t>(AlignToVector(static_cast<uint32_t>(storageC0)));
            return inputNeed <= INPUT_TILE_NUM;
        }
        const uint64_t alignedW = static_cast<uint64_t>(AlignToVector(static_cast<uint32_t>(tiling_->outW)));
        return storageC0 * alignedW <= OUTPUT_TILE_NUM && storageC0 * alignedW <= INPUT_TILE_NUM;
    }

    __aicore__ inline uint32_t ProcessNdc1hwc0StoragePartialRange(uint64_t cur, uint64_t outEnd)
    {
        const uint64_t remain = outEnd - cur;
        const uint32_t curCount = remain > OUTPUT_TILE_NUM ? OUTPUT_TILE_NUM : static_cast<uint32_t>(remain);
        LocalTensor<T> yLocal = yOutQue_.AllocTensor<T>();
        for (uint32_t i = 0; i < curCount; ++i) {
            yLocal.SetValue(i, ComputeNdc1hwc0OutputValue(cur + static_cast<uint64_t>(i)));
        }
        yOutQue_.EnQue(yLocal);
        LocalTensor<T> yOut = yOutQue_.DeQue<T>();
        DataCopyExtParams copyParams{1U, static_cast<uint32_t>(curCount * sizeof(T)), 0U, 0U, 0U};
        DataCopyPad(yGm_[cur], yOut, copyParams);
        yOutQue_.FreeTensor(yOut);
        return curCount;
    }

    __aicore__ inline void ProcessNdc1hwc0StorageRowVector()
    {
        const uint64_t storageD = Ndc1hwc0StorageD();
        const uint64_t storageH = Ndc1hwc0StorageH();
        const uint64_t storageW = Ndc1hwc0StorageW();
        const uint64_t storageC1 = Ndc1hwc0StorageC1();
        const uint64_t storageC0 = Ndc1hwc0StorageC0();
        if (storageD == 0U || storageH == 0U || storageW == 0U || storageC1 == 0U || storageC0 == 0U) {
            return;
        }
        const uint64_t rowElements = storageW * storageC0;
        if (rowElements == 0U || rowElements > OUTPUT_TILE_NUM) {
            ProcessAivDirectFast();
            return;
        }
        const uint64_t outOffset = Ndc1hwc0ValidCoreStartOffset(tiling_->totalOut, rowElements);
        const uint64_t outCount = Ndc1hwc0ValidCoreElementCount(tiling_->totalOut, rowElements, outOffset);
        const uint64_t outEnd = outOffset + outCount;
        uint64_t cur = outOffset;
        while (cur < outEnd) {
            if (cur % rowElements != 0U || outEnd - cur < rowElements) {
                cur += ProcessNdc1hwc0StoragePartialRange(cur, outEnd);
                continue;
            }
            const uint32_t rowElements32 = static_cast<uint32_t>(rowElements);
            uint32_t maxTileRows = OUTPUT_TILE_NUM / rowElements32;
            if (maxTileRows == 0U) {
                maxTileRows = 1U;
            }
            const uint64_t remainRows64 = (outEnd - cur) / rowElements;
            uint32_t tileRows = remainRows64 > maxTileRows ? maxTileRows : static_cast<uint32_t>(remainRows64);
            if (tileRows == 0U) {
                ProcessNdc1hwc0StorageRowVectorByRow(cur / rowElements, cur, static_cast<uint32_t>(storageC0),
                                                     static_cast<uint32_t>(storageW));
                cur += rowElements;
                continue;
            }
            ProcessNdc1hwc0StorageRowVectorTile(cur / rowElements, tileRows, cur, static_cast<uint32_t>(storageC0),
                                                static_cast<uint32_t>(storageW));
            cur += static_cast<uint64_t>(tileRows) * rowElements;
        }
    }

    __aicore__ inline void ProcessNdc1hwc0StorageRowVectorTile(uint64_t startRow, uint32_t rowCount,
                                                               uint64_t outputOffset, uint32_t storageC0,
                                                               uint32_t storageW)
    {
        const uint32_t rowElements = storageW * storageC0;
        LocalTensor<T> yLocal = maskBuf_.Get<T>();
        Duplicate(yLocal, ZeroValue(), rowElements * rowCount);
        PipeBarrier<PIPE_V>();
        for (uint32_t rowIdx = 0; rowIdx < rowCount; ++rowIdx) {
            FillNdc1hwc0StorageRowVector(startRow + rowIdx, yLocal[static_cast<uint64_t>(rowIdx) * rowElements],
                                         storageC0, storageW);
        }
        CopyOutVector(outputOffset, yLocal, rowElements * rowCount);
    }

    __aicore__ inline void ProcessNdc1hwc0StorageRowVectorByRow(uint64_t row, uint64_t outputOffset, uint32_t storageC0,
                                                                uint32_t storageW)
    {
        const uint32_t rowElements = storageW * storageC0;
        LocalTensor<T> rowLocal = maskBuf_.Get<T>();
        Duplicate(rowLocal, ZeroValue(), rowElements);
        PipeBarrier<PIPE_V>();
        FillNdc1hwc0StorageRowVector(row, rowLocal, storageC0, storageW);
        CopyOutVector(outputOffset, rowLocal, rowElements);
    }

    __aicore__ inline void FillNdc1hwc0StorageRowVector(uint64_t row, LocalTensor<T> rowLocal, uint32_t storageC0,
                                                        uint32_t storageW)
    {
        int64_t nIdx = 0;
        int64_t od = 0;
        int64_t c1Idx = 0;
        int64_t oh = 0;
        DecodeNdc1hwc0StorageRow(row, nIdx, od, c1Idx, oh);
        if (nIdx < tiling_->n && od < tiling_->outD && oh < tiling_->outH &&
            storageW >= static_cast<uint32_t>(tiling_->outW)) {
            const int64_t cBase = c1Idx * static_cast<int64_t>(storageC0);
            int64_t activeChannels = tiling_->c - cBase;
            if (activeChannels > static_cast<int64_t>(storageC0)) {
                activeChannels = static_cast<int64_t>(storageC0);
            }
            if (activeChannels > 0) {
                if (tiling_->dataFormat == FORMAT_NDHWC_VALUE) {
                    ComputeNdc1hwc0NdhwcRowVector(rowLocal, nIdx, od, cBase, oh, static_cast<uint32_t>(activeChannels),
                                                  storageC0);
                } else {
                    ComputeNdc1hwc0NcdhwRowVector(rowLocal, nIdx, od, cBase, oh, static_cast<uint32_t>(activeChannels),
                                                  storageC0);
                }
            }
        }
    }

    __aicore__ inline bool CanUseNdc1hwc0NdhwcD3H3Dil2GroupPath() const
    {
        if (tiling_->dataFormat != FORMAT_NDHWC_VALUE || tiling_->outW <= 0 || tiling_->outH <= 0 ||
            tiling_->inH <= 0 || tiling_->c <= 0) {
            return false;
        }
        const uint64_t block = Ndc1hwc0Block();
        if (block == 0U || tiling_->outputC1 != 1 || static_cast<uint64_t>(tiling_->c) > block) {
            return false;
        }
        if (!HasNdc1hwc0D3H3Dil2PoolSpec() || tiling_->outW != tiling_->inW) {
            return false;
        }
        const uint32_t cCount = static_cast<uint32_t>(tiling_->c);
        const uint32_t outW = static_cast<uint32_t>(tiling_->outW);
        const uint32_t rowCount = outW * cCount;
        const uint32_t zeroIndex = AlignToVector(rowCount);
        const uint32_t compactStride = AlignToVector(zeroIndex + 1U);
        const uint32_t rowElements = static_cast<uint32_t>(outW * static_cast<uint32_t>(block));
        const uint32_t scatterScratch = Ndc1hwc0GatherTempOffset(rowElements);
        const uint64_t dmaxCount = static_cast<uint64_t>(tiling_->inH) * compactStride;
        const bool fullDWindow = tiling_->outD > 0 && (tiling_->outD - 1) * 3 + 2 < tiling_->inD;
        return HasNdhwcD3H3BufferCapacity(rowCount, rowElements, zeroIndex, compactStride, dmaxCount, scatterScratch,
                                          cCount, static_cast<uint32_t>(block)) &&
               fullDWindow;
    }

    __aicore__ inline bool HasNdc1hwc0D3H3Dil2PoolSpec() const
    {
        return tiling_->kD == 3 && tiling_->kH == 3 && tiling_->kW == 1 && tiling_->sD == 3 && tiling_->sH == 1 &&
               tiling_->sW == 1 && tiling_->dilationD == 1 && tiling_->dilationH == 2 && tiling_->dilationW == 1 &&
               tiling_->padFront == 0 && tiling_->padTop == 2 && tiling_->padLeft == 0;
    }

    __aicore__ inline bool HasNdhwcD3H3BufferCapacity(uint32_t rowCount, uint32_t rowElements, uint32_t zeroIndex,
                                                      uint32_t compactStride, uint64_t dmaxCount,
                                                      uint32_t scatterScratch, uint32_t cCount, uint32_t block) const
    {
        return rowCount > 0U && rowElements > 0U && rowElements <= OUTPUT_TILE_NUM && zeroIndex < compactStride &&
               compactStride <= INPUT_TILE_NUM && dmaxCount + scatterScratch <= NDC1HWC0_D3H3_OUTPUT_TILE_NUM &&
               dmaxCount <= NDC1HWC0_D3H3_OUTPUT_TILE_NUM && zeroIndex <= INPUT_TILE_NUM && cCount <= block &&
               block > 0U;
    }

    __aicore__ inline bool CanUseNdc1hwc0NdhwcK1DirectGroupPath() const
    {
        if (tiling_->dataFormat != FORMAT_NDHWC_VALUE || tiling_->outW <= 0 || tiling_->outH <= 0 || tiling_->c <= 0) {
            return false;
        }
        if (!CanUseNdc1hwc0NdhwcCompactK1Direct()) {
            return false;
        }
        const uint64_t block = Ndc1hwc0Block();
        if (block == 0U || tiling_->outputC1 != 1 || static_cast<uint64_t>(tiling_->c) > block) {
            return false;
        }
        const uint32_t cCount = static_cast<uint32_t>(tiling_->c);
        const uint32_t outW = static_cast<uint32_t>(tiling_->outW);
        const uint32_t rowCount = outW * cCount;
        const uint32_t rowElements = static_cast<uint32_t>(static_cast<uint64_t>(outW) * block);
        const uint32_t scatterScratch = Ndc1hwc0GatherTempOffset(rowElements);
        return rowCount > 0U && rowElements > 0U && rowElements <= OUTPUT_TILE_NUM && rowCount + 1U <= INPUT_TILE_NUM &&
               rowCount + 1U <= OUTPUT_TILE_NUM && rowCount + 1U + scatterScratch <= OUTPUT_TILE_NUM;
    }

    __aicore__ inline bool CanUseNdc1hwc0NdhwcK1BalancedDirectPath() const
    {
        if (!CanUseNdc1hwc0NdhwcK1DirectGroupPath()) {
            return false;
        }
        const uint64_t block = Ndc1hwc0Block();
        const uint32_t cCount = static_cast<uint32_t>(tiling_->c);
        const uint32_t alignedC = AlignToVector(cCount);
        const uint32_t outW = static_cast<uint32_t>(tiling_->outW);
        const uint32_t rowElements = static_cast<uint32_t>(static_cast<uint64_t>(outW) * block);
        return block == 16U && alignedC == static_cast<uint32_t>(block) && rowElements > 0U &&
               tiling_->outD == tiling_->inD && tiling_->outH == tiling_->inH && tiling_->outW == tiling_->inW;
    }

    __aicore__ inline bool CanUseNdc1hwc0NdhwcK1MinimalBufferPath() const
    {
        if (tiling_->dataFormat != FORMAT_NDHWC_VALUE || tiling_->inputLayout == INPUT_LAYOUT_NDC1HWC0_VALUE ||
            tiling_->outputLayout != OUTPUT_LAYOUT_NDC1HWC0_VALUE) {
            return false;
        }
        const uint64_t rowElements = static_cast<uint64_t>(tiling_->outW) * Ndc1hwc0Block();
        return rowElements > 0U && rowElements <= INPUT_TILE_NUM && CanUseNdc1hwc0NdhwcK1BalancedDirectPath();
    }

    __aicore__ inline void ProcessNdc1hwc0NdhwcK1BalancedDirect()
    {
        const uint64_t block = Ndc1hwc0Block();
        const uint64_t validC1 = Ndc1hwc0ValidC1(block);
        const uint64_t validOut = Ndc1hwc0ValidOut(block, validC1);
        if (block == 0U || validC1 != 1U || validOut == 0U) {
            return;
        }
        const uint32_t outW = static_cast<uint32_t>(tiling_->outW);
        const uint32_t rowElements = static_cast<uint32_t>(static_cast<uint64_t>(outW) * block);
        if (rowElements == 0U) {
            return;
        }
        const uint64_t totalRows = validOut / rowElements;
        const uint64_t blockDim = static_cast<uint64_t>(ActiveBlockDim());
        const uint64_t blockIdx = static_cast<uint64_t>(GetBlockIdx());
        const uint64_t workerDim = blockDim > totalRows ? totalRows : blockDim;
        if (blockIdx >= workerDim) {
            return;
        }
        const uint64_t startRow = totalRows * blockIdx / workerDim;
        const uint64_t endRow = totalRows * (blockIdx + 1U) / workerDim;
        if (endRow > startRow) {
            int64_t nIdx = 0;
            int64_t od = 0;
            int64_t c1Idx = 0;
            int64_t oh = 0;
            DecodeNdc1hwc0Row(startRow, validC1, nIdx, od, c1Idx, oh);
            if (c1Idx == 0) {
                const uint32_t rows = static_cast<uint32_t>(endRow - startRow);
                const uint32_t cCount = static_cast<uint32_t>(tiling_->c);
                CopyInVectorWBlocksPadStrideValue(InputOffset(nIdx, od, oh, 0, 0), rows * outW, cCount,
                                                  static_cast<uint32_t>(block), 0U, ZeroValue());
                LocalTensor<T> xLocal = xInQue_.DeQue<T>();
                CopyOutVectorFromMte2Fast(startRow * static_cast<uint64_t>(rowElements), xLocal, rows * rowElements);
                xInQue_.FreeTensor(xLocal);
            }
        }
        if (validOut < tiling_->totalOut) {
            CopyOutZeroRangeByWorker(validOut, tiling_->totalOut - validOut, workerDim);
        }
    }

    __aicore__ inline void CopyOutZeroRangeByWorker(uint64_t outOffset, uint64_t outCount, uint64_t workerDim)
    {
        if (outCount == 0U || workerDim == 0U) {
            return;
        }
        const uint64_t blockIdx = static_cast<uint64_t>(GetBlockIdx());
        if (blockIdx >= workerDim) {
            return;
        }
        const uint64_t start = outCount * blockIdx / workerDim;
        const uint64_t end = outCount * (blockIdx + 1U) / workerDim;
        if (end > start) {
            CopyOutZeroRange(outOffset + start, end - start);
        }
    }

    __aicore__ inline void ProcessNdc1hwc0NdhwcK1DirectGroupStep(uint64_t& cur, uint64_t outEnd, uint64_t block,
                                                                 uint64_t validC1, uint32_t cCount,
                                                                 uint64_t rowElements)
    {
        if (rowElements == 0U) {
            cur = outEnd;
            return;
        }
        if (!PrepareNdc1hwc0FullRow(cur, outEnd, block, rowElements, validC1)) {
            return;
        }
        const uint64_t row = cur / rowElements;
        int64_t nIdx = 0;
        int64_t od = 0;
        int64_t c1Idx = 0;
        int64_t oh = 0;
        DecodeNdc1hwc0Row(row, validC1, nIdx, od, c1Idx, oh);
        if (c1Idx != 0) {
            ProcessNdc1hwc0RowVectorByRow(row, cur, block, validC1);
            cur += rowElements;
            return;
        }
        const bool linearRows = validC1 == 1U && tiling_->outD == tiling_->inD && tiling_->outH == tiling_->inH &&
                                tiling_->outW == tiling_->inW;
        uint32_t rows = linearRows ? static_cast<uint32_t>((outEnd - cur) / rowElements) :
                                     static_cast<uint32_t>(tiling_->outH - oh);
        const uint64_t remainRows = (outEnd - cur) / rowElements;
        if (static_cast<uint64_t>(rows) > remainRows) {
            rows = static_cast<uint32_t>(remainRows);
        }
        const uint32_t maxRowsByOutput = OUTPUT_TILE_NUM / static_cast<uint32_t>(rowElements);
        if (rows > maxRowsByOutput) {
            rows = maxRowsByOutput;
        }
        const uint32_t alignedCount = AlignToVector(static_cast<uint32_t>(tiling_->outW) * cCount + 1U);
        const uint32_t maxRowsByInput = alignedCount == 0U ? 1U : INPUT_TILE_NUM / alignedCount;
        if (maxRowsByInput > 0U && rows > maxRowsByInput) {
            rows = maxRowsByInput;
        }
        if (rows == 0U) {
            ProcessNdc1hwc0RowVectorByRow(row, cur, block, validC1);
            cur += rowElements;
            return;
        }
        ProcessNdc1hwc0NdhwcK1DirectGroupTile(cur, nIdx, od, static_cast<uint32_t>(oh), rows, cCount,
                                              static_cast<uint32_t>(block));
        cur += static_cast<uint64_t>(rows) * rowElements;
    }

    __aicore__ inline void ProcessNdc1hwc0NdhwcK1DirectGroup()
    {
        Ndc1hwc0LinearRangeContext range{};
        if (!InitNdc1hwc0ActiveLinearRange(range)) {
            return;
        }

        const uint32_t cCount = static_cast<uint32_t>(tiling_->c);
        uint64_t cur = range.outOffset;
        while (cur < range.outEnd) {
            ProcessNdc1hwc0NdhwcK1DirectGroupStep(cur, range.outEnd, range.block, range.validC1, cCount,
                                                  range.rowElements);
        }
        if (range.validOut < tiling_->totalOut) {
            CopyOutZeroRangeByCore(range.validOut, tiling_->totalOut - range.validOut);
        }
    }

    __aicore__ inline void ProcessNdc1hwc0NdhwcK1DirectGroupTile(uint64_t outputOffset, int64_t nIdx, int64_t od,
                                                                 uint32_t ohStart, uint32_t rows, uint32_t cCount,
                                                                 uint32_t block)
    {
        const uint32_t outW = static_cast<uint32_t>(tiling_->outW);
        const uint32_t rowCount = outW * cCount;
        const uint32_t alignedCount = AlignToVector(rowCount + 1U);
        const uint32_t rowElements = outW * block;
        const uint32_t alignedC = AlignToVector(cCount);
        const uint32_t directRowStride = outW * alignedC;
        if (alignedC >= cCount && directRowStride > 0U &&
            static_cast<uint64_t>(rows) * directRowStride <= INPUT_TILE_NUM &&
            (static_cast<uint64_t>(alignedC) * sizeof(T)) % UB_BLOCK_BYTES == 0U) {
            CopyInVectorWBlocksPadStrideValue(InputOffset(nIdx, od, ohStart, 0, 0), rows * outW, cCount, alignedC,
                                              static_cast<uint32_t>(tiling_->c - static_cast<int64_t>(cCount)),
                                              ZeroValue());
            LocalTensor<T> xLocal = xInQue_.DeQue<T>();
            if (alignedC == block && directRowStride == rowElements) {
                CopyOutVectorFromMte2(outputOffset, xLocal, rows * rowElements);
                xInQue_.FreeTensor(xLocal);
                return;
            }
            LocalTensor<T> outLocal = maskBuf_.Get<T>();
            if (CopyOutNdc1hwc0NdhwcAlignedCDirect(outputOffset, xLocal, outLocal, rows, directRowStride, cCount,
                                                   alignedC, block, outW, true, OUTPUT_TILE_NUM)) {
                xInQue_.FreeTensor(xLocal);
                return;
            }
            xInQue_.FreeTensor(xLocal);
        }
        LocalTensor<T> scratchLocal = tmpBuf_.Get<T>();
        LocalTensor<uint32_t> offsetLocal = scratchLocal.template ReinterpretCast<uint32_t>();
        InitNdc1hwc0NdhwcCompactGatherOffsets(offsetLocal, cCount, block, outW, rowCount);
        LocalTensor<T> outLocal = maskBuf_.Get<T>();
        CopyInVectorWBlocksPadStrideValue(InputOffset(nIdx, od, ohStart, 0, 0), rows, rowCount, alignedCount, 0U,
                                          ZeroValue());
        LocalTensor<T> xLocal = xInQue_.DeQue<T>();
        for (uint32_t rowIdx = 0; rowIdx < rows; ++rowIdx) {
            Gather(outLocal[static_cast<uint64_t>(rowIdx) * rowElements],
                   xLocal[static_cast<uint64_t>(rowIdx) * alignedCount], offsetLocal, static_cast<uint32_t>(0),
                   rowElements);
            PipeBarrier<PIPE_V>();
        }
        xInQue_.FreeTensor(xLocal);
        CopyOutVector(outputOffset, outLocal, rows * rowElements);
    }

    __aicore__ inline bool CanUseNdc1hwc0NdhwcK1FullC1PlanePath() const
    {
        if (tiling_->dataFormat != FORMAT_NDHWC_VALUE || !CanUseNdc1hwc0NdhwcCompactK1Direct() || tiling_->outW <= 0 ||
            tiling_->outH <= 0 || tiling_->outD <= 0 || tiling_->c <= 0) {
            return false;
        }
        const uint64_t block = Ndc1hwc0Block();
        const uint64_t validC1 = Ndc1hwc0ValidC1(block);
        if (block == 0U || validC1 <= 1U || validC1 > 8U || block > 255U) {
            return false;
        }
        const uint32_t outW = static_cast<uint32_t>(tiling_->outW);
        const uint32_t outH = static_cast<uint32_t>(tiling_->outH);
        const uint32_t rowElements = static_cast<uint32_t>(static_cast<uint64_t>(outW) * block);
        const uint32_t planeRows = static_cast<uint32_t>(validC1) * outH;
        const uint32_t outputCount = planeRows * rowElements;
        const uint32_t c0Count = static_cast<uint32_t>(
            block < static_cast<uint64_t>(tiling_->c) ? block : static_cast<uint64_t>(tiling_->c));
        const uint32_t alignedC0 = AlignToVector(c0Count);
        const uint32_t inputNeed = planeRows * alignedC0;
        return rowElements > 0U && outputCount > 0U && outputCount <= OUTPUT_TILE_NUM && c0Count > 0U &&
               alignedC0 <= 255U && inputNeed <= INPUT_TILE_NUM && tiling_->normalCoreOut >= outputCount &&
               tiling_->normalCoreOut % outputCount == 0U;
    }

    struct Ndc1hwc0PlaneGroupContext {
        int64_t nIdx = 0;
        int64_t od = 0;
        uint32_t groupPlanes = 0U;
    };

    __aicore__ inline bool PrepareNdc1hwc0FullPlane(uint64_t& cur, uint64_t outEnd, uint64_t block, uint64_t validC1,
                                                    uint32_t rowElements, uint32_t outputCount)
    {
        if (rowElements == 0U || outputCount == 0U) {
            cur = outEnd;
            return false;
        }
        if (cur % outputCount == 0U && outEnd - cur >= outputCount) {
            return true;
        }
        const uint64_t row = cur / rowElements;
        ProcessNdc1hwc0RowVectorByRow(row, cur, block, validC1);
        cur += rowElements - (cur - row * static_cast<uint64_t>(rowElements));
        return false;
    }

    __aicore__ inline Ndc1hwc0PlaneGroupContext GetNdc1hwc0PlaneGroupContext(uint64_t cur, uint64_t outEnd,
                                                                             uint32_t outputCount,
                                                                             uint32_t maxGroupPlanes) const
    {
        Ndc1hwc0PlaneGroupContext context{};
        if (outputCount == 0U || tiling_->outD <= 0) {
            return context;
        }
        const uint64_t planeIdx = cur / outputCount;
        context.od = static_cast<int64_t>(planeIdx % static_cast<uint64_t>(tiling_->outD));
        context.nIdx = static_cast<int64_t>(planeIdx / static_cast<uint64_t>(tiling_->outD));
        context.groupPlanes = maxGroupPlanes;
        const uint64_t remainPlanes = (outEnd - cur) / outputCount;
        if (static_cast<uint64_t>(context.groupPlanes) > remainPlanes) {
            context.groupPlanes = static_cast<uint32_t>(remainPlanes);
        }
        const uint32_t remainD = static_cast<uint32_t>(tiling_->outD - context.od);
        if (context.groupPlanes > remainD) {
            context.groupPlanes = remainD;
        }
        return context;
    }

    __aicore__ inline void ProcessNdc1hwc0NdhwcK1FullC1Plane()
    {
        uint64_t block = 0U, validC1 = 0U, validOut = 0U;
        uint32_t outW = 0U, outH = 0U, rowElements = 0U, outputCount = 0U;
        if (!InitNdc1hwc0FullPlaneGeometry(block, validC1, validOut, outW, outH, rowElements, outputCount)) {
            return;
        }
        const uint32_t maxGroupPlanes = NdhwcK1FullC1MaxDGroup(static_cast<uint32_t>(validC1),
                                                               static_cast<uint32_t>(block), outH);
        uint64_t outOffset = 0U, outCount = 0U, outEnd = 0U;
        GetNdc1hwc0ValidCoreRange(validOut, rowElements, outOffset, outCount, outEnd);
        uint64_t cur = outOffset;
        while (cur < outEnd) {
            if (!PrepareNdc1hwc0FullPlane(cur, outEnd, block, validC1, rowElements, outputCount)) {
                continue;
            }
            const Ndc1hwc0PlaneGroupContext context = GetNdc1hwc0PlaneGroupContext(cur, outEnd, outputCount,
                                                                                   maxGroupPlanes);
            if (context.groupPlanes > 1U &&
                ProcessNdc1hwc0NdhwcK1FullC1DGroupTile(cur, context.nIdx, context.od, context.groupPlanes,
                                                       static_cast<uint32_t>(validC1), static_cast<uint32_t>(block),
                                                       outH, outW)) {
                cur += static_cast<uint64_t>(context.groupPlanes) * outputCount;
                continue;
            }
            ProcessNdc1hwc0NdhwcK1FullC1PlaneTile(cur, context.nIdx, context.od, static_cast<uint32_t>(validC1),
                                                  static_cast<uint32_t>(block), outH, outW);
            cur += outputCount;
        }
        if (validOut < tiling_->totalOut) {
            CopyOutZeroRangeByCore(validOut, tiling_->totalOut - validOut);
        }
    }

    __aicore__ inline uint32_t NdhwcK1FullC1MaxDGroup(uint32_t validC1, uint32_t block, uint32_t outH) const
    {
        if (validC1 == 0U || block == 0U || outH == 0U || tiling_->outW <= 0) {
            return 1U;
        }
        const uint32_t rowElements = static_cast<uint32_t>(static_cast<uint64_t>(tiling_->outW) * block);
        const uint32_t outputBlock = outH * rowElements;
        uint32_t maxGroup = (outputBlock == 0U || rowElements == 0U) ? 1U : OUTPUT_TILE_NUM / outputBlock;
        if (maxGroup == 0U) {
            return 1U;
        }
        if (maxGroup > static_cast<uint32_t>(tiling_->outD)) {
            maxGroup = static_cast<uint32_t>(tiling_->outD);
        }
        const uint32_t c0Count = static_cast<uint32_t>(
            block < static_cast<uint64_t>(tiling_->c) ? block : static_cast<uint64_t>(tiling_->c));
        const uint32_t alignedC0 = AlignToVector(c0Count);
        while (maxGroup > 1U) {
            if (static_cast<uint64_t>(outH) * static_cast<uint64_t>(tiling_->outW) * static_cast<uint64_t>(maxGroup) *
                    alignedC0 <=
                INPUT_TILE_NUM) {
                break;
            }
            --maxGroup;
        }
        return maxGroup == 0U ? 1U : maxGroup;
    }

    __aicore__ inline bool ProcessNdc1hwc0NdhwcK1FullC1DGroupTile(uint64_t outputOffset, int64_t nIdx, int64_t od,
                                                                  uint32_t groupPlanes, uint32_t validC1,
                                                                  uint32_t block, uint32_t outH, uint32_t outW)
    {
        const uint32_t planeRows = validC1 * outH;
        const uint32_t rowElements = outW * block;
        const uint32_t outputCount = planeRows * rowElements;
        const uint32_t outputBlock = outH * rowElements;
        if (groupPlanes <= 1U || planeRows == 0U || outputCount == 0U || outputBlock == 0U ||
            outputBlock > OUTPUT_TILE_NUM) {
            return false;
        }
        bool allFast = true;
        for (uint32_t c1 = 0; c1 < validC1; ++c1) {
            if (!ProcessNdc1hwc0NdhwcK1FullC1PackDGroupRows(outputOffset + static_cast<uint64_t>(c1) * outputBlock,
                                                            nIdx, od, groupPlanes, c1, block, outH, outW, validC1)) {
                allFast = false;
                break;
            }
        }
        if (allFast) {
            return true;
        }
        for (uint32_t gd = 0; gd < groupPlanes; ++gd) {
            for (uint32_t c1 = 0; c1 < validC1; ++c1) {
                const uint64_t c1OutputOffset = outputOffset + static_cast<uint64_t>(gd) * outputCount +
                                                static_cast<uint64_t>(c1) * outputBlock;
                ProcessNdc1hwc0NdhwcK1FullC1PackRows(c1OutputOffset, nIdx, od + static_cast<int64_t>(gd), 0U, outH, c1,
                                                     block, outW);
            }
        }
        return true;
    }

    __aicore__ inline bool ProcessNdc1hwc0NdhwcK1FullC1PackDGroupRows(uint64_t outputOffset, int64_t nIdx, int64_t od,
                                                                      uint32_t groupPlanes, uint32_t c1, uint32_t block,
                                                                      uint32_t outH, uint32_t outW, uint32_t validC1)
    {
        const uint32_t cBase = c1 * block;
        if (cBase >= static_cast<uint32_t>(tiling_->c) || groupPlanes == 0U || outH == 0U || outW == 0U ||
            block == 0U || validC1 <= 1U) {
            return false;
        }
        uint32_t cCount = static_cast<uint32_t>(tiling_->c) - cBase;
        if (cCount > block) {
            cCount = block;
        }
        const uint32_t alignedC0 = AlignToVector(cCount);
        const uint32_t rowElements = outW * block;
        const uint32_t outputBlock = outH * rowElements;
        const uint32_t blockCount = groupPlanes * outH * outW;
        const uint32_t inputCount = blockCount * alignedC0;
        if (cCount == 0U || alignedC0 != block || inputCount == 0U || inputCount > INPUT_TILE_NUM ||
            outputBlock == 0U || outputBlock > OUTPUT_TILE_NUM) {
            return false;
        }
        CopyInVectorWBlocksPadStrideValue(InputOffset(nIdx, od, 0, 0, cBase), blockCount, cCount, alignedC0,
                                          static_cast<uint32_t>(tiling_->c - static_cast<int64_t>(cCount)),
                                          ZeroValue());
        LocalTensor<T> xLocal = xInQue_.DeQue<T>();
        CopyOutNdc1hwc0K1C1DGroupFromMte2(outputOffset, xLocal, outputBlock, groupPlanes, validC1);
        xInQue_.FreeTensor(xLocal);
        return true;
    }

    __aicore__ inline void CopyOutNdc1hwc0K1C1DGroupFromMte2(uint64_t outputOffset, LocalTensor<T> srcLocal,
                                                             uint32_t outputBlock, uint32_t groupPlanes,
                                                             uint32_t validC1)
    {
        SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        DataCopyExtParams copyParams{static_cast<uint16_t>(groupPlanes), static_cast<uint32_t>(outputBlock * sizeof(T)),
                                     0, static_cast<uint32_t>((validC1 - 1U) * outputBlock * sizeof(T)), 0};
        DataCopyPad(yGm_[outputOffset], srcLocal, copyParams);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }

    __aicore__ inline void ProcessNdc1hwc0NdhwcK1FullC1PlaneTile(uint64_t outputOffset, int64_t nIdx, int64_t od,
                                                                 uint32_t validC1, uint32_t block, uint32_t outH,
                                                                 uint32_t outW)
    {
        const uint32_t rowElements = outW * block;
        const uint32_t outputBlock = outH * rowElements;
        for (uint32_t c1 = 0; c1 < validC1; ++c1) {
            ProcessNdc1hwc0NdhwcK1FullC1PackRows(outputOffset + static_cast<uint64_t>(c1) * outputBlock, nIdx, od, 0U,
                                                 outH, c1, block, outW);
        }
    }

    __aicore__ inline void ProcessNdc1hwc0NdhwcK1FullC1PackRows(uint64_t outputOffset, int64_t nIdx, int64_t od,
                                                                uint32_t ohStart, uint32_t rows, uint32_t c1,
                                                                uint32_t block, uint32_t outW)
    {
        const uint32_t cBase = c1 * block;
        uint32_t cCount = 0U;
        if (cBase < static_cast<uint32_t>(tiling_->c)) {
            cCount = static_cast<uint32_t>(tiling_->c) - cBase;
            if (cCount > block) {
                cCount = block;
            }
        }
        const uint32_t alignedC0 = AlignToVector(cCount == 0U ? 1U : cCount);
        const uint32_t rowElements = outW * block;
        LocalTensor<T> outLocal = maskBuf_.Get<T>();
        if (cCount == 0U) {
            Duplicate(outLocal, ZeroValue(), rows * rowElements);
            PipeBarrier<PIPE_V>();
            CopyOutVector(outputOffset, outLocal, rows * rowElements);
            return;
        }
        CopyInVectorWBlocksPadStrideValue(InputOffset(nIdx, od, ohStart, 0, cBase), rows * outW, cCount, alignedC0,
                                          static_cast<uint32_t>(tiling_->c - static_cast<int64_t>(cCount)),
                                          ZeroValue());
        LocalTensor<T> xLocal = xInQue_.DeQue<T>();
        if (alignedC0 == block) {
            CopyOutVectorFromMte2(outputOffset, xLocal, rows * rowElements);
            xInQue_.FreeTensor(xLocal);
            return;
        } else {
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            ScatterNdhwcK1PackedRows(outLocal, xLocal, rows, outW, block, cCount, alignedC0);
        }
        xInQue_.FreeTensor(xLocal);
        CopyOutVector(outputOffset, outLocal, rows * rowElements);
    }

    __aicore__ inline void CopyOutVectorFromMte2(uint64_t outputOffset, LocalTensor<T> srcLocal, uint32_t count)
    {
        SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        if (count % VectorAlignNum() == 0U && outputOffset % VectorAlignNum() == 0U) {
            DataCopy(yGm_[outputOffset], srcLocal, count);
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
            DataCopyPad(yGm_[outputOffset], srcLocal, copyParams);
        }
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }

    __aicore__ inline void CopyOutVectorFromMte2Fast(uint64_t outputOffset, LocalTensor<T> srcLocal, uint32_t count)
    {
        SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        if (count % VectorAlignNum() == 0U && outputOffset % VectorAlignNum() == 0U) {
            DataCopy(yGm_[outputOffset], srcLocal, count);
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
            DataCopyPad(yGm_[outputOffset], srcLocal, copyParams);
        }
    }

    __aicore__ inline void ScatterNdhwcK1PackedRows(LocalTensor<T> dstLocal, LocalTensor<T> srcLocal, uint32_t rows,
                                                    uint32_t outW, uint32_t block, uint32_t cCount, uint32_t alignedC0)
    {
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        for (uint32_t row = 0; row < rows; ++row) {
            const uint64_t rowOutBase = static_cast<uint64_t>(row) * outW * block;
            const uint64_t rowSrcBase = static_cast<uint64_t>(row) * outW * alignedC0;
            for (uint32_t ow = 0; ow < outW; ++ow) {
                const uint64_t outBase = rowOutBase + static_cast<uint64_t>(ow) * block;
                const uint64_t srcBase = rowSrcBase + static_cast<uint64_t>(ow) * alignedC0;
                for (uint32_t c0 = 0; c0 < block; ++c0) {
                    const T value = c0 < cCount ? srcLocal.GetValue(srcBase + c0) : ZeroValue();
                    dstLocal.SetValue(outBase + c0, value);
                }
            }
        }
    }

    __aicore__ inline bool HasNdc1hwc0NdhwcStride2Shape() const
    {
        return tiling_->dataFormat == FORMAT_NDHWC_VALUE && tiling_->outW > 0 && tiling_->outH > 0 &&
               tiling_->inW > 0 && tiling_->c > 0;
    }

    __aicore__ inline bool HasNdc1hwc0Stride2Pool2Spec() const
    {
        const bool kernelAndStride = tiling_->kD == 2 && tiling_->kH == 2 && tiling_->kW == 2 && tiling_->sD == 2 &&
                                     tiling_->sH == 2 && tiling_->sW == 2;
        const bool dilationAndPad = tiling_->dilationD == 1 && tiling_->dilationH == 1 && tiling_->dilationW == 1 &&
                                    tiling_->padFront == 0 && tiling_->padTop == 0 && tiling_->padLeft == 0;
        return kernelAndStride && dilationAndPad;
    }

    __aicore__ inline bool HasNdc1hwc0NdhwcStride2DualC1Layout(uint64_t block) const
    {
        const uint64_t validC1 = Ndc1hwc0ValidC1(block);
        return block > 0U && validC1 == 2U && Ndc1hwc0ValidOut(block, validC1) == tiling_->totalOut &&
               static_cast<uint64_t>(tiling_->c) > block && static_cast<uint64_t>(tiling_->c) <= block * 2U;
    }

    __aicore__ inline uint32_t Ndc1hwc0NdhwcStride2AlignedInputCount(uint32_t validInputCount) const
    {
        const uint32_t blockElements = UB_BLOCK_BYTES / sizeof(T);
        if constexpr (AscendC::Std::is_same<T, half>::value) {
            return (validInputCount + 1U + blockElements - 1U) / blockElements * blockElements;
        }
        return AlignToVector(validInputCount + 1U);
    }

    __aicore__ inline bool HasNdc1hwc0NdhwcStride2DualC1Capacity(uint64_t block) const
    {
        const uint32_t outW = static_cast<uint32_t>(tiling_->outW);
        const uint32_t rowElements = static_cast<uint32_t>(static_cast<uint64_t>(outW) * block);
        const uint32_t validInputCount = static_cast<uint32_t>(tiling_->inW) * static_cast<uint32_t>(tiling_->c);
        const uint32_t alignedInputCount = Ndc1hwc0NdhwcStride2AlignedInputCount(validInputCount);
        const uint32_t offsetElements = Ndc1hwc0GatherTempOffset(rowElements);
        const uint32_t c0Count = static_cast<uint32_t>(block);
        const uint32_t c1Count = static_cast<uint32_t>(static_cast<uint64_t>(tiling_->c) - block);
        const uint32_t scratchNeed = offsetElements * 4U + rowElements * 3U;
        const uint64_t slabInputNeed = static_cast<uint64_t>(alignedInputCount) * static_cast<uint64_t>(tiling_->outH) *
                                       2U;
        return c1Count > 0U && rowElements > 0U && rowElements * 2U <= OUTPUT_TILE_NUM &&
               rowElements * static_cast<uint32_t>(tiling_->outH) * 2U <= OUTPUT_TILE_NUM &&
               slabInputNeed <= INPUT_TILE_NUM && scratchNeed <= OUTPUT_TILE_NUM &&
               CanUseNdc1hwc0NdhwcStride2Pool2Row(0, c0Count, static_cast<uint32_t>(block)) &&
               CanUseNdc1hwc0NdhwcStride2Pool2Row(static_cast<int64_t>(block), c1Count, static_cast<uint32_t>(block));
    }

    __aicore__ inline bool CanUseNdc1hwc0NdhwcStride2DualC1GroupPath() const
    {
        const uint64_t block = Ndc1hwc0Block();
        return HasNdc1hwc0NdhwcStride2Shape() && HasNdc1hwc0NdhwcStride2DualC1Layout(block) &&
               HasNdc1hwc0Stride2Pool2Spec() && HasNdc1hwc0NdhwcStride2DualC1Capacity(block);
    }

    __aicore__ inline void ProcessNdc1hwc0NdhwcStride2DualC1Group()
    {
        Ndc1hwc0LinearRangeContext range{};
        if (!InitNdc1hwc0ActiveLinearRange(range)) {
            return;
        }
        uint64_t cur = range.outOffset;
        while (cur < range.outEnd) {
            const uint64_t row = cur / range.rowElements;
            const uint64_t rowOffset = cur - row * range.rowElements;
            if (rowOffset != 0U || range.outEnd - cur < range.rowElements * 2U) {
                ProcessNdc1hwc0RowVectorByRow(row, cur, range.block, range.validC1);
                cur += range.rowElements - rowOffset;
                continue;
            }
            int64_t nIdx = 0;
            int64_t od = 0;
            int64_t c1Idx = 0;
            int64_t oh = 0;
            DecodeNdc1hwc0Row(row, range.validC1, nIdx, od, c1Idx, oh);
            if (c1Idx != 0 || oh != 0) {
                ProcessNdc1hwc0RowVectorByRow(row, cur, range.block, range.validC1);
                cur += range.rowElements;
                continue;
            }
            const uint64_t groupElements = range.rowElements * static_cast<uint64_t>(tiling_->outH) * 2U;
            if (range.outEnd - cur < groupElements) {
                ProcessNdc1hwc0RowVectorByRow(row, cur, range.block, range.validC1);
                cur += range.rowElements;
                continue;
            }
            ProcessNdc1hwc0NdhwcStride2DualC1HGroup(cur, nIdx, od, static_cast<uint32_t>(range.block));
            cur += groupElements;
        }
        if (range.validOut < tiling_->totalOut) {
            CopyOutZeroRangeByCore(range.validOut, tiling_->totalOut - range.validOut);
        }
    }

    __aicore__ inline void ReduceNdc1hwc0NdhwcStride2DualC1Slab(
        LocalTensor<T> xLocal, LocalTensor<T> outLocal, LocalTensor<T> even0Local, LocalTensor<T> odd0Local,
        LocalTensor<T> even1Local, LocalTensor<T> odd1Local, LocalTensor<uint32_t> even0Offset,
        LocalTensor<uint32_t> odd0Offset, LocalTensor<uint32_t> even1Offset, LocalTensor<uint32_t> odd1Offset,
        uint32_t slabRows, uint32_t alignedInputCount, uint32_t rowElements, uint32_t outH)
    {
        for (uint32_t oh = 0; oh < outH; ++oh) {
            LocalTensor<T> row0Local = outLocal[static_cast<uint64_t>(oh) * rowElements];
            LocalTensor<T> row1Local = outLocal[static_cast<uint64_t>(outH + oh) * rowElements];
            for (int64_t kh = 0; kh < 2; ++kh) {
                const uint32_t ihLocal = oh * 2U + static_cast<uint32_t>(kh);
                if (ihLocal >= slabRows) {
                    continue;
                }
                LocalTensor<T> hLocal = xLocal[static_cast<uint64_t>(ihLocal) * alignedInputCount];
                Gather(even0Local, hLocal, even0Offset, static_cast<uint32_t>(0), rowElements);
                Gather(odd0Local, hLocal, odd0Offset, static_cast<uint32_t>(0), rowElements);
                Gather(even1Local, hLocal, even1Offset, static_cast<uint32_t>(0), rowElements);
                Gather(odd1Local, hLocal, odd1Offset, static_cast<uint32_t>(0), rowElements);
                PipeBarrier<PIPE_V>();
                Max(even0Local, even0Local, odd0Local, rowElements);
                Max(even1Local, even1Local, odd1Local, rowElements);
                PipeBarrier<PIPE_V>();
                Max(row0Local, row0Local, even0Local, rowElements);
                Max(row1Local, row1Local, even1Local, rowElements);
                PipeBarrier<PIPE_V>();
            }
        }
    }

    __aicore__ inline void ProcessNdc1hwc0NdhwcStride2DualC1HGroup(uint64_t outputOffset, int64_t nIdx, int64_t od,
                                                                   uint32_t block)
    {
        const uint32_t outW = static_cast<uint32_t>(tiling_->outW);
        const uint32_t outH = static_cast<uint32_t>(tiling_->outH);
        const uint32_t rowElements = outW * block;
        const uint32_t validInputCount = static_cast<uint32_t>(tiling_->inW) * static_cast<uint32_t>(tiling_->c);
        const uint32_t alignedInputCount = Ndc1hwc0NdhwcStride2AlignedInputCount(validInputCount);
        const uint32_t offsetElements = Ndc1hwc0GatherTempOffset(rowElements);
        const uint32_t c1Count = static_cast<uint32_t>(static_cast<uint64_t>(tiling_->c) - block);
        LocalTensor<T> outLocal = maskBuf_.Get<T>();
        LocalTensor<T> scratchLocal = tmpBuf_.Get<T>();
        LocalTensor<T> even0Local = calcBuf_.Get<T>();
        LocalTensor<uint32_t> even0Offset = scratchLocal.template ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> odd0Offset = scratchLocal[offsetElements].template ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> even1Offset = scratchLocal[offsetElements * 2U].template ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> odd1Offset = scratchLocal[offsetElements * 3U].template ReinterpretCast<uint32_t>();
        LocalTensor<T> odd0Local = scratchLocal[offsetElements * 4U];
        LocalTensor<T> even1Local = odd0Local[rowElements];
        LocalTensor<T> odd1Local = even1Local[rowElements];
        InitNdc1hwc0NdhwcStride2GatherOffsets(even0Offset, odd0Offset, 0, block, block, outW);
        InitNdc1hwc0NdhwcStride2GatherOffsets(even1Offset, odd1Offset, static_cast<int64_t>(block), c1Count, block,
                                              outW);
        Duplicate(outLocal, NegInfValue(), rowElements * outH * 2U);
        PipeBarrier<PIPE_V>();
        for (int64_t kd = 0; kd < 2; ++kd) {
            const int64_t id = Pool2InputD(od, kd);
            if (IsOutOfRange(id, tiling_->inD)) {
                continue;
            }
            uint32_t slabRows = outH * 2U;
            if (slabRows > static_cast<uint32_t>(tiling_->inH)) {
                slabRows = static_cast<uint32_t>(tiling_->inH);
            }
            CopyInVectorWBlocksPadStrideValue(InputOffset(nIdx, id, 0, 0, 0), slabRows, validInputCount,
                                              alignedInputCount, 0U, NegInfValue());
            LocalTensor<T> xLocal = xInQue_.DeQue<T>();
            ReduceNdc1hwc0NdhwcStride2DualC1Slab(xLocal, outLocal, even0Local, odd0Local, even1Local, odd1Local,
                                                 even0Offset, odd0Offset, even1Offset, odd1Offset, slabRows,
                                                 alignedInputCount, rowElements, outH);
            xInQue_.FreeTensor(xLocal);
        }
        for (uint32_t oh = 0; oh < outH; ++oh) {
            ZeroNdc1hwc0RowTail(outLocal[static_cast<uint64_t>(outH + oh) * rowElements], c1Count, block, outW);
        }
        CopyOutVector(outputOffset, outLocal, rowElements * outH * 2U);
    }

    __aicore__ inline bool CanUseNdc1hwc0NdhwcStride2GroupPath() const
    {
        if (!HasNdc1hwc0NdhwcStride2Shape() || !HasNdc1hwc0Stride2Pool2Spec()) {
            return false;
        }
        const uint64_t block = Ndc1hwc0Block();
        const uint32_t outW = static_cast<uint32_t>(tiling_->outW);
        const uint32_t rowElements = static_cast<uint32_t>(static_cast<uint64_t>(outW) * block);
        const uint32_t cCount = static_cast<uint32_t>(
            block < static_cast<uint64_t>(tiling_->c) ? block : static_cast<uint64_t>(tiling_->c));
        return block > 0U && rowElements > 0U && rowElements <= OUTPUT_TILE_NUM &&
               CanUseNdc1hwc0NdhwcStride2Pool2Row(0, cCount, static_cast<uint32_t>(block));
    }

    __aicore__ inline uint32_t Ndc1hwc0NdhwcStride2TileRows(int64_t oh, uint64_t remainElements,
                                                            uint64_t rowElements) const
    {
        if (rowElements == 0U) {
            return 0U;
        }
        uint32_t rows = static_cast<uint32_t>(tiling_->outH - oh);
        const uint64_t remainRows = remainElements / rowElements;
        if (static_cast<uint64_t>(rows) > remainRows) {
            rows = static_cast<uint32_t>(remainRows);
        }
        const uint32_t maxRowsByOutput = OUTPUT_TILE_NUM / static_cast<uint32_t>(rowElements);
        if (rows > maxRowsByOutput) {
            rows = maxRowsByOutput;
        }
        const uint32_t validInputCount = static_cast<uint32_t>(tiling_->inW) * static_cast<uint32_t>(tiling_->c);
        const uint32_t alignedInputCount = AlignToVector(validInputCount + 1U);
        const uint32_t maxRowsByInput = alignedInputCount == 0U ? 1U : INPUT_TILE_NUM / (alignedInputCount * 2U);
        if (maxRowsByInput > 0U && rows > maxRowsByInput) {
            rows = maxRowsByInput;
        }
        return rows;
    }

    __aicore__ inline void ProcessNdc1hwc0NdhwcStride2GroupStep(uint64_t& cur, uint64_t outEnd, uint64_t block,
                                                                uint64_t validC1, uint64_t rowElements)
    {
        Ndc1hwc0DecodedRow context{};
        if (!PrepareNdc1hwc0DecodedFullRow(cur, outEnd, block, rowElements, validC1, context)) {
            return;
        }
        const int64_t cBase = context.c1Idx * static_cast<int64_t>(block);
        const int64_t activeChannels = Ndc1hwc0ActiveChannels(cBase, block);
        const uint32_t rows = Ndc1hwc0NdhwcStride2TileRows(context.oh, outEnd - cur, rowElements);
        if (rows == 0U) {
            ProcessNdc1hwc0RowVectorByRow(context.row, cur, block, validC1);
            cur += rowElements;
            return;
        }
        if (activeChannels > 0 && CanUseNdc1hwc0NdhwcStride2Pool2Row(cBase, static_cast<uint32_t>(activeChannels),
                                                                     static_cast<uint32_t>(block))) {
            ProcessNdc1hwc0NdhwcStride2GroupTile(cur, context.nIdx, context.od, cBase,
                                                 static_cast<uint32_t>(context.oh), rows,
                                                 static_cast<uint32_t>(activeChannels), static_cast<uint32_t>(block));
        } else {
            ProcessNdc1hwc0RowVectorTile(context.row, rows, cur, block, validC1);
        }
        cur += static_cast<uint64_t>(rows) * rowElements;
    }

    __aicore__ inline void ProcessNdc1hwc0NdhwcStride2Group()
    {
        Ndc1hwc0LinearRangeContext range{};
        if (!InitNdc1hwc0ActiveLinearRange(range)) {
            return;
        }

        uint64_t cur = range.outOffset;
        while (cur < range.outEnd) {
            ProcessNdc1hwc0NdhwcStride2GroupStep(cur, range.outEnd, range.block, range.validC1, range.rowElements);
        }
        if (range.validOut < tiling_->totalOut) {
            CopyOutZeroRangeByCore(range.validOut, tiling_->totalOut - range.validOut);
        }
    }

    __aicore__ inline void ReduceNdc1hwc0NdhwcStride2Slab(LocalTensor<T> xLocal, LocalTensor<T> outLocal,
                                                          LocalTensor<T> evenLocal, LocalTensor<T> oddLocal,
                                                          LocalTensor<uint32_t> evenOffset,
                                                          LocalTensor<uint32_t> oddOffset, uint32_t rows,
                                                          uint32_t slabRows, uint32_t alignedInputCount,
                                                          uint32_t rowElements)
    {
        for (uint32_t rowIdx = 0; rowIdx < rows; ++rowIdx) {
            LocalTensor<T> rowLocal = outLocal[static_cast<uint64_t>(rowIdx) * rowElements];
            for (int64_t kh = 0; kh < 2; ++kh) {
                const uint32_t ihLocal = rowIdx * 2U + static_cast<uint32_t>(kh);
                if (ihLocal >= slabRows) {
                    continue;
                }
                LocalTensor<T> hLocal = xLocal[static_cast<uint64_t>(ihLocal) * alignedInputCount];
                Gather(evenLocal, hLocal, evenOffset, static_cast<uint32_t>(0), rowElements);
                Gather(oddLocal, hLocal, oddOffset, static_cast<uint32_t>(0), rowElements);
                PipeBarrier<PIPE_V>();
                Max(evenLocal, evenLocal, oddLocal, rowElements);
                PipeBarrier<PIPE_V>();
                Max(rowLocal, rowLocal, evenLocal, rowElements);
                PipeBarrier<PIPE_V>();
            }
        }
    }

    __aicore__ inline void ProcessNdc1hwc0NdhwcStride2GroupTile(uint64_t outputOffset, int64_t nIdx, int64_t od,
                                                                int64_t cBase, uint32_t ohStart, uint32_t rows,
                                                                uint32_t cCount, uint32_t block)
    {
        const uint32_t outW = static_cast<uint32_t>(tiling_->outW);
        const uint32_t rowElements = outW * block;
        const uint32_t validInputCount = static_cast<uint32_t>(tiling_->inW) * static_cast<uint32_t>(tiling_->c);
        const uint32_t alignedInputCount = AlignToVector(validInputCount + 1U);
        const uint32_t offsetElements = Ndc1hwc0GatherTempOffset(rowElements);
        LocalTensor<T> evenLocal = calcBuf_.Get<T>();
        LocalTensor<T> scratchLocal = tmpBuf_.Get<T>();
        LocalTensor<T> outLocal = maskBuf_.Get<T>();
        LocalTensor<uint32_t> evenOffset = scratchLocal.template ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> oddOffset = scratchLocal[offsetElements].template ReinterpretCast<uint32_t>();
        LocalTensor<T> oddLocal = scratchLocal[offsetElements * 2U];
        InitNdc1hwc0NdhwcStride2GatherOffsets(evenOffset, oddOffset, cBase, cCount, block, outW);
        Duplicate(outLocal, NegInfValue(), rows * rowElements);
        PipeBarrier<PIPE_V>();
        for (int64_t kd = 0; kd < 2; ++kd) {
            const int64_t id = Pool2InputD(od, kd);
            if (IsOutOfRange(id, tiling_->inD)) {
                continue;
            }
            const int64_t ihStart = static_cast<int64_t>(ohStart) * 2;
            uint32_t slabRows = rows * 2U;
            if (ihStart + static_cast<int64_t>(slabRows) > tiling_->inH) {
                slabRows = static_cast<uint32_t>(tiling_->inH - ihStart);
            }
            CopyInVectorWBlocksPadStrideValue(InputOffset(nIdx, id, ihStart, 0, 0), slabRows, validInputCount,
                                              alignedInputCount, 0U, NegInfValue());
            LocalTensor<T> xLocal = xInQue_.DeQue<T>();
            ReduceNdc1hwc0NdhwcStride2Slab(xLocal, outLocal, evenLocal, oddLocal, evenOffset, oddOffset, rows, slabRows,
                                           alignedInputCount, rowElements);
            xInQue_.FreeTensor(xLocal);
        }
        for (uint32_t rowIdx = 0; rowIdx < rows; ++rowIdx) {
            LocalTensor<T> rowLocal = outLocal[static_cast<uint64_t>(rowIdx) * rowElements];
            ZeroNdc1hwc0RowTail(rowLocal, cCount, block, outW);
        }
        CopyOutVector(outputOffset, outLocal, rows * rowElements);
    }

    struct Ndc1hwc0GroupStepContext {
        int64_t nIdx;
        int64_t od;
        int64_t oh;
        uint32_t rows;
    };

    __aicore__ inline bool PrepareNdc1hwc0GroupStep(uint64_t& cur, uint64_t outEnd, uint64_t block, uint64_t validC1,
                                                    uint32_t rowElements, bool requireFullRow,
                                                    Ndc1hwc0GroupStepContext& context)
    {
        if (rowElements == 0U) {
            cur = outEnd;
            return false;
        }
        const uint64_t row = cur / rowElements;
        const uint64_t rowOffset = cur - row * static_cast<uint64_t>(rowElements);
        if (rowOffset != 0U || (requireFullRow && outEnd - cur < rowElements)) {
            if (requireFullRow) {
                cur += ProcessNdc1hwc0PartialValidRange(cur, outEnd, block, rowElements, validC1);
            } else {
                ProcessNdc1hwc0RowVectorByRow(row, cur, block, validC1);
                cur += rowElements - rowOffset;
            }
            return false;
        }
        int64_t c1Idx = 0;
        DecodeNdc1hwc0Row(row, validC1, context.nIdx, context.od, c1Idx, context.oh);
        if (c1Idx != 0) {
            ProcessNdc1hwc0RowVectorByRow(row, cur, block, validC1);
            cur += rowElements;
            return false;
        }
        context.rows = static_cast<uint32_t>(tiling_->outH - context.oh);
        const uint64_t remainRows = (outEnd - cur) / rowElements;
        if (static_cast<uint64_t>(context.rows) > remainRows) {
            context.rows = static_cast<uint32_t>(remainRows);
        }
        if (context.rows == 0U) {
            cur = outEnd;
            return false;
        }
        return true;
    }

    __aicore__ inline void ProcessNdc1hwc0NdhwcD3H3Dil2GroupStep(uint64_t& cur, uint64_t outEnd, uint64_t block,
                                                                 uint64_t validC1, uint32_t cCount, uint32_t outW,
                                                                 uint32_t rowElements)
    {
        Ndc1hwc0GroupStepContext context{};
        if (!PrepareNdc1hwc0GroupStep(cur, outEnd, block, validC1, rowElements, true, context)) {
            return;
        }
        ProcessNdc1hwc0NdhwcD3H3Dil2GroupTile(cur, context.nIdx, context.od, static_cast<uint32_t>(context.oh),
                                              context.rows, cCount, static_cast<uint32_t>(block), outW);
        cur += static_cast<uint64_t>(context.rows) * rowElements;
    }

    __aicore__ inline void ProcessNdc1hwc0NdhwcD3H3Dil2Group()
    {
        uint64_t block = 0U, validC1 = 0U, validOut = 0U;
        uint64_t outOffset = 0U, outCount = 0U, outEnd = 0U;
        uint32_t outW = 0U, rowElements = 0U;
        if (!InitNdc1hwc0ActiveGroupRange(block, validC1, validOut, outW, rowElements, outOffset, outCount, outEnd)) {
            return;
        }

        const uint32_t cCount = static_cast<uint32_t>(tiling_->c);
        uint64_t cur = outOffset;
        while (cur < outEnd) {
            ProcessNdc1hwc0NdhwcD3H3Dil2GroupStep(cur, outEnd, block, validC1, cCount, outW, rowElements);
        }
        if (validOut < tiling_->totalOut) {
            CopyOutZeroRangeByCore(validOut, tiling_->totalOut - validOut);
        }
    }

    __aicore__ inline void ReduceNdc1hwc0NdhwcD3H3Depth(LocalTensor<T> dmaxLocal, int64_t nIdx, int64_t od,
                                                        uint32_t rowCount, uint32_t zeroIndex, uint32_t compactStride)
    {
        for (int64_t kd = 0; kd < 3; ++kd) {
            const int64_t id = od * 3 + kd;
            if (IsOutOfRange(id, tiling_->inD)) {
                continue;
            }
            for (uint32_t ih = 0U; ih < static_cast<uint32_t>(tiling_->inH); ++ih) {
                CopyInVectorPadValue(InputOffset(nIdx, id, static_cast<int64_t>(ih), 0, 0), rowCount, zeroIndex,
                                     NegInfValue());
                LocalTensor<T> xLocal = xInQue_.DeQue<T>();
                Max(dmaxLocal[static_cast<uint64_t>(ih) * compactStride],
                    dmaxLocal[static_cast<uint64_t>(ih) * compactStride], xLocal, zeroIndex);
                PipeBarrier<PIPE_V>();
                xInQue_.FreeTensor(xLocal);
            }
        }
    }

    __aicore__ inline void StoreNdc1hwc0NdhwcD3H3Rows(LocalTensor<T> compactLocal, LocalTensor<T> dmaxLocal,
                                                      LocalTensor<T> outLocal, uint64_t outputOffset, uint32_t ohStart,
                                                      uint32_t rows, uint32_t maxTileRows, uint32_t compactStride,
                                                      uint32_t cCount, uint32_t block, uint32_t outW,
                                                      uint32_t rowElements, uint32_t zeroIndex)
    {
        uint32_t doneRows = 0U;
        while (doneRows < rows) {
            uint32_t tileRows = rows - doneRows;
            if (tileRows > maxTileRows) {
                tileRows = maxTileRows;
            }
            MaxNdc1hwc0NdhwcD3H3CompactRows(compactLocal, dmaxLocal, ohStart + doneRows, tileRows, compactStride);
            const uint32_t offsetBase = AlignToVector(tileRows * compactStride);
            LocalTensor<uint32_t> offsetLocal = compactLocal[offsetBase].template ReinterpretCast<uint32_t>();
            if (!ScatterNdc1hwc0CompactRowsReuseRowOffset(outLocal, compactLocal, offsetLocal, tileRows, compactStride,
                                                          cCount, block, outW, cCount, 1U, zeroIndex)) {
                ScatterNdc1hwc0CompactRowsChecked(outLocal, compactLocal, offsetLocal, tileRows, compactStride, cCount,
                                                  block, outW, cCount, 1U, zeroIndex);
            }
            CopyOutVector(outputOffset + static_cast<uint64_t>(doneRows) * rowElements, outLocal,
                          tileRows * rowElements);
            doneRows += tileRows;
        }
    }

    __aicore__ inline void ProcessNdc1hwc0NdhwcD3H3Dil2GroupTile(uint64_t outputOffset, int64_t nIdx, int64_t od,
                                                                 uint32_t ohStart, uint32_t rows, uint32_t cCount,
                                                                 uint32_t block, uint32_t outW)
    {
        if (ProcessNdc1hwc0NdhwcD3H3Dil2PlaneCopyTile(outputOffset, nIdx, od, ohStart, rows, cCount, block, outW)) {
            return;
        }
        const uint32_t rowCount = outW * cCount;
        const uint32_t zeroIndex = AlignToVector(rowCount);
        const uint32_t compactStride = AlignToVector(zeroIndex + 1U);
        const uint32_t dmaxCount = static_cast<uint32_t>(tiling_->inH) * compactStride;
        const uint32_t rowElements = outW * block;
        LocalTensor<T> dmaxLocal = calcBuf_.Get<T>();
        LocalTensor<T> compactLocal = tmpBuf_.Get<T>();
        LocalTensor<T> outLocal = maskBuf_.Get<T>();
        Duplicate(dmaxLocal, NegInfValue(), dmaxCount);
        PipeBarrier<PIPE_V>();
        ReduceNdc1hwc0NdhwcD3H3Depth(dmaxLocal, nIdx, od, rowCount, zeroIndex, compactStride);

        uint32_t maxTileRows = rows;
        const uint32_t maxRowsByOutput = rowElements == 0U ? 1U : NDC1HWC0_D3H3_OUTPUT_TILE_NUM / rowElements;
        if (maxRowsByOutput > 0U && maxTileRows > maxRowsByOutput) {
            maxTileRows = maxRowsByOutput;
        }
        while (maxTileRows > 1U && static_cast<uint64_t>(maxTileRows) * static_cast<uint64_t>(compactStride) >
                                       NDC1HWC0_D3H3_OUTPUT_TILE_NUM) {
            --maxTileRows;
        }
        if (maxTileRows == 0U) {
            maxTileRows = 1U;
        }

        StoreNdc1hwc0NdhwcD3H3Rows(compactLocal, dmaxLocal, outLocal, outputOffset, ohStart, rows, maxTileRows,
                                   compactStride, cCount, block, outW, rowElements, zeroIndex);
    }

    struct NdhwcD3H3PlaneCopyContext {
        uint32_t inH;
        uint32_t rowCount;
        uint32_t zeroIndex;
        uint32_t compactStride;
        uint32_t rowElements;
        uint32_t rowOffsetNeed;
        uint32_t maxTileRows;
    };

    __aicore__ inline bool IsValidNdhwcD3H3PlaneCopyCapacity(uint32_t rowCount, uint32_t dmaxCount,
                                                             uint32_t copyPlaneCount, uint32_t rowElements,
                                                             uint32_t zeroIndex, uint32_t compactStride,
                                                             uint32_t rowOffsetNeed) const
    {
        return rowCount > 0U && dmaxCount > 0U && copyPlaneCount > 0U && copyPlaneCount <= INPUT_TILE_NUM &&
               dmaxCount <= NDC1HWC0_D3H3_OUTPUT_TILE_NUM && rowElements > 0U && rowElements <= OUTPUT_TILE_NUM &&
               zeroIndex < compactStride && compactStride >= rowCount && compactStride - rowCount <= 255U &&
               rowOffsetNeed <= NDC1HWC0_D3H3_OUTPUT_TILE_NUM;
    }

    __aicore__ inline bool InitNdhwcD3H3PlaneCopyContext(uint32_t rows, uint32_t cCount, uint32_t block, uint32_t outW,
                                                         NdhwcD3H3PlaneCopyContext& context) const
    {
        if (rows == 0U || cCount == 0U || cCount > block || outW == 0U || tiling_->inH <= 0) {
            return false;
        }
        const uint32_t inH = static_cast<uint32_t>(tiling_->inH);
        const uint32_t rowCount = outW * cCount;
        const uint32_t zeroIndex = AlignToVector(rowCount);
        const uint32_t compactStride = AlignToVector(zeroIndex + 1U);
        const uint32_t dmaxCount = inH * compactStride;
        const uint32_t copyPlaneCount = inH * zeroIndex;
        const uint32_t rowElements = outW * block;
        const uint32_t rowOffsetNeed = Ndc1hwc0GatherTempOffset(rowElements);
        if (!IsValidNdhwcD3H3PlaneCopyCapacity(rowCount, dmaxCount, copyPlaneCount, rowElements, zeroIndex,
                                               compactStride, rowOffsetNeed)) {
            return false;
        }
        uint32_t maxTileRows = Ndc1hwc0MaxCompactTileRowsLimit(rows, compactStride, rowElements,
                                                               NDC1HWC0_D3H3_OUTPUT_TILE_NUM);
        if (maxTileRows == 0U) {
            maxTileRows = 1U;
        }
        while (maxTileRows > 1U &&
               (static_cast<uint64_t>(maxTileRows) * rowElements > NDC1HWC0_D3H3_OUTPUT_TILE_NUM ||
                AlignToVector(maxTileRows * compactStride) + rowOffsetNeed > NDC1HWC0_D3H3_OUTPUT_TILE_NUM)) {
            --maxTileRows;
        }
        context = {inH, rowCount, zeroIndex, compactStride, rowElements, rowOffsetNeed, maxTileRows};
        return true;
    }

    __aicore__ inline bool LoadNdhwcD3H3PlaneCopyDmax(LocalTensor<T> dmaxLocal, int64_t nIdx, int64_t od,
                                                      const NdhwcD3H3PlaneCopyContext& context)
    {
        bool dmaxInitialized = false;
        for (int64_t kd = 0; kd < 3; ++kd) {
            const int64_t id = od * 3 + kd;
            if (IsOutOfRange(id, tiling_->inD)) {
                continue;
            }
            CopyInVectorWBlocksPadStrideValue(InputOffset(nIdx, id, 0, 0, 0), context.inH, context.rowCount,
                                              context.zeroIndex, 0U, NegInfValue());
            LocalTensor<T> xLocal = xInQue_.DeQue<T>();
            for (uint32_t ih = 0U; ih < context.inH; ++ih) {
                const uint64_t dstBase = static_cast<uint64_t>(ih) * context.compactStride;
                const uint64_t srcBase = static_cast<uint64_t>(ih) * context.zeroIndex;
                if (!dmaxInitialized) {
                    CopyLocalTensor(dmaxLocal[dstBase], xLocal[srcBase], context.zeroIndex);
                } else {
                    Max(dmaxLocal[dstBase], dmaxLocal[dstBase], xLocal[srcBase], context.zeroIndex);
                }
                PipeBarrier<PIPE_V>();
            }
            dmaxInitialized = true;
            xInQue_.FreeTensor(xLocal);
        }
        return dmaxInitialized;
    }

    __aicore__ inline void StoreNdhwcD3H3PlaneCopyRows(LocalTensor<T> compactLocal, LocalTensor<T> dmaxLocal,
                                                       LocalTensor<T> outLocal, LocalTensor<uint32_t> offsetLocal,
                                                       uint64_t outputOffset, uint32_t ohStart, uint32_t rows,
                                                       uint32_t cCount, uint32_t block, uint32_t outW,
                                                       const NdhwcD3H3PlaneCopyContext& context)
    {
        uint32_t doneRows = 0U;
        while (doneRows < rows) {
            uint32_t tileRows = rows - doneRows;
            if (tileRows > context.maxTileRows) {
                tileRows = context.maxTileRows;
            }
            MaxNdc1hwc0NdhwcD3H3CompactRows(compactLocal, dmaxLocal, ohStart + doneRows, tileRows,
                                            context.compactStride);
            if (!ScatterNdc1hwc0CompactRowsWithRowOffset(outLocal, compactLocal, offsetLocal, tileRows,
                                                         context.compactStride, block, outW, context.zeroIndex)) {
                ScatterNdc1hwc0CompactRowsChecked(outLocal, compactLocal, offsetLocal, tileRows, context.compactStride,
                                                  cCount, block, outW, cCount, 1U, context.zeroIndex);
            }
            CopyOutVector(outputOffset + static_cast<uint64_t>(doneRows) * context.rowElements, outLocal,
                          tileRows * context.rowElements);
            doneRows += tileRows;
        }
    }

    __aicore__ inline bool ProcessNdc1hwc0NdhwcD3H3Dil2PlaneCopyTile(uint64_t outputOffset, int64_t nIdx, int64_t od,
                                                                     uint32_t ohStart, uint32_t rows, uint32_t cCount,
                                                                     uint32_t block, uint32_t outW)
    {
        NdhwcD3H3PlaneCopyContext context;
        if (!InitNdhwcD3H3PlaneCopyContext(rows, cCount, block, outW, context)) {
            return false;
        }
        LocalTensor<T> dmaxLocal = calcBuf_.Get<T>();
        if (!LoadNdhwcD3H3PlaneCopyDmax(dmaxLocal, nIdx, od, context)) {
            return false;
        }
        LocalTensor<T> compactLocal = tmpBuf_.Get<T>();
        LocalTensor<T> outLocal = maskBuf_.Get<T>();
        const uint32_t offsetBase = AlignToVector(context.maxTileRows * context.compactStride);
        LocalTensor<uint32_t> offsetLocal = compactLocal[offsetBase].template ReinterpretCast<uint32_t>();
        InitNdc1hwc0CompactRowActiveOffsets(offsetLocal, 0U, cCount, block, outW, cCount, 1U, context.zeroIndex);
        StoreNdhwcD3H3PlaneCopyRows(compactLocal, dmaxLocal, outLocal, offsetLocal, outputOffset, ohStart, rows, cCount,
                                    block, outW, context);
        return true;
    }

    __aicore__ inline bool HasNdc1hwc0DilatedWDirectShape(uint32_t dataFormat) const
    {
        return tiling_->dataFormat == dataFormat && tiling_->outW > 0 && tiling_->outH > 0 && tiling_->inW > 0 &&
               tiling_->c > 0;
    }

    __aicore__ inline bool HasNdc1hwc0DilatedWDirectLayout(uint64_t block) const
    {
        return block > 0U && tiling_->outputC1 == 1 && static_cast<uint64_t>(tiling_->c) <= block;
    }

    __aicore__ inline bool HasNdc1hwc0DilatedWDirectPoolSpec() const
    {
        const bool kernelAndStride = tiling_->kD == 1 && tiling_->kH == 1 && tiling_->kW == 3 && tiling_->sD == 3 &&
                                     tiling_->sH == 3 && tiling_->sW == 3;
        const bool dilation = tiling_->dilationD == 1 && tiling_->dilationH == 1 && tiling_->dilationW == 2;
        return kernelAndStride && dilation;
    }

    __aicore__ inline bool HasNdc1hwc0NcdhwDilatedWDirectCapacity(uint64_t block) const
    {
        const uint32_t outW = static_cast<uint32_t>(tiling_->outW);
        const uint32_t rowElements = static_cast<uint32_t>(static_cast<uint64_t>(outW) * block);
        const uint32_t alignedInputW = AlignToVector(static_cast<uint32_t>(tiling_->inW));
        const uint32_t alignedW = AlignToVector(outW);
        const uint32_t cCount = static_cast<uint32_t>(tiling_->c);
        const uint32_t compactCount = cCount * alignedW;
        const uint32_t accStride = AlignToVector(compactCount + 1U);
        const uint32_t inputOffsetNeed = Ndc1hwc0GatherTempOffset(compactCount);
        const uint32_t rowOffsetNeed = Ndc1hwc0GatherTempOffset(rowElements);
        const uint32_t gatheredOffset = inputOffsetNeed * 3U;
        const uint32_t scatterOffset = AlignToVector(gatheredOffset + accStride);
        const uint32_t scratchNeed = scatterOffset + rowOffsetNeed;
        return rowElements > 0U && rowElements <= OUTPUT_TILE_NUM &&
               static_cast<uint64_t>(cCount) * alignedInputW + 1U <= INPUT_TILE_NUM && accStride <= OUTPUT_TILE_NUM &&
               scratchNeed <= OUTPUT_TILE_NUM &&
               CanUseNdc1hwc0NcdhwRowGather(cCount, static_cast<uint32_t>(block), outW, alignedW);
    }

    __aicore__ inline bool CanUseNdc1hwc0NcdhwDilatedWDirectPath() const
    {
        const uint64_t block = Ndc1hwc0Block();
        const bool supportedPad = tiling_->padFront == 2 && tiling_->padTop == 1 && tiling_->padLeft == 1;
        return HasNdc1hwc0DilatedWDirectShape(FORMAT_NCDHW_VALUE) && HasNdc1hwc0DilatedWDirectLayout(block) &&
               HasNdc1hwc0DilatedWDirectPoolSpec() && supportedPad && HasNdc1hwc0NcdhwDilatedWDirectCapacity(block);
    }

    __aicore__ inline void ProcessNdc1hwc0DilatedWDirectStep(uint64_t& cur, uint64_t outEnd, uint64_t block,
                                                             uint64_t validC1, uint32_t rowElements, uint32_t outW,
                                                             LocalTensor<T> outLocal, bool isNcdhw)
    {
        if (rowElements == 0U) {
            cur = outEnd;
            return;
        }
        if (!PrepareNdc1hwc0FullRow(cur, outEnd, block, rowElements, validC1)) {
            return;
        }
        const uint64_t remainRows64 = (outEnd - cur) / rowElements;
        uint32_t tileRows = remainRows64 > static_cast<uint64_t>(OUTPUT_TILE_NUM / rowElements) ?
                                OUTPUT_TILE_NUM / rowElements :
                                static_cast<uint32_t>(remainRows64);
        if (tileRows == 0U) {
            tileRows = 1U;
        }
        int64_t nIdx = 0;
        int64_t od = 0;
        int64_t c1Idx = 0;
        int64_t oh = 0;
        DecodeNdc1hwc0Row(cur / rowElements, validC1, nIdx, od, c1Idx, oh);
        if (c1Idx != 0) {
            tileRows = 1U;
        }
        if (isNcdhw) {
            FillNdc1hwc0NcdhwDilatedWDirectRows(cur / rowElements, tileRows, validC1, outLocal,
                                                static_cast<uint32_t>(block), outW);
        } else {
            FillNdc1hwc0NdhwcDilatedWDirectRows(cur / rowElements, tileRows, validC1, outLocal,
                                                static_cast<uint32_t>(block), outW);
        }
        CopyOutVector(cur, outLocal, tileRows * rowElements);
        cur += static_cast<uint64_t>(tileRows) * rowElements;
    }

    __aicore__ inline void ProcessNdc1hwc0DilatedWDirect(bool isNcdhw)
    {
        uint64_t block = 0U, validC1 = 0U, validOut = 0U;
        uint64_t outOffset = 0U, outCount = 0U, outEnd = 0U;
        uint32_t outW = 0U, rowElements = 0U;
        if (!InitNdc1hwc0GroupRange(block, validC1, validOut, outW, rowElements, outOffset, outCount, outEnd)) {
            return;
        }
        LocalTensor<T> outLocal = maskBuf_.Get<T>();
        uint64_t cur = outOffset;
        while (cur < outEnd) {
            ProcessNdc1hwc0DilatedWDirectStep(cur, outEnd, block, validC1, rowElements, outW, outLocal, isNcdhw);
        }
        if (validOut < tiling_->totalOut) {
            CopyOutZeroRangeByCore(validOut, tiling_->totalOut - validOut);
        }
    }

    __aicore__ inline void ProcessNdc1hwc0NcdhwDilatedWDirect() { ProcessNdc1hwc0DilatedWDirect(true); }

    __aicore__ inline void FillNdc1hwc0NcdhwDilatedWDirectRow(
        uint64_t startRow, uint32_t rowIdx, uint64_t validC1, LocalTensor<T> outLocal, LocalTensor<T> accLocal,
        LocalTensor<T> gatheredLocal, LocalTensor<uint32_t> offsetLocal, LocalTensor<uint32_t> offset1Local,
        LocalTensor<uint32_t> offset2Local, LocalTensor<uint32_t> scatterOffsetLocal, uint32_t rowElements,
        uint32_t cCount, uint32_t inW, uint32_t alignedInputW, uint32_t compactCount, uint32_t zeroIndex,
        uint32_t accZeroIndex, uint32_t srcStrideElements)
    {
        int64_t nIdx = 0;
        int64_t od = 0;
        int64_t c1Idx = 0;
        int64_t oh = 0;
        DecodeNdc1hwc0Row(startRow + rowIdx, validC1, nIdx, od, c1Idx, oh);
        if (c1Idx != 0) {
            return;
        }
        const int64_t id = od * tiling_->sD - tiling_->padFront;
        const int64_t ih = oh * tiling_->sH - tiling_->padTop;
        LocalTensor<T> rowLocal = outLocal[static_cast<uint64_t>(rowIdx) * rowElements];
        if (!IsOutOfRange(id, tiling_->inD) && !IsOutOfRange(ih, tiling_->inH)) {
            CopyInVectorWBlocksPadStrideValue(InputOffset(nIdx, id, ih, 0, 0), cCount, inW, alignedInputW,
                                              srcStrideElements, NegInfValue());
            LocalTensor<T> xLocal = xInQue_.DeQue<T>();
            Duplicate(xLocal[zeroIndex], NegInfValue(), 1);
            PipeBarrier<PIPE_V>();
            GatherNdc1hwc0CompactTile(accLocal, xLocal, offsetLocal, compactCount);
            GatherNdc1hwc0CompactTile(gatheredLocal, xLocal, offset1Local, compactCount);
            Max(accLocal, accLocal, gatheredLocal, compactCount);
            PipeBarrier<PIPE_V>();
            GatherNdc1hwc0CompactTile(gatheredLocal, xLocal, offset2Local, compactCount);
            Max(accLocal, accLocal, gatheredLocal, compactCount);
            PipeBarrier<PIPE_V>();
            ScatterNdc1hwc0CompactRowWithOffsets(rowLocal, accLocal, scatterOffsetLocal, rowElements, accZeroIndex);
            xInQue_.FreeTensor(xLocal);
            return;
        }
        Duplicate(accLocal, NegInfValue(), compactCount);
        PipeBarrier<PIPE_V>();
        ScatterNdc1hwc0CompactRowWithOffsets(rowLocal, accLocal, scatterOffsetLocal, rowElements, accZeroIndex);
    }

    struct Ndc1hwc0DilatedWScratchLayout {
        uint32_t inputOffsetNeed;
        uint32_t gatheredOffset;
        uint32_t scatterOffset;
    };

    __aicore__ inline Ndc1hwc0DilatedWScratchLayout GetNdc1hwc0DilatedWScratchLayout(uint32_t compactCount,
                                                                                     uint32_t accStride) const
    {
        const uint32_t inputOffsetNeed = Ndc1hwc0GatherTempOffset(compactCount);
        const uint32_t gatheredOffset = inputOffsetNeed * 3U;
        return {inputOffsetNeed, gatheredOffset, AlignToVector(gatheredOffset + accStride)};
    }

    __aicore__ inline void FillNdc1hwc0NcdhwDilatedWDirectRows(uint64_t startRow, uint32_t tileRows, uint64_t validC1,
                                                               LocalTensor<T> outLocal, uint32_t block, uint32_t outW)
    {
        const uint32_t rowElements = outW * block;
        const uint32_t cCount = static_cast<uint32_t>(tiling_->c);
        const uint32_t inW = static_cast<uint32_t>(tiling_->inW);
        const uint32_t alignedInputW = AlignToVector(inW);
        const uint32_t alignedW = AlignToVector(outW);
        const uint32_t compactCount = cCount * alignedW;
        const uint32_t accStride = AlignToVector(compactCount + 1U);
        const Ndc1hwc0DilatedWScratchLayout scratchLayout = GetNdc1hwc0DilatedWScratchLayout(compactCount, accStride);
        LocalTensor<T> accLocal = calcBuf_.Get<T>();
        LocalTensor<T> tmpLocal = tmpBuf_.Get<T>();
        LocalTensor<uint32_t> offsetLocal = tmpLocal.template ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> offset1Local = tmpLocal[scratchLayout.inputOffsetNeed]
                                                 .template ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> offset2Local = tmpLocal[scratchLayout.inputOffsetNeed * 2U]
                                                 .template ReinterpretCast<uint32_t>();
        LocalTensor<T> gatheredLocal = tmpLocal[scratchLayout.gatheredOffset];
        LocalTensor<uint32_t> scatterOffsetLocal = tmpLocal[scratchLayout.scatterOffset]
                                                       .template ReinterpretCast<uint32_t>();
        const uint32_t zeroIndex = cCount * alignedInputW;
        const uint32_t accZeroIndex = compactCount;
        InitNdc1hwc0NcdhwDilatedWDirectOffsets(offsetLocal, cCount, outW, alignedW, alignedInputW, zeroIndex, 0);
        InitNdc1hwc0NcdhwDilatedWDirectOffsets(offset1Local, cCount, outW, alignedW, alignedInputW, zeroIndex, 1);
        InitNdc1hwc0NcdhwDilatedWDirectOffsets(offset2Local, cCount, outW, alignedW, alignedInputW, zeroIndex, 2);
        InitNdc1hwc0CompactRowActiveOffsets(scatterOffsetLocal, 0U, cCount, block, outW, 1U, alignedW, accZeroIndex);
        const uint64_t channelStride = static_cast<uint64_t>(tiling_->inD) * static_cast<uint64_t>(tiling_->inH) *
                                       static_cast<uint64_t>(tiling_->inW);
        const uint32_t srcStrideElements = static_cast<uint32_t>(channelStride - static_cast<uint64_t>(inW));
        for (uint32_t rowIdx = 0; rowIdx < tileRows; ++rowIdx) {
            FillNdc1hwc0NcdhwDilatedWDirectRow(startRow, rowIdx, validC1, outLocal, accLocal, gatheredLocal,
                                               offsetLocal, offset1Local, offset2Local, scatterOffsetLocal, rowElements,
                                               cCount, inW, alignedInputW, compactCount, zeroIndex, accZeroIndex,
                                               srcStrideElements);
        }
    }

    __aicore__ inline void InitNdc1hwc0NcdhwDilatedWDirectOffsets(LocalTensor<uint32_t> offsetLocal, uint32_t cCount,
                                                                  uint32_t outW, uint32_t alignedW,
                                                                  uint32_t alignedInputW, uint32_t zeroIndex,
                                                                  int64_t kw)
    {
        LocalTensor<int32_t> offsetI32 = offsetLocal.template ReinterpretCast<int32_t>();
        const int32_t zeroOffset = static_cast<int32_t>(zeroIndex * sizeof(T));
        for (uint32_t c0 = 0; c0 < cCount; ++c0) {
            const uint32_t dstBase = c0 * alignedW;
            const uint32_t srcBase = c0 * alignedInputW;
            for (uint32_t ow = 0; ow < alignedW; ++ow) {
                int32_t srcOffset = zeroOffset;
                if (ow < outW) {
                    const int64_t iw = static_cast<int64_t>(ow) * tiling_->sW + kw * tiling_->dilationW -
                                       tiling_->padLeft;
                    if (!IsOutOfRange(iw, tiling_->inW)) {
                        srcOffset = static_cast<int32_t>((srcBase + static_cast<uint32_t>(iw)) * sizeof(T));
                    }
                }
                offsetI32.SetValue(dstBase + ow, srcOffset);
            }
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline bool HasNdc1hwc0NdhwcDilatedWDirectCapacity(uint64_t block) const
    {
        const uint32_t cCount = static_cast<uint32_t>(tiling_->c);
        const uint32_t outW = static_cast<uint32_t>(tiling_->outW);
        const uint32_t inW = static_cast<uint32_t>(tiling_->inW);
        const uint32_t rowElements = static_cast<uint32_t>(static_cast<uint64_t>(outW) * block);
        const uint32_t inputCount = inW * cCount;
        const uint32_t compactCount = outW * cCount;
        const uint32_t inputZeroIndex = inputCount;
        const uint32_t alignedInputBase = AlignToVector(inputCount);
        const uint32_t alignedInputCount = alignedInputBase == inputCount ? AlignToVector(inputCount + 1U) :
                                                                            alignedInputBase;
        const uint32_t accZeroIndex = AlignToVector(compactCount);
        const uint32_t accStride = AlignToVector(accZeroIndex + 1U);
        const uint32_t inputOffsetNeed = Ndc1hwc0GatherTempOffset(compactCount);
        const uint32_t rowOffsetNeed = Ndc1hwc0GatherTempOffset(rowElements);
        const uint32_t gatheredOffset = inputOffsetNeed * 3U;
        const uint32_t scatterOffset = AlignToVector(gatheredOffset + accStride);
        const uint32_t scratchNeed = scatterOffset + rowOffsetNeed;
        return rowElements > 0U && rowElements <= OUTPUT_TILE_NUM && compactCount > 0U && inputCount > 0U &&
               alignedInputCount <= INPUT_TILE_NUM && accStride <= OUTPUT_TILE_NUM && scratchNeed <= OUTPUT_TILE_NUM;
    }

    __aicore__ inline bool CanUseNdc1hwc0NdhwcDilatedWDirectPath() const
    {
        const uint64_t block = Ndc1hwc0Block();
        const bool nonnegativePad = tiling_->padFront >= 0 && tiling_->padTop >= 0 && tiling_->padLeft >= 0;
        return HasNdc1hwc0DilatedWDirectShape(FORMAT_NDHWC_VALUE) && HasNdc1hwc0DilatedWDirectLayout(block) &&
               HasNdc1hwc0DilatedWDirectPoolSpec() && nonnegativePad && HasNdc1hwc0NdhwcDilatedWDirectCapacity(block);
    }

    __aicore__ inline void ProcessNdc1hwc0NdhwcDilatedWDirect() { ProcessNdc1hwc0DilatedWDirect(false); }

    __aicore__ inline void ReduceNdc1hwc0NdhwcDilatedWDirectRow(
        LocalTensor<T> rowLocal, LocalTensor<T> inputLocal, LocalTensor<T> accLocal, LocalTensor<T> gatheredLocal,
        LocalTensor<uint32_t> offsetLocal, LocalTensor<uint32_t> offset1Local, LocalTensor<uint32_t> offset2Local,
        LocalTensor<uint32_t> scatterOffsetLocal, uint32_t compactCount, uint32_t rowElements)
    {
        GatherNdc1hwc0CompactTile(accLocal, inputLocal, offsetLocal, compactCount);
        GatherNdc1hwc0CompactTile(gatheredLocal, inputLocal, offset1Local, compactCount);
        Max(accLocal, accLocal, gatheredLocal, compactCount);
        PipeBarrier<PIPE_V>();
        GatherNdc1hwc0CompactTile(gatheredLocal, inputLocal, offset2Local, compactCount);
        Max(accLocal, accLocal, gatheredLocal, compactCount);
        PipeBarrier<PIPE_V>();
        ScatterNdc1hwc0CompactRowWithOffsetsNoZero(rowLocal, accLocal, scatterOffsetLocal, rowElements);
    }

    __aicore__ inline uint32_t Ndc1hwc0NdhwcDilatedWBatchRows(uint64_t startRow, uint32_t rowIdx, uint32_t tileRows,
                                                              uint64_t validC1, int64_t nIdx, int64_t od, int64_t id,
                                                              int64_t ih, uint32_t alignedInputCount) const
    {
        uint32_t batchRows = 1U;
        const uint32_t maxBatchRows = alignedInputCount == 0U ? 1U : INPUT_TILE_NUM / alignedInputCount;
        while (rowIdx + batchRows < tileRows && batchRows < maxBatchRows) {
            int64_t nextN = 0;
            int64_t nextOd = 0;
            int64_t nextC1 = 0;
            int64_t nextOh = 0;
            DecodeNdc1hwc0Row(startRow + rowIdx + batchRows, validC1, nextN, nextOd, nextC1, nextOh);
            const int64_t nextId = nextOd * tiling_->sD - tiling_->padFront;
            const int64_t nextIh = nextOh * tiling_->sH - tiling_->padTop;
            if (nextN != nIdx || nextOd != od || nextC1 != 0 || nextId != id || IsOutOfRange(nextIh, tiling_->inH) ||
                nextIh != ih + static_cast<int64_t>(batchRows) * tiling_->sH) {
                break;
            }
            ++batchRows;
        }
        return batchRows;
    }

    __aicore__ inline void FillNdc1hwc0NdhwcDilatedWBatch(
        uint32_t rowIdx, uint32_t batchRows, LocalTensor<T> outLocal, LocalTensor<T> accLocal,
        LocalTensor<T> gatheredLocal, LocalTensor<uint32_t> offsetLocal, LocalTensor<uint32_t> offset1Local,
        LocalTensor<uint32_t> offset2Local, LocalTensor<uint32_t> scatterOffsetLocal, uint32_t rowElements,
        uint32_t inputCount, uint32_t alignedInputCount, uint32_t compactCount, int64_t nIdx, int64_t id, int64_t ih,
        uint32_t srcStrideElements)
    {
        CopyInVectorWBlocksPadStrideValue(InputOffset(nIdx, id, ih, 0, 0), batchRows, inputCount, alignedInputCount,
                                          srcStrideElements, NegInfValue());
        LocalTensor<T> xLocal = xInQue_.DeQue<T>();
        for (uint32_t batchIdx = 0U; batchIdx < batchRows; ++batchIdx) {
            LocalTensor<T> batchRowLocal = outLocal[static_cast<uint64_t>(rowIdx + batchIdx) * rowElements];
            LocalTensor<T> batchInput = xLocal[static_cast<uint64_t>(batchIdx) * alignedInputCount];
            ReduceNdc1hwc0NdhwcDilatedWDirectRow(batchRowLocal, batchInput, accLocal, gatheredLocal, offsetLocal,
                                                 offset1Local, offset2Local, scatterOffsetLocal, compactCount,
                                                 rowElements);
        }
        xInQue_.FreeTensor(xLocal);
    }

    __aicore__ inline uint32_t FillNdc1hwc0NdhwcDilatedWRowGroup(
        uint64_t startRow, uint32_t rowIdx, uint32_t tileRows, uint64_t validC1, LocalTensor<T> outLocal,
        LocalTensor<T> accLocal, LocalTensor<T> gatheredLocal, LocalTensor<uint32_t> offsetLocal,
        LocalTensor<uint32_t> offset1Local, LocalTensor<uint32_t> offset2Local,
        LocalTensor<uint32_t> scatterOffsetLocal, uint32_t rowElements, uint32_t inputCount, uint32_t alignedInputCount,
        uint32_t compactCount)
    {
        int64_t nIdx = 0;
        int64_t od = 0;
        int64_t c1Idx = 0;
        int64_t oh = 0;
        DecodeNdc1hwc0Row(startRow + rowIdx, validC1, nIdx, od, c1Idx, oh);
        LocalTensor<T> rowLocal = outLocal[static_cast<uint64_t>(rowIdx) * rowElements];
        const int64_t id = od * tiling_->sD - tiling_->padFront;
        const int64_t ih = oh * tiling_->sH - tiling_->padTop;
        if (c1Idx != 0 || IsOutOfRange(id, tiling_->inD) || IsOutOfRange(ih, tiling_->inH)) {
            Duplicate(accLocal, NegInfValue(), compactCount);
            PipeBarrier<PIPE_V>();
            ScatterNdc1hwc0CompactRowWithOffsetsNoZero(rowLocal, accLocal, scatterOffsetLocal, rowElements);
            return 1U;
        }
        const uint32_t batchRows = Ndc1hwc0NdhwcDilatedWBatchRows(startRow, rowIdx, tileRows, validC1, nIdx, od, id, ih,
                                                                  alignedInputCount);
        const uint32_t srcStrideElements = static_cast<uint32_t>((tiling_->sH - 1) * tiling_->inW * tiling_->c);
        if (batchRows > 1U && srcStrideElements <= 65535U) {
            FillNdc1hwc0NdhwcDilatedWBatch(rowIdx, batchRows, outLocal, accLocal, gatheredLocal, offsetLocal,
                                           offset1Local, offset2Local, scatterOffsetLocal, rowElements, inputCount,
                                           alignedInputCount, compactCount, nIdx, id, ih, srcStrideElements);
            return batchRows;
        }
        CopyInVectorPadValue(InputOffset(nIdx, id, ih, 0, 0), inputCount, alignedInputCount, NegInfValue());
        LocalTensor<T> xLocal = xInQue_.DeQue<T>();
        ReduceNdc1hwc0NdhwcDilatedWDirectRow(rowLocal, xLocal, accLocal, gatheredLocal, offsetLocal, offset1Local,
                                             offset2Local, scatterOffsetLocal, compactCount, rowElements);
        xInQue_.FreeTensor(xLocal);
        return 1U;
    }

    __aicore__ inline void FillNdc1hwc0NdhwcDilatedWDirectRows(uint64_t startRow, uint32_t tileRows, uint64_t validC1,
                                                               LocalTensor<T> outLocal, uint32_t block, uint32_t outW)
    {
        const uint32_t rowElements = outW * block;
        const uint32_t cCount = static_cast<uint32_t>(tiling_->c);
        const uint32_t inputCount = static_cast<uint32_t>(tiling_->inW) * cCount;
        const uint32_t compactCount = outW * cCount;
        const uint32_t accZeroIndex = AlignToVector(compactCount);
        const uint32_t accStride = AlignToVector(accZeroIndex + 1U);
        const Ndc1hwc0DilatedWScratchLayout scratchLayout = GetNdc1hwc0DilatedWScratchLayout(compactCount, accStride);
        LocalTensor<T> tmpLocal = tmpBuf_.Get<T>();
        LocalTensor<uint32_t> offsetLocal = tmpLocal.template ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> offset1Local = tmpLocal[scratchLayout.inputOffsetNeed]
                                                 .template ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> offset2Local = tmpLocal[scratchLayout.inputOffsetNeed * 2U]
                                                 .template ReinterpretCast<uint32_t>();
        LocalTensor<T> gatheredLocal = tmpLocal[scratchLayout.gatheredOffset];
        LocalTensor<uint32_t> scatterOffsetLocal = tmpLocal[scratchLayout.scatterOffset]
                                                       .template ReinterpretCast<uint32_t>();
        LocalTensor<T> accLocal = calcBuf_.Get<T>();
        const uint32_t inputZeroIndex = inputCount;
        const uint32_t alignedInputBase = AlignToVector(inputCount);
        const uint32_t alignedInputCount = alignedInputBase == inputCount ? AlignToVector(inputCount + 1U) :
                                                                            alignedInputBase;
        InitNdc1hwc0NdhwcDilatedWDirectOffsets(offsetLocal, cCount, outW, compactCount, compactCount, inputZeroIndex,
                                               0);
        InitNdc1hwc0NdhwcDilatedWDirectOffsets(offset1Local, cCount, outW, compactCount, compactCount, inputZeroIndex,
                                               1);
        InitNdc1hwc0NdhwcDilatedWDirectOffsets(offset2Local, cCount, outW, compactCount, compactCount, inputZeroIndex,
                                               2);
        InitNdc1hwc0CompactRowActiveOffsets(scatterOffsetLocal, 0U, cCount, block, outW, cCount, 1U, accZeroIndex);
        Duplicate(accLocal[accZeroIndex], ZeroValue(), 1);
        PipeBarrier<PIPE_V>();
        uint32_t rowIdx = 0U;
        while (rowIdx < tileRows) {
            rowIdx += FillNdc1hwc0NdhwcDilatedWRowGroup(
                startRow, rowIdx, tileRows, validC1, outLocal, accLocal, gatheredLocal, offsetLocal, offset1Local,
                offset2Local, scatterOffsetLocal, rowElements, inputCount, alignedInputCount, compactCount);
        }
    }

    __aicore__ inline void InitNdc1hwc0NdhwcDilatedWDirectOffsets(LocalTensor<uint32_t> offsetLocal, uint32_t cCount,
                                                                  uint32_t outW, uint32_t compactCount,
                                                                  uint32_t alignedCompactCount, uint32_t zeroIndex,
                                                                  int64_t kw)
    {
        LocalTensor<int32_t> offsetI32 = offsetLocal.template ReinterpretCast<int32_t>();
        const int32_t zeroOffset = static_cast<int32_t>(zeroIndex * sizeof(T));
        for (uint32_t ow = 0; ow < outW; ++ow) {
            const uint32_t dstBase = ow * cCount;
            for (uint32_t c0 = 0; c0 < cCount; ++c0) {
                int32_t srcOffset = zeroOffset;
                const int64_t iw = static_cast<int64_t>(ow) * tiling_->sW + kw * tiling_->dilationW - tiling_->padLeft;
                if (!IsOutOfRange(iw, tiling_->inW)) {
                    srcOffset = static_cast<int32_t>((static_cast<uint32_t>(iw) * cCount + c0) * sizeof(T));
                }
                offsetI32.SetValue(dstBase + c0, srcOffset);
            }
        }
        for (uint32_t i = compactCount; i < alignedCompactCount; ++i) {
            offsetI32.SetValue(i, zeroOffset);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline bool HasNdc1hwc0HOnlyStride3Shape() const
    {
        return tiling_->outW > 0 && tiling_->outH > 0 && tiling_->inH > 0 && tiling_->c > 0;
    }

    __aicore__ inline bool HasNdc1hwc0HOnlyStride3PoolSpec() const
    {
        const bool kernelAndStride = tiling_->kD == 1 && tiling_->kH == 3 && tiling_->kW == 1 && tiling_->sD == 1 &&
                                     tiling_->sH == 3 && tiling_->sW == 1;
        const bool dilationAndPad = tiling_->dilationD == 1 && tiling_->dilationH == 1 && tiling_->dilationW == 1 &&
                                    tiling_->padFront == 0 && tiling_->padLeft == 0;
        return kernelAndStride && dilationAndPad && tiling_->outW == tiling_->inW && tiling_->outD <= tiling_->inD;
    }

    __aicore__ inline bool HasNdc1hwc0HOnlyStride3Layout(uint64_t block) const
    {
        return block > 0U && tiling_->outputC1 == 1 && static_cast<uint64_t>(tiling_->c) <= block;
    }

    __aicore__ inline bool HasNdc1hwc0NcdhwHOnlyStride3Capacity(uint64_t block) const
    {
        const uint32_t cCount = static_cast<uint32_t>(tiling_->c);
        const uint32_t outW = static_cast<uint32_t>(tiling_->outW);
        const uint32_t rowElements = static_cast<uint32_t>(static_cast<uint64_t>(outW) * block);
        const uint32_t blockElements = UB_BLOCK_BYTES / sizeof(T);
        const uint32_t alignedW = (outW + blockElements - 1U) / blockElements * blockElements;
        const uint32_t rowData = cCount * alignedW;
        const uint32_t rowStride = (rowData + 1U + blockElements - 1U) / blockElements * blockElements;
        const uint32_t planeCompact = static_cast<uint32_t>(tiling_->outH) * rowStride;
        const uint32_t planeInput = static_cast<uint32_t>(tiling_->inH) * alignedW;
        return planeCompact + 1U <= OUTPUT_TILE_NUM && planeInput <= INPUT_TILE_NUM &&
               Ndc1hwc0GatherTempOffset(rowElements) <= OUTPUT_TILE_NUM;
    }

    __aicore__ inline bool HasNdc1hwc0NdhwcHOnlyStride3Capacity(uint64_t block) const
    {
        const uint32_t cCount = static_cast<uint32_t>(tiling_->c);
        const uint32_t outW = static_cast<uint32_t>(tiling_->outW);
        const uint32_t rowElements = static_cast<uint32_t>(static_cast<uint64_t>(outW) * block);
        const uint32_t compactCount = outW * cCount;
        const uint32_t alignedCompactCount = AlignToVector(compactCount);
        const uint32_t blockElements = UB_BLOCK_BYTES / sizeof(T);
        const uint32_t rowStride = (alignedCompactCount + 1U + blockElements - 1U) / blockElements * blockElements;
        const uint32_t planeCompact = static_cast<uint32_t>(tiling_->outH) * rowStride;
        const uint32_t planeInput = static_cast<uint32_t>(tiling_->inH) * alignedCompactCount;
        return compactCount > 0U && planeInput <= INPUT_TILE_NUM && planeCompact <= OUTPUT_TILE_NUM &&
               Ndc1hwc0GatherTempOffset(rowElements) <= OUTPUT_TILE_NUM;
    }

    __aicore__ inline bool CanUseNdc1hwc0HOnlyStride3GroupPath() const
    {
        const uint64_t block = Ndc1hwc0Block();
        if (!HasNdc1hwc0HOnlyStride3Shape() || !HasNdc1hwc0HOnlyStride3Layout(block) ||
            !HasNdc1hwc0HOnlyStride3PoolSpec()) {
            return false;
        }
        const uint64_t rowElements = static_cast<uint64_t>(tiling_->outW) * block;
        const uint32_t planeElements = static_cast<uint32_t>(static_cast<uint64_t>(tiling_->outH) * rowElements);
        if (rowElements == 0U || rowElements > OUTPUT_TILE_NUM || planeElements == 0U ||
            planeElements > OUTPUT_TILE_NUM) {
            return false;
        }
        if (tiling_->dataFormat == FORMAT_NCDHW_VALUE) {
            return HasNdc1hwc0NcdhwHOnlyStride3Capacity(block);
        }
        return HasNdc1hwc0NdhwcHOnlyStride3Capacity(block);
    }

    __aicore__ inline bool CanUseNdc1hwc0LogicalHOnlyPlanePath() const
    {
        const uint64_t block = Ndc1hwc0Block();
        const uint64_t validC1 = Ndc1hwc0ValidC1(block);
        if constexpr (AscendC::Std::is_same<T, bfloat16_t>::value) {
            return false;
        }
        if (tiling_->inputLayout == INPUT_LAYOUT_NDC1HWC0_VALUE ||
            tiling_->outputLayout != OUTPUT_LAYOUT_NDC1HWC0_VALUE || block != 16U || validC1 != 1U ||
            tiling_->outputC1 != 1 || tiling_->c <= 0 || static_cast<uint64_t>(tiling_->c) > block ||
            !HasNdc1hwc0HOnlyStride3Shape() || !HasNdc1hwc0HOnlyStride3PoolSpec()) {
            return false;
        }
        const uint64_t inputPlane = static_cast<uint64_t>(tiling_->inH) * tiling_->inW;
        const uint64_t alignedInputPlane = AlignToVector(static_cast<uint32_t>(inputPlane));
        const uint64_t blockedInputPlane = inputPlane * block;
        const uint64_t outputPlane = static_cast<uint64_t>(tiling_->outH) * tiling_->outW * block;
        const uint64_t rowElements = static_cast<uint64_t>(tiling_->outW) * block;
        const uint64_t rowStride = rowElements * sizeof(T) / UB_BLOCK_BYTES;
        const uint64_t ncdhwNeed = (static_cast<uint64_t>(tiling_->c) + 1U) * alignedInputPlane + blockedInputPlane;
        const uint64_t ndhwcNeed = blockedInputPlane;
        return inputPlane > 0U && alignedInputPlane >= inputPlane && alignedInputPlane - inputPlane <= 255U &&
               ((tiling_->dataFormat == FORMAT_NCDHW_VALUE && ncdhwNeed <= INPUT_TILE_NUM) ||
                (tiling_->dataFormat == FORMAT_NDHWC_VALUE && ndhwcNeed <= INPUT_TILE_NUM)) &&
               outputPlane > 0U && outputPlane <= OUTPUT_TILE_NUM && rowStride > 0U && rowStride <= 255U &&
               tiling_->outH > 2 && tiling_->outH - 2 <= 255;
    }

    __aicore__ inline LocalTensor<T> LoadNdc1hwc0LogicalHOnlyPlane(LocalTensor<T> inputLocal, uint32_t plane,
                                                                   uint32_t block, uint32_t inputPlane,
                                                                   uint32_t alignedInputPlane)
    {
        const uint32_t nIdx = plane / static_cast<uint32_t>(tiling_->inD);
        const uint32_t dIdx = plane - nIdx * static_cast<uint32_t>(tiling_->inD);
        if (tiling_->dataFormat == FORMAT_NDHWC_VALUE) {
            const DataCopyExtParams inputCopy{static_cast<uint16_t>(inputPlane),
                                              static_cast<uint32_t>(tiling_->c * sizeof(T)), 0U, 0U, 0U};
            const DataCopyPadExtParams<T> inputPad{
                true, 0U, static_cast<uint8_t>(block - static_cast<uint32_t>(tiling_->c)), ZeroValue()};
            const uint64_t inputOffset = (static_cast<uint64_t>(nIdx) * tiling_->inD + dIdx) * inputPlane * tiling_->c;
            DataCopyPad(inputLocal, xGm_[inputOffset], inputCopy, inputPad);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            return inputLocal;
        }
        const uint64_t nBase = static_cast<uint64_t>(nIdx) * tiling_->c * tiling_->inD * inputPlane;
        const uint64_t inputOffset = nBase + static_cast<uint64_t>(dIdx) * inputPlane;
        const DataCopyExtParams inputCopy{static_cast<uint16_t>(tiling_->c),
                                          static_cast<uint32_t>(inputPlane * sizeof(T)),
                                          static_cast<uint32_t>((tiling_->inD - 1) * inputPlane * sizeof(T)), 0U, 0U};
        const DataCopyPadExtParams<T> inputPad{true, 0U, static_cast<uint8_t>(alignedInputPlane - inputPlane),
                                               ZeroValue()};
        DataCopyPad(inputLocal, xGm_[inputOffset], inputCopy, inputPad);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        LocalTensor<T> zeroLocal = inputLocal[static_cast<uint64_t>(tiling_->c) * alignedInputPlane];
        Duplicate(zeroLocal, ZeroValue(), alignedInputPlane);
        PipeBarrier<PIPE_V>();
        LocalTensor<T> blockedLocal = zeroLocal[alignedInputPlane];
        TransposeNdc1hwc0C0PlaneBlock(blockedLocal, inputLocal, zeroLocal, alignedInputPlane,
                                      static_cast<uint32_t>(tiling_->c), block);
        return blockedLocal;
    }

    __aicore__ inline void ReduceNdc1hwc0LogicalHOnlyPlane(LocalTensor<T> inputLocal, LocalTensor<T> outputLocal,
                                                           uint32_t block)
    {
        const uint32_t rowElements = static_cast<uint32_t>(tiling_->outW) * block;
        Max(outputLocal, inputLocal, inputLocal[rowElements], rowElements);
        Max(outputLocal[static_cast<uint64_t>(tiling_->outH - 1) * rowElements],
            inputLocal[static_cast<uint64_t>(tiling_->inH - 2) * rowElements],
            inputLocal[static_cast<uint64_t>(tiling_->inH - 1) * rowElements], rowElements);
        const uint8_t rowStride = static_cast<uint8_t>(rowElements * sizeof(T) / UB_BLOCK_BYTES);
        const uint8_t middleRows = static_cast<uint8_t>(tiling_->outH - 2);
        const BinaryRepeatParams middleParams{
            1U, 1U, 1U, rowStride, static_cast<uint8_t>(3U * rowStride), static_cast<uint8_t>(3U * rowStride)};
        ReduceNdc1hwc0LogicalTinyK3Rows(outputLocal[rowElements], inputLocal[2U * rowElements],
                                        inputLocal[3U * rowElements], rowElements, middleRows, middleParams);
        PipeBarrier<PIPE_V>();
        const BinaryRepeatParams finishParams{1U, 1U, 1U, rowStride, rowStride, static_cast<uint8_t>(3U * rowStride)};
        ReduceNdc1hwc0LogicalTinyK3Rows(outputLocal[rowElements], outputLocal[rowElements],
                                        inputLocal[4U * rowElements], rowElements, middleRows, finishParams);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ProcessNdc1hwc0LogicalHOnlyPlanes()
    {
        const uint32_t block = static_cast<uint32_t>(Ndc1hwc0Block());
        const uint32_t inputPlane = static_cast<uint32_t>(tiling_->inH * tiling_->inW);
        const uint32_t alignedInputPlane = AlignToVector(inputPlane);
        const uint32_t outputPlane = static_cast<uint32_t>(tiling_->outH * tiling_->outW) * block;
        const uint32_t totalPlanes = static_cast<uint32_t>(tiling_->n * tiling_->inD);
        const uint32_t worker = GetBlockIdx();
        const uint32_t workerDim = ActiveBlockDim();
        for (uint32_t plane = worker; plane < totalPlanes; plane += workerDim) {
            LocalTensor<T> inputLocal = xInQue_.AllocTensor<T>();
            LocalTensor<T> blockedLocal = LoadNdc1hwc0LogicalHOnlyPlane(inputLocal, plane, block, inputPlane,
                                                                        alignedInputPlane);
            LocalTensor<T> outputLocal = calcBuf_.Get<T>();
            ReduceNdc1hwc0LogicalHOnlyPlane(blockedLocal, outputLocal, block);
            CopyOutVector(static_cast<uint64_t>(plane) * outputPlane, outputLocal, outputPlane);
            xInQue_.FreeTensor(inputLocal);
        }
        const uint64_t validOut = static_cast<uint64_t>(totalPlanes) * outputPlane;
        if (validOut < tiling_->totalOut) {
            CopyOutZeroRangeByCore(validOut, tiling_->totalOut - validOut);
        }
    }

    __aicore__ inline bool CanUseNdc1hwc0LogicalD3H3PlanePath() const
    {
        const uint64_t block = Ndc1hwc0Block();
        const uint64_t validC1 = Ndc1hwc0ValidC1(block);
        if constexpr (AscendC::Std::is_same<T, bfloat16_t>::value) {
            return false;
        }
        if (tiling_->inputLayout == INPUT_LAYOUT_NDC1HWC0_VALUE ||
            tiling_->outputLayout != OUTPUT_LAYOUT_NDC1HWC0_VALUE || block != 16U || validC1 != 1U ||
            tiling_->outputC1 != 1 || tiling_->c <= 0 || static_cast<uint64_t>(tiling_->c) > block ||
            !MatchesPoolSpec(3, 3, 1, 3, 1, 1, 1, 2, 1, 0, 2, 0) || tiling_->inD != tiling_->outD * 3 ||
            tiling_->outH != tiling_->inH || tiling_->outW != tiling_->inW || tiling_->inH < 4) {
            return false;
        }
        const uint64_t plane = static_cast<uint64_t>(tiling_->inH) * tiling_->inW;
        const uint64_t slab = plane * 3U;
        const uint64_t alignedSlab = AlignToVector(static_cast<uint32_t>(slab));
        const uint64_t blockedSlab = slab * block;
        const uint64_t outputPlane = plane * block;
        return plane > 0U && alignedSlab >= slab && alignedSlab - slab <= 255U &&
               ((tiling_->dataFormat == FORMAT_NCDHW_VALUE &&
                 static_cast<uint64_t>(tiling_->c) * alignedSlab + blockedSlab <= INPUT_TILE_NUM) ||
                (tiling_->dataFormat == FORMAT_NDHWC_VALUE && blockedSlab <= INPUT_TILE_NUM)) &&
               outputPlane <= OUTPUT_TILE_NUM;
    }

    __aicore__ inline LocalTensor<T> LoadNdc1hwc0LogicalD3H3Slab(LocalTensor<T> inputLocal, LocalTensor<T> zeroLocal,
                                                                 uint32_t nIdx, uint32_t od, uint32_t block,
                                                                 uint32_t plane, uint32_t slab, uint32_t alignedSlab)
    {
        const uint32_t inputDepthStart = od * 3U;
        if (tiling_->dataFormat == FORMAT_NDHWC_VALUE) {
            const DataCopyExtParams inputCopy{static_cast<uint16_t>(slab),
                                              static_cast<uint32_t>(tiling_->c * sizeof(T)), 0U, 0U, 0U};
            const DataCopyPadExtParams<T> inputPad{
                true, 0U, static_cast<uint8_t>(block - static_cast<uint32_t>(tiling_->c)), ZeroValue()};
            const uint64_t inputOffset = (static_cast<uint64_t>(nIdx) * tiling_->inD + inputDepthStart) * plane *
                                         tiling_->c;
            DataCopyPad(inputLocal, xGm_[inputOffset], inputCopy, inputPad);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            return inputLocal;
        }
        const uint64_t nBase = static_cast<uint64_t>(nIdx) * tiling_->c * tiling_->inD * plane;
        const uint64_t inputOffset = nBase + static_cast<uint64_t>(inputDepthStart) * plane;
        const DataCopyExtParams inputCopy{static_cast<uint16_t>(tiling_->c), static_cast<uint32_t>(slab * sizeof(T)),
                                          static_cast<uint32_t>((tiling_->inD * plane - slab) * sizeof(T)), 0U, 0U};
        const DataCopyPadExtParams<T> inputPad{true, 0U, static_cast<uint8_t>(alignedSlab - slab), ZeroValue()};
        DataCopyPad(inputLocal, xGm_[inputOffset], inputCopy, inputPad);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Duplicate(zeroLocal, ZeroValue(), alignedSlab);
        PipeBarrier<PIPE_V>();
        LocalTensor<T> blockedLocal = inputLocal[static_cast<uint64_t>(tiling_->c) * alignedSlab];
        TransposeNdc1hwc0C0PlaneBlock(blockedLocal, inputLocal, zeroLocal, alignedSlab,
                                      static_cast<uint32_t>(tiling_->c), block);
        return blockedLocal;
    }

    __aicore__ inline void ReduceNdc1hwc0LogicalD3H3Slab(LocalTensor<T> blockedLocal, LocalTensor<T> depthLocal,
                                                         LocalTensor<T> outputLocal, uint32_t block, uint32_t plane)
    {
        const uint32_t blockedPlane = plane * block;
        const uint32_t rowElements = static_cast<uint32_t>(tiling_->inW) * block;
        Max(depthLocal, blockedLocal, blockedLocal[blockedPlane], blockedPlane);
        PipeBarrier<PIPE_V>();
        Max(depthLocal, depthLocal, blockedLocal[2U * blockedPlane], blockedPlane);
        PipeBarrier<PIPE_V>();
        Max(outputLocal[2U * rowElements], depthLocal[2U * rowElements], depthLocal,
            static_cast<uint32_t>(tiling_->inH - 2) * rowElements);
        Max(outputLocal, depthLocal, depthLocal[2U * rowElements], 2U * rowElements);
        PipeBarrier<PIPE_V>();
        Max(outputLocal[2U * rowElements], outputLocal[2U * rowElements], depthLocal[4U * rowElements],
            static_cast<uint32_t>(tiling_->inH - 4) * rowElements);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ProcessNdc1hwc0LogicalD3H3Planes()
    {
        const uint32_t block = static_cast<uint32_t>(Ndc1hwc0Block());
        const uint32_t plane = static_cast<uint32_t>(tiling_->inH * tiling_->inW);
        const uint32_t slab = 3U * plane;
        const uint32_t alignedSlab = AlignToVector(slab);
        const uint32_t outputPlane = plane * block;
        const uint32_t totalUnits = static_cast<uint32_t>(tiling_->n * tiling_->outD);
        const uint32_t worker = GetBlockIdx();
        const uint32_t workerDim = ActiveBlockDim();
        for (uint32_t unit = worker; unit < totalUnits; unit += workerDim) {
            const uint32_t nIdx = unit / static_cast<uint32_t>(tiling_->outD);
            const uint32_t od = unit - nIdx * static_cast<uint32_t>(tiling_->outD);
            LocalTensor<T> inputLocal = xInQue_.AllocTensor<T>();
            LocalTensor<T> depthLocal = calcBuf_.Get<T>();
            LocalTensor<T> blockedLocal = LoadNdc1hwc0LogicalD3H3Slab(inputLocal, depthLocal, nIdx, od, block, plane,
                                                                      slab, alignedSlab);
            LocalTensor<T> outputLocal = maskBuf_.Get<T>();
            ReduceNdc1hwc0LogicalD3H3Slab(blockedLocal, depthLocal, outputLocal, block, plane);
            CopyOutVector(static_cast<uint64_t>(unit) * outputPlane, outputLocal, outputPlane);
            xInQue_.FreeTensor(inputLocal);
        }
        const uint64_t validOut = static_cast<uint64_t>(totalUnits) * outputPlane;
        if (validOut < tiling_->totalOut) {
            CopyOutZeroRangeByCore(validOut, tiling_->totalOut - validOut);
        }
    }

    __aicore__ inline bool CanUseNdc1hwc0LogicalD3W3RowGroupPath() const
    {
        const uint64_t block = Ndc1hwc0Block();
        const uint64_t validC1 = Ndc1hwc0ValidC1(block);
        if constexpr (AscendC::Std::is_same<T, bfloat16_t>::value) {
            return false;
        }
        if (tiling_->inputLayout == INPUT_LAYOUT_NDC1HWC0_VALUE || tiling_->dataFormat != FORMAT_NDHWC_VALUE ||
            tiling_->outputLayout != OUTPUT_LAYOUT_NDC1HWC0_VALUE || block != 16U || validC1 != 1U ||
            tiling_->outputC1 != 1 || tiling_->c <= 0 || static_cast<uint64_t>(tiling_->c) > block ||
            !MatchesPoolSpec(3, 1, 3, 3, 1, 3, 2, 1, 1, 0, 0, 0) || tiling_->outD <= 0 ||
            tiling_->outH != tiling_->inH || tiling_->outW * 3 != tiling_->inW) {
            return false;
        }
        const uint64_t totalPlanes = static_cast<uint64_t>(tiling_->n) * tiling_->outD;
        const uint64_t workerDim = ActiveBlockDim();
        if (totalPlanes == 0U || workerDim == 0U) {
            return false;
        }
        uint64_t groupsPerPlane = workerDim / totalPlanes;
        if (groupsPerPlane == 0U) {
            groupsPerPlane = 1U;
        }
        if (groupsPerPlane > static_cast<uint64_t>(tiling_->outH)) {
            groupsPerPlane = static_cast<uint64_t>(tiling_->outH);
        }
        const uint64_t maxRows = (static_cast<uint64_t>(tiling_->outH) + groupsPerPlane - 1U) / groupsPerPlane;
        const uint64_t inputNeed = 3U * maxRows * tiling_->inW * block;
        const uint64_t outputNeed = maxRows * tiling_->outW * block;
        return inputNeed > 0U && inputNeed <= INPUT_TILE_NUM && outputNeed > 0U && outputNeed <= OUTPUT_TILE_NUM &&
               maxRows * static_cast<uint64_t>(tiling_->outW) <= 255U;
    }

    __aicore__ inline void LoadNdc1hwc0LogicalD3W3Rows(LocalTensor<T> inputLocal, uint32_t nIdx, uint32_t od,
                                                       uint32_t hStart, uint32_t rows, uint32_t block)
    {
        const uint32_t inputPoints = rows * static_cast<uint32_t>(tiling_->inW);
        const uint32_t bankElements = inputPoints * block;
        const DataCopyExtParams inputCopy{static_cast<uint16_t>(inputPoints),
                                          static_cast<uint32_t>(tiling_->c * sizeof(T)), 0U, 0U, 0U};
        const DataCopyPadExtParams<T> inputPad{
            true, 0U, static_cast<uint8_t>(block - static_cast<uint32_t>(tiling_->c)), ZeroValue()};
        for (uint32_t kd = 0U; kd < 3U; ++kd) {
            const uint32_t inputDepth = od * static_cast<uint32_t>(tiling_->sD) + kd * 2U;
            const uint64_t inputOffset = (((static_cast<uint64_t>(nIdx) * tiling_->inD + inputDepth) * tiling_->inH +
                                           hStart) *
                                          tiling_->inW) *
                                         tiling_->c;
            DataCopyPad(inputLocal[static_cast<uint64_t>(kd) * bankElements], xGm_[inputOffset], inputCopy, inputPad);
        }
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }

    __aicore__ inline void ReduceNdc1hwc0LogicalD3W3Rows(LocalTensor<T> inputLocal, LocalTensor<T> outputLocal,
                                                         uint32_t rows, uint32_t block)
    {
        const uint32_t bankElements = rows * static_cast<uint32_t>(tiling_->inW) * block;
        Max(inputLocal, inputLocal, inputLocal[bankElements], bankElements);
        PipeBarrier<PIPE_V>();
        Max(inputLocal, inputLocal, inputLocal[2U * bankElements], bankElements);
        PipeBarrier<PIPE_V>();
        const uint8_t outputStride = static_cast<uint8_t>(block * sizeof(T) / UB_BLOCK_BYTES);
        const uint8_t inputStride = static_cast<uint8_t>(3U * block * sizeof(T) / UB_BLOCK_BYTES);
        const BinaryRepeatParams firstParams{1U, 1U, 1U, outputStride, inputStride, inputStride};
        const uint8_t repeats = static_cast<uint8_t>(rows * static_cast<uint32_t>(tiling_->outW));
        Max(outputLocal, inputLocal, inputLocal[block], block, repeats, firstParams);
        PipeBarrier<PIPE_V>();
        const BinaryRepeatParams finishParams{1U, 1U, 1U, outputStride, outputStride, inputStride};
        Max(outputLocal, outputLocal, inputLocal[2U * block], block, repeats, finishParams);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ProcessNdc1hwc0LogicalD3W3RowGroups()
    {
        const uint32_t block = static_cast<uint32_t>(Ndc1hwc0Block());
        const uint32_t totalPlanes = static_cast<uint32_t>(tiling_->n * tiling_->outD);
        const uint32_t worker = GetBlockIdx();
        const uint32_t workerDim = ActiveBlockDim();
        uint32_t groupsPerPlane = workerDim / totalPlanes;
        if (groupsPerPlane == 0U) {
            groupsPerPlane = 1U;
        }
        if (groupsPerPlane > static_cast<uint32_t>(tiling_->outH)) {
            groupsPerPlane = static_cast<uint32_t>(tiling_->outH);
        }
        const uint32_t totalGroups = totalPlanes * groupsPerPlane;
        for (uint32_t unit = worker; unit < totalGroups; unit += workerDim) {
            const uint32_t plane = unit / groupsPerPlane;
            const uint32_t group = unit - plane * groupsPerPlane;
            const uint32_t nIdx = plane / static_cast<uint32_t>(tiling_->outD);
            const uint32_t od = plane - nIdx * static_cast<uint32_t>(tiling_->outD);
            const uint32_t baseRows = static_cast<uint32_t>(tiling_->outH) / groupsPerPlane;
            const uint32_t extraRows = static_cast<uint32_t>(tiling_->outH) - baseRows * groupsPerPlane;
            const uint32_t rows = baseRows + (group < extraRows ? 1U : 0U);
            const uint32_t hStart = group * baseRows + (group < extraRows ? group : extraRows);
            LocalTensor<T> inputLocal = xInQue_.AllocTensor<T>();
            LoadNdc1hwc0LogicalD3W3Rows(inputLocal, nIdx, od, hStart, rows, block);
            LocalTensor<T> outputLocal = calcBuf_.Get<T>();
            ReduceNdc1hwc0LogicalD3W3Rows(inputLocal, outputLocal, rows, block);
            const uint64_t outputOffset = (static_cast<uint64_t>(plane) * tiling_->outH + hStart) * tiling_->outW *
                                          block;
            CopyOutVector(outputOffset, outputLocal, rows * static_cast<uint32_t>(tiling_->outW) * block);
            xInQue_.FreeTensor(inputLocal);
        }
        const uint64_t validOut = static_cast<uint64_t>(totalPlanes) * tiling_->outH * tiling_->outW * block;
        if (validOut < tiling_->totalOut) {
            CopyOutZeroRangeByCore(validOut, tiling_->totalOut - validOut);
        }
    }

    __aicore__ inline bool CanUseNdc1hwc0LogicalD2H3W2DepthGroupPath() const
    {
        if constexpr (!AscendC::Std::is_same<T, half>::value) {
            return false;
        }
        const uint64_t block = Ndc1hwc0Block();
        const uint64_t validC1 = Ndc1hwc0ValidC1(block);
        if (tiling_->inputLayout == INPUT_LAYOUT_NDC1HWC0_VALUE ||
            tiling_->outputLayout != OUTPUT_LAYOUT_NDC1HWC0_VALUE || block != 16U || validC1 != 1U ||
            tiling_->outputC1 != 1 || tiling_->c <= 0 || static_cast<uint64_t>(tiling_->c) > block ||
            !MatchesPoolSpec(2, 3, 2, 1, 2, 1, 2, 2, 1, 1, 2, 0) || tiling_->outD <= 0 || tiling_->outH <= 2 ||
            tiling_->outW != tiling_->inW) {
            return false;
        }
        const uint64_t workerDim = ActiveBlockDim();
        if (tiling_->n <= 0 || workerDim == 0U) {
            return false;
        }
        uint64_t groupsPerN = workerDim / static_cast<uint64_t>(tiling_->n);
        if (groupsPerN == 0U) {
            groupsPerN = 1U;
        }
        if (groupsPerN > static_cast<uint64_t>(tiling_->outD)) {
            groupsPerN = static_cast<uint64_t>(tiling_->outD);
        }
        const uint64_t maxOutputDepths = (static_cast<uint64_t>(tiling_->outD) + groupsPerN - 1U) / groupsPerN;
        const uint64_t inputPlane = static_cast<uint64_t>(tiling_->inH) * tiling_->inW;
        const uint64_t alignedInputPlane = AlignToVector(static_cast<uint32_t>(inputPlane));
        const uint64_t blockedInputPlane = inputPlane * block;
        const uint64_t outputPlane = static_cast<uint64_t>(tiling_->outH) * tiling_->outW * block;
        const uint64_t ncdhwNeed = static_cast<uint64_t>(tiling_->c) * alignedInputPlane + blockedInputPlane;
        return inputPlane > 0U && alignedInputPlane >= inputPlane && alignedInputPlane - inputPlane <= 255U &&
               ((tiling_->dataFormat == FORMAT_NCDHW_VALUE && ncdhwNeed <= INPUT_TILE_NUM) ||
                (tiling_->dataFormat == FORMAT_NDHWC_VALUE && blockedInputPlane <= INPUT_TILE_NUM)) &&
               blockedInputPlane <= OUTPUT_TILE_NUM && outputPlane <= OUTPUT_TILE_NUM &&
               maxOutputDepths * outputPlane <= OUTPUT_TILE_NUM && tiling_->outH - 2 <= 255;
    }

    __aicore__ inline LocalTensor<T> LoadNdc1hwc0LogicalD2H3W2Plane(LocalTensor<T> inputLocal, LocalTensor<T> zeroLocal,
                                                                    uint32_t nIdx, uint32_t id, uint32_t block,
                                                                    uint32_t inputPlane, uint32_t alignedInputPlane)
    {
        if (tiling_->dataFormat == FORMAT_NDHWC_VALUE) {
            const DataCopyExtParams inputCopy{static_cast<uint16_t>(inputPlane),
                                              static_cast<uint32_t>(tiling_->c * sizeof(T)), 0U, 0U, 0U};
            const DataCopyPadExtParams<T> inputPad{
                true, 0U, static_cast<uint8_t>(block - static_cast<uint32_t>(tiling_->c)), ZeroValue()};
            const uint64_t inputOffset = (static_cast<uint64_t>(nIdx) * tiling_->inD + id) * inputPlane * tiling_->c;
            DataCopyPad(inputLocal, xGm_[inputOffset], inputCopy, inputPad);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            return inputLocal;
        }
        const uint64_t nBase = static_cast<uint64_t>(nIdx) * tiling_->c * tiling_->inD * inputPlane;
        const uint64_t inputOffset = nBase + static_cast<uint64_t>(id) * inputPlane;
        const DataCopyExtParams inputCopy{static_cast<uint16_t>(tiling_->c),
                                          static_cast<uint32_t>(inputPlane * sizeof(T)),
                                          static_cast<uint32_t>((tiling_->inD - 1) * inputPlane * sizeof(T)), 0U, 0U};
        const DataCopyPadExtParams<T> inputPad{true, 0U, static_cast<uint8_t>(alignedInputPlane - inputPlane),
                                               ZeroValue()};
        DataCopyPad(inputLocal, xGm_[inputOffset], inputCopy, inputPad);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Duplicate(zeroLocal, ZeroValue(), alignedInputPlane);
        PipeBarrier<PIPE_V>();
        LocalTensor<T> blockedLocal = inputLocal[static_cast<uint64_t>(tiling_->c) * alignedInputPlane];
        TransposeNdc1hwc0C0PlaneBlock(blockedLocal, inputLocal, zeroLocal, alignedInputPlane,
                                      static_cast<uint32_t>(tiling_->c), block);
        return blockedLocal;
    }

    __aicore__ inline void ReduceNdc1hwc0LogicalD2H3W2Plane(LocalTensor<T> inputPlaneLocal, LocalTensor<T> widthLocal,
                                                            LocalTensor<T> outputPlaneLocal, uint32_t block)
    {
        const uint32_t inputRow = static_cast<uint32_t>(tiling_->inW) * block;
        const uint32_t outputRow = static_cast<uint32_t>(tiling_->outW) * block;
        const uint8_t rowStride = static_cast<uint8_t>(inputRow * sizeof(T) / UB_BLOCK_BYTES);
        const BinaryRepeatParams widthParams{1U, 1U, 1U, rowStride, rowStride, rowStride};
        Max(widthLocal, inputPlaneLocal, inputPlaneLocal[block], 2U * block, static_cast<uint8_t>(tiling_->inH),
            widthParams);
        Max(widthLocal[2U * block], inputPlaneLocal[2U * block], inputPlaneLocal[2U * block], block,
            static_cast<uint8_t>(tiling_->inH), widthParams);
        PipeBarrier<PIPE_V>();
        Max(outputPlaneLocal, widthLocal, widthLocal[2U * outputRow], outputRow);
        const uint8_t middleRows = static_cast<uint8_t>(tiling_->outH - 2);
        const uint8_t outputStride = static_cast<uint8_t>(outputRow * sizeof(T) / UB_BLOCK_BYTES);
        const BinaryRepeatParams heightParams{
            1U, 1U, 1U, outputStride, static_cast<uint8_t>(2U * outputStride), static_cast<uint8_t>(2U * outputStride)};
        Max(outputPlaneLocal[outputRow], widthLocal, widthLocal[2U * outputRow], outputRow, middleRows, heightParams);
        PipeBarrier<PIPE_V>();
        const BinaryRepeatParams heightFinishParams{
            1U, 1U, 1U, outputStride, outputStride, static_cast<uint8_t>(2U * outputStride)};
        Max(outputPlaneLocal[outputRow], outputPlaneLocal[outputRow], widthLocal[4U * outputRow], outputRow, middleRows,
            heightFinishParams);
        Max(outputPlaneLocal[static_cast<uint64_t>(tiling_->outH - 1) * outputRow],
            widthLocal[static_cast<uint64_t>(tiling_->inH - 3) * outputRow],
            widthLocal[static_cast<uint64_t>(tiling_->inH - 1) * outputRow], outputRow);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void AccumulateNdc1hwc0LogicalD2H3W2Depth(uint32_t id, uint32_t odStart, uint32_t odCount,
                                                                LocalTensor<T> planeLocal, LocalTensor<T> outputLocal,
                                                                uint32_t outputPlane)
    {
        for (uint32_t kd = 0U; kd < static_cast<uint32_t>(tiling_->kD); ++kd) {
            const int64_t numerator = static_cast<int64_t>(id) + tiling_->padFront -
                                      static_cast<int64_t>(kd) * tiling_->dilationD;
            if (numerator < 0 || numerator % tiling_->sD != 0) {
                continue;
            }
            const uint32_t od = static_cast<uint32_t>(numerator / tiling_->sD);
            if (od < odStart || od >= odStart + odCount) {
                continue;
            }
            Max(outputLocal[static_cast<uint64_t>(od - odStart) * outputPlane],
                outputLocal[static_cast<uint64_t>(od - odStart) * outputPlane], planeLocal, outputPlane);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ProcessNdc1hwc0LogicalD2H3W2DepthGroups()
    {
        const uint32_t block = static_cast<uint32_t>(Ndc1hwc0Block());
        const uint32_t inputPlane = static_cast<uint32_t>(tiling_->inH * tiling_->inW);
        const uint32_t alignedInputPlane = AlignToVector(inputPlane);
        const uint32_t outputPlane = static_cast<uint32_t>(tiling_->outH * tiling_->outW) * block;
        const uint32_t worker = GetBlockIdx();
        const uint32_t workerDim = ActiveBlockDim();
        uint32_t groupsPerN = workerDim / static_cast<uint32_t>(tiling_->n);
        if (groupsPerN == 0U) {
            groupsPerN = 1U;
        }
        if (groupsPerN > static_cast<uint32_t>(tiling_->outD)) {
            groupsPerN = static_cast<uint32_t>(tiling_->outD);
        }
        const uint32_t totalGroups = static_cast<uint32_t>(tiling_->n) * groupsPerN;
        for (uint32_t unit = worker; unit < totalGroups; unit += workerDim) {
            const uint32_t nIdx = unit / groupsPerN;
            const uint32_t group = unit - nIdx * groupsPerN;
            const uint32_t baseDepths = static_cast<uint32_t>(tiling_->outD) / groupsPerN;
            const uint32_t extraDepths = static_cast<uint32_t>(tiling_->outD) - baseDepths * groupsPerN;
            const uint32_t odCount = baseDepths + (group < extraDepths ? 1U : 0U);
            const uint32_t odStart = group * baseDepths + (group < extraDepths ? group : extraDepths);
            int64_t idStart = static_cast<int64_t>(odStart) * tiling_->sD - tiling_->padFront;
            int64_t idEnd = static_cast<int64_t>(odStart + odCount - 1U) * tiling_->sD +
                            (tiling_->kD - 1) * tiling_->dilationD - tiling_->padFront;
            if (idStart < 0) {
                idStart = 0;
            }
            if (idEnd >= tiling_->inD) {
                idEnd = tiling_->inD - 1;
            }
            LocalTensor<T> outputLocal = maskBuf_.Get<T>();
            Duplicate(outputLocal, NegInfValue(), odCount * outputPlane);
            PipeBarrier<PIPE_V>();
            for (int64_t id = idStart; id <= idEnd; ++id) {
                LocalTensor<T> inputLocal = xInQue_.AllocTensor<T>();
                LocalTensor<T> planeLocal = calcBuf_.Get<T>();
                LocalTensor<T> blockedLocal = LoadNdc1hwc0LogicalD2H3W2Plane(
                    inputLocal, planeLocal, nIdx, static_cast<uint32_t>(id), block, inputPlane, alignedInputPlane);
                LocalTensor<T> widthLocal = tmpBuf_.Get<T>();
                ReduceNdc1hwc0LogicalD2H3W2Plane(blockedLocal, widthLocal, planeLocal, block);
                AccumulateNdc1hwc0LogicalD2H3W2Depth(static_cast<uint32_t>(id), odStart, odCount, planeLocal,
                                                     outputLocal, outputPlane);
                xInQue_.FreeTensor(inputLocal);
            }
            const uint64_t outputOffset = (static_cast<uint64_t>(nIdx) * tiling_->outD + odStart) * outputPlane;
            CopyOutVector(outputOffset, outputLocal, odCount * outputPlane);
        }
        const uint64_t validOut = static_cast<uint64_t>(tiling_->n) * tiling_->outD * outputPlane;
        if (validOut < tiling_->totalOut) {
            CopyOutZeroRangeByCore(validOut, tiling_->totalOut - validOut);
        }
    }

    __aicore__ inline void ProcessNdc1hwc0HOnlyStride3GroupStep(uint64_t& cur, uint64_t outEnd, uint64_t block,
                                                                uint64_t validC1, uint32_t rowElements,
                                                                uint32_t maxTileRows, uint32_t outW,
                                                                LocalTensor<T> outLocal)
    {
        Ndc1hwc0GroupStepContext context{};
        if (!PrepareNdc1hwc0GroupStep(cur, outEnd, block, validC1, rowElements, true, context)) {
            return;
        }
        uint32_t rows = context.rows;
        if (rows > maxTileRows) {
            rows = maxTileRows;
        }
        if (tiling_->dataFormat == FORMAT_NCDHW_VALUE) {
            FillNdc1hwc0NcdhwHOnlyStride3Rows(context.nIdx, context.od, static_cast<uint32_t>(context.oh), rows,
                                              outLocal, static_cast<uint32_t>(block), outW);
        } else {
            FillNdc1hwc0NdhwcHOnlyStride3Rows(context.nIdx, context.od, static_cast<uint32_t>(context.oh), rows,
                                              outLocal, static_cast<uint32_t>(block), outW);
        }
        CopyOutVector(cur, outLocal, rows * rowElements);
        cur += static_cast<uint64_t>(rows) * rowElements;
    }

    __aicore__ inline void ProcessNdc1hwc0HOnlyStride3Group()
    {
        uint64_t block = 0U, validC1 = 0U, validOut = 0U;
        uint64_t outOffset = 0U, outCount = 0U, outEnd = 0U;
        uint32_t outW = 0U, rowElements = 0U;
        if (!InitNdc1hwc0GroupRange(block, validC1, validOut, outW, rowElements, outOffset, outCount, outEnd)) {
            return;
        }
        LocalTensor<T> outLocal = maskBuf_.Get<T>();
        const uint32_t maxTileRows = OUTPUT_TILE_NUM / rowElements == 0U ? 1U : OUTPUT_TILE_NUM / rowElements;
        uint64_t cur = outOffset;
        while (cur < outEnd) {
            ProcessNdc1hwc0HOnlyStride3GroupStep(cur, outEnd, block, validC1, rowElements, maxTileRows, outW, outLocal);
        }
        if (validOut < tiling_->totalOut) {
            CopyOutZeroRangeByCore(validOut, tiling_->totalOut - validOut);
        }
    }

    __aicore__ inline void ReduceNdc1hwc0NcdhwHOnlyRepeatChannel(LocalTensor<T> accPlaneLocal, LocalTensor<T> xLocal,
                                                                 uint32_t tileRows, uint32_t c0, uint32_t alignedW,
                                                                 uint32_t rowStride, uint32_t dstRepStride,
                                                                 uint32_t srcRepStride)
    {
        BinaryRepeatParams params{1,
                                  1,
                                  1,
                                  static_cast<uint8_t>(dstRepStride),
                                  static_cast<uint8_t>(dstRepStride),
                                  static_cast<uint8_t>(srcRepStride)};
        for (int64_t kh = 0; kh < 3; ++kh) {
            uint32_t beginRow = 0U;
            while (beginRow < tileRows && static_cast<int64_t>(beginRow) * tiling_->sH + kh - tiling_->padTop < 0) {
                ++beginRow;
            }
            uint32_t endRow = tileRows;
            while (endRow > beginRow &&
                   static_cast<int64_t>(endRow - 1U) * tiling_->sH + kh - tiling_->padTop >= tiling_->inH) {
                --endRow;
            }
            if (beginRow >= endRow) {
                continue;
            }
            const uint32_t ih = static_cast<uint32_t>(static_cast<int64_t>(beginRow) * tiling_->sH + kh -
                                                      tiling_->padTop);
            const uint64_t dstBase = static_cast<uint64_t>(beginRow) * rowStride + static_cast<uint64_t>(c0) * alignedW;
            Max(accPlaneLocal[dstBase], accPlaneLocal[dstBase], xLocal[static_cast<uint64_t>(ih) * alignedW], alignedW,
                static_cast<uint8_t>(endRow - beginRow), params);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ReduceNdc1hwc0NcdhwHOnlyScalarChannel(LocalTensor<T> accPlaneLocal, LocalTensor<T> xLocal,
                                                                 uint32_t tileRows, uint32_t c0, uint32_t alignedW,
                                                                 uint32_t rowStride, uint32_t outW)
    {
        for (uint32_t oh = 0; oh < tileRows; ++oh) {
            LocalTensor<T> accRow = accPlaneLocal[static_cast<uint64_t>(oh) * rowStride];
            for (int64_t kh = 0; kh < 3; ++kh) {
                const int64_t ih = static_cast<int64_t>(oh) * tiling_->sH + kh - tiling_->padTop;
                if (IsOutOfRange(ih, tiling_->inH)) {
                    continue;
                }
                const uint64_t dstBase = static_cast<uint64_t>(c0) * alignedW;
                Max(accRow[dstBase], accRow[dstBase], xLocal[static_cast<uint64_t>(ih) * alignedW], outW);
                PipeBarrier<PIPE_V>();
            }
        }
    }

    __aicore__ inline void FillNdc1hwc0NcdhwHOnlyFullPlane(int64_t nIdx, int64_t od, uint32_t tileRows,
                                                           LocalTensor<T> outLocal, uint32_t cCount, uint32_t block,
                                                           uint32_t outW, uint32_t alignedW, uint32_t compactCount,
                                                           uint32_t rowStride, bool allowRepeat)
    {
        const uint32_t inH = static_cast<uint32_t>(tiling_->inH);
        LocalTensor<T> accPlaneLocal = calcBuf_.Get<T>();
        LocalTensor<T> tmpLocal = tmpBuf_.Get<T>();
        LocalTensor<uint32_t> offsetLocal = tmpLocal.template ReinterpretCast<uint32_t>();
        Duplicate(accPlaneLocal, NegInfValue(), tileRows * rowStride);
        PipeBarrier<PIPE_V>();
        for (uint32_t c0 = 0; c0 < cCount; ++c0) {
            CopyInVectorWBlocksPadStrideValue(InputOffset(nIdx, od, 0, 0, static_cast<int64_t>(c0)), inH, outW,
                                              alignedW, 0U, NegInfValue());
            LocalTensor<T> xLocal = xInQue_.DeQue<T>();
            const uint32_t dstRepStride = rowStride * sizeof(T) / UB_BLOCK_BYTES;
            const uint32_t srcRepStride = static_cast<uint32_t>(tiling_->sH) * alignedW * sizeof(T) / UB_BLOCK_BYTES;
            const bool useRepeat = allowRepeat && dstRepStride > 0U && dstRepStride <= 255U && srcRepStride > 0U &&
                                   srcRepStride <= 255U;
            if (useRepeat) {
                ReduceNdc1hwc0NcdhwHOnlyRepeatChannel(accPlaneLocal, xLocal, tileRows, c0, alignedW, rowStride,
                                                      dstRepStride, srcRepStride);
            } else {
                ReduceNdc1hwc0NcdhwHOnlyScalarChannel(accPlaneLocal, xLocal, tileRows, c0, alignedW, rowStride, outW);
            }
            xInQue_.FreeTensor(xLocal);
        }
        if (!ScatterNdc1hwc0CompactRowsReuseRowOffset(outLocal, accPlaneLocal, offsetLocal, tileRows, rowStride, cCount,
                                                      block, outW, 1U, alignedW, compactCount)) {
            ScatterNdc1hwc0CompactRowsChecked(outLocal, accPlaneLocal, offsetLocal, tileRows, rowStride, cCount, block,
                                              outW, 1U, alignedW, compactCount);
        }
    }

    __aicore__ inline void FillNdc1hwc0NcdhwHOnlyRow(int64_t nIdx, int64_t od, uint32_t oh, uint32_t rowIdx,
                                                     LocalTensor<T> outLocal, uint32_t cCount, uint32_t block,
                                                     uint32_t outW, uint32_t alignedW, uint32_t compactCount,
                                                     uint32_t rowStride, uint32_t rowElements,
                                                     uint32_t srcStrideElements)
    {
        LocalTensor<T> accLocal = calcBuf_.Get<T>();
        Duplicate(accLocal, NegInfValue(), compactCount);
        PipeBarrier<PIPE_V>();
        for (int64_t kh = 0; kh < 3; ++kh) {
            const int64_t ih = static_cast<int64_t>(oh) * tiling_->sH + kh - tiling_->padTop;
            if (IsOutOfRange(ih, tiling_->inH)) {
                continue;
            }
            CopyInVectorWBlocksPadStrideValue(InputOffset(nIdx, od, ih, 0, 0), cCount, outW, alignedW,
                                              srcStrideElements, NegInfValue());
            LocalTensor<T> xLocal = xInQue_.DeQue<T>();
            Max(accLocal, accLocal, xLocal, compactCount);
            PipeBarrier<PIPE_V>();
            xInQue_.FreeTensor(xLocal);
        }
        LocalTensor<T> tmpLocal = tmpBuf_.Get<T>();
        LocalTensor<uint32_t> offsetLocal = tmpLocal.template ReinterpretCast<uint32_t>();
        LocalTensor<T> rowLocal = outLocal[static_cast<uint64_t>(rowIdx) * rowElements];
        if (!ScatterNdc1hwc0CompactRowsReuseRowOffset(rowLocal, accLocal, offsetLocal, 1U, rowStride, cCount, block,
                                                      outW, 1U, alignedW, compactCount)) {
            ScatterNdc1hwc0CompactRowsChecked(rowLocal, accLocal, offsetLocal, 1U, rowStride, cCount, block, outW, 1U,
                                              alignedW, compactCount);
        }
    }

    __aicore__ inline void FillNdc1hwc0NcdhwHOnlyStride3Rows(int64_t nIdx, int64_t od, uint32_t ohStart,
                                                             uint32_t tileRows, LocalTensor<T> outLocal, uint32_t block,
                                                             uint32_t outW)
    {
        const uint32_t cCount = static_cast<uint32_t>(tiling_->c);
        const uint32_t blockElements = UB_BLOCK_BYTES / sizeof(T);
        const uint32_t alignedW = (outW + blockElements - 1U) / blockElements * blockElements;
        const uint32_t compactCount = cCount * alignedW;
        const uint32_t rowStride = (compactCount + 1U + blockElements - 1U) / blockElements * blockElements;
        if (ohStart == 0U && tileRows == static_cast<uint32_t>(tiling_->outH)) {
            FillNdc1hwc0NcdhwHOnlyFullPlane(nIdx, od, tileRows, outLocal, cCount, block, outW, alignedW, compactCount,
                                            rowStride, true);
            return;
        }
        const uint32_t rowElements = outW * block;
        const uint64_t channelStride = static_cast<uint64_t>(tiling_->inD) * static_cast<uint64_t>(tiling_->inH) *
                                       static_cast<uint64_t>(tiling_->inW);
        const uint32_t srcStrideElements = static_cast<uint32_t>(channelStride - static_cast<uint64_t>(outW));
        for (uint32_t rowIdx = 0; rowIdx < tileRows; ++rowIdx) {
            FillNdc1hwc0NcdhwHOnlyRow(nIdx, od, ohStart + rowIdx, rowIdx, outLocal, cCount, block, outW, alignedW,
                                      compactCount, rowStride, rowElements, srcStrideElements);
        }
    }

    __aicore__ inline void FillNdc1hwc0NdhwcHOnlyFullPlane(int64_t nIdx, int64_t od, uint32_t tileRows,
                                                           LocalTensor<T> outLocal, uint32_t cCount, uint32_t block,
                                                           uint32_t outW, uint32_t compactCount,
                                                           uint32_t alignedCompactCount, uint32_t rowStride)
    {
        LocalTensor<T> accPlaneLocal = calcBuf_.Get<T>();
        LocalTensor<T> tmpLocal = tmpBuf_.Get<T>();
        LocalTensor<uint32_t> offsetLocal = tmpLocal.template ReinterpretCast<uint32_t>();
        CopyInVectorWBlocksPadStrideValue(InputOffset(nIdx, od, 0, 0, 0), static_cast<uint32_t>(tiling_->inH),
                                          compactCount, alignedCompactCount, 0U, NegInfValue());
        LocalTensor<T> xLocal = xInQue_.DeQue<T>();
        Duplicate(accPlaneLocal, NegInfValue(), tileRows * rowStride);
        PipeBarrier<PIPE_V>();
        for (uint32_t oh = 0; oh < tileRows; ++oh) {
            LocalTensor<T> accRow = accPlaneLocal[static_cast<uint64_t>(oh) * rowStride];
            for (int64_t kh = 0; kh < 3; ++kh) {
                const int64_t ih = static_cast<int64_t>(oh) * tiling_->sH + kh - tiling_->padTop;
                if (IsOutOfRange(ih, tiling_->inH)) {
                    continue;
                }
                Max(accRow, accRow, xLocal[static_cast<uint64_t>(ih) * alignedCompactCount], alignedCompactCount);
                PipeBarrier<PIPE_V>();
            }
        }
        xInQue_.FreeTensor(xLocal);
        if (!ScatterNdc1hwc0CompactRowsReuseRowOffset(outLocal, accPlaneLocal, offsetLocal, tileRows, rowStride, cCount,
                                                      block, outW, cCount, 1U, alignedCompactCount)) {
            ScatterNdc1hwc0CompactRowsChecked(outLocal, accPlaneLocal, offsetLocal, tileRows, rowStride, cCount, block,
                                              outW, cCount, 1U, alignedCompactCount);
        }
    }

    __aicore__ inline void FillNdc1hwc0NdhwcHOnlyRow(int64_t nIdx, int64_t od, uint32_t oh, uint32_t rowIdx,
                                                     LocalTensor<T> outLocal, uint32_t cCount, uint32_t block,
                                                     uint32_t outW, uint32_t compactCount, uint32_t alignedCompactCount,
                                                     uint32_t rowStride, uint32_t rowElements)
    {
        LocalTensor<T> accLocal = calcBuf_.Get<T>();
        Duplicate(accLocal, NegInfValue(), alignedCompactCount);
        PipeBarrier<PIPE_V>();
        for (int64_t kh = 0; kh < 3; ++kh) {
            const int64_t ih = static_cast<int64_t>(oh) * tiling_->sH + kh - tiling_->padTop;
            if (IsOutOfRange(ih, tiling_->inH)) {
                continue;
            }
            CopyInVectorPadValue(InputOffset(nIdx, od, ih, 0, 0), compactCount, alignedCompactCount, NegInfValue());
            LocalTensor<T> xLocal = xInQue_.DeQue<T>();
            Max(accLocal, accLocal, xLocal, alignedCompactCount);
            PipeBarrier<PIPE_V>();
            xInQue_.FreeTensor(xLocal);
        }
        LocalTensor<T> rowLocal = outLocal[static_cast<uint64_t>(rowIdx) * rowElements];
        if constexpr (AscendC::Std::is_same<T, float>::value) {
            if (ScatterNdc1hwc0CompactTileActiveChannels(rowLocal, accLocal, 1U, alignedCompactCount, cCount, block,
                                                         outW, cCount, 1U)) {
                return;
            }
        }
        LocalTensor<T> tmpLocal = tmpBuf_.Get<T>();
        LocalTensor<uint32_t> offsetLocal = tmpLocal.template ReinterpretCast<uint32_t>();
        ScatterNdc1hwc0CompactRowsChecked(rowLocal, accLocal, offsetLocal, 1U, rowStride, cCount, block, outW, cCount,
                                          1U, alignedCompactCount);
    }

    __aicore__ inline void FillNdc1hwc0NdhwcHOnlyStride3Rows(int64_t nIdx, int64_t od, uint32_t ohStart,
                                                             uint32_t tileRows, LocalTensor<T> outLocal, uint32_t block,
                                                             uint32_t outW)
    {
        const uint32_t cCount = static_cast<uint32_t>(tiling_->c);
        const uint32_t compactCount = outW * cCount;
        const uint32_t alignedCompactCount = AlignToVector(compactCount);
        const uint32_t blockElements = UB_BLOCK_BYTES / sizeof(T);
        const uint32_t rowStride = (alignedCompactCount + 1U + blockElements - 1U) / blockElements * blockElements;
        if (ohStart == 0U && tileRows == static_cast<uint32_t>(tiling_->outH)) {
            FillNdc1hwc0NdhwcHOnlyFullPlane(nIdx, od, tileRows, outLocal, cCount, block, outW, compactCount,
                                            alignedCompactCount, rowStride);
            return;
        }
        const uint32_t rowElements = outW * block;
        for (uint32_t rowIdx = 0; rowIdx < tileRows; ++rowIdx) {
            FillNdc1hwc0NdhwcHOnlyRow(nIdx, od, ohStart + rowIdx, rowIdx, outLocal, cCount, block, outW, compactCount,
                                      alignedCompactCount, rowStride, rowElements);
        }
    }

    __aicore__ inline bool HasNdc1hwc0NcdhwD3H3Shape() const
    {
        return tiling_->dataFormat == FORMAT_NCDHW_VALUE && tiling_->outW > 0 && tiling_->outH > 0 &&
               tiling_->inH > 0 && tiling_->c > 0;
    }

    __aicore__ inline bool HasNdc1hwc0NcdhwD3H3Layout(uint64_t block) const
    {
        return block > 0U && tiling_->outputC1 == 1 && static_cast<uint64_t>(tiling_->c) <= block &&
               tiling_->outW == tiling_->inW;
    }

    __aicore__ inline bool HasNdc1hwc0NcdhwD3H3Capacity(uint64_t block) const
    {
        const uint32_t cCount = static_cast<uint32_t>(tiling_->c);
        const uint32_t outW = static_cast<uint32_t>(tiling_->outW);
        const uint32_t blockAlignedW = UB_BLOCK_BYTES / sizeof(T);
        const uint32_t alignedW = (outW + blockAlignedW - 1U) / blockAlignedW * blockAlignedW;
        const uint32_t compactCount = cCount * alignedW;
        const uint32_t rowStride = (compactCount + 1U + blockAlignedW - 1U) / blockAlignedW * blockAlignedW;
        const uint32_t dmaxCount = compactCount * static_cast<uint32_t>(tiling_->inH);
        const uint32_t rowElements = static_cast<uint32_t>(static_cast<uint64_t>(outW) * block);
        const uint32_t scatterOffset = AlignToVector(rowStride);
        const uint32_t scratchNeed = scatterOffset + Ndc1hwc0GatherTempOffset(rowElements);
        const uint64_t slabCount = static_cast<uint64_t>(tiling_->inH) * alignedW * 3U;
        const bool fullDWindow = tiling_->outD > 0 && (tiling_->outD - 1) * 3 + 2 < tiling_->inD;
        return outW > 0U && rowElements > 0U && rowElements <= OUTPUT_TILE_NUM &&
               rowStride <= NDC1HWC0_D3H3_OUTPUT_TILE_NUM && dmaxCount <= NDC1HWC0_D3H3_OUTPUT_TILE_NUM &&
               slabCount <= INPUT_TILE_NUM && scratchNeed <= NDC1HWC0_D3H3_OUTPUT_TILE_NUM && fullDWindow &&
               CanUseNdc1hwc0NcdhwRowGather(cCount, static_cast<uint32_t>(block), outW, alignedW);
    }

    __aicore__ inline bool CanUseNdc1hwc0NcdhwD3H3Dil2GroupPath() const
    {
        const uint64_t block = Ndc1hwc0Block();
        return HasNdc1hwc0NcdhwD3H3Shape() && HasNdc1hwc0NcdhwD3H3Layout(block) && HasNdc1hwc0D3H3Dil2PoolSpec() &&
               HasNdc1hwc0NcdhwD3H3Capacity(block);
    }

    __aicore__ inline void ProcessNdc1hwc0NcdhwD3H3Dil2GroupStep(uint64_t& cur, uint64_t outEnd, uint64_t block,
                                                                 uint64_t validC1, uint32_t rowElements,
                                                                 uint32_t cCount, uint32_t outW)
    {
        Ndc1hwc0GroupStepContext context{};
        if (!PrepareNdc1hwc0GroupStep(cur, outEnd, block, validC1, rowElements, true, context)) {
            return;
        }
        ProcessNdc1hwc0NcdhwD3H3Dil2GroupTile(cur, context.nIdx, context.od, static_cast<uint32_t>(context.oh),
                                              context.rows, cCount, static_cast<uint32_t>(block), outW);
        cur += static_cast<uint64_t>(context.rows) * rowElements;
    }

    __aicore__ inline void ProcessNdc1hwc0NcdhwD3H3Dil2Group()
    {
        uint64_t block = 0U, validC1 = 0U, validOut = 0U;
        uint64_t outOffset = 0U, outCount = 0U, outEnd = 0U;
        uint32_t outW = 0U, rowElements = 0U;
        if (!InitNdc1hwc0GroupRange(block, validC1, validOut, outW, rowElements, outOffset, outCount, outEnd)) {
            return;
        }
        const uint32_t cCount = static_cast<uint32_t>(tiling_->c);
        uint64_t cur = outOffset;
        while (cur < outEnd) {
            ProcessNdc1hwc0NcdhwD3H3Dil2GroupStep(cur, outEnd, block, validC1, rowElements, cCount, outW);
        }
        if (validOut < tiling_->totalOut) {
            CopyOutZeroRangeByCore(validOut, tiling_->totalOut - validOut);
        }
    }

    __aicore__ inline void ReduceNdc1hwc0NcdhwD3H3Depth(LocalTensor<T> dmaxLocal, int64_t nIdx, int64_t od,
                                                        uint32_t cCount, uint32_t outW, uint32_t alignedW)
    {
        const uint32_t planeCount = static_cast<uint32_t>(tiling_->inH) * alignedW;
        for (uint32_t c0 = 0; c0 < cCount; ++c0) {
            const int64_t cIdx = static_cast<int64_t>(c0);
            CopyInVectorWBlocksPadStrideValue(InputOffset(nIdx, od * 3, 0, 0, cIdx),
                                              static_cast<uint32_t>(tiling_->inH) * 3U, outW, alignedW, 0U,
                                              NegInfValue());
            LocalTensor<T> xLocal = xInQue_.DeQue<T>();
            LocalTensor<T> channelDmax = dmaxLocal[static_cast<uint64_t>(c0) * planeCount];
            Max(channelDmax, xLocal, xLocal[planeCount], planeCount);
            PipeBarrier<PIPE_V>();
            Max(channelDmax, channelDmax, xLocal[static_cast<uint64_t>(planeCount) * 2U], planeCount);
            PipeBarrier<PIPE_V>();
            xInQue_.FreeTensor(xLocal);
        }
    }

    __aicore__ inline uint32_t Ndc1hwc0NcdhwD3H3MaxTileRows(uint32_t rows, uint32_t rowStride,
                                                            uint32_t rowElements) const
    {
        uint32_t maxTileRows = Ndc1hwc0MaxCompactTileRowsLimit(rows, rowStride, rowElements,
                                                               NDC1HWC0_D3H3_OUTPUT_TILE_NUM);
        if (maxTileRows == 0U) {
            maxTileRows = 1U;
        }
        const uint32_t rowOffsetNeed = Ndc1hwc0GatherTempOffset(rowElements);
        while (maxTileRows > 1U &&
               (maxTileRows * rowStride > NDC1HWC0_D3H3_OUTPUT_TILE_NUM ||
                AlignToVector(maxTileRows * rowStride) + rowOffsetNeed > NDC1HWC0_D3H3_OUTPUT_TILE_NUM)) {
            --maxTileRows;
        }
        return maxTileRows;
    }

    __aicore__ inline void StoreNdc1hwc0NcdhwD3H3Tiles(uint64_t outputOffset, uint32_t ohStart, uint32_t rows,
                                                       uint32_t cCount, uint32_t block, uint32_t outW,
                                                       uint32_t alignedW, uint32_t compactCount, uint32_t rowStride,
                                                       uint32_t rowElements, LocalTensor<T> dmaxLocal,
                                                       LocalTensor<T> compactLocal, LocalTensor<T> outLocal)
    {
        const uint32_t maxTileRows = Ndc1hwc0NcdhwD3H3MaxTileRows(rows, rowStride, rowElements);
        const uint32_t zeroIndex = compactCount;
        const uint32_t offsetBase = AlignToVector(maxTileRows * rowStride);
        LocalTensor<uint32_t> scatterOffsetLocal = compactLocal[offsetBase].template ReinterpretCast<uint32_t>();
        uint32_t doneRows = 0U;
        while (doneRows < rows) {
            uint32_t tileRows = rows - doneRows;
            if (tileRows > maxTileRows) {
                tileRows = maxTileRows;
            }
            MaxNdc1hwc0NcdhwD3H3CompactRows(compactLocal, dmaxLocal, ohStart + doneRows, tileRows, cCount, alignedW,
                                            rowStride);
            if (!ScatterNdc1hwc0CompactRowsReuseRowOffset(outLocal, compactLocal, scatterOffsetLocal, tileRows,
                                                          rowStride, cCount, block, outW, 1U, alignedW, zeroIndex)) {
                ScatterNdc1hwc0CompactRowsChecked(outLocal, compactLocal, scatterOffsetLocal, tileRows, rowStride,
                                                  cCount, block, outW, 1U, alignedW, zeroIndex);
            }
            CopyOutVector(outputOffset + static_cast<uint64_t>(doneRows) * rowElements, outLocal,
                          tileRows * rowElements);
            doneRows += tileRows;
        }
    }

    __aicore__ inline void ProcessNdc1hwc0NcdhwD3H3Dil2GroupTile(uint64_t outputOffset, int64_t nIdx, int64_t od,
                                                                 uint32_t ohStart, uint32_t rows, uint32_t cCount,
                                                                 uint32_t block, uint32_t outW)
    {
        const uint32_t blockAlignedW = UB_BLOCK_BYTES / sizeof(T);
        const uint32_t alignedW = (outW + blockAlignedW - 1U) / blockAlignedW * blockAlignedW;
        const uint32_t compactCount = cCount * alignedW;
        const uint32_t rowStride = (compactCount + 1U + blockAlignedW - 1U) / blockAlignedW * blockAlignedW;
        const uint32_t rowElements = outW * block;
        LocalTensor<T> dmaxLocal = calcBuf_.Get<T>();
        LocalTensor<T> compactLocal = tmpBuf_.Get<T>();
        LocalTensor<T> outLocal = maskBuf_.Get<T>();
        ReduceNdc1hwc0NcdhwD3H3Depth(dmaxLocal, nIdx, od, cCount, outW, alignedW);
        StoreNdc1hwc0NcdhwD3H3Tiles(outputOffset, ohStart, rows, cCount, block, outW, alignedW, compactCount, rowStride,
                                    rowElements, dmaxLocal, compactLocal, outLocal);
    }

    __aicore__ inline bool CanUseNdc1hwc0NcdhwStride2GroupPath() const
    {
        if (tiling_->dataFormat != FORMAT_NCDHW_VALUE || tiling_->outW <= 0 || tiling_->outH <= 0 || tiling_->c <= 0) {
            return false;
        }
        if (!MatchesPoolSpec(2, 2, 2, 2, 2, 2, 1, 1, 1, 0, 0, 0)) {
            return false;
        }
        const uint64_t block = Ndc1hwc0Block();
        const uint32_t outW = static_cast<uint32_t>(tiling_->outW);
        const uint32_t rowElements = static_cast<uint32_t>(static_cast<uint64_t>(outW) * block);
        return block > 0U && rowElements > 0U && rowElements <= OUTPUT_TILE_NUM &&
               CanUseNdc1hwc0NcdhwStride2Pool2Row(
                   static_cast<uint32_t>(block < static_cast<uint64_t>(tiling_->c) ? block : tiling_->c),
                   static_cast<uint32_t>(block));
    }

    __aicore__ inline bool CanUseNdc1hwc0NcdhwStride2FullC1PlanePath() const
    {
        if (!CanUseNdc1hwc0NcdhwStride2GroupPath() || tiling_->outD <= 0 || tiling_->outW <= 0 || tiling_->outH <= 0 ||
            tiling_->c <= 0) {
            return false;
        }
        const uint64_t block = Ndc1hwc0Block();
        const uint64_t validC1 = Ndc1hwc0ValidC1(block);
        if (block == 0U || validC1 == 0U || validC1 > 8U) {
            return false;
        }
        const uint32_t outW = static_cast<uint32_t>(tiling_->outW);
        const uint32_t outH = static_cast<uint32_t>(tiling_->outH);
        const uint32_t rowElements = static_cast<uint32_t>(static_cast<uint64_t>(outW) * block);
        const uint32_t outputCount = static_cast<uint32_t>(validC1) * outH * rowElements;
        const uint32_t alignedW = AlignToVector(outW);
        const uint32_t inputNeedW = outW * 2U;
        const uint32_t alignedInputW = AlignToVector(inputNeedW);
        const uint32_t compactCount = static_cast<uint32_t>(block) * alignedW;
        const uint32_t channelOffset = AlignToVector(compactCount);
        const uint32_t oddOffset = channelOffset + Ndc1hwc0GatherTempOffset(compactCount);
        const uint32_t compactOffset = oddOffset + compactCount;
        const uint32_t scatterOffset = compactOffset + compactCount;
        const uint32_t scratchNeed = scatterOffset + Ndc1hwc0GatherTempOffset(rowElements);
        return outputCount > 0U && outputCount <= OUTPUT_TILE_NUM && compactCount + 1U <= OUTPUT_TILE_NUM &&
               compactCount <= OUTPUT_TILE_NUM && scratchNeed <= OUTPUT_TILE_NUM &&
               alignedInputW * static_cast<uint32_t>(block) <= INPUT_TILE_NUM &&
               tiling_->normalCoreOut >= outputCount && tiling_->normalCoreOut % outputCount == 0U;
    }

    __aicore__ inline void ProcessNdc1hwc0NcdhwStride2FullC1Plane()
    {
        uint64_t block = 0U, validC1 = 0U, validOut = 0U;
        uint32_t outW = 0U, outH = 0U, rowElements = 0U, outputCount = 0U;
        if (!InitNdc1hwc0FullPlaneGeometry(block, validC1, validOut, outW, outH, rowElements, outputCount)) {
            return;
        }
        uint64_t outOffset = 0U, outCount = 0U, outEnd = 0U;
        GetNdc1hwc0ValidCoreRange(validOut, rowElements, outOffset, outCount, outEnd);
        uint64_t cur = outOffset;
        while (cur < outEnd) {
            if (!PrepareNdc1hwc0FullPlane(cur, outEnd, block, validC1, rowElements, outputCount)) {
                continue;
            }
            const Ndc1hwc0PlaneGroupContext context = GetNdc1hwc0PlaneGroupContext(cur, outEnd, outputCount, 1U);
            ProcessNdc1hwc0NcdhwStride2FullC1PlaneTile(cur, context.nIdx, context.od, static_cast<uint32_t>(validC1),
                                                       static_cast<uint32_t>(block), outH, outW);
            cur += outputCount;
        }
        if (validOut < tiling_->totalOut) {
            CopyOutZeroRangeByCore(validOut, tiling_->totalOut - validOut);
        }
    }

    __aicore__ inline void ReduceNdc1hwc0NcdhwStride2Window(LocalTensor<T> accLocal, LocalTensor<T> compactLocal,
                                                            LocalTensor<T> oddLocal,
                                                            LocalTensor<uint32_t> channelOffsetLocal, int64_t nIdx,
                                                            int64_t od, int64_t oh, int64_t cBase, uint32_t cCount,
                                                            uint32_t validInputW, uint32_t alignedInputW,
                                                            uint32_t srcStrideElements, uint32_t compactCount)
    {
        ReduceNdc1hwc0NcdhwStride2Pool2Row(accLocal, compactLocal, oddLocal, channelOffsetLocal, nIdx, od, cBase, oh,
                                           cCount, validInputW, alignedInputW, compactCount, srcStrideElements);
    }

    __aicore__ inline void ProcessNdc1hwc0NcdhwStride2FullC1Channel(
        int64_t nIdx, int64_t od, uint32_t c1, uint32_t block, uint32_t outH, uint32_t outW, uint32_t alignedW,
        uint32_t validInputW, uint32_t alignedInputW, uint32_t rowElements, uint32_t srcStrideElements,
        LocalTensor<T> accLocal, LocalTensor<T> compactLocal, LocalTensor<T> oddLocal, LocalTensor<T> outLocal,
        LocalTensor<uint32_t> channelOffsetLocal, LocalTensor<uint32_t> scatterOffsetLocal)
    {
        const int64_t cBase = static_cast<int64_t>(c1 * block);
        int64_t activeChannels = tiling_->c - cBase;
        if (activeChannels <= 0) {
            return;
        }
        if (activeChannels > static_cast<int64_t>(block)) {
            activeChannels = static_cast<int64_t>(block);
        }
        const uint32_t cCount = static_cast<uint32_t>(activeChannels);
        const uint32_t compactCount = cCount * alignedW;
        InitNdc1hwc0NcdhwStride2ChannelGatherOffsets(channelOffsetLocal, cCount, outW, alignedW, alignedInputW);
        InitNdc1hwc0NcdhwRowGatherOffsets(scatterOffsetLocal, cCount, block, outW, alignedW, compactCount);
        for (uint32_t oh = 0; oh < outH; ++oh) {
            Duplicate(accLocal, NegInfValue(), compactCount);
            PipeBarrier<PIPE_V>();
            ReduceNdc1hwc0NcdhwStride2Window(accLocal, compactLocal, oddLocal, channelOffsetLocal, nIdx, od, oh, cBase,
                                             cCount, validInputW, alignedInputW, srcStrideElements, compactCount);
            Duplicate(accLocal[compactCount], ZeroValue(), 1);
            PipeBarrier<PIPE_V>();
            const uint32_t outRow = c1 * outH + oh;
            Gather(outLocal[static_cast<uint64_t>(outRow) * rowElements], accLocal, scatterOffsetLocal,
                   static_cast<uint32_t>(0), rowElements);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ProcessNdc1hwc0NcdhwStride2FullC1PlaneTile(uint64_t outputOffset, int64_t nIdx, int64_t od,
                                                                      uint32_t validC1, uint32_t block, uint32_t outH,
                                                                      uint32_t outW)
    {
        if (ProcessNdc1hwc0NcdhwStride2SmallCSlabPlaneTile(outputOffset, nIdx, od, validC1, block, outH, outW) ||
            ProcessNdc1hwc0NcdhwStride2DualC1PlaneTile(outputOffset, nIdx, od, validC1, block, outH, outW)) {
            return;
        }
        const uint32_t alignedW = AlignToVector(outW);
        const uint32_t inputNeedW = outW * 2U;
        const uint32_t validInputW = static_cast<uint32_t>(
            static_cast<int64_t>(inputNeedW) < tiling_->inW ? inputNeedW : tiling_->inW);
        const uint32_t alignedInputW = AlignToVector(inputNeedW);
        const uint32_t rowElements = outW * block;
        const uint32_t outputCount = validC1 * outH * rowElements;
        const uint64_t channelStride = static_cast<uint64_t>(tiling_->inD) * static_cast<uint64_t>(tiling_->inH) *
                                       static_cast<uint64_t>(tiling_->inW);
        const uint32_t srcStrideElements = static_cast<uint32_t>(channelStride - static_cast<uint64_t>(validInputW));
        LocalTensor<T> accLocal = calcBuf_.Get<T>();
        LocalTensor<T> tmpLocal = tmpBuf_.Get<T>();
        LocalTensor<T> outLocal = maskBuf_.Get<T>();
        const uint32_t maxCompactCount = block * alignedW;
        const uint32_t channelOffset = AlignToVector(maxCompactCount);
        const uint32_t oddOffset = channelOffset + Ndc1hwc0GatherTempOffset(maxCompactCount);
        const uint32_t compactOffset = oddOffset + maxCompactCount;
        const uint32_t scatterOffset = compactOffset + maxCompactCount;
        LocalTensor<uint32_t> channelOffsetLocal = tmpLocal[channelOffset].template ReinterpretCast<uint32_t>();
        LocalTensor<T> oddLocal = tmpLocal[oddOffset];
        LocalTensor<T> compactLocal = tmpLocal[compactOffset];
        LocalTensor<uint32_t> scatterOffsetLocal = tmpLocal[scatterOffset].template ReinterpretCast<uint32_t>();
        for (uint32_t c1 = 0; c1 < validC1; ++c1) {
            ProcessNdc1hwc0NcdhwStride2FullC1Channel(
                nIdx, od, c1, block, outH, outW, alignedW, validInputW, alignedInputW, rowElements, srcStrideElements,
                accLocal, compactLocal, oddLocal, outLocal, channelOffsetLocal, scatterOffsetLocal);
        }
        CopyOutVector(outputOffset, outLocal, outputCount);
    }

    __aicore__ inline bool IsNdc1hwc0NcdhwStride2SmallCSlabShape(uint32_t outH, uint32_t outW) const
    {
        return tiling_->dataFormat == FORMAT_NCDHW_VALUE && tiling_->n > 0 && tiling_->c > 0 && tiling_->inD > 0 &&
               tiling_->inH > 0 && tiling_->inW > 0 && tiling_->outD == (tiling_->inD + 1) / 2 &&
               outH == static_cast<uint32_t>((tiling_->inH + 1) / 2) &&
               outW == static_cast<uint32_t>((tiling_->inW + 1) / 2);
    }

    __aicore__ inline bool HasNdc1hwc0NcdhwStride2SmallCSlabLayout(uint32_t validC1, uint32_t block) const
    {
        return (block == 16U && validC1 == 2U) || (block == 32U && validC1 == 1U);
    }

    __aicore__ inline bool HasNdc1hwc0NcdhwStride2SmallCSlabCapacity(uint32_t validC1, uint32_t block, uint32_t outH,
                                                                     uint32_t outW) const
    {
        const uint32_t inputPlane = static_cast<uint32_t>(tiling_->inH) * static_cast<uint32_t>(tiling_->inW);
        const uint32_t alignedInputPlane = AlignToVector(inputPlane);
        const uint32_t rowElements = outW * block;
        const uint32_t gatherCount = validC1 * rowElements;
        const uint32_t outputCount = validC1 * outH * rowElements;
        const uint32_t zeroIndex = static_cast<uint32_t>(tiling_->c) * alignedInputPlane;
        const uint32_t inputNeed = zeroIndex + alignedInputPlane * 2U;
        const uint32_t wReducedOffset = 2U * Ndc1hwc0GatherTempOffset(gatherCount);
        const uint32_t wReducedCount = static_cast<uint32_t>(tiling_->inH) * gatherCount;
        return inputPlane > 0U && alignedInputPlane > 0U && rowElements > 0U && gatherCount > 0U && outputCount > 0U &&
               inputNeed <= INPUT_TILE_NUM && outputCount <= OUTPUT_TILE_NUM && gatherCount <= OUTPUT_TILE_NUM &&
               wReducedOffset + wReducedCount <= OUTPUT_TILE_NUM;
    }

    __aicore__ inline void ReduceNdc1hwc0NcdhwStride2SmallCSlabWidth(LocalTensor<T> wReducedLocal,
                                                                     LocalTensor<T> oddLocal, LocalTensor<T> xLocal,
                                                                     LocalTensor<uint32_t> evenOffset,
                                                                     LocalTensor<uint32_t> oddOffset, uint32_t inH,
                                                                     uint32_t inW, uint32_t gatherCount)
    {
        for (uint32_t ih = 0U; ih < inH; ++ih) {
            const uint64_t rowSrcBase = static_cast<uint64_t>(ih) * inW;
            const uint64_t rowReducedBase = static_cast<uint64_t>(ih) * gatherCount;
            Gather(wReducedLocal[rowReducedBase], xLocal[rowSrcBase], evenOffset, static_cast<uint32_t>(0),
                   gatherCount);
            Gather(oddLocal, xLocal[rowSrcBase], oddOffset, static_cast<uint32_t>(0), gatherCount);
            PipeBarrier<PIPE_V>();
            Max(wReducedLocal[rowReducedBase], wReducedLocal[rowReducedBase], oddLocal, gatherCount);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ReduceNdc1hwc0NcdhwStride2SmallCSlabHeight(LocalTensor<T> accLocal,
                                                                      LocalTensor<T> wReducedLocal, uint32_t outH,
                                                                      uint32_t validC1, uint32_t rowElements,
                                                                      uint32_t gatherCount)
    {
        for (uint32_t oh = 0U; oh < outH; ++oh) {
            for (int64_t kh = 0; kh < 2; ++kh) {
                const int64_t ih = static_cast<int64_t>(oh) * tiling_->sH + kh - tiling_->padTop;
                if (IsOutOfRange(ih, tiling_->inH)) {
                    continue;
                }
                for (uint32_t c1 = 0U; c1 < validC1; ++c1) {
                    const uint64_t dstOffset = (static_cast<uint64_t>(c1) * outH + oh) * rowElements;
                    const uint64_t srcOffset = static_cast<uint64_t>(ih) * gatherCount +
                                               static_cast<uint64_t>(c1) * rowElements;
                    Max(accLocal[dstOffset], accLocal[dstOffset], wReducedLocal[srcOffset], rowElements);
                    PipeBarrier<PIPE_V>();
                }
            }
        }
    }

    __aicore__ inline void ReduceNdc1hwc0NcdhwStride2SmallCSlabDepth(
        int64_t nIdx, int64_t od, uint32_t cAll, uint32_t inH, uint32_t inW, uint32_t inputPlane,
        uint32_t alignedInputPlane, uint32_t zeroIndex, uint32_t negIndex, uint32_t outH, uint32_t validC1,
        uint32_t rowElements, uint32_t gatherCount, uint32_t srcStrideElements, LocalTensor<T> accLocal,
        LocalTensor<T> wReducedLocal, LocalTensor<T> oddLocal, LocalTensor<uint32_t> evenOffset,
        LocalTensor<uint32_t> oddOffset)
    {
        for (int64_t kd = 0; kd < 2; ++kd) {
            const int64_t id = Pool2InputD(od, kd);
            if (IsOutOfRange(id, tiling_->inD)) {
                continue;
            }
            CopyInVectorWBlocksPadStrideValue(InputOffset(nIdx, id, 0, 0, 0), cAll, inputPlane, alignedInputPlane,
                                              srcStrideElements, NegInfValue());
            LocalTensor<T> xLocal = xInQue_.DeQue<T>();
            Duplicate(xLocal[zeroIndex], ZeroValue(), alignedInputPlane);
            Duplicate(xLocal[negIndex], NegInfValue(), alignedInputPlane);
            PipeBarrier<PIPE_V>();
            ReduceNdc1hwc0NcdhwStride2SmallCSlabWidth(wReducedLocal, oddLocal, xLocal, evenOffset, oddOffset, inH, inW,
                                                      gatherCount);
            xInQue_.FreeTensor(xLocal);
            ReduceNdc1hwc0NcdhwStride2SmallCSlabHeight(accLocal, wReducedLocal, outH, validC1, rowElements,
                                                       gatherCount);
        }
    }

    __aicore__ inline void ExecuteNdc1hwc0NcdhwStride2SmallCSlabPlaneTile(uint64_t outputOffset, int64_t nIdx,
                                                                          int64_t od, uint32_t validC1, uint32_t block,
                                                                          uint32_t outH, uint32_t outW)
    {
        const uint32_t cAll = static_cast<uint32_t>(tiling_->c);
        const uint32_t inH = static_cast<uint32_t>(tiling_->inH);
        const uint32_t inW = static_cast<uint32_t>(tiling_->inW);
        const uint32_t inputPlane = inH * inW;
        const uint32_t alignedInputPlane = AlignToVector(inputPlane);
        const uint32_t rowElements = outW * block;
        const uint32_t gatherCount = validC1 * rowElements;
        const uint32_t outputCount = validC1 * outH * rowElements;
        const uint32_t zeroIndex = cAll * alignedInputPlane;
        const uint32_t negIndex = zeroIndex + alignedInputPlane;
        const uint32_t offsetNeed = Ndc1hwc0GatherTempOffset(gatherCount);
        const uint32_t wReducedOffset = 2U * offsetNeed;
        LocalTensor<T> accLocal = calcBuf_.Get<T>();
        LocalTensor<T> tmpLocal = tmpBuf_.Get<T>();
        LocalTensor<T> oddLocal = maskBuf_.Get<T>();
        LocalTensor<uint32_t> evenOffset = tmpLocal.template ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> oddOffset = tmpLocal[offsetNeed].template ReinterpretCast<uint32_t>();
        LocalTensor<T> wReducedLocal = tmpLocal[wReducedOffset];
        InitNdc1hwc0NcdhwStride2SlabGatherOffsets(evenOffset, oddOffset, cAll, validC1, block, outW, inW,
                                                  alignedInputPlane, zeroIndex, negIndex);
        Duplicate(accLocal, NegInfValue(), outputCount);
        PipeBarrier<PIPE_V>();
        const uint64_t channelStride = static_cast<uint64_t>(tiling_->inD) * static_cast<uint64_t>(tiling_->inH) *
                                       static_cast<uint64_t>(tiling_->inW);
        const uint32_t srcStrideElements = static_cast<uint32_t>(channelStride - inputPlane);
        ReduceNdc1hwc0NcdhwStride2SmallCSlabDepth(nIdx, od, cAll, inH, inW, inputPlane, alignedInputPlane, zeroIndex,
                                                  negIndex, outH, validC1, rowElements, gatherCount, srcStrideElements,
                                                  accLocal, wReducedLocal, oddLocal, evenOffset, oddOffset);
        CopyOutVector(outputOffset, accLocal, outputCount);
    }

    __aicore__ inline bool ProcessNdc1hwc0NcdhwStride2SmallCSlabPlaneTile(uint64_t outputOffset, int64_t nIdx,
                                                                          int64_t od, uint32_t validC1, uint32_t block,
                                                                          uint32_t outH, uint32_t outW)
    {
        if constexpr (AscendC::Std::is_same<T, bfloat16_t>::value) {
            return false;
        } else {
            if (!IsNdc1hwc0NcdhwStride2SmallCSlabShape(outH, outW) || !HasNdc1hwc0Stride2Pool2Spec() ||
                !HasNdc1hwc0NcdhwStride2SmallCSlabLayout(validC1, block) ||
                !HasNdc1hwc0NcdhwStride2SmallCSlabCapacity(validC1, block, outH, outW)) {
                return false;
            }
            ExecuteNdc1hwc0NcdhwStride2SmallCSlabPlaneTile(outputOffset, nIdx, od, validC1, block, outH, outW);
            return true;
        }
    }

    __aicore__ inline void InitNdc1hwc0NcdhwStride2SlabGatherOffsets(LocalTensor<uint32_t> evenOffset,
                                                                     LocalTensor<uint32_t> oddOffset, uint32_t cAll,
                                                                     uint32_t validC1, uint32_t block, uint32_t outW,
                                                                     uint32_t inW, uint32_t alignedInputPlane,
                                                                     uint32_t zeroIndex, uint32_t negIndex)
    {
        LocalTensor<int32_t> evenI32 = evenOffset.template ReinterpretCast<int32_t>();
        LocalTensor<int32_t> oddI32 = oddOffset.template ReinterpretCast<int32_t>();
        const int32_t zeroOffset = static_cast<int32_t>(zeroIndex * sizeof(T));
        const int32_t negOffset = static_cast<int32_t>(negIndex * sizeof(T));
        const uint32_t rowElements = outW * block;
        for (uint32_t c1 = 0U; c1 < validC1; ++c1) {
            for (uint32_t ow = 0U; ow < outW; ++ow) {
                const uint32_t evenW = ow * 2U;
                const uint32_t oddW = evenW + 1U;
                const uint32_t outBase = c1 * rowElements + ow * block;
                for (uint32_t c0 = 0U; c0 < block; ++c0) {
                    const uint32_t outIndex = outBase + c0;
                    const uint32_t cIdx = c1 * block + c0;
                    int32_t evenSrc = zeroOffset;
                    int32_t oddSrc = zeroOffset;
                    if (cIdx < cAll) {
                        const uint32_t channelBase = cIdx * alignedInputPlane;
                        evenSrc = evenW < inW ? static_cast<int32_t>((channelBase + evenW) * sizeof(T)) : negOffset;
                        oddSrc = oddW < inW ? static_cast<int32_t>((channelBase + oddW) * sizeof(T)) : negOffset;
                    }
                    evenI32.SetValue(outIndex, evenSrc);
                    oddI32.SetValue(outIndex, oddSrc);
                }
            }
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline bool HasNdc1hwc0NcdhwStride2DualC1Layout(uint32_t validC1, uint32_t block) const
    {
        return tiling_->dataFormat == FORMAT_NCDHW_VALUE && validC1 == 2U && block == 16U &&
               tiling_->c > static_cast<int64_t>(block) && tiling_->c <= static_cast<int64_t>(block * validC1) &&
               tiling_->outW > 0 && tiling_->inW > 0;
    }

    __aicore__ inline bool HasNdc1hwc0NcdhwStride2DualC1Capacity(uint32_t validC1, uint32_t block, uint32_t outH,
                                                                 uint32_t outW) const
    {
        const uint32_t alignedW = AlignToVector(outW);
        const uint32_t inputNeedW = outW * 2U;
        const uint32_t validInputW = static_cast<uint32_t>(
            static_cast<int64_t>(inputNeedW) < tiling_->inW ? inputNeedW : tiling_->inW);
        const uint32_t alignedInputW = AlignToVector(inputNeedW);
        const uint32_t rowElements = outW * block;
        const uint32_t outputCount = validC1 * outH * rowElements;
        const uint32_t cAll = static_cast<uint32_t>(tiling_->c);
        const uint32_t tailCompact = (cAll - block) * alignedW;
        const uint32_t compactAll = cAll * alignedW;
        const uint64_t channelStride = static_cast<uint64_t>(tiling_->inD) * static_cast<uint64_t>(tiling_->inH) *
                                       static_cast<uint64_t>(tiling_->inW);
        const uint32_t channelOffset = AlignToVector(compactAll);
        const uint32_t oddOffset = channelOffset + Ndc1hwc0GatherTempOffset(compactAll);
        const uint32_t compactOffset = oddOffset + compactAll;
        const uint32_t fullScatterOffset = compactOffset + compactAll;
        const uint32_t tailScatterOffset = fullScatterOffset + Ndc1hwc0GatherTempOffset(rowElements);
        const uint32_t scratchNeed = tailScatterOffset + Ndc1hwc0GatherTempOffset(rowElements);
        const bool inputAndOutputFit = validInputW > 0U && alignedW > 0U && alignedInputW > 0U &&
                                       channelStride >= static_cast<uint64_t>(validInputW) && rowElements > 0U &&
                                       rowElements <= 255U && outputCount > 0U && outputCount <= OUTPUT_TILE_NUM;
        return inputAndOutputFit && compactAll + 1U <= OUTPUT_TILE_NUM && cAll * alignedInputW <= INPUT_TILE_NUM &&
               scratchNeed <= OUTPUT_TILE_NUM && tailCompact + 1U <= OUTPUT_TILE_NUM;
    }

    __aicore__ inline void StoreNdc1hwc0NcdhwStride2DualC1Rows(
        int64_t nIdx, int64_t od, uint32_t outH, uint32_t rowElements, uint32_t cAll, uint32_t validInputW,
        uint32_t alignedInputW, uint32_t srcStrideElements, uint32_t compactAll, uint32_t fullCompact,
        LocalTensor<T> accLocal, LocalTensor<T> compactLocal, LocalTensor<T> oddLocal, LocalTensor<T> outLocal,
        LocalTensor<uint32_t> channelOffsetLocal, LocalTensor<uint32_t> fullScatterLocal,
        LocalTensor<uint32_t> tailScatterLocal)
    {
        for (uint32_t oh = 0; oh < outH; ++oh) {
            Duplicate(accLocal, NegInfValue(), compactAll);
            PipeBarrier<PIPE_V>();
            ReduceNdc1hwc0NcdhwStride2Window(accLocal, compactLocal, oddLocal, channelOffsetLocal, nIdx, od, oh, 0,
                                             cAll, validInputW, alignedInputW, srcStrideElements, compactAll);
            Gather(outLocal[static_cast<uint64_t>(oh) * rowElements], accLocal, fullScatterLocal,
                   static_cast<uint32_t>(0), rowElements);
            PipeBarrier<PIPE_V>();
            Gather(outLocal[static_cast<uint64_t>(outH + oh) * rowElements], accLocal[fullCompact], tailScatterLocal,
                   static_cast<uint32_t>(0), rowElements);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ExecuteNdc1hwc0NcdhwStride2DualC1PlaneTile(uint64_t outputOffset, int64_t nIdx, int64_t od,
                                                                      uint32_t validC1, uint32_t block, uint32_t outH,
                                                                      uint32_t outW)
    {
        const uint32_t alignedW = AlignToVector(outW);
        const uint32_t inputNeedW = outW * 2U;
        const uint32_t validInputW = static_cast<uint32_t>(
            static_cast<int64_t>(inputNeedW) < tiling_->inW ? inputNeedW : tiling_->inW);
        const uint32_t alignedInputW = AlignToVector(inputNeedW);
        const uint32_t rowElements = outW * block;
        const uint32_t outputCount = validC1 * outH * rowElements;
        const uint32_t cAll = static_cast<uint32_t>(tiling_->c);
        const uint32_t tailC = cAll - block;
        const uint32_t compactAll = cAll * alignedW;
        const uint32_t fullCompact = block * alignedW;
        const uint32_t tailCompact = tailC * alignedW;
        const uint64_t channelStride = static_cast<uint64_t>(tiling_->inD) * static_cast<uint64_t>(tiling_->inH) *
                                       static_cast<uint64_t>(tiling_->inW);
        const uint32_t srcStrideElements = static_cast<uint32_t>(channelStride - static_cast<uint64_t>(validInputW));
        LocalTensor<T> accLocal = calcBuf_.Get<T>();
        LocalTensor<T> tmpLocal = tmpBuf_.Get<T>();
        LocalTensor<T> outLocal = maskBuf_.Get<T>();
        const uint32_t channelOffset = AlignToVector(compactAll);
        const uint32_t oddOffset = channelOffset + Ndc1hwc0GatherTempOffset(compactAll);
        const uint32_t compactOffset = oddOffset + compactAll;
        const uint32_t fullScatterOffset = compactOffset + compactAll;
        const uint32_t tailScatterOffset = fullScatterOffset + Ndc1hwc0GatherTempOffset(rowElements);
        LocalTensor<uint32_t> channelOffsetLocal = tmpLocal[channelOffset].template ReinterpretCast<uint32_t>();
        LocalTensor<T> oddLocal = tmpLocal[oddOffset];
        LocalTensor<T> compactLocal = tmpLocal[compactOffset];
        LocalTensor<uint32_t> fullScatterLocal = tmpLocal[fullScatterOffset].template ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> tailScatterLocal = tmpLocal[tailScatterOffset].template ReinterpretCast<uint32_t>();
        InitNdc1hwc0NcdhwStride2ChannelGatherOffsets(channelOffsetLocal, cAll, outW, alignedW, alignedInputW);
        InitNdc1hwc0NcdhwRowGatherOffsets(fullScatterLocal, block, block, outW, alignedW, fullCompact);
        InitNdc1hwc0NcdhwRowGatherOffsets(tailScatterLocal, tailC, block, outW, alignedW, tailCompact);
        Duplicate(accLocal[compactAll], ZeroValue(), 1);
        PipeBarrier<PIPE_V>();
        StoreNdc1hwc0NcdhwStride2DualC1Rows(nIdx, od, outH, rowElements, cAll, validInputW, alignedInputW,
                                            srcStrideElements, compactAll, fullCompact, accLocal, compactLocal,
                                            oddLocal, outLocal, channelOffsetLocal, fullScatterLocal, tailScatterLocal);
        CopyOutVector(outputOffset, outLocal, outputCount);
    }

    __aicore__ inline bool ProcessNdc1hwc0NcdhwStride2DualC1PlaneTile(uint64_t outputOffset, int64_t nIdx, int64_t od,
                                                                      uint32_t validC1, uint32_t block, uint32_t outH,
                                                                      uint32_t outW)
    {
        if constexpr (AscendC::Std::is_same<T, bfloat16_t>::value) {
            return false;
        } else {
            if (!HasNdc1hwc0NcdhwStride2DualC1Layout(validC1, block) || !HasNdc1hwc0Stride2Pool2Spec() ||
                !HasNdc1hwc0NcdhwStride2DualC1Capacity(validC1, block, outH, outW)) {
                return false;
            }
            ExecuteNdc1hwc0NcdhwStride2DualC1PlaneTile(outputOffset, nIdx, od, validC1, block, outH, outW);
            return true;
        }
    }

    __aicore__ inline void ProcessNdc1hwc0NcdhwStride2GroupStep(uint64_t& cur, uint64_t outEnd, uint64_t block,
                                                                uint64_t validC1, uint64_t rowElements)
    {
        Ndc1hwc0DecodedRow context{};
        if (!PrepareNdc1hwc0DecodedFullRow(cur, outEnd, block, rowElements, validC1, context)) {
            return;
        }
        const int64_t cBase = context.c1Idx * static_cast<int64_t>(block);
        const int64_t activeChannels = Ndc1hwc0ActiveChannels(cBase, block);
        uint32_t rows = static_cast<uint32_t>(tiling_->outH - context.oh);
        if (rowElements == 0U) {
            cur = outEnd;
            return;
        }
        const uint64_t remainRows = (outEnd - cur) / rowElements;
        if (static_cast<uint64_t>(rows) > remainRows) {
            rows = static_cast<uint32_t>(remainRows);
        }
        const uint32_t maxRowsByOutput = OUTPUT_TILE_NUM / static_cast<uint32_t>(rowElements);
        if (rows > maxRowsByOutput) {
            rows = maxRowsByOutput;
        }
        if (rows == 0U) {
            ProcessNdc1hwc0RowVectorByRow(context.row, cur, block, validC1);
            cur += rowElements;
            return;
        }
        if (activeChannels > 0 &&
            CanUseNdc1hwc0NcdhwStride2GroupTile(static_cast<uint32_t>(activeChannels), static_cast<uint32_t>(block))) {
            ProcessNdc1hwc0NcdhwStride2GroupTile(cur, context.nIdx, context.od, cBase,
                                                 static_cast<uint32_t>(context.oh), rows,
                                                 static_cast<uint32_t>(activeChannels), static_cast<uint32_t>(block));
        } else {
            ProcessNdc1hwc0RowVectorTile(context.row, rows, cur, block, validC1);
        }
        cur += static_cast<uint64_t>(rows) * rowElements;
    }

    __aicore__ inline void ProcessNdc1hwc0NcdhwStride2Group()
    {
        uint64_t block = 0U, validC1 = 0U, validOut = 0U, rowElements = 0U;
        uint64_t outOffset = 0U, outCount = 0U, outEnd = 0U;
        if (!InitNdc1hwc0LinearRange(block, validC1, validOut, rowElements, outOffset, outCount, outEnd)) {
            return;
        }
        uint64_t cur = outOffset;
        while (cur < outEnd) {
            ProcessNdc1hwc0NcdhwStride2GroupStep(cur, outEnd, block, validC1, rowElements);
        }
        if (validOut < tiling_->totalOut) {
            CopyOutZeroRangeByCore(validOut, tiling_->totalOut - validOut);
        }
    }

    __aicore__ inline bool CanUseNdc1hwc0NcdhwStride2GroupTile(uint32_t cCount, uint32_t block) const
    {
        if (!CanUseNdc1hwc0NcdhwStride2Pool2Row(cCount, block)) {
            return false;
        }
        const uint32_t outW = static_cast<uint32_t>(tiling_->outW);
        const uint32_t alignedW = AlignToVector(outW);
        const uint32_t inputNeedW = outW * 2U;
        const uint32_t alignedInputW = AlignToVector(inputNeedW);
        const uint32_t compactCount = cCount * alignedW;
        const uint32_t rowElements = outW * block;
        const uint32_t channelOffset = AlignToVector(compactCount);
        const uint32_t oddOffset = channelOffset + Ndc1hwc0GatherTempOffset(compactCount);
        const uint32_t compactOffset = oddOffset + compactCount;
        const uint32_t scatterOffset = compactOffset + compactCount;
        const uint32_t scratchNeed = scatterOffset + Ndc1hwc0GatherTempOffset(rowElements);
        return rowElements <= OUTPUT_TILE_NUM && compactCount + 1U <= OUTPUT_TILE_NUM &&
               compactCount <= OUTPUT_TILE_NUM && scratchNeed <= OUTPUT_TILE_NUM && rowElements <= 255U;
    }

    __aicore__ inline void ProcessNdc1hwc0NcdhwStride2GroupTile(uint64_t outputOffset, int64_t nIdx, int64_t od,
                                                                int64_t cBase, uint32_t ohStart, uint32_t rows,
                                                                uint32_t cCount, uint32_t block)
    {
        const uint32_t outW = static_cast<uint32_t>(tiling_->outW);
        const uint32_t alignedW = AlignToVector(outW);
        const uint32_t inputNeedW = outW * 2U;
        const uint32_t validInputW = static_cast<uint32_t>(
            static_cast<int64_t>(inputNeedW) < tiling_->inW ? inputNeedW : tiling_->inW);
        const uint32_t alignedInputW = AlignToVector(inputNeedW);
        const uint32_t compactCount = cCount * alignedW;
        const uint32_t rowElements = outW * block;
        const uint64_t channelStride = static_cast<uint64_t>(tiling_->inD) * static_cast<uint64_t>(tiling_->inH) *
                                       static_cast<uint64_t>(tiling_->inW);
        const uint32_t srcStrideElements = static_cast<uint32_t>(channelStride - static_cast<uint64_t>(validInputW));

        LocalTensor<T> accLocal = calcBuf_.Get<T>();
        LocalTensor<T> tmpLocal = tmpBuf_.Get<T>();
        LocalTensor<T> outLocal = maskBuf_.Get<T>();
        const uint32_t channelOffset = AlignToVector(compactCount);
        const uint32_t oddOffset = channelOffset + Ndc1hwc0GatherTempOffset(compactCount);
        const uint32_t compactOffset = oddOffset + compactCount;
        const uint32_t scatterOffset = compactOffset + compactCount;
        LocalTensor<uint32_t> channelOffsetLocal = tmpLocal[channelOffset].template ReinterpretCast<uint32_t>();
        LocalTensor<T> oddLocal = tmpLocal[oddOffset];
        LocalTensor<T> compactLocal = tmpLocal[compactOffset];
        LocalTensor<uint32_t> scatterOffsetLocal = tmpLocal[scatterOffset].template ReinterpretCast<uint32_t>();
        InitNdc1hwc0NcdhwStride2ChannelGatherOffsets(channelOffsetLocal, cCount, outW, alignedW, alignedInputW);
        InitNdc1hwc0NcdhwRowGatherOffsets(scatterOffsetLocal, cCount, block, outW, alignedW, compactCount);
        for (uint32_t rowIdx = 0; rowIdx < rows; ++rowIdx) {
            Duplicate(accLocal, NegInfValue(), compactCount);
            PipeBarrier<PIPE_V>();
            const int64_t oh = static_cast<int64_t>(ohStart + rowIdx);
            ReduceNdc1hwc0NcdhwStride2Window(accLocal, compactLocal, oddLocal, channelOffsetLocal, nIdx, od, oh, cBase,
                                             cCount, validInputW, alignedInputW, srcStrideElements, compactCount);
            Duplicate(accLocal[compactCount], ZeroValue(), 1);
            PipeBarrier<PIPE_V>();
            Gather(outLocal[static_cast<uint64_t>(rowIdx) * rowElements], accLocal, scatterOffsetLocal,
                   static_cast<uint32_t>(0), rowElements);
            PipeBarrier<PIPE_V>();
        }
        CopyOutVector(outputOffset, outLocal, rows * rowElements);
    }

    __aicore__ inline bool HasNdc1hwc0NdhwcD3H3Dil2ReuseShape() const
    {
        return tiling_->dataFormat == FORMAT_NDHWC_VALUE && tiling_->outW > 0 && tiling_->outH > 0 &&
               tiling_->inH > 0 && tiling_->c > 0;
    }

    __aicore__ inline bool HasNdc1hwc0NdhwcD3H3Dil2ReuseLayout(uint64_t block) const
    {
        return block > 0U && tiling_->outputC1 == 1 && static_cast<uint64_t>(tiling_->c) <= block &&
               tiling_->outW == tiling_->inW;
    }

    __aicore__ inline bool HasNdc1hwc0NdhwcD3H3Dil2ReuseCapacity(uint64_t block) const
    {
        const uint32_t cCount = static_cast<uint32_t>(tiling_->c);
        const uint32_t outW = static_cast<uint32_t>(tiling_->outW);
        const uint32_t rowCount = outW * cCount;
        const uint32_t alignedRowCount = AlignToVector(rowCount);
        const uint32_t rowElements = static_cast<uint32_t>(outW * block);
        const uint32_t scatterScratch = Ndc1hwc0GatherTempOffset(rowElements);
        return rowCount > 0U && rowElements > 0U && rowElements <= OUTPUT_TILE_NUM &&
               rowCount + 1U <= OUTPUT_TILE_NUM && rowElements + scatterScratch <= OUTPUT_TILE_NUM &&
               alignedRowCount <= INPUT_TILE_NUM;
    }

    __aicore__ inline bool CanUseNdc1hwc0NdhwcD3H3Dil2ReusePath() const
    {
        const uint64_t block = Ndc1hwc0Block();
        return HasNdc1hwc0NdhwcD3H3Dil2ReuseShape() && HasNdc1hwc0NdhwcD3H3Dil2ReuseLayout(block) &&
               HasNdc1hwc0D3H3Dil2PoolSpec() && HasNdc1hwc0NdhwcD3H3Dil2ReuseCapacity(block);
    }

    __aicore__ inline void ProcessNdc1hwc0NdhwcD3H3Dil2ReuseStep(uint64_t& cur, uint64_t outEnd, uint64_t block,
                                                                 uint64_t validC1, uint32_t rowElements,
                                                                 uint32_t cCount, uint32_t outW)
    {
        Ndc1hwc0GroupStepContext context{};
        if (!PrepareNdc1hwc0GroupStep(cur, outEnd, block, validC1, rowElements, false, context)) {
            return;
        }
        ProcessNdc1hwc0NdhwcD3H3Dil2ReuseGroup(cur, context.nIdx, context.od, static_cast<uint32_t>(context.oh),
                                               context.rows, cCount, static_cast<uint32_t>(block), outW);
        cur += static_cast<uint64_t>(context.rows) * rowElements;
    }

    __aicore__ inline void ProcessNdc1hwc0NdhwcD3H3Dil2Reuse()
    {
        uint64_t block = 0U, validC1 = 0U, validOut = 0U;
        uint64_t outOffset = 0U, outCount = 0U, outEnd = 0U;
        uint32_t outW = 0U, rowElements = 0U;
        const bool initialized = InitNdc1hwc0GroupRange(block, validC1, validOut, outW, rowElements, outOffset,
                                                        outCount, outEnd);
        if (!initialized) {
            return;
        }
        const uint32_t cCount = static_cast<uint32_t>(tiling_->c);
        uint64_t cur = outOffset;
        while (cur < outEnd) {
            ProcessNdc1hwc0NdhwcD3H3Dil2ReuseStep(cur, outEnd, block, validC1, rowElements, cCount, outW);
        }
        if (validOut < tiling_->totalOut) {
            CopyOutZeroRangeByCore(validOut, tiling_->totalOut - validOut);
        }
    }

    __aicore__ inline void ProcessNdc1hwc0NdhwcD3H3Dil2ReuseGroup(uint64_t outputOffset, int64_t nIdx, int64_t od,
                                                                  uint32_t ohStart, uint32_t rows, uint32_t cCount,
                                                                  uint32_t block, uint32_t outW)
    {
        const uint32_t rowCount = outW * cCount;
        const uint32_t alignedRowCount = AlignToVector(rowCount);
        LocalTensor<T> compactLocal = tmpBuf_.Get<T>();
        LocalTensor<T> outLocal = maskBuf_.Get<T>();

        const uint32_t rowElements = outW * block;
        uint32_t doneRows = 0U;
        while (doneRows < rows) {
            uint32_t tileRows = rows - doneRows;
            const uint32_t maxTileRows = OUTPUT_TILE_NUM / rowElements;
            if (tileRows > maxTileRows) {
                tileRows = maxTileRows;
            }
            if (tileRows == 0U) {
                tileRows = 1U;
            }
            Duplicate(outLocal, ZeroValue(), tileRows * rowElements);
            PipeBarrier<PIPE_V>();
            for (uint32_t rowIdx = 0; rowIdx < tileRows; ++rowIdx) {
                const uint32_t oh = ohStart + doneRows + rowIdx;
                Duplicate(compactLocal, NegInfValue(), alignedRowCount);
                PipeBarrier<PIPE_V>();
                for (int64_t kd = 0; kd < 3; ++kd) {
                    const int64_t id = od * 3 + kd;
                    if (IsOutOfRange(id, tiling_->inD)) {
                        continue;
                    }
                    for (int64_t kh = 0; kh < 3; ++kh) {
                        const int64_t ih = static_cast<int64_t>(oh) + kh * 2 - 2;
                        if (IsOutOfRange(ih, tiling_->inH)) {
                            continue;
                        }
                        CopyInVectorPadValue(InputOffset(nIdx, id, ih, 0, 0), rowCount, alignedRowCount, NegInfValue());
                        LocalTensor<T> xLocal = xInQue_.DeQue<T>();
                        Max(compactLocal, compactLocal, xLocal, alignedRowCount);
                        PipeBarrier<PIPE_V>();
                        xInQue_.FreeTensor(xLocal);
                    }
                }
                ScatterNdc1hwc0NdhwcCompactRowScalar(outLocal[static_cast<uint64_t>(rowIdx) * rowElements],
                                                     compactLocal, cCount, block, outW);
            }
            CopyOutVector(outputOffset + static_cast<uint64_t>(doneRows) * rowElements, outLocal,
                          tileRows * rowElements);
            doneRows += tileRows;
        }
    }

    __aicore__ inline void ProcessNdc1hwc0NcdhwRowScalarByRow(uint64_t row, uint64_t outputOffset, uint64_t block,
                                                              uint64_t validC1)
    {
        int64_t nIdx = 0;
        int64_t od = 0;
        int64_t c1Idx = 0;
        int64_t oh = 0;
        DecodeNdc1hwc0Row(row, validC1, nIdx, od, c1Idx, oh);
        const int64_t cBase = c1Idx * static_cast<int64_t>(block);
        const int64_t activeChannels = Ndc1hwc0ActiveChannels(cBase, block);

        const uint32_t outW = static_cast<uint32_t>(tiling_->outW);
        const uint32_t rowElements = static_cast<uint32_t>(static_cast<uint64_t>(outW) * block);
        LocalTensor<T> rowLocal = maskBuf_.Get<T>();
        Duplicate(rowLocal, ZeroValue(), rowElements);
        PipeBarrier<PIPE_V>();
        if (activeChannels > 0) {
            if (CanUseNdc1hwc0NcdhwNoWPadRowScalar()) {
                ComputeNdc1hwc0NcdhwNoWPadRowScalar(rowLocal, nIdx, od, cBase, oh,
                                                    static_cast<uint32_t>(activeChannels), static_cast<uint32_t>(block),
                                                    outW);
            } else {
                ComputeNdc1hwc0NcdhwGenericRowScalar(rowLocal, nIdx, od, cBase, oh,
                                                     static_cast<uint32_t>(activeChannels),
                                                     static_cast<uint32_t>(block), outW);
            }
        }
        CopyOutVector(outputOffset, rowLocal, rowElements);
    }

    __aicore__ inline void FillNdc1hwc0NcdhwRowScalarTile(uint64_t startRow, uint32_t rowCount, uint64_t block,
                                                          uint64_t validC1, LocalTensor<T> yLocal)
    {
        const uint32_t outW = static_cast<uint32_t>(tiling_->outW);
        const uint32_t rowElements = static_cast<uint32_t>(static_cast<uint64_t>(outW) * block);
        LocalTensor<T> rowTmp = tmpBuf_.Get<T>();
        Duplicate(yLocal, ZeroValue(), rowElements * rowCount);
        PipeBarrier<PIPE_V>();
        for (uint32_t rowIdx = 0; rowIdx < rowCount; ++rowIdx) {
            const Ndc1hwc0DecodedRow context = DecodeNdc1hwc0RowContext(startRow + rowIdx, validC1);
            const int64_t cBase = context.c1Idx * static_cast<int64_t>(block);
            const int64_t activeChannels = Ndc1hwc0ActiveChannels(cBase, block);
            if (activeChannels <= 0) {
                continue;
            }
            Duplicate(rowTmp, ZeroValue(), rowElements);
            PipeBarrier<PIPE_V>();
            if (CanUseNdc1hwc0NcdhwNoWPadRowScalar()) {
                ComputeNdc1hwc0NcdhwNoWPadRowScalar(rowTmp, context.nIdx, context.od, cBase, context.oh,
                                                    static_cast<uint32_t>(activeChannels), static_cast<uint32_t>(block),
                                                    outW);
            } else {
                ComputeNdc1hwc0NcdhwGenericRowScalar(rowTmp, context.nIdx, context.od, cBase, context.oh,
                                                     static_cast<uint32_t>(activeChannels),
                                                     static_cast<uint32_t>(block), outW);
            }
            const uint64_t dstOffset = static_cast<uint64_t>(rowIdx) * rowElements;
            for (uint32_t i = 0; i < rowElements; ++i) {
                yLocal.SetValue(dstOffset + i, rowTmp.GetValue(i));
            }
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ProcessNdc1hwc0NcdhwRowScalarTile(uint64_t startRow, uint32_t rowCount,
                                                             uint64_t outputOffset, uint64_t block, uint64_t validC1)
    {
        const uint32_t rowElements = static_cast<uint32_t>(static_cast<uint64_t>(tiling_->outW) * block);
        LocalTensor<T> yLocal = maskBuf_.Get<T>();
        FillNdc1hwc0NcdhwRowScalarTile(startRow, rowCount, block, validC1, yLocal);
        CopyOutVector(outputOffset, yLocal, rowElements * rowCount);
    }

    __aicore__ inline void ComputeNdc1hwc0NcdhwGenericRowScalar(LocalTensor<T> rowLocal, int64_t nIdx, int64_t od,
                                                                int64_t cBase, int64_t oh, uint32_t cCount,
                                                                uint32_t block, uint32_t outW)
    {
        for (uint32_t c0 = 0; c0 < cCount; ++c0) {
            const int64_t cIdx = cBase + static_cast<int64_t>(c0);
            for (uint32_t ow = 0; ow < outW; ++ow) {
                rowLocal.SetValue(static_cast<uint64_t>(ow) * block + c0,
                                  ComputeValueAt(nIdx, od, oh, static_cast<int64_t>(ow), cIdx));
            }
        }
    }

    __aicore__ inline void ReduceNdc1hwc0NcdhwNoWPadDepth(LocalTensor<T> rowLocal, int64_t nIdx, int64_t id,
                                                          int64_t cIdx, int64_t oh, uint32_t c0, uint32_t block,
                                                          uint32_t outW)
    {
        for (int64_t kh = 0; kh < tiling_->kH; ++kh) {
            const int64_t ih = DilatedInputH(oh, kh);
            if (IsOutOfRange(ih, tiling_->inH)) {
                continue;
            }
            const uint64_t inputBase = InputOffset(nIdx, id, ih, 0, cIdx);
            for (uint32_t ow = 0; ow < outW; ++ow) {
                const uint64_t pos = static_cast<uint64_t>(ow) * block + c0;
                T maxValue = rowLocal.GetValue(pos);
                float maxValueFp32 = ValueToFloat(maxValue);
                UpdateMaxValueLoaded(xGm_.GetValue(inputBase + ow), maxValue, maxValueFp32);
                rowLocal.SetValue(pos, maxValue);
            }
        }
    }

    __aicore__ inline void ComputeNdc1hwc0NcdhwNoWPadRowScalar(LocalTensor<T> rowLocal, int64_t nIdx, int64_t od,
                                                               int64_t cBase, int64_t oh, uint32_t cCount,
                                                               uint32_t block, uint32_t outW)
    {
        for (uint32_t c0 = 0; c0 < cCount; ++c0) {
            for (uint32_t ow = 0; ow < outW; ++ow) {
                rowLocal.SetValue(static_cast<uint64_t>(ow) * block + c0, NegInfValue());
            }
            const int64_t cIdx = cBase + static_cast<int64_t>(c0);
            for (int64_t kd = 0; kd < tiling_->kD; ++kd) {
                const int64_t id = DilatedInputD(od, kd);
                if (!IsOutOfRange(id, tiling_->inD)) {
                    ReduceNdc1hwc0NcdhwNoWPadDepth(rowLocal, nIdx, id, cIdx, oh, c0, block, outW);
                }
            }
        }
    }

    __aicore__ inline void FillNdc1hwc0NcdhwK1Row(LocalTensor<T> rowLocal, uint32_t rowOffset, int64_t nIdx, int64_t od,
                                                  int64_t cBase, int64_t oh, uint32_t cCount, uint32_t block,
                                                  uint32_t outW)
    {
        if (FillNdc1hwc0NcdhwK1RowGather(rowLocal, rowOffset, nIdx, od, cBase, oh, cCount, block, outW)) {
            return;
        }
        for (uint32_t c0 = 0; c0 < cCount; ++c0) {
            const int64_t cIdx = cBase + static_cast<int64_t>(c0);
            const uint64_t inputBase = InputOffset(nIdx, od, oh, 0, cIdx);
            for (uint32_t ow = 0; ow < outW; ++ow) {
                rowLocal.SetValue(rowOffset + static_cast<uint64_t>(ow) * block + c0, xGm_.GetValue(inputBase + ow));
            }
        }
        for (uint32_t c0 = cCount; c0 < block; ++c0) {
            for (uint32_t ow = 0; ow < outW; ++ow) {
                rowLocal.SetValue(rowOffset + static_cast<uint64_t>(ow) * block + c0, ZeroValue());
            }
        }
    }

    __aicore__ inline void GatherNdc1hwc0NcdhwK1Row(LocalTensor<T> rowLocal, uint32_t rowOffset, LocalTensor<T> xIn,
                                                    LocalTensor<uint32_t> offsetLocal, uint32_t cCount, uint32_t block,
                                                    uint32_t outW, uint32_t alignedW, uint32_t zeroIndex,
                                                    uint32_t rowElements, bool initWideOffsets)
    {
        if (CanUseNdc1hwc0WideGather(rowElements)) {
            if (initWideOffsets) {
                InitNdc1hwc0NcdhwK1GatherOffsets(offsetLocal, cCount, block, outW, alignedW, zeroIndex);
            }
            Gather(rowLocal[static_cast<uint64_t>(rowOffset)], xIn, offsetLocal, static_cast<uint32_t>(0), rowElements);
            PipeBarrier<PIPE_V>();
            return;
        }
        uint32_t done = 0U;
        while (done < rowElements) {
            const uint32_t curCount = Ndc1hwc0SafeGatherChunk(rowElements - done);
            InitNdc1hwc0NcdhwRowGatherOffsetsChunk(offsetLocal, done, curCount, cCount, block, alignedW, zeroIndex);
            Gather(rowLocal[static_cast<uint64_t>(rowOffset) + done], xIn, offsetLocal, static_cast<uint32_t>(0),
                   curCount);
            PipeBarrier<PIPE_V>();
            done += curCount;
        }
    }

    __aicore__ inline bool FillNdc1hwc0NcdhwK1RowGather(LocalTensor<T> rowLocal, uint32_t rowOffset, int64_t nIdx,
                                                        int64_t od, int64_t cBase, int64_t oh, uint32_t cCount,
                                                        uint32_t block, uint32_t outW)
    {
        if (cCount == 0U || block == 0U || outW == 0U || cCount > block) {
            return false;
        }
        const uint32_t alignedW = AlignToVector(outW);
        const uint32_t rowElements = outW * block;
        const uint32_t inputElements = cCount * alignedW;
        const uint64_t channelStride = static_cast<uint64_t>(tiling_->inD) * static_cast<uint64_t>(tiling_->inH) *
                                       static_cast<uint64_t>(tiling_->inW);
        if (alignedW == 0U || rowElements > OUTPUT_TILE_NUM || inputElements + 1U > INPUT_TILE_NUM ||
            channelStride < static_cast<uint64_t>(outW)) {
            return false;
        }
        LocalTensor<T> xLocal = xInQue_.AllocTensor<T>();
        DataCopyExtParams copyParams{static_cast<uint16_t>(cCount), static_cast<uint32_t>(outW * sizeof(T)),
                                     static_cast<uint32_t>((channelStride - static_cast<uint64_t>(outW)) * sizeof(T)),
                                     0, 0};
        DataCopyPadExtParams<T> padParams{true, 0, static_cast<uint8_t>(alignedW - outW), ZeroValue()};
        DataCopyPad(xLocal, xGm_[InputOffset(nIdx, od, oh, 0, cBase)], copyParams, padParams);
        xInQue_.EnQue(xLocal);
        LocalTensor<T> xIn = xInQue_.DeQue<T>();
        LocalTensor<T> scratchLocal = tmpBuf_.Get<T>();
        LocalTensor<uint32_t> offsetLocal = scratchLocal.template ReinterpretCast<uint32_t>();
        const uint32_t zeroIndex = inputElements;
        Duplicate(xIn[zeroIndex], ZeroValue(), 1);
        PipeBarrier<PIPE_V>();
        GatherNdc1hwc0NcdhwK1Row(rowLocal, rowOffset, xIn, offsetLocal, cCount, block, outW, alignedW, zeroIndex,
                                 rowElements, true);
        xInQue_.FreeTensor(xIn);
        return true;
    }

    __aicore__ inline void InitNdc1hwc0NcdhwK1GatherOffsets(LocalTensor<uint32_t> offsetLocal, uint32_t cCount,
                                                            uint32_t block, uint32_t outW, uint32_t alignedW,
                                                            uint32_t zeroIndex)
    {
        InitNdc1hwc0NcdhwRowGatherOffsets(offsetLocal, cCount, block, outW, alignedW, zeroIndex);
    }

    __aicore__ inline void ProcessNdc1hwc0NcdhwK1DirectByRow(uint64_t row, uint64_t outputOffset, uint64_t block,
                                                             uint64_t validC1)
    {
        const Ndc1hwc0DecodedRow context = DecodeNdc1hwc0RowContext(row, validC1);
        const int64_t cBase = context.c1Idx * static_cast<int64_t>(block);
        const int64_t activeChannels = Ndc1hwc0ActiveChannels(cBase, block);

        const uint32_t outW = static_cast<uint32_t>(tiling_->outW);
        const uint32_t rowElements = static_cast<uint32_t>(static_cast<uint64_t>(outW) * block);
        LocalTensor<T> rowLocal = maskBuf_.Get<T>();
        if (activeChannels > 0) {
            FillNdc1hwc0NcdhwK1Row(rowLocal, 0U, context.nIdx, context.od, cBase, context.oh,
                                   static_cast<uint32_t>(activeChannels), static_cast<uint32_t>(block), outW);
        } else {
            Duplicate(rowLocal, ZeroValue(), rowElements);
            PipeBarrier<PIPE_V>();
        }
        CopyOutVector(outputOffset, rowLocal, rowElements);
    }

    __aicore__ inline void FillNdc1hwc0NcdhwK1DirectTile(uint64_t startRow, uint32_t rowCount, uint64_t block,
                                                         uint64_t validC1, LocalTensor<T> yLocal)
    {
        const uint32_t outW = static_cast<uint32_t>(tiling_->outW);
        const uint32_t rowElements = static_cast<uint32_t>(static_cast<uint64_t>(outW) * block);
        for (uint32_t rowIdx = 0; rowIdx < rowCount; ++rowIdx) {
            int64_t nIdx = 0;
            int64_t od = 0;
            int64_t c1Idx = 0;
            int64_t oh = 0;
            DecodeNdc1hwc0Row(startRow + rowIdx, validC1, nIdx, od, c1Idx, oh);
            const int64_t cBase = c1Idx * static_cast<int64_t>(block);
            const int64_t activeChannels = Ndc1hwc0ActiveChannels(cBase, block);

            const uint32_t rowOffset = rowIdx * rowElements;
            if (activeChannels > 0) {
                FillNdc1hwc0NcdhwK1Row(yLocal, rowOffset, nIdx, od, cBase, oh, static_cast<uint32_t>(activeChannels),
                                       static_cast<uint32_t>(block), outW);
            } else {
                for (uint32_t i = 0; i < rowElements; ++i) {
                    yLocal.SetValue(static_cast<uint64_t>(rowOffset) + i, ZeroValue());
                }
            }
        }
    }

    __aicore__ inline void ProcessNdc1hwc0NcdhwK1DirectTile(uint64_t startRow, uint32_t rowCount, uint64_t outputOffset,
                                                            uint64_t block, uint64_t validC1)
    {
        const uint32_t outW = static_cast<uint32_t>(tiling_->outW);
        const uint32_t rowElements = static_cast<uint32_t>(static_cast<uint64_t>(outW) * block);
        LocalTensor<T> yLocal = maskBuf_.Get<T>();
        FillNdc1hwc0NcdhwK1DirectTile(startRow, rowCount, block, validC1, yLocal);
        CopyOutVector(outputOffset, yLocal, rowElements * rowCount);
    }

    __aicore__ inline void ProcessNdc1hwc0NcdhwTiledRowStep(uint64_t& cur, uint64_t validEnd, uint64_t block,
                                                            uint64_t validC1, uint64_t rowElements, bool k1Direct)
    {
        if (rowElements == 0U) {
            cur = validEnd;
            return;
        }
        const uint64_t rowOffset = cur % rowElements;
        if (rowOffset != 0U || validEnd - cur < rowElements) {
            cur += ProcessNdc1hwc0PartialValidRange(cur, validEnd, block, rowElements, validC1);
            return;
        }
        const uint32_t rowElements32 = static_cast<uint32_t>(rowElements);
        uint32_t maxTileRows = OUTPUT_TILE_NUM / rowElements32;
        if (maxTileRows == 0U) {
            maxTileRows = 1U;
        }
        const uint64_t remainRows64 = (validEnd - cur) / rowElements;
        const uint32_t tileRows = remainRows64 > maxTileRows ? maxTileRows : static_cast<uint32_t>(remainRows64);
        if (tileRows == 0U) {
            if (k1Direct) {
                ProcessNdc1hwc0NcdhwK1DirectByRow(cur / rowElements, cur, block, validC1);
            } else {
                ProcessNdc1hwc0NcdhwRowScalarByRow(cur / rowElements, cur, block, validC1);
            }
            cur += rowElements;
            return;
        }
        if (k1Direct) {
            ProcessNdc1hwc0NcdhwK1DirectTile(cur / rowElements, tileRows, cur, block, validC1);
        } else {
            ProcessNdc1hwc0NcdhwRowScalarTile(cur / rowElements, tileRows, cur, block, validC1);
        }
        cur += static_cast<uint64_t>(tileRows) * rowElements;
    }

    __aicore__ inline void ProcessNdc1hwc0NcdhwTiledRows(bool k1Direct)
    {
        uint64_t block = 0U, validC1 = 0U, validOut = 0U;
        if (!InitNdc1hwc0ValidOutput(block, validC1, validOut)) {
            return;
        }
        const uint64_t rowElements = static_cast<uint64_t>(tiling_->outW) * block;
        const uint64_t outOffset = Ndc1hwc0ValidCoreStartOffset(validOut, rowElements);
        const uint64_t outCount = Ndc1hwc0ValidCoreElementCount(validOut, rowElements, outOffset);
        const uint64_t validEnd = outOffset + outCount;
        uint64_t cur = outOffset;
        while (cur < validEnd) {
            ProcessNdc1hwc0NcdhwTiledRowStep(cur, validEnd, block, validC1, rowElements, k1Direct);
        }
        if (validOut < tiling_->totalOut) {
            CopyOutZeroRangeByCore(validOut, tiling_->totalOut - validOut);
        }
    }

    __aicore__ inline void ProcessNdc1hwc0NcdhwK1Direct() { ProcessNdc1hwc0NcdhwTiledRows(true); }

    __aicore__ inline void ProcessNdc1hwc0NcdhwRowScalar() { ProcessNdc1hwc0NcdhwTiledRows(false); }

    __aicore__ inline void FillNdc1hwc0ValidTile(LocalTensor<T> yLocal, uint64_t outLinear, uint32_t curCount,
                                                 uint64_t block, uint64_t rowElements, uint64_t validC1)
    {
        Duplicate(yLocal, ZeroValue(), curCount);
        PipeBarrier<PIPE_V>();
        uint32_t offset = 0;
        while (offset < curCount) {
            const uint64_t linear = outLinear + offset;
            if (rowElements == 0U || block == 0U) {
                return;
            }
            const uint64_t row = linear / rowElements;
            const uint64_t rowOffset = linear - row * rowElements;
            const uint64_t owBase = rowOffset / block;
            const uint32_t c0Base = static_cast<uint32_t>(rowOffset - owBase * block);
            uint32_t blockCount = static_cast<uint32_t>(block) - c0Base;
            const uint32_t remain = curCount - offset;
            if (blockCount > remain) {
                blockCount = remain;
            }

            int64_t nIdx = 0;
            int64_t od = 0;
            int64_t c1Idx = 0;
            int64_t oh = 0;
            DecodeNdc1hwc0Row(row, validC1, nIdx, od, c1Idx, oh);
            const int64_t cBase = c1Idx * static_cast<int64_t>(block) + static_cast<int64_t>(c0Base);
            const int64_t validChannelCount = tiling_->c - cBase;
            if (validChannelCount > 0) {
                uint32_t activeCount = blockCount;
                if (static_cast<int64_t>(activeCount) > validChannelCount) {
                    activeCount = static_cast<uint32_t>(validChannelCount);
                }
                const int64_t ow = static_cast<int64_t>(owBase);
                for (uint32_t i = 0; i < activeCount; ++i) {
                    yLocal.SetValue(offset + i, ComputeValueAt(nIdx, od, oh, ow, cBase + i));
                }
            }
            offset += blockCount;
        }
    }

    __aicore__ inline void ProcessNdc1hwc0()
    {
        Ndc1hwc0LinearRangeContext range{};
        if (!InitNdc1hwc0ActiveLinearRange(range)) {
            return;
        }
        const uint64_t validEnd = range.outEnd;
        uint64_t processed = 0;
        while (range.outOffset + processed < validEnd) {
            const uint64_t remain = validEnd - (range.outOffset + processed);
            const uint32_t curCount = remain > OUTPUT_TILE_NUM ? OUTPUT_TILE_NUM : static_cast<uint32_t>(remain);
            LocalTensor<T> yLocal = yOutQue_.AllocTensor<T>();
            FillNdc1hwc0ValidTile(yLocal, range.outOffset + processed, curCount, range.block, range.rowElements,
                                  range.validC1);
            yOutQue_.EnQue(yLocal);

            LocalTensor<T> yOut = yOutQue_.DeQue<T>();
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(curCount * sizeof(T)), 0, 0, 0};
            DataCopyPad(yGm_[range.outOffset + processed], yOut, copyParams);
            yOutQue_.FreeTensor(yOut);
            processed += curCount;
        }
        if (range.validOut < tiling_->totalOut) {
            CopyOutZeroRangeByCore(range.validOut, tiling_->totalOut - range.validOut);
        }
    }

    __aicore__ inline void DecodeNdhwcRow(uint64_t row, int64_t& nIdx, int64_t& od, int64_t& oh, int64_t& ow) const
    {
        ow = static_cast<int64_t>(row % tiling_->outW);
        row /= tiling_->outW;
        oh = static_cast<int64_t>(row % tiling_->outH);
        row /= tiling_->outH;
        od = static_cast<int64_t>(row % tiling_->outD);
        nIdx = static_cast<int64_t>(row / tiling_->outD);
    }

    __aicore__ inline void CopyInVector(uint64_t inputOffset, uint32_t count)
    {
        LocalTensor<T> xLocal = xInQue_.AllocTensor<T>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(xLocal, xGm_[inputOffset], copyParams, padParams);
        xInQue_.EnQue(xLocal);
    }

    __aicore__ inline void CopyInVectorPad(uint64_t inputOffset, uint32_t validCount, uint32_t alignedCount)
    {
        LocalTensor<T> xLocal = xInQue_.AllocTensor<T>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(validCount * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{true, 0, static_cast<uint8_t>(alignedCount - validCount), NegInfValue()};
        DataCopyPad(xLocal, xGm_[inputOffset], copyParams, padParams);
        xInQue_.EnQue(xLocal);
    }

    __aicore__ inline void CopyInVectorWBlocksPad(uint64_t inputOffset, uint32_t blockCount, uint32_t validCount,
                                                  uint32_t alignedCount)
    {
        LocalTensor<T> xLocal = xInQue_.AllocTensor<T>();
        if (validCount == alignedCount && inputOffset % VectorAlignNum() == 0U) {
            const uint32_t totalCount = blockCount * validCount;
            DataCopy(xLocal, xGm_[inputOffset], totalCount);
            xInQue_.EnQue(xLocal);
            return;
        }
        DataCopyExtParams copyParams{static_cast<uint16_t>(blockCount), static_cast<uint32_t>(validCount * sizeof(T)),
                                     0, 0, 0};
        DataCopyPadExtParams<T> padParams{true, 0, static_cast<uint8_t>(alignedCount - validCount), NegInfValue()};
        DataCopyPad(xLocal, xGm_[inputOffset], copyParams, padParams);
        xInQue_.EnQue(xLocal);
    }

    __aicore__ inline void CopyInVectorWBlocksPadStride(uint64_t inputOffset, uint32_t blockCount, uint32_t validCount,
                                                        uint32_t alignedCount, uint32_t srcStrideElements)
    {
        CopyInVectorWBlocksPadStrideValue(inputOffset, blockCount, validCount, alignedCount, srcStrideElements,
                                          NegInfValue());
    }

    __aicore__ inline void CopyInVectorWBlocksPadStrideValue(uint64_t inputOffset, uint32_t blockCount,
                                                             uint32_t validCount, uint32_t alignedCount,
                                                             uint32_t srcStrideElements, T padValue)
    {
        LocalTensor<T> xLocal = xInQue_.AllocTensor<T>();
        if (srcStrideElements == 0U && validCount == alignedCount && inputOffset % VectorAlignNum() == 0U) {
            const uint32_t totalCount = blockCount * validCount;
            DataCopy(xLocal, xGm_[inputOffset], totalCount);
            xInQue_.EnQue(xLocal);
            return;
        }
        DataCopyExtParams copyParams{static_cast<uint16_t>(blockCount), static_cast<uint32_t>(validCount * sizeof(T)),
                                     static_cast<uint32_t>(srcStrideElements * sizeof(T)), 0, 0};
        DataCopyPadExtParams<T> padParams{true, 0, static_cast<uint8_t>(alignedCount - validCount), padValue};
        DataCopyPad(xLocal, xGm_[inputOffset], copyParams, padParams);
        xInQue_.EnQue(xLocal);
    }

    __aicore__ inline void CopyOutVector(uint64_t outputOffset, LocalTensor<T> yLocal, uint32_t count)
    {
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        if (count % VectorAlignNum() == 0U && outputOffset % VectorAlignNum() == 0U) {
            DataCopy(yGm_[outputOffset], yLocal, count);
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
            DataCopyPad(yGm_[outputOffset], yLocal, copyParams);
        }
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }

    __aicore__ inline void CopyOutVectorWithQueueSync(uint64_t outputOffset, LocalTensor<T> yLocal, uint32_t count,
                                                      event_t eventIdVToMte3, event_t eventIdMte3ToV)
    {
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        if (count % VectorAlignNum() == 0U && outputOffset % VectorAlignNum() == 0U) {
            DataCopy(yGm_[outputOffset], yLocal, count);
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
            DataCopyPad(yGm_[outputOffset], yLocal, copyParams);
        }
        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
    }

    __aicore__ inline void CopyOutVectorPlain(uint64_t outputOffset, LocalTensor<T> yLocal, uint32_t count)
    {
        if (count % VectorAlignNum() == 0U && outputOffset % VectorAlignNum() == 0U) {
            DataCopy(yGm_[outputOffset], yLocal, count);
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
            DataCopyPad(yGm_[outputOffset], yLocal, copyParams);
        }
    }

    __aicore__ inline void CopyOutVectorLast(uint64_t outputOffset, LocalTensor<T> yLocal, uint32_t count)
    {
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        if (count % VectorAlignNum() == 0U && outputOffset % VectorAlignNum() == 0U) {
            DataCopy(yGm_[outputOffset], yLocal, count);
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
            DataCopyPad(yGm_[outputOffset], yLocal, copyParams);
        }
    }

    __aicore__ inline bool CopyOutNdc1hwc0NdhwcAlignedCDirect(uint64_t outputOffset, LocalTensor<T> compactLocal,
                                                              LocalTensor<T> zeroLocal, uint32_t rows,
                                                              uint32_t rowStride, uint32_t cCount, uint32_t alignedC,
                                                              uint32_t block, uint32_t outW, bool compactTailIsZero,
                                                              uint32_t tileLimit)
    {
        if (rows == 0U || rowStride == 0U || cCount == 0U || alignedC < cCount || block < alignedC || outW == 0U ||
            outW > 4095U) {
            return false;
        }
        if ((static_cast<uint64_t>(alignedC) * sizeof(T)) % UB_BLOCK_BYTES != 0U) {
            return false;
        }
        const uint32_t rowElements = outW * block;
        const uint32_t totalElements = rows * rowElements;
        if (rowElements == 0U || totalElements == 0U || totalElements > tileLimit || rowStride < outW * alignedC) {
            return false;
        }
        const uint32_t blockGapBytes = static_cast<uint32_t>((block - alignedC) * sizeof(T));
        if (blockGapBytes > 65535U) {
            return false;
        }

        const bool wholeC0Written = alignedC == block && compactTailIsZero;
        if (!wholeC0Written) {
            Duplicate(zeroLocal, ZeroValue(), totalElements);
            PipeBarrier<PIPE_V>();
            CopyOutVector(outputOffset, zeroLocal, totalElements);
        }

        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyExtParams copyParams{static_cast<uint16_t>(outW), static_cast<uint32_t>(alignedC * sizeof(T)), 0,
                                     blockGapBytes, 0};
        for (uint32_t rowIdx = 0U; rowIdx < rows; ++rowIdx) {
            if (!compactTailIsZero && alignedC > cCount) {
                ZeroNdhwcAlignedCTail(compactLocal[static_cast<uint64_t>(rowIdx) * rowStride], cCount, alignedC, outW);
            }
            const uint64_t outBase = outputOffset + static_cast<uint64_t>(rowIdx) * rowElements;
            const uint64_t srcBase = static_cast<uint64_t>(rowIdx) * rowStride;
            DataCopyPad(yGm_[outBase], compactLocal[srcBase], copyParams);
        }
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        return true;
    }

    __aicore__ inline void ZeroNdhwcAlignedCTail(LocalTensor<T> rowLocal, uint32_t cCount, uint32_t alignedC,
                                                 uint32_t outW)
    {
        if (cCount >= alignedC) {
            return;
        }
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        for (uint32_t ow = 0U; ow < outW; ++ow) {
            const uint32_t base = ow * alignedC;
            for (uint32_t c0 = cCount; c0 < alignedC; ++c0) {
                rowLocal.SetValue(base + c0, ZeroValue());
            }
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopyOutSmallCWPair(uint64_t outputOffset, LocalTensor<T> yLocal, uint32_t validCount,
                                              uint32_t alignedCount)
    {
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(validCount * sizeof(T)), 0, 0, 0};
        DataCopyPad(yGm_[outputOffset], yLocal, copyParams);
        DataCopyPad(yGm_[outputOffset + validCount], yLocal[alignedCount], copyParams);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }

    __aicore__ inline void CopyOutSmallCWRow(uint64_t outputOffset, LocalTensor<T> yLocal, uint32_t validCount,
                                             uint32_t alignedCount, uint32_t countW)
    {
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        if (validCount == alignedCount && outputOffset % VectorAlignNum() == 0U &&
            (validCount * countW) % VectorAlignNum() == 0U) {
            DataCopy(yGm_[outputOffset], yLocal, validCount * countW);
        } else {
            const uint32_t alignedValidCount = AlignToVector(validCount);
            uint32_t srcStrideBlocks = 0U;
            if (alignedCount > alignedValidCount) {
                srcStrideBlocks = static_cast<uint32_t>(
                    (static_cast<uint64_t>(alignedCount - alignedValidCount) * sizeof(T)) / UB_BLOCK_BYTES);
            }
            DataCopyExtParams copyParams{static_cast<uint16_t>(countW), static_cast<uint32_t>(validCount * sizeof(T)),
                                         srcStrideBlocks, 0, 0};
            DataCopyPad(yGm_[outputOffset], yLocal, copyParams);
        }
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }

    __aicore__ inline bool CopyOutSmallCWRowCompact(uint64_t outputOffset, LocalTensor<T> srcLocal,
                                                    LocalTensor<T> scratchLocal, uint32_t validCount,
                                                    uint32_t alignedCount, uint32_t countW)
    {
        if (validCount == 0U || alignedCount == 0U || countW == 0U || validCount > alignedCount) {
            return false;
        }
        if (validCount == alignedCount && outputOffset % VectorAlignNum() == 0U &&
            (validCount * countW) % VectorAlignNum() == 0U) {
            CopyOutVector(outputOffset, srcLocal, validCount * countW);
            return true;
        }
        return false;
    }

    template <typename U>
    __aicore__ inline void CopyLocalTensor(LocalTensor<U> dstLocal, LocalTensor<U> srcLocal, uint32_t count)
    {
        Max(dstLocal, srcLocal, srcLocal, count);
    }

    __aicore__ inline bool IsOutOfRange(int64_t index, int64_t limit) const
    {
        if (index < 0) {
            return true;
        }
        return index >= limit;
    }

    __aicore__ inline bool CanUseNdhwcNoPad2x2x2Path() const
    {
        return tiling_->kD == 2 && tiling_->kH == 2 && tiling_->kW == 2 && tiling_->padFront == 0 &&
               tiling_->padTop == 0 && tiling_->padLeft == 0 && tiling_->dilationD == 1 && tiling_->dilationH == 1 &&
               tiling_->dilationW == 1;
    }

    __aicore__ inline bool CanUseNdhwcScalar2x2x2Path() const
    {
        if (tiling_->dataFormat != FORMAT_NDHWC_VALUE) {
            return false;
        }
        if (tiling_->kD != 2 || tiling_->kH != 2 || tiling_->kW != 2) {
            return false;
        }
        if (tiling_->c >= static_cast<int64_t>(VectorAlignNum())) {
            return false;
        }
        return true;
    }

    __aicore__ inline bool HasComplete2x2x2Windows() const
    {
        return (tiling_->outD - 1) * tiling_->sD + 1 < tiling_->inD &&
               (tiling_->outH - 1) * tiling_->sH + 1 < tiling_->inH &&
               (tiling_->outW - 1) * tiling_->sW + 1 < tiling_->inW;
    }

    __aicore__ inline bool CanUseNcdhwScalar2x2x2Path() const
    {
        return tiling_->dataFormat == FORMAT_NCDHW_VALUE && tiling_->kD == 2 && tiling_->kH == 2 && tiling_->kW == 2 &&
               tiling_->padFront == 0 && tiling_->padTop == 0 && tiling_->padLeft == 0 && tiling_->dilationD == 1 &&
               tiling_->dilationH == 1 && tiling_->dilationW == 1 && HasComplete2x2x2Windows();
    }

    __aicore__ inline bool CanUseNcdhwFloatStride1RowReusePath() const
    {
        if constexpr (!AscendC::Std::is_same<T, float>::value) {
            return false;
        } else {
            if (!CanUseNcdhwScalar2x2x2Path() || tiling_->sW != 1 || tiling_->outW <= 0 ||
                tiling_->outW > static_cast<int64_t>(OUTPUT_TILE_NUM)) {
                return false;
            }
            const uint64_t outW = static_cast<uint64_t>(tiling_->outW);
            return outW != 0 && tiling_->normalCoreOut >= outW && tiling_->normalCoreOut % outW == 0;
        }
    }

    __aicore__ inline bool CanUseNcdhwStride2RowVectorPath() const
    {
        if constexpr (AscendC::Std::is_same<T, bfloat16_t>::value) {
            return false;
        } else {
            uint32_t outW = 0U;
            if (!InitNcdhwStride2VectorWidth(outW)) {
                return false;
            }
            const uint32_t inputRowCount = outW * 2U;
            const uint32_t alignedInputRowCount = AlignToVector(inputRowCount);
            const uint32_t copyCount = inputRowCount * 2U;
            const uint32_t alignedCopyCount = AlignToVector(copyCount);
            if (alignedInputRowCount > OUTPUT_TILE_NUM || alignedCopyCount > INPUT_TILE_NUM || outW > OUTPUT_TILE_NUM) {
                return false;
            }
            return tiling_->normalCoreOut >= outW && tiling_->normalCoreOut % outW == 0;
        }
    }

    __aicore__ inline bool CanUseNcdhwStride2WholeDDirectPath() const
    {
        if constexpr (AscendC::Std::is_same<T, bfloat16_t>::value) {
            return false;
        } else {
            if (!CanUseNcdhwScalar2x2x2Path() || tiling_->sD != 2 || tiling_->sH != 2 || tiling_->sW != 2 ||
                tiling_->outD != 2 || tiling_->inD < 4 || tiling_->outW <= 0 || tiling_->outH <= 0 || tiling_->c <= 0) {
                return false;
            }
            const uint32_t outW = static_cast<uint32_t>(tiling_->outW);
            const uint32_t outH = static_cast<uint32_t>(tiling_->outH);
            const uint32_t inputRowCount = outW * 2U;
            const uint32_t alignedInputRowCount = AlignToVector(inputRowCount);
            const uint32_t alignedOutputRowCount = AlignToVector(outW);
            const uint32_t inputNeed = alignedInputRowCount * outH * 4U;
            const uint32_t accNeed = alignedInputRowCount * outH * 2U;
            const uint32_t outNeed = alignedOutputRowCount * outH * 2U;
            const uint32_t compactOut = outW * outH * 2U;
            if (inputNeed == 0U || inputNeed > INPUT_TILE_NUM || accNeed > OUTPUT_TILE_NUM ||
                outNeed > OUTPUT_TILE_NUM || compactOut > OUTPUT_TILE_NUM) {
                return false;
            }
            const uint64_t channelOut = static_cast<uint64_t>(outW) * outH * 2U;
            return channelOut > 0U && tiling_->normalCoreOut >= channelOut && tiling_->normalCoreOut % channelOut == 0U;
        }
    }

    __aicore__ inline bool CanUseNcdhwStride2DPlaneDirectPath() const
    {
        if constexpr (AscendC::Std::is_same<T, bfloat16_t>::value) {
            return false;
        } else {
            if (!CanUseNcdhwScalar2x2x2Path() || tiling_->sD != 2 || tiling_->sH != 2 || tiling_->sW != 2 ||
                tiling_->outD <= 2 || tiling_->outW <= 0 || tiling_->outH <= 0 || tiling_->c <= 0) {
                return false;
            }
            const uint32_t outW = static_cast<uint32_t>(tiling_->outW);
            const uint32_t outH = static_cast<uint32_t>(tiling_->outH);
            const uint32_t inputRowCount = outW * 2U;
            const uint32_t alignedInputRowCount = AlignToVector(inputRowCount);
            const uint32_t alignedOutputRowCount = AlignToVector(outW);
            const uint32_t blockRows = NcdhwStride2MaxHBlockRows(outW);
            if (blockRows == 0U) {
                return false;
            }
            const uint32_t inputNeed = alignedInputRowCount * blockRows * 2U;
            const uint32_t accNeed = alignedInputRowCount * blockRows;
            const uint32_t outNeed = alignedOutputRowCount * blockRows;
            const uint32_t compactOut = outW * blockRows;
            if (inputNeed == 0U || inputNeed > INPUT_TILE_NUM || accNeed > OUTPUT_TILE_NUM ||
                outNeed > OUTPUT_TILE_NUM || compactOut > OUTPUT_TILE_NUM) {
                return false;
            }
            const uint64_t planeOut = static_cast<uint64_t>(outW) * outH;
            return planeOut > 0U && tiling_->normalCoreOut >= planeOut && tiling_->normalCoreOut % planeOut == 0U;
        }
    }

    __aicore__ inline bool CanUseNcdhwStride2MicroHBlockPath() const
    {
        if constexpr (!AscendC::Std::is_same<T, half>::value) {
            return false;
        } else {
            uint32_t outW = 0U;
            if (!InitNcdhwStride2VectorWidth(outW)) {
                return false;
            }
            const uint32_t maxRows = NcdhwStride2MaxHBlockRows(outW);
            return maxRows > 0U && tiling_->normalCoreOut >= outW && tiling_->normalCoreOut % outW == 0U;
        }
    }

    __aicore__ inline bool InitNcdhwStride2VectorWidth(uint32_t& outW) const
    {
        if (!CanUseNcdhwScalar2x2x2Path() || tiling_->sD != 2 || tiling_->sH != 2 || tiling_->sW != 2 ||
            tiling_->outW <= 0) {
            return false;
        }
        outW = static_cast<uint32_t>(tiling_->outW);
        return outW >= 16U;
    }

    __aicore__ inline bool CanUseNdhwcSmallCWStride1PairPath() const
    {
        return CanUseNdhwcNoPad2x2x2Path() && tiling_->sW == 1 && tiling_->c > 1 &&
               tiling_->c < static_cast<int64_t>(VectorAlignNum());
    }

    __aicore__ inline bool CanUseNdhwcBf16SmallCVectorPath() const
    {
        if constexpr (!AscendC::Std::is_same<T, bfloat16_t>::value) {
            return false;
        }
        return tiling_->dataFormat == FORMAT_NDHWC_VALUE && tiling_->c > 1 &&
               tiling_->c < static_cast<int64_t>(VectorAlignNum()) && tiling_->sW == 1 && CanUseNdhwcNoPad2x2x2Path() &&
               HasComplete2x2x2Windows();
    }

    __aicore__ inline bool CanUseNdhwcHalfC8Stride2ScalarRowPath() const
    {
        if constexpr (!AscendC::Std::is_same<T, half>::value) {
            return false;
        }
        return tiling_->dataFormat == FORMAT_NDHWC_VALUE && tiling_->c == 8 && tiling_->sD == 2 && tiling_->sH == 2 &&
               tiling_->sW == 2 && CanUseNdhwcNoPad2x2x2Path() && HasComplete2x2x2Windows();
    }

    __aicore__ inline bool CanUseNdhwcFloatC3NoPad2x2x2DirectPath() const
    {
        if constexpr (!AscendC::Std::is_same<T, float>::value) {
            return false;
        }
        return tiling_->dataFormat == FORMAT_NDHWC_VALUE && tiling_->c == 3 && tiling_->sW == 1 &&
               CanUseNdhwcNoPad2x2x2Path() && HasComplete2x2x2Windows();
    }

    __aicore__ inline bool InitNdc1hwc0RowTile(uint32_t dataFormat, uint64_t& block, uint64_t& rowElements) const
    {
        if (tiling_->dataFormat != dataFormat) {
            return false;
        }
        block = Ndc1hwc0Block();
        if (block == 0U || tiling_->outW <= 0 || tiling_->c <= 0) {
            return false;
        }
        rowElements = static_cast<uint64_t>(tiling_->outW) * block;
        return rowElements != 0U && rowElements <= OUTPUT_TILE_NUM;
    }

    __aicore__ inline bool CanUseNdc1hwc0RowVectorPath() const
    {
        uint64_t block = 0U, rowElements = 0U;
        if (!InitNdc1hwc0RowTile(FORMAT_NDHWC_VALUE, block, rowElements)) {
            return false;
        }
        if (tiling_->sW <= 0 || tiling_->dilationW <= 0 || tiling_->kW <= 0) {
            return false;
        }
        if (tiling_->outW * static_cast<int64_t>(AlignToVector(static_cast<uint32_t>(block))) > INPUT_TILE_NUM) {
            return false;
        }
        return true;
    }

    __aicore__ inline bool CanUseNdc1hwc0NcdhwRowVectorPath() const
    {
        uint64_t block = 0U, rowElements = 0U;
        if (!InitNdc1hwc0RowTile(FORMAT_NCDHW_VALUE, block, rowElements)) {
            return false;
        }
        if (tiling_->sW <= 0 || tiling_->dilationW <= 0 || tiling_->kW <= 0) {
            return false;
        }
        uint64_t maxSpanW = 0U;
        for (int64_t kw = 0; kw < tiling_->kW; ++kw) {
            int64_t first = 0;
            while (first < tiling_->outW && first * tiling_->sW + kw * tiling_->dilationW - tiling_->padLeft < 0) {
                ++first;
            }
            int64_t last = tiling_->outW - 1;
            while (last >= first && last * tiling_->sW + kw * tiling_->dilationW - tiling_->padLeft >= tiling_->inW) {
                --last;
            }
            if (last < first) {
                continue;
            }
            const uint64_t spanW = static_cast<uint64_t>(last - first) * static_cast<uint64_t>(tiling_->sW) + 1U;
            if (maxSpanW < spanW) {
                maxSpanW = spanW;
            }
        }
        if (maxSpanW == 0U || maxSpanW > static_cast<uint64_t>(tiling_->inW)) {
            return false;
        }
        const uint64_t inputTileNeed = static_cast<uint64_t>(block) *
                                       static_cast<uint64_t>(AlignToVector(static_cast<uint32_t>(maxSpanW)));
        if (inputTileNeed > INPUT_TILE_NUM) {
            return false;
        }
        return tiling_->normalCoreOut >= rowElements && tiling_->normalCoreOut % rowElements == 0U;
    }

    __aicore__ inline bool CanUseNdhwcStride2WBlockVectorPath() const
    {
        if (tiling_->dataFormat != FORMAT_NDHWC_VALUE) {
            return false;
        }
        if (!MatchesPoolSpec(2, 2, 2, 2, 2, 2, 1, 1, 1, 0, 0, 0)) {
            return false;
        }
        if (tiling_->c < static_cast<int64_t>(VectorAlignNum()) || tiling_->c <= 0) {
            return false;
        }
        const uint64_t cCount = static_cast<uint64_t>(tiling_->c);
        if (static_cast<uint32_t>(tiling_->c) % VectorAlignNum() != 0U || cCount * 2U > OUTPUT_TILE_NUM ||
            cCount * 4U > INPUT_TILE_NUM) {
            return false;
        }
        return HasComplete2x2x2Windows() && tiling_->normalCoreOut >= cCount && tiling_->normalCoreOut % cCount == 0U;
    }

    __aicore__ inline bool CanUseNdhwcStride2AlignedCPath() const
    {
        if (tiling_->dataFormat != FORMAT_NDHWC_VALUE || tiling_->outW <= 0 || tiling_->c <= 0) {
            return false;
        }
        const uint32_t cCount = static_cast<uint32_t>(tiling_->c);
        if (!HasNdc1hwc0Stride2Pool2Spec() || tiling_->c < static_cast<int64_t>(VectorAlignNum()) ||
            cCount % VectorAlignNum() != 0U) {
            return false;
        }
        return true;
    }

    __aicore__ inline bool CanUseNdhwcStride2SingleRowFusedDPath() const
    {
        if (!CanUseNdhwcStride2AlignedCPath()) {
            return false;
        }
        const uint32_t cCount = static_cast<uint32_t>(tiling_->c);
        const uint32_t outputRowCount = static_cast<uint32_t>(tiling_->outW) * cCount;
        const uint32_t inputRowCount = outputRowCount * 2U;
        const uint32_t copyCount = inputRowCount * 2U;
        const uint32_t alignedCopyCount = AlignToVector(copyCount);
        if (outputRowCount == 0U || tiling_->normalCoreOut != outputRowCount || outputRowCount > OUTPUT_TILE_NUM ||
            alignedCopyCount * 2U > INPUT_TILE_NUM) {
            return false;
        }
        return HasComplete2x2x2Windows();
    }

    __aicore__ inline bool CanUseNdhwcStride2FullRowDirectPath() const
    {
        if (!CanUseNdhwcStride2LargeHBlockBuffers()) {
            return false;
        }
        const uint64_t rowElements = static_cast<uint64_t>(tiling_->outW) * static_cast<uint64_t>(tiling_->c);
        return rowElements != 0U && tiling_->normalCoreOut >= rowElements * 2U &&
               tiling_->normalCoreOut % rowElements == 0U;
    }

    __aicore__ inline bool CanUseNdhwcFloatStride2CompactHBlockPath() const
    {
        if constexpr (!AscendC::Std::is_same<T, float>::value) {
            return false;
        }
        return CanUseNdhwcStride2FullRowDirectPath();
    }

    __aicore__ inline bool CanUseNdhwcStride2LargeHBlockBuffers() const
    {
        if (!CanUseNdhwcStride2AlignedCPath()) {
            return false;
        }
        const uint32_t cCount = static_cast<uint32_t>(tiling_->c);
        const uint64_t rowElements = static_cast<uint64_t>(tiling_->outW) * static_cast<uint64_t>(cCount);
        if (rowElements == 0U || tiling_->normalCoreOut < rowElements * NDHWC_STRIDE2_HBLOCK_ROWS ||
            tiling_->normalCoreOut % rowElements != 0U) {
            return false;
        }
        if (rowElements * NDHWC_STRIDE2_HBLOCK_ROWS > NDHWC_STRIDE2_LARGE_OUTPUT_TILE_NUM ||
            rowElements * 4U * NDHWC_STRIDE2_HBLOCK_ROWS > NDHWC_STRIDE2_LARGE_INPUT_TILE_NUM) {
            return false;
        }
        return HasComplete2x2x2Windows();
    }

    __aicore__ inline bool CanUseNdhwcStride2TwoRowDThenWPath() const
    {
        if (!CanUseNdhwcStride2AlignedCPath()) {
            return false;
        }
        const uint32_t cCount = static_cast<uint32_t>(tiling_->c);
        const uint64_t rowElements = static_cast<uint64_t>(tiling_->outW) * static_cast<uint64_t>(cCount);
        if (rowElements == 0U || tiling_->normalCoreOut < rowElements * NDHWC_STRIDE2_DTHENW_ROWS ||
            tiling_->normalCoreOut % rowElements != 0U) {
            return false;
        }
        if (rowElements * NDHWC_STRIDE2_DTHENW_ROWS > OUTPUT_TILE_NUM ||
            rowElements * 4U * NDHWC_STRIDE2_DTHENW_ROWS > INPUT_TILE_NUM) {
            return false;
        }
        return HasComplete2x2x2Windows();
    }

    __aicore__ inline bool CanUseNdc1hwc0NcdhwRowScalarPath() const
    {
        uint64_t block = 0U, rowElements = 0U;
        if (!InitNdc1hwc0RowTile(FORMAT_NCDHW_VALUE, block, rowElements)) {
            return false;
        }
        return tiling_->normalCoreOut >= rowElements && tiling_->normalCoreOut % rowElements == 0U;
    }

    __aicore__ inline bool CanUseNdc1hwc0NcdhwK1DirectPath() const
    {
        if (tiling_->dataFormat != FORMAT_NCDHW_VALUE) {
            return false;
        }
        if (tiling_->kD != 1 || tiling_->kH != 1 || tiling_->kW != 1 || tiling_->sD != 1 || tiling_->sH != 1 ||
            tiling_->sW != 1 || tiling_->dilationD != 1 || tiling_->dilationH != 1 || tiling_->dilationW != 1 ||
            tiling_->padFront != 0 || tiling_->padTop != 0 || tiling_->padLeft != 0) {
            return false;
        }
        if (tiling_->outD > tiling_->inD || tiling_->outH > tiling_->inH || tiling_->outW > tiling_->inW) {
            return false;
        }
        return CanUseNdc1hwc0NcdhwRowScalarPath();
    }

    __aicore__ inline bool CanFitNdc1hwc0NcdhwK1FullC1Plane(uint64_t block, uint32_t outW, uint32_t outH,
                                                            uint32_t alignedPlane, uint32_t rowElements,
                                                            uint32_t outputCount, uint32_t outputBlock,
                                                            uint32_t inputNeed, uint32_t offsetNeed,
                                                            uint64_t channelStride) const
    {
        return (block == 16U || block == 32U) && rowElements > 0U && outputCount > 0U && outputBlock > 0U &&
               outputBlock <= OUTPUT_TILE_NUM && inputNeed > 0U && inputNeed + alignedPlane <= INPUT_TILE_NUM &&
               alignedPlane > outW * outH && alignedPlane - outW * outH <= 255U && outputCount <= OUTPUT_TILE_NUM &&
               inputNeed <= INPUT_TILE_NUM && offsetNeed <= OUTPUT_TILE_NUM &&
               channelStride >= static_cast<uint64_t>(outW) * outH && tiling_->normalCoreOut >= outputCount &&
               tiling_->normalCoreOut % outputCount == 0U;
    }

    __aicore__ inline bool CanUseNdc1hwc0NcdhwK1FullC1PlanePath() const
    {
        if (!CanUseNdc1hwc0NcdhwK1DirectPath() || tiling_->outW <= 0 || tiling_->outH <= 0 || tiling_->outD <= 0 ||
            tiling_->c <= 0 || tiling_->outW != tiling_->inW) {
            return false;
        }
        const uint64_t block = Ndc1hwc0Block();
        const uint64_t validC1 = Ndc1hwc0ValidC1(block);
        if (block == 0U || validC1 == 0U || validC1 > 8U) {
            return false;
        }
        const uint32_t outW = static_cast<uint32_t>(tiling_->outW);
        const uint32_t outH = static_cast<uint32_t>(tiling_->outH);
        const uint32_t alignedPlane = AlignToVector(outW * outH + 1U);
        const uint32_t rowElements = static_cast<uint32_t>(static_cast<uint64_t>(outW) * block);
        const uint32_t planeRows = static_cast<uint32_t>(validC1) * outH;
        const uint32_t outputCount = planeRows * rowElements;
        const uint32_t outputBlock = outH * outW * static_cast<uint32_t>(block);
        const uint32_t inputNeed = static_cast<uint32_t>(tiling_->c) * alignedPlane;
        const uint32_t offsetNeed = Ndc1hwc0GatherTempOffset(outputCount);
        const uint64_t channelStride = static_cast<uint64_t>(tiling_->inD) * static_cast<uint64_t>(tiling_->inH) *
                                       static_cast<uint64_t>(tiling_->inW);
        return CanFitNdc1hwc0NcdhwK1FullC1Plane(block, outW, outH, alignedPlane, rowElements, outputCount, outputBlock,
                                                inputNeed, offsetNeed, channelStride);
    }

    __aicore__ inline void ProcessNdc1hwc0NcdhwK1FullC1PlaneStep(uint64_t& cur, uint64_t outEnd, uint64_t block,
                                                                 uint64_t validC1, uint32_t rowElements,
                                                                 uint32_t outputCount, uint32_t planeValid,
                                                                 uint32_t maxGroupPlanes, uint32_t outH, uint32_t outW)
    {
        if (rowElements == 0U || outputCount == 0U) {
            cur = outEnd;
            return;
        }
        if (cur % outputCount != 0U || outEnd - cur < outputCount) {
            const uint64_t row = cur / rowElements;
            ProcessNdc1hwc0NcdhwK1DirectByRow(row, cur, block, validC1);
            cur += rowElements - (cur - row * static_cast<uint64_t>(rowElements));
            return;
        }
        const Ndc1hwc0PlaneGroupContext context = GetNdc1hwc0PlaneGroupContext(cur, outEnd, outputCount,
                                                                               maxGroupPlanes);
        if (context.groupPlanes > 1U && ProcessNdc1hwc0NcdhwK1FullC1DGroupTile(
                                            cur, context.nIdx, context.od, context.groupPlanes,
                                            static_cast<uint32_t>(validC1), static_cast<uint32_t>(block), planeValid)) {
            cur += static_cast<uint64_t>(context.groupPlanes) * outputCount;
            return;
        }
        ProcessNdc1hwc0NcdhwK1FullC1PlaneTile(cur, context.nIdx, context.od, static_cast<uint32_t>(validC1),
                                              static_cast<uint32_t>(block), outH, outW);
        cur += outputCount;
    }

    __aicore__ inline void ProcessNdc1hwc0NcdhwK1FullC1Plane()
    {
        uint64_t block = 0U, validC1 = 0U, validOut = 0U;
        if (!InitNdc1hwc0ValidOutput(block, validC1, validOut)) {
            return;
        }
        const uint32_t outW = static_cast<uint32_t>(tiling_->outW);
        const uint32_t outH = static_cast<uint32_t>(tiling_->outH);
        const uint32_t rowElements = static_cast<uint32_t>(static_cast<uint64_t>(outW) * block);
        const uint32_t outputCount = static_cast<uint32_t>(validC1) * outH * rowElements;
        const uint32_t planeValid = outH * outW;
        const uint32_t maxGroupPlanes = K1FullC1PlaneMaxDGroup(static_cast<uint32_t>(validC1),
                                                               static_cast<uint32_t>(block), planeValid);
        uint64_t outOffset = 0U, outCount = 0U, outEnd = 0U;
        GetNdc1hwc0ValidCoreRange(validOut, rowElements, outOffset, outCount, outEnd);
        uint64_t cur = outOffset;
        while (cur < outEnd) {
            ProcessNdc1hwc0NcdhwK1FullC1PlaneStep(cur, outEnd, block, validC1, rowElements, outputCount, planeValid,
                                                  maxGroupPlanes, outH, outW);
        }
        if (validOut < tiling_->totalOut) {
            CopyOutZeroRangeByCore(validOut, tiling_->totalOut - validOut);
        }
    }

    __aicore__ inline uint32_t K1FullC1PlaneMaxDGroup(uint32_t validC1, uint32_t block, uint32_t planeValid) const
    {
        if (validC1 == 0U || block != 16U || planeValid == 0U) {
            return 1U;
        }
        uint32_t maxGroup = OUTPUT_TILE_NUM / (planeValid * block);
        if (maxGroup == 0U) {
            return 1U;
        }
        if (maxGroup > static_cast<uint32_t>(tiling_->outD)) {
            maxGroup = static_cast<uint32_t>(tiling_->outD);
        }
        const uint32_t channelCount = static_cast<uint32_t>(tiling_->c);
        while (maxGroup > 1U) {
            const uint32_t groupValid = maxGroup * planeValid;
            const uint32_t alignedGroup = AlignToVector(groupValid + 1U);
            const uint32_t inputNeed = channelCount * alignedGroup + alignedGroup;
            const uint32_t transWrite = alignedGroup * block;
            if (inputNeed <= INPUT_TILE_NUM && transWrite <= OUTPUT_TILE_NUM) {
                break;
            }
            --maxGroup;
        }
        return maxGroup == 0U ? 1U : maxGroup;
    }

    __aicore__ inline bool ProcessNdc1hwc0NcdhwK1FullC1DGroupChannel(uint64_t outputOffset, uint32_t c1,
                                                                     uint32_t groupPlanes, uint32_t validC1,
                                                                     uint32_t block, uint32_t outputBlock,
                                                                     uint32_t alignedGroup, uint32_t channelCount,
                                                                     uint32_t zeroRowOffset, LocalTensor<T> xLocal,
                                                                     LocalTensor<T> outLocal)
    {
        const uint32_t cBase = c1 * block;
        uint32_t activeChannels = 0U;
        if (cBase < channelCount) {
            activeChannels = channelCount - cBase;
            if (activeChannels > block) {
                activeChannels = block;
            }
        }
        if (activeChannels == 0U) {
            Duplicate(outLocal, ZeroValue(), groupPlanes * outputBlock);
            PipeBarrier<PIPE_V>();
        } else if (!TransposeNdc1hwc0C0PlaneBlock(outLocal, xLocal[static_cast<uint64_t>(cBase) * alignedGroup],
                                                  xLocal[zeroRowOffset], alignedGroup, activeChannels, block)) {
            return false;
        }
        CopyOutNdc1hwc0K1C1DGroup(outputOffset + static_cast<uint64_t>(c1) * outputBlock, outLocal, outputBlock,
                                  groupPlanes, validC1);
        return true;
    }

    __aicore__ inline bool ProcessNdc1hwc0NcdhwK1FullC1DGroupTile(uint64_t outputOffset, int64_t nIdx, int64_t od,
                                                                  uint32_t groupPlanes, uint32_t validC1,
                                                                  uint32_t block, uint32_t planeValid)
    {
        if (groupPlanes <= 1U || block != 16U || validC1 == 0U || planeValid == 0U) {
            return false;
        }
        const uint32_t groupValid = groupPlanes * planeValid;
        const uint32_t alignedGroup = AlignToVector(groupValid + 1U);
        const uint32_t outputBlock = planeValid * block;
        const uint32_t transWrite = alignedGroup * block;
        if (outputBlock == 0U || transWrite > OUTPUT_TILE_NUM || groupPlanes * outputBlock > OUTPUT_TILE_NUM) {
            return false;
        }
        const uint32_t channelCount = static_cast<uint32_t>(tiling_->c);
        const uint32_t zeroRowOffset = channelCount * alignedGroup;
        if (zeroRowOffset + alignedGroup > INPUT_TILE_NUM) {
            return false;
        }
        const uint64_t channelStride = static_cast<uint64_t>(tiling_->inD) * static_cast<uint64_t>(tiling_->inH) *
                                       static_cast<uint64_t>(tiling_->inW);
        if (channelStride < static_cast<uint64_t>(groupValid)) {
            return false;
        }

        CopyInVectorWBlocksPadStrideValue(InputOffset(nIdx, od, 0, 0, 0), channelCount, groupValid, alignedGroup,
                                          static_cast<uint32_t>(channelStride - static_cast<uint64_t>(groupValid)),
                                          ZeroValue());
        LocalTensor<T> xLocal = xInQue_.DeQue<T>();
        Duplicate(xLocal[zeroRowOffset], ZeroValue(), alignedGroup);
        PipeBarrier<PIPE_V>();

        LocalTensor<T> outLocal = maskBuf_.Get<T>();
        for (uint32_t c1 = 0; c1 < validC1; ++c1) {
            if (!ProcessNdc1hwc0NcdhwK1FullC1DGroupChannel(outputOffset, c1, groupPlanes, validC1, block, outputBlock,
                                                           alignedGroup, channelCount, zeroRowOffset, xLocal,
                                                           outLocal)) {
                xInQue_.FreeTensor(xLocal);
                return false;
            }
        }
        xInQue_.FreeTensor(xLocal);
        return true;
    }

    __aicore__ inline void CopyOutNdc1hwc0K1C1DGroup(uint64_t outputOffset, LocalTensor<T> srcLocal,
                                                     uint32_t outputBlock, uint32_t groupPlanes, uint32_t validC1)
    {
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyExtParams copyParams{static_cast<uint16_t>(groupPlanes), static_cast<uint32_t>(outputBlock * sizeof(T)),
                                     0, static_cast<uint32_t>((validC1 - 1U) * outputBlock * sizeof(T)), 0};
        DataCopyPad(yGm_[outputOffset], srcLocal, copyParams);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }

    __aicore__ inline void ProcessNdc1hwc0NcdhwK1FullC1PlaneTile(uint64_t outputOffset, int64_t nIdx, int64_t od,
                                                                 uint32_t validC1, uint32_t block, uint32_t outH,
                                                                 uint32_t outW)
    {
        const uint32_t planeValid = outH * outW;
        const uint32_t alignedPlane = AlignToVector(planeValid + 1U);
        const uint32_t channelCount = static_cast<uint32_t>(tiling_->c);
        const uint32_t rowElements = outW * block;
        const uint32_t outputCount = validC1 * outH * rowElements;
        const uint32_t outputBlock = planeValid * block;
        const uint64_t channelStride = static_cast<uint64_t>(tiling_->inD) * static_cast<uint64_t>(tiling_->inH) *
                                       static_cast<uint64_t>(tiling_->inW);
        CopyInVectorWBlocksPadStrideValue(InputOffset(nIdx, od, 0, 0, 0), channelCount, planeValid, alignedPlane,
                                          static_cast<uint32_t>(channelStride - static_cast<uint64_t>(planeValid)),
                                          ZeroValue());
        LocalTensor<T> xLocal = xInQue_.DeQue<T>();
        LocalTensor<T> outLocal = maskBuf_.Get<T>();
        if (TransposeNdc1hwc0NcdhwK1FullC1Plane(outputOffset, xLocal, outLocal, validC1, block, alignedPlane,
                                                outputBlock)) {
            xInQue_.FreeTensor(xLocal);
            return;
        }
        LocalTensor<T> scratchLocal = tmpBuf_.Get<T>();
        LocalTensor<uint32_t> offsetLocal = scratchLocal.template ReinterpretCast<uint32_t>();
        InitNdc1hwc0NcdhwK1FullC1PlaneOffsets(offsetLocal, validC1, block, outH, outW, alignedPlane, planeValid);
        GatherNdc1hwc0CompactTile(outLocal, xLocal, offsetLocal, outputCount);
        xInQue_.FreeTensor(xLocal);
        CopyOutVector(outputOffset, outLocal, outputCount);
    }

    __aicore__ inline bool TransposeNdc1hwc0NcdhwK1FullC1Plane(uint64_t outputOffset, LocalTensor<T> xLocal,
                                                               LocalTensor<T> outLocal, uint32_t validC1,
                                                               uint32_t block, uint32_t alignedPlane,
                                                               uint32_t outputBlock)
    {
        if (block != 16U || alignedPlane == 0U || alignedPlane % VectorAlignNum() != 0U || outputBlock == 0U ||
            outputBlock > OUTPUT_TILE_NUM) {
            return false;
        }
        const uint32_t totalOutput = validC1 * outputBlock;
        const uint32_t transWriteCount = (validC1 - 1U) * outputBlock + alignedPlane * static_cast<uint32_t>(block);
        if (totalOutput == 0U || totalOutput > OUTPUT_TILE_NUM || transWriteCount > OUTPUT_TILE_NUM) {
            return false;
        }
        const uint32_t channelCount = static_cast<uint32_t>(tiling_->c);
        const uint32_t zeroRowOffset = channelCount * alignedPlane;
        if (zeroRowOffset + alignedPlane > INPUT_TILE_NUM) {
            return false;
        }
        Duplicate(xLocal[zeroRowOffset], ZeroValue(), alignedPlane);
        PipeBarrier<PIPE_V>();

        for (uint32_t c1 = 0; c1 < validC1; ++c1) {
            const uint32_t cBase = c1 * block;
            uint32_t activeChannels = 0U;
            if (cBase < channelCount) {
                activeChannels = channelCount - cBase;
                if (activeChannels > block) {
                    activeChannels = block;
                }
            }
            if (activeChannels == 0U) {
                Duplicate(outLocal[static_cast<uint64_t>(c1) * outputBlock], ZeroValue(), outputBlock);
                PipeBarrier<PIPE_V>();
            } else if (!TransposeNdc1hwc0C0PlaneBlock(outLocal[static_cast<uint64_t>(c1) * outputBlock],
                                                      xLocal[static_cast<uint64_t>(cBase) * alignedPlane],
                                                      xLocal[zeroRowOffset], alignedPlane, activeChannels, block)) {
                return false;
            }
        }
        CopyOutVector(outputOffset, outLocal, totalOutput);
        return true;
    }

    __aicore__ inline bool TransposeNdc1hwc0C0PlaneBlock(LocalTensor<T> dstLocal, LocalTensor<T> srcLocal,
                                                         LocalTensor<T> zeroLocal, uint32_t alignedPlane,
                                                         uint32_t activeRows, uint32_t block)
    {
        if (block != 16U || activeRows == 0U || activeRows > block || alignedPlane == 0U ||
            alignedPlane % VectorAlignNum() != 0U) {
            return false;
        }
        uint64_t srcAddrList[16];
        uint64_t dstAddrList[16];
        for (uint32_t i = 0; i < 16U; ++i) {
            if (i < activeRows) {
                srcAddrList[i] = reinterpret_cast<uint64_t>(
                    srcLocal[static_cast<uint64_t>(i) * alignedPlane].GetPhyAddr());
            } else {
                srcAddrList[i] = reinterpret_cast<uint64_t>(zeroLocal.GetPhyAddr());
            }
            if constexpr (AscendC::Std::is_same<T, float>::value) {
                dstAddrList[i] = reinterpret_cast<uint64_t>(
                    dstLocal[static_cast<uint64_t>(i / 2U) * block + (i % 2U) * 8U].GetPhyAddr());
            } else {
                dstAddrList[i] = reinterpret_cast<uint64_t>(dstLocal[static_cast<uint64_t>(i) * block].GetPhyAddr());
            }
        }
        TransDataTo5HDParams transDataParams;
        transDataParams.dstHighHalf = false;
        transDataParams.srcHighHalf = false;
        transDataParams.repeatTimes = alignedPlane / VectorAlignNum();
        if (transDataParams.repeatTimes == 1U) {
            transDataParams.srcRepStride = 0;
            transDataParams.dstRepStride = 0;
        } else {
            transDataParams.srcRepStride = 1;
            transDataParams.dstRepStride = block;
        }
        TransDataTo5HD<T>(dstAddrList, srcAddrList, transDataParams);
        PipeBarrier<PIPE_V>();
        return true;
    }

    __aicore__ inline void InitNdc1hwc0NcdhwK1FullC1PlaneOffsets(LocalTensor<uint32_t> offsetLocal, uint32_t validC1,
                                                                 uint32_t block, uint32_t outH, uint32_t outW,
                                                                 uint32_t alignedPlane, uint32_t zeroIndex)
    {
        LocalTensor<int32_t> offsetI32 = offsetLocal.template ReinterpretCast<int32_t>();
        const int32_t zeroOffset = static_cast<int32_t>(zeroIndex * sizeof(T));
        for (uint32_t c1 = 0; c1 < validC1; ++c1) {
            for (uint32_t oh = 0; oh < outH; ++oh) {
                const uint32_t outRowBase = (c1 * outH + oh) * outW * block;
                const uint32_t srcRowBase = oh * outW;
                for (uint32_t ow = 0; ow < outW; ++ow) {
                    const uint32_t outBase = outRowBase + ow * block;
                    for (uint32_t c0 = 0; c0 < block; ++c0) {
                        const uint32_t cIdx = c1 * block + c0;
                        int32_t srcOffset = zeroOffset;
                        if (cIdx < static_cast<uint32_t>(tiling_->c)) {
                            srcOffset = static_cast<int32_t>((cIdx * alignedPlane + srcRowBase + ow) * sizeof(T));
                        }
                        offsetI32.SetValue(outBase + c0, srcOffset);
                    }
                }
            }
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline bool CanUseNdc1hwc0NcdhwK1DirectGroupPath() const
    {
        if (!CanUseNdc1hwc0NcdhwK1DirectPath() || tiling_->outW <= 0 || tiling_->outH <= 0) {
            return false;
        }
        const uint64_t block = Ndc1hwc0Block();
        const uint64_t validC1 = Ndc1hwc0ValidC1(block);
        if (block == 0U || validC1 == 0U) {
            return false;
        }
        const uint32_t outW = static_cast<uint32_t>(tiling_->outW);
        const uint32_t alignedW = AlignToVector(outW);
        const uint32_t cCount = static_cast<uint32_t>(
            block < static_cast<uint64_t>(tiling_->c) ? block : static_cast<uint64_t>(tiling_->c));
        const uint32_t compactStride = cCount * alignedW;
        const uint32_t rowElements = static_cast<uint32_t>(static_cast<uint64_t>(outW) * block);
        const uint32_t offsetElements = Ndc1hwc0GatherTempOffset(rowElements);
        const uint32_t maxRows = Ndc1hwc0MaxCompactTileRows(static_cast<uint32_t>(tiling_->outH), compactStride,
                                                            rowElements);
        return rowElements > 0U && compactStride > 0U && rowElements <= OUTPUT_TILE_NUM &&
               compactStride + 1U <= INPUT_TILE_NUM && offsetElements <= OUTPUT_TILE_NUM &&
               compactStride + 1U <= OUTPUT_TILE_NUM && alignedW <= INPUT_TILE_NUM && maxRows > 1U &&
               CanUseNdc1hwc0NcdhwRowGather(cCount, static_cast<uint32_t>(block), outW, alignedW);
    }

    __aicore__ inline void ProcessNdc1hwc0NcdhwK1DirectGroupStep(uint64_t& cur, uint64_t outEnd, uint64_t block,
                                                                 uint64_t validC1, uint32_t rowElements, uint32_t outW)
    {
        if (rowElements == 0U) {
            cur = outEnd;
            return;
        }
        const uint64_t row = cur / rowElements;
        const uint64_t rowOffset = cur - row * static_cast<uint64_t>(rowElements);
        if (rowOffset != 0U || outEnd - cur < rowElements) {
            ProcessNdc1hwc0NcdhwK1DirectByRow(row, cur, block, validC1);
            cur += rowElements - rowOffset;
            return;
        }
        int64_t nIdx = 0;
        int64_t od = 0;
        int64_t c1Idx = 0;
        int64_t oh = 0;
        DecodeNdc1hwc0Row(row, validC1, nIdx, od, c1Idx, oh);
        const int64_t cBase = c1Idx * static_cast<int64_t>(block);
        const int64_t activeChannels = Ndc1hwc0ActiveChannels(cBase, block);
        if (activeChannels <= 0) {
            ProcessNdc1hwc0NcdhwK1DirectByRow(row, cur, block, validC1);
            cur += rowElements;
            return;
        }
        uint32_t rows = static_cast<uint32_t>(tiling_->outH - oh);
        const uint64_t remainRows = (outEnd - cur) / rowElements;
        if (static_cast<uint64_t>(rows) > remainRows) {
            rows = static_cast<uint32_t>(remainRows);
        }
        const uint32_t alignedW = AlignToVector(outW);
        rows = Ndc1hwc0MaxCompactTileRows(rows, static_cast<uint32_t>(activeChannels) * alignedW, rowElements);
        if (rows == 0U) {
            ProcessNdc1hwc0NcdhwK1DirectByRow(row, cur, block, validC1);
            cur += rowElements;
            return;
        }
        ProcessNdc1hwc0NcdhwK1DirectGroupTile(cur, nIdx, od, cBase, static_cast<uint32_t>(oh), rows,
                                              static_cast<uint32_t>(activeChannels), static_cast<uint32_t>(block),
                                              outW);
        cur += static_cast<uint64_t>(rows) * rowElements;
    }

    __aicore__ inline void ProcessNdc1hwc0NcdhwK1DirectGroup()
    {
        uint64_t block = 0U, validC1 = 0U, validOut = 0U;
        uint64_t outOffset = 0U, outCount = 0U, outEnd = 0U;
        uint32_t outW = 0U, rowElements = 0U;
        if (!InitNdc1hwc0GroupRange(block, validC1, validOut, outW, rowElements, outOffset, outCount, outEnd)) {
            return;
        }
        uint64_t cur = outOffset;
        while (cur < outEnd) {
            ProcessNdc1hwc0NcdhwK1DirectGroupStep(cur, outEnd, block, validC1, rowElements, outW);
        }
        if (validOut < tiling_->totalOut) {
            CopyOutZeroRangeByCore(validOut, tiling_->totalOut - validOut);
        }
    }

    __aicore__ inline void ProcessNdc1hwc0NcdhwK1DirectGroupTile(uint64_t outputOffset, int64_t nIdx, int64_t od,
                                                                 int64_t cBase, uint32_t ohStart, uint32_t rows,
                                                                 uint32_t cCount, uint32_t block, uint32_t outW)
    {
        if (ProcessNdc1hwc0NcdhwK1DirectPlaneTile(outputOffset, nIdx, od, cBase, ohStart, rows, cCount, block, outW)) {
            return;
        }
        const uint32_t alignedW = AlignToVector(outW);
        const uint32_t rowElements = outW * block;
        LocalTensor<T> scratchLocal = tmpBuf_.Get<T>();
        LocalTensor<T> outLocal = maskBuf_.Get<T>();
        LocalTensor<uint32_t> offsetLocal = scratchLocal.template ReinterpretCast<uint32_t>();
        const uint64_t channelStride = static_cast<uint64_t>(tiling_->inD) * static_cast<uint64_t>(tiling_->inH) *
                                       static_cast<uint64_t>(tiling_->inW);
        const uint32_t srcStrideElements = static_cast<uint32_t>(channelStride - static_cast<uint64_t>(outW));
        const uint32_t zeroIndex = cCount * alignedW;
        if (CanUseNdc1hwc0WideGather(rowElements)) {
            InitNdc1hwc0NcdhwK1GatherOffsets(offsetLocal, cCount, block, outW, alignedW, zeroIndex);
        }
        for (uint32_t rowIdx = 0; rowIdx < rows; ++rowIdx) {
            CopyInVectorWBlocksPadStrideValue(InputOffset(nIdx, od, ohStart + rowIdx, 0, cBase), cCount, outW, alignedW,
                                              srcStrideElements, ZeroValue());
            LocalTensor<T> xLocal = xInQue_.DeQue<T>();
            Duplicate(xLocal[zeroIndex], ZeroValue(), 1);
            PipeBarrier<PIPE_V>();
            GatherNdc1hwc0NcdhwK1Row(outLocal, rowIdx * rowElements, xLocal, offsetLocal, cCount, block, outW, alignedW,
                                     zeroIndex, rowElements, false);
            xInQue_.FreeTensor(xLocal);
        }
        CopyOutVector(outputOffset, outLocal, rows * rowElements);
    }

    __aicore__ inline bool ProcessNdc1hwc0NcdhwK1DirectPlaneTile(uint64_t outputOffset, int64_t nIdx, int64_t od,
                                                                 int64_t cBase, uint32_t ohStart, uint32_t rows,
                                                                 uint32_t cCount, uint32_t block, uint32_t outW)
    {
        if (rows <= 1U || cCount == 0U || block == 0U || outW == 0U || cCount > block ||
            tiling_->outW != tiling_->inW) {
            return false;
        }
        const uint32_t planeValid = rows * outW;
        const uint32_t alignedPlane = AlignToVector(planeValid + (cCount < block ? 1U : 0U));
        if (planeValid == 0U || alignedPlane == 0U || alignedPlane < planeValid || alignedPlane - planeValid > 255U) {
            return false;
        }
        const uint32_t inputNeed = cCount * alignedPlane;
        const uint32_t rowElements = outW * block;
        const uint32_t totalElements = rows * rowElements;
        const uint32_t offsetElements = Ndc1hwc0GatherTempOffset(totalElements);
        if (inputNeed == 0U || inputNeed > INPUT_TILE_NUM || totalElements == 0U || totalElements > OUTPUT_TILE_NUM ||
            offsetElements > OUTPUT_TILE_NUM) {
            return false;
        }
        const uint64_t channelStride = static_cast<uint64_t>(tiling_->inD) * static_cast<uint64_t>(tiling_->inH) *
                                       static_cast<uint64_t>(tiling_->inW);
        if (channelStride < static_cast<uint64_t>(planeValid)) {
            return false;
        }

        CopyInVectorWBlocksPadStrideValue(InputOffset(nIdx, od, ohStart, 0, cBase), cCount, planeValid, alignedPlane,
                                          static_cast<uint32_t>(channelStride - static_cast<uint64_t>(planeValid)),
                                          ZeroValue());
        LocalTensor<T> xLocal = xInQue_.DeQue<T>();
        LocalTensor<T> outLocal = maskBuf_.Get<T>();
        LocalTensor<T> scratchLocal = tmpBuf_.Get<T>();
        LocalTensor<uint32_t> offsetLocal = scratchLocal.template ReinterpretCast<uint32_t>();
        InitNdc1hwc0NcdhwK1PlaneGatherOffsets(offsetLocal, rows, cCount, block, outW, alignedPlane, planeValid);

        uint32_t done = 0U;
        while (done < totalElements) {
            const uint32_t curCount = Ndc1hwc0SafeGatherChunk(totalElements - done);
            Gather(outLocal[done], xLocal, offsetLocal[done], static_cast<uint32_t>(0), curCount);
            PipeBarrier<PIPE_V>();
            done += curCount;
        }
        xInQue_.FreeTensor(xLocal);
        CopyOutVector(outputOffset, outLocal, totalElements);
        return true;
    }

    __aicore__ inline void InitNdc1hwc0NcdhwK1PlaneGatherOffsets(LocalTensor<uint32_t> offsetLocal, uint32_t rows,
                                                                 uint32_t cCount, uint32_t block, uint32_t outW,
                                                                 uint32_t alignedPlane, uint32_t zeroIndex)
    {
        LocalTensor<int32_t> offsetI32 = offsetLocal.template ReinterpretCast<int32_t>();
        const int32_t zeroOffset = static_cast<int32_t>(zeroIndex * sizeof(T));
        for (uint32_t rowIdx = 0; rowIdx < rows; ++rowIdx) {
            const uint32_t rowSrcBase = rowIdx * outW;
            const uint32_t rowOutBase = rowIdx * outW * block;
            for (uint32_t ow = 0; ow < outW; ++ow) {
                const uint32_t outBase = rowOutBase + ow * block;
                for (uint32_t c0 = 0; c0 < block; ++c0) {
                    int32_t srcOffset = zeroOffset;
                    if (c0 < cCount) {
                        srcOffset = static_cast<int32_t>((c0 * alignedPlane + rowSrcBase + ow) * sizeof(T));
                    }
                    offsetI32.SetValue(outBase + c0, srcOffset);
                }
            }
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline bool CanUseNdc1hwc0NcdhwNoWPadRowScalar() const
    {
        if (tiling_->sW != 1 || tiling_->kW != 1 || tiling_->padLeft != 0) {
            return false;
        }
        return tiling_->outW <= tiling_->inW;
    }

    __aicore__ inline void UpdateMaxValueLoaded(T curValue, T& maxValue, float& maxValueFp32)
    {
        const float cur = ValueToFloat(curValue);
        if (cur > maxValueFp32 || IsNan(cur)) {
            maxValue = curValue;
            maxValueFp32 = cur;
        }
    }

    __aicore__ inline void UpdateMaxValueByOffset(uint64_t inputOffset, T& maxValue, float& maxValueFp32)
    {
        UpdateMaxValueLoaded(xGm_.GetValue(inputOffset), maxValue, maxValueFp32);
    }

    __aicore__ inline void UpdateFloatMaxFast(float value, float& maxValue)
    {
        if (value > maxValue) {
            maxValue = value;
        }
    }

    __aicore__ inline float LoadNcdhwFloat2x2Column(uint64_t base, uint64_t hStride, uint64_t dStride, uint32_t wOffset)
    {
        const uint64_t offset = base + static_cast<uint64_t>(wOffset);
        float value = xGm_.GetValue(offset);
        UpdateFloatMaxFast(xGm_.GetValue(offset + hStride), value);
        UpdateFloatMaxFast(xGm_.GetValue(offset + dStride), value);
        UpdateFloatMaxFast(xGm_.GetValue(offset + dStride + hStride), value);
        return value;
    }

    __aicore__ inline void LoadNdhwcFloatC3Column(uint64_t base, uint64_t hStride, uint64_t dStride, float& out0,
                                                  float& out1, float& out2)
    {
        out0 = xGm_.GetValue(base);
        out1 = xGm_.GetValue(base + 1U);
        out2 = xGm_.GetValue(base + 2U);
        UpdateFloatMaxFast(xGm_.GetValue(base + hStride), out0);
        UpdateFloatMaxFast(xGm_.GetValue(base + hStride + 1U), out1);
        UpdateFloatMaxFast(xGm_.GetValue(base + hStride + 2U), out2);
        UpdateFloatMaxFast(xGm_.GetValue(base + dStride), out0);
        UpdateFloatMaxFast(xGm_.GetValue(base + dStride + 1U), out1);
        UpdateFloatMaxFast(xGm_.GetValue(base + dStride + 2U), out2);
        UpdateFloatMaxFast(xGm_.GetValue(base + dStride + hStride), out0);
        UpdateFloatMaxFast(xGm_.GetValue(base + dStride + hStride + 1U), out1);
        UpdateFloatMaxFast(xGm_.GetValue(base + dStride + hStride + 2U), out2);
    }

    __aicore__ inline void ComputeNdhwcFloatC3NoPad2x2x2Row(uint64_t base, uint32_t countW, LocalTensor<T> yLocal,
                                                            uint32_t offset)
    {
        const uint64_t hStride = static_cast<uint64_t>(tiling_->inW) * 3U;
        const uint64_t dStride = static_cast<uint64_t>(tiling_->inH) * hStride;
        float left0 = 0.0F;
        float left1 = 0.0F;
        float left2 = 0.0F;
        LoadNdhwcFloatC3Column(base, hStride, dStride, left0, left1, left2);
        for (uint32_t ow = 0; ow < countW; ++ow) {
            float right0 = 0.0F;
            float right1 = 0.0F;
            float right2 = 0.0F;
            LoadNdhwcFloatC3Column(base + static_cast<uint64_t>(ow + 1U) * 3U, hStride, dStride, right0, right1,
                                   right2);
            float out0 = left0;
            float out1 = left1;
            float out2 = left2;
            UpdateFloatMaxFast(right0, out0);
            UpdateFloatMaxFast(right1, out1);
            UpdateFloatMaxFast(right2, out2);
            const uint32_t outOffset = offset + ow * 3U;
            yLocal.SetValue(outOffset, out0);
            yLocal.SetValue(outOffset + 1U, out1);
            yLocal.SetValue(outOffset + 2U, out2);
            left0 = right0;
            left1 = right1;
            left2 = right2;
        }
    }

    __aicore__ inline void ProcessNdhwcFloatC3NoPad2x2x2Direct()
    {
        const uint64_t outOffset = CoreStartOffset();
        const uint64_t outCount = CoreElementCount(outOffset);
        uint64_t processed = 0;
        while (processed < outCount) {
            const uint64_t remain = outCount - processed;
            constexpr uint32_t c3TileCount = OUTPUT_TILE_NUM / 3U * 3U;
            uint32_t curCount = remain > c3TileCount ? c3TileCount : static_cast<uint32_t>(remain);
            const uint32_t rowElements = static_cast<uint32_t>(tiling_->outW) * 3U;
            if (rowElements != 0U && rowElements <= c3TileCount && curCount > rowElements) {
                curCount = curCount / rowElements * rowElements;
            }
            LocalTensor<T> yLocal = calcBuf_.Get<T>();
            uint32_t offset = 0;
            while (offset < curCount) {
                const uint64_t row = (outOffset + processed + offset) / 3U;
                int64_t nIdx = 0;
                int64_t od = 0;
                int64_t oh = 0;
                int64_t ow = 0;
                DecodeNdhwcRow(row, nIdx, od, oh, ow);
                const uint64_t base = InputOffset(nIdx, od * tiling_->sD, oh * tiling_->sH, ow, 0);
                uint32_t countW = static_cast<uint32_t>(tiling_->outW - ow);
                const uint32_t remainW = (curCount - offset) / 3U;
                if (countW > remainW) {
                    countW = remainW;
                }
                ComputeNdhwcFloatC3NoPad2x2x2Row(base, countW, yLocal, offset);
                offset += countW * 3U;
            }
            CopyOutVectorPlain(outOffset + processed, yLocal, curCount);
            processed += curCount;
        }
    }

    __aicore__ inline T ComputeNdhwcNoPad2x2x2ScalarValue(int64_t nIdx, int64_t id, int64_t ih, int64_t iw,
                                                          int64_t cIdx)
    {
        const uint64_t base = InputOffset(nIdx, id, ih, iw, cIdx);
        const uint64_t wStride = static_cast<uint64_t>(tiling_->c);
        const uint64_t hStride = static_cast<uint64_t>(tiling_->inW) * wStride;
        const uint64_t dStride = static_cast<uint64_t>(tiling_->inH) * hStride;
        return ComputeNdhwcNoPad2x2x2ScalarValueByBase(base, wStride, hStride, dStride);
    }

    __aicore__ inline T ComputeNdhwcNoPad2x2x2ScalarValueByBase(uint64_t base, uint64_t wStride, uint64_t hStride,
                                                                uint64_t dStride)
    {
        T maxValue = xGm_.GetValue(base);
        float maxValueFp32 = ValueToFloat(maxValue);
        UpdateMaxValueByOffset(base + wStride, maxValue, maxValueFp32);
        UpdateMaxValueByOffset(base + hStride, maxValue, maxValueFp32);
        UpdateMaxValueByOffset(base + hStride + wStride, maxValue, maxValueFp32);
        UpdateMaxValueByOffset(base + dStride, maxValue, maxValueFp32);
        UpdateMaxValueByOffset(base + dStride + wStride, maxValue, maxValueFp32);
        UpdateMaxValueByOffset(base + dStride + hStride, maxValue, maxValueFp32);
        UpdateMaxValueByOffset(base + dStride + hStride + wStride, maxValue, maxValueFp32);
        return maxValue;
    }

    __aicore__ inline void FillNdhwcHalfC8Stride2ScalarTile(LocalTensor<T> yLocal, uint64_t outLinearStart,
                                                            uint32_t curCount, uint64_t wStride, uint64_t hStride,
                                                            uint64_t dStride)
    {
        constexpr uint32_t c8 = 8U;
        uint32_t offset = 0;
        while (offset < curCount) {
            const uint64_t row = (outLinearStart + offset) / c8;
            int64_t nIdx = 0;
            int64_t od = 0;
            int64_t oh = 0;
            int64_t ow = 0;
            DecodeNdhwcRow(row, nIdx, od, oh, ow);
            const uint64_t base = InputOffset(nIdx, od * tiling_->sD, oh * tiling_->sH, ow * tiling_->sW, 0);
            if (ow + 1 < tiling_->outW && offset + 2U * c8 <= curCount &&
                (ow * tiling_->sW + tiling_->sW + 1) < tiling_->inW) {
                for (uint32_t i = 0; i < c8; ++i) {
                    yLocal.SetValue(offset + i,
                                    ComputeNdhwcNoPad2x2x2ScalarValueByBase(base + i, wStride, hStride, dStride));
                    yLocal.SetValue(offset + c8 + i, ComputeNdhwcNoPad2x2x2ScalarValueByBase(
                                                         base + static_cast<uint64_t>(tiling_->sW) * wStride + i,
                                                         wStride, hStride, dStride));
                }
                offset += 2U * c8;
                continue;
            }
            for (uint32_t i = 0; i < c8; ++i) {
                yLocal.SetValue(offset + i,
                                ComputeNdhwcNoPad2x2x2ScalarValueByBase(base + i, wStride, hStride, dStride));
            }
            offset += c8;
        }
    }

    __aicore__ inline void ProcessNdhwcHalfC8Stride2ScalarRow()
    {
        constexpr uint32_t c8 = 8U;
        const uint64_t outOffset = CoreStartOffset();
        const uint64_t outCount = CoreElementCount(outOffset);
        const uint64_t wStride = c8;
        const uint64_t hStride = static_cast<uint64_t>(tiling_->inW) * wStride;
        const uint64_t dStride = static_cast<uint64_t>(tiling_->inH) * hStride;
        uint64_t processed = 0;
        while (processed < outCount) {
            const uint64_t remain = outCount - processed;
            uint32_t curCount = remain > OUTPUT_TILE_NUM ? OUTPUT_TILE_NUM : static_cast<uint32_t>(remain);
            curCount = curCount / c8 * c8;
            if (curCount == 0U) {
                ProcessGenericRange(outOffset + processed, remain);
                return;
            }
            LocalTensor<T> yLocal = yOutQue_.AllocTensor<T>();
            FillNdhwcHalfC8Stride2ScalarTile(yLocal, outOffset + processed, curCount, wStride, hStride, dStride);
            yOutQue_.EnQue(yLocal);

            LocalTensor<T> yOut = yOutQue_.DeQue<T>();
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(curCount * sizeof(T)), 0, 0, 0};
            DataCopyPad(yGm_[outOffset + processed], yOut, copyParams);
            yOutQue_.FreeTensor(yOut);
            processed += curCount;
        }
    }

    __aicore__ inline void ComputeNdhwcNoPad2x2x2ScalarPairByBase(uint64_t base, T& out0, T& out1)
    {
        const uint64_t wStride = static_cast<uint64_t>(tiling_->c);
        const uint64_t hStride = static_cast<uint64_t>(tiling_->inW) * wStride;
        const uint64_t dStride = static_cast<uint64_t>(tiling_->inH) * hStride;

        const T leftValue = xGm_.GetValue(base);
        const T midValue = xGm_.GetValue(base + wStride);
        const T rightValue = xGm_.GetValue(base + 2U * wStride);
        out0 = leftValue;
        float out0Fp32 = ValueToFloat(out0);
        UpdateMaxValueLoaded(midValue, out0, out0Fp32);
        out1 = midValue;
        float out1Fp32 = ValueToFloat(out1);
        UpdateMaxValueLoaded(rightValue, out1, out1Fp32);

        UpdateNdhwcPairPlane(base + hStride, wStride, out0, out0Fp32, out1, out1Fp32);
        UpdateNdhwcPairPlane(base + dStride, wStride, out0, out0Fp32, out1, out1Fp32);
        UpdateNdhwcPairPlane(base + dStride + hStride, wStride, out0, out0Fp32, out1, out1Fp32);
    }

    __aicore__ inline void UpdateNdhwcPairPlane(uint64_t base, uint64_t wStride, T& out0, float& out0Fp32, T& out1,
                                                float& out1Fp32)
    {
        const T leftValue = xGm_.GetValue(base);
        const T midValue = xGm_.GetValue(base + wStride);
        const T rightValue = xGm_.GetValue(base + 2U * wStride);
        UpdateMaxValueLoaded(leftValue, out0, out0Fp32);
        UpdateMaxValueLoaded(midValue, out0, out0Fp32);
        UpdateMaxValueLoaded(midValue, out1, out1Fp32);
        UpdateMaxValueLoaded(rightValue, out1, out1Fp32);
    }

    __aicore__ inline void UpdateNdhwcScalar2x2x2Depth(uint64_t dBase, int64_t h0, int64_t h1, int64_t w0, int64_t w1,
                                                       bool h0Valid, bool h1Valid, bool w0Valid, bool w1Valid,
                                                       uint64_t hStride, uint64_t wStride, T& maxValue,
                                                       float& maxValueFp32)
    {
        if (h0Valid) {
            const uint64_t hBase = dBase + static_cast<uint64_t>(h0) * hStride;
            if (w0Valid) {
                UpdateMaxValueByOffset(hBase + static_cast<uint64_t>(w0) * wStride, maxValue, maxValueFp32);
            }
            if (w1Valid) {
                UpdateMaxValueByOffset(hBase + static_cast<uint64_t>(w1) * wStride, maxValue, maxValueFp32);
            }
        }
        if (h1Valid) {
            const uint64_t hBase = dBase + static_cast<uint64_t>(h1) * hStride;
            if (w0Valid) {
                UpdateMaxValueByOffset(hBase + static_cast<uint64_t>(w0) * wStride, maxValue, maxValueFp32);
            }
            if (w1Valid) {
                UpdateMaxValueByOffset(hBase + static_cast<uint64_t>(w1) * wStride, maxValue, maxValueFp32);
            }
        }
    }

    __aicore__ inline T ComputeNdhwcScalar2x2x2Value(int64_t nIdx, int64_t od, int64_t oh, int64_t ow, int64_t cIdx)
    {
        const int64_t d0 = od * tiling_->sD - tiling_->padFront;
        const int64_t h0 = oh * tiling_->sH - tiling_->padTop;
        const int64_t w0 = ow * tiling_->sW - tiling_->padLeft;
        const int64_t d1 = d0 + tiling_->dilationD;
        const int64_t h1 = h0 + tiling_->dilationH;
        const int64_t w1 = w0 + tiling_->dilationW;
        T maxValue = NegInfValue();
        float maxValueFp32 = ValueToFloat(maxValue);
        const bool d0Valid = !IsOutOfRange(d0, tiling_->inD);
        const bool d1Valid = !IsOutOfRange(d1, tiling_->inD);
        const bool h0Valid = !IsOutOfRange(h0, tiling_->inH);
        const bool h1Valid = !IsOutOfRange(h1, tiling_->inH);
        const bool w0Valid = !IsOutOfRange(w0, tiling_->inW);
        const bool w1Valid = !IsOutOfRange(w1, tiling_->inW);
        const uint64_t wStride = static_cast<uint64_t>(tiling_->c);
        const uint64_t hStride = static_cast<uint64_t>(tiling_->inW) * wStride;
        const uint64_t dStride = static_cast<uint64_t>(tiling_->inH) * hStride;
        const uint64_t nBase = static_cast<uint64_t>(nIdx) * static_cast<uint64_t>(tiling_->inD) * dStride +
                               static_cast<uint64_t>(cIdx);

        if (d0Valid) {
            UpdateNdhwcScalar2x2x2Depth(nBase + static_cast<uint64_t>(d0) * dStride, h0, h1, w0, w1, h0Valid, h1Valid,
                                        w0Valid, w1Valid, hStride, wStride, maxValue, maxValueFp32);
        }
        if (d1Valid) {
            UpdateNdhwcScalar2x2x2Depth(nBase + static_cast<uint64_t>(d1) * dStride, h0, h1, w0, w1, h0Valid, h1Valid,
                                        w0Valid, w1Valid, hStride, wStride, maxValue, maxValueFp32);
        }
        return maxValue;
    }

    __aicore__ inline bool FillNdhwcScalar2x2x2Pair(LocalTensor<T> yLocal, uint32_t& offset, uint32_t curCount,
                                                    int64_t nIdx, int64_t id, int64_t ih, int64_t iw, int64_t ow,
                                                    uint32_t cBase, uint32_t rowCount, uint32_t cCount,
                                                    bool useNoPadFast)
    {
        const bool commonPair = useNoPadFast && cBase == 0U && rowCount == cCount && offset + 2U * cCount <= curCount &&
                                ow + 1 < tiling_->outW;
        if (commonPair && tiling_->sW == 1 && iw + 2 < tiling_->inW) {
            const uint64_t base = InputOffset(nIdx, id, ih, iw, 0);
            for (uint32_t i = 0; i < cCount; ++i) {
                T out0;
                T out1;
                ComputeNdhwcNoPad2x2x2ScalarPairByBase(base + i, out0, out1);
                yLocal.SetValue(offset + i, out0);
                yLocal.SetValue(offset + cCount + i, out1);
            }
            offset += 2U * cCount;
            return true;
        }
        if (commonPair && tiling_->sW == 2 && iw + tiling_->sW + 1 < tiling_->inW) {
            const uint64_t base = InputOffset(nIdx, id, ih, iw, 0);
            const uint64_t wStride = static_cast<uint64_t>(tiling_->c);
            const uint64_t hStride = static_cast<uint64_t>(tiling_->inW) * wStride;
            const uint64_t dStride = static_cast<uint64_t>(tiling_->inH) * hStride;
            for (uint32_t i = 0; i < cCount; ++i) {
                yLocal.SetValue(offset + i,
                                ComputeNdhwcNoPad2x2x2ScalarValueByBase(base + i, wStride, hStride, dStride));
                yLocal.SetValue(offset + cCount + i, ComputeNdhwcNoPad2x2x2ScalarValueByBase(
                                                         base + static_cast<uint64_t>(tiling_->sW) * wStride + i,
                                                         wStride, hStride, dStride));
            }
            offset += 2U * cCount;
            return true;
        }
        return false;
    }

    __aicore__ inline void FillNdhwcScalar2x2x2Tile(LocalTensor<T> yLocal, uint64_t outLinearStart, uint32_t curCount)
    {
        uint32_t offset = 0;
        while (offset < curCount) {
            const uint64_t outLinear = outLinearStart + offset;
            const uint64_t row = outLinear / static_cast<uint64_t>(tiling_->c);
            const uint32_t cBase = static_cast<uint32_t>(outLinear - row * static_cast<uint64_t>(tiling_->c));
            uint32_t rowCount = static_cast<uint32_t>(tiling_->c) - cBase;
            const uint32_t remainInTile = curCount - offset;
            if (rowCount > remainInTile) {
                rowCount = remainInTile;
            }
            int64_t nIdx = 0;
            int64_t od = 0;
            int64_t oh = 0;
            int64_t ow = 0;
            DecodeNdhwcRow(row, nIdx, od, oh, ow);
            const int64_t id = od * tiling_->sD;
            const int64_t ih = oh * tiling_->sH;
            const int64_t iw = ow * tiling_->sW;
            const bool useNoPadFast = CanUseNdhwcNoPad2x2x2Path() && id + 1 < tiling_->inD && ih + 1 < tiling_->inH &&
                                      iw + 1 < tiling_->inW;
            const uint32_t cCount = static_cast<uint32_t>(tiling_->c);
            if (FillNdhwcScalar2x2x2Pair(yLocal, offset, curCount, nIdx, id, ih, iw, ow, cBase, rowCount, cCount,
                                         useNoPadFast)) {
                continue;
            }
            for (uint32_t i = 0; i < rowCount; ++i) {
                const int64_t cIdx = static_cast<int64_t>(cBase + i);
                const T outValue = useNoPadFast ? ComputeNdhwcNoPad2x2x2ScalarValue(nIdx, id, ih, iw, cIdx) :
                                                  ComputeNdhwcScalar2x2x2Value(nIdx, od, oh, ow, cIdx);
                yLocal.SetValue(offset + i, outValue);
            }
            offset += rowCount;
        }
    }

    __aicore__ inline void ProcessNdhwcScalar2x2x2()
    {
        const uint64_t outOffset = CoreStartOffset();
        const uint64_t outCount = CoreElementCount(outOffset);
        uint64_t processed = 0;
        while (processed < outCount) {
            const uint64_t remain = outCount - processed;
            const uint32_t curCount = remain > OUTPUT_TILE_NUM ? OUTPUT_TILE_NUM : static_cast<uint32_t>(remain);
            LocalTensor<T> yLocal = yOutQue_.AllocTensor<T>();
            FillNdhwcScalar2x2x2Tile(yLocal, outOffset + processed, curCount);
            yOutQue_.EnQue(yLocal);
            LocalTensor<T> yOut = yOutQue_.DeQue<T>();
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(curCount * sizeof(T)), 0, 0, 0};
            DataCopyPad(yGm_[outOffset + processed], yOut, copyParams);
            yOutQue_.FreeTensor(yOut);
            processed += curCount;
        }
    }

    __aicore__ inline void FillNcdhwScalar2x2x2Tile(LocalTensor<T> yLocal, uint64_t outLinearStart, uint32_t curCount)
    {
        uint32_t offset = 0;
        while (offset < curCount) {
            const uint64_t outLinear = outLinearStart + offset;
            uint64_t row = outLinear / static_cast<uint64_t>(tiling_->outW);
            const uint32_t wBase = static_cast<uint32_t>(outLinear - row * static_cast<uint64_t>(tiling_->outW));
            uint32_t rowCount = static_cast<uint32_t>(tiling_->outW) - wBase;
            const uint32_t remainInTile = curCount - offset;
            if (rowCount > remainInTile) {
                rowCount = remainInTile;
            }
            const int64_t oh = static_cast<int64_t>(row % static_cast<uint64_t>(tiling_->outH));
            row /= static_cast<uint64_t>(tiling_->outH);
            const int64_t od = static_cast<int64_t>(row % static_cast<uint64_t>(tiling_->outD));
            row /= static_cast<uint64_t>(tiling_->outD);
            const int64_t cIdx = static_cast<int64_t>(row % static_cast<uint64_t>(tiling_->c));
            const int64_t nIdx = static_cast<int64_t>(row / static_cast<uint64_t>(tiling_->c));
            const int64_t id = od * tiling_->sD;
            const int64_t ih = oh * tiling_->sH;
            uint64_t base = InputOffset(nIdx, id, ih, static_cast<int64_t>(wBase) * tiling_->sW, cIdx);
            uint32_t i = 0;
            while (i < rowCount) {
                if (tiling_->sW == 1 && i + 1U < rowCount) {
                    T out0;
                    T out1;
                    ComputeNcdhwNoPad2x2x2ScalarPairByBase(base, out0, out1);
                    yLocal.SetValue(offset + i, out0);
                    yLocal.SetValue(offset + i + 1U, out1);
                    base += 2U;
                    i += 2U;
                    continue;
                }
                yLocal.SetValue(offset + i, ComputeNcdhwNoPad2x2x2ScalarValueByBase(base));
                base += static_cast<uint64_t>(tiling_->sW);
                ++i;
            }
            offset += rowCount;
        }
    }

    __aicore__ inline void ProcessNcdhwScalar2x2x2()
    {
        const uint64_t outOffset = CoreStartOffset();
        const uint64_t outCount = CoreElementCount(outOffset);
        uint64_t processed = 0;
        while (processed < outCount) {
            const uint64_t remain = outCount - processed;
            const uint32_t curCount = remain > OUTPUT_TILE_NUM ? OUTPUT_TILE_NUM : static_cast<uint32_t>(remain);
            LocalTensor<T> yLocal = yOutQue_.AllocTensor<T>();
            FillNcdhwScalar2x2x2Tile(yLocal, outOffset + processed, curCount);
            yOutQue_.EnQue(yLocal);

            LocalTensor<T> yOut = yOutQue_.DeQue<T>();
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(curCount * sizeof(T)), 0, 0, 0};
            DataCopyPad(yGm_[outOffset + processed], yOut, copyParams);
            yOutQue_.FreeTensor(yOut);
            processed += curCount;
        }
    }

    __aicore__ inline void ProcessNcdhwFloatStride1RowReuse()
    {
        const uint64_t outOffset = CoreStartOffset();
        const uint64_t outCount = CoreElementCount(outOffset);
        const uint32_t outW = static_cast<uint32_t>(tiling_->outW);
        uint64_t processed = 0;
        while (processed < outCount) {
            const uint64_t remain = outCount - processed;
            uint32_t curCount = remain > OUTPUT_TILE_NUM ? OUTPUT_TILE_NUM : static_cast<uint32_t>(remain);
            if (curCount > outW) {
                curCount = curCount / outW * outW;
            }
            LocalTensor<T> yLocal = yOutQue_.AllocTensor<T>();
            uint32_t offset = 0;
            while (offset < curCount) {
                const uint64_t outLinear = outOffset + processed + offset;
                uint64_t row = outLinear / static_cast<uint64_t>(outW);
                const uint32_t wBase = static_cast<uint32_t>(outLinear - row * static_cast<uint64_t>(outW));
                uint32_t rowCount = outW - wBase;
                const uint32_t remainInTile = curCount - offset;
                if (rowCount > remainInTile) {
                    rowCount = remainInTile;
                }
                if (wBase == 0 && rowCount == outW) {
                    ComputeNcdhwFloatStride1RowReuseByRow(row, yLocal, offset);
                    offset += outW;
                    continue;
                }
                ComputeNcdhwFloatStride1PartialRow(row, wBase, rowCount, yLocal, offset);
                offset += rowCount;
            }
            yOutQue_.EnQue(yLocal);

            LocalTensor<T> yOut = yOutQue_.DeQue<T>();
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(curCount * sizeof(T)), 0, 0, 0};
            DataCopyPad(yGm_[outOffset + processed], yOut, copyParams);
            yOutQue_.FreeTensor(yOut);
            processed += curCount;
        }
    }

    __aicore__ inline void ProcessNdhwcStride2WBlockVectorStep(uint64_t& cur, uint64_t outEnd, uint32_t cCount,
                                                               uint32_t maxCountW)
    {
        const uint64_t row = cur / static_cast<uint64_t>(cCount);
        if (cur != row * static_cast<uint64_t>(cCount)) {
            ProcessGenericRange(cur, outEnd - cur);
            cur = outEnd;
            return;
        }
        int64_t nIdx = 0;
        int64_t od = 0;
        int64_t oh = 0;
        int64_t ow = 0;
        DecodeNdhwcRow(row, nIdx, od, oh, ow);
        if (TryProcessNdhwcStride2FullHBlock(cur, outEnd, cCount, nIdx, od, oh, ow)) {
            return;
        }
        uint32_t countW = static_cast<uint32_t>(tiling_->outW - ow);
        const uint64_t remainRows = (outEnd - cur) / static_cast<uint64_t>(cCount);
        if (static_cast<uint64_t>(countW) > remainRows) {
            countW = static_cast<uint32_t>(remainRows);
        }
        if (countW > maxCountW) {
            countW = maxCountW;
        }
        if (countW == 0U) {
            cur = outEnd;
            return;
        }
        const uint64_t base = InputOffset(nIdx, od * tiling_->sD, oh * tiling_->sH, ow * tiling_->sW, 0);
        const bool isLastSegment = cur + static_cast<uint64_t>(countW) * static_cast<uint64_t>(cCount) >= outEnd;
        ProcessNdhwcStride2WBlockVectorSegment(base, cur, countW, cCount, isLastSegment);
        cur += static_cast<uint64_t>(countW) * static_cast<uint64_t>(cCount);
    }

    __aicore__ inline bool TryProcessNdhwcStride2FullHBlock(uint64_t& cur, uint64_t outEnd, uint32_t cCount,
                                                            int64_t nIdx, int64_t od, int64_t oh, int64_t ow)
    {
        if (ow != 0 || tiling_->outW <= 0) {
            return false;
        }
        const uint32_t fullCountW = static_cast<uint32_t>(tiling_->outW);
        uint32_t blockRows = NdhwcStride2MaxHBlockRows(fullCountW, cCount);
        const uint64_t remainOutputRows = (outEnd - cur) /
                                          (static_cast<uint64_t>(fullCountW) * static_cast<uint64_t>(cCount));
        if (static_cast<uint64_t>(blockRows) > remainOutputRows) {
            blockRows = static_cast<uint32_t>(remainOutputRows);
        }
        const uint32_t rowsToHBoundary = static_cast<uint32_t>(tiling_->outH - oh);
        if (blockRows > rowsToHBoundary) {
            blockRows = rowsToHBoundary;
        }
        if (blockRows <= 1U) {
            return false;
        }
        const uint64_t base = InputOffset(nIdx, od * tiling_->sD, oh * tiling_->sH, 0, 0);
        const uint64_t segmentCount = static_cast<uint64_t>(blockRows) * static_cast<uint64_t>(fullCountW) *
                                      static_cast<uint64_t>(cCount);
        ProcessNdhwcStride2HBlockVectorSegment(base, cur, fullCountW, cCount, blockRows, cur + segmentCount >= outEnd);
        cur += segmentCount;
        return true;
    }

    __aicore__ inline void ProcessNdhwcStride2WBlockVector()
    {
        const uint64_t outOffset = CoreStartOffset();
        const uint64_t outCount = CoreElementCount(outOffset);
        const uint64_t outEnd = outOffset + outCount;
        const uint32_t cCount = static_cast<uint32_t>(tiling_->c);
        if (cCount == 0U || outOffset % cCount != 0U) {
            ProcessGenericRange(outOffset, outCount);
            return;
        }
        uint32_t maxCountW = OUTPUT_TILE_NUM / (2U * cCount);
        const uint32_t maxCountWByInput = INPUT_TILE_NUM / (4U * cCount);
        if (maxCountW > maxCountWByInput) {
            maxCountW = maxCountWByInput;
        }
        if (maxCountW == 0U) {
            ProcessGenericRange(outOffset, outCount);
            return;
        }
        uint64_t cur = outOffset;
        while (cur < outEnd) {
            ProcessNdhwcStride2WBlockVectorStep(cur, outEnd, cCount, maxCountW);
        }
    }

    __aicore__ inline void ProcessNdhwcStride2TwoRowDThenW()
    {
        ProcessNdhwcStride2HBlockCommon<NDHWC_STRIDE2_DTHENW_ROWS, false>();
    }

    __aicore__ inline void ProcessNdhwcFloatStride2CompactHBlockDirect()
    {
        ProcessNdhwcStride2HBlockCommon<NDHWC_STRIDE2_HBLOCK_ROWS, true>();
    }

    __aicore__ inline void ProcessNdhwcStride2FullRowDirect()
    {
        ProcessNdhwcStride2HBlockCommon<NDHWC_STRIDE2_HBLOCK_ROWS, false>();
    }

    template <uint32_t maxBlockRows, bool compactFloat>
    __aicore__ inline void ProcessNdhwcStride2HBlockCommon()
    {
        const uint64_t outOffset = CoreStartOffset();
        const uint64_t outCount = CoreElementCount(outOffset);
        const uint32_t cCount = static_cast<uint32_t>(tiling_->c);
        const uint32_t countW = static_cast<uint32_t>(tiling_->outW);
        const uint64_t rowElements = static_cast<uint64_t>(countW) * static_cast<uint64_t>(cCount);
        if (outCount == 0U) {
            return;
        }
        if (rowElements == 0U || outOffset % rowElements != 0U || outCount % rowElements != 0U) {
            ProcessNdhwcStride2WBlockVector();
            return;
        }

        uint64_t row = outOffset / rowElements;
        uint64_t outputOffset = outOffset;
        uint64_t remainRows = outCount / rowElements;
        while (remainRows > 0U) {
            if (!ProcessNdhwcStride2HBlockStep<maxBlockRows, compactFloat>(row, outputOffset, remainRows, rowElements,
                                                                           countW, cCount)) {
                break;
            }
        }
    }

    template <uint32_t maxBlockRows, bool compactFloat>
    __aicore__ inline bool ProcessNdhwcStride2HBlockStep(uint64_t& row, uint64_t& outputOffset, uint64_t& remainRows,
                                                         uint64_t rowElements, uint32_t countW, uint32_t cCount)
    {
        uint64_t rowInTensor = row;
        const int64_t oh = static_cast<int64_t>(rowInTensor % static_cast<uint64_t>(tiling_->outH));
        rowInTensor /= static_cast<uint64_t>(tiling_->outH);
        const int64_t od = static_cast<int64_t>(rowInTensor % static_cast<uint64_t>(tiling_->outD));
        const int64_t nIdx = static_cast<int64_t>(rowInTensor / static_cast<uint64_t>(tiling_->outD));
        uint32_t blockRows = remainRows > maxBlockRows ? maxBlockRows : static_cast<uint32_t>(remainRows);
        const uint32_t rowsToHBoundary = static_cast<uint32_t>(tiling_->outH - oh);
        if (blockRows > rowsToHBoundary) {
            blockRows = rowsToHBoundary;
        }
        if (blockRows == 0U) {
            return false;
        }
        const uint64_t inputBase = InputOffset(nIdx, od * tiling_->sD, oh * tiling_->sH, 0, 0);
        if constexpr (compactFloat) {
            ProcessNdhwcFloatStride2CompactHBlockSegment(inputBase, outputOffset, countW, cCount, blockRows);
        } else {
            if (blockRows > 1U) {
                const bool isLastSegment = remainRows == static_cast<uint64_t>(blockRows);
                ProcessNdhwcStride2HBlockDThenWVectorSegment(inputBase, outputOffset, countW, cCount, blockRows,
                                                             isLastSegment);
            } else {
                ProcessNdhwcStride2WBlockVectorSegment(inputBase, outputOffset, countW, cCount,
                                                       remainRows == static_cast<uint64_t>(blockRows));
            }
        }
        outputOffset += static_cast<uint64_t>(blockRows) * rowElements;
        row += static_cast<uint64_t>(blockRows);
        remainRows -= static_cast<uint64_t>(blockRows);
        return true;
    }

    __aicore__ inline void ProcessNdhwcFloatStride2CompactHBlockSegment(uint64_t inputBase, uint64_t outputOffset,
                                                                        uint32_t countW, uint32_t cCount,
                                                                        uint32_t blockRows)
    {
        const uint32_t inputRowCount = countW * 2U * cCount;
        const uint32_t alignedInputRowCount = AlignToVector(inputRowCount);
        const uint32_t outputRowCount = countW * cCount;
        const uint32_t outputCount = blockRows * outputRowCount;
        const uint64_t hStride = static_cast<uint64_t>(tiling_->inW) * static_cast<uint64_t>(cCount);
        const uint64_t dStride = static_cast<uint64_t>(tiling_->inH) * hStride;
        const uint32_t srcStrideElements = static_cast<uint32_t>(hStride - static_cast<uint64_t>(inputRowCount));

        LocalTensor<T> accLocal = calcBuf_.Get<T>();
        LocalTensor<T> tmpLocal = tmpBuf_.Get<T>();

        CopyInVectorWBlocksPadStride(inputBase, blockRows * 2U, inputRowCount, alignedInputRowCount, srcStrideElements);
        LocalTensor<T> d0Local = xInQue_.DeQue<T>();
        for (uint32_t rowIdx = 0; rowIdx < blockRows; ++rowIdx) {
            const uint32_t inOffset = rowIdx * alignedInputRowCount * 2U;
            Max(accLocal[static_cast<uint64_t>(rowIdx) * alignedInputRowCount], d0Local[inOffset],
                d0Local[inOffset + alignedInputRowCount], inputRowCount);
        }
        PipeBarrier<PIPE_V>();
        xInQue_.FreeTensor(d0Local);

        CopyInVectorWBlocksPadStride(inputBase + dStride, blockRows * 2U, inputRowCount, alignedInputRowCount,
                                     srcStrideElements);
        LocalTensor<T> d1Local = xInQue_.DeQue<T>();
        for (uint32_t rowIdx = 0; rowIdx < blockRows; ++rowIdx) {
            const uint32_t inOffset = rowIdx * alignedInputRowCount * 2U;
            Max(tmpLocal[static_cast<uint64_t>(rowIdx) * alignedInputRowCount], d1Local[inOffset],
                d1Local[inOffset + alignedInputRowCount], inputRowCount);
        }
        PipeBarrier<PIPE_V>();
        xInQue_.FreeTensor(d1Local);

        for (uint32_t rowIdx = 0; rowIdx < blockRows; ++rowIdx) {
            LocalTensor<T> accRow = accLocal[static_cast<uint64_t>(rowIdx) * alignedInputRowCount];
            LocalTensor<T> tmpRow = tmpLocal[static_cast<uint64_t>(rowIdx) * alignedInputRowCount];
            Max(accRow, accRow, tmpRow, inputRowCount);
        }
        PipeBarrier<PIPE_V>();

        for (uint32_t rowIdx = 0; rowIdx < blockRows; ++rowIdx) {
            CompressNdhwcStride2WPairNoBarrier(tmpLocal[static_cast<uint64_t>(rowIdx) * outputRowCount],
                                               accLocal[static_cast<uint64_t>(rowIdx) * alignedInputRowCount], countW,
                                               cCount);
        }
        CopyOutVector(outputOffset, tmpLocal, outputCount);
    }

    __aicore__ inline void ProcessNdhwcStride2SingleRowFusedD()
    {
        const uint64_t outOffset = CoreStartOffset();
        const uint64_t outCount = CoreElementCount(outOffset);
        const uint64_t outEnd = outOffset + outCount;
        const uint32_t cCount = static_cast<uint32_t>(tiling_->c);
        const uint32_t countW = static_cast<uint32_t>(tiling_->outW);
        const uint32_t outputRowCount = countW * cCount;
        if (outputRowCount == 0U || outOffset % outputRowCount != 0U || outCount % outputRowCount != 0U) {
            ProcessNdhwcStride2WBlockVector();
            return;
        }

        uint64_t cur = outOffset;
        while (cur < outEnd) {
            const uint64_t row = cur / static_cast<uint64_t>(cCount);
            int64_t nIdx = 0;
            int64_t od = 0;
            int64_t oh = 0;
            int64_t ow = 0;
            DecodeNdhwcRow(row, nIdx, od, oh, ow);
            if (ow != 0) {
                ProcessNdhwcStride2WBlockVector();
                return;
            }
            const uint64_t base = InputOffset(nIdx, od * tiling_->sD, oh * tiling_->sH, 0, 0);
            ProcessNdhwcStride2SingleRowFusedDSegment(base, cur, countW, cCount);
            cur += outputRowCount;
        }
    }

    __aicore__ inline void ProcessNdhwcStride2SingleRowFusedDSegment(uint64_t inputBase, uint64_t outputOffset,
                                                                     uint32_t countW, uint32_t cCount)
    {
        const uint32_t inputRowCount = countW * 2U * cCount;
        const uint32_t alignedInputRowCount = AlignToVector(inputRowCount);
        const uint32_t dPlaneOffset = alignedInputRowCount * 2U;
        const uint64_t hStride = static_cast<uint64_t>(tiling_->inW) * static_cast<uint64_t>(cCount);
        const uint64_t dStride = static_cast<uint64_t>(tiling_->inH) * hStride;
        const uint32_t srcStrideElements = static_cast<uint32_t>(hStride - static_cast<uint64_t>(inputRowCount));

        LocalTensor<T> xLocal = xInQue_.AllocTensor<T>();
        if (srcStrideElements == 0U && inputRowCount == alignedInputRowCount && inputBase % VectorAlignNum() == 0U &&
            (inputBase + dStride) % VectorAlignNum() == 0U) {
            const uint32_t copyCount = inputRowCount * 2U;
            DataCopy(xLocal, xGm_[inputBase], copyCount);
            DataCopy(xLocal[dPlaneOffset], xGm_[inputBase + dStride], copyCount);
        } else {
            DataCopyExtParams copyParams{static_cast<uint16_t>(2U), static_cast<uint32_t>(inputRowCount * sizeof(T)),
                                         static_cast<uint32_t>(srcStrideElements * sizeof(T)), 0, 0};
            DataCopyPadExtParams<T> padParams{true, 0, static_cast<uint8_t>(alignedInputRowCount - inputRowCount),
                                              NegInfValue()};
            DataCopyPad(xLocal, xGm_[inputBase], copyParams, padParams);
            DataCopyPad(xLocal[dPlaneOffset], xGm_[inputBase + dStride], copyParams, padParams);
        }
        xInQue_.EnQue(xLocal);

        LocalTensor<T> srcLocal = xInQue_.DeQue<T>();
        LocalTensor<T> rowLocal = maskBuf_.Get<T>();
        LocalTensor<T> tmpLocal = tmpBuf_.Get<T>();
        Max(rowLocal, srcLocal, srcLocal[alignedInputRowCount], inputRowCount);
        Max(tmpLocal, srcLocal[dPlaneOffset], srcLocal[dPlaneOffset + alignedInputRowCount], inputRowCount);
        PipeBarrier<PIPE_V>();
        Max(rowLocal, rowLocal, tmpLocal, inputRowCount);
        PipeBarrier<PIPE_V>();

        LocalTensor<T> yLocal = calcBuf_.Get<T>();
        CompressNdhwcStride2WPair(yLocal, rowLocal, countW, cCount);
        CopyOutVector(outputOffset, yLocal, countW * cCount);
        xInQue_.FreeTensor(srcLocal);
    }

    __aicore__ inline uint32_t NdhwcStride2MaxHBlockRows(uint32_t countW, uint32_t cCount) const
    {
        const uint32_t inputRowCount = countW * 2U * cCount;
        const uint32_t alignedInputRowCount = AlignToVector(inputRowCount);
        const uint32_t copyPerOutputH = alignedInputRowCount * 2U;
        const uint32_t outputRowCount = countW * cCount;
        if (copyPerOutputH == 0U || outputRowCount == 0U || inputRowCount > OUTPUT_TILE_NUM) {
            return 1U;
        }
        uint32_t inputTileNum = INPUT_TILE_NUM;
        uint32_t outputTileNum = OUTPUT_TILE_NUM;
        if (CanUseNdhwcStride2LargeHBlockBuffers() && countW == static_cast<uint32_t>(tiling_->outW)) {
            inputTileNum = NDHWC_STRIDE2_LARGE_INPUT_TILE_NUM;
            outputTileNum = NDHWC_STRIDE2_LARGE_OUTPUT_TILE_NUM;
        }
        uint32_t rows = inputTileNum / copyPerOutputH;
        const uint32_t rowsByOutput = outputTileNum / outputRowCount;
        if (rows > rowsByOutput) {
            rows = rowsByOutput;
        }
        if (rows == 0U) {
            rows = 1U;
        }
        if (static_cast<uint64_t>(rows) > static_cast<uint64_t>(tiling_->outH)) {
            rows = static_cast<uint32_t>(tiling_->outH);
        }
        return rows;
    }

    __aicore__ inline void ProcessNdhwcStride2HBlockVectorSegment(uint64_t inputBase, uint64_t outputOffset,
                                                                  uint32_t countW, uint32_t cCount, uint32_t blockRows,
                                                                  bool isLastSegment)
    {
        if (CanUseNdhwcStride2LargeHBlockBuffers() && countW == static_cast<uint32_t>(tiling_->outW)) {
            ProcessNdhwcStride2HBlockDThenWVectorSegment(inputBase, outputOffset, countW, cCount, blockRows,
                                                         isLastSegment);
            return;
        }
        const uint32_t outputCount = blockRows * countW * cCount;
        const uint64_t hStride = static_cast<uint64_t>(tiling_->inW) * static_cast<uint64_t>(cCount);
        const uint64_t dStride = static_cast<uint64_t>(tiling_->inH) * hStride;

        LocalTensor<T> yLocal = calcBuf_.Get<T>();
        LocalTensor<T> tmpLocal = tmpBuf_.Get<T>();
        ComputeNdhwcStride2HBlockPlane(inputBase, countW, cCount, blockRows, yLocal);
        ComputeNdhwcStride2HBlockPlane(inputBase + dStride, countW, cCount, blockRows, tmpLocal);
        Max(yLocal, yLocal, tmpLocal, outputCount);
        PipeBarrier<PIPE_V>();
        CopyOutVector(outputOffset, yLocal, outputCount);
    }

    __aicore__ inline void ReduceNdhwcStride2HBlockD0(uint64_t inputBase, uint32_t blockRows, uint32_t inputRowCount,
                                                      uint32_t alignedInputRowCount, uint32_t srcStrideElements,
                                                      LocalTensor<T> accLocal)
    {
        CopyInVectorWBlocksPadStride(inputBase, blockRows * 2U, inputRowCount, alignedInputRowCount, srcStrideElements);
        LocalTensor<T> d0Local = xInQue_.DeQue<T>();
        for (uint32_t rowIdx = 0; rowIdx < blockRows; ++rowIdx) {
            const uint32_t inOffset = rowIdx * alignedInputRowCount * 2U;
            Max(accLocal[static_cast<uint64_t>(rowIdx) * alignedInputRowCount], d0Local[inOffset],
                d0Local[inOffset + alignedInputRowCount], inputRowCount);
        }
        PipeBarrier<PIPE_V>();
        xInQue_.FreeTensor(d0Local);
    }

    __aicore__ inline void ReduceNdhwcStride2HBlockD1(uint64_t inputBase, uint32_t blockRows, uint32_t inputRowCount,
                                                      uint32_t alignedInputRowCount, uint32_t outputCount,
                                                      uint32_t srcStrideElements, LocalTensor<T> accLocal,
                                                      LocalTensor<T> rowLocal, LocalTensor<T> yLocal)
    {
        CopyInVectorWBlocksPadStride(inputBase, blockRows * 2U, inputRowCount, alignedInputRowCount, srcStrideElements);
        LocalTensor<T> d1Local = xInQue_.DeQue<T>();
        if (blockRows == NDHWC_STRIDE2_DTHENW_ROWS && outputCount >= inputRowCount) {
            const uint32_t secondRowOffset = alignedInputRowCount * 2U;
            Max(rowLocal, d1Local, d1Local[alignedInputRowCount], inputRowCount);
            Max(yLocal, d1Local[secondRowOffset], d1Local[secondRowOffset + alignedInputRowCount], inputRowCount);
            PipeBarrier<PIPE_V>();
            Max(accLocal, accLocal, rowLocal, inputRowCount);
            Max(accLocal[alignedInputRowCount], accLocal[alignedInputRowCount], yLocal, inputRowCount);
            PipeBarrier<PIPE_V>();
        } else {
            for (uint32_t rowIdx = 0; rowIdx < blockRows; ++rowIdx) {
                const uint32_t inOffset = rowIdx * alignedInputRowCount * 2U;
                LocalTensor<T> accRow = accLocal[static_cast<uint64_t>(rowIdx) * alignedInputRowCount];
                Max(rowLocal, d1Local[inOffset], d1Local[inOffset + alignedInputRowCount], inputRowCount);
                PipeBarrier<PIPE_V>();
                Max(accRow, accRow, rowLocal, inputRowCount);
                PipeBarrier<PIPE_V>();
            }
        }
        xInQue_.FreeTensor(d1Local);
    }

    __aicore__ inline void ProcessNdhwcStride2HBlockDThenWVectorSegment(uint64_t inputBase, uint64_t outputOffset,
                                                                        uint32_t countW, uint32_t cCount,
                                                                        uint32_t blockRows, bool isLastSegment)
    {
        const uint32_t inputRowCount = countW * 2U * cCount;
        const uint32_t alignedInputRowCount = AlignToVector(inputRowCount);
        const uint32_t outputRowCount = countW * cCount;
        const uint32_t outputCount = blockRows * outputRowCount;
        const uint64_t hStride = static_cast<uint64_t>(tiling_->inW) * static_cast<uint64_t>(cCount);
        const uint64_t dStride = static_cast<uint64_t>(tiling_->inH) * hStride;
        const uint32_t srcStrideElements = static_cast<uint32_t>(hStride - static_cast<uint64_t>(inputRowCount));
        LocalTensor<T> accLocal = calcBuf_.Get<T>();
        LocalTensor<T> rowLocal = tmpBuf_.Get<T>();
        LocalTensor<T> yLocal = maskBuf_.Get<T>();
        ReduceNdhwcStride2HBlockD0(inputBase, blockRows, inputRowCount, alignedInputRowCount, srcStrideElements,
                                   accLocal);
        ReduceNdhwcStride2HBlockD1(inputBase + dStride, blockRows, inputRowCount, alignedInputRowCount, outputCount,
                                   srcStrideElements, accLocal, rowLocal, yLocal);
        if (!CompressNdhwcStride2WPairRowsNoBarrier(yLocal, accLocal, countW, cCount, blockRows)) {
            for (uint32_t rowIdx = 0; rowIdx < blockRows; ++rowIdx) {
                CompressNdhwcStride2WPairNoBarrier(yLocal[static_cast<uint64_t>(rowIdx) * outputRowCount],
                                                   accLocal[static_cast<uint64_t>(rowIdx) * alignedInputRowCount],
                                                   countW, cCount);
            }
        }
        if (isLastSegment) {
            CopyOutVectorLast(outputOffset, yLocal, outputCount);
        } else {
            CopyOutVector(outputOffset, yLocal, outputCount);
        }
    }

    __aicore__ inline void ComputeNdhwcStride2HBlockPlane(uint64_t inputOffset, uint32_t countW, uint32_t cCount,
                                                          uint32_t blockRows, LocalTensor<T> dstLocal)
    {
        const uint32_t inputRowCount = countW * 2U * cCount;
        const uint32_t alignedInputRowCount = AlignToVector(inputRowCount);
        const uint64_t hStride = static_cast<uint64_t>(tiling_->inW) * static_cast<uint64_t>(cCount);
        const uint32_t srcStrideElements = static_cast<uint32_t>(hStride - static_cast<uint64_t>(inputRowCount));
        CopyInVectorWBlocksPadStride(inputOffset, blockRows * 2U, inputRowCount, alignedInputRowCount,
                                     srcStrideElements);

        LocalTensor<T> xLocal = xInQue_.DeQue<T>();
        LocalTensor<T> rowLocal = maskBuf_.Get<T>();
        const uint32_t outputRowCount = countW * cCount;
        for (uint32_t rowIdx = 0; rowIdx < blockRows; ++rowIdx) {
            const uint32_t inOffset = rowIdx * alignedInputRowCount * 2U;
            Max(rowLocal, xLocal[inOffset], xLocal[inOffset + alignedInputRowCount], inputRowCount);
            PipeBarrier<PIPE_V>();
            CompressNdhwcStride2WPair(dstLocal[static_cast<uint64_t>(rowIdx) * outputRowCount], rowLocal, countW,
                                      cCount);
        }
        xInQue_.FreeTensor(xLocal);
    }

    __aicore__ inline void ProcessNdhwcStride2WBlockVectorSegment(uint64_t inputBase, uint64_t outputOffset,
                                                                  uint32_t countW, uint32_t cCount,
                                                                  bool isLastSegment = false)
    {
        const uint32_t inputRowCount = countW * 2U * cCount;
        const uint32_t alignedInputRowCount = AlignToVector(inputRowCount);
        const uint64_t hStride = static_cast<uint64_t>(tiling_->inW) * static_cast<uint64_t>(cCount);
        const uint64_t dStride = static_cast<uint64_t>(tiling_->inH) * hStride;

        LocalTensor<T> accLocal = calcBuf_.Get<T>();
        LocalTensor<T> tmpLocal = tmpBuf_.Get<T>();
        LocalTensor<T> rowLocal = maskBuf_.Get<T>();
        bool hasValue = false;
        bool accInTmp = false;
        ReduceNdhwcStride2WBlock(inputBase, inputRowCount, alignedInputRowCount, hStride, accLocal, tmpLocal, rowLocal,
                                 hasValue, accInTmp);
        ReduceNdhwcStride2WBlock(inputBase + dStride, inputRowCount, alignedInputRowCount, hStride, accLocal, tmpLocal,
                                 rowLocal, hasValue, accInTmp);

        LocalTensor<T> resultLocal = accInTmp ? tmpLocal : accLocal;
        const uint32_t outputCount = countW * cCount;
        CompressNdhwcStride2WPair(rowLocal, resultLocal, countW, cCount);
        if (isLastSegment) {
            CopyOutVectorLast(outputOffset, rowLocal, outputCount);
        } else {
            CopyOutVector(outputOffset, rowLocal, outputCount);
        }
    }

    __aicore__ inline void ReduceNdhwcStride2WBlock(uint64_t inputOffset, uint32_t inputRowCount,
                                                    uint32_t alignedInputRowCount, uint64_t hStride,
                                                    LocalTensor<T> accLocal, LocalTensor<T> tmpLocal,
                                                    LocalTensor<T> rowLocal, bool& hasValue, bool& accInTmp)
    {
        const uint32_t srcStrideElements = static_cast<uint32_t>(hStride - static_cast<uint64_t>(inputRowCount));
        CopyInVectorWBlocksPadStride(inputOffset, 2U, inputRowCount, alignedInputRowCount, srcStrideElements);
        LocalTensor<T> xLocal = xInQue_.DeQue<T>();
        LocalTensor<T> dstLocal = hasValue ? rowLocal : accLocal;
        Max(dstLocal, xLocal, xLocal[alignedInputRowCount], inputRowCount);
        PipeBarrier<PIPE_V>();
        UpdateStride2ReductionAccumulator<true>(accLocal, tmpLocal, rowLocal, inputRowCount, hasValue, accInTmp);
        PipeBarrier<PIPE_V>();
        xInQue_.FreeTensor(xLocal);
    }

    template <bool firstValueInAccumulator>
    __aicore__ inline void UpdateStride2ReductionAccumulator(LocalTensor<T> accLocal, LocalTensor<T> tmpLocal,
                                                             LocalTensor<T> rowLocal, uint32_t count, bool& hasValue,
                                                             bool& accInTmp)
    {
        if (!hasValue) {
            if constexpr (!firstValueInAccumulator) {
                CopyLocalTensor(accLocal, rowLocal, count);
            }
            hasValue = true;
            accInTmp = false;
            return;
        }
        if (accInTmp) {
            Max(accLocal, tmpLocal, rowLocal, count);
        } else {
            Max(tmpLocal, accLocal, rowLocal, count);
        }
        accInTmp = !accInTmp;
    }

    __aicore__ inline void CompressNdhwcStride2WPair(LocalTensor<T> yLocal, LocalTensor<T> resultLocal, uint32_t countW,
                                                     uint32_t cCount)
    {
        if (CompressNdhwcStride2WPairRepeat(yLocal, resultLocal, countW, cCount)) {
            return;
        }
        for (uint32_t ow = 0; ow < countW; ++ow) {
            const uint32_t inputOffset = ow * 2U * cCount;
            Max(yLocal[static_cast<uint64_t>(ow) * cCount], resultLocal[inputOffset], resultLocal[inputOffset + cCount],
                cCount);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CompressNdhwcStride2WPairNoBarrier(LocalTensor<T> yLocal, LocalTensor<T> resultLocal,
                                                              uint32_t countW, uint32_t cCount)
    {
        if (CompressNdhwcStride2WPairRepeatNoBarrier(yLocal, resultLocal, countW, cCount)) {
            return;
        }
        for (uint32_t ow = 0; ow < countW; ++ow) {
            const uint32_t inputOffset = ow * 2U * cCount;
            Max(yLocal[static_cast<uint64_t>(ow) * cCount], resultLocal[inputOffset], resultLocal[inputOffset + cCount],
                cCount);
        }
    }

    __aicore__ inline bool CompressNdhwcStride2WPairRepeat(LocalTensor<T> yLocal, LocalTensor<T> resultLocal,
                                                           uint32_t countW, uint32_t cCount)
    {
        if (!CompressNdhwcStride2WPairRepeatCommon(yLocal, resultLocal, cCount, countW)) {
            return false;
        }
        PipeBarrier<PIPE_V>();
        return true;
    }

    __aicore__ inline bool CompressNdhwcStride2WPairRepeatNoBarrier(LocalTensor<T> yLocal, LocalTensor<T> resultLocal,
                                                                    uint32_t countW, uint32_t cCount)
    {
        return CompressNdhwcStride2WPairRepeatCommon(yLocal, resultLocal, cCount, countW);
    }

    __aicore__ inline bool CompressNdhwcStride2WPairRowsNoBarrier(LocalTensor<T> yLocal, LocalTensor<T> resultLocal,
                                                                  uint32_t countW, uint32_t cCount, uint32_t blockRows)
    {
        const uint32_t repeatTimes = countW * blockRows;
        if (countW == 0U || cCount == 0U || blockRows == 0U || repeatTimes > 255U) {
            return false;
        }
        const uint32_t inputRowCount = countW * 2U * cCount;
        if (AlignToVector(inputRowCount) != inputRowCount) {
            return false;
        }
        return CompressNdhwcStride2WPairRepeatCommon(yLocal, resultLocal, cCount, repeatTimes);
    }

    __aicore__ inline bool CompressNdhwcStride2WPairRepeatCommon(LocalTensor<T> yLocal, LocalTensor<T> resultLocal,
                                                                 uint32_t cCount, uint32_t repeatTimes)
    {
        if (cCount == 0U || repeatTimes == 0U || repeatTimes > 255U) {
            return false;
        }
        const uint32_t dstRepStride = static_cast<uint32_t>(static_cast<uint64_t>(cCount) * sizeof(T) / UB_BLOCK_BYTES);
        const uint32_t srcRepStride = dstRepStride * 2U;
        if (dstRepStride == 0U || dstRepStride > 255U || srcRepStride > 255U) {
            return false;
        }

        constexpr uint32_t repeatElements = 256U / sizeof(T);
        const BinaryRepeatParams params = MakeStride2BinaryRepeatParams(dstRepStride, srcRepStride);
        uint32_t cBase = 0U;
        while (cBase < cCount) {
            uint32_t curCount = cCount - cBase;
            if (curCount > repeatElements) {
                curCount = repeatElements;
            }
            Max(yLocal[cBase], resultLocal[cBase], resultLocal[cCount + cBase], curCount,
                static_cast<uint8_t>(repeatTimes), params);
            cBase += curCount;
        }
        return true;
    }

    __aicore__ inline BinaryRepeatParams MakeStride2BinaryRepeatParams(uint32_t dstRepStride,
                                                                       uint32_t srcRepStride) const
    {
        return {1,
                1,
                1,
                static_cast<uint8_t>(dstRepStride),
                static_cast<uint8_t>(srcRepStride),
                static_cast<uint8_t>(srcRepStride)};
    }

    __aicore__ inline void ProcessNcdhwStride2MicroHBlock()
    {
        const uint64_t outOffset = CoreStartOffset();
        const uint64_t outCount = CoreElementCount(outOffset);
        const uint32_t outW = static_cast<uint32_t>(tiling_->outW);
        if (outW == 0U || outOffset % outW != 0U || outCount % outW != 0U) {
            ProcessNcdhwStride2RowVector();
            return;
        }

        uint64_t processed = 0;
        while (processed < outCount) {
            uint64_t row = 0U;
            const uint32_t blockRows = NcdhwStride2BlockRows(outOffset, outCount, processed, outW, row);
            if (blockRows == 0U) {
                return;
            }
            ProcessNcdhwStride2HBlockVector(row, outOffset + processed, outW, blockRows);
            processed += static_cast<uint64_t>(blockRows) * static_cast<uint64_t>(outW);
        }
    }

    __aicore__ inline uint32_t NcdhwStride2BlockRows(uint64_t outOffset, uint64_t outCount, uint64_t processed,
                                                     uint32_t outW, uint64_t& row) const
    {
        row = (outOffset + processed) / static_cast<uint64_t>(outW);
        const uint32_t oh = static_cast<uint32_t>(row % static_cast<uint64_t>(tiling_->outH));
        uint32_t blockRows = NcdhwStride2MaxHBlockRows(outW);
        const uint64_t remainRows = (outCount - processed) / static_cast<uint64_t>(outW);
        if (static_cast<uint64_t>(blockRows) > remainRows) {
            blockRows = static_cast<uint32_t>(remainRows);
        }
        const uint32_t rowsToHBoundary = static_cast<uint32_t>(tiling_->outH) - oh;
        return blockRows > rowsToHBoundary ? rowsToHBoundary : blockRows;
    }

    __aicore__ inline void ProcessNcdhwStride2WholeDDirect()
    {
        const uint64_t outOffset = CoreStartOffset();
        const uint64_t outCount = CoreElementCount(outOffset);
        const uint32_t outW = static_cast<uint32_t>(tiling_->outW);
        const uint32_t outH = static_cast<uint32_t>(tiling_->outH);
        const uint64_t channelOut = static_cast<uint64_t>(outW) * outH * 2U;
        if (outW == 0U || outH == 0U || channelOut == 0U || outOffset % channelOut != 0U ||
            outCount % channelOut != 0U) {
            ProcessNcdhwStride2RowVector();
            return;
        }
        uint64_t processed = 0;
        while (processed < outCount) {
            const uint64_t channelLinear = (outOffset + processed) / channelOut;
            const int64_t cIdx = static_cast<int64_t>(channelLinear % static_cast<uint64_t>(tiling_->c));
            const int64_t nIdx = static_cast<int64_t>(channelLinear / static_cast<uint64_t>(tiling_->c));
            ProcessNcdhwStride2WholeDChannel(nIdx, cIdx, outOffset + processed, outW, outH);
            processed += channelOut;
        }
    }

    __aicore__ inline void ReduceNcdhwStride2HPlane(uint64_t inputOffset, LocalTensor<T> dstLocal, uint32_t inputRows,
                                                    uint32_t inputRowCount, uint32_t alignedInputRowCount,
                                                    uint32_t outputRows, uint32_t srcStrideElements)
    {
        CopyInVectorWBlocksPadStride(inputOffset, inputRows, inputRowCount, alignedInputRowCount, srcStrideElements);
        LocalTensor<T> srcLocal = xInQue_.DeQue<T>();
        if (!MaxNcdhwStride2HRowsRepeat(dstLocal, srcLocal, inputRowCount, alignedInputRowCount, outputRows)) {
            for (uint32_t rowIdx = 0; rowIdx < outputRows; ++rowIdx) {
                const uint32_t inOffset = rowIdx * alignedInputRowCount * 2U;
                Max(dstLocal[static_cast<uint64_t>(rowIdx) * alignedInputRowCount], srcLocal[inOffset],
                    srcLocal[inOffset + alignedInputRowCount], inputRowCount);
            }
        }
        PipeBarrier<PIPE_V>();
        xInQue_.FreeTensor(srcLocal);
    }

    __aicore__ inline void CompressNcdhwStride2Rows(LocalTensor<T> yLocal, LocalTensor<T> accLocal,
                                                    LocalTensor<T> tmpLocal, uint32_t outW, uint32_t inputRowCount,
                                                    uint32_t alignedInputRowCount, uint32_t alignedOutputRowCount,
                                                    uint32_t outH)
    {
        if (CompressNcdhwStride2WPairRowsPairMaxGather(yLocal, accLocal, tmpLocal, outW, inputRowCount,
                                                       alignedInputRowCount, alignedOutputRowCount, outH) ||
            CompressNcdhwStride2WPairRowsAlignedGather(yLocal, accLocal, tmpLocal, outW, alignedInputRowCount,
                                                       alignedOutputRowCount, outH)) {
            return;
        }
        for (uint32_t rowIdx = 0; rowIdx < outH; ++rowIdx) {
            CompressNcdhwStride2WPairWithScratch(yLocal[static_cast<uint64_t>(rowIdx) * alignedOutputRowCount],
                                                 accLocal[static_cast<uint64_t>(rowIdx) * alignedInputRowCount],
                                                 tmpLocal, outW);
        }
    }

    __aicore__ inline void ProcessNcdhwStride2WholeDChannel(int64_t nIdx, int64_t cIdx, uint64_t outputOffset,
                                                            uint32_t outW, uint32_t outH)
    {
        if (ProcessNcdhwStride2WholeDChannelSingleCopy(nIdx, cIdx, outputOffset, outW, outH)) {
            return;
        }
        const uint32_t inputRowCount = outW * 2U;
        const uint32_t alignedInputRowCount = AlignToVector(inputRowCount);
        const uint32_t alignedOutputRowCount = AlignToVector(outW);
        const uint64_t hStride = static_cast<uint64_t>(tiling_->inW);
        const uint64_t dStride = static_cast<uint64_t>(tiling_->inH) * hStride;
        const uint64_t base = InputOffset(nIdx, 0, 0, 0, cIdx);
        const uint32_t srcStrideElements = static_cast<uint32_t>(static_cast<uint64_t>(tiling_->inW) -
                                                                 static_cast<uint64_t>(inputRowCount));
        const uint32_t inRowsPerD = outH * 2U;
        const uint32_t accPlaneCount = outH * alignedInputRowCount;
        const uint32_t outPlaneCount = outH * alignedOutputRowCount;
        LocalTensor<T> accLocal = calcBuf_.Get<T>();
        LocalTensor<T> tmpLocal = tmpBuf_.Get<T>();
        LocalTensor<T> yLocal = maskBuf_.Get<T>();
        for (uint32_t od = 0; od < 2U; ++od) {
            const uint64_t odBase = base + static_cast<uint64_t>(od) * 2U * dStride;
            ReduceNcdhwStride2HPlane(odBase, accLocal, inRowsPerD, inputRowCount, alignedInputRowCount, outH,
                                     srcStrideElements);
            ReduceNcdhwStride2HPlane(odBase + dStride, tmpLocal, inRowsPerD, inputRowCount, alignedInputRowCount, outH,
                                     srcStrideElements);
            Max(accLocal, accLocal, tmpLocal, accPlaneCount);
            PipeBarrier<PIPE_V>();
            CompressNcdhwStride2Rows(yLocal[static_cast<uint64_t>(od) * outPlaneCount], accLocal, tmpLocal, outW,
                                     inputRowCount, alignedInputRowCount, alignedOutputRowCount, outH);
        }
        if (!CopyOutSmallCWRowCompact(outputOffset, yLocal, tmpLocal, outW, alignedOutputRowCount, outH * 2U)) {
            CopyOutSmallCWRow(outputOffset, yLocal, outW, alignedOutputRowCount, outH * 2U);
        }
    }

    __aicore__ inline void ReduceNcdhwStride2HPlaneFromLocal(LocalTensor<T> dstLocal, LocalTensor<T> srcLocal,
                                                             uint32_t inputRowCount, uint32_t alignedInputRowCount,
                                                             uint32_t outH)
    {
        if (!MaxNcdhwStride2HRowsRepeat(dstLocal, srcLocal, inputRowCount, alignedInputRowCount, outH)) {
            for (uint32_t oh = 0; oh < outH; ++oh) {
                const uint32_t row0 = oh * 2U * alignedInputRowCount;
                LocalTensor<T> dstRow = dstLocal[static_cast<uint64_t>(oh) * alignedInputRowCount];
                Max(dstRow, srcLocal[row0], srcLocal[row0 + alignedInputRowCount], inputRowCount);
            }
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline bool ProcessNcdhwStride2WholeDChannelSingleCopy(int64_t nIdx, int64_t cIdx, uint64_t outputOffset,
                                                                      uint32_t outW, uint32_t outH)
    {
        if constexpr (!AscendC::Std::is_same<T, half>::value) {
            return false;
        }
        if (tiling_->outD != 2 || tiling_->inD != 4 || tiling_->inH != static_cast<int64_t>(outH * 2U) ||
            tiling_->inW != static_cast<int64_t>(outW * 2U) || outW == 0U || outH == 0U) {
            return false;
        }
        const uint32_t inputRowCount = outW * 2U;
        const uint32_t alignedInputRowCount = AlignToVector(inputRowCount);
        const uint32_t alignedOutputRowCount = AlignToVector(outW);
        const uint32_t inputRows = static_cast<uint32_t>(tiling_->inD) * static_cast<uint32_t>(tiling_->inH);
        const uint32_t inputNeed = inputRows * alignedInputRowCount;
        const uint32_t outPlaneCount = outH * alignedOutputRowCount;
        if (alignedInputRowCount == 0U || inputNeed == 0U || inputNeed > INPUT_TILE_NUM || outPlaneCount == 0U ||
            outPlaneCount * 2U > OUTPUT_TILE_NUM) {
            return false;
        }

        const uint64_t base = InputOffset(nIdx, 0, 0, 0, cIdx);
        CopyInVectorWBlocksPadStride(base, inputRows, inputRowCount, alignedInputRowCount, 0U);
        LocalTensor<T> xLocal = xInQue_.DeQue<T>();
        LocalTensor<T> accLocal = calcBuf_.Get<T>();
        LocalTensor<T> tmpLocal = tmpBuf_.Get<T>();
        LocalTensor<T> yLocal = maskBuf_.Get<T>();
        const uint32_t inH = static_cast<uint32_t>(tiling_->inH);
        const uint32_t accPlaneCount = outH * alignedInputRowCount;
        for (uint32_t od = 0; od < 2U; ++od) {
            const uint32_t d0Base = od * 2U * inH * alignedInputRowCount;
            const uint32_t d1Base = d0Base + inH * alignedInputRowCount;
            ReduceNcdhwStride2HPlaneFromLocal(accLocal, xLocal[d0Base], inputRowCount, alignedInputRowCount, outH);
            ReduceNcdhwStride2HPlaneFromLocal(tmpLocal, xLocal[d1Base], inputRowCount, alignedInputRowCount, outH);
            Max(accLocal, accLocal, tmpLocal, accPlaneCount);
            PipeBarrier<PIPE_V>();
            CompressNcdhwStride2Rows(yLocal[static_cast<uint64_t>(od) * outPlaneCount], accLocal, tmpLocal, outW,
                                     inputRowCount, alignedInputRowCount, alignedOutputRowCount, outH);
        }
        xInQue_.FreeTensor(xLocal);

        if (!CopyOutSmallCWRowCompact(outputOffset, yLocal, tmpLocal, outW, alignedOutputRowCount, outH * 2U)) {
            CopyOutSmallCWRow(outputOffset, yLocal, outW, alignedOutputRowCount, outH * 2U);
        }
        return true;
    }

    __aicore__ inline void ProcessNcdhwStride2DPlaneDirect()
    {
        const uint64_t outOffset = CoreStartOffset();
        const uint64_t outCount = CoreElementCount(outOffset);
        const uint32_t outW = static_cast<uint32_t>(tiling_->outW);
        const uint32_t outH = static_cast<uint32_t>(tiling_->outH);
        const uint64_t planeOut = static_cast<uint64_t>(outW) * outH;
        if (outW == 0U || outH == 0U || planeOut == 0U || outOffset % planeOut != 0U || outCount % planeOut != 0U) {
            ProcessNcdhwStride2RowVector();
            return;
        }

        uint64_t processed = 0;
        while (processed < outCount) {
            uint64_t planeLinear = (outOffset + processed) / planeOut;
            const uint64_t od = planeLinear % static_cast<uint64_t>(tiling_->outD);
            planeLinear /= static_cast<uint64_t>(tiling_->outD);
            const uint64_t cIdx = planeLinear % static_cast<uint64_t>(tiling_->c);
            const uint64_t nIdx = planeLinear / static_cast<uint64_t>(tiling_->c);
            const uint64_t row = ((nIdx * static_cast<uint64_t>(tiling_->c) + cIdx) *
                                      static_cast<uint64_t>(tiling_->outD) +
                                  od) *
                                 static_cast<uint64_t>(tiling_->outH);
            uint32_t blockRows = NcdhwStride2MaxHBlockRows(outW);
            if (blockRows == 0U) {
                ProcessNcdhwStride2RowVector();
                return;
            }
            uint32_t ohBase = 0U;
            while (ohBase < outH) {
                uint32_t curRows = blockRows;
                const uint32_t remainRows = outH - ohBase;
                if (curRows > remainRows) {
                    curRows = remainRows;
                }
                ProcessNcdhwStride2HBlockVector(
                    row + ohBase, outOffset + processed + static_cast<uint64_t>(ohBase) * outW, outW, curRows);
                ohBase += curRows;
            }
            processed += planeOut;
        }
    }

    __aicore__ inline void ProcessNcdhwStride2RowVector()
    {
        const uint64_t outOffset = CoreStartOffset();
        const uint64_t outCount = CoreElementCount(outOffset);
        const uint32_t outW = static_cast<uint32_t>(tiling_->outW);
        if (outW == 0U || outOffset % outW != 0U || outCount % outW != 0U) {
            ProcessNcdhwScalar2x2x2();
            return;
        }

        uint64_t processed = 0;
        while (processed < outCount) {
            uint64_t row = 0U;
            const uint32_t blockRows = NcdhwStride2BlockRows(outOffset, outCount, processed, outW, row);
            if (blockRows > 1U) {
                ProcessNcdhwStride2HBlockVector(row, outOffset + processed, outW, blockRows);
                processed += static_cast<uint64_t>(blockRows) * static_cast<uint64_t>(outW);
            } else {
                ProcessNcdhwStride2RowVectorByRow(row, outOffset + processed, outW);
                processed += outW;
            }
        }
    }

    __aicore__ inline uint32_t NcdhwStride2MaxHBlockRows(uint32_t outW) const
    {
        if constexpr (AscendC::Std::is_same<T, bfloat16_t>::value) {
            (void)outW;
            return 1U;
        } else {
            const uint32_t inputRowCount = outW * 2U;
            const uint32_t alignedInputRowCount = AlignToVector(inputRowCount);
            const uint32_t alignedOutputRowCount = AlignToVector(outW);
            const uint32_t copyPerOutputH = alignedInputRowCount * 2U;
            if (copyPerOutputH == 0U || outW == 0U || alignedOutputRowCount == 0U) {
                return 1U;
            }
            uint32_t rows = INPUT_TILE_NUM / copyPerOutputH;
            if (alignedInputRowCount == 0U) {
                return 1U;
            }
            const uint32_t rowsByUncompressed = OUTPUT_TILE_NUM / alignedInputRowCount;
            if (rows > rowsByUncompressed) {
                rows = rowsByUncompressed;
            }
            const uint32_t rowsByOutput = OUTPUT_TILE_NUM / alignedOutputRowCount;
            if (rows > rowsByOutput) {
                rows = rowsByOutput;
            }
            if constexpr (AscendC::Std::is_same<T, float>::value || AscendC::Std::is_same<T, half>::value) {
                const uint32_t gatherOffsetElements = NcdhwStride2GatherTempOffset(outW);
                if (OUTPUT_TILE_NUM > gatherOffsetElements) {
                    const uint32_t rowsByGatherScratch = (OUTPUT_TILE_NUM - gatherOffsetElements) /
                                                         alignedOutputRowCount;
                    if (rows > rowsByGatherScratch) {
                        rows = rowsByGatherScratch;
                    }
                }
            }
            if (rows == 0U) {
                rows = 1U;
            }
            if (static_cast<uint64_t>(rows) > static_cast<uint64_t>(tiling_->outH)) {
                rows = static_cast<uint32_t>(tiling_->outH);
            }
            return rows;
        }
    }

    __aicore__ inline void CompressNcdhwStride2HBlockRows(LocalTensor<T> yLocal, LocalTensor<T> accLocal,
                                                          LocalTensor<T> tmpLocal, uint32_t outW,
                                                          uint32_t inputRowCount, uint32_t alignedInputRowCount,
                                                          uint32_t alignedOutputRowCount, uint32_t blockRows)
    {
        if constexpr (!AscendC::Std::is_same<T, bfloat16_t>::value) {
            if (CompressNcdhwStride2WPairRowsPairMaxGather(yLocal, accLocal, tmpLocal, outW, inputRowCount,
                                                           alignedInputRowCount, alignedOutputRowCount, blockRows) ||
                CompressNcdhwStride2WPairRowsAlignedGather(yLocal, accLocal, tmpLocal, outW, alignedInputRowCount,
                                                           alignedOutputRowCount, blockRows)) {
                return;
            }
            LocalTensor<uint32_t> offsetLocal = tmpLocal.template ReinterpretCast<uint32_t>();
            const uint32_t oddRowsOffset = NcdhwStride2GatherTempOffset(outW);
            LocalTensor<T> oddLocal = tmpLocal[oddRowsOffset];
            InitNcdhwStride2GatherOffsets(offsetLocal, outW);
            for (uint32_t rowIdx = 0; rowIdx < blockRows; ++rowIdx) {
                Gather(yLocal[static_cast<uint64_t>(rowIdx) * alignedOutputRowCount],
                       accLocal[static_cast<uint64_t>(rowIdx) * alignedInputRowCount], offsetLocal,
                       static_cast<uint32_t>(0), outW);
                Gather(oddLocal[static_cast<uint64_t>(rowIdx) * alignedOutputRowCount],
                       accLocal[static_cast<uint64_t>(rowIdx) * alignedInputRowCount], offsetLocal,
                       static_cast<uint32_t>(sizeof(T)), outW);
            }
            PipeBarrier<PIPE_V>();
            for (uint32_t rowIdx = 0; rowIdx < blockRows; ++rowIdx) {
                Max(yLocal[static_cast<uint64_t>(rowIdx) * alignedOutputRowCount],
                    yLocal[static_cast<uint64_t>(rowIdx) * alignedOutputRowCount],
                    oddLocal[static_cast<uint64_t>(rowIdx) * alignedOutputRowCount], outW);
            }
            PipeBarrier<PIPE_V>();
        } else {
            for (uint32_t rowIdx = 0; rowIdx < blockRows; ++rowIdx) {
                CompressNcdhwStride2WPairScalar(yLocal[static_cast<uint64_t>(rowIdx) * alignedOutputRowCount],
                                                accLocal[static_cast<uint64_t>(rowIdx) * alignedInputRowCount], outW);
            }
        }
    }

    __aicore__ inline void ProcessNcdhwStride2HBlockVector(uint64_t row, uint64_t outputOffset, uint32_t outW,
                                                           uint32_t blockRows)
    {
        const int64_t oh = static_cast<int64_t>(row % static_cast<uint64_t>(tiling_->outH));
        row /= static_cast<uint64_t>(tiling_->outH);
        const int64_t od = static_cast<int64_t>(row % static_cast<uint64_t>(tiling_->outD));
        row /= static_cast<uint64_t>(tiling_->outD);
        const int64_t cIdx = static_cast<int64_t>(row % static_cast<uint64_t>(tiling_->c));
        const int64_t nIdx = static_cast<int64_t>(row / static_cast<uint64_t>(tiling_->c));

        const uint32_t inputRowCount = outW * 2U;
        const uint32_t alignedInputRowCount = AlignToVector(inputRowCount);
        const uint32_t alignedOutputRowCount = AlignToVector(outW);
        const uint64_t hStride = static_cast<uint64_t>(tiling_->inW);
        const uint64_t dStride = static_cast<uint64_t>(tiling_->inH) * hStride;
        const uint64_t base = InputOffset(nIdx, od * tiling_->sD, oh * tiling_->sH, 0, cIdx);
        const uint32_t srcStrideElements = static_cast<uint32_t>(static_cast<uint64_t>(tiling_->inW) -
                                                                 static_cast<uint64_t>(inputRowCount));

        LocalTensor<T> accLocal = calcBuf_.Get<T>();
        LocalTensor<T> tmpLocal = tmpBuf_.Get<T>();
        LocalTensor<T> yLocal = maskBuf_.Get<T>();

        ReduceNcdhwStride2HPlane(base, accLocal, blockRows * 2U, inputRowCount, alignedInputRowCount, blockRows,
                                 srcStrideElements);
        ReduceNcdhwStride2HPlane(base + dStride, tmpLocal, blockRows * 2U, inputRowCount, alignedInputRowCount,
                                 blockRows, srcStrideElements);
        Max(accLocal, accLocal, tmpLocal, blockRows * alignedInputRowCount);
        PipeBarrier<PIPE_V>();
        CompressNcdhwStride2HBlockRows(yLocal, accLocal, tmpLocal, outW, inputRowCount, alignedInputRowCount,
                                       alignedOutputRowCount, blockRows);
        if (!CopyOutSmallCWRowCompact(outputOffset, yLocal, tmpLocal, outW, alignedOutputRowCount, blockRows)) {
            CopyOutSmallCWRow(outputOffset, yLocal, outW, alignedOutputRowCount, blockRows);
        }
    }

    __aicore__ inline bool MaxNcdhwStride2HRowsRepeat(LocalTensor<T> dstLocal, LocalTensor<T> srcLocal,
                                                      uint32_t inputRowCount, uint32_t alignedInputRowCount,
                                                      uint32_t blockRows)
    {
        if constexpr (!(AscendC::Std::is_same<T, float>::value || AscendC::Std::is_same<T, half>::value)) {
            return false;
        }
        if (inputRowCount == 0U || alignedInputRowCount == 0U || blockRows == 0U || blockRows > 255U) {
            return false;
        }
        const uint32_t dstRepStride = static_cast<uint32_t>(static_cast<uint64_t>(alignedInputRowCount) * sizeof(T) /
                                                            UB_BLOCK_BYTES);
        const uint32_t srcRepStride = dstRepStride * 2U;
        if (dstRepStride == 0U || dstRepStride > 255U || srcRepStride > 255U) {
            return false;
        }

        constexpr uint32_t repeatElements = 256U / sizeof(T);
        const BinaryRepeatParams params = MakeStride2BinaryRepeatParams(dstRepStride, srcRepStride);
        uint32_t offset = 0U;
        while (offset < alignedInputRowCount) {
            uint32_t curCount = alignedInputRowCount - offset;
            if (curCount > repeatElements) {
                curCount = repeatElements;
            }
            Max(dstLocal[offset], srcLocal[offset], srcLocal[alignedInputRowCount + offset], curCount,
                static_cast<uint8_t>(blockRows), params);
            offset += curCount;
        }
        return true;
    }

    __aicore__ inline void CompressNcdhwStride2WPair(LocalTensor<T> yLocal, LocalTensor<T> resultLocal, uint32_t outW)
    {
        if constexpr (!AscendC::Std::is_same<T, bfloat16_t>::value) {
            LocalTensor<T> scratchLocal = maskBuf_.Get<T>();
            CompressNcdhwStride2WPairGatherAligned(yLocal, resultLocal, scratchLocal, outW, AlignToVector(outW));
        } else {
            CompressNcdhwStride2WPairScalar(yLocal, resultLocal, outW);
        }
    }

    __aicore__ inline void CompressNcdhwStride2WPairWithScratch(LocalTensor<T> yLocal, LocalTensor<T> resultLocal,
                                                                LocalTensor<T> scratchLocal, uint32_t outW)
    {
        if constexpr (!AscendC::Std::is_same<T, bfloat16_t>::value) {
            CompressNcdhwStride2WPairGatherAligned(yLocal, resultLocal, scratchLocal, outW, AlignToVector(outW));
        } else {
            CompressNcdhwStride2WPairScalar(yLocal, resultLocal, outW);
        }
    }

    __aicore__ inline void CompressNcdhwStride2WPairScalar(LocalTensor<T> yLocal, LocalTensor<T> resultLocal,
                                                           uint32_t outW)
    {
        for (uint32_t ow = 0; ow < outW; ++ow) {
            const uint32_t inputOffset = ow * 2U;
            T outValue = resultLocal.GetValue(inputOffset);
            float outFp32 = ValueToFloat(outValue);
            UpdateMaxValueLoaded(resultLocal.GetValue(inputOffset + 1U), outValue, outFp32);
            yLocal.SetValue(ow, outValue);
        }
    }

    __aicore__ inline uint32_t NcdhwStride2GatherTempOffset(uint32_t outW) const
    {
        const uint32_t offsetBytes = outW * sizeof(uint32_t);
        const uint32_t offsetElements = static_cast<uint32_t>((offsetBytes + sizeof(T) - 1U) / sizeof(T));
        return AlignToVector(offsetElements);
    }

    __aicore__ inline void InitNcdhwStride2GatherOffsets(LocalTensor<uint32_t> offsetLocal, uint32_t outW)
    {
        LocalTensor<int32_t> offsetI32 = offsetLocal.template ReinterpretCast<int32_t>();
        ArithProgression(offsetI32, static_cast<int32_t>(0), static_cast<int32_t>(2U * sizeof(T)), outW);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void InitNcdhwStride2GatherOffsetsAligned(LocalTensor<uint32_t> offsetLocal, uint32_t outW,
                                                                uint32_t alignedOutW)
    {
        LocalTensor<int32_t> offsetI32 = offsetLocal.template ReinterpretCast<int32_t>();
        ArithProgression(offsetI32, static_cast<int32_t>(0), static_cast<int32_t>(2U * sizeof(T)), outW);
        for (uint32_t i = outW; i < alignedOutW; ++i) {
            offsetI32.SetValue(i, 0);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline uint32_t NcdhwStride2RowsGatherTempOffset(uint32_t totalCount) const
    {
        const uint32_t offsetBytes = totalCount * sizeof(uint32_t);
        const uint32_t offsetElements = static_cast<uint32_t>((offsetBytes + sizeof(T) - 1U) / sizeof(T));
        return AlignToVector(offsetElements);
    }

    __aicore__ inline bool CanUseNcdhwStride2PairMaxGather(uint32_t alignedOutputRowCount,
                                                           uint32_t alignedInputRowCount, uint32_t blockRows) const
    {
        const uint32_t outputCount = alignedOutputRowCount * blockRows;
        if (alignedOutputRowCount < VectorAlignNum() || alignedInputRowCount == 0U || blockRows == 0U ||
            outputCount > OUTPUT_TILE_NUM) {
            return false;
        }
        const uint32_t oddBase = AlignToVector(outputCount);
        const uint32_t offsetBase = oddBase + outputCount;
        const uint32_t offsetElements = NcdhwStride2GatherTempOffset(alignedOutputRowCount);
        return offsetBase < NCDHW_STRIDE2_TMP_TILE_NUM && offsetElements <= NCDHW_STRIDE2_TMP_TILE_NUM - offsetBase;
    }

    __aicore__ inline bool CompressNcdhwStride2WPairRowsPairMaxGather(
        LocalTensor<T> yLocal, LocalTensor<T> resultLocal, LocalTensor<T> scratchLocal, uint32_t outW,
        uint32_t inputRowCount, uint32_t alignedInputRowCount, uint32_t alignedOutputRowCount, uint32_t blockRows)
    {
        if (!CanUseNcdhwStride2PairMaxGather(alignedOutputRowCount, alignedInputRowCount, blockRows) ||
            inputRowCount < 2U) {
            return false;
        }
        const uint32_t outputCount = alignedOutputRowCount * blockRows;
        const uint32_t oddBase = AlignToVector(outputCount);
        const uint32_t offsetBase = oddBase + outputCount;
        LocalTensor<T> oddLocal = scratchLocal[oddBase];
        LocalTensor<uint32_t> offsetLocal = scratchLocal[offsetBase].template ReinterpretCast<uint32_t>();
        InitNcdhwStride2GatherOffsetsAligned(offsetLocal, outW, alignedOutputRowCount);
        for (uint32_t rowIdx = 0; rowIdx < blockRows; ++rowIdx) {
            Gather(yLocal[static_cast<uint64_t>(rowIdx) * alignedOutputRowCount],
                   resultLocal[static_cast<uint64_t>(rowIdx) * alignedInputRowCount], offsetLocal,
                   static_cast<uint32_t>(0), alignedOutputRowCount);
            Gather(oddLocal[static_cast<uint64_t>(rowIdx) * alignedOutputRowCount],
                   resultLocal[static_cast<uint64_t>(rowIdx) * alignedInputRowCount], offsetLocal,
                   static_cast<uint32_t>(sizeof(T)), alignedOutputRowCount);
        }
        PipeBarrier<PIPE_V>();
        Max(yLocal, yLocal, oddLocal, outputCount);
        PipeBarrier<PIPE_V>();
        return true;
    }

    __aicore__ inline bool CanUseNcdhwStride2RowsAlignedGather(uint32_t alignedOutputRowCount, uint32_t blockRows) const
    {
        const uint32_t totalCount = alignedOutputRowCount * blockRows;
        if (totalCount == 0U || totalCount > OUTPUT_TILE_NUM) {
            return false;
        }
        const uint32_t oddOffset = NcdhwStride2RowsGatherTempOffset(totalCount);
        const uint32_t tmpLimit = OUTPUT_TILE_NUM;
        return oddOffset < tmpLimit && totalCount <= tmpLimit - oddOffset;
    }

    __aicore__ inline void InitNcdhwStride2RowsAlignedGatherOffsets(LocalTensor<uint32_t> offsetLocal, uint32_t outW,
                                                                    uint32_t alignedInputRowCount,
                                                                    uint32_t alignedOutputRowCount, uint32_t blockRows)
    {
        LocalTensor<int32_t> offsetI32 = offsetLocal.template ReinterpretCast<int32_t>();
        for (uint32_t rowIdx = 0; rowIdx < blockRows; ++rowIdx) {
            const int32_t rowBase = static_cast<int32_t>(rowIdx * alignedInputRowCount * sizeof(T));
            const uint32_t outBase = rowIdx * alignedOutputRowCount;
            ArithProgression(offsetI32[outBase], rowBase, static_cast<int32_t>(2U * sizeof(T)), outW);
            for (uint32_t tail = outW; tail < alignedOutputRowCount; ++tail) {
                offsetI32.SetValue(outBase + tail, rowBase);
            }
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CompressNcdhwStride2WPairGatherAligned(LocalTensor<T> yLocal, LocalTensor<T> resultLocal,
                                                                  LocalTensor<T> scratchLocal, uint32_t outW,
                                                                  uint32_t alignedOutW)
    {
        LocalTensor<uint32_t> offsetLocal = scratchLocal.template ReinterpretCast<uint32_t>();
        InitNcdhwStride2GatherOffsetsAligned(offsetLocal, outW, alignedOutW);
        LocalTensor<T> oddLocal = scratchLocal[NcdhwStride2GatherTempOffset(alignedOutW)];
        Gather(yLocal, resultLocal, offsetLocal, static_cast<uint32_t>(0), alignedOutW);
        Gather(oddLocal, resultLocal, offsetLocal, static_cast<uint32_t>(sizeof(T)), alignedOutW);
        PipeBarrier<PIPE_V>();
        Max(yLocal, yLocal, oddLocal, alignedOutW);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline bool CompressNcdhwStride2WPairRowsAlignedGather(LocalTensor<T> yLocal, LocalTensor<T> resultLocal,
                                                                      LocalTensor<T> scratchLocal, uint32_t outW,
                                                                      uint32_t alignedInputRowCount,
                                                                      uint32_t alignedOutputRowCount,
                                                                      uint32_t blockRows)
    {
        if (!CanUseNcdhwStride2RowsAlignedGather(alignedOutputRowCount, blockRows)) {
            return false;
        }
        const uint32_t totalCount = alignedOutputRowCount * blockRows;
        LocalTensor<uint32_t> offsetLocal = scratchLocal.template ReinterpretCast<uint32_t>();
        InitNcdhwStride2RowsAlignedGatherOffsets(offsetLocal, outW, alignedInputRowCount, alignedOutputRowCount,
                                                 blockRows);
        LocalTensor<T> oddLocal = scratchLocal[NcdhwStride2RowsGatherTempOffset(totalCount)];
        uint32_t done = 0U;
        while (done < totalCount) {
            const uint32_t curCount = Ndc1hwc0SafeGatherChunk(totalCount - done);
            Gather(yLocal[done], resultLocal, offsetLocal[done], static_cast<uint32_t>(0), curCount);
            Gather(oddLocal[done], resultLocal, offsetLocal[done], static_cast<uint32_t>(sizeof(T)), curCount);
            PipeBarrier<PIPE_V>();
            done += curCount;
        }
        Max(yLocal, yLocal, oddLocal, totalCount);
        PipeBarrier<PIPE_V>();
        return true;
    }

    __aicore__ inline void ProcessNcdhwStride2RowVectorByRow(uint64_t row, uint64_t outputOffset, uint32_t outW)
    {
        const int64_t oh = static_cast<int64_t>(row % static_cast<uint64_t>(tiling_->outH));
        row /= static_cast<uint64_t>(tiling_->outH);
        const int64_t od = static_cast<int64_t>(row % static_cast<uint64_t>(tiling_->outD));
        row /= static_cast<uint64_t>(tiling_->outD);
        const int64_t cIdx = static_cast<int64_t>(row % static_cast<uint64_t>(tiling_->c));
        const int64_t nIdx = static_cast<int64_t>(row / static_cast<uint64_t>(tiling_->c));

        const uint32_t inputRowCount = outW * 2U;
        const uint32_t copyCount = inputRowCount * 2U;
        const uint32_t alignedCopyCount = AlignToVector(copyCount);
        const uint64_t hStride = static_cast<uint64_t>(tiling_->inW);
        const uint64_t dStride = static_cast<uint64_t>(tiling_->inH) * hStride;
        const uint64_t base = InputOffset(nIdx, od * tiling_->sD, oh * tiling_->sH, 0, cIdx);

        LocalTensor<T> accLocal = calcBuf_.Get<T>();
        LocalTensor<T> tmpLocal = tmpBuf_.Get<T>();
        LocalTensor<T> rowLocal = maskBuf_.Get<T>();
        bool hasValue = false;
        bool accInTmp = false;
        ReduceNcdhwStride2TwoRows(base, inputRowCount, copyCount, alignedCopyCount, accLocal, tmpLocal, rowLocal,
                                  hasValue, accInTmp);
        ReduceNcdhwStride2TwoRows(base + dStride, inputRowCount, copyCount, alignedCopyCount, accLocal, tmpLocal,
                                  rowLocal, hasValue, accInTmp);

        LocalTensor<T> resultLocal = accInTmp ? tmpLocal : accLocal;
        LocalTensor<T> yLocal = yOutQue_.AllocTensor<T>();
        CompressNcdhwStride2WPair(yLocal, resultLocal, outW);
        yOutQue_.EnQue(yLocal);

        LocalTensor<T> yOut = yOutQue_.DeQue<T>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(outW * sizeof(T)), 0, 0, 0};
        DataCopyPad(yGm_[outputOffset], yOut, copyParams);
        yOutQue_.FreeTensor(yOut);
    }

    __aicore__ inline void ReduceNcdhwStride2TwoRows(uint64_t inputOffset, uint32_t inputRowCount, uint32_t copyCount,
                                                     uint32_t alignedCopyCount, LocalTensor<T> accLocal,
                                                     LocalTensor<T> tmpLocal, LocalTensor<T> rowLocal, bool& hasValue,
                                                     bool& accInTmp)
    {
        CopyInVectorPad(inputOffset, copyCount, alignedCopyCount);
        LocalTensor<T> xLocal = xInQue_.DeQue<T>();
        Max(rowLocal, xLocal, xLocal[inputRowCount], inputRowCount);
        PipeBarrier<PIPE_V>();
        UpdateStride2ReductionAccumulator<false>(accLocal, tmpLocal, rowLocal, inputRowCount, hasValue, accInTmp);
        PipeBarrier<PIPE_V>();
        xInQue_.FreeTensor(xLocal);
    }

    __aicore__ inline void ComputeNcdhwFloatStride1RowReuseByRow(uint64_t row, LocalTensor<T> yLocal, uint32_t offset)
    {
        const int64_t oh = static_cast<int64_t>(row % static_cast<uint64_t>(tiling_->outH));
        row /= static_cast<uint64_t>(tiling_->outH);
        const int64_t od = static_cast<int64_t>(row % static_cast<uint64_t>(tiling_->outD));
        row /= static_cast<uint64_t>(tiling_->outD);
        const int64_t cIdx = static_cast<int64_t>(row % static_cast<uint64_t>(tiling_->c));
        const int64_t nIdx = static_cast<int64_t>(row / static_cast<uint64_t>(tiling_->c));
        const uint64_t hStride = static_cast<uint64_t>(tiling_->inW);
        const uint64_t dStride = static_cast<uint64_t>(tiling_->inH) * hStride;
        const uint64_t base = InputOffset(nIdx, od * tiling_->sD, oh * tiling_->sH, 0, cIdx);

        float left = LoadNcdhwFloat2x2Column(base, hStride, dStride, 0);
        for (uint32_t ow = 0; ow < static_cast<uint32_t>(tiling_->outW); ++ow) {
            const float right = LoadNcdhwFloat2x2Column(base, hStride, dStride, ow + 1U);
            float outValue = left;
            UpdateFloatMaxFast(right, outValue);
            yLocal.SetValue(offset + ow, outValue);
            left = right;
        }
    }

    __aicore__ inline void ComputeNcdhwFloatStride1PartialRow(uint64_t row, uint32_t wBase, uint32_t rowCount,
                                                              LocalTensor<T> yLocal, uint32_t offset)
    {
        const int64_t oh = static_cast<int64_t>(row % static_cast<uint64_t>(tiling_->outH));
        row /= static_cast<uint64_t>(tiling_->outH);
        const int64_t od = static_cast<int64_t>(row % static_cast<uint64_t>(tiling_->outD));
        row /= static_cast<uint64_t>(tiling_->outD);
        const int64_t cIdx = static_cast<int64_t>(row % static_cast<uint64_t>(tiling_->c));
        const int64_t nIdx = static_cast<int64_t>(row / static_cast<uint64_t>(tiling_->c));
        uint64_t base = InputOffset(nIdx, od * tiling_->sD, oh * tiling_->sH, static_cast<int64_t>(wBase), cIdx);
        for (uint32_t i = 0; i < rowCount; ++i) {
            yLocal.SetValue(offset + i, ComputeNcdhwNoPad2x2x2ScalarValueByBase(base + i));
        }
    }

    __aicore__ inline T ComputeNcdhwNoPad2x2x2ScalarValueByBase(uint64_t base)
    {
        constexpr uint64_t wStride = 1U;
        const uint64_t hStride = static_cast<uint64_t>(tiling_->inW);
        const uint64_t dStride = static_cast<uint64_t>(tiling_->inH) * hStride;
        return ComputeNdhwcNoPad2x2x2ScalarValueByBase(base, wStride, hStride, dStride);
    }

    __aicore__ inline void ComputeNcdhwNoPad2x2x2ScalarPairByBase(uint64_t base, T& out0, T& out1)
    {
        const uint64_t hStride = static_cast<uint64_t>(tiling_->inW);
        const uint64_t dStride = static_cast<uint64_t>(tiling_->inH) * hStride;

        T leftValue = xGm_.GetValue(base);
        T midValue = xGm_.GetValue(base + 1U);
        T rightValue = xGm_.GetValue(base + 2U);
        out0 = leftValue;
        float out0Fp32 = ValueToFloat(out0);
        UpdateMaxValueLoaded(midValue, out0, out0Fp32);
        out1 = midValue;
        float out1Fp32 = ValueToFloat(out1);
        UpdateMaxValueLoaded(rightValue, out1, out1Fp32);

        UpdateNcdhwPairPlane(base + hStride, out0, out0Fp32, out1, out1Fp32);
        UpdateNcdhwPairPlane(base + dStride, out0, out0Fp32, out1, out1Fp32);
        UpdateNcdhwPairPlane(base + dStride + hStride, out0, out0Fp32, out1, out1Fp32);
    }

    __aicore__ inline void UpdateNcdhwPairPlane(uint64_t base, T& out0, float& out0Fp32, T& out1, float& out1Fp32)
    {
        const T leftValue = xGm_.GetValue(base);
        const T midValue = xGm_.GetValue(base + 1U);
        const T rightValue = xGm_.GetValue(base + 2U);
        UpdateMaxValueLoaded(leftValue, out0, out0Fp32);
        UpdateMaxValueLoaded(midValue, out0, out0Fp32);
        UpdateMaxValueLoaded(midValue, out1, out1Fp32);
        UpdateMaxValueLoaded(rightValue, out1, out1Fp32);
    }

    __aicore__ inline void ReduceVectorInput(uint64_t inputOffset, uint32_t count, LocalTensor<T> accLocal,
                                             LocalTensor<T> tmpLocal, bool& hasValue, bool& accInTmp)
    {
        CopyInVector(inputOffset, count);
        LocalTensor<T> xLocal = xInQue_.DeQue<T>();
        if (!hasValue) {
            CopyLocalTensor(accLocal, xLocal, count);
            hasValue = true;
            accInTmp = false;
        } else {
            if (accInTmp) {
                Max(accLocal, tmpLocal, xLocal, count);
            } else {
                Max(tmpLocal, accLocal, xLocal, count);
            }
            accInTmp = !accInTmp;
        }
        PipeBarrier<PIPE_V>();
        xInQue_.FreeTensor(xLocal);
    }

    __aicore__ inline void ReduceVectorInputWStride1RowBlocks(uint64_t inputOffset, uint32_t cCount,
                                                              uint32_t alignedCount, uint32_t countW,
                                                              LocalTensor<T> rowLocal, LocalTensor<T> accLocal,
                                                              LocalTensor<T> tmpLocal, bool& hasValue, bool& accInTmp)
    {
        CopyInVectorWBlocksPad(inputOffset, countW + 1U, cCount, alignedCount);
        LocalTensor<T> xLocal = xInQue_.DeQue<T>();
        for (uint32_t i = 0; i < countW; ++i) {
            const uint32_t outOffset = i * alignedCount;
            Max(rowLocal[outOffset], xLocal[outOffset], xLocal[outOffset + alignedCount], alignedCount);
        }
        PipeBarrier<PIPE_V>();
        const uint32_t rowCount = countW * alignedCount;
        if (!hasValue) {
            CopyLocalTensor(accLocal, rowLocal, rowCount);
            hasValue = true;
            accInTmp = false;
        } else {
            if (accInTmp) {
                Max(accLocal, tmpLocal, rowLocal, rowCount);
            } else {
                Max(tmpLocal, accLocal, rowLocal, rowCount);
            }
            accInTmp = !accInTmp;
        }
        PipeBarrier<PIPE_V>();
        xInQue_.FreeTensor(xLocal);
    }

    __aicore__ inline void ReduceBf16VectorInputPad(uint64_t inputOffset, uint32_t validCount, uint32_t alignedCount,
                                                    LocalTensor<float> accLocal, LocalTensor<float> tmpLocal,
                                                    bool& hasValue, bool& accInTmp)
    {
        CopyInVectorPad(inputOffset, validCount, alignedCount);
        LocalTensor<T> xLocal = xInQue_.DeQue<T>();
        LocalTensor<float> xFloatLocal = maskBuf_.Get<float>();
        Cast(xFloatLocal, xLocal, RoundMode::CAST_NONE, alignedCount);
        PipeBarrier<PIPE_V>();
        if (!hasValue) {
            CopyLocalTensor(accLocal, xFloatLocal, alignedCount);
            hasValue = true;
            accInTmp = false;
        } else {
            if (accInTmp) {
                Max(accLocal, tmpLocal, xFloatLocal, alignedCount);
            } else {
                Max(tmpLocal, accLocal, xFloatLocal, alignedCount);
            }
            accInTmp = !accInTmp;
        }
        PipeBarrier<PIPE_V>();
        xInQue_.FreeTensor(xLocal);
    }

    __aicore__ inline void ReduceBf16VectorInputWStride1PairBlocks(uint64_t inputOffset, uint32_t validCount,
                                                                   uint32_t alignedCount, LocalTensor<float> accLocal,
                                                                   LocalTensor<float> tmpLocal, bool& hasValue,
                                                                   bool& accInTmp)
    {
        constexpr uint32_t wBlockCount = 3;
        CopyInVectorWBlocksPad(inputOffset, wBlockCount, validCount, alignedCount);
        LocalTensor<T> xLocal = xInQue_.DeQue<T>();
        LocalTensor<float> xFloatLocal = maskBuf_.Get<float>();
        Cast(xFloatLocal, xLocal, RoundMode::CAST_NONE, alignedCount * wBlockCount);
        PipeBarrier<PIPE_V>();
        Max(xFloatLocal, xFloatLocal, xFloatLocal[alignedCount], alignedCount);
        Max(xFloatLocal[alignedCount], xFloatLocal[alignedCount], xFloatLocal[alignedCount * 2U], alignedCount);
        PipeBarrier<PIPE_V>();
        const uint32_t pairCount = alignedCount * 2U;
        if (!hasValue) {
            CopyLocalTensor(accLocal, xFloatLocal, pairCount);
            hasValue = true;
            accInTmp = false;
        } else {
            if (accInTmp) {
                Max(accLocal, tmpLocal, xFloatLocal, pairCount);
            } else {
                Max(tmpLocal, accLocal, xFloatLocal, pairCount);
            }
            accInTmp = !accInTmp;
        }
        PipeBarrier<PIPE_V>();
        xInQue_.FreeTensor(xLocal);
    }

    __aicore__ inline uint32_t AlignToVector(uint32_t count) const
    {
        const uint32_t alignNum = VectorAlignNum();
        if (alignNum == 0U) {
            return count;
        }
        return (count + alignNum - 1U) / alignNum * alignNum;
    }

    __aicore__ inline void ReduceBf16VectorInputWStride1Row(uint64_t inputOffset, uint32_t inputCount,
                                                            uint32_t alignedInputCount, uint32_t outputCount,
                                                            uint32_t cCount, LocalTensor<float> accLocal,
                                                            LocalTensor<float> tmpLocal, bool& hasValue, bool& accInTmp)
    {
        CopyInVectorPad(inputOffset, inputCount, alignedInputCount);
        LocalTensor<T> xLocal = xInQue_.DeQue<T>();
        LocalTensor<float> xFloatLocal = maskBuf_.Get<float>();
        Cast(xFloatLocal, xLocal, RoundMode::CAST_NONE, alignedInputCount);
        PipeBarrier<PIPE_V>();
        if (!hasValue) {
            Max(accLocal, xFloatLocal, xFloatLocal[cCount], outputCount);
            PipeBarrier<PIPE_V>();
            hasValue = true;
            accInTmp = false;
        } else {
            if (accInTmp) {
                Max(accLocal, xFloatLocal, xFloatLocal[cCount], outputCount);
                PipeBarrier<PIPE_V>();
                Max(accLocal, tmpLocal, accLocal, outputCount);
                accInTmp = false;
            } else {
                Max(tmpLocal, xFloatLocal, xFloatLocal[cCount], outputCount);
                PipeBarrier<PIPE_V>();
                Max(tmpLocal, accLocal, tmpLocal, outputCount);
                accInTmp = true;
            }
            PipeBarrier<PIPE_V>();
        }
        xInQue_.FreeTensor(xLocal);
    }

    struct NdhwcInputPoint {
        uint64_t base;
        uint64_t wStride;
        uint64_t hStride;
        uint64_t dStride;
    };

    __aicore__ inline NdhwcInputPoint GetNdhwcInputPoint(uint64_t outLinear) const
    {
        const uint64_t row = outLinear / static_cast<uint64_t>(tiling_->c);
        const int64_t cBase = static_cast<int64_t>(outLinear - row * static_cast<uint64_t>(tiling_->c));
        int64_t nIdx = 0;
        int64_t od = 0;
        int64_t oh = 0;
        int64_t ow = 0;
        DecodeNdhwcRow(row, nIdx, od, oh, ow);
        const uint64_t wStride = static_cast<uint64_t>(tiling_->c);
        const uint64_t hStride = static_cast<uint64_t>(tiling_->inW) * wStride;
        return {InputOffset(nIdx, od * tiling_->sD, oh * tiling_->sH, ow * tiling_->sW, cBase), wStride, hStride,
                static_cast<uint64_t>(tiling_->inH) * hStride};
    }

    __aicore__ inline void ProcessNdhwcBf16NoPad2x2x2Segment(uint64_t outLinear, uint32_t validCount,
                                                             uint32_t alignedCount)
    {
        const NdhwcInputPoint point = GetNdhwcInputPoint(outLinear);

        LocalTensor<float> accLocal = calcBuf_.Get<float>();
        LocalTensor<float> tmpLocal = tmpBuf_.Get<float>();
        bool hasValue = false;
        bool accInTmp = false;
        ReduceBf16VectorInputPad(point.base, validCount, alignedCount, accLocal, tmpLocal, hasValue, accInTmp);
        ReduceBf16VectorInputPad(point.base + point.wStride, validCount, alignedCount, accLocal, tmpLocal, hasValue,
                                 accInTmp);
        ReduceBf16VectorInputPad(point.base + point.hStride, validCount, alignedCount, accLocal, tmpLocal, hasValue,
                                 accInTmp);
        ReduceBf16VectorInputPad(point.base + point.hStride + point.wStride, validCount, alignedCount, accLocal,
                                 tmpLocal, hasValue, accInTmp);
        ReduceBf16VectorInputPad(point.base + point.dStride, validCount, alignedCount, accLocal, tmpLocal, hasValue,
                                 accInTmp);
        ReduceBf16VectorInputPad(point.base + point.dStride + point.wStride, validCount, alignedCount, accLocal,
                                 tmpLocal, hasValue, accInTmp);
        ReduceBf16VectorInputPad(point.base + point.dStride + point.hStride, validCount, alignedCount, accLocal,
                                 tmpLocal, hasValue, accInTmp);
        ReduceBf16VectorInputPad(point.base + point.dStride + point.hStride + point.wStride, validCount, alignedCount,
                                 accLocal, tmpLocal, hasValue, accInTmp);

        LocalTensor<T> yLocal = yOutQue_.AllocTensor<T>();
        Cast(yLocal, accInTmp ? tmpLocal : accLocal, RoundMode::CAST_ROUND, alignedCount);
        PipeBarrier<PIPE_V>();
        yOutQue_.EnQue(yLocal);
        LocalTensor<T> yOut = yOutQue_.DeQue<T>();
        CopyOutVector(outLinear, yOut, validCount);
        yOutQue_.FreeTensor(yOut);
    }

    __aicore__ inline void ProcessNdhwcBf16NoPad2x2x2SmallCWStride1PairSegment(uint64_t outLinear, uint32_t validCount,
                                                                               uint32_t alignedCount)
    {
        const NdhwcInputPoint point = GetNdhwcInputPoint(outLinear);

        LocalTensor<float> accLocal = calcBuf_.Get<float>();
        LocalTensor<float> tmpLocal = tmpBuf_.Get<float>();
        bool hasValue = false;
        bool accInTmp = false;
        ReduceBf16VectorInputWStride1PairBlocks(point.base, validCount, alignedCount, accLocal, tmpLocal, hasValue,
                                                accInTmp);
        ReduceBf16VectorInputWStride1PairBlocks(point.base + point.hStride, validCount, alignedCount, accLocal,
                                                tmpLocal, hasValue, accInTmp);
        ReduceBf16VectorInputWStride1PairBlocks(point.base + point.dStride, validCount, alignedCount, accLocal,
                                                tmpLocal, hasValue, accInTmp);
        ReduceBf16VectorInputWStride1PairBlocks(point.base + point.dStride + point.hStride, validCount, alignedCount,
                                                accLocal, tmpLocal, hasValue, accInTmp);

        const uint32_t pairCount = alignedCount * 2U;
        LocalTensor<T> yLocal = yOutQue_.AllocTensor<T>();
        Cast(yLocal, accInTmp ? tmpLocal : accLocal, RoundMode::CAST_ROUND, pairCount);
        PipeBarrier<PIPE_V>();
        yOutQue_.EnQue(yLocal);
        LocalTensor<T> yOut = yOutQue_.DeQue<T>();
        CopyOutSmallCWPair(outLinear, yOut, validCount, alignedCount);
        yOutQue_.FreeTensor(yOut);
    }

    __aicore__ inline void ProcessNdhwcBf16NoPad2x2x2SmallCWStride1RowSegment(uint64_t outLinear, uint32_t cCount,
                                                                              uint32_t countW)
    {
        const NdhwcInputPoint point = GetNdhwcInputPoint(outLinear);
        const uint32_t outputCount = countW * cCount;
        const uint32_t inputCount = (countW + 1U) * cCount;
        const uint32_t alignedInputCount = AlignToVector(inputCount);

        LocalTensor<float> accLocal = calcBuf_.Get<float>();
        LocalTensor<float> tmpLocal = tmpBuf_.Get<float>();
        bool hasValue = false;
        bool accInTmp = false;
        ReduceBf16VectorInputWStride1Row(point.base, inputCount, alignedInputCount, outputCount, cCount, accLocal,
                                         tmpLocal, hasValue, accInTmp);
        ReduceBf16VectorInputWStride1Row(point.base + point.hStride, inputCount, alignedInputCount, outputCount, cCount,
                                         accLocal, tmpLocal, hasValue, accInTmp);
        ReduceBf16VectorInputWStride1Row(point.base + point.dStride, inputCount, alignedInputCount, outputCount, cCount,
                                         accLocal, tmpLocal, hasValue, accInTmp);
        ReduceBf16VectorInputWStride1Row(point.base + point.dStride + point.hStride, inputCount, alignedInputCount,
                                         outputCount, cCount, accLocal, tmpLocal, hasValue, accInTmp);

        LocalTensor<T> yLocal = yOutQue_.AllocTensor<T>();
        Cast(yLocal, accInTmp ? tmpLocal : accLocal, RoundMode::CAST_ROUND, outputCount);
        PipeBarrier<PIPE_V>();
        yOutQue_.EnQue(yLocal);
        LocalTensor<T> yOut = yOutQue_.DeQue<T>();
        CopyOutVector(outLinear, yOut, outputCount);
        yOutQue_.FreeTensor(yOut);
    }

    __aicore__ inline void ProcessNdhwcBf16SmallCVector()
    {
        const uint64_t outOffset = CoreStartOffset();
        const uint64_t outCount = CoreElementCount(outOffset);
        const uint64_t outEnd = outOffset + outCount;
        const uint32_t alignedCount = VectorAlignNum();
        uint64_t cur = outOffset;
        while (cur < outEnd) {
            const uint64_t row = cur / static_cast<uint64_t>(tiling_->c);
            const uint64_t cBase = cur - row * static_cast<uint64_t>(tiling_->c);
            const uint64_t remainInRow = static_cast<uint64_t>(tiling_->c) - cBase;
            const uint64_t remainInCore = outEnd - cur;
            uint32_t validCount = remainInRow > remainInCore ? static_cast<uint32_t>(remainInCore) :
                                                               static_cast<uint32_t>(remainInRow);
            if (CanUseNdhwcSmallCWStride1PairPath() && cBase == 0U &&
                remainInCore >= static_cast<uint64_t>(tiling_->c) * 2U) {
                const uint64_t ow = row % static_cast<uint64_t>(tiling_->outW);
                const uint64_t maxW = static_cast<uint64_t>(tiling_->outW) - ow;
                const uint64_t coreW = remainInCore / static_cast<uint64_t>(tiling_->c);
                uint32_t countW = static_cast<uint32_t>(maxW < coreW ? maxW : coreW);
                if (countW >= 2U && ow + countW < static_cast<uint64_t>(tiling_->inW)) {
                    ProcessNdhwcBf16NoPad2x2x2SmallCWStride1RowSegment(cur, static_cast<uint32_t>(tiling_->c), countW);
                    cur += static_cast<uint64_t>(countW) * static_cast<uint64_t>(tiling_->c);
                    continue;
                }
            }
            if (CanUseNdhwcSmallCWStride1PairPath() && cBase == 0U &&
                remainInCore >= static_cast<uint64_t>(validCount) * 2U) {
                const uint64_t ow = row % static_cast<uint64_t>(tiling_->outW);
                if (ow + 1U < static_cast<uint64_t>(tiling_->outW) && ow + 2U < static_cast<uint64_t>(tiling_->inW)) {
                    ProcessNdhwcBf16NoPad2x2x2SmallCWStride1PairSegment(cur, validCount, alignedCount);
                    cur += static_cast<uint64_t>(validCount) * 2U;
                    continue;
                }
            }
            ProcessNdhwcBf16NoPad2x2x2Segment(cur, validCount, alignedCount);
            cur += validCount;
        }
    }

    __aicore__ inline void ProcessNdhwcNoPad2x2x2SmallCWStride1RowSegment(uint64_t outLinear, uint32_t cCount,
                                                                          uint32_t countW)
    {
        const NdhwcInputPoint point = GetNdhwcInputPoint(outLinear);
        uint32_t alignedCount = VectorAlignNum();
        if (alignedCount < cCount) {
            alignedCount = cCount;
        }

        LocalTensor<T> accLocal = calcBuf_.Get<T>();
        LocalTensor<T> tmpLocal = tmpBuf_.Get<T>();
        LocalTensor<T> rowLocal = maskBuf_.Get<T>();
        bool hasValue = false;
        bool accInTmp = false;
        ReduceVectorInputWStride1RowBlocks(point.base, cCount, alignedCount, countW, rowLocal, accLocal, tmpLocal,
                                           hasValue, accInTmp);
        ReduceVectorInputWStride1RowBlocks(point.base + point.hStride, cCount, alignedCount, countW, rowLocal, accLocal,
                                           tmpLocal, hasValue, accInTmp);
        ReduceVectorInputWStride1RowBlocks(point.base + point.dStride, cCount, alignedCount, countW, rowLocal, accLocal,
                                           tmpLocal, hasValue, accInTmp);
        ReduceVectorInputWStride1RowBlocks(point.base + point.dStride + point.hStride, cCount, alignedCount, countW,
                                           rowLocal, accLocal, tmpLocal, hasValue, accInTmp);

        CopyOutSmallCWRow(outLinear, accInTmp ? tmpLocal : accLocal, cCount, alignedCount, countW);
    }

    __aicore__ inline void ProcessNdhwcNoPad2x2x2Segment(uint64_t outLinear, uint32_t count)
    {
        const NdhwcInputPoint point = GetNdhwcInputPoint(outLinear);

        LocalTensor<T> accLocal = calcBuf_.Get<T>();
        LocalTensor<T> tmpLocal = tmpBuf_.Get<T>();
        bool hasValue = false;
        bool accInTmp = false;
        ReduceVectorInput(point.base, count, accLocal, tmpLocal, hasValue, accInTmp);
        ReduceVectorInput(point.base + point.wStride, count, accLocal, tmpLocal, hasValue, accInTmp);
        ReduceVectorInput(point.base + point.hStride, count, accLocal, tmpLocal, hasValue, accInTmp);
        ReduceVectorInput(point.base + point.hStride + point.wStride, count, accLocal, tmpLocal, hasValue, accInTmp);
        ReduceVectorInput(point.base + point.dStride, count, accLocal, tmpLocal, hasValue, accInTmp);
        ReduceVectorInput(point.base + point.dStride + point.wStride, count, accLocal, tmpLocal, hasValue, accInTmp);
        ReduceVectorInput(point.base + point.dStride + point.hStride, count, accLocal, tmpLocal, hasValue, accInTmp);
        ReduceVectorInput(point.base + point.dStride + point.hStride + point.wStride, count, accLocal, tmpLocal,
                          hasValue, accInTmp);
        CopyOutVector(outLinear, accInTmp ? tmpLocal : accLocal, count);
    }

    __aicore__ inline void ProcessNdhwcVectorSegment(uint64_t outLinear, uint32_t count)
    {
        if (count < VectorAlignNum()) {
            ProcessGenericRange(outLinear, count);
            return;
        }
        if (count % VectorAlignNum() != 0U) {
            ProcessGenericRange(outLinear, count);
            return;
        }
        if (CanUseNdhwcNoPad2x2x2Path()) {
            ProcessNdhwcNoPad2x2x2Segment(outLinear, count);
            return;
        }

        const uint64_t row = outLinear / static_cast<uint64_t>(tiling_->c);
        const int64_t cBase = static_cast<int64_t>(outLinear - row * static_cast<uint64_t>(tiling_->c));
        int64_t nIdx = 0;
        int64_t od = 0;
        int64_t oh = 0;
        int64_t ow = 0;
        DecodeNdhwcRow(row, nIdx, od, oh, ow);

        LocalTensor<T> accLocal = calcBuf_.Get<T>();
        LocalTensor<T> tmpLocal = tmpBuf_.Get<T>();
        bool hasValue = false;
        bool accInTmp = false;
        ReduceNdhwcVectorWindow(nIdx, od, oh, ow, cBase, count, accLocal, tmpLocal, hasValue, accInTmp);
        if (!hasValue) {
            Duplicate(accLocal, NegInfValue(), count);
            PipeBarrier<PIPE_V>();
        }
        CopyOutVector(outLinear, accInTmp ? tmpLocal : accLocal, count);
    }

    __aicore__ inline void ReduceNdhwcVectorDepth(int64_t nIdx, int64_t id, int64_t oh, int64_t ow, int64_t cBase,
                                                  uint32_t count, LocalTensor<T> accLocal, LocalTensor<T> tmpLocal,
                                                  bool& hasValue, bool& accInTmp)
    {
        for (int64_t kh = 0; kh < tiling_->kH; ++kh) {
            const int64_t ih = DilatedInputH(oh, kh);
            if (IsOutOfRange(ih, tiling_->inH)) {
                continue;
            }
            for (int64_t kw = 0; kw < tiling_->kW; ++kw) {
                const int64_t iw = DilatedInputW(ow, kw);
                if (IsOutOfRange(iw, tiling_->inW)) {
                    continue;
                }
                const uint64_t inputOffset = InputOffset(nIdx, id, ih, iw, cBase);
                ReduceVectorInput(inputOffset, count, accLocal, tmpLocal, hasValue, accInTmp);
            }
        }
    }

    __aicore__ inline void ReduceNdhwcVectorWindow(int64_t nIdx, int64_t od, int64_t oh, int64_t ow, int64_t cBase,
                                                   uint32_t count, LocalTensor<T> accLocal, LocalTensor<T> tmpLocal,
                                                   bool& hasValue, bool& accInTmp)
    {
        for (int64_t kd = 0; kd < tiling_->kD; ++kd) {
            const int64_t id = DilatedInputD(od, kd);
            if (!IsOutOfRange(id, tiling_->inD)) {
                ReduceNdhwcVectorDepth(nIdx, id, oh, ow, cBase, count, accLocal, tmpLocal, hasValue, accInTmp);
            }
        }
    }

    __aicore__ inline void ProcessNdhwcVector()
    {
        const uint64_t outOffset = CoreStartOffset();
        const uint64_t outCount = CoreElementCount(outOffset);
        const uint64_t outEnd = outOffset + outCount;
        uint64_t cur = outOffset;
        while (cur < outEnd) {
            const uint64_t row = cur / static_cast<uint64_t>(tiling_->c);
            const uint64_t cBase = cur - row * static_cast<uint64_t>(tiling_->c);
            const uint64_t remainInRow = static_cast<uint64_t>(tiling_->c) - cBase;
            const uint64_t remainInCore = outEnd - cur;
            uint32_t count = remainInRow > OUTPUT_TILE_NUM ? OUTPUT_TILE_NUM : static_cast<uint32_t>(remainInRow);
            if (static_cast<uint64_t>(count) > remainInCore) {
                count = static_cast<uint32_t>(remainInCore);
            }
            if (CanUseNdhwcNoPad2x2x2Path() && tiling_->sW == 1 && cBase == 0U && tiling_->c > 0 &&
                static_cast<uint64_t>(tiling_->c) * 2U <= remainInCore) {
                const uint64_t ow = row % static_cast<uint64_t>(tiling_->outW);
                const uint64_t maxW = static_cast<uint64_t>(tiling_->outW) - ow;
                const uint64_t coreW = remainInCore / static_cast<uint64_t>(tiling_->c);
                uint32_t countW = static_cast<uint32_t>(maxW < coreW ? maxW : coreW);
                const uint32_t maxCountW = OUTPUT_TILE_NUM / static_cast<uint32_t>(tiling_->c) - 1U;
                if (countW > maxCountW) {
                    countW = maxCountW;
                }
                if (countW >= 2U && ow + countW < static_cast<uint64_t>(tiling_->inW)) {
                    ProcessNdhwcNoPad2x2x2SmallCWStride1RowSegment(cur, static_cast<uint32_t>(tiling_->c), countW);
                    cur += static_cast<uint64_t>(countW) * static_cast<uint64_t>(tiling_->c);
                    continue;
                }
            }
            ProcessNdhwcVectorSegment(cur, count);
            cur += count;
        }
    }

    __aicore__ inline uint64_t InputOffset(int64_t nIdx, int64_t dIdx, int64_t hIdx, int64_t wIdx, int64_t cIdx) const
    {
        if (tiling_->inputLayout == INPUT_LAYOUT_NDC1HWC0_VALUE) {
            const uint64_t block = InputNdc1hwc0Block();
            const uint64_t c1 = block == 0U ? 0U : static_cast<uint64_t>(cIdx) / block;
            const uint64_t c0 = block == 0U ? 0U : static_cast<uint64_t>(cIdx) - c1 * block;
            return (((static_cast<uint64_t>(nIdx) * static_cast<uint64_t>(tiling_->inD) + static_cast<uint64_t>(dIdx)) *
                         static_cast<uint64_t>(tiling_->inputC1) +
                     c1) *
                        static_cast<uint64_t>(tiling_->inH) +
                    static_cast<uint64_t>(hIdx)) *
                       static_cast<uint64_t>(tiling_->inW) * block +
                   static_cast<uint64_t>(wIdx) * block + c0;
        }
        if (tiling_->dataFormat == FORMAT_NCDHW_VALUE) {
            return (((static_cast<uint64_t>(nIdx) * tiling_->c + cIdx) * tiling_->inD + dIdx) * tiling_->inH + hIdx) *
                       tiling_->inW +
                   wIdx;
        }
        return (((static_cast<uint64_t>(nIdx) * tiling_->inD + dIdx) * tiling_->inH + hIdx) * tiling_->inW + wIdx) *
                   tiling_->c +
               cIdx;
    }

    __aicore__ inline void DecodeOutputIndex(uint64_t linear, int64_t& nIdx, int64_t& od, int64_t& oh, int64_t& ow,
                                             int64_t& cIdx) const
    {
        if (tiling_->dataFormat == FORMAT_NCDHW_VALUE) {
            ow = static_cast<int64_t>(linear % tiling_->outW);
            linear /= tiling_->outW;
            oh = static_cast<int64_t>(linear % tiling_->outH);
            linear /= tiling_->outH;
            od = static_cast<int64_t>(linear % tiling_->outD);
            linear /= tiling_->outD;
            cIdx = static_cast<int64_t>(linear % tiling_->c);
            nIdx = static_cast<int64_t>(linear / tiling_->c);
        } else {
            cIdx = static_cast<int64_t>(linear % tiling_->c);
            linear /= tiling_->c;
            ow = static_cast<int64_t>(linear % tiling_->outW);
            linear /= tiling_->outW;
            oh = static_cast<int64_t>(linear % tiling_->outH);
            linear /= tiling_->outH;
            od = static_cast<int64_t>(linear % tiling_->outD);
            nIdx = static_cast<int64_t>(linear / tiling_->outD);
        }
    }

    __aicore__ inline float ValueToFloat(T value) const
    {
        if constexpr (AscendC::Std::is_same<T, bfloat16_t>::value) {
            return ToFloat(value);
        } else {
            return static_cast<float>(value);
        }
    }

    __aicore__ inline bool IsNan(float value) const { return value != value; }

    __aicore__ inline T NegInfValue() const
    {
        if constexpr (AscendC::Std::is_same<T, bfloat16_t>::value) {
            constexpr uint16_t negInfBits = 0xFF80U;
            return *reinterpret_cast<const bfloat16_t*>(&negInfBits);
        } else if constexpr (AscendC::Std::is_same<T, half>::value) {
            constexpr uint16_t negInfBits = 0xFC00U;
            return *reinterpret_cast<const half*>(&negInfBits);
        } else {
            constexpr uint32_t negInfBits = 0xFF800000U;
            return *reinterpret_cast<const float*>(&negInfBits);
        }
    }

    __aicore__ inline void ReduceValueAtDepth(int64_t nIdx, int64_t id, int64_t oh, int64_t ow, int64_t cIdx,
                                              T& maxValue, float& maxValueFp32)
    {
        for (int64_t kh = 0; kh < tiling_->kH; ++kh) {
            const int64_t ih = DilatedInputH(oh, kh);
            if (!IsOutOfRange(ih, tiling_->inH)) {
                for (int64_t kw = 0; kw < tiling_->kW; ++kw) {
                    const int64_t iw = DilatedInputW(ow, kw);
                    if (!IsOutOfRange(iw, tiling_->inW)) {
                        const uint64_t inputOffset = InputOffset(nIdx, id, ih, iw, cIdx);
                        const T curValue = xGm_.GetValue(inputOffset);
                        const float cur = ValueToFloat(curValue);
                        if (cur > maxValueFp32 || IsNan(cur)) {
                            maxValue = curValue;
                            maxValueFp32 = cur;
                        }
                    }
                }
            }
        }
    }

    __aicore__ inline T ComputeValueAt(int64_t nIdx, int64_t od, int64_t oh, int64_t ow, int64_t cIdx)
    {
        T maxValue = NegInfValue();
        float maxValueFp32 = ValueToFloat(maxValue);
        for (int64_t kd = 0; kd < tiling_->kD; ++kd) {
            const int64_t id = DilatedInputD(od, kd);
            if (!IsOutOfRange(id, tiling_->inD)) {
                ReduceValueAtDepth(nIdx, id, oh, ow, cIdx, maxValue, maxValueFp32);
            }
        }

        return maxValue;
    }

    __aicore__ inline T ComputeValue(uint64_t outLinear)
    {
        int64_t nIdx = 0;
        int64_t od = 0;
        int64_t oh = 0;
        int64_t ow = 0;
        int64_t cIdx = 0;
        DecodeOutputIndex(outLinear, nIdx, od, oh, ow, cIdx);
        return ComputeValueAt(nIdx, od, oh, ow, cIdx);
    }

    TPipe pipe_;
    TQue<TPosition::VECIN, BUFFER_NUM> xInQue_;
    TBuf<TPosition::VECCALC> simtTilingDataBuf_;
    TBuf<TPosition::VECCALC> simtParamBuf_;
    TBuf<TPosition::VECCALC> calcBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    TBuf<TPosition::VECCALC> maskBuf_;
    TQue<TPosition::VECOUT, BUFFER_NUM> yOutQue_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    const MaxPool3DTilingData* tiling_ = nullptr;
};

} // namespace MaxPool3DExp

#endif // MAX_POOL3_D_H_
