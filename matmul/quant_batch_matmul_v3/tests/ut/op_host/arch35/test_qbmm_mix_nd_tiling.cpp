/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <utility>
#include "matmul/quant_batch_matmul_v3/op_host/op_tiling/arch35/adaptive_sliding_window_mix_basic_api_tiling.h"
#include "matmul/quant_batch_matmul_v3/op_host/op_tiling/arch35/quant_batch_matmul_v3_tiling_util.h"

namespace {
class MixCapabilityProbe : public optiling::AdaptiveSlidingWindowMixBasicAPITiling {
public:
    MixCapabilityProbe() : AdaptiveSlidingWindowMixBasicAPITiling(nullptr) {}
    bool Check(const optiling::QuantBatchMatmulInfo& info)
    {
        const auto saved = inputParams_;
        inputParams_ = info;
        const bool result = IsCapable();
        inputParams_ = saved;
        return result;
    }
};

optiling::QuantBatchMatmulInfo MakeNdMixInfo()
{
    optiling::QuantBatchMatmulInfo info{};
    info.bFormat = ge::FORMAT_ND;
    info.aDtype = ge::DT_INT8;
    info.bDtype = ge::DT_INT8;
    info.cDtype = ge::DT_BF16;
    info.scaleDtype = ge::DT_FLOAT;
    info.hasBias = true;
    info.biasDtype = ge::DT_INT32;
    info.isPerChannel = true;
    return info;
}
} // namespace

TEST(QbmmMixNdCapability, OrdinaryMixUsesBlazeForEveryTranspose)
{
    MixCapabilityProbe tiling;
    const bool supported = optiling::IsTensorapiCapable();
    for (auto dtype : {ge::DT_INT8, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_HIFLOAT8}) {
        for (auto output : {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT}) {
            if (dtype == ge::DT_INT8 && output == ge::DT_FLOAT) {
                continue;
            }
            for (int transA : {0, 1}) {
                for (int transB : {0, 1}) {
                    auto info = MakeNdMixInfo();
                    info.aDtype = info.bDtype = dtype;
                    info.cDtype = output;
                    info.biasDtype = ge::DT_FLOAT;
                    info.transA = transA;
                    info.transB = transB;
                    EXPECT_EQ(tiling.Check(info), supported);
                    info.isPerChannel = false;
                    info.isPertoken = true;
                    EXPECT_EQ(tiling.Check(info), supported);
                }
            }
        }
    }
}

TEST(QbmmMixNdCapability, Int8BiasAndFp8MixedPairs)
{
    MixCapabilityProbe tiling;
    const bool supported = optiling::IsTensorapiCapable();
    auto info = MakeNdMixInfo();
    for (auto bias : {ge::DT_INT32, ge::DT_FLOAT, ge::DT_BF16}) {
        info.biasDtype = bias;
        EXPECT_EQ(tiling.Check(info), supported);
    }
    info.aDtype = ge::DT_FLOAT8_E4M3FN;
    info.bDtype = ge::DT_FLOAT8_E5M2;
    info.biasDtype = ge::DT_FLOAT;
    EXPECT_EQ(tiling.Check(info), supported);
    std::swap(info.aDtype, info.bDtype);
    EXPECT_EQ(tiling.Check(info), supported);
    info.isPerChannel = false;
    info.isPerTensor = info.isDoubleScale = true;
    info.perTokenScaleDtype = ge::DT_FLOAT;
    EXPECT_EQ(tiling.Check(info), supported);
}

TEST(QbmmMixNdCapability, CubeAndExcludedModesDoNotEnterOrdinaryMix)
{
    MixCapabilityProbe tiling;
    auto info = MakeNdMixInfo();
    info.isPerBlock = true;
    EXPECT_FALSE(tiling.Check(info));
    info = MakeNdMixInfo();
    info.isMxPerGroup = true;
    EXPECT_FALSE(tiling.Check(info));
    info = MakeNdMixInfo();
    info.aDtype = info.bDtype = ge::DT_INT4;
    EXPECT_FALSE(tiling.Check(info));
    info = MakeNdMixInfo();
    info.cDtype = ge::DT_INT32;
    EXPECT_FALSE(tiling.Check(info));
    info = MakeNdMixInfo();
    info.scaleDtype = ge::DT_UINT64;
    EXPECT_FALSE(tiling.Check(info));
    info = MakeNdMixInfo();
    info.isPerChannel = false;
    info.isPerTensor = true; // INT32 bias stays in the Cube per-tensor route.
    EXPECT_FALSE(tiling.Check(info));
    info.biasDtype = ge::DT_FLOAT;
    EXPECT_EQ(tiling.Check(info), optiling::IsTensorapiCapable());
}

TEST(QbmmMixNdCapability, Int8NzMixUsesBlazeForEveryTranspose)
{
    MixCapabilityProbe tiling;
    const bool supported = optiling::IsTensorapiCapable();
    for (auto output : {ge::DT_FLOAT16, ge::DT_BF16}) {
        for (auto bias : {ge::DT_INT32, ge::DT_FLOAT, output}) {
            for (int transA : {0, 1}) {
                for (int transB : {0, 1}) {
                    auto info = MakeNdMixInfo();
                    info.bFormat = ge::FORMAT_FRACTAL_NZ;
                    info.cDtype = output;
                    info.biasDtype = bias;
                    info.transA = transA;
                    info.transB = transB;
                    EXPECT_EQ(tiling.Check(info), supported);
                    info.isPerChannel = false;
                    info.isPertoken = true;
                    EXPECT_EQ(tiling.Check(info), supported);
                }
            }
        }
    }
}
