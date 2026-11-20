/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Tiling UT for AscendDequant with an independent, hand-derived oracle.
//
// The oracle below re-implements the documented formulas (numpy-broadcastable
// deq_scale with positional alignment, UB / multicore split) directly. It never
// calls the tiling function under test to derive an expectation. Format carries
// no semantics: the origin format is not read, so every case is tiled as ND.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "exe_graph/runtime/continuous_vector.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/tiling_data.h"
#include "kernel_run_context_facker.h"
#include "log/log.h"
#include "platform/platform_infos_def.h"
#include "register/op_impl_registry.h"
#include "ut_op_common.h"

#include "../../../../op_kernel/arch35/ascend_dequant_tilingdata.h"

using namespace ge;

namespace {

constexpr int64_t kUbSize = 262144;
constexpr int64_t kCoreNum = 24;
constexpr int64_t kPhysNodes = 3;
constexpr int64_t kUbBlockSize = 32;
constexpr int64_t kFp32Bytes = 4;
constexpr int64_t kPerBufBytes = (kUbSize / kPhysNodes) & ~(kUbBlockSize - 1); // 87360
constexpr int64_t kPerBufElems = kPerBufBytes / kFp32Bytes;                    // 21840

int64_t CeilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

// Right-aligned numpy broadcast check: scale rank <= x rank, and every aligned
// axis must equal x's axis or be 1. Alignment to x's rank is returned in `aligned`.
bool BroadcastAligned(const std::vector<int64_t>& x, const std::vector<int64_t>& scale, std::vector<int64_t>& aligned)
{
    const int64_t xRank = static_cast<int64_t>(x.size());
    const int64_t sRank = static_cast<int64_t>(scale.size());
    if (sRank > xRank) {
        return false;
    }
    const int64_t offset = xRank - sRank;
    aligned.assign(static_cast<size_t>(xRank), 1);
    for (int64_t d = 0; d < sRank; ++d) {
        const int64_t sLen = scale[static_cast<size_t>(d)];
        const int64_t xLen = x[static_cast<size_t>(offset + d)];
        if (sLen != 1 && sLen != xLen) {
            return false;
        }
        aligned[static_cast<size_t>(offset + d)] = sLen;
    }
    return true;
}

struct Normalized {
    std::vector<int64_t> maxBro;
    std::vector<std::vector<int64_t>> in;
    std::vector<std::vector<int64_t>> out;
    std::vector<std::vector<int64_t>> paddedIn;
    std::vector<std::vector<int64_t>> paddedOut;
};

void PadAndSqueeze(const std::vector<std::vector<int64_t>>& inputs, const std::vector<std::vector<int64_t>>& outputs,
                   Normalized& n)
{
    int64_t maxRank = 0;
    for (const auto& s : inputs) {
        maxRank = std::max(maxRank, static_cast<int64_t>(s.size()));
    }
    for (const auto& s : outputs) {
        maxRank = std::max(maxRank, static_cast<int64_t>(s.size()));
    }
    auto pad = [maxRank](const std::vector<std::vector<int64_t>>& shapes) {
        std::vector<std::vector<int64_t>> res(shapes.size());
        for (size_t i = 0; i < shapes.size(); ++i) {
            res[i].assign(static_cast<size_t>(maxRank - static_cast<int64_t>(shapes[i].size())), 1);
            res[i].insert(res[i].end(), shapes[i].begin(), shapes[i].end());
        }
        return res;
    };
    n.paddedIn = pad(inputs);
    n.paddedOut = pad(outputs);

    n.maxBro.clear();
    n.in.assign(inputs.size(), {});
    n.out.assign(outputs.size(), {});
    for (int64_t d = 0; d < maxRank; ++d) {
        bool allOne = true;
        bool hasZero = false;
        int64_t maxDim = 0;
        for (const auto& s : n.paddedIn) {
            allOne = allOne && (s[static_cast<size_t>(d)] == 1);
            hasZero = hasZero || (s[static_cast<size_t>(d)] == 0);
            maxDim = std::max(maxDim, s[static_cast<size_t>(d)]);
        }
        for (const auto& s : n.paddedOut) {
            allOne = allOne && (s[static_cast<size_t>(d)] == 1);
            hasZero = hasZero || (s[static_cast<size_t>(d)] == 0);
            maxDim = std::max(maxDim, s[static_cast<size_t>(d)]);
        }
        if (!allOne) {
            n.maxBro.push_back(hasZero ? 0 : maxDim);
            for (size_t i = 0; i < n.paddedIn.size(); ++i) {
                n.in[i].push_back(n.paddedIn[i][static_cast<size_t>(d)]);
            }
            for (size_t i = 0; i < n.paddedOut.size(); ++i) {
                n.out[i].push_back(n.paddedOut[i][static_cast<size_t>(d)]);
            }
        }
    }
    if (n.maxBro.empty()) {
        n.maxBro.push_back(1);
        for (auto& s : n.in) {
            s.push_back(1);
        }
        for (auto& s : n.out) {
            s.push_back(1);
        }
    }
}

bool CheckBroadcast(const std::vector<std::vector<int64_t>>& in, const std::vector<std::vector<int64_t>>& out,
                    int64_t rank)
{
    for (int64_t d = 0; d < rank; ++d) {
        int64_t ref = -1;
        for (const auto& s : in) {
            const int64_t v = s[static_cast<size_t>(d)];
            if (v == 1) {
                continue;
            }
            if (ref == -1) {
                ref = v;
            } else if (v != ref) {
                return false;
            }
        }
        for (const auto& s : out) {
            const int64_t v = s[static_cast<size_t>(d)];
            if (v == 1) {
                continue;
            }
            if (ref == -1) {
                ref = v;
            } else if (v != ref) {
                return false;
            }
        }
        if (ref != -1) {
            for (const auto& s : out) {
                if (s[static_cast<size_t>(d)] == 1) {
                    return false;
                }
            }
        }
    }
    return true;
}

std::vector<int64_t> MergeByGroups(const std::vector<int64_t>& s, const std::vector<std::vector<int64_t>>& groups)
{
    std::vector<int64_t> merged;
    merged.reserve(groups.size());
    for (const auto& g : groups) {
        int64_t prod = 1;
        for (int64_t k = g[0]; k <= g[1]; ++k) {
            prod *= s[static_cast<size_t>(k)];
        }
        merged.push_back(prod);
    }
    return merged;
}

void MergeAxis(std::vector<int64_t>& maxBro, std::vector<std::vector<int64_t>>& in,
               std::vector<std::vector<int64_t>>& out)
{
    const int64_t oldRank = static_cast<int64_t>(maxBro.size());
    if (oldRank <= 1) {
        return;
    }
    std::vector<std::vector<int64_t>> groups;
    int64_t low = oldRank - 1;
    int64_t high = oldRank - 1;
    for (int64_t d = oldRank - 2; d >= 0; --d) {
        int64_t maxProd = 1;
        for (int64_t k = d; k <= high; ++k) {
            maxProd *= maxBro[static_cast<size_t>(k)];
        }
        bool mergeable = true;
        for (size_t i = 0; i < in.size() && mergeable; ++i) {
            int64_t prod = 1;
            for (int64_t k = d; k <= high; ++k) {
                prod *= in[i][static_cast<size_t>(k)];
            }
            mergeable = (prod == 1 || prod == maxProd);
        }
        for (size_t i = 0; i < out.size() && mergeable; ++i) {
            int64_t prod = 1;
            for (int64_t k = d; k <= high; ++k) {
                prod *= out[i][static_cast<size_t>(k)];
            }
            mergeable = (prod == 1 || prod == maxProd);
        }
        if (mergeable) {
            low = d;
        } else {
            groups.push_back({low, high});
            low = d;
            high = d;
        }
    }
    groups.push_back({low, high});
    std::reverse(groups.begin(), groups.end());

    maxBro = MergeByGroups(maxBro, groups);
    for (auto& s : in) {
        s = MergeByGroups(s, groups);
    }
    for (auto& s : out) {
        s = MergeByGroups(s, groups);
    }
}

SplitResult FindSplit(const std::vector<int64_t>& maxBro)
{
    SplitResult out{};
    const int64_t rank = static_cast<int64_t>(maxBro.size());
    int64_t inner = 1;
    for (int64_t k = rank - 1; k >= 0; --k) {
        if (maxBro[static_cast<size_t>(k)] * inner > kPerBufElems) {
            out.ubFactor = kPerBufElems / inner;
            out.ubOuter = CeilDiv(maxBro[static_cast<size_t>(k)], out.ubFactor);
            const int64_t rem = maxBro[static_cast<size_t>(k)] % out.ubFactor;
            out.ubTail = (rem == 0) ? out.ubFactor : rem;
            out.ubSplitIdx = k;
            return out;
        }
        if (k == 0) {
            out.ubSplitIdx = 0;
            out.ubFactor = maxBro[0];
            out.ubOuter = 1;
            out.ubTail = out.ubFactor;
            return out;
        }
        inner *= maxBro[static_cast<size_t>(k)];
    }
    return out;
}

MultiCoreResult MultiCoreSplit(const std::vector<int64_t>& maxBro, const SplitResult& split)
{
    MultiCoreResult out{};
    int64_t outerProd = 1;
    for (int64_t j = 0; j < split.ubSplitIdx; ++j) {
        outerProd *= maxBro[static_cast<size_t>(j)];
    }
    out.totalTiles = outerProd * split.ubOuter;
    out.usedCoreNum = (out.totalTiles < kCoreNum) ? out.totalTiles : kCoreNum;
    out.mainTiles = CeilDiv(out.totalTiles, out.usedCoreNum);
    out.mainCoreNum = out.totalTiles - (out.mainTiles - 1) * out.usedCoreNum;
    return out;
}

std::vector<int64_t> CalcStrides(const std::vector<int64_t>& shape, const std::vector<int64_t>& maxBro, bool u64View)
{
    const int64_t rank = static_cast<int64_t>(shape.size());
    std::vector<int64_t> strides(static_cast<size_t>(rank), 0);
    int64_t stride = 1;
    for (int64_t d = rank - 1; d >= 0; --d) {
        const bool brc = (shape[static_cast<size_t>(d)] == 1 && maxBro[static_cast<size_t>(d)] != 1);
        strides[static_cast<size_t>(d)] = brc ? 0 : stride;
        if (!brc) {
            stride *= shape[static_cast<size_t>(d)];
        }
    }
    if (u64View) {
        for (auto& s : strides) {
            s *= 2;
        }
    }
    return strides;
}

struct OracleCase {
    std::vector<int64_t> xShape;
    std::vector<int64_t> scaleShape;
    ge::DataType xDtype = ge::DT_INT32;
    ge::DataType scaleDtype = ge::DT_UINT64;
    // Declared y dtype; DT_UNDEFINED means "the dtype derived from attr dtype".
    ge::DataType yDtype = ge::DT_UNDEFINED;
    int64_t dtypeAttr = 0; // prototype default: float32 output
    bool sqrtMode = false;
    bool reluFlag = false;
    ge::Format xFormat = ge::FORMAT_ND;
    ge::Format yFormat = ge::FORMAT_ND;
    ge::Format scaleFormat = ge::FORMAT_ND;
};

struct OracleResult {
    bool ok = false;
    bool empty = false;
    int64_t rank = 0;
    int64_t R = 0;
    SplitResult split{};
    MultiCoreResult multicore{};
    std::vector<int64_t> maxBro;
    std::vector<std::vector<int64_t>> inShapes;
    std::vector<std::vector<int64_t>> inStrides;
    std::vector<std::vector<int64_t>> outShapes;
    std::vector<std::vector<int64_t>> outStrides;
    int64_t sqrtMode = 0;
    int64_t reluFlag = 0;
    int64_t outDtype = 0;
};

// attr dtype -> y dtype (0 -> float32, 1 -> float16); DT_UNDEFINED when out of range.
ge::DataType AttrOutputDtype(int64_t dtypeAttr)
{
    if (dtypeAttr == 0) {
        return ge::DT_FLOAT;
    }
    if (dtypeAttr == 1) {
        return ge::DT_FLOAT16;
    }
    return ge::DT_UNDEFINED;
}

OracleResult ComputeOracle(const OracleCase& c)
{
    OracleResult r;
    if (c.xDtype != ge::DT_INT32 || c.scaleDtype != ge::DT_UINT64) {
        return r;
    }
    const int64_t xRank = static_cast<int64_t>(c.xShape.size());
    if (xRank < 1 || xRank > 8) {
        return r;
    }
    const ge::DataType attrYDtype = AttrOutputDtype(c.dtypeAttr);
    if (attrYDtype == ge::DT_UNDEFINED) {
        return r;
    }
    if (c.yDtype != ge::DT_UNDEFINED && c.yDtype != attrYDtype) {
        return r;
    }
    for (const auto& s : {c.xShape, c.scaleShape}) {
        for (int64_t d : s) {
            if (d < 0) {
                return r;
            }
        }
    }
    std::vector<int64_t> aligned;
    if (!BroadcastAligned(c.xShape, c.scaleShape, aligned)) {
        return r;
    }
    Normalized n;
    PadAndSqueeze({c.xShape, aligned}, {c.xShape}, n);
    const int64_t keyRank = static_cast<int64_t>(n.maxBro.size());
    if (keyRank > 8 || !CheckBroadcast(n.paddedIn, n.paddedOut, keyRank)) {
        return r;
    }
    r.ok = true;
    r.R = (keyRank <= 4) ? 4 : 8;
    r.sqrtMode = c.sqrtMode ? 1 : 0;
    r.reluFlag = c.reluFlag ? 1 : 0;
    r.outDtype = c.dtypeAttr;

    bool empty = false;
    for (int64_t d : c.xShape) {
        empty = empty || (d == 0);
    }
    r.empty = empty;
    if (empty) {
        r.rank = keyRank;
        r.split = SplitResult{};
        r.multicore = MultiCoreResult{};
        r.multicore.usedCoreNum = 0;
    } else {
        MergeAxis(n.maxBro, n.in, n.out);
        r.rank = static_cast<int64_t>(n.maxBro.size());
        r.split = FindSplit(n.maxBro);
        r.multicore = MultiCoreSplit(n.maxBro, r.split);
    }

    const int64_t R = r.R;
    const int64_t rank = r.rank;
    r.maxBro.assign(static_cast<size_t>(R), 1);
    for (int64_t d = 0; d < rank; ++d) {
        r.maxBro[static_cast<size_t>(d)] = n.maxBro[static_cast<size_t>(d)];
    }
    r.inShapes.assign(n.in.size(), std::vector<int64_t>(static_cast<size_t>(R), 1));
    r.inStrides.assign(n.in.size(), std::vector<int64_t>(static_cast<size_t>(R), 0));
    for (size_t i = 0; i < n.in.size(); ++i) {
        // The uint64 deq_scale is read through the low 32-bit word view.
        const auto strides = CalcStrides(n.in[i], n.maxBro, i == 1);
        for (int64_t d = 0; d < rank; ++d) {
            r.inShapes[i][static_cast<size_t>(d)] = n.in[i][static_cast<size_t>(d)];
            r.inStrides[i][static_cast<size_t>(d)] = strides[static_cast<size_t>(d)];
        }
    }
    r.outShapes.assign(n.out.size(), std::vector<int64_t>(static_cast<size_t>(R), 1));
    r.outStrides.assign(n.out.size(), std::vector<int64_t>(static_cast<size_t>(R), 0));
    for (size_t i = 0; i < n.out.size(); ++i) {
        const auto strides = CalcStrides(n.out[i], n.maxBro, false);
        for (int64_t d = 0; d < rank; ++d) {
            r.outShapes[i][static_cast<size_t>(d)] = n.out[i][static_cast<size_t>(d)];
            r.outStrides[i][static_cast<size_t>(d)] = strides[static_cast<size_t>(d)];
        }
    }
    return r;
}

gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape s;
    for (int64_t d : dims) {
        s.MutableOriginShape().AppendDim(d);
        s.MutableStorageShape().AppendDim(d);
    }
    return s;
}

void InitPlatform(fe::PlatFormInfos& pi)
{
    pi.Init();
    std::map<std::string, std::string> socInfos;
    socInfos["ai_core_cnt"] = std::to_string(kCoreNum);
    socInfos["cube_core_cnt"] = std::to_string(kCoreNum / 2);
    socInfos["vector_core_cnt"] = std::to_string(kCoreNum);
    socInfos["core_type_list"] = "AICore";
    socInfos["l2_size"] = "33554432";
    pi.SetPlatformRes("SoCInfo", socInfos);

    std::map<std::string, std::string> aicoreSpec;
    aicoreSpec["ub_size"] = std::to_string(kUbSize);
    aicoreSpec["l0_a_size"] = "65536";
    aicoreSpec["l0_b_size"] = "65536";
    aicoreSpec["l0_c_size"] = "131072";
    aicoreSpec["l1_size"] = "524288";
    pi.SetPlatformRes("AICoreSpec", aicoreSpec);
    pi.SetCoreNumByCoreType("AICore");

    std::map<std::string, std::string> version;
    version["Short_SoC_version"] = "Ascend950";
    version["NpuArch"] = "3510";
    pi.SetPlatformRes("version", version);
}

class AscendDequantTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AscendDequantTilingTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "AscendDequantTilingTest TearDown" << std::endl; }

    bool InvokeTiling(const OracleCase& c)
    {
        const auto* impl = gert::OpImplRegistry::GetInstance().GetOpImpl("AscendDequant");
        if (impl == nullptr || impl->tiling == nullptr) {
            return false;
        }
        const ge::DataType declaredY = (c.yDtype == ge::DT_UNDEFINED) ? AttrOutputDtype(c.dtypeAttr) : c.yDtype;
        gert::StorageShape xShape = MakeStorageShape(c.xShape);
        gert::StorageShape sShape = MakeStorageShape(c.scaleShape);
        gert::StorageShape yShape = MakeStorageShape(c.xShape);
        tilingData_ = gert::TilingData::CreateCap(4096);
        auto wsHolder = gert::ContinuousVector::Create<size_t>(16);
        auto* ws = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());
        InitPlatform(platformInfo_);
        struct EmptyCompileInfo {
        } compileInfo;

        auto holder = gert::TilingContextFaker()
                          .SetOpType("AscendDequant")
                          .NodeIoNum(2, 1)
                          .IrInstanceNum({1, 1})
                          .InputShapes({&xShape, &sShape})
                          .OutputShapes({&yShape})
                          .CompileInfo(&compileInfo)
                          .PlatformInfo(reinterpret_cast<char*>(&platformInfo_))
                          .NodeInputTd(0, c.xDtype, c.xFormat, c.xFormat)
                          .NodeInputTd(1, c.scaleDtype, c.scaleFormat, c.scaleFormat)
                          .NodeOutputTd(0, declaredY, c.yFormat, c.yFormat)
                          .NodeAttrs({{"sqrt_mode", Ops::NN::AnyValue::CreateFrom<bool>(c.sqrtMode)},
                                      {"relu_flag", Ops::NN::AnyValue::CreateFrom<bool>(c.reluFlag)},
                                      {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(c.dtypeAttr)}})
                          .TilingData(tilingData_.get())
                          .Workspace(ws)
                          .Build();
        auto* ctx = holder.GetContext<gert::TilingContext>();
        if (ctx == nullptr) {
            return false;
        }
        InitPlatform(*ctx->GetPlatformInfo());
        const ge::graphStatus ret = impl->tiling(ctx);
        blockDim_ = ctx->GetBlockDim();
        tilingKey_ = ctx->GetTilingKey();
        ws0_ = (ctx->GetWorkspaceSizes(1) != nullptr) ? ctx->GetWorkspaceSizes(1)[0] : static_cast<size_t>(-1);
        return ret == ge::GRAPH_SUCCESS;
    }

    template <int64_t R>
    void ExpectMatchesOracle(const OracleResult& e)
    {
        const auto* td = reinterpret_cast<const AscendDequantTilingData<R>*>(
            reinterpret_cast<gert::TilingData*>(tilingData_.get())->GetData());
        ASSERT_NE(td, nullptr);
        EXPECT_EQ(td->rank, e.rank);
        EXPECT_EQ(td->perBufBytes, kPerBufBytes);
        EXPECT_EQ(td->numInputs, kMaxInputSlots);
        EXPECT_EQ(td->numOutputs, kMaxOutputSlots);
        EXPECT_EQ(td->split.ubSplitIdx, e.split.ubSplitIdx);
        EXPECT_EQ(td->split.ubFactor, e.split.ubFactor);
        EXPECT_EQ(td->split.ubOuter, e.split.ubOuter);
        EXPECT_EQ(td->split.ubTail, e.split.ubTail);
        EXPECT_EQ(td->multicore.usedCoreNum, e.multicore.usedCoreNum);
        EXPECT_EQ(td->multicore.totalTiles, e.multicore.totalTiles);
        EXPECT_EQ(td->multicore.mainTiles, e.multicore.mainTiles);
        EXPECT_EQ(td->multicore.mainCoreNum, e.multicore.mainCoreNum);
        for (int64_t d = 0; d < R; ++d) {
            EXPECT_EQ(td->maxBroShape[d], e.maxBro[static_cast<size_t>(d)]) << "maxBroShape[" << d << "]";
        }
        for (int64_t i = 0; i < kMaxInputSlots; ++i) {
            for (int64_t d = 0; d < R; ++d) {
                EXPECT_EQ(td->inputShapes[i][d], e.inShapes[static_cast<size_t>(i)][static_cast<size_t>(d)])
                    << "inputShapes[" << i << "][" << d << "]";
                EXPECT_EQ(td->inputStrides[i][d], e.inStrides[static_cast<size_t>(i)][static_cast<size_t>(d)])
                    << "inputStrides[" << i << "][" << d << "]";
            }
        }
        for (int64_t i = 0; i < kMaxOutputSlots; ++i) {
            for (int64_t d = 0; d < R; ++d) {
                EXPECT_EQ(td->outputShapes[i][d], e.outShapes[static_cast<size_t>(i)][static_cast<size_t>(d)])
                    << "outputShapes[" << i << "][" << d << "]";
                EXPECT_EQ(td->outputStrides[i][d], e.outStrides[static_cast<size_t>(i)][static_cast<size_t>(d)])
                    << "outputStrides[" << i << "][" << d << "]";
            }
        }
        EXPECT_EQ(td->sqrtMode, e.sqrtMode);
        EXPECT_EQ(td->reluFlag, e.reluFlag);
        EXPECT_EQ(td->outDtype, e.outDtype);
    }

    void RunSuccess(const OracleCase& c)
    {
        ASSERT_TRUE(InvokeTiling(c));
        const OracleResult e = ComputeOracle(c);
        ASSERT_TRUE(e.ok);
        EXPECT_EQ(ws0_, static_cast<size_t>(0));
        EXPECT_EQ(blockDim_, static_cast<uint32_t>(e.multicore.usedCoreNum == 0 ? 1 : e.multicore.usedCoreNum));
        EXPECT_EQ(tilingKey_, static_cast<uint64_t>(e.R == 4 ? 0 : 1));
        if (e.R == 4) {
            ExpectMatchesOracle<4>(e);
        } else {
            ExpectMatchesOracle<8>(e);
        }
    }

    // The origin format carries no semantics: tiling the same case as ND and with
    // the format under test must yield identical blockDim / tilingKey / workspace
    // and byte-identical TilingData.
    void RunSameAsNd(const OracleCase& c)
    {
        OracleCase nd = c;
        nd.xFormat = ge::FORMAT_ND;
        nd.yFormat = ge::FORMAT_ND;
        nd.scaleFormat = ge::FORMAT_ND;
        RunSuccess(nd);

        const OracleResult e = ComputeOracle(nd);
        ASSERT_TRUE(e.ok);
        const size_t bytes = (e.R == 4) ? sizeof(AscendDequantTilingData<4>) : sizeof(AscendDequantTilingData<8>);
        std::vector<uint8_t> ndData(bytes);
        std::memcpy(ndData.data(), reinterpret_cast<gert::TilingData*>(tilingData_.get())->GetData(), bytes);
        const uint32_t ndBlockDim = blockDim_;
        const uint64_t ndTilingKey = tilingKey_;
        const size_t ndWs0 = ws0_;

        RunSuccess(c);
        EXPECT_EQ(blockDim_, ndBlockDim);
        EXPECT_EQ(tilingKey_, ndTilingKey);
        EXPECT_EQ(ws0_, ndWs0);
        EXPECT_EQ(std::memcmp(ndData.data(), reinterpret_cast<gert::TilingData*>(tilingData_.get())->GetData(), bytes),
                  0)
            << "non-ND format produced different TilingData than ND";
    }

    fe::PlatFormInfos platformInfo_;
    std::unique_ptr<uint8_t[]> tilingData_;
    uint32_t blockDim_ = 0;
    uint64_t tilingKey_ = 0;
    size_t ws0_ = 0;
};

TEST_F(AscendDequantTilingTest, Rank1PerChannel)
{
    OracleCase c{{8}, {8}};
    RunSuccess(c);
    // hand-computed: perBufBytes = (262144/3)&~31 = 87360
    EXPECT_EQ(kPerBufBytes, 87360);
}

TEST_F(AscendDequantTilingTest, Rank2PerTensor)
{
    OracleCase c{{2, 4}, {1}};
    RunSuccess(c);
}

TEST_F(AscendDequantTilingTest, Rank2PerChannel)
{
    OracleCase c{{4, 8}, {8}};
    RunSuccess(c);
    const auto* td = reinterpret_cast<const AscendDequantTilingData<4>*>(
        reinterpret_cast<gert::TilingData*>(tilingData_.get())->GetData());
    ASSERT_NE(td, nullptr);
    EXPECT_EQ(td->rank, 2);
    EXPECT_EQ(td->maxBroShape[0], 4);
    EXPECT_EQ(td->maxBroShape[1], 8);
    EXPECT_EQ(td->inputStrides[1][0], 0);
    EXPECT_EQ(td->inputStrides[1][1], 2); // uint64 low-word view
}

// A 1-D deq_scale binds to the LAST axis of x, whatever its length means for
// the other axes: x={4,8} accepts 8 and rejects 4 (a non-last axis length).
TEST_F(AscendDequantTilingTest, OneDimScaleBindsLastAxis)
{
    OracleCase c{{4, 8}, {8}};
    RunSuccess(c);

    OracleCase nonLast = c;
    nonLast.scaleShape = {4};
    EXPECT_FALSE(InvokeTiling(nonLast));
}

// Arbitrary broadcast axes are allowed: x={2,3,4} with scale={2,1,4} is legal
// because the right-aligned axes are (2,1,4) vs (2,3,4).
TEST_F(AscendDequantTilingTest, Rank3BroadcastMiddleAxis)
{
    OracleCase c{{2, 3, 4}, {2, 1, 4}};
    RunSuccess(c);
    const auto* td = reinterpret_cast<const AscendDequantTilingData<4>*>(
        reinterpret_cast<gert::TilingData*>(tilingData_.get())->GetData());
    ASSERT_NE(td, nullptr);
    EXPECT_EQ(td->rank, 3);
    EXPECT_EQ(td->maxBroShape[0], 2);
    EXPECT_EQ(td->maxBroShape[1], 3);
    EXPECT_EQ(td->maxBroShape[2], 4);
    EXPECT_EQ(td->inputShapes[1][0], 2);
    EXPECT_EQ(td->inputShapes[1][1], 1);
    EXPECT_EQ(td->inputShapes[1][2], 4);
    EXPECT_EQ(td->inputStrides[1][0], 8); // one (1,1,4) plane: 4 uint64 -> 8 words
    EXPECT_EQ(td->inputStrides[1][1], 0); // broadcast axis
    EXPECT_EQ(td->inputStrides[1][2], 2); // uint64 low-word view
}

// x={2,32,4,4} with a 1-D scale of length 4 (last axis) is legal; length 32
// (a non-last axis) is not.
TEST_F(AscendDequantTilingTest, Rank4OneDimScaleLastAxisOnly)
{
    OracleCase c{{2, 32, 4, 4}, {4}};
    RunSuccess(c);

    OracleCase nonLast = c;
    nonLast.scaleShape = {32};
    EXPECT_FALSE(InvokeTiling(nonLast));
}

// Full same-rank broadcast form with a tail axis and a broadcast head axis.
TEST_F(AscendDequantTilingTest, Rank4AlignedScale)
{
    OracleCase c{{2, 32, 4, 4}, {1, 32, 1, 1}};
    RunSuccess(c);
}

// Scale rank greater than x rank is never broadcastable.
TEST_F(AscendDequantTilingTest, RejectsScaleRankAboveX)
{
    OracleCase c{{4, 8}, {1, 4, 8}};
    EXPECT_FALSE(InvokeTiling(c));
}

TEST_F(AscendDequantTilingTest, Rank5PerTensorRoutesKey1)
{
    OracleCase c{{2, 3, 4, 5, 6}, {1}};
    RunSuccess(c);
    const auto* td = reinterpret_cast<const AscendDequantTilingData<8>*>(
        reinterpret_cast<gert::TilingData*>(tilingData_.get())->GetData());
    ASSERT_NE(td, nullptr);
    EXPECT_EQ(td->rank, 1);    // merge collapses the per-tensor form
    EXPECT_EQ(tilingKey_, 1U); // branch still follows the pre-merge rank 5
}

TEST_F(AscendDequantTilingTest, Rank5PerChannelRoutesKey1)
{
    OracleCase c{{2, 3, 4, 5, 6}, {6}}; // 1-D scale binds to the last axis
    RunSuccess(c);
    EXPECT_EQ(tilingKey_, 1U);
}

TEST_F(AscendDequantTilingTest, Rank6PerChannelLastAxis)
{
    OracleCase c{{2, 3, 2, 2, 2, 8}, {8}};
    RunSuccess(c);
    EXPECT_EQ(tilingKey_, 1U);
}

TEST_F(AscendDequantTilingTest, Rank5LargeShapeMultiCore)
{
    OracleCase c{{2, 3, 4, 5, 4096}, {1, 3, 1, 1, 1}}; // broadcast on dim1
    RunSuccess(c);
    const OracleResult e = ComputeOracle(c);
    ASSERT_TRUE(e.ok);
    EXPECT_EQ(e.split.ubSplitIdx, 2); // the dim1 boundary is the only non-mergeable one
    EXPECT_EQ(e.split.ubFactor, 21840);
    EXPECT_EQ(e.split.ubOuter, 4);
    EXPECT_EQ(e.multicore.totalTiles, 24);
    EXPECT_EQ(e.multicore.usedCoreNum, 24);
    EXPECT_EQ(e.multicore.mainTiles, 1);
    EXPECT_EQ(e.multicore.mainCoreNum, 24);
}

TEST_F(AscendDequantTilingTest, EmptyRank2)
{
    OracleCase c{{0, 3}, {3}};
    RunSuccess(c);
    const auto* td = reinterpret_cast<const AscendDequantTilingData<4>*>(
        reinterpret_cast<gert::TilingData*>(tilingData_.get())->GetData());
    ASSERT_NE(td, nullptr);
    EXPECT_EQ(td->multicore.usedCoreNum, 0);
    EXPECT_EQ(blockDim_, 1U);
}

TEST_F(AscendDequantTilingTest, EmptyRank5KeepsKey1)
{
    OracleCase c{{0, 2, 3, 4, 5}, {1}};
    RunSuccess(c);
    EXPECT_EQ(tilingKey_, 1U);
    EXPECT_EQ(blockDim_, 1U);
}

// No format validation: a registered non-ND format must tile exactly like the
// same shapes declared as ND, because broadcasting is positional.
TEST_F(AscendDequantTilingTest, AcceptsRegisteredNonNdFormats)
{
    for (const ge::Format format : {ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NCDHW, ge::FORMAT_NDHWC}) {
        OracleCase c{{2, 32, 4, 4}, {4}};
        c.xFormat = format;
        c.yFormat = format;
        RunSameAsNd(c);
    }
}

TEST_F(AscendDequantTilingTest, AcceptsNc1hwc0Format)
{
    OracleCase c{{2, 32, 4, 4}, {4}};
    c.xFormat = ge::FORMAT_NC1HWC0;
    c.yFormat = ge::FORMAT_NC1HWC0;
    RunSameAsNd(c);
}

// The deq_scale format is not validated either.
TEST_F(AscendDequantTilingTest, AcceptsNonNdScaleFormat)
{
    OracleCase c{{2, 32, 4, 4}, {4}};
    c.scaleFormat = ge::FORMAT_NHWC;
    RunSameAsNd(c);
}

TEST_F(AscendDequantTilingTest, RejectsScaleLengthMismatch)
{
    OracleCase c{{2, 3}, {4}};
    EXPECT_FALSE(InvokeTiling(c));
}

// (x=int32) x (deq_scale=uint64) x (y=float16|float32): both prototype
// combinations must tile, and the output dtype must reach TilingData.
TEST_F(AscendDequantTilingTest, BothOutputDtypeCombinations)
{
    struct Combo {
        int64_t dtypeAttr;
        int64_t outDtype;
        int64_t scaleStrideHigh; // deq_scale last-axis stride (uint64 word view -> 2)
    };
    const Combo combos[2] = {
        {1, 1, 2}, // uint64 scale -> float16
        {0, 0, 2}, // uint64 scale -> float32
    };
    for (const Combo& combo : combos) {
        OracleCase c{{4, 8}, {8}};
        c.scaleDtype = ge::DT_UINT64;
        c.dtypeAttr = combo.dtypeAttr;
        RunSuccess(c);
        const auto* td = reinterpret_cast<const AscendDequantTilingData<4>*>(
            reinterpret_cast<gert::TilingData*>(tilingData_.get())->GetData());
        ASSERT_NE(td, nullptr);
        EXPECT_EQ(td->outDtype, combo.outDtype);
        // per-channel scale on the last axis: uint64 is read through the uint32 word view
        EXPECT_EQ(td->inputStrides[1][1], combo.scaleStrideHigh);
    }
}

// deq_scale is uint64 only: a float16 declaration must be rejected.
TEST_F(AscendDequantTilingTest, RejectsFloat16ScaleDtype)
{
    OracleCase c{{2, 3}, {3}};
    c.scaleDtype = ge::DT_FLOAT16;
    c.dtypeAttr = 1;
    EXPECT_FALSE(InvokeTiling(c));
}

TEST_F(AscendDequantTilingTest, RejectsDtypeAttrOutOfRange)
{
    OracleCase c{{2, 3}, {3}};
    c.dtypeAttr = 2;
    c.yDtype = ge::DT_FLOAT;
    EXPECT_FALSE(InvokeTiling(c));
}

TEST_F(AscendDequantTilingTest, RejectsOutputDtypeContradictingAttr)
{
    // dtype=0 asks for float32, but the declared y is float16.
    OracleCase c{{2, 3}, {3}};
    c.dtypeAttr = 0;
    c.yDtype = ge::DT_FLOAT16;
    EXPECT_FALSE(InvokeTiling(c));

    // dtype=1 asks for float16, but the declared y is float32.
    OracleCase c2{{2, 3}, {3}};
    c2.dtypeAttr = 1;
    c2.yDtype = ge::DT_FLOAT;
    EXPECT_FALSE(InvokeTiling(c2));
}

TEST_F(AscendDequantTilingTest, RejectsInvalidScaleDtype)
{
    OracleCase c{{2, 3}, {3}};
    c.scaleDtype = ge::DT_FLOAT;
    EXPECT_FALSE(InvokeTiling(c));
}

TEST_F(AscendDequantTilingTest, RejectsInvalidInputDtype)
{
    OracleCase c{{2, 3}, {3}};
    c.xDtype = ge::DT_FLOAT;
    EXPECT_FALSE(InvokeTiling(c));
}

TEST_F(AscendDequantTilingTest, RejectsRank9)
{
    OracleCase c{{1, 1, 1, 1, 1, 1, 1, 1, 1}, {1}};
    EXPECT_FALSE(InvokeTiling(c));
}

TEST_F(AscendDequantTilingTest, RejectsNegativeDim)
{
    OracleCase c{{-1, 3}, {3}};
    EXPECT_FALSE(InvokeTiling(c));
}

} // namespace
