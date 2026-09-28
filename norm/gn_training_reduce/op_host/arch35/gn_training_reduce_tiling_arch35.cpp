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
 * \file gn_training_reduce_tiling_arch35.cpp
 * \brief GNTrainingReduce host tiling implementation for ascend950 (arch35).
 */

#include "../../op_kernel/arch35/gn_training_reduce_tiling_struct.h"

#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "gn_training_reduce_tiling_arch35.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>

using namespace Ops::Base;

namespace optiling {

namespace {

constexpr const char* kOpName = "GNTrainingReduce";
constexpr int32_t kMaxPatternRank = GN_TRAINING_REDUCE_MAX_PATTERN_RANK;

// UB buffer 份数。
constexpr int64_t kPpre = 2;                   // preInBuf + preReduceResult
constexpr int64_t kPpreExt = 1;                // preReduceResultTail
constexpr int64_t kPpost = 1;                  // outBuf
constexpr int64_t kCacheBufUbSize = 16 * 1024; // 二分缓存树固定 16KB

// tilingKey 位编码：GET_TPL_TILING_KEY(isGroup, isEmptyTensor)
//   base  = 0<<0 | 0<<1 = 0
//   group = 1<<0 | 0<<1 = 1
//   empty = 0<<0 | 1<<1 = 2
constexpr uint64_t kTilingKeyBase = 0;
constexpr uint64_t kTilingKeyGroup = 1;
constexpr uint64_t kTilingKeyEmpty = 2;

inline int64_t CeilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }
inline int64_t AlignUp(int64_t v, int64_t f) { return CeilDiv(v, f) * f; }
inline int64_t AlignDown(int64_t v, int64_t f) { return (v / f) * f; }

struct AxisView {
    int64_t size;
    int64_t stride;
    bool isR;
};

// 取 tiling data 数据区；调用方若把整个 TilingData buffer（含头）清零，数据区紧
// 随 gert::TilingData 头之后，此处按需重建头（与 TileD 参考实现一致）。
template <typename T>
T* AcquireTilingData(gert::TilingContext* context)
{
    auto* raw = context->GetRawTilingData();
    if (raw == nullptr) {
        return nullptr;
    }
    if (raw->GetData() == nullptr) {
        raw->Init(sizeof(T), reinterpret_cast<uint8_t*>(raw) + sizeof(gert::TilingData));
    } else if (raw->GetCapacity() < sizeof(T)) {
        return nullptr;
    }
    raw->SetDataSize(sizeof(T));
    return reinterpret_cast<T*>(raw->GetData());
}

ge::graphStatus ReadPlatform(gert::TilingContext* context, int64_t& coreNum, int64_t& ubSize)
{
    // 硬件参数全走 platform 接口：优先从 GetPlatformInfo 取 coreNum / ubSize；
    // platform 不可用或未给出有效值时才回退 CompileInfo。
    auto* platformInfo = context->GetPlatformInfo();
    if (platformInfo != nullptr) {
        platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);
        coreNum = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
        uint64_t ub = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
        ubSize = static_cast<int64_t>(ub);
        if (coreNum > 0 && ubSize > 0) {
            return ge::GRAPH_SUCCESS;
        }
    }
    auto* compileInfo = context->GetCompileInfo<GNTrainingReduceCompileInfo>();
    if (compileInfo != nullptr) {
        coreNum = compileInfo->coreNum;
        ubSize = compileInfo->ubSize;
        return ge::GRAPH_SUCCESS;
    }
    OP_LOGE(kOpName, "platform info is nullptr and compile info is nullptr");
    return ge::GRAPH_FAILED;
}

// 合轴四步。view 为 format 视图的 A/R 类型表。
int64_t FuseAxes(const AxisView* view, int64_t viewNum, AxisView* fused)
{
    // 步骤 1：去 1 轴（全 1 退化保留占位 A=1）
    AxisView dropped[kMaxPatternRank + 1];
    int64_t droppedNum = 0;
    for (int64_t i = 0; i < viewNum; ++i) {
        if (view[i].size != 1) {
            dropped[droppedNum++] = view[i];
        }
    }
    if (droppedNum == 0) {
        dropped[droppedNum++] = AxisView{1, 1, false};
    }

    // 步骤 2：相邻同类型轴合并（保留前导轴 stride）
    int64_t fusedNum = 0;
    for (int64_t i = 0; i < droppedNum; ++i) {
        if (fusedNum > 0 && fused[fusedNum - 1].isR == dropped[i].isR) {
            fused[fusedNum - 1].size *= dropped[i].size;
        } else {
            fused[fusedNum++] = dropped[i];
        }
    }

    // 步骤 3：补 leading A（第 0 轴是 R 时前置 A=1，stride = ∏所有轴 size）
    if (fusedNum > 0 && fused[0].isR) {
        int64_t total = 1;
        for (int64_t i = 0; i < fusedNum; ++i) {
            total *= fused[i].size;
        }
        for (int64_t i = fusedNum; i > 0; --i) {
            fused[i] = fused[i - 1];
        }
        fused[0] = AxisView{1, total, false};
        ++fusedNum;
    }

    // 步骤 4：补 R 增广（纯 A 退化路径）
    bool hasR = false;
    for (int64_t i = 0; i < fusedNum; ++i) {
        if (fused[i].isR) {
            hasR = true;
        }
    }
    if (!hasR) {
        const AxisView a = fused[0];
        if (a.size == 1) {
            fused[fusedNum++] = AxisView{1, 1, true}; // AR (tail-R)
        } else {
            const int64_t outer = a.size * a.stride;
            for (int64_t i = fusedNum + 1; i >= 2; --i) {
                fused[i] = fused[i - 2];
            }
            fused[0] = AxisView{1, outer, false}; // A=1
            fused[1] = AxisView{1, outer, true};  // R=1
            fusedNum += 2;                        // ARA (tail-A)
        }
    }
    return fusedNum;
}

ge::graphStatus ComputeEmptyRSplit(int64_t aTotal, int64_t coreNum, int64_t ubSize, int64_t blockSize,
                                   GNTrainingReduceEmptyTilingData* et)
{
    const int64_t maxDtypeSize = 4; // max(sizeof(D_T), sizeof(float))
    // a) 4KB buffer 下界；b) 优先多核；c) UB 上限（单 buf ≤ 64KB）；d) aTotal 兜底
    const int64_t minAPerCore = CeilDiv(4096, maxDtypeSize);
    const int64_t maxBufSize = std::min(ubSize, static_cast<int64_t>(65536));
    const int64_t maxUbFactor = maxBufSize / maxDtypeSize;
    int64_t aUb = std::min(std::max(minAPerCore, CeilDiv(aTotal, coreNum)), maxUbFactor);
    aUb = std::min(aUb, aTotal);
    if (aUb < 1) {
        return ge::GRAPH_FAILED;
    }
    et->aUbFactor = aUb;
    const int64_t aLoopCntTotal = CeilDiv(aTotal, aUb);
    et->aSmallCoreLoopCnt = aLoopCntTotal / coreNum;
    et->aBigCoreCnt = static_cast<int32_t>(aLoopCntTotal % coreNum);
    et->aBigCoreLoopCnt = et->aSmallCoreLoopCnt + ((et->aBigCoreCnt > 0) ? 1 : 0);
    et->usedCoreNum = static_cast<int32_t>((et->aSmallCoreLoopCnt > 0) ? coreNum : et->aBigCoreCnt);
    const int64_t postBufSize = std::max(aUb * maxDtypeSize, blockSize);
    et->postBufSize = AlignUp(postBufSize, blockSize);
    return ge::GRAPH_SUCCESS;
}

} // namespace

ge::graphStatus TilingForGNTrainingReduce(gert::TilingContext* context)
{
    if (context == nullptr) {
        OP_LOGE(kOpName, "TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }

    // ===== 0) 平台信息 =====
    int64_t coreNum = 0;
    int64_t ubSize = 0;
    if (ReadPlatform(context, coreNum, ubSize) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (coreNum <= 0) {
        OP_LOGE(kOpName, "incorrect value: platform coreNum=%ld must be positive", coreNum);
        return ge::GRAPH_FAILED;
    }
    if (ubSize <= 0) {
        OP_LOGE(kOpName, "incorrect value: platform ubSize=%ld must be positive", ubSize);
        return ge::GRAPH_FAILED;
    }
    const int64_t blockSize = static_cast<int64_t>(Ops::Base::GetUbBlockSize(context));
    const int64_t cacheLineSize = static_cast<int64_t>(Ops::Base::GetCacheLineSize(context));

    // ===== 1) 异常值校验：dtype -> format -> rank -> attr -> shape =====
    auto* inDesc = context->GetInputDesc(0);
    if (inDesc == nullptr) {
        OP_LOGE(kOpName, "incorrect dtype: x tensor desc is nullptr");
        return ge::GRAPH_FAILED;
    }
    const ge::DataType inDtype = inDesc->GetDataType();
    if (inDtype != ge::DT_FLOAT16 && inDtype != ge::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE(kOpName, "x", ToString(inDtype).c_str(), "float16 or float32");
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i < 2; ++i) {
        auto* outDesc = context->GetOutputDesc(i);
        if (outDesc == nullptr || outDesc->GetDataType() != ge::DT_FLOAT) {
            const std::string outName = "output[" + std::to_string(i) + "]";
            const std::string actualDtype = (outDesc == nullptr) ? "nullptr" : ToString(outDesc->GetDataType());
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(kOpName, outName.c_str(), actualDtype.c_str(),
                                                  "The dtype of output must be float32");
            return ge::GRAPH_FAILED;
        }
    }

    // 输入 format 仅支持 NCHW / NHWC（design/Interface.md「数据 Format 支持」）；
    // 与 InferShape 保持同一口径，ND 也不接受。
    const ge::Format inFormat = inDesc->GetOriginFormat();
    if (inFormat != ge::FORMAT_NCHW && inFormat != ge::FORMAT_NHWC) {
        OP_LOGE_FOR_INVALID_FORMAT(kOpName, "x", ToString(inFormat).c_str(), "NCHW or NHWC");
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i < 2; ++i) {
        auto* outDesc = context->GetOutputDesc(i);
        if (outDesc == nullptr || outDesc->GetOriginFormat() != ge::FORMAT_ND) {
            const std::string outName = "output[" + std::to_string(i) + "]";
            const std::string actualFormat = (outDesc == nullptr) ? "nullptr" : ToString(outDesc->GetOriginFormat());
            OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(kOpName, outName.c_str(), actualFormat.c_str(),
                                                   "The origin format of output must be ND");
            return ge::GRAPH_FAILED;
        }
    }

    auto* xShapePtr = context->GetInputShape(0);
    if (xShapePtr == nullptr) {
        OP_LOGE(kOpName, "incorrect shape dim: x shape is nullptr");
        return ge::GRAPH_FAILED;
    }
    const gert::Shape& xOrigin = xShapePtr->GetOriginShape();
    const gert::Shape& xStorage = xShapePtr->GetStorageShape();
    if (xOrigin.GetDimNum() != 4) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(kOpName, "x", std::to_string(xOrigin.GetDimNum()).c_str(),
                                                 "The shape dim of input x must be 4");
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i < 4; ++i) {
        const int64_t d = xOrigin.GetDim(i);
        if (d < 0) {
            const std::string dimName = "x[" + std::to_string(i) + "]";
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(kOpName, dimName.c_str(), std::to_string(d).c_str(),
                                                     "The dim of input x must not be negative/unknown");
            return ge::GRAPH_FAILED;
        }
    }

    // attr num_groups（按索引访问）
    const auto* attrs = context->GetAttrs();
    const int64_t* numGroupsPtr = (attrs != nullptr) ? attrs->GetInt(0) : nullptr;
    const int64_t numGroups = (numGroupsPtr != nullptr) ? *numGroupsPtr : 2;
    if (numGroups < 1) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "num_groups", std::to_string(numGroups).c_str(),
                                              "The attribute num_groups must be greater than or equal to 1");
        return ge::GRAPH_FAILED;
    }
    const int64_t N = xOrigin.GetDim(0);
    const int64_t C = (inFormat == ge::FORMAT_NCHW) ? xOrigin.GetDim(1) : xOrigin.GetDim(3);
    // C % num_groups 是无条件硬约束：空 tensor（N==0）不构成豁免，仅当 C 能被 G
    // 整除时才走 EMPTY_A / EMPTY_R（C=0 的空分组合法）。与 A2/A3 TBE 实现行为一致，
    // 也与本算子 InferShape 保持同一口径。
    if (C % numGroups != 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "num_groups", std::to_string(numGroups).c_str(),
            ("The channel dim C=" + std::to_string(C) + " of input x must be divisible by num_groups").c_str());
        return ge::GRAPH_FAILED;
    }

    // shape 一致性：storageShape == originShape (单输入算子)
    if (xStorage.GetDimNum() != xOrigin.GetDimNum()) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            kOpName, "x",
            ("storageShape dim " + std::to_string(xStorage.GetDimNum()) + " vs originShape dim " +
             std::to_string(xOrigin.GetDimNum()))
                .c_str(),
            "The storage shape of input x must have the same dim as its origin shape");
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i < 4; ++i) {
        if (xStorage.GetDim(i) != xOrigin.GetDim(i)) {
            const std::string dimName = "x[" + std::to_string(i) + "]";
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                kOpName, dimName.c_str(),
                ("storageShape " + std::to_string(xStorage.GetDim(i)) + " vs originShape " +
                 std::to_string(xOrigin.GetDim(i)))
                    .c_str(),
                "The storage shape of input x must be identical to its origin shape");
            return ge::GRAPH_FAILED;
        }
    }
    // 输出 shape 规则：NCHW [N,G,1,1,1]；NHWC [N,1,1,G,1]
    int64_t expOutShape[5];
    if (inFormat == ge::FORMAT_NCHW) {
        expOutShape[0] = N;
        expOutShape[1] = numGroups;
        expOutShape[2] = 1;
        expOutShape[3] = 1;
        expOutShape[4] = 1;
    } else {
        expOutShape[0] = N;
        expOutShape[1] = 1;
        expOutShape[2] = 1;
        expOutShape[3] = numGroups;
        expOutShape[4] = 1;
    }
    for (size_t o = 0; o < 2; ++o) {
        auto* outShapePtr = context->GetOutputShape(o);
        if (outShapePtr == nullptr) {
            OP_LOGE(kOpName, "incorrect shape: output[%zu] shape is nullptr", o);
            return ge::GRAPH_FAILED;
        }
        const gert::Shape& outShape = outShapePtr->GetOriginShape();
        const std::string outName = "output[" + std::to_string(o) + "]";
        if (outShape.GetDimNum() != 5) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(kOpName, outName.c_str(),
                                                     std::to_string(outShape.GetDimNum()).c_str(),
                                                     "The shape dim of output must be 5");
            return ge::GRAPH_FAILED;
        }
        for (size_t i = 0; i < 5; ++i) {
            if (outShape.GetDim(i) != expOutShape[i]) {
                const std::string dimName = outName + "[" + std::to_string(i) + "]";
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    kOpName, dimName.c_str(),
                    (std::to_string(outShape.GetDim(i)) + " vs expected " + std::to_string(expOutShape[i])).c_str(),
                    "The shape of output must be [N, num_groups, 1, 1, 1] (NCHW) or [N, 1, 1, num_groups, 1] (NHWC)");
                return ge::GRAPH_FAILED;
            }
        }
    }

    const int64_t sysWorkspace = static_cast<int64_t>(Ops::Base::GetWorkspaceSize(context));

    // ===== 2) 空 tensor 短路（EMPTY_A 优先于 EMPTY_R）=====
    const int64_t D = C / numGroups;
    const int64_t H = (inFormat == ge::FORMAT_NCHW) ? xOrigin.GetDim(2) : xOrigin.GetDim(1);
    const int64_t W = (inFormat == ge::FORMAT_NCHW) ? xOrigin.GetDim(3) : xOrigin.GetDim(2);

    AxisView view[5];
    if (inFormat == ge::FORMAT_NCHW) {
        view[0] = AxisView{N, C * H * W, false};
        view[1] = AxisView{numGroups, D * H * W, false};
        view[2] = AxisView{D, H * W, true};
        view[3] = AxisView{H, W, true};
        view[4] = AxisView{W, 1, true};
    } else {
        view[0] = AxisView{N, H * W * C, false};
        view[1] = AxisView{H, W * C, true};
        view[2] = AxisView{W, C, true};
        view[3] = AxisView{numGroups, D, false};
        view[4] = AxisView{D, 1, true};
    }
    bool hasZeroA = false;
    bool hasZeroR = false;
    for (int64_t i = 0; i < 5; ++i) {
        if (view[i].size == 0) {
            if (view[i].isR) {
                hasZeroR = true;
            } else {
                hasZeroA = true;
            }
        }
    }
    if (hasZeroA || hasZeroR) {
        auto* et = AcquireTilingData<GNTrainingReduceEmptyTilingData>(context);
        if (et == nullptr) {
            OP_LOGE(kOpName, "empty tensor: tiling data buffer unavailable");
            return ge::GRAPH_FAILED;
        }
        std::memset(et, 0, sizeof(*et));
        if (hasZeroA) {
            // EMPTY_A：输出空 tensor，所有核早退（严禁 SetBlockDim(0)）
            et->usedCoreNum = 0;
            context->SetBlockDim(1);
        } else {
            // EMPTY_R：A 全非 0、R 含 0，按 a 元素数切分
            int64_t aTotal = 1;
            for (int64_t i = 0; i < 5; ++i) {
                if (!view[i].isR) {
                    aTotal *= view[i].size;
                }
            }
            et->aTotal = aTotal;
            if (ComputeEmptyRSplit(aTotal, coreNum, ubSize, blockSize, et) != ge::GRAPH_SUCCESS) {
                OP_LOGE(kOpName, "empty tensor: empty R split failed");
                return ge::GRAPH_FAILED;
            }
            context->SetBlockDim(static_cast<uint32_t>(et->usedCoreNum));
        }
        size_t* currentWorkspace = context->GetWorkspaceSizes(1);
        if (currentWorkspace == nullptr) {
            OP_LOGE(kOpName, "empty tensor: workspace sizes unavailable");
            return ge::GRAPH_FAILED;
        }
        currentWorkspace[0] = static_cast<size_t>(sysWorkspace);
        context->SetTilingKey(kTilingKeyEmpty);
        return ge::GRAPH_SUCCESS;
    }

    // ===== 3) 非空：合轴四步 =====
    AxisView fused[kMaxPatternRank + 2];
    const int64_t axisNum = FuseAxes(view, 5, fused);
    if (axisNum < 2 || axisNum > kMaxPatternRank) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            kOpName, "x", std::to_string(axisNum).c_str(),
            ("The fused axis num of input x must be in [2, " + std::to_string(kMaxPatternRank) + "]").c_str());
        return ge::GRAPH_FAILED;
    }
    int64_t axisShape[kMaxPatternRank] = {0, 0, 0, 0};
    int64_t axisStride[kMaxPatternRank] = {0, 0, 0, 0};
    for (int64_t i = 0; i < axisNum; ++i) {
        axisShape[i] = fused[i].size;
    }
    // 合轴后各轴仍是内存连续的：合并轴必须以「最内层」stride（= 右侧所有轴 size 之积）
    // 作为统一步长，kernel 的 Unravel/偏移公式才能按轴 uniform stride 累加。沿用前导轴
    // stride（如 N·G 轴取 C·H·W）会使 A 向多核步进越过张量边界（507035）。
    {
        int64_t strideAcc = 1;
        for (int64_t i = axisNum - 1; i >= 0; --i) {
            axisStride[i] = strideAcc;
            strideAcc *= axisShape[i];
        }
    }
    const bool isTailR = (axisNum % 2 == 0);
    const bool isTailA = !isTailR;

    const int64_t sizeofDType = (inDtype == ge::DT_FLOAT16) ? 2 : 4;
    const int64_t maxDtypeSize = std::max(sizeofDType, static_cast<int64_t>(4));
    const int64_t bsElem = blockSize / sizeofDType;
    const int64_t cachelineTmp = AlignDown(cacheLineSize / sizeofDType, bsElem);

    // ===== 4) UB 切分三步 =====
    // ---- Step 1: ComputeAUbFactor ----
    int32_t aSplitIdx = 0;
    int64_t aUbFactor = 0;
    int64_t innerAProdAlign = 1;
    int64_t aUnit = 0;
    {
        int64_t product = 1;
        int64_t idx = axisNum - 1;
        while (idx >= 0) {
            const int64_t axisSize = (idx == axisNum - 1) ? AlignUp(axisShape[idx], bsElem) : axisShape[idx];
            if (product * axisSize > cachelineTmp) {
                break;
            }
            product *= axisSize;
            --idx;
        }
        if (idx < 0) {
            aSplitIdx = 0;
            aUbFactor = axisShape[0];
        } else if (idx % 2 == 0) {
            aSplitIdx = static_cast<int32_t>(idx);
            aUbFactor = cachelineTmp / product;
            aUbFactor = std::min(aUbFactor, axisShape[idx]);
        } else {
            aSplitIdx = static_cast<int32_t>(idx - 1);
            aUbFactor = 1;
        }
        innerAProdAlign = 1;
        for (int64_t k = aSplitIdx + 2; k < axisNum; k += 2) {
            if (k == axisNum - 1 && isTailA) {
                innerAProdAlign *= AlignUp(axisShape[k], bsElem);
            } else {
                innerAProdAlign *= axisShape[k];
            }
        }
        aUnit = aUbFactor * innerAProdAlign;
    }

    // ---- Step 2: ComputeRUbFactor ----
    // UB 预算：pre 阶段三份 buffer 与二分缓存除逻辑大小外各多分配一个向量的读余量
    // （无 mask 的 LoadAlign 读满整个向量寄存器，见 GN_TRAINING_REDUCE_UB_READ_SLACK），
    // 故可用池要把这 4 份余量一并扣除，避免实际占用越过 ubSize。
    const int64_t ubReadSlack = (kPpre + kPpreExt + 1) * GN_TRAINING_REDUCE_UB_READ_SLACK;
    const int64_t ubAvailable = ubSize - kCacheBufUbSize - ubReadSlack;
    int64_t postBufSize = AlignUp(aUnit * maxDtypeSize, blockSize);
    int32_t rSplitIdx = 0;
    int64_t rUbFactor = 0;
    int64_t rUbFactorAlign = 0;
    int64_t innerRProdAlign = 1;
    {
        const int64_t aOnlyBytes = kPpost * postBufSize;
        const int64_t bytesPerRElem = (kPpre + kPpreExt) * aUnit * maxDtypeSize;
        if (ubAvailable <= 0 || bytesPerRElem <= 0) {
            OP_LOGE(kOpName, "tiling fail: ub budget invalid ubAvailable=%ld bytesPerRElem=%ld", ubAvailable,
                    bytesPerRElem);
            return ge::GRAPH_FAILED;
        }
        const int64_t rIMax = (ubAvailable - aOnlyBytes) / bytesPerRElem;
        if (rIMax < 1) {
            OP_LOGE(kOpName, "tiling fail: rIMax=%ld < 1 (cannot fit one R block)", rIMax);
            return ge::GRAPH_FAILED;
        }
        const int64_t lastR = (axisNum % 2 == 0) ? (axisNum - 1) : (axisNum - 2);
        innerRProdAlign = 1;
        rSplitIdx = static_cast<int32_t>(lastR);
        while (rSplitIdx > 1) {
            const int64_t axisSize = (rSplitIdx == lastR && isTailR) ? AlignUp(axisShape[rSplitIdx], bsElem) :
                                                                       axisShape[rSplitIdx];
            if (axisSize * innerRProdAlign > rIMax) {
                break;
            }
            innerRProdAlign *= axisSize;
            rSplitIdx -= 2;
        }
        rUbFactor = rIMax / innerRProdAlign;
        rUbFactor = std::min(rUbFactor, axisShape[rSplitIdx]);
        const bool isBurstTailR = (isTailR && rSplitIdx == lastR);
        if (isBurstTailR) {
            if (rUbFactor < axisShape[rSplitIdx]) {
                rUbFactor = AlignDown(rUbFactor, bsElem);
                if (rUbFactor < 1) {
                    OP_LOGE(kOpName, "tiling fail: rUbFactor underflow on burst tail-R");
                    return ge::GRAPH_FAILED;
                }
                rUbFactorAlign = rUbFactor;
            } else {
                rUbFactorAlign = AlignUp(rUbFactor, bsElem);
                if (rUbFactorAlign * innerRProdAlign > rIMax) {
                    rUbFactor = AlignDown(rUbFactor, bsElem);
                    if (rUbFactor < 1) {
                        OP_LOGE(kOpName, "tiling fail: rUbFactor underflow after align-down");
                        return ge::GRAPH_FAILED;
                    }
                    rUbFactorAlign = rUbFactor;
                }
            }
        } else {
            rUbFactorAlign = rUbFactor;
            if (rUbFactor < 1) {
                OP_LOGE(kOpName, "tiling fail: rUbFactor < 1");
                return ge::GRAPH_FAILED;
            }
        }
    }

    // ---- Step 3: ExpandAIfRFullyLoaded ----
    if (rUbFactor == axisShape[rSplitIdx] && rSplitIdx == 1) {
        const int64_t rPadded = rUbFactorAlign * innerRProdAlign;
        const int64_t denom = maxDtypeSize * ((kPpre + kPpreExt) * rPadded + kPpost);
        int64_t aUnitMax = (denom > 0) ? (ubAvailable / denom) : aUnit;
        int64_t allA = 1;
        for (int64_t k = 0; k < axisNum; k += 2) {
            allA *= axisShape[k];
        }
        aUnitMax = std::min(aUnitMax, allA);
        aUnitMax = std::min(aUnitMax, static_cast<int64_t>(kCacheBufUbSize / 4)); // ≤ 4096
        if (isTailA) {
            aUnitMax = AlignDown(aUnitMax, bsElem);
        }
        if (aUnitMax > aUnit) {
            int64_t product = 1;
            int64_t idx = (axisNum % 2 == 0) ? (axisNum - 2) : (axisNum - 1);
            while (idx >= 0) {
                const int64_t axisSize = (idx == axisNum - 1 && isTailA) ? AlignUp(axisShape[idx], bsElem) :
                                                                           axisShape[idx];
                if (product * axisSize > aUnitMax) {
                    break;
                }
                product *= axisSize;
                idx -= 2;
            }
            if (idx < 0) {
                aSplitIdx = 0;
                aUbFactor = axisShape[0];
            } else {
                aSplitIdx = static_cast<int32_t>(idx);
                aUbFactor = std::min(aUnitMax / product, axisShape[idx]);
            }
            innerAProdAlign = 1;
            for (int64_t k = aSplitIdx + 2; k < axisNum; k += 2) {
                if (k == axisNum - 1 && isTailA) {
                    innerAProdAlign *= AlignUp(axisShape[k], bsElem);
                } else {
                    innerAProdAlign *= axisShape[k];
                }
            }
            aUnit = aUbFactor * innerAProdAlign;
        }
    }

    // A′ 规避（见问题记录 A4）：tail-A（如 NHWC D=1 的 [A,R,A]）下若一个 A chunk 含
    // >1 个外层 A 行（aUbFactor>1），kernel 的 UB 搬运会把 chunk 内第 2 个 A 行写成第 1
    // 行的 group-0。强制 aUbFactor=1，使每个 A chunk 只含 1 个外层 A 行。须置于
    // Step 3 (ExpandAIfRFullyLoaded) 之后，否则会被其重算覆盖。
    if (isTailA && aSplitIdx != static_cast<int32_t>(axisNum - 1)) {
        aUbFactor = 1;
        aUnit = aUbFactor * innerAProdAlign;
    }

    // ---- ComputeUbSizes ----
    const int64_t preBufSize = aUnit * rUbFactorAlign * innerRProdAlign * maxDtypeSize;
    postBufSize = AlignUp(aUnit * maxDtypeSize, blockSize);

    // ===== 5) 多核切分 =====
    // ---- ComputeFusedALoopSplit ----
    const int64_t aSplitChunkCnt = CeilDiv(axisShape[aSplitIdx], aUbFactor);
    int64_t outerAProd = 1;
    for (int64_t k = 0; k < aSplitIdx; k += 2) {
        outerAProd *= axisShape[k];
    }
    const int64_t aLoopCntTotal = outerAProd * aSplitChunkCnt;
    const int64_t aSmallCoreLoopCnt = aLoopCntTotal / coreNum;
    const int64_t aBigCoreCnt = aLoopCntTotal % coreNum;
    const int64_t aBigCoreLoopCnt = aSmallCoreLoopCnt + ((aBigCoreCnt > 0) ? 1 : 0);
    int64_t usedCoreNum = (aSmallCoreLoopCnt > 0) ? coreNum : aBigCoreCnt;

    // ---- ComputeRLoopCnt ----
    int64_t outerRProd = 1;
    for (int64_t k = 1; k < rSplitIdx; k += 2) {
        outerRProd *= axisShape[k];
    }
    const int64_t rLoopCntTotal = outerRProd * CeilDiv(axisShape[rSplitIdx], rUbFactor);

    // ===== 6) Group 判定与 2D 分核 =====
    bool isGroup = false;
    if (!(aLoopCntTotal > coreNum / 2) && rLoopCntTotal > 1) {
        isGroup = true;
    }
    int64_t aTotal = 1;
    for (int64_t k = 0; k < axisNum; k += 2) {
        aTotal *= axisShape[k];
    }
    int64_t rGroupCnt = 0;
    if (isGroup) {
        // ---- ComputeGroupSplit ----
        const int64_t totalOuter = aLoopCntTotal * rLoopCntTotal;
        const int64_t perCoreNum = CeilDiv(totalOuter, coreNum);
        int64_t numBlocks = CeilDiv(totalOuter, perCoreNum);
        if (AlignUp(numBlocks, aLoopCntTotal) <= coreNum) {
            numBlocks = AlignUp(numBlocks, aLoopCntTotal);
        } else {
            numBlocks = AlignDown(numBlocks, aLoopCntTotal);
        }
        usedCoreNum = numBlocks;
        rGroupCnt = numBlocks / aLoopCntTotal;
        if (context->SetScheduleMode(1) != ge::GRAPH_SUCCESS) {
            OP_LOGE(kOpName, "tiling fail: cannot set ScheduleMode for group template");
            return ge::GRAPH_FAILED;
        }
    }

    // ---- workspace ----
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace == nullptr) {
        OP_LOGE(kOpName, "tiling fail: workspace sizes unavailable");
        return ge::GRAPH_FAILED;
    }
    if (isGroup) {
        currentWorkspace[0] = static_cast<size_t>(rGroupCnt) * static_cast<size_t>(aTotal) * sizeof(float) +
                              static_cast<size_t>(sysWorkspace);
    } else {
        currentWorkspace[0] = static_cast<size_t>(sysWorkspace);
    }

    // ===== 7) 填 TilingData + SetTilingKey + SetBlockDim =====
    auto* td = AcquireTilingData<GNTrainingReduceTilingData>(context);
    if (td == nullptr) {
        OP_LOGE(kOpName, "tiling fail: tiling data buffer unavailable");
        return ge::GRAPH_FAILED;
    }
    std::memset(td, 0, sizeof(*td));
    td->axisNum = static_cast<int32_t>(axisNum);
    for (int64_t i = 0; i < kMaxPatternRank; ++i) {
        td->axisShape[i] = axisShape[i];
        td->axisStride[i] = axisStride[i];
    }
    td->aLoopCntTotal = aLoopCntTotal;
    td->aSplitChunkCnt = aSplitChunkCnt;
    td->aBigCoreLoopCnt = aBigCoreLoopCnt;
    td->aSmallCoreLoopCnt = aSmallCoreLoopCnt;
    td->aBigCoreCnt = static_cast<int32_t>(aBigCoreCnt);
    td->usedCoreNum = static_cast<int32_t>(usedCoreNum);
    td->aSplitIdx = aSplitIdx;
    td->rSplitIdx = rSplitIdx;
    td->aUbFactor = aUbFactor;
    td->rUbFactor = rUbFactor;
    td->rUbFactorAlign = rUbFactorAlign;
    td->innerAProdAlign = innerAProdAlign;
    td->innerRProdAlign = innerRProdAlign;
    td->rLoopCntTotal = rLoopCntTotal;
    td->preBufSize = preBufSize;
    td->postBufSize = postBufSize;
    td->cacheBufUbSize = kCacheBufUbSize;
    td->rGroupCnt = isGroup ? rGroupCnt : 0;

    context->SetTilingKey(isGroup ? kTilingKeyGroup : kTilingKeyBase);
    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));

    OP_LOGI(kOpName,
            "[gn_training_reduce] axisNum=%d aLoopCntTotal=%ld aSplitChunkCnt=%ld aBigCoreLoopCnt=%ld "
            "aSmallCoreLoopCnt=%ld aBigCoreCnt=%d usedCoreNum=%d aSplitIdx=%d rSplitIdx=%d aUbFactor=%ld "
            "rUbFactor=%ld rUbFactorAlign=%ld innerAProdAlign=%ld innerRProdAlign=%ld rLoopCntTotal=%ld "
            "preBufSize=%ld postBufSize=%ld cacheBufUbSize=%ld rGroupCnt=%ld tilingKey=%lu",
            td->axisNum, td->aLoopCntTotal, td->aSplitChunkCnt, td->aBigCoreLoopCnt, td->aSmallCoreLoopCnt,
            td->aBigCoreCnt, td->usedCoreNum, td->aSplitIdx, td->rSplitIdx, td->aUbFactor, td->rUbFactor,
            td->rUbFactorAlign, td->innerAProdAlign, td->innerRProdAlign, td->rLoopCntTotal, td->preBufSize,
            td->postBufSize, td->cacheBufUbSize, td->rGroupCnt,
            static_cast<unsigned long>(isGroup ? kTilingKeyGroup : kTilingKeyBase));
    return ge::GRAPH_SUCCESS;
}

// 编译期准备：解析 platform 并填充 CompileInfo。
ge::graphStatus TilingPrepareForGNTrainingReduce(gert::TilingParseContext* context)
{
    if (context == nullptr) {
        OP_LOGE(kOpName, "TilingParseContext is nullptr");
        return ge::GRAPH_FAILED;
    }
    auto* platformInfo = context->GetPlatformInfo();
    auto* compileInfo = context->GetCompiledInfo<GNTrainingReduceCompileInfo>();
    if (platformInfo == nullptr || compileInfo == nullptr) {
        OP_LOGE(kOpName, "TilingParse: platform info or compiled info is nullptr");
        return ge::GRAPH_FAILED;
    }
    platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);
    compileInfo->coreNum = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = static_cast<int64_t>(ubSize);
    return ge::GRAPH_SUCCESS;
}

// 注册 Tiling 入口与 TilingParse。
IMPL_OP_OPTILING(GNTrainingReduce)
    .Tiling(TilingForGNTrainingReduce)
    .TilingParse<GNTrainingReduceCompileInfo>(TilingPrepareForGNTrainingReduce);

} // namespace optiling
