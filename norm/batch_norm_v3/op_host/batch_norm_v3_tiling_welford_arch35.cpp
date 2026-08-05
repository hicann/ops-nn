/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file batch_norm_v3_tiling_welford_arch35.cpp
 * \brief
 */
#include <vector>
#include <algorithm>
#include "batch_norm_v3_tiling.h"
using namespace ge;

namespace {
constexpr int64_t TILINGKEY_WELFORD_REDUCE = 300000;

constexpr int64_t FLOAT32_BYTES = 4;
constexpr int64_t FLOAT16_BYTES = 2;
constexpr int64_t SMALL_BUFFER_NUM = 9;
constexpr int64_t SMALL_BUFFER_NUM_FP32 = 6;
constexpr int64_t SMALL_BUFFER_NUM_T = 2;
constexpr int64_t LARGE_BUFFER_NUM_QUEUE = 2;
constexpr int64_t LARGE_BUFFER_NUM_TMP = 2;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t NCHW_DIM_NUM = 4;
constexpr int64_t NCDHW_DIM_NUM = 5;
constexpr int64_t NHWC_DIM_NUM = 4;
constexpr int64_t NDHWC_DIM_NUM = 5;
constexpr int64_t MIN_ND_DIM_NUM = 2;
constexpr int64_t BINARY_ADD_COEF = 2;
constexpr int64_t MAX_COMMON_PARELLEL = 256;

// 6 for large case, 1 for extra
constexpr int64_t BLOCK_RESERVE_NUMBER = 7;

constexpr int64_t DIM_0 = 0;
constexpr int64_t DIM_1 = 1;
constexpr int64_t DIM_2 = 2;
constexpr int64_t DIM_3 = 3;
constexpr int64_t DIM_4 = 4;

static const int32_t INDEX_EPSILON = 0;
static const int32_t INDEX_MOMENTUM = 1;
static const int32_t INDEX_IS_TRAINING = 2;
constexpr float DEFAULT_EPSILON = 1e-5;
constexpr float DEFAULT_MOMENTUM = 0.1;

constexpr uint32_t MINIMAL_WORKSPACE = 16 * 1024 * 1024;
} // namespace

namespace optiling {
class BatchNormV3WelfordReduceTilingBase : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit BatchNormV3WelfordReduceTilingBase(gert::TilingContext* context)
        : Ops::NN::Optiling::TilingBaseClass(context)
    {
        Reset();
    }
    ~BatchNormV3WelfordReduceTilingBase() override = default;

    void Reset(gert::TilingContext* context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    bool IsCapable() override
    {
        // 二分折叠要求 binaryAddQuotient 是整数个 vlFp32（kernel 侧按 dichotomyAddPower / VL_FP32
        // 取整块循环次数），而它的下限就是一个 vlFp32。因此归约长度 r1 * r0 不超过一个
        // vlFp32 时，不存在合法取值，本模板不支持。
        if (r1 * r0 <= static_cast<int64_t>(vlFp32)) {
            OP_LOGI(context_->GetNodeName(), "BatchNormV3WelfordReduce not capable: r1(%ld) * r0(%ld) <= vlFp32(%lu).",
                    r1, r0, vlFp32);
            return false;
        }
        return true;
    }
    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    // 3、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    // 4、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;

    void Reset();

private:
    const char* opName = "BatchNormV3WelfordReduce";
    int64_t a0;
    int64_t r0;
    int64_t r1;
    int64_t blockNum;
    int64_t binaryAddQuotient;
    int64_t parallelN;
    uint64_t blockSize;
    uint64_t vlFp32;
    ge::DataType dataType;
    BatchNormV3WelfordRegbaseTilingData tilingData;
};

void BatchNormV3WelfordReduceTilingBase::Reset()
{
    opName = nullptr;
    a0 = 0;
    r0 = 0;
    r1 = 0;
    blockNum = 0;
    binaryAddQuotient = 0;
    parallelN = 0;
    blockSize = 0;
    vlFp32 = 0;
    return;
}

inline static int64_t CeilDiv(int64_t value, int64_t factor)
{
    int64_t valueNum = 0;
    if (factor == 0) {
        return value;
    }

    if (value % factor == 0) {
        valueNum = value / factor;
    } else {
        valueNum = value / factor + 1;
    }

    return valueNum;
}

inline static int64_t RoundUp(int64_t a, int64_t b) { return CeilDiv(a, b) * b; }

ge::graphStatus BatchNormV3WelfordReduceTilingBase::GetPlatformInfo()
{
    auto compileInfo = reinterpret_cast<const BatchNormV3CompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    blockSize = static_cast<uint64_t>(compileInfo->blockSize);
    vlFp32 = static_cast<uint64_t>(compileInfo->vectorLength) / sizeof(float);

    opName = context_->GetNodeName();
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        aicoreParams_.numBlocks = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        aicoreParams_.ubSize = ubSizePlatForm;
    } else {
        aicoreParams_.numBlocks = compileInfo->coreNum;
        aicoreParams_.ubSize = compileInfo->ubSize;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormV3WelfordReduceTilingBase::GetShapeAttrsInfo()
{
    if (context_ == nullptr) {
        OP_LOGE("BatchNormV3", "TilingContext is nullptr.");
        return ge::GRAPH_FAILED;
    }

    // 获取输入shape
    auto xShape = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShape);
    auto xStorageShape = xShape->GetStorageShape();
    auto xDesc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    dataType = xDesc->GetDataType();
    auto format = xDesc->GetFormat().GetStorageFormat();

    // 获取attr
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const float* epsilonPtr = attrs->GetFloat(INDEX_EPSILON);
    const float epsilon = (epsilonPtr == nullptr) ? DEFAULT_EPSILON : *epsilonPtr;
    const float* momentumPtr = attrs->GetFloat(INDEX_MOMENTUM);
    const float momentum = (momentumPtr == nullptr) ? DEFAULT_MOMENTUM : *momentumPtr;
    const bool* isTrainingPtr = attrs->GetBool(INDEX_IS_TRAINING);
    bool isTraining = (isTrainingPtr == nullptr) ? true : *isTrainingPtr;
    if (!isTraining) {
        OP_LOGI(context_, "This node not support infer.");
        return ge::GRAPH_PARAM_INVALID;
    }
    tilingData.set_epsilon(epsilon);
    tilingData.set_momentum(momentum);

    if (format == FORMAT_ND) {
        // 与 BatchNormV3RegbaseTilingBase::GetShapeAttrsInfo 的折算保持一致：
        // r1 = dim0，a0 = dim1，r0 = 其余各维之积
        OP_CHECK_IF(xStorageShape.GetDimNum() < MIN_ND_DIM_NUM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "x",
                                                 std::to_string(xStorageShape.GetDimNum()).c_str(),
                                                 "at least 2D with ND format"),
                    return ge::GRAPH_FAILED);
        r1 = xStorageShape.GetDim(DIM_0);
        a0 = xStorageShape.GetDim(DIM_1);
        r0 = 1;
        for (size_t i = static_cast<size_t>(DIM_2); i < xStorageShape.GetDimNum(); ++i) {
            r0 *= xStorageShape.GetDim(i);
        }
    } else if (format == FORMAT_NCHW) {
        OP_CHECK_IF(
            xStorageShape.GetDimNum() != NCHW_DIM_NUM,
            OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "x",
                                         std::to_string(xStorageShape.GetDimNum()).c_str(), "4D with NCHW format"),
            return ge::GRAPH_FAILED);
        r1 = xStorageShape.GetDim(DIM_0);
        a0 = xStorageShape.GetDim(DIM_1);
        r0 = xStorageShape.GetDim(DIM_2) * xStorageShape.GetDim(DIM_3);
    } else if (format == FORMAT_NCDHW) {
        OP_CHECK_IF(
            xStorageShape.GetDimNum() != NCDHW_DIM_NUM,
            OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "x",
                                         std::to_string(xStorageShape.GetDimNum()).c_str(), "5D with NCDHW format"),
            return ge::GRAPH_FAILED);
        r1 = xStorageShape.GetDim(DIM_0);
        a0 = xStorageShape.GetDim(DIM_1);
        r0 = xStorageShape.GetDim(DIM_2) * xStorageShape.GetDim(DIM_3) * xStorageShape.GetDim(DIM_4);
    } else if (format == FORMAT_NHWC) {
        OP_CHECK_IF(
            xStorageShape.GetDimNum() != NHWC_DIM_NUM,
            OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "x",
                                         std::to_string(xStorageShape.GetDimNum()).c_str(), "4D with NHWC format"),
            return ge::GRAPH_FAILED);
        r1 = xStorageShape.GetDim(DIM_0) * xStorageShape.GetDim(DIM_1) * xStorageShape.GetDim(DIM_2);
        a0 = xStorageShape.GetDim(DIM_3);
        r0 = 1;
    } else if (format == FORMAT_NDHWC) {
        OP_CHECK_IF(
            xStorageShape.GetDimNum() != NDHWC_DIM_NUM,
            OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "x",
                                         std::to_string(xStorageShape.GetDimNum()).c_str(), "5D with NDHWC format"),
            return ge::GRAPH_FAILED);
        r1 = xStorageShape.GetDim(DIM_0) * xStorageShape.GetDim(DIM_1) * xStorageShape.GetDim(DIM_2) *
             xStorageShape.GetDim(DIM_3);
        a0 = xStorageShape.GetDim(DIM_4);
        r0 = 1;
    } else {
        OP_LOGE_FOR_INVALID_FORMAT(context_->GetNodeName(), "x", ge::TypeUtils::FormatToSerialString(format).c_str(),
                                   "ND, NCHW, NCDHW, NHWC or NDHWC");
        return ge::GRAPH_FAILED;
    }

    // 五个分支只负责折算出 (r1, a0, r0)，setter 在此统一调用一次，避免逐分支重复。
    tilingData.set_r1(r1);
    tilingData.set_a0(a0);
    tilingData.set_r0(r0);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormV3WelfordReduceTilingBase::DoOpTiling()
{
    // block tiling
    tilingData.set_aBlockFactor(CeilDiv(a0, static_cast<int64_t>(aicoreParams_.numBlocks)));
    tilingData.set_realCoreNum(CeilDiv(a0, tilingData.get_aBlockFactor()));
    tilingData.set_numLastCore(a0 % tilingData.get_aBlockFactor());
    blockNum = tilingData.get_realCoreNum();

    tilingData.set_elemNum(r1 * r0);
    tilingData.set_vlLenFp32(vlFp32);

    int64_t elemSize = FLOAT16_BYTES;
    if (dataType == ge::DT_FLOAT) {
        elemSize = FLOAT32_BYTES;
    }
    int64_t elemAlignNum = blockSize / elemSize;

    // ub tiling
    int64_t aGatherLimit = tilingData.get_aBlockFactor() > MAX_COMMON_PARELLEL ? MAX_COMMON_PARELLEL :
                                                                                 tilingData.get_aBlockFactor();
    tilingData.set_aGatherLimit(aGatherLimit);

    int32_t totalUBSize = aicoreParams_.ubSize;
    uint64_t smallUbNum = RoundUp(tilingData.get_aGatherLimit() * FLOAT32_BYTES, blockSize);
    uint64_t smallUbSize = (smallUbNum * SMALL_BUFFER_NUM * DOUBLE_BUFFER) * FLOAT32_BYTES;

    int64_t binaryAddBufNum = (totalUBSize / (DOUBLE_BUFFER * LARGE_BUFFER_NUM_QUEUE * elemSize)) /
                              tilingData.get_vlLenFp32();
    int64_t binaryAddBufSize = ((binaryAddBufNum * FLOAT32_BYTES + blockSize - 1) / blockSize) * blockSize;

    // smallUbSize / blockSize 是无符号数，直接相减会在表达式内下溢（算术转换发生在赋值之前，
    // 只改左值类型无效），这里统一转成有符号数计算。
    int64_t ubRemain = static_cast<int64_t>(totalUBSize) - static_cast<int64_t>(smallUbSize) - binaryAddBufSize -
                       static_cast<int64_t>(blockSize) * BLOCK_RESERVE_NUMBER;

    // processSize is max ub size.
    int64_t ubSize = ubRemain /
                     (DOUBLE_BUFFER * elemSize * LARGE_BUFFER_NUM_QUEUE + FLOAT32_BYTES * LARGE_BUFFER_NUM_TMP);
    int64_t ubSizeAlign = ubSize / elemAlignNum * elemAlignNum;
    // 放在对齐之后校验：ubRemain 为负时 ubSizeAlign 同样不为正，为正但不足一个对齐单位时会被
    // 抹成 0，这一条同时覆盖两种情况。
    OP_CHECK_IF(ubSizeAlign <= 0,
                OP_LOGE(opName, "ub size %d is not enough, ubSizeAlign %ld.", totalUBSize, ubSizeAlign),
                return ge::GRAPH_FAILED);

    if (r0 >= ubSizeAlign) {
        tilingData.set_r0Factor(ubSizeAlign);
        tilingData.set_loopR0outer(CeilDiv(r0, ubSizeAlign));
        tilingData.set_r1Factor(1);
        tilingData.set_loopR1outer(r1);
        tilingData.set_ubSize(ubSizeAlign);
        parallelN = ubSizeAlign;
        tilingData.set_parallelN(parallelN);
        tilingData.set_processSize(ubSizeAlign);
        tilingData.set_cutR1OrR0(0);
    } else {
        int64_t r1Factor = ubSizeAlign / r0;
        r1Factor = r1Factor > r1 ? r1 : r1Factor;

        tilingData.set_r0Factor(r0);
        tilingData.set_loopR0outer(1);
        tilingData.set_r1Factor(r1Factor);
        tilingData.set_loopR1outer(CeilDiv(r1, r1Factor));
        int64_t processSize = r0 * r1Factor;
        ubSizeAlign = (processSize + elemAlignNum - 1) / elemAlignNum * elemAlignNum;
        tilingData.set_ubSize(ubSizeAlign);
        parallelN = processSize;
        tilingData.set_parallelN(parallelN);
        tilingData.set_processSize(processSize);
        tilingData.set_cutR1OrR0(1);
    }

    // parallelN 才是二分折叠的实际输入：它取 ubSizeAlign（r0 >= ubSizeAlign 分支）或
    // r0 * r1Factor（另一分支），两者恒小于等于 r1 * r0。UB 紧张时 ubSizeAlign 可以小到
    // 一个 vlFp32 以内，此时 IsCapable 里按 r1 * r0 判的守卫拦不住：binaryAddQuotient
    // 仍会退化成半个 VL，vcaddNum 与 binaryAddLastNum 双双为 0，归约结果恒为 0。
    OP_CHECK_IF(parallelN <= static_cast<int64_t>(vlFp32),
                OP_LOGE(opName, "ub size %d is not enough, parallelN %ld should be greater than vlFp32 %lu.",
                        totalUBSize, parallelN, vlFp32),
                return ge::GRAPH_FAILED);

    // binary add param
    int64_t vlLenFp32 = tilingData.get_vlLenFp32();
    binaryAddQuotient = vlLenFp32;
    while (binaryAddQuotient < parallelN) {
        binaryAddQuotient = binaryAddQuotient * BINARY_ADD_COEF;
    }
    binaryAddQuotient = binaryAddQuotient / BINARY_ADD_COEF;
    tilingData.set_binaryAddQuotient(binaryAddQuotient);

    OP_CHECK_IF(vlLenFp32 == 0, OP_LOGE(opName, "vlLenFp32 should not be 0."), return ge::GRAPH_FAILED);
    int64_t vcaddNum = binaryAddQuotient / vlLenFp32;
    if (vcaddNum <= vlLenFp32) {
        tilingData.set_binaryAddK(0);
        tilingData.set_binaryAddLastNum(vcaddNum);
    } else {
        int64_t binaryAddNum = vcaddNum / vlLenFp32;
        int64_t binaryAddK = 0;
        int64_t tmpBinaryAddNum = 1;
        while (tmpBinaryAddNum < binaryAddNum) {
            binaryAddK = binaryAddK + 1;
            tmpBinaryAddNum = tmpBinaryAddNum * BINARY_ADD_COEF;
        }
        tilingData.set_binaryAddK(binaryAddK);
        tilingData.set_binaryAddLastNum(vlLenFp32);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormV3WelfordReduceTilingBase::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

uint64_t BatchNormV3WelfordReduceTilingBase::GetTilingKey() const { return TILINGKEY_WELFORD_REDUCE; }

ge::graphStatus BatchNormV3WelfordReduceTilingBase::GetWorkspaceSize()
{
    // 计算workspace大小
    workspaceSize_ = MINIMAL_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormV3WelfordReduceTilingBase::PostTiling()
{
    context_->SetBlockDim(blockNum);
    auto rawTilingData = context_->GetRawTilingData();
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = workspaceSize_;
    OP_CHECK_IF(tilingData.GetDataSize() > rawTilingData->GetCapacity(),
                OP_LOGE(context_->GetNodeName(), "actual tiling data size %zu > context tiling data size %zu",
                        tilingData.GetDataSize(), rawTilingData->GetCapacity()),
                return ge::GRAPH_FAILED);
    tilingData.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("BatchNormV3", BatchNormV3WelfordReduceTilingBase, 30000);
} // namespace optiling
