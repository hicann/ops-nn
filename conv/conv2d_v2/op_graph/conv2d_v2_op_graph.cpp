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
 * \file conv2d_v2_op_graph.cpp
 * \brief Conv2DV2 OpSelectFormat实现：算子信息库侧的format、unknownShapeFormat列表置空后，
 *        dtype/format组合全量由本实现下发。根据fmap C轴大小、groups、Win上限及C04 HwSplit
 *        模式的L1装载量校验结果，动态选择FRACTAL_Z或FRACTAL_Z_C04的dtype/format组合；
 *        filter（weight）额外下发sub_format=0（单个值，对全部组合生效）。
 */

#include <algorithm>
#include <array>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "graph/utils/type_utils.h"
#include "log/log.h"
#include "register/op_impl_registry.h"

namespace ops {
namespace {
static const std::string kFp16 = "float16";
static const std::string kInt8 = "int8";
static const std::string kNchw = "NCHW";
static const std::string kNhwc = "NHWC";
static const std::string kFractalZ = "FRACTAL_Z";
static const std::string kFractalZC04 = "FRACTAL_Z_C04";
static const std::string kNd = "ND";

struct DtFmtSpec {
    std::string dtype;
    std::string format;
    std::string unknownshapeFormat;
};

enum TensorIdx { X = 0, FILTER, BIAS, OFFSET_W, Y, END };

struct TensorSpec {
    std::string classify;
    struct {
        std::string name;
        std::vector<std::string> dtypes;
        std::vector<std::string> formats;
        std::vector<std::string> dynFormats;
        std::vector<std::string> subFormats;
    } desc;
};

template <typename T>
std::string Join(const std::vector<T>& sequence, std::string const& separator)
{
    std::ostringstream result;
    typename std::vector<T>::const_iterator iter = sequence.begin();
    if (iter != sequence.end()) {
        result << *iter++;
    }
    while (iter != sequence.end()) {
        result << separator << *iter++;
    }
    return result.str();
}

// 由nlohmann序列化，字符串转义由库保证；sub_format非空时下发（逗号分隔），
// FE解析后写入ext_dynamic_sub_format作为该tensor支持的subformat列表
inline void ToJson(nlohmann::json& jsonResult, const std::vector<TensorSpec>& result)
{
    for (const auto& op : result) {
        nlohmann::json descJson = {
            {"name", op.desc.name},
            {"dtype", Join(op.desc.dtypes, ",")},
            {"format", Join(op.desc.formats, ",")},
            {"unknownshape_format", Join(op.desc.dynFormats, ",")},
        };
        if (!op.desc.subFormats.empty()) {
            descJson["sub_format"] = Join(op.desc.subFormats, ",");
        }
        jsonResult[op.classify] = descJson;
    }
}

using DtFmtSpecList = std::vector<std::array<DtFmtSpec, TensorIdx::END>>;

// 全量组合按filter格式拆分：filter为FRACTAL_Z的组合进listConv2DV2FractalZ，
// filter为FRACTAL_Z_C04的组合进listConv2DV2FractalZC04
const DtFmtSpecList listConv2DV2FractalZ = {
    // {{[X],              [FILTER],                 [BIAS],          [OFFSET_W],      [Y]}}
    {{{kFp16, kNchw, kNchw},
      {kFp16, kFractalZ, kFractalZ},
      {kFp16, kNd, kNd},
      {kInt8, kNd, kNd},
      {kFp16, kNchw, kNchw}}},
    {{{kFp16, kNhwc, kNhwc},
      {kFp16, kFractalZ, kFractalZ},
      {kFp16, kNd, kNd},
      {kInt8, kNd, kNd},
      {kFp16, kNhwc, kNhwc}}},
    {{{kFp16, kNchw, kNchw},
      {kFp16, kFractalZ, kFractalZ},
      {kFp16, kNd, kNd},
      {kInt8, kNd, kNd},
      {kFp16, kNhwc, kNhwc}}},
    {{{kFp16, kNhwc, kNhwc},
      {kFp16, kFractalZ, kFractalZ},
      {kFp16, kNd, kNd},
      {kInt8, kNd, kNd},
      {kFp16, kNchw, kNchw}}},
};

const DtFmtSpecList listConv2DV2FractalZC04 = {
    // {{[X],              [FILTER],                       [BIAS],          [OFFSET_W],      [Y]}}
    {{{kFp16, kNchw, kNchw},
      {kFp16, kFractalZC04, kFractalZC04},
      {kFp16, kNd, kNd},
      {kInt8, kNd, kNd},
      {kFp16, kNchw, kNchw}}},
    {{{kFp16, kNhwc, kNhwc},
      {kFp16, kFractalZC04, kFractalZC04},
      {kFp16, kNd, kNd},
      {kInt8, kNd, kNd},
      {kFp16, kNhwc, kNhwc}}},
    {{{kFp16, kNchw, kNchw},
      {kFp16, kFractalZC04, kFractalZC04},
      {kFp16, kNd, kNd},
      {kInt8, kNd, kNd},
      {kFp16, kNhwc, kNhwc}}},
    {{{kFp16, kNhwc, kNhwc},
      {kFp16, kFractalZC04, kFractalZC04},
      {kFp16, kNd, kNd},
      {kInt8, kNd, kNd},
      {kFp16, kNchw, kNchw}}},
};

struct CubeMkn {
    int64_t m;
    int64_t k;
    int64_t n;
};

inline CubeMkn GetCubeMkn(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_INT8:
        case ge::DT_UINT8:
            return {16, 32, 16};
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
        case ge::DT_INT16:
        case ge::DT_UINT16:
            return {16, 16, 16};
        default:
            return {16, 16, 16};
    }
}

inline int64_t GetBitRatio(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_INT8:
        case ge::DT_UINT8:
        case ge::DT_INT4:
            return 1;
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
        case ge::DT_INT16:
        case ge::DT_UINT16:
            return 2;
        case ge::DT_INT32:
        case ge::DT_UINT32:
        case ge::DT_FLOAT:
            return 4;
        case ge::DT_INT64:
        case ge::DT_UINT64:
        case ge::DT_DOUBLE:
            return 8;
        default:
            return 1;
    }
}

inline int64_t AlignB(int64_t a, int64_t b)
{
    if (b == 0) {
        return a;
    }
    return ((a + b - 1) / b) * b;
}

inline int64_t CeilDiv(int64_t a, int64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

inline bool CheckC04HwSplitModeL1Valid(const gert::OpCheckContext* context)
{
    const auto* fmapDesc = context->GetInputDesc(0);
    const auto* fmapShape = context->GetInputShape(0);
    const auto* weightDesc = context->GetInputDesc(1);
    const auto* weightShape = context->GetInputShape(1);
    const auto* attrs = context->GetAttrs();
    if (fmapDesc == nullptr || fmapShape == nullptr || weightDesc == nullptr || weightShape == nullptr ||
        attrs == nullptr) {
        OP_LOGD(context->GetNodeName(), "[OpSelectFormat] CheckC04HwSplitModeL1Valid: nullptr input, return false.");
        return false;
    }

    const char* dataFormatPtr = attrs->GetStr(4);
    if (dataFormatPtr == nullptr) {
        OP_LOGD(context->GetNodeName(),
                "[OpSelectFormat] CheckC04HwSplitModeL1Valid: data_format attr is null, return false.");
        return false;
    }
    std::string dataFormat = dataFormatPtr;
    size_t posH = dataFormat.find('H');
    size_t posW = dataFormat.find('W');
    if (posH == std::string::npos || posW == std::string::npos) {
        OP_LOGD(context->GetNodeName(),
                "[OpSelectFormat] CheckC04HwSplitModeL1Valid: H/W not found in data_format[%s], return false.",
                dataFormat.c_str());
        return false;
    }

    const auto* stridesList = attrs->GetListInt(0);
    const auto* padsList = attrs->GetListInt(1);
    const auto* dilationsList = attrs->GetListInt(2);
    if (stridesList == nullptr || padsList == nullptr || dilationsList == nullptr) {
        OP_LOGD(context->GetNodeName(),
                "[OpSelectFormat] CheckC04HwSplitModeL1Valid: strides/pads/dilations attr is null, return false.");
        return false;
    }
    const int64_t* strides = stridesList->GetData();
    const int64_t* pads = padsList->GetData();
    const int64_t* dilations = dilationsList->GetData();
    if (stridesList->GetSize() <= posH || stridesList->GetSize() <= posW || dilationsList->GetSize() <= posH ||
        dilationsList->GetSize() <= posW || padsList->GetSize() < 4) {
        OP_LOGD(context->GetNodeName(),
                "[OpSelectFormat] CheckC04HwSplitModeL1Valid: strides/dilations/pads size too small, return false.");
        return false;
    }
    int64_t strideH = strides[posH];
    int64_t strideW = strides[posW];
    int64_t dilateH = dilations[posH];
    int64_t dilateW = dilations[posW];
    int64_t padLeft = pads[2];
    int64_t padRight = pads[3];

    std::string fmapOriFormat = ge::TypeUtils::FormatToSerialString(fmapDesc->GetOriginFormat());
    size_t fmapPosH = fmapOriFormat.find('H');
    size_t fmapPosW = fmapOriFormat.find('W');
    if (fmapPosH == std::string::npos || fmapPosW == std::string::npos) {
        OP_LOGD(context->GetNodeName(),
                "[OpSelectFormat] CheckC04HwSplitModeL1Valid: H/W not found in fmap ori_format[%s], return false.",
                fmapOriFormat.c_str());
        return false;
    }
    const auto& fmapOriShape = fmapShape->GetOriginShape();
    if (fmapOriShape.GetDimNum() <= fmapPosH || fmapOriShape.GetDimNum() <= fmapPosW) {
        OP_LOGD(context->GetNodeName(),
                "[OpSelectFormat] CheckC04HwSplitModeL1Valid: fmap shape dim_num too small, return false.");
        return false;
    }
    int64_t hIn = fmapOriShape.GetDim(fmapPosH);
    int64_t wIn = fmapOriShape.GetDim(fmapPosW);

    std::string weightOriFormat = ge::TypeUtils::FormatToSerialString(weightDesc->GetOriginFormat());
    size_t weightPosH = weightOriFormat.find('H');
    size_t weightPosW = weightOriFormat.find('W');
    if (weightPosH == std::string::npos || weightPosW == std::string::npos) {
        OP_LOGD(context->GetNodeName(),
                "[OpSelectFormat] CheckC04HwSplitModeL1Valid: H/W not found in weight ori_format[%s], return false.",
                weightOriFormat.c_str());
        return false;
    }
    const auto& weightOriShape = weightShape->GetOriginShape();
    if (weightOriShape.GetDimNum() <= weightPosH || weightOriShape.GetDimNum() <= weightPosW) {
        OP_LOGD(context->GetNodeName(),
                "[OpSelectFormat] CheckC04HwSplitModeL1Valid: weight shape dim_num too small, return false.");
        return false;
    }
    int64_t hK = weightOriShape.GetDim(weightPosH);
    int64_t wK = weightOriShape.GetDim(weightPosW);

    int64_t hkDilation = (hK - 1) * dilateH + 1;
    int64_t wkDilation = (wK - 1) * dilateW + 1;
    int64_t wOut = (wIn + padLeft + padRight - wkDilation) / strideW + 1;
    if (wOut <= 0) {
        OP_LOGD(context->GetNodeName(), "[OpSelectFormat] CheckC04HwSplitModeL1Valid: wOut=%ld <= 0, return false.",
                wOut);
        return false;
    }

    CubeMkn cube = GetCubeMkn(fmapDesc->GetDataType());
    int64_t m0 = cube.m;
    int64_t k0 = cube.k;
    int64_t n0 = cube.n;
    int64_t wBitRatio = GetBitRatio(weightDesc->GetDataType());
    int64_t fmapBitRatio = GetBitRatio(fmapDesc->GetDataType());

    constexpr int64_t kC04CinValue = 4;
    int64_t kbMin = AlignB(kC04CinValue * hK * wK, k0);
    int64_t weightUsedL1Size = AlignB(kbMin * n0 * wBitRatio, k0);
    int64_t hoAl1Min = (wOut < m0) ? CeilDiv(m0, wOut) : 1;
    int64_t hiAl1Min = std::min(hIn, (hoAl1Min - 1) * strideH + hkDilation);
    int64_t fmapUsedL1Size = AlignB(hiAl1Min * wIn * kC04CinValue * fmapBitRatio, k0);
    int64_t minL1LoadSize = fmapUsedL1Size + weightUsedL1Size;

    constexpr int64_t kL1BufferSize = 1048576; // 1M
    constexpr int64_t kFb0Size = 4096;
    constexpr int64_t kFb1Size = 2048;
    constexpr int64_t kBtSize = 4096;
    int64_t l1BufferSize = kL1BufferSize - kFb0Size - kFb1Size - kBtSize;

    OP_LOGD(context->GetNodeName(),
            "[OpSelectFormat] CheckC04HwSplitModeL1Valid: hIn=%ld, wIn=%ld, hK=%ld, wK=%ld, strideH=%ld, strideW=%ld, "
            "dilateH=%ld, dilateW=%ld, padLeft=%ld, padRight=%ld, wOut=%ld, m0=%ld, k0=%ld, n0=%ld, "
            "wBitRatio=%ld, fmapBitRatio=%ld, kbMin=%ld, weightUsedL1Size=%ld, hoAl1Min=%ld, hiAl1Min=%ld, "
            "fmapUsedL1Size=%ld, minL1LoadSize=%ld, l1BufferSize=%ld.",
            hIn, wIn, hK, wK, strideH, strideW, dilateH, dilateW, padLeft, padRight, wOut, m0, k0, n0, wBitRatio,
            fmapBitRatio, kbMin, weightUsedL1Size, hoAl1Min, hiAl1Min, fmapUsedL1Size, minL1LoadSize, l1BufferSize);

    bool valid = minL1LoadSize <= l1BufferSize;
    OP_LOGD(context->GetNodeName(), "[OpSelectFormat] CheckC04HwSplitModeL1Valid: result=%d.", valid);
    return valid;
}

// 从context判定是否走FRACTAL_Z_C04分支（c04 HwSplit模式条件全满足时使能）
bool GetC04EnableFlag(const gert::OpCheckContext* context)
{
    OP_LOGD(context == nullptr ? "" : context->GetNodeName(), "[OpSelectFormat] Conv2dV2CheckHelper enter, context=%p.",
            context);
    bool c04EnableFlag = false;
    if (context != nullptr) {
        const auto* fmapDesc = context->GetInputDesc(0);
        const auto* fmapShape = context->GetInputShape(0);
        const auto* attrs = context->GetAttrs();
        if (fmapDesc != nullptr && fmapShape != nullptr && attrs != nullptr) {
            int cIndex = 1;
            int wIndex = 3;
            if (fmapDesc->GetOriginFormat() == ge::FORMAT_NHWC) {
                cIndex = 3;
                wIndex = 2;
            }
            const auto& originShape = fmapShape->GetOriginShape();
            if (originShape.GetDimNum() > static_cast<size_t>(wIndex)) {
                int64_t fmapC = originShape.GetDim(static_cast<size_t>(cIndex));
                int64_t fmapW = originShape.GetDim(static_cast<size_t>(wIndex));
                const int64_t* groupsPtr = attrs->GetInt(3);
                if (groupsPtr != nullptr && fmapC > 0 && fmapW > 0) {
                    constexpr int64_t kLoad3dv2WinLimit = 32767;
                    bool load3dv2WinLimitFlag = (fmapW <= kLoad3dv2WinLimit);
                    bool c04LoadL1ValidFlag = CheckC04HwSplitModeL1Valid(context);
                    // weight为int8时不走c04
                    const auto* weightDesc = context->GetInputDesc(1);
                    bool weightNotInt8Flag = (weightDesc != nullptr && weightDesc->GetDataType() != ge::DT_INT8);
                    c04EnableFlag = (fmapC <= 4 && *groupsPtr == 1 && load3dv2WinLimitFlag && c04LoadL1ValidFlag &&
                                     weightNotInt8Flag);
                    OP_LOGD(context->GetNodeName(),
                            "[OpSelectFormat] Conv2dV2CheckHelper: fmapC=%ld, fmapW=%ld, groups=%ld, "
                            "load3dv2WinLimitFlag=%d, c04LoadL1ValidFlag=%d, weightNotInt8Flag=%d, c04EnableFlag=%d.",
                            fmapC, fmapW, *groupsPtr, load3dv2WinLimitFlag, c04LoadL1ValidFlag, weightNotInt8Flag,
                            c04EnableFlag);
                } else {
                    OP_LOGD(context->GetNodeName(),
                            "[OpSelectFormat] Conv2dV2CheckHelper: skip c04 check, groupsPtr=%p, fmapC=%ld, fmapW=%ld.",
                            groupsPtr, fmapC, fmapW);
                }
            } else {
                OP_LOGD(context->GetNodeName(),
                        "[OpSelectFormat] Conv2dV2CheckHelper: originShape dim_num=%zu <= wIndex=%d, c04 disabled.",
                        originShape.GetDimNum(), wIndex);
            }
        } else {
            OP_LOGD(
                context->GetNodeName(),
                "[OpSelectFormat] Conv2dV2CheckHelper: nullptr desc/shape/attrs, fmapDesc=%p, fmapShape=%p, attrs=%p.",
                fmapDesc, fmapShape, attrs);
        }
    }
    OP_LOGD(context == nullptr ? "" : context->GetNodeName(), "[OpSelectFormat] GetC04EnableFlag: c04EnableFlag=%d.",
            c04EnableFlag);
    return c04EnableFlag;
}

class Conv2dV2CheckHelper {
public:
    explicit Conv2dV2CheckHelper(bool c04EnableFlag);
    virtual ~Conv2dV2CheckHelper() = default;
    ge::graphStatus OpSelectFormat(ge::AscendString& result) const;
    DtFmtSpecList GetDtFmtSpecList() const;

private:
    DtFmtSpecList dtFmtSpecList_;
};

Conv2dV2CheckHelper::Conv2dV2CheckHelper(bool c04EnableFlag)
{
    dtFmtSpecList_ = c04EnableFlag ? listConv2DV2FractalZC04 : listConv2DV2FractalZ;
}

DtFmtSpecList Conv2dV2CheckHelper::GetDtFmtSpecList() const { return dtFmtSpecList_; }

ge::graphStatus Conv2dV2CheckHelper::OpSelectFormat(ge::AscendString& result) const
{
    DtFmtSpecList dtfmtSpecList = GetDtFmtSpecList();
    auto getDtypeList = [&dtfmtSpecList](enum TensorIdx idx) -> std::vector<std::string> {
        std::vector<std::string> output;
        for (auto& spec : dtfmtSpecList) {
            output.push_back(spec[idx].dtype);
        }
        return output;
    };
    auto getFormatList = [&dtfmtSpecList](enum TensorIdx idx) -> std::vector<std::string> {
        std::vector<std::string> output;
        for (auto& spec : dtfmtSpecList) {
            output.push_back(spec[idx].format);
        }
        return output;
    };
    auto getUnknownShapeFormatList = [&dtfmtSpecList](enum TensorIdx idx) -> std::vector<std::string> {
        std::vector<std::string> output;
        for (auto& spec : dtfmtSpecList) {
            output.push_back(spec[idx].unknownshapeFormat);
        }
        return output;
    };

    auto genTensorSpec = [getDtypeList, getFormatList, getUnknownShapeFormatList](
                             std::string classify, std::string name, enum TensorIdx idx) -> TensorSpec {
        TensorSpec spec = {.classify = classify,
                           .desc = {.name = name,
                                    .dtypes = getDtypeList(idx),
                                    .formats = getFormatList(idx),
                                    .dynFormats = getUnknownShapeFormatList(idx)}};
        return spec;
    };
    std::vector<TensorSpec> opInfo;
    opInfo.push_back(genTensorSpec("input0", "x", TensorIdx::X));
    opInfo.push_back(genTensorSpec("input1", "filter", TensorIdx::FILTER));
    // weight的sub_format固定为0：仅下发一个0，对该tensor的全部format组合生效
    opInfo.back().desc.subFormats = {"0"};
    opInfo.push_back(genTensorSpec("input2", "bias", TensorIdx::BIAS));
    opInfo.push_back(genTensorSpec("input3", "offset_w", TensorIdx::OFFSET_W));
    opInfo.push_back(genTensorSpec("output0", "y", TensorIdx::Y));

    nlohmann::json jsonResult = nlohmann::json::object();
    ToJson(jsonResult, opInfo);
    const std::string jsonStr = jsonResult.dump();
    result = ge::AscendString(jsonStr.c_str());
    return ge::GRAPH_SUCCESS;
}
} // namespace

static ge::graphStatus Conv2dV2OpSelectFormat(const gert::OpCheckContext* context, ge::AscendString& result)
{
    Conv2dV2CheckHelper helper(GetC04EnableFlag(context));
    ge::graphStatus ret = helper.OpSelectFormat(result);
    OP_LOGD(context == nullptr ? "" : context->GetNodeName(),
            "[OpSelectFormat] Conv2dV2OpSelectFormat: json result = %s", result.GetString());
    return ret;
}

// OpSelectFormat仅在算子信息库format、unknownShapeFormat为空时生效
IMPL_OP(Conv2DV2).OpSelectFormat(Conv2dV2OpSelectFormat);
} // namespace ops
