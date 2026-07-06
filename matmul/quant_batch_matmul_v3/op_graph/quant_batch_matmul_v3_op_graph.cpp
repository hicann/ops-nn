/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <nlohmann/json.hpp>
#include <vector>
#include <sstream>

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "runtime/runtime/base.h"
#include "tiling/platform/platform_ascendc.h"

namespace OpCheckHelper {

const std::string INT4 = "int4";
const std::string INT8 = "int8";
const std::string INT32 = "int32";
const std::string INT64 = "int64";
const std::string UINT64 = "uint64";
const std::string FP16 = "float16";
const std::string FP32 = "float32";
const std::string ND = "ND";
const std::string NZ = "FRACTAL_NZ";

struct TensorSpec {
    std::string classify;
    struct {
        std::string name;
        std::vector<std::string> dtypes;
        std::vector<std::string> formats;
        std::vector<std::string> dynFormats;
    } desc;
};

struct DtFmtSpec {
    std::string dtype;
    std::string format;
    std::string unknownshapeFormat;
};

enum TensorIdx { X1 = 0, X2, SCALE, OFFSET, BIAS, PERTOKEN_SCALE, Y, END };

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

void ToJson(nlohmann::json& j, const std::vector<TensorSpec>& result)
{
    for (auto& op : result) {
        j[op.classify] = {
            {"name", op.desc.name},
            {"dtype", Join(op.desc.dtypes, ",")},
            {"format", Join(op.desc.formats, ",")},
            {"unknownshape_format", Join(op.desc.dynFormats, ",")},
        };
    }
}

const std::vector<std::array<DtFmtSpec, TensorIdx::END>> listWeightNZ = {
    // 实现QbmmV3 a8w8场景常量折叠时，weight只保留NZ格式
    // {{[X1],             [X2],         [SCALE],          [OFFSET],        [BIAS],     [PERTOKEN_SCALE],    [Y]}}
    {{{INT8, ND, ND},
      {INT8, NZ, NZ},
      {UINT64, ND, ND},
      {FP32, ND, ND},
      {INT32, ND, ND},
      {FP32, ND, ND},
      {FP16, ND, ND}}},
    {{{INT8, ND, ND},
      {INT8, NZ, NZ},
      {UINT64, ND, ND},
      {FP32, ND, ND},
      {INT32, ND, ND},
      {FP32, ND, ND},
      {INT8, ND, ND}}},
    {{{INT8, ND, ND},
      {INT8, NZ, NZ},
      {INT64, ND, ND},
      {FP32, ND, ND},
      {INT32, ND, ND},
      {FP32, ND, ND},
      {FP16, ND, ND}}},
    {{{INT8, ND, ND},
      {INT8, NZ, NZ},
      {INT64, ND, ND},
      {FP32, ND, ND},
      {INT32, ND, ND},
      {FP32, ND, ND},
      {INT8, ND, ND}}},
    {{{INT8, ND, ND},
      {INT4, ND, ND},
      {UINT64, ND, ND},
      {FP32, ND, ND},
      {INT32, ND, ND},
      {FP32, ND, ND},
      {FP16, ND, ND}}},
    {{{INT8, ND, ND},
      {INT4, ND, ND},
      {UINT64, ND, ND},
      {FP32, ND, ND},
      {INT32, ND, ND},
      {FP32, ND, ND},
      {INT8, ND, ND}}},
    {{{INT8, ND, ND},
      {INT4, ND, ND},
      {INT64, ND, ND},
      {FP32, ND, ND},
      {INT32, ND, ND},
      {FP32, ND, ND},
      {FP16, ND, ND}}},
    {{{INT8, ND, ND},
      {INT4, ND, ND},
      {INT64, ND, ND},
      {FP32, ND, ND},
      {INT32, ND, ND},
      {FP32, ND, ND},
      {INT8, ND, ND}}},
    {{{INT8, ND, ND},
      {INT4, NZ, NZ},
      {UINT64, ND, ND},
      {FP32, ND, ND},
      {INT32, ND, ND},
      {FP32, ND, ND},
      {FP16, ND, ND}}},
    {{{INT8, ND, ND},
      {INT4, NZ, NZ},
      {UINT64, ND, ND},
      {FP32, ND, ND},
      {INT32, ND, ND},
      {FP32, ND, ND},
      {INT8, ND, ND}}},
    {{{INT8, ND, ND},
      {INT4, NZ, NZ},
      {INT64, ND, ND},
      {FP32, ND, ND},
      {INT32, ND, ND},
      {FP32, ND, ND},
      {FP16, ND, ND}}},
    {{{INT8, ND, ND},
      {INT4, NZ, NZ},
      {INT64, ND, ND},
      {FP32, ND, ND},
      {INT32, ND, ND},
      {FP32, ND, ND},
      {INT8, ND, ND}}}};

const std::vector<std::array<DtFmtSpec, TensorIdx::END>> listQbmmV3 = {
    // 非常量折叠场景与原始QbmmV3算子信息库内容一致
    // {{[X1],            [X2],           [SCALE],         [OFFSET],       [BIAS],      [PERTOKEN_SCALE],      [Y]}}
    {{{INT8, ND, ND},
      {INT8, ND, ND},
      {UINT64, ND, ND},
      {FP32, ND, ND},
      {INT32, ND, ND},
      {FP32, ND, ND},
      {FP16, ND, ND}}},
    {{{INT8, ND, ND},
      {INT8, ND, ND},
      {UINT64, ND, ND},
      {FP32, ND, ND},
      {INT32, ND, ND},
      {FP32, ND, ND},
      {INT8, ND, ND}}},
    {{{INT8, ND, ND},
      {INT8, NZ, NZ},
      {UINT64, ND, ND},
      {FP32, ND, ND},
      {INT32, ND, ND},
      {FP32, ND, ND},
      {FP16, ND, ND}}},
    {{{INT8, ND, ND},
      {INT8, NZ, NZ},
      {UINT64, ND, ND},
      {FP32, ND, ND},
      {INT32, ND, ND},
      {FP32, ND, ND},
      {INT8, ND, ND}}},
    {{{INT8, ND, ND},
      {INT8, ND, ND},
      {INT64, ND, ND},
      {FP32, ND, ND},
      {INT32, ND, ND},
      {FP32, ND, ND},
      {FP16, ND, ND}}},
    {{{INT8, ND, ND},
      {INT8, ND, ND},
      {INT64, ND, ND},
      {FP32, ND, ND},
      {INT32, ND, ND},
      {FP32, ND, ND},
      {INT8, ND, ND}}},
    {{{INT8, ND, ND},
      {INT8, NZ, NZ},
      {INT64, ND, ND},
      {FP32, ND, ND},
      {INT32, ND, ND},
      {FP32, ND, ND},
      {FP16, ND, ND}}},
    {{{INT8, ND, ND},
      {INT8, NZ, NZ},
      {INT64, ND, ND},
      {FP32, ND, ND},
      {INT32, ND, ND},
      {FP32, ND, ND},
      {INT8, ND, ND}}},
    {{{INT8, ND, ND},
      {INT4, ND, ND},
      {UINT64, ND, ND},
      {FP32, ND, ND},
      {INT32, ND, ND},
      {FP32, ND, ND},
      {FP16, ND, ND}}},
    {{{INT8, ND, ND},
      {INT4, ND, ND},
      {UINT64, ND, ND},
      {FP32, ND, ND},
      {INT32, ND, ND},
      {FP32, ND, ND},
      {INT8, ND, ND}}},
    {{{INT8, ND, ND},
      {INT4, NZ, NZ},
      {UINT64, ND, ND},
      {FP32, ND, ND},
      {INT32, ND, ND},
      {FP32, ND, ND},
      {FP16, ND, ND}}},
    {{{INT8, ND, ND},
      {INT4, NZ, NZ},
      {UINT64, ND, ND},
      {FP32, ND, ND},
      {INT32, ND, ND},
      {FP32, ND, ND},
      {INT8, ND, ND}}},
    {{{INT8, ND, ND},
      {INT4, ND, ND},
      {INT64, ND, ND},
      {FP32, ND, ND},
      {INT32, ND, ND},
      {FP32, ND, ND},
      {FP16, ND, ND}}},
    {{{INT8, ND, ND},
      {INT4, ND, ND},
      {INT64, ND, ND},
      {FP32, ND, ND},
      {INT32, ND, ND},
      {FP32, ND, ND},
      {INT8, ND, ND}}},
    {{{INT8, ND, ND},
      {INT4, NZ, NZ},
      {INT64, ND, ND},
      {FP32, ND, ND},
      {INT32, ND, ND},
      {FP32, ND, ND},
      {FP16, ND, ND}}},
    {{{INT8, ND, ND},
      {INT4, NZ, NZ},
      {INT64, ND, ND},
      {FP32, ND, ND},
      {INT32, ND, ND},
      {FP32, ND, ND},
      {INT8, ND, ND}}}};

class QbmmV3OpCheckHelper {
public:
    explicit QbmmV3OpCheckHelper(const gert::OpCheckContext* context);
    virtual ~QbmmV3OpCheckHelper() = default;
    ge::graphStatus OpSelectFormat(ge::AscendString& result) const;
    using DtFmtSpecList = std::vector<std::array<DtFmtSpec, TensorIdx::END>>;
    DtFmtSpecList GetDtFmtSpecList() const;

private:
    DtFmtSpecList dtFmtSpecList_;
};

QbmmV3OpCheckHelper::QbmmV3OpCheckHelper(const gert::OpCheckContext* context)
{
    const ge::AscendString weight = "x2";
    if (context != nullptr && context->IsConstInput(weight)) {
        dtFmtSpecList_ = listWeightNZ;
    } else {
        dtFmtSpecList_ = listQbmmV3;
    }
}

QbmmV3OpCheckHelper::DtFmtSpecList QbmmV3OpCheckHelper::GetDtFmtSpecList() const { return dtFmtSpecList_; }

ge::graphStatus QbmmV3OpCheckHelper::OpSelectFormat(ge::AscendString& result) const
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

    // 将dtFmtSpecList_内容拼成算子信息库的形式
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
    opInfo.push_back(genTensorSpec("input0", "x1", TensorIdx::X1));
    opInfo.push_back(genTensorSpec("input1", "x2", TensorIdx::X2));
    opInfo.push_back(genTensorSpec("input2", "scale", TensorIdx::SCALE));
    opInfo.push_back(genTensorSpec("input3", "offset", TensorIdx::OFFSET));
    opInfo.push_back(genTensorSpec("input4", "bias", TensorIdx::BIAS));
    opInfo.push_back(genTensorSpec("input5", "pertoken_scale", TensorIdx::PERTOKEN_SCALE));
    opInfo.push_back(genTensorSpec("output0", "y", TensorIdx::Y));

    nlohmann::json jsonResult;
    OpCheckHelper::ToJson(jsonResult, opInfo);
    result = ge::AscendString(jsonResult.dump().c_str());
    return ge::GRAPH_SUCCESS;
}
} // namespace OpCheckHelper

namespace optiling {
ge::graphStatus QbmmV3OpSelectFormat(const gert::OpCheckContext* context, ge::AscendString& result)
{
    OpCheckHelper::QbmmV3OpCheckHelper qbmmV3OpCheckHelper(context);
    qbmmV3OpCheckHelper.OpSelectFormat(result);
    return ge::GRAPH_SUCCESS;
}

// OpSelectFormat仅在算子信息库为format、unknownShapeFormat为空时生效；当前只应用于supportMmadS8S4场景
IMPL_OP(QuantBatchMatmulV3).OpSelectFormat(QbmmV3OpSelectFormat);
} // namespace optiling