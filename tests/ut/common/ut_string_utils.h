/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NN_TESTS_UT_COMMON_UT_STRING_UTILS_H_
#define NN_TESTS_UT_COMMON_UT_STRING_UTILS_H_

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef _WIN32
#include <limits.h>
#include <unistd.h>
#endif

#if __has_include("aclnn/aclnn_base.h")
#include "aclnn/aclnn_base.h"
#if __has_include("opdev/op_errno.h")
#include "opdev/op_errno.h"
#endif
#if __has_include("opdev/platform.h")
#include "opdev/platform.h"
#endif
#define NN_TESTS_UT_COMMON_HAS_ACLNN_TYPES
// Prefer aclnn status codes from opdev/op_errno.h; fall back to CANN doc values if absent.
#ifndef ACLNN_SUCCESS
#define ACLNN_SUCCESS static_cast<aclnnStatus>(0)
#endif
#ifndef ACLNN_ERR_PARAM_NULLPTR
#define ACLNN_ERR_PARAM_NULLPTR static_cast<aclnnStatus>(161001)
#endif
#ifndef ACLNN_ERR_PARAM_INVALID
#define ACLNN_ERR_PARAM_INVALID static_cast<aclnnStatus>(161002)
#endif
#endif

#if __has_include("exe_graph/runtime/shape.h") && __has_include("exe_graph/runtime/storage_shape.h") && \
                                                                __has_include("graph/types.h")
#include "exe_graph/runtime/shape.h"
#include "exe_graph/runtime/storage_shape.h"
#include "graph/types.h"
#define NN_TESTS_UT_COMMON_HAS_GE_TYPES
#endif

namespace ut_str {

inline std::string GetExeDirPath()
{
#ifdef _WIN32
    return "./";
#else
    std::string exePath("./");
    char path[PATH_MAX];
    ssize_t n = readlink("/proc/self/exe", path, sizeof(path));
    if (n > 0) {
        path[n] = '\0';
        exePath.assign(path);
        auto pos = exePath.find_last_of('/');
        if (pos != std::string::npos) {
            exePath.erase(pos + 1);
        } else {
            exePath.assign("./");
        }
    }
    return exePath;
#endif
}

inline std::string Trim(const std::string& input)
{
    const auto first = std::find_if_not(input.begin(), input.end(),
                                        [](unsigned char ch) { return std::isspace(ch) != 0; });
    if (first == input.end()) {
        return "";
    }
    const auto last = std::find_if_not(input.rbegin(), input.rend(), [](unsigned char ch) {
                          return std::isspace(ch) != 0;
                      }).base();
    return std::string(first, last);
}

inline void SplitStr2Vec(const std::string& input, const std::string& delimiter, std::vector<std::string>& output)
{
    output.clear();
    if (delimiter.empty()) {
        if (!input.empty()) {
            output.emplace_back(input);
        }
        return;
    }
    const auto delimiterLen = delimiter.size();
    std::string::size_type currPos = 0;
    std::string::size_type nextPos = input.find(delimiter, currPos);
    while (nextPos != std::string::npos) {
        output.emplace_back(input.substr(currPos, nextPos - currPos));
        currPos = nextPos + delimiterLen;
        nextPos = input.find(delimiter, currPos);
    }
    if (currPos <= input.size()) {
        output.emplace_back(input.substr(currPos));
    }
}

inline std::vector<int64_t> ParseInt64Vec(const std::string& value)
{
    std::vector<int64_t> result;
    const std::string trimmed = Trim(value);
    if (trimmed.empty()) {
        return result;
    }
    std::vector<std::string> items;
    SplitStr2Vec(trimmed, " ", items);
    for (size_t tokenIdx = 0; tokenIdx < items.size(); ++tokenIdx) {
        const auto& item = items[tokenIdx];
        const std::string token = Trim(item);
        if (token.empty()) {
            continue;
        }
        try {
            result.push_back(static_cast<int64_t>(std::stoll(token)));
        } catch (const std::exception& e) {
            std::cerr << "ParseInt64Vec skip invalid token, original value: \"" << value
                      << "\", token index: " << tokenIdx << ", token: \"" << token << "\", error: " << e.what()
                      << std::endl;
            continue;
        }
    }
    return result;
}

inline bool ParseBool(const std::string& value)
{
    std::string t = Trim(value);
    std::transform(t.begin(), t.end(), t.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return t == "true" || t == "1";
}

inline int ParseIntOrDefault(const std::string& value, int defaultValue)
{
    const std::string trimmed = Trim(value);
    if (trimmed.empty()) {
        return defaultValue;
    }
    try {
        return std::stoi(trimmed);
    } catch (const std::exception&) {
        return defaultValue;
    }
}

inline int64_t ParseInt64OrDefault(const std::string& value, int64_t defaultValue)
{
    const std::string trimmed = Trim(value);
    if (trimmed.empty()) {
        return defaultValue;
    }
    try {
        return static_cast<int64_t>(std::stoll(trimmed));
    } catch (const std::exception&) {
        return defaultValue;
    }
}

#ifdef NN_TESTS_UT_COMMON_HAS_GE_TYPES
inline ge::graphStatus ParseGraphStatus(const std::string& value, ge::graphStatus defaultValue = ge::GRAPH_FAILED)
{
    std::string lowered = Trim(value);
    std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    if (lowered == "graph_success" || lowered == "success" || lowered == "1") {
        return ge::GRAPH_SUCCESS;
    }
    if (lowered == "graph_failed" || lowered == "failed" || lowered == "0") {
        return ge::GRAPH_FAILED;
    }
    return defaultValue;
}

inline gert::Shape ToShape(const std::vector<int64_t>& dims)
{
    gert::Shape shape;
    for (const auto dim : dims) {
        shape.AppendDim(dim);
    }
    return shape;
}

inline gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape;
    shape.MutableStorageShape() = ToShape(dims);
    shape.MutableOriginShape() = shape.MutableStorageShape();
    return shape;
}

inline gert::Shape ParseShape(const std::string& value) { return ToShape(ParseInt64Vec(value)); }

inline ge::DataType ParseDtype(const std::string& value, ge::DataType defaultValue = ge::DT_FLOAT16)
{
    static const std::map<std::string, ge::DataType> kDtypeMap = {
        {"FLOAT16", ge::DT_FLOAT16},
        {"FP16", ge::DT_FLOAT16},
        {"DT_FLOAT16", ge::DT_FLOAT16},
        {"FLOAT", ge::DT_FLOAT},
        {"FLOAT32", ge::DT_FLOAT},
        {"FP32", ge::DT_FLOAT},
        {"DT_FLOAT", ge::DT_FLOAT},
        {"DT_FLOAT32", ge::DT_FLOAT},
        {"BF16", ge::DT_BF16},
        {"DT_BF16", ge::DT_BF16},
        {"INT8", ge::DT_INT8},
        {"DT_INT8", ge::DT_INT8},
        {"INT4", ge::DT_INT4},
        {"DT_INT4", ge::DT_INT4},
        {"INT2", ge::DT_INT2},
        {"DT_INT2", ge::DT_INT2},
        {"UINT1", ge::DT_UINT1},
        {"DT_UINT1", ge::DT_UINT1},
        {"UINT64", ge::DT_UINT64},
        {"DT_UINT64", ge::DT_UINT64},
        {"INT32", ge::DT_INT32},
        {"DT_INT32", ge::DT_INT32},
        {"INT64", ge::DT_INT64},
        {"DT_INT64", ge::DT_INT64},
        {"HIFLOAT8", ge::DT_HIFLOAT8},
        {"DT_HIFLOAT8", ge::DT_HIFLOAT8},
        {"FLOAT8-E8M0", ge::DT_FLOAT8_E8M0},
        {"FP8-E8M0", ge::DT_FLOAT8_E8M0},
        {"DT_FLOAT8_E8M0", ge::DT_FLOAT8_E8M0},
        {"FLOAT8-E5M2", ge::DT_FLOAT8_E5M2},
        {"FP8-E5M2", ge::DT_FLOAT8_E5M2},
        {"DT_FLOAT8_E5M2", ge::DT_FLOAT8_E5M2},
        {"FLOAT8-E4M3", ge::DT_FLOAT8_E4M3FN},
        {"FP8-E4M3", ge::DT_FLOAT8_E4M3FN},
        {"DT_FLOAT8_E4M3FN", ge::DT_FLOAT8_E4M3FN},
        {"FLOAT4-E2M1", ge::DT_FLOAT4_E2M1},
        {"FP4-E2M1", ge::DT_FLOAT4_E2M1},
        {"DT_FLOAT4_E2M1", ge::DT_FLOAT4_E2M1},
    };
    const auto dtypeStr = Trim(value);
    const auto it = kDtypeMap.find(dtypeStr);
    if (it != kDtypeMap.end()) {
        return it->second;
    }
    return defaultValue;
}

inline ge::Format ParseFormat(const std::string& value, ge::Format defaultValue = ge::FORMAT_ND)
{
    static const std::map<std::string, ge::Format> kFormatMap = {
        {"ND", ge::FORMAT_ND},
        {"FORMAT_ND", ge::FORMAT_ND},
        {"NZ", ge::FORMAT_FRACTAL_NZ},
        {"FORMAT_FRACTAL_NZ", ge::FORMAT_FRACTAL_NZ},
    };
    const auto formatStr = Trim(value);
    const auto it = kFormatMap.find(formatStr);
    if (it != kFormatMap.end()) {
        return it->second;
    }
    return defaultValue;
}
#endif

#ifdef NN_TESTS_UT_COMMON_HAS_ACLNN_TYPES
inline aclDataType ParseAclDataType(const std::string& value, aclDataType defaultValue = ACL_FLOAT16)
{
    static const std::map<std::string, aclDataType> kDtypeMap = {
        {"ACL_INT4", ACL_INT4},
        {"ACL_INT8", ACL_INT8},
        {"ACL_INT64", ACL_INT64},
        {"ACL_INT32", ACL_INT32},
        {"ACL_UINT64", ACL_UINT64},
        {"ACL_FLOAT", ACL_FLOAT},
        {"ACL_FLOAT16", ACL_FLOAT16},
        {"ACL_BF16", ACL_BF16},
        {"ACL_HIFLOAT8", ACL_HIFLOAT8},
        {"ACL_FLOAT8_E4M3FN", ACL_FLOAT8_E4M3FN},
        {"ACL_FLOAT8_E5M2", ACL_FLOAT8_E5M2},
        {"ACL_FLOAT8_E8M0", ACL_FLOAT8_E8M0},
        {"ACL_FLOAT4_E2M1", ACL_FLOAT4_E2M1},
    };
    const auto it = kDtypeMap.find(Trim(value));
    if (it != kDtypeMap.end()) {
        return it->second;
    }
    return defaultValue;
}

inline aclnnStatus ParseAclnnStatus(const std::string& value)
{
    static const std::map<std::string, aclnnStatus> kStatusMap = {
        {"ACLNN_SUCCESS", ACLNN_SUCCESS},
        {"ACLNN_ERR_PARAM_INVALID", ACLNN_ERR_PARAM_INVALID},
        {"ACLNN_ERR_PARAM_NULLPTR", ACLNN_ERR_PARAM_NULLPTR},
    };
    const auto it = kStatusMap.find(Trim(value));
    if (it != kStatusMap.end()) {
        return it->second;
    }
    return ACLNN_ERR_PARAM_INVALID;
}

inline aclFormat ParseAclFormat(const std::string& value)
{
    static const std::map<std::string, aclFormat> kFormatMap = {
        {"ACL_FORMAT_ND", ACL_FORMAT_ND},
        {"ACL_FORMAT_NCHW", ACL_FORMAT_NCHW},
        {"ACL_FORMAT_FRACTAL_NZ", ACL_FORMAT_FRACTAL_NZ},
        {"ACL_FORMAT_FRACTAL_NZ_C0_32", ACL_FORMAT_FRACTAL_NZ_C0_32},
    };
    const auto it = kFormatMap.find(Trim(value));
    if (it != kFormatMap.end()) {
        return it->second;
    }
    return ACL_FORMAT_ND;
}
#endif

inline std::vector<int64_t> MakeDefaultStride(const std::vector<int64_t>& shape)
{
    std::vector<int64_t> stride(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        stride[static_cast<size_t>(i)] = stride[static_cast<size_t>(i + 1)] * shape[static_cast<size_t>(i + 1)];
    }
    return stride;
}

} // namespace ut_str

#endif // NN_TESTS_UT_COMMON_UT_STRING_UTILS_H_
