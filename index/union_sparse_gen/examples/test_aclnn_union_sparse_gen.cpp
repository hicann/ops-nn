/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstdio>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_union_sparse_gen.h"

#define CHECK_RET(cond, expr) \
    do {                      \
        if (!(cond)) {        \
            expr;             \
        }                     \
    } while (0)

namespace {
constexpr int64_t kSampleBlockY = 128;
constexpr size_t kBlockShapeDimensions = 2;
constexpr int64_t kSampleUnionSize = 8, kSampleLayout = 4;
constexpr int64_t kCausalMask = 1, kExplicitBlocks = 0, kUnspecifiedMaxSeqLen = -1;
constexpr int32_t kFirstQueryBit = 1 << 0, kSecondQueryBit = 1 << 1;
enum SampleStatus {
    kInitFailed = 1,
    kSetDeviceFailed = 2,
    kCreateStreamFailed = 3,
    kCreateTensorFailed = 4,
    kWorkspaceQueryFailed = 5,
    kWorkspaceAllocFailed = 6,
    kLaunchFailed = 7,
    kSynchronizeFailed = 8,
    kReadbackFailed = 9,
    kOutputMismatch = 10
};

int64_t ShapeSize(const std::vector<int64_t>& shape)
{
    int64_t size = 1;
    for (const int64_t dim : shape) {
        size *= dim;
    }
    return size;
}

template <typename T>
int CreateTensor(const std::vector<T>& host, const std::vector<int64_t>& shape, void** device, aclTensor** tensor)
{
    const auto bytes = static_cast<size_t>(ShapeSize(shape)) * sizeof(T);
    auto ret = aclrtMalloc(device, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclrtMemcpy(*device, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = shape[static_cast<size_t>(i + 1)] * strides[static_cast<size_t>(i + 1)];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), sizeof(T) == sizeof(int64_t) ? ACL_INT64 : ACL_INT32,
                              strides.data(), 0, ACL_FORMAT_ND, shape.data(), shape.size(), *device);
    return *tensor == nullptr ? ACL_ERROR_FAILURE : ACL_SUCCESS;
}

struct SampleData {
    explicit SampleData(int64_t y) : blockY(y), maskHost(static_cast<size_t>(y), 0) {}
    int64_t blockY;
    std::vector<int32_t> unionHost{0}, unionCountHost{0}, maskHost;
    void* sparseDevice = nullptr;
    void* countDevice = nullptr;
    void* qDevice = nullptr;
    void* kvDevice = nullptr;
    void* seqQDevice = nullptr;
    void* seqKvDevice = nullptr;
    void* unionDevice = nullptr;
    void* unionCountDevice = nullptr;
    void* maskDevice = nullptr;
    void* workspace = nullptr;
    aclTensor* sparse = nullptr;
    aclTensor* count = nullptr;
    aclTensor* qLengths = nullptr;
    aclTensor* kvLengths = nullptr;
    aclTensor* seqQ = nullptr;
    aclTensor* seqKv = nullptr;
    aclTensor* unionIdx = nullptr;
    aclTensor* unionCount = nullptr;
    aclTensor* exactMask = nullptr;
    aclIntArray* blockShape = nullptr;
};

int CreateSampleTensors(SampleData& sample)
{
    // One head, three stored Q rows, two valid entries per row at most.
    // seqUsedQ keeps two rows; seqUsedKv keeps exactly one complete KV block.
    const std::vector<int64_t> sparseShape = {1, 3, 2};
    const std::vector<int64_t> countShape = {1, 3};
    const std::vector<int64_t> lengthShape = {2};
    const std::vector<int64_t> seqShape = {1};
    const std::vector<int64_t> unionShape = {1, 1, 1};
    const std::vector<int64_t> unionCountShape = {1, 1};
    const std::vector<int64_t> maskShape = {1, 1, sample.blockY};
    const std::vector<int32_t> sparseHost = {0, -1, 0, 1, 1, -1};
    const std::vector<int32_t> countHost = {1, 2, 1};
    const std::vector<int64_t> qHost = {0, 3};
    const std::vector<int64_t> kvHost = {0, static_cast<int32_t>(sample.blockY + 2)};
    const std::vector<int32_t> seqQHost = {2};
    const std::vector<int32_t> seqKvHost = {static_cast<int32_t>(sample.blockY)};

    CHECK_RET(CreateTensor(sparseHost, sparseShape, &sample.sparseDevice, &sample.sparse) == ACL_SUCCESS,
              return kCreateTensorFailed);
    CHECK_RET(CreateTensor(countHost, countShape, &sample.countDevice, &sample.count) == ACL_SUCCESS,
              return kCreateTensorFailed);
    CHECK_RET(CreateTensor(qHost, lengthShape, &sample.qDevice, &sample.qLengths) == ACL_SUCCESS,
              return kCreateTensorFailed);
    CHECK_RET(CreateTensor(kvHost, lengthShape, &sample.kvDevice, &sample.kvLengths) == ACL_SUCCESS,
              return kCreateTensorFailed);
    CHECK_RET(CreateTensor(seqQHost, seqShape, &sample.seqQDevice, &sample.seqQ) == ACL_SUCCESS,
              return kCreateTensorFailed);
    CHECK_RET(CreateTensor(seqKvHost, seqShape, &sample.seqKvDevice, &sample.seqKv) == ACL_SUCCESS,
              return kCreateTensorFailed);
    CHECK_RET(CreateTensor(sample.unionHost, unionShape, &sample.unionDevice, &sample.unionIdx) == ACL_SUCCESS,
              return kCreateTensorFailed);
    CHECK_RET(CreateTensor(sample.unionCountHost, unionCountShape, &sample.unionCountDevice, &sample.unionCount) ==
                  ACL_SUCCESS,
              return kCreateTensorFailed);
    CHECK_RET(CreateTensor(sample.maskHost, maskShape, &sample.maskDevice, &sample.exactMask) == ACL_SUCCESS,
              return kCreateTensorFailed);

    const int64_t shapeData[] = {1, sample.blockY};
    sample.blockShape = aclCreateIntArray(shapeData, kBlockShapeDimensions);
    CHECK_RET(sample.blockShape != nullptr, return kCreateTensorFailed);
    return ACL_SUCCESS;
}

int ExecuteSample(SampleData& sample, aclrtStream stream)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnUnionSparseGenGetWorkspaceSize(
        sample.sparse, sample.count, sample.qLengths, sample.kvLengths, sample.seqQ, sample.seqKv, sample.blockShape,
        kSampleUnionSize, kCausalMask, kUnspecifiedMaxSeqLen, kUnspecifiedMaxSeqLen, kSampleLayout, true,
        kExplicitBlocks, 1, 1, sample.unionIdx, sample.unionCount, sample.exactMask, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, return kWorkspaceQueryFailed);
    if (workspaceSize > 0) {
        CHECK_RET(aclrtMalloc(&sample.workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS,
                  return kWorkspaceAllocFailed);
    }
    CHECK_RET(aclnnUnionSparseGen(sample.workspace, workspaceSize, executor, stream) == ACL_SUCCESS,
              return kLaunchFailed);
    CHECK_RET(aclrtSynchronizeStream(stream) == ACL_SUCCESS, return kSynchronizeFailed);
    return ACL_SUCCESS;
}

int ReadSampleOutputs(SampleData& sample, bool& ok)
{
    CHECK_RET(aclrtMemcpy(sample.unionHost.data(), sample.unionHost.size() * sizeof(int32_t), sample.unionDevice,
                          sample.unionHost.size() * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS,
              return kReadbackFailed);
    CHECK_RET(aclrtMemcpy(sample.unionCountHost.data(), sample.unionCountHost.size() * sizeof(int32_t),
                          sample.unionCountDevice, sample.unionCountHost.size() * sizeof(int32_t),
                          ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS,
              return kReadbackFailed);
    CHECK_RET(aclrtMemcpy(sample.maskHost.data(), sample.maskHost.size() * sizeof(int32_t), sample.maskDevice,
                          sample.maskHost.size() * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS,
              return kReadbackFailed);
    printf("observed union_count=%d union_prefix=[%d] mask_prefix=[%d]\n", sample.unionCountHost[0],
           sample.unionHost[0], sample.maskHost[0]);

    ok = sample.unionCountHost[0] == 1 && sample.unionHost[0] == 0;
    for (int64_t i = 0; i < sample.blockY - 1; ++i) {
        ok = ok && sample.maskHost[static_cast<size_t>(i)] == (kFirstQueryBit | kSecondQueryBit);
    }
    ok = ok && sample.maskHost[static_cast<size_t>(sample.blockY - 1)] == kSecondQueryBit;

    return ACL_SUCCESS;
}

void ReleaseSample(SampleData& sample, aclrtStream stream)
{
    aclDestroyTensor(sample.sparse);
    aclDestroyTensor(sample.count);
    aclDestroyTensor(sample.qLengths);
    aclDestroyTensor(sample.kvLengths);
    aclDestroyTensor(sample.seqQ);
    aclDestroyTensor(sample.seqKv);
    aclDestroyTensor(sample.unionIdx);
    aclDestroyTensor(sample.unionCount);
    aclDestroyTensor(sample.exactMask);
    aclrtFree(sample.sparseDevice);
    aclrtFree(sample.countDevice);
    aclrtFree(sample.qDevice);
    aclrtFree(sample.kvDevice);
    aclrtFree(sample.seqQDevice);
    aclrtFree(sample.seqKvDevice);
    aclrtFree(sample.unionDevice);
    aclrtFree(sample.unionCountDevice);
    aclrtFree(sample.maskDevice);
    if (sample.workspace != nullptr) {
        aclrtFree(sample.workspace);
    }
    aclDestroyIntArray(sample.blockShape);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
}

int Run(int64_t blockY)
{
    aclrtStream stream = nullptr;
    CHECK_RET(aclInit(nullptr) == ACL_SUCCESS, return kInitFailed);
    CHECK_RET(aclrtSetDevice(0) == ACL_SUCCESS, return kSetDeviceFailed);
    CHECK_RET(aclrtCreateStream(&stream) == ACL_SUCCESS, return kCreateStreamFailed);

    SampleData sample(blockY);
    CHECK_RET(CreateSampleTensors(sample) == ACL_SUCCESS, return kCreateTensorFailed);
    const auto executeStatus = ExecuteSample(sample, stream);
    CHECK_RET(executeStatus == ACL_SUCCESS, return executeStatus);
    bool ok = false;
    const auto readStatus = ReadSampleOutputs(sample, ok);
    CHECK_RET(readStatus == ACL_SUCCESS, return readStatus);
    ReleaseSample(sample, stream);
    if (!ok) {
        printf("UnionSparseGen mismatch\n");
        return kOutputMismatch;
    }
    printf("UnionSparseGen OK: union_count=%d, union=[%d], mask_prefix=[%d]\n", sample.unionCountHost[0],
           sample.unionHost[0], sample.maskHost[0]);
    return 0;
}

} // namespace

int main() { return Run(kSampleBlockY); }
