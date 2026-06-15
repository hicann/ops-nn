#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_swiglu_group_quant.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto dim : shape) {
    shapeSize *= dim;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return ACL_SUCCESS;
}

bool CheckHardwareSupport() {
  const char* socName = aclrtGetSocName();
  if (socName == nullptr) {
    LOG_PRINT("Warning: Cannot get SOC name, skip hardware check\n");
    return true;
  }

  LOG_PRINT("Current SOC: %s\n", socName);
  if (strstr(socName, "Ascend950") != nullptr || strstr(socName, "ascend950") != nullptr) {
    return true;
  }

  LOG_PRINT("Warning: SwigluGroupQuant only supports Ascend950, current SOC '%s' is not supported. Skip test.\n",
            socName);
  return false;
}

void Finalize(int32_t deviceId, aclrtStream stream) {
  (void)aclrtDestroyStream(stream);
  (void)aclrtResetDevice(deviceId);
  (void)aclFinalize();
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return ACL_SUCCESS;
}

int main() {
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  if (!CheckHardwareSupport()) {
    LOG_PRINT("\n=== Test SKIPPED (hardware not supported) ===\n");
    Finalize(deviceId, stream);
    return ACL_SUCCESS;
  }

  std::vector<int64_t> xShape = {2, 256};
  std::vector<int64_t> yShape = {2, 128};
  std::vector<int64_t> scaleOutShape = {2, 1};
  std::vector<int64_t> yOriginShape = {2, 128};

  std::vector<uint16_t> xHostData(GetShapeSize(xShape), 0);
  for (size_t i = 0; i < xHostData.size(); ++i) {
    xHostData[i] = static_cast<uint16_t>(i % 23);
  }
  std::vector<uint8_t> yHostData(GetShapeSize(yShape), 0);
  std::vector<float> scaleOutHostData(GetShapeSize(scaleOutShape), 0.0f);
  std::vector<uint16_t> yOriginHostData(GetShapeSize(yOriginShape), 0);

  void* xDeviceAddr = nullptr;
  void* yDeviceAddr = nullptr;
  void* scaleOutDeviceAddr = nullptr;
  void* yOriginDeviceAddr = nullptr;
  aclTensor* x = nullptr;
  aclTensor* y = nullptr;
  aclTensor* scaleOut = nullptr;
  aclTensor* yOrigin = nullptr;

  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, ACL_FLOAT16, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, ACL_FLOAT8_E4M3FN, &y);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(scaleOutHostData, scaleOutShape, &scaleOutDeviceAddr, ACL_FLOAT, &scaleOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(yOriginHostData, yOriginShape, &yOriginDeviceAddr, ACL_FLOAT16, &yOrigin);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  int64_t dstType = 36;
  int64_t quantMode = 0;
  int64_t blockSize = 0;
  bool roundScale = false;
  double clampLimit = -1.0;
  bool outputOrigin = false;

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  ret = aclnnSwigluGroupQuantGetWorkspaceSize(x, nullptr, nullptr, dstType, quantMode, blockSize, roundScale,
                                              clampLimit, outputOrigin, y, scaleOut, yOrigin, &workspaceSize,
                                              &executor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnSwigluGroupQuantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  ret = aclnnSwigluGroupQuant(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwigluGroupQuant failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  std::vector<uint8_t> resultData(GetShapeSize(yShape), 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), yDeviceAddr,
                    resultData.size() * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  LOG_PRINT("result[0] is: %d\n", resultData[0]);

  aclDestroyTensor(x);
  aclDestroyTensor(y);
  aclDestroyTensor(scaleOut);
  aclDestroyTensor(yOrigin);
  aclrtFree(xDeviceAddr);
  aclrtFree(yDeviceAddr);
  aclrtFree(scaleOutDeviceAddr);
  aclrtFree(yOriginDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  Finalize(deviceId, stream);
  return ACL_SUCCESS;
}
