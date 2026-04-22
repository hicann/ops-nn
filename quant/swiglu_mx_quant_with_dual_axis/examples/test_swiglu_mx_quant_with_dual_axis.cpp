#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_swiglu_mx_quant_with_dual_axis.h"

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
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // 固定写法，资源初始化
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // 调用aclrtMalloc申请device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // 计算连续tensor的strides
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // 调用aclCreateTensor接口创建aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. （固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  // 输入 x shape: [M, 2N] = [64, 256]
  std::vector<int64_t> xShape = {64, 256};
  // group_index shape: [G] = [1]，采用 cumsum 模式，值为 64 表示一个 group 包含所有 64 行
  std::vector<int64_t> groupIndexShape = {1};
  // SwiGLU 输出 shape: [M, N] = [64, 128]
  std::vector<int64_t> y1OutShape = {64, 128};
  std::vector<int64_t> y2OutShape = {64, 128};
  // mxscale1 shape: [M, (ceil(N/32)+2-1)/2, 2] = [64, 2, 2]
  std::vector<int64_t> mxscale1OutShape = {64, 2, 2};
  // mxscale2 shape: [floor(M/64) + G, N, 2] = [1 + 1, 128, 2] = [2, 128, 2]
  std::vector<int64_t> mxscale2OutShape = {2, 128, 2};

  void* xDeviceAddr = nullptr;
  void* groupIndexDeviceAddr = nullptr;
  void* y1OutDeviceAddr = nullptr;
  void* mxscale1OutDeviceAddr = nullptr;
  void* y2OutDeviceAddr = nullptr;
  void* mxscale2OutDeviceAddr = nullptr;

  aclTensor* x = nullptr;
  aclTensor* groupIndex = nullptr;
  aclTensor* y1Out = nullptr;
  aclTensor* mxscale1Out = nullptr;
  aclTensor* y2Out = nullptr;
  aclTensor* mxscale2Out = nullptr;

  // 输入数据初始化（BF16）
  std::vector<uint16_t> xHostData(64 * 256, 0);
  for (int64_t i = 0; i < 64 * 256; i++) {
    xHostData[i] = static_cast<uint16_t>(i % 100);
  }
  // group_index 数据：cumsum 模式，值为 64 表示只有一个 group，包含所有行
  std::vector<int64_t> groupIndexHostData = {64};
  std::vector<uint8_t> y1OutHostData(64 * 128, 0);
  std::vector<uint8_t> y2OutHostData(64 * 128, 0);
  std::vector<uint8_t> mxscale1OutHostData(64 * 2 * 2, 0);
  std::vector<uint8_t> mxscale2OutHostData(2 * 128 * 2, 0);

  bool activateLeft = true;
  int64_t dstType = 36;  // FLOAT8_E4M3FN
  int64_t scaleAlg = 0;  // OCP
  double maxDtypeValue = 0.0;

  // 创建 x aclTensor
  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建 groupIndex aclTensor
  ret = CreateAclTensor(groupIndexHostData, groupIndexShape, &groupIndexDeviceAddr, aclDataType::ACL_INT64, &groupIndex);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建 y1Out aclTensor
  ret = CreateAclTensor(y1OutHostData, y1OutShape, &y1OutDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &y1Out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建 mxscale1Out aclTensor
  ret = CreateAclTensor(mxscale1OutHostData, mxscale1OutShape, &mxscale1OutDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &mxscale1Out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建 y2Out aclTensor
  ret = CreateAclTensor(y2OutHostData, y2OutShape, &y2OutDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &y2Out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建 mxscale2Out aclTensor
  ret = CreateAclTensor(mxscale2OutHostData, mxscale2OutShape, &mxscale2OutDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &mxscale2Out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnSwigluMxQuantWithDualAxis第一段接口（带 groupIndex 输入）
  ret = aclnnSwigluMxQuantWithDualAxisGetWorkspaceSize(x, groupIndex, activateLeft, "rint",
      dstType, scaleAlg, maxDtypeValue, y1Out, mxscale1Out, y2Out, mxscale2Out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwigluMxQuantWithDualAxisGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnSwigluMxQuantWithDualAxis第二段接口
  ret = aclnnSwigluMxQuantWithDualAxis(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwigluMxQuantWithDualAxis failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(y1OutShape);
  std::vector<uint8_t> y1ResultData(size, 0);
  ret = aclrtMemcpy(y1ResultData.data(), y1ResultData.size() * sizeof(y1ResultData[0]), y1OutDeviceAddr, size * sizeof(y1ResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy y1Out result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < 10 && i < size; i++) {
    LOG_PRINT("y1Out[%ld] is: %d\n", i, y1ResultData[i]);
  }

  size = GetShapeSize(y2OutShape);
  std::vector<uint8_t> y2ResultData(size, 0);
  ret = aclrtMemcpy(y2ResultData.data(), y2ResultData.size() * sizeof(y2ResultData[0]), y2OutDeviceAddr, size * sizeof(y2ResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy y2Out result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < 10 && i < size; i++) {
    LOG_PRINT("y2Out[%ld] is: %d\n", i, y2ResultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(x);
  aclDestroyTensor(groupIndex);
  aclDestroyTensor(y1Out);
  aclDestroyTensor(mxscale1Out);
  aclDestroyTensor(y2Out);
  aclDestroyTensor(mxscale2Out);
  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(xDeviceAddr);
  aclrtFree(groupIndexDeviceAddr);
  aclrtFree(y1OutDeviceAddr);
  aclrtFree(mxscale1OutDeviceAddr);
  aclrtFree(y2OutDeviceAddr);
  aclrtFree(mxscale2OutDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}