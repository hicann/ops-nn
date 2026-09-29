# aclnnTransSparse4to2Para

## Supported Products

|Product            |  Supported |
|:-------------------------|:----------:|
|  <term>Atlas A3 training products/Atlas A3 inference products</term>  |     √    |
|  <term>Atlas A2 training products/Atlas A2 inference products</term>    |     √    |

## Function

- Description:

  Performs preprocessing on the weight matrix of structured sparsity and outputs the compressed sparse matrix and the corresponding index matrix.
  If there are at least two zeros in every four elements of the original sparse matrix, the compressed sparse matrix is a matrix in which two zeros are filtered out from every four elements. During matrix compression, an index matrix is generated. For every four elements in the original matrix, two 2-bit indexes are generated in the index matrix and encoded according to the rules.

## Prototype

```Cpp
aclnnStatus aclnnTransSparse4to2Para(
    const int8_t* weight, 
    aclIntArray*  shape, 
    int8_t**      sparseWeight, 
    int64_t**     sparseWeightDims,
    uint64_t*     sparseWeightDimsNum, 
    uint8_t**     index, 
    int64_t**     indexDims, 
    uint64_t*     indexDimsNum)
```

## aclnnTransSparse4to2Para

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1505px"><colgroup>
  <col style="width: 150px">
  <col style="width: 120px">
  <col style="width: 350px">
  <col style="width: 360px">
  <col style="width: 130px">
  <col style="width: 120px">
  <col style="width: 130px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Usage</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (Shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
   <tbody>
      <tr>
        <td>weight</td>
        <td>Input</td>
        <td>Uncompressed sparse right matrix in the matrix multiplication operation.</td>
        <td><ul><li>The shape can be expressed as (n, k). </li><li>At least two zeros are required for every four elements.</li></ul></td>
        <td>INT8</td>
        <td>ND</td>
        <td>2</td>
        <td>×</td>
      </tr>
      <tr>
        <td>shape</td>
        <td>Input</td>
        <td>Shape of the uncompressed sparse right matrix in the matrix multiplication operation.</td>
        <td><ul><li>It is created by using the aclCreateIntArray API.</li></ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>sparseWeight</td>
        <td>Output</td>
        <td>Compressed right matrix in the matrix multiplication operation.</td>
        <td><ul><li>The memory is released by the caller.</li></ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>sparseWeightDims</td>
        <td>Output</td>
        <td>Pointer to the start address of the StorageShape array of the compressed right matrix in the matrix multiplication operation.</td>
        <td><ul><li>The memory is released by the caller. </li><li>StorageShape can be expressed as (ceil(k_half / 32), ceil(n / 16), 16, 32), where k_half = ceil(k / 8) * 8 / 2.</li></ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>sparseWeightDimsNum</td>
        <td>Output</td>
        <td>Dimension of the StorageShape array of the compressed right matrix in the matrix multiplication operation.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>index</td>
        <td>Output</td>
        <td>Index matrix corresponding to the compressed right matrix in the matrix multiplication operation.</td>
        <td><ul><li>The memory is released by the caller.</li></ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>indexDims</td>
        <td>Output</td>
        <td>Pointer address of the first element in the StorageShape array, which is the index matrix corresponding to the compressed right matrix in the matrix multiplication operation.</td>
        <td><ul><li>The memory is released by the caller. </li><li>StorageShape can be expressed as (ceil(k_half / 32), ceil(n / 16), 16, 8), where k_half = ceil(k / 8) * 8 / 2.</li></ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>indexDimsNum</td>
        <td>Output</td>
        <td>Dimension of the StorageShape array, which is the index matrix corresponding to the compressed right matrix in the matrix multiplication operation.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
  </tbody>
  </table>

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter validation. The following error codes may be returned.
    <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
    <col style="width: 291px">
    <col style="width: 135px">
    <col style="width: 723px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Code</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The input weight is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="8">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="8">161002</td>
      <td>The shape, format, and data type of weight do not meet the requirements. The data does not meet the requirement that at least two of every four elements are zeros.</td>
    </tr>
  </tbody></table>

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

- <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>:

  ```Cpp
  #include <iostream>
  #include <memory>
  #include <vector>
  #include <stdlib.h>

  #include "acl/acl.h"
  #include "aclnnop/aclnn_sparse4to2quant_matmul_weight_nz.h"

  #define CHECK_RET(cond, return_expr) \
      do {                             \
          if (!(cond)) {               \
              return_expr;             \
          }                            \
      } while (0)

  #define CHECK_FREE_RET(cond, return_expr) \
      do {                                  \
          if (!(cond)) {                    \
              Finalize(deviceId, stream);   \
              return_expr;                  \
          }                                 \
      } while (0)

  #define LOG_PRINT(message, ...)         \
      do {                                \
          printf(message, ##__VA_ARGS__); \
      } while (0)

  #define CREATE_TENSOR(hostData, shape, deviceAddr, dtype, tensor)                                        \
      ret = CreateAclTensor(hostData, shape, &deviceAddr, dtype, &tensor);                                 \
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> tensor##Ptr(tensor, aclDestroyTensor); \
      std::unique_ptr<void, aclError (*)(void*)> deviceAddr##Ptr(deviceAddr, aclrtFree);                  \
      CHECK_RET(ret == ACL_SUCCESS, return ret)

  #define CREATE_SPARSE_TENSOR(hostData, weightShape, storageShape, deviceAddr, dataType, tensor)          \
      ret = CreateSparseTensor(hostData, weightShape, storageShape, &deviceAddr, dataType, &tensor);       \
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> tensor##Ptr(tensor, aclDestroyTensor); \
      std::unique_ptr<void, aclError (*)(void*)> deviceAddr##Ptr(deviceAddr, aclrtFree);                  \
      CHECK_RET(ret == ACL_SUCCESS, return ret)

  int64_t GetShapeSize(const std::vector<int64_t>& shape)
  {
      int64_t shapeSize = 1;
      for (auto i : shape) {
          shapeSize *= i;
      }
      return shapeSize;
  }

  int Init(int32_t deviceId, aclrtStream* stream)
  {
      // (Fixed writing) Initialize resources.
      auto ret = aclInit(nullptr);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
      ret = aclrtSetDevice(deviceId);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
      ret = aclrtCreateStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
      return 0;
  }

  template <typename T>
  int CreateSparseTensor(
      const T* sparseWeightData, const std::vector<int64_t>& viewShape, const std::vector<int64_t>& storageShape,
      void** deviceAddr, aclDataType dataType, aclTensor** tensor)
  {
      auto size = static_cast<uint64_t>(GetShapeSize(storageShape)) * sizeof(T);

      // Call aclrtMalloc to allocate memory on the device.
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call aclrtMemcpy to copy the data on the host to the memory on the device. 
      ret = aclrtMemcpy(*deviceAddr, size, sparseWeightData, size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // Compute the strides of the contiguous tensor.
      std::vector<int64_t> strides(viewShape.size(), 1);
      for (int64_t i = viewShape.size() - 2; i >= 0; i--) {
          strides[i] = viewShape[i + 1] * strides[i + 1];
      }

      // Call aclCreateTensor to create an aclTensor.
      *tensor = aclCreateTensor(
          viewShape.data(), viewShape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, storageShape.data(),
          storageShape.size(), *deviceAddr);
      return 0;
  }

  template <typename T>
  int CreateAclTensor(
      const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
      aclTensor** tensor)
  {
      auto size = GetShapeSize(shape) * sizeof(T);
      // Call aclrtMalloc to allocate memory on the device.
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call aclrtMemcpy to copy the data on the host to the memory on the device. 
      if (hostData.size() > 0) {
          ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
      }

      // Compute the strides of the contiguous tensor.
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      // Call aclCreateTensor to create an aclTensor.
      *tensor = aclCreateTensor(
          shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
          *deviceAddr);
      return 0;
  }

  void Finalize(int32_t deviceId, aclrtStream stream)
  {
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
  }

  void GenRandomMask(std::vector<size_t>& masks)
  {
      masks[0] = random() % 4;
      masks[1] = random() % 4;
      while (masks[1] == masks[0]) {
          masks[1] = random() % 4;
      }
  }

  void GenRandomSparseData(std::vector<int8_t>& weightHostData)
  {
      srandom(233U);
      std::vector<size_t> masks(2, 0UL);
      constexpr size_t step = 4UL;
      for (size_t i = 0; i < weightHostData.size(); i += step) {
          GenRandomMask(masks);
          for (auto mask : masks) {
              weightHostData[i + mask] = 0;
          }
      }
  }

  std::vector<int64_t> GenStorageShape(int64_t* dims, uint64_t dimsNum)
  {
      std::vector<int64_t> storageShape;
      for (uint64_t i = 0UL; i < dimsNum; i++) {
          storageShape.push_back(dims[i]);
      }
      return storageShape;
  }

  int aclnnSparse4to2QuantMatmulTest(int32_t deviceId, aclrtStream& stream)
  {
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. Construct the input and output according to the API.
      int64_t m = 64L;
      int64_t k = 512L;
      int64_t n = 128L;
      std::vector<int64_t> xShape = {m, k};
      std::vector<int64_t> weightShape = {n, k};
      std::vector<int64_t> indexShape = {n, (k + 7) / 8};
      std::vector<int64_t> biasShape = {n};
      std::vector<int64_t> xScaleShape = {m};
      std::vector<int64_t> weightScaleShape = {n};
      std::vector<int64_t> outShape = {m, n};
      void* xDeviceAddr = nullptr;
      void* sparseWeightDeviceAddr = nullptr;
      void* indexDeviceAddr = nullptr;
      void* biasDeviceAddr = nullptr;
      void* xScaleDeviceAddr = nullptr;
      void* weightScaleDeviceAddr = nullptr;
      void* outDeviceAddr = nullptr;
      aclTensor* x = nullptr;
      aclTensor* sparseWeight = nullptr;
      aclTensor* index = nullptr;
      aclTensor* bias = nullptr;
      aclTensor* xScale = nullptr;
      aclTensor* weightScale = nullptr;
      aclTensor* out = nullptr;
      std::vector<int8_t> xHostData(GetShapeSize(xShape), 1);
      std::vector<int8_t> weightHostData(GetShapeSize(weightShape), 1);
      std::vector<uint16_t> biasHostData(GetShapeSize(biasShape), 1); // is actually the bfloat16 half-precision mode.
      std::vector<float> xScaleHostData(GetShapeSize(xScaleShape), 1);
      std::vector<float> weightScaleHostData(GetShapeSize(weightScaleShape), 1);
      GenRandomSparseData(weightHostData);

      int8_t* sparseWeightHostData = nullptr;
      uint8_t* indexHostData = nullptr;
      int64_t* sparseWeightDims = nullptr;
      uint64_t sparseWeightDimsNum = 0UL;
      int64_t* indexDims = nullptr;
      uint64_t indexDimsNum = 0UL;
      aclIntArray* weightShapeArray = aclCreateIntArray(weightShape.data(), weightShape.size());
      ret = aclnnTransSparse4to2Para(
          weightHostData.data(), weightShapeArray, &sparseWeightHostData, &sparseWeightDims, &sparseWeightDimsNum,
          &indexHostData, &indexDims, &indexDimsNum);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransSparse4to2Para failed. ERROR: %d\n", ret); return ret);
      std::unique_ptr<int8_t[]> sparseWeightHostDataPtr(sparseWeightHostData);
      std::unique_ptr<uint8_t[]> indexHostDataPtr(indexHostData);
      std::unique_ptr<int64_t[]> sparseWeightDimsPtr(sparseWeightDims);
      std::unique_ptr<int64_t[]> indexDimsPtr(indexDims);

      CREATE_TENSOR(xHostData, xShape, xDeviceAddr, aclDataType::ACL_INT8, x);

      weightShape.back() = (weightShape.back() + 7) / 8 * 8 / 2; // The K axis is aligned to the nearest multiple of 8 after the 2 out of 4 selection, and then halved.
      auto sparseWeightStorageShape = GenStorageShape(sparseWeightDims, sparseWeightDimsNum);
      CREATE_SPARSE_TENSOR(
          sparseWeightHostData, weightShape, sparseWeightStorageShape, sparseWeightDeviceAddr, aclDataType::ACL_INT8,
          sparseWeight);

      auto indexStorageShape = GenStorageShape(indexDims, indexDimsNum);
      CREATE_SPARSE_TENSOR(indexHostData, indexShape, indexStorageShape, indexDeviceAddr, aclDataType::ACL_UINT8, index);
      CREATE_TENSOR(biasHostData, biasShape, biasDeviceAddr, aclDataType::ACL_BF16, bias);
      CREATE_TENSOR(xScaleHostData, xScaleShape, xScaleDeviceAddr, aclDataType::ACL_FLOAT, xScale);
      CREATE_TENSOR(weightScaleHostData, weightScaleShape, weightScaleDeviceAddr, aclDataType::ACL_FLOAT, weightScale);
      CREATE_TENSOR(std::vector<uint16_t>(), outShape, outDeviceAddr, aclDataType::ACL_BF16, out);

      uint64_t workspaceSize = 0;
      aclOpExecutor* executor;
      void* workspaceAddr = nullptr;

      // Call the first segment of the aclnnSparse4to2QuantMatmul API.
      ret = aclnnSparse4to2QuantMatmulWeightNzGetWorkspaceSize(
          x, sparseWeight, index, xScale, weightScale, bias, out, &workspaceSize, &executor);

      CHECK_RET(
          ret == ACL_SUCCESS, LOG_PRINT("aclnnSparse4to2QuantMatmulWeightNzGetWorkspaceSize failed. ERROR: %d\n", ret);
          return ret);
      // Allocate device memory based on workspaceSize computed by the first-phase API.
      std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtrTrans(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtrTrans.reset(workspaceAddr);
      }
      // Call the second phase of the aclnnSparse4to2QuantMatmul API.
      ret = aclnnSparse4to2QuantMatmulWeightNz(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSparse4to2QuantMatmulWeightNz failed. ERROR: %d\n", ret); return ret);

      // 4. (Fixed writing) Wait until the task execution is complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
      auto size = GetShapeSize(outShape);
      // In C language, the bf16 data cannot be directly printed. You need to read the data using uint16 and convert it to bf16 in binary mode.
      std::vector<uint16_t> resultData(size, 0);
      ret = aclrtMemcpy(
          resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]),
          ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("result[%ld] is: %u\n", i, resultData[i]);
      }
      return ACL_SUCCESS;
  }

  int main()
  {
      // 1. (fixed) Initialize the device or stream. For details, see the ACL API manual.
      // Set the device ID in use.
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = aclnnSparse4to2QuantMatmulTest(deviceId, stream);
      CHECK_FREE_RET(
          ret == ACL_SUCCESS, LOG_PRINT("aclnnSparse4to2QuantMatmulTest failed. ERROR: %d\n", ret); return ret);

      Finalize(deviceId, stream);
      return 0;
  }
