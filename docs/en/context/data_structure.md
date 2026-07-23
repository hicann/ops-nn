# Data Structures

This chapter provides the basic data structures required for calling CANN operator APIs. **Developers do not need to focus on their internal implementation and can use them directly.**

Note that these basic data structures can be created through the "Public Interfaces" section in [Operator Library Interface](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/API/aolapi/operatorlist_00001.html), such as aclCreateTensor.

- **aclTensor**

  A structure defined by the framework to manage and store tensor data (such as multi-dimensional data like vectors and matrices). You can create this object through the **aclCreateTensor** interface.

  ```bash
  typedef struct aclTensor aclTensor
  ```

- **aclScalar**

  A structure defined by the framework to manage and store scalar data (that is, a single value). You can create this object through the **aclCreateScalar** interface.

  ```bash
  typedef struct aclScalar aclScalar
  ```

- **aclIntArray**

  An array structure defined by the framework to manage and store integer data. You can create this object through the **aclCreateIntArray** interface.

  ```bash
  typedef struct aclIntArray aclIntArray
  ```

- **aclFloatArray**

  An array structure defined by the framework to manage and store float32 data. You can create this object through the **aclCreateFloatArray** interface.

  ```bash
  typedef struct aclFloatArray aclFloatArray
  ```

- **aclBoolArray**

  An array structure defined by the framework to manage and store boolean data. You can create this object through the **aclCreateBoolArray** interface.

  ```bash
  typedef struct aclBoolArray aclBoolArray
  ```

- **aclTensorList**

  An array structure defined by the framework to manage and store multiple tensor data. You can create this object through the **aclCreateTensorList** interface.

  ```bash
  typedef struct aclTensorList aclTensorList
  ```

- **aclScalarList**

  An array structure defined by the framework to manage and store scalar data. You can create this object through the **aclCreateScalarList** interface.

  ```bash
  typedef struct aclScalarList aclScalarList
  ```

- **aclOpExecutor**

  An executor data structure defined by the framework, which is a container used to execute operator calculations.

  Typically, when calling the first-stage interface aclxxXxxGetWorkspaceSize, the framework automatically creates an aclOpExecutor; after calling the second-stage interface aclxxXxx, the object is automatically released.

  ```bash
  typedef struct aclOpExecutor aclOpExecutor
  ```

- **aclrtStream**

  A stream processing data structure defined by the framework, used to manage and maintain the execution order of some asynchronous operations.

  ```bash
  typedef void *aclrtStream
  ```
