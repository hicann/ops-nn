# Compilation and Running Examples

## Prerequisites

- If you need to compile and execute operator APIs, ensure that the basic environment has been set up, including driver, firmware, CANN software package, ops package, etc.
- For the operator API calling process and compilation and running operations, refer to [Application Development (C&C++)](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/programug/acldevg/aclcppdevg_000006.html) under "Single Operator Invocation > Single Operator API Execution > Calling aclnn Interface Example Code".

## Pre-compilation Preparation

This chapter takes the development and runtime environment co-location scenario as an example, that is, the machine with AI processor serves as both the development environment and the runtime environment. In this scenario, code development and code running are on the same machine. Here we take the **AddMatMul operator** as an example. The calling logic, process, and compilation script of other operators are roughly the same as the AddMatMul operator. Please modify the API calling script (*.cpp) and compilation script (CMakeLists) according to the actual situation.

- **Example Code**

   The AddMatMul operator implements tensor addition operation, and the calculation formula is: out = β * self + α * (mat1 @ mat2). You can obtain the example code from the "Calling Example" section in [aclnnAddmm&aclnnInplaceAddmm.md](../../../matmul/mat_mul_v3/docs/aclnnAddmm&aclnnInplaceAddmm.md) and name the code file "**test\_addmm.cpp**".

- **CMakeLists File**

    The CMake file example is as follows. Please modify according to the actual situation:

    ```bash
    # Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

    # CMake lowest version requirement
    cmake_minimum_required(VERSION 3.14)

    # Set project name
    project(ACLNN_EXAMPLE)

    # Compile options
    add_compile_options(-std=c++11)

    # Set compilation options
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "./bin")
    set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
    set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

    # Set executable file name (such as opapi_test) and specify the directory where the operator file *.cpp to be run is located
    add_executable(opapi_test
                   test_addmm.cpp)

    # Set ASCEND_PATH (CANN software package directory, please modify according to the actual path) and INCLUDE_BASE_DIR (header file directory)
    if(NOT "$ENV{ASCEND_CUSTOM_PATH}" STREQUAL "")
        set(ASCEND_PATH $ENV{ASCEND_CUSTOM_PATH})
    else()
        set(ASCEND_PATH "/usr/local/Ascend/cann")
    endif()
    set(INCLUDE_BASE_DIR "${ASCEND_PATH}/include")
    include_directories(
        ${INCLUDE_BASE_DIR}
        ${INCLUDE_BASE_DIR}/aclnn
    )

    # Set linked library file path
    target_link_libraries(opapi_test PRIVATE
                          ${ASCEND_PATH}/lib64/libascendcl.so
                          ${ASCEND_PATH}/lib64/libnnopbase.so
                          ${ASCEND_PATH}/lib64/libopapi_math.so
                          ${ASCEND_PATH}/lib64/libopapi_nn.so)

    # The executable file is in the bin directory under the CMakeLists file directory
    install(TARGETS opapi_test DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    ```

    For operators that combine collective communication and MatMul calculation, and run in parallel, they are collectively called MC2 operators (communication-computation fusion operators), including AllGatherMatmul, AlltoAllAllGatherBatchMatMul, BatchMatMulReduceScatterAlltoAll, MatmulAllReduce, MatmulAllReduceAddRmsNorm, MatmulReduceScatter, etc. When calling such operator APIs, multi-threading and HCCL (Huawei Collective Communication Library) are generally involved. Therefore, the CMake file needs to additionally import the following content, otherwise compilation will fail.

  ```bash
  # Set linked library file path
  find_package(Threads REQUIRED)
  target_link_libraries(opapi_test PRIVATE
                        ${ASCEND_PATH}/lib64/libascendcl.so
                        ${ASCEND_PATH}/lib64/libnnopbase.so
                        ${ASCEND_PATH}/lib64/libopapi_math.so
                        ${ASCEND_PATH}/lib64/libopapi_nn.so
                        ${ASCEND_PATH}/lib64/libhccl.so      # Collective communication library file
                        ${CMAKE_THREAD_LIBS_INIT})           # Library file that multi-threading depends on
  ```

  Where "find_package(Threads REQUIRED)" is a CMake command used to find the thread library, which can automatically link the header files or indirectly dependent library files that the thread library depends on.

## Compilation and Running

  1. Prepare the operator calling code (*.cpp) and compilation script (CMakeLists.txt) in advance.
  2. Configure environment variables.

     After installing the CANN software, log in to the environment as the CANN runtime user and execute the following command to make the environment variables effective.

        ```bash
        source ${INSTALL_DIR}/set_env.sh
        ```

     Where ${INSTALL_DIR} is the storage path after CANN software installation. Please replace according to the actual situation.
  3. Compile and run.
        - Enter the directory where CMakeLists.txt is located and execute the following command to create a new build directory to store the generated compilation files.

            ```bash
            mkdir -p build
            ```

        - Enter the build directory, execute the cmake command to compile, and then execute the make command to generate the executable file.

          ```bash
          cd build
          cmake ../ -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE
          make
          ```

          After successful compilation, the opapi\_test executable file will be generated in the bin folder under the build directory.

        - Enter the bin directory and run the executable file opapi_test.

          ```bash
          cd bin
          ./opapi_test
          ```

          Taking the running result of the AddMatMul operator as an example, the result after running is shown below:

          ```bash
          result[0] is: 1.200000
          result[1] is: 2.200000
          result[2] is: 3.200000
          result[3] is: 5.400000
          result[4] is: 6.400000
          result[5] is: 7.400000
          result[6] is: 9.600000
          result[7] is: 10.600000
          ```

          If the execution result reports an error and the expected result does not appear, you can use the aclGetRecentErrMsg interface to obtain the specific error information.
          Example of obtaining exception information when calling aclnnAddmmGetWorkspaceSize fails:

          ```bash
          // self is nullptr
          ret = aclnnAddmmGetWorkspaceSize(self, mat1, mat2, beta, alpha, out, cubeMathType, &workspaceSize, &executor);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddmmGetWorkspaceSize failed. ERROR: %d\n[ERROR msg]%s", ret, aclGetRecentErrMsg()); return ret);
          ```

          The above null pointer construction problem obtains error information as shown below:

          ```bash
          aclnnAddmmGetWorkspaceSize failed. ERROR: 161001
          [ERROR msg][PID:xxxx] xxx(timesamp) AclNN_Parameter_Error(EZ1001): Expected a proper Tensor but got null for argument addmmTennsor.self.
          ```
