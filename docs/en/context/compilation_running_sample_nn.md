# Compilation and Running Sample

## Prerequisites

- To compile and execute operator APIs, ensure that the basic environment has been set up, including the driver, firmware, CANN software package, and ops package.
- For details about the operator API call process and compilation and running operations, see "Single-Operator Calling > Single-Operator API Execution > Sample Code for Calling aclnn APIs" in [Application Development Guide (C&C++)](https://www.hiascend.com/document/detail/zh/canncommercial/850/appdevg/acldevg/aclcppdevg_000000.html).

## Preparations

Assume that the development environment and operating environment are deployed on the same server (equipped with the AI processor). In this scenario, code development and code running are performed on the same machine. The following uses the AddMatMul operator as an example. The calling logic, process, and compilation script of other operators are similar to those of the AddMatMul operator. You should modify the API calling script (**\*.cpp**) and compilation script (**CMakeLists**) as required.

- **Sample code**

   The AddMatMul operator implements the tensor addition operation. The calculation formula is as follows: out = β  self + α  (mat1 @ mat2). You can obtain the sample code from "Example" in [aclnnAddmm&aclnnInplaceAddmm](../../../matmul/batch_mat_mul_v3/docs/aclnnAddbmm&aclnnInplaceAddbmm_en.md) and name the code file ****test\_addmm.cpp****.

- **CMakeLists file**

    The following is an example of the CMake file, which should be modified as required.
    ```
    # Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

    # CMake lowest version requirement
    cmake_minimum_required(VERSION 3.14)

    # Set the project name.
    project(ACLNN_EXAMPLE)

    # Compile options
    add_compile_options(-std=c++11)

    # Set compilation options.
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "./bin")
    set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
    set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

    # Set the executable file name (for example, opapi_test) and specify the directory where the operator file (*.cpp) is stored.
    add_executable(opapi_test
                   test_addmm.cpp)

    # Set ASCEND_PATH (CANN package path, which should be replaced with the actual path) and INCLUDE_BASE_DIR (header file directory).
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
    
    # Set the link library file paths.
    target_link_libraries(opapi_test PRIVATE
                          ${ASCEND_PATH}/lib64/libascendcl.so
                          ${ASCEND_PATH}/lib64/libnnopbase.so
                          ${ASCEND_PATH}/lib64/libopapi_math.so
                          ${ASCEND_PATH}/lib64/libopapi_nn.so)

    # The executable file is in the bin folder of the directory where the CMakeLists file is located.
    install(TARGETS opapi_test DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    ```
    Operators with integrated and parallel collective communication and MatMul computation are collectively referred to as merged compute and communication operators (abbreviated as MC2 operators), including AllGatherMatmul, AlltoAllAllGatherBatchMatMul, BatchMatMulReduceScatterAlltoAll, MatmulAllReduce, MatmulAllReduceAddRmsNorm, and MatmulReduceScatter. When these operator APIs are called, multi-threading and Huawei Collective Communication Library (HCCL) are generally involved. Therefore, the following content needs to be imported to the CMake file to ensure successful compilation:
  
  ```
  # Set the link library file paths.
  find_package(Threads REQUIRED)
  target_link_libraries(opapi_test PRIVATE
                        ${ASCEND_PATH}/lib64/libascendcl.so
                        ${ASCEND_PATH}/lib64/libnnopbase.so
                        ${ASCEND_PATH}/lib64/libopapi_math.so
                        ${ASCEND_PATH}/lib64/libopapi_nn.so
                        ${ASCEND_PATH}/lib64/libhccl.so      # Collective communication library file
                        ${CMAKE_THREAD_LIBS_INIT})           # Library file on which multi-threading depends
  ```
  You can use the **find_package(Threads REQUIRED)** of CMake to search for the thread library. The command can automatically link the header files on which the thread library depends or the library files on which the thread library indirectly depends.
  
## Compilation and Running

  1. Prepare the calling code (**\*.cpp**) and compilation script (**CMakeLists.txt**) of the operator.
  2. Set environment variables.

     After installing the CANN software, log in to the environment as the CANN running user and run the following command to make the environment variables take effect:

        ```
        source ${INSTALL_DIR}/set_env.sh
        ```

     **${INSTALL_DIR}** indicates the CANN component directory, which should be replaced with the actual directory.

   3. Compile and run the script.
      - Go to the directory where **CMakeLists.txt** is stored and run the following command to create the **build** folder to store the generated compilation file.
          
          ```
          mkdir -p build 
          ```
        
      - Go to the **build** directory, run the **cmake** command to compile the code, and then run the **make** command to generate an executable file.
          
          ```
          cd build
          cmake ../ -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE
          make
          ```

          After the compilation is successful, the **opapi\_test** executable file is generated in the **bin** folder in the **build** directory.
          
      - Go to the **bin** directory and run the executable file **opapi_test**.
        
        ```
        cd bin
        ./opapi_test
        ```
        
        The following uses the running result of the AddMatMul operator as an example:
        
        ```
        result[0] is: 1.200000
        result[1] is: 2.200000
        result[2] is: 3.200000
        result[3] is: 5.400000
        result[4] is: 6.400000
        result[5] is: 7.400000
        result[6] is: 9.600000
        result[7] is: 10.600000
        ```
