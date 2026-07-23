# Security Statement

## Running User Recommendations

Based on security considerations, we do not recommend using root or other administrator type accounts to execute any commands. Follow the principle of minimum permissions.

## File Permission Control

- We recommend that users set the running system umask value to 0027 or above on the host machine (including the host machine) and in the container to ensure that the default maximum permission for new folders is 750 and the default maximum permission for new files is 640.
- We recommend that users take security measures such as permission control for sensitive content such as personal privacy data, business assets, source files, and various files saved during operator development. For example, for project installation directory permission control and input public data file permission control, the set permissions should refer to [A-File (Folder) Permission Control Recommended Maximum Values in Various Scenarios](#a-file-folder-permission-control-recommended-maximum-values-in-various-scenarios).
- When the operator runs, it may cache operator compilation files, which are stored in the `kernel_meta_*` folder under the running directory to speed up subsequent operator invocation. Users can perform permission control on the generated related files as needed.
- Users need to perform permission control during installation and use. We recommend referring to [A-File (Folder) Permission Control Recommended Maximum Values in Various Scenarios](#a-file-folder-permission-control-recommended-maximum-values-in-various-scenarios) for file permission reference settings.

## Build Security Statement

When compiling and installing this project from source code, you need to compile it yourself. During the compilation process, some intermediate files will be generated. We recommend that you perform permission control on the intermediate files after compilation to ensure file security.

## Running Security Statement

- We recommend that users write corresponding operator invocation scripts based on the running environment resource status. If the operator invocation script does not match the resource status, such as the space used for generating input data or benchmark calculation results exceeding the memory capacity limit, or the script saving data locally exceeding the disk space size, it may cause errors and lead to unexpected process exit.
- When the operator runs abnormally, it will exit the process and print error information. We recommend locating the specific error cause based on the error prompt, including setting operator synchronous execution, viewing log files, and other methods.
- When the operator is invoked through [PyTorch](https://gitee.com/ascend/pytorch), running errors may occur due to version mismatch. For details, please refer to [PyTorch Security Statement](https://gitee.com/ascend/pytorch#%E5%AE%89%E5%85%A8%E5%A3%B0%E6%98%8E).

## Public Network Address Statement

The public network addresses contained in this project code are declared as follows:

|      Type      |                                           Open Source Code Address                                           |                            File Name                             |             Public Network IP Address/Public Network URL Address/Domain Name/Email Address/Compressed File Address             |                   Usage Description                    |
| :------------: |:------------------------------------------------------------------------------------------:|:----------------------------------------------------------| :---------------------------------------------------------- |:-----------------------------------------|
|  Dependency  | Not involved  | cmake/third_party/makeself-fetch.cmake | [https://gitcode.com/cann-src-third-party/makeself/releases/download/release-2.5.0-patch1.0/makeself-release-2.5.0-patch1.tar.gz](https://gitcode.com/cann-src-third-party/makeself/releases/download/release-2.5.0-patch1.0/makeself-release-2.5.0-patch1.tar.gz) | Download makeself source code from gitcode, used as compilation dependency |
|  Dependency  | Not involved  | cmake/third_party/nlohmann_json.cmake | [https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/include.zip](https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/include.zip) | Download json source code from gitcode, used as compilation dependency |
|  Dependency  | Not involved  | cmake/third_party/gtest.cmake | [https://gitcode.com/cann-src-third-party/googletest/releases/download/v1.14.0/googletest-1.14.0.tar.gz](https://gitcode.com/cann-src-third-party/googletest/releases/download/v1.14.0/googletest-1.14.0.tar.gz) | Download googletest source code from gitcode, used as compilation dependency |
|  Dependency  | Not involved  | cmake/third_party/eigen.cmake | [https://gitcode.com/cann-src-third-party/eigen/releases/download/5.0.0-h0.trunk/eigen-5.0.0.tar.gz](https://gitcode.com/cann-src-third-party/eigen/releases/download/5.0.0-h0.trunk/eigen-5.0.0.tar.gz) | Download eigen source code from gitcode, used as compilation dependency |
|  Dependency  | Not involved  | ops-nn/install_deps.sh | [https://apt.kitware.com/keys/kitware-archive-latest.asc](https://apt.kitware.com/keys/kitware-archive-latest.asc) | Download install_deps source code from gitcode, used as compilation dependency |
|  Dependency  | Not involved  | ops-nn/install_deps.sh | [https://apt.kitware.com/ubuntu/](https://apt.kitware.com/ubuntu/) | Download install_deps source code from gitcode, used as compilation dependency |
|  Dependency  | Not involved  | cmake | [https://apt.kitware.com/keys/kitware-archive-latest.asc](https://apt.kitware.com/keys/kitware-archive-latest.asc) | Download cmake software from kitware, used as compilation dependency |
|  Dependency  | Not involved  | cmake | [https://apt.kitware.com/ubuntu/](https://apt.kitware.com/ubuntu/) | Download cmake software from kitware, used as compilation dependency |

## Vulnerability Mechanism Description

[Vulnerability Management](https://gitcode.com/cann/community/blob/master/security/security.md)

## Appendix

### A-File (Folder) Permission Control Recommended Maximum Values in Various Scenarios

| Type           | Linux Permission Reference Maximum Value |
| -------------- | ---------------  |
| User Home Directory                        |   750 (rwxr-x---)            |
| Program Files (including script files, library files, etc.)       |   550 (r-xr-x---)             |
| Program File Directory                      |   550 (r-xr-x---)            |
| Configuration File                          |  640 (rw-r-----)             |
| Configuration File Directory                      |   750 (rwxr-x---)            |
| Log File (recording completed or archived)        |  440 (r--r-----)             |
| Log File (currently recording)                |    640 (rw-r-----)           |
| Log File Directory                      |   750 (rwxr-x---)            |
| Debug File                         |  640 (rw-r-----)         |
| Debug File Directory                     |   750 (rwxr-x---)  |
| Temporary File Directory                      |   750 (rwxr-x---)   |
| Maintenance Upgrade File Directory                  |   770 (rwxrwx---)    |
| Business Data File                      |   640 (rw-r-----)    |
| Business Data File Directory                  |   750 (rwxr-x---)      |
| Key Component, Private Key, Certificate, Ciphertext File Directory    |  700 (rwx-----)      |
| Key Component, Private Key, Certificate, Encrypted Ciphertext        | 600 (rw-------)      |
| Encryption/Decryption Interface, Encryption/Decryption Script            |   500 (r-x------)        |
