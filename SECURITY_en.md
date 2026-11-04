# Security Statement

## Recommendations for the Running User

For security reasons, using administrator accounts such as root to execute any command is not recommended. Follow the principle of least privilege.

## File Permission Control

- You are advised to set the umask value of the running system to 0027 or higher on the host (including the host machine) and in containers. This ensures that the default maximum permission of a newly created folder is 750 and the default maximum permission of a newly created file is 640.
- You are advised to take security measures such as permission control for sensitive content, including personal privacy data, commercial assets, source files, and various files saved during operator development. For example, for permission control over the installation directory of this project and over input public data files, refer to [A-Recommended Maximum Permissions for Files (Folders) in Each Scenario](#a-recommended-maximum-permissions-for-files-folders-in-each-scenario) when setting permissions.
- During operator running, the operator compilation files may be cached and stored in the `kernel_meta_*` folder in the running directory to accelerate subsequent operator invocation. You can perform permission control on the generated files as needed.
- During installation and usage, you need to perform proper permission control. For the settings, refer to the file permission reference in [A-Recommended Maximum Permissions for Files (Folders) in Each Scenario](#a-recommended-maximum-permissions-for-files-folders-in-each-scenario).

## Build Security Statement

When compiling and installing this project from source code, you need to compile it yourself. The compilation process generates some intermediate files. After the compilation is complete, you are advised to perform proper permission control on the intermediate files to ensure file security.

## Runtime Security Statement

- You are advised to write the operator invocation script according to the resource status of the running environment. If the operator invocation script does not match the resource status, for example, the space used for generating input data or golden computation results exceeds the memory capacity limit, or the data saved locally by the script exceeds the disk space, errors may occur and the process may exit unexpectedly.
- When an operator running error occurs, the process exits and prints error information. You are advised to locate the specific error cause based on the error message, for example, by setting the operator to run synchronously or viewing log files.
- When the operator is invoked through [PyTorch](https://gitee.com/ascend/pytorch), a running error may occur due to version mismatch. For details, refer to the [PyTorch Security Statement](https://gitee.com/ascend/pytorch#%E5%AE%89%E5%85%A8%E5%A3%B0%E6%98%8E).

## Public Network Address Statement

The public network addresses contained in the code of this project are declared as follows:

|      Type      |                                           Open-source code address                                           |                            File name                             |             Public IP address/Public URL address/Domain name/Email address/Compressed file address             |                   Purpose description                    |
| :------------: |:------------------------------------------------------------------------------------------:|:----------------------------------------------------------| :---------------------------------------------------------- |:-----------------------------------------|
|  Dependency  | Not involved  | cmake/third_party/makeself-fetch.cmake | [https://gitcode.com/cann-src-third-party/makeself/releases/download/release-2.5.0-patch1.0/makeself-release-2.5.0-patch1.tar.gz](https://gitcode.com/cann-src-third-party/makeself/releases/download/release-2.5.0-patch1.0/makeself-release-2.5.0-patch1.tar.gz) | Downloads the makeself source code from gitcode as a compilation dependency |
|  Dependency  | Not involved  | cmake/third_party/nlohmann_json.cmake | [https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/include.zip](https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/include.zip) | Downloads the json source code from gitcode as a compilation dependency |
|  Dependency  | Not involved  | cmake/third_party/gtest.cmake | [https://gitcode.com/cann-src-third-party/googletest/releases/download/v1.14.0/googletest-1.14.0.tar.gz](https://gitcode.com/cann-src-third-party/googletest/releases/download/v1.14.0/googletest-1.14.0.tar.gz) | Downloads the googletest source code from gitcode as a compilation dependency |
|  Dependency  | Not involved  | cmake/third_party/eigen.cmake | [https://gitcode.com/cann-src-third-party/eigen/releases/download/5.0.0-h0.trunk/eigen-5.0.0.tar.gz](https://gitcode.com/cann-src-third-party/eigen/releases/download/5.0.0-h0.trunk/eigen-5.0.0.tar.gz) | Downloads the eigen source code from gitcode as a compilation dependency |
|  Dependency  | Not involved  | ops-nn/install_deps.sh | [https://apt.kitware.com/keys/kitware-archive-latest.asc](https://apt.kitware.com/keys/kitware-archive-latest.asc) | Downloads the Kitware signature key from apt.kitware.com as a compilation dependency |
|  Dependency  | Not involved  | ops-nn/install_deps.sh | [https://apt.kitware.com/ubuntu/](https://apt.kitware.com/ubuntu/) | Downloads the Kitware signature key from apt.kitware.com as a compilation dependency |
|  Dependency  | Not involved  | cmake | [https://apt.kitware.com/keys/kitware-archive-latest.asc](https://apt.kitware.com/keys/kitware-archive-latest.asc) | Downloads the cmake software from kitware as a compilation dependency |
|  Dependency  | Not involved  | cmake | [https://apt.kitware.com/ubuntu/](https://apt.kitware.com/ubuntu/) | Downloads the cmake software from kitware as a compilation dependency |

## Vulnerability Mechanism Statement

[Vulnerability Management](https://gitcode.com/cann/community/blob/master/security/security.md)

## Appendix

### A-Recommended Maximum Permissions for Files (Folders) in Each Scenario

| Type           | Linux permission reference maximum |
| -------------- | ---------------  |
| User home directory                        |   750 (rwxr-x---)            |
| Program files (including script files, library files, and so on)       |   550 (r-xr-x---)             |
| Program file directory                      |   550 (r-xr-x---)            |
| Configuration file                          |  640 (rw-r-----)             |
| Configuration file directory                      |   750 (rwxr-x---)            |
| Log file (recording completed or archived)        |  440 (r--r-----)             |
| Log file (recording in progress)                |    640 (rw-r-----)           |
| Log file directory                      |   750 (rwxr-x---)            |
| Debug file                         |  640 (rw-r-----)         |
| Debug file directory                     |   750 (rwxr-x---)  |
| Temporary file directory                      |   750 (rwxr-x---)   |
| Maintenance and upgrade file directory                  |   770 (rwxrwx---)    |
| Service data file                      |   640 (rw-r-----)    |
| Service data file directory                  |   750 (rwxr-x---)      |
| Key component, private key, certificate, and ciphertext file directory    |  700 (rwx------)      |
| Key component, private key, certificate, and encrypted ciphertext        | 600 (rw-------)      |
| Encryption/decryption interface and encryption/decryption script            |   500 (r-x------)        |
