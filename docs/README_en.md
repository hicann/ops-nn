# Project Documentation

## Directory Description

The key directory structure is as follows:

```text
├── context                            # public directory, storing documents including basic concepts, project directory introduction, build parameter description, and so on
│   ├── dir_structure.md
│   ├── build.md
│   └── ...
├── debug                              # operator debugging and tuning document directory
│   ├── op_debug_prof.md
│   └── ...
├── develop                            # operator development document directory (including AI Core and AI CPU operator development guides)
│   ├── aicore_develop_guide.md
│   ├── aicpu_develop_guide.md
│   └── ...
├── figures                            # image directory
├── invocation                         # operator invocation document directory (including aclnn invocation, graph mode invocation, and so on)
│   ├──op_invocation.md
│   └── ...
├── op_api_list.md                     # complete operator interface list (aclnn)
├── op_list.md                         # complete operator list
└── README
```

## Document Description

The complete documentation of the project is as follows. Please obtain the corresponding content as needed.

| Document                                             | Description                                                         |
| ------------------------------------------------ | ------------------------------------------------------------ |
| [Operator List](zh/op_list.md)                        | Introduces the list of all operators included in the project.                                 |
| [aclnn List](zh/op_api_list.md)                   | Introduces all operator APIs included in the project. You can directly invoke operators through this API.             |
| [Environment Deployment](en/context/quick_install.md)          | Introduces the basic environment setup process, including the acquisition and installation of software packages and third-party dependencies in different scenarios. |
| [Operator Invocation](en/invocation/quick_op_invocation.md) | Introduces how to compile source code and execute operators, including operator package compilation, operator sample execution, UT execution, and so on in different scenarios. |
| [Operator Development](en/develop/aicore_develop_guide.md)   | Introduces how to develop new operators based on this project engineering, including operator prototype definition, Tiling implementation, Kernel implementation, and so on. |
| [Operator Invocation Methods](en/invocation/op_invocation.md)   | Introduces multiple operator invocation methods and invocation processes, such as aclnn invocation, graph mode invocation, and so on. |
| [Operator Debugging and Tuning](en/debug/op_debug_prof.md)        | Introduces common operator debugging and tuning methods.                               |

## Appendix

| Document                                | Description                                                         |
| ----------------------------------- | ------------------------------------------------------------ |
| [Operator Basic Concepts](en/context/basic_concept.md) | Introduces basic concepts and terminology in the operator domain, such as quantization/sparse, data type, data format, and so on. |
| [build Parameter Description](en/context/build.md)   | Introduces the functions and parameter meanings of the build.sh script in this project.                 |
