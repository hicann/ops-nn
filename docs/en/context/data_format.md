# Data Formats

Data format (format) is used to describe the business semantics of the axes of a multi-dimensional Tensor, representing the physical layout format of data, such as 1D, 2D, 3D, 4D, 5D, and so on. Generally, CNN (Convolutional Neural Networks) APIs require specific formats to be described.

For the **full range of data formats** supported by aclTensor, refer to [ACL API (C)](https://www.hiascend.com/document/detail/en/canncommercial/latest/API/appdevgapi/aclcppdevg_03_0004.html) under "Data Types and Their Operation Interfaces > aclFormat".

For an introduction to **data format layout principles**, refer to [Ascend C Operator Development Guide](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/programug/Ascendcopdevg/atlas_ascendc_map_10_0002.html) under "Concept Principles and Terminology > Neural Networks and Operators > Data Layout Formats".

## Usage Instructions

Currently, most operator APIs support the ND data format. For example, the aclnnAdd interface indicates that the supported data format is ND (that is, the rule of low-dimensional priority continuous layout for multi-dimensional Tensors). For aclnnConvolution, which is a CNN-type API, the input aclTensor is required to be set with a format that has business semantics, rather than the ND format. Such operators need to know the business semantics in the Tensor during the calculation process to perform the corresponding computation. For example, in 2D convolution, you need to know the correspondence between the Batch dimension, Channel dimension, Height dimension, Width dimension, and the Tensor dimensions.

>**Note:**
>
>- For the parameter description of two-stage interfaces, to simplify the description, **the original data format "ACL\_FORMAT\_XXXX_" is abbreviated as "_XXXX_"**.
>- The meaning of each dimension in the data format: N (Batch) represents the batch size, H (Height) represents the feature map height, W (Width) represents the feature map width, C (Channels) represents the feature map channels, D (Depth) represents the feature map depth, L (Length) represents the feature map length.

## Common Data Formats

When creating an aclTensor through the **aclCreateTensor** interface, you need to set the data format according to the API business requirements. The **supported data formats** are:

ACL\_FORMAT\_ND, ACL\_FORMAT\_NCHW, ACL\_FORMAT\_NHWC, ACL\_FORMAT\_HWCN, ACL\_FORMAT\_NDHWC, ACL\_FORMAT\_NCDHW, ACL\_FORMAT\_NC, ACL\_FORMAT\_NCL.

For non-ND Tensors, the Tensor dimension requirements are consistent with the format description. For example:

- 5D Tensor: Requires ACL\_FORMAT\_NCDHW, ACL\_FORMAT\_NDHWC, or ACL\_FORMAT\_ND (if the API parameter description does not indicate support for ND, setting the ND format will result in an API validation error).
- 4D Tensor: Requires ACL\_FORMAT\_NCHW, ACL\_FORMAT\_NHWC, ACL\_FORMAT\_HWCN, or ACL\_FORMAT\_ND.
- 3D Tensor: Requires ACL\_FORMAT\_NCL or ACL\_FORMAT\_ND.
- 2D Tensor: Requires ACL\_FORMAT\_NC or ACL\_FORMAT\_ND.
- Other dimension Tensors: Require ACL\_FORMAT\_ND.

## Private Data Formats

In addition to the common data formats mentioned above, there are other data formats, such as ACL\_FORMAT\_NC1HWC0, ACL\_FORMAT\_FRACTAL\_Z, ACL\_FORMAT\_NC1HWC0\_C04, ACL\_FORMAT\_FRACTAL\_NZ, ACL\_FORMAT\_NDC1HWC0, ACL\_FORMAT\_FRACTAL\_Z\_3D, and so on.

These formats are private formats of the NPU. Currently, most aclnn APIs do not support these formats. If an individual API declares supported data formats, refer to the actual description of that API.
