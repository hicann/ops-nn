# Non-contiguous Tensor

Currently, most operator APIs support "**non-contiguous Tensor**" as input aclTensor, that is, a Tensor can be represented by (shape, strides, offset).

Note: You can create an aclTensor through the "Public Interfaces > aclCreateTensor" section in [Operator Library Interface](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/API/aolapi/operatorlist_00001.html).

## Example 1

For example, consider a Tensor with shape=(6, 5), strides=(10, 1), and offset=22. Its memory layout is as follows:
> a<sub>0,0</sub> , a<sub>0,1</sub> , a<sub>0,2</sub> , a<sub>0,3</sub> , a<sub>0,4</sub> , a<sub>0,5</sub> , a<sub>0,6</sub> , a<sub>0,7</sub> , a<sub>0,8</sub> , a<sub>0,9</sub>
> a<sub>1,0</sub> , a<sub>1,1</sub> , a<sub>1,2</sub> , a<sub>1,3</sub> , a<sub>1,4</sub> , a<sub>1,5</sub> , a<sub>1,6</sub> , a<sub>1,7</sub> , a<sub>1,8</sub> , a<sub>1,9</sub>
> a<sub>2,0</sub> , a<sub>2,1</sub> , **a<sub>2,2</sub> , a<sub>2,3</sub> , a<sub>2,4</sub> , a<sub>2,5</sub> , a<sub>2,6</sub>** , a<sub>2,7</sub> , a<sub>2,8</sub> , a<sub>2,9</sub>
> a<sub>3,0</sub> , a<sub>3,1</sub> , **a<sub>3,2</sub> , a<sub>3,3</sub> , a<sub>3,4</sub> , a<sub>3,5</sub> , a<sub>3,6</sub>** , a<sub>3,7</sub> , a<sub>3,8</sub> , a<sub>3,9</sub>
> a<sub>4,0</sub> , a<sub>4,1</sub> , **a<sub>4,2</sub> , a<sub>4,3</sub> , a<sub>4,4</sub> , a<sub>4,5</sub> , a<sub>4,6</sub>** , a<sub>4,7</sub> , a<sub>4,8</sub> , a<sub>4,9</sub>
> a<sub>5,0</sub> , a<sub>5,1</sub> , **a<sub>5,2</sub> , a<sub>5,3</sub> , a<sub>5,4</sub> , a<sub>5,5</sub> , a<sub>5,6</sub>** , a<sub>5,7</sub> , a<sub>5,8</sub> , a<sub>5,9</sub>
> a<sub>6,0</sub> , a<sub>6,1</sub> , **a<sub>6,2</sub> , a<sub>6,3</sub> , a<sub>6,4</sub> , a<sub>6,5</sub> , a<sub>6,6</sub>** , a<sub>6,7</sub> , a<sub>6,8</sub> , a<sub>6,9</sub>
> a<sub>7,0</sub> , a<sub>7,1</sub> , **a<sub>7,2</sub> , a<sub>7,3</sub> , a<sub>7,4</sub> , a<sub>7,5</sub> , a<sub>7,6</sub>** , a<sub>7,7</sub> , a<sub>7,8</sub> , a<sub>7,9</sub>
> a<sub>8,0</sub> , a<sub>8,1</sub> , a<sub>8,2</sub> , a<sub>8,3</sub> , a<sub>8,4</sub> , a<sub>8,5</sub> , a<sub>8,6</sub> , a<sub>8,7</sub> , a<sub>8,8</sub> , a<sub>8,9</sub>
> a<sub>9,0</sub> , a<sub>9,1</sub> , a<sub>9,2</sub> , a<sub>9,3</sub> , a<sub>9,4</sub> , a<sub>9,5</sub> , a<sub>9,6</sub> , a<sub>9,7</sub> , a<sub>9,8</sub> , a<sub>9,9</sub>

That is, the Tensor is laid out in the dark positions shown above. This complete Tensor is non-contiguous in memory layout. Strides describe the interval between two adjacent elements in the Tensor dimension. If the stride in dimension 1 is 1, that dimension is contiguous; if the stride in dimension 0 is 10, then adjacent elements are separated by 10 elements, which is non-contiguous. Offset represents the offset of the first element of this Tensor relative to addr.

## Example 2

For example, consider a Tensor with shape=(4, 3), strides=(20, 2), and offset=22. Its memory layout is as follows:

> a<sub>0,0</sub> , a<sub>0,1</sub> , a<sub>0,2</sub> , a<sub>0,3</sub> , a<sub>0,4</sub> , a<sub>0,5</sub> , a<sub>0,6</sub> , a<sub>0,7</sub> , a<sub>0,8</sub> , a<sub>0,9</sub>
> a<sub>1,0</sub> , a<sub>1,1</sub> , a<sub>1,2</sub> , a<sub>1,3</sub> , a<sub>1,4</sub> , a<sub>1,5</sub> , a<sub>1,6</sub> , a<sub>1,7</sub> , a<sub>1,8</sub> , a<sub>1,9</sub>
> a<sub>2,0</sub> , a<sub>2,1</sub> , **a<sub>2,2</sub>** , a<sub>2,3</sub> , **a<sub>2,4</sub>** , a<sub>2,5</sub> , **a<sub>2,6</sub>** , a<sub>2,7</sub> , a<sub>2,8</sub> , a<sub>2,9</sub>
> a<sub>3,0</sub> , a<sub>3,1</sub> , a<sub>3,2</sub> , a<sub>3,3</sub> , a<sub>3,4</sub> , a<sub>3,5</sub> , a<sub>3,6</sub> , a<sub>3,7</sub> , a<sub>3,8</sub> , a<sub>3,9</sub>
> a<sub>4,0</sub> , a<sub>4,1</sub> , **a<sub>4,2</sub>** , a<sub>4,3</sub> , **a<sub>4,4</sub>** , a<sub>4,5</sub> , **a<sub>4,6</sub>** , a<sub>4,7</sub> , a<sub>4,8</sub> , a<sub>4,9</sub>
> a<sub>5,0</sub> , a<sub>5,1</sub> , a<sub>5,2</sub> , a<sub>5,3</sub> , a<sub>5,4</sub> , a<sub>5,5</sub> , a<sub>5,6</sub> , a<sub>5,7</sub> , a<sub>5,8</sub> , a<sub>5,9</sub>
> a<sub>6,0</sub> , a<sub>6,1</sub> , **a<sub>6,2</sub>** , a<sub>6,3</sub> , **a<sub>6,4</sub>** , a<sub>6,5</sub> , **a<sub>6,6</sub>** , a<sub>6,7</sub> , a<sub>6,8</sub> , a<sub>6,9</sub>
> a<sub>7,0</sub> , a<sub>7,1</sub> , a<sub>7,2</sub> , a<sub>7,3</sub> , a<sub>7,4</sub> , a<sub>7,5</sub> , a<sub>7,6</sub> , a<sub>7,7</sub> , a<sub>7,8</sub> , a<sub>7,9</sub>
> a<sub>8,0</sub> , a<sub>8,1</sub> , **a<sub>8,2</sub>** , a<sub>8,3</sub> , **a<sub>8,4</sub>** , a<sub>8,5</sub> , **a<sub>8,6</sub>** , a<sub>8,7</sub> , a<sub>8,8</sub> , a<sub>8,9</sub>
> a<sub>9,0</sub> , a<sub>9,1</sub> , a<sub>9,2</sub> , a<sub>9,3</sub> , a<sub>9,4</sub> , a<sub>9,5</sub> , a<sub>9,6</sub> , a<sub>9,7</sub> , a<sub>9,8</sub> , a<sub>9,9</sub>

That is, the Tensor is laid out in the dark positions shown above. This complete Tensor is non-contiguous in memory layout. Strides describe the interval between two adjacent elements in the Tensor dimension. If the stride in dimension 1 is 2, that dimension has an interval of 1 element; if the stride in dimension 0 is 20, then adjacent elements are separated by 20 elements, which is non-contiguous. Offset represents the offset of the first element of this Tensor relative to addr.
