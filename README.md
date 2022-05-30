# Hybrid Test

- Test the performance of calling cuda from python.
- Test the usage of TVM's c++ interface. 

## Usage

Run `source setup.sh` to build the lib.

## Result

### Scatter Op 

depth = 2, seqlen = 3
|(batch size, hidden size)|(64, 64)| (128, 128)|(256, 256)| (512, 512)|
|--|--|--|--|--|
|TVM relay(ms)|0.0164| 0.0173|0.0144| 0.1141|
|Our(ms)|0.0045|0.0029| 0.0041| 0.0088|

> 经测试同一网络下一次gather和一次scatter用时较为接近。

### Total
以(10, 100, 64, 64)为例，近似估算kernel开销
- cell: 0.0127ms
  - 调用次数
    - mm ：seqlen \* depth
    - 或bmm : 2 (seqlen + depth) - 1
- gather/scatter: 0.0045ms
  - 理想情况下一次copy时间近似为一次scatter
  - 调用次数
    - mm ：约 5 \* seqlen \* depth
    - 或bmm : 约 3 \* seqlen \* depth + 5 (seqlen + depth) 
    - **当前实现上瓶颈为copy**(对应python中scan结果列表的append)的次数

在使用bmm情况下约为2.781 + 15.749 = 18.531 ms
- FT库中测试结果：ft = 7.962 ms, CuDNN = 14.298 ms

