# IREE-torch u-kernel benchmarks

A set of python scripts to benchmark u-kernels produced by IREE against torch.

### Dependencies

Before running the benchmarks install the python `requirements.txt`, this will
install pytorch and iree-turbine.

### Example:

```bash
python attention.py -c gfx942 -b 4 -k1 128 -n 128 -k2 128 -m 128 -B 4 4 --dtype bf16 --dynamic-dims k2
```

Produces:
```
********************************************************************************
Attention:
        b=Dim(size=4, dynamic=False), m=Dim(size=128, dynamic=False), n=Dim(size=128, dynamic=False), k1=Dim(size=128, dynamic=False), k2=Dim(size=128, dynamic=True)
        batch=(4, 4)
        is_casual=False, enable_gqa=False, dtype=torch.bfloat16
        num_its=10


========================================
Compiling IREE...
module @module {
  func.func @main(%arg0: !torch.vtensor<[4,4,4,128,128],bf16>, %arg1: !torch.vtensor<[4,4,4,?,128],bf16>, %arg2: !torch.vtensor<[4,4,4,?,128],bf16>) -> !torch.vtensor<[4,4,4,128,128],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.symbolic_int "s0" {min_val = 0, max_val = 9223372036854775807} : !torch.int
    torch.bind_symbolic_shape %arg1, [%0], affine_map<()[s0] -> (4, 4, 4, s0, 128)> : !torch.vtensor<[4,4,4,?,128],bf16>
    torch.bind_symbolic_shape %arg2, [%0], affine_map<()[s0] -> (4, 4, 4, s0, 128)> : !torch.vtensor<[4,4,4,?,128],bf16>
    %none = torch.constant.none
    %float0.000000e00 = torch.constant.float 0.000000e+00
    %false = torch.constant.bool false
    %none_0 = torch.constant.none
    %false_1 = torch.constant.bool false
    %1 = torch.aten.scaled_dot_product_attention %arg0, %arg1, %arg2, %none, %float0.000000e00, %false, %none_0, %false_1 : !torch.vtensor<[4,4,4,128,128],bf16>, !torch.vtensor<[4,4,4,?,128],bf16>, !torch.vtensor<[4,4,4,?,128],bf16>, !torch.none, !torch.float, !torch.bool, !torch.none, !torch.bool -> !torch.vtensor<[4,4,4,128,128],bf16>
    return %1 : !torch.vtensor<[4,4,4,128,128],bf16>
  }
}

Done compiling

========================================
Profiling torch:
[W721 11:49:47.533001874 collection.cpp:1110] Warning: ROCTracer produced duplicate flow start: 118 (function operator())
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Cijk_Ailk_Bljk_SB_Bias_AS_SAV_UserArgs_MT32x256x16_M...         0.00%       0.000us         0.00%       0.000us       0.000us     446.762us        62.10%     446.762us      22.338us            20
                       triton_per_fused__safe_softmax_2         0.00%       0.000us         0.00%       0.000us       0.000us      94.970us        13.20%      94.970us       9.497us            10
                        triton_poi_fused__to_copy_mul_0         0.00%       0.000us         0.00%       0.000us       0.000us      54.988us         7.64%      54.988us       5.499us            10
                            triton_poi_fused__to_copy_4         0.00%       0.000us         0.00%       0.000us       0.000us      46.500us         6.46%      46.500us       4.650us            10
                                 triton_poi_fused_mul_1         0.00%       0.000us         0.00%       0.000us       0.000us      44.295us         6.16%      44.295us       4.430us            10
                            triton_poi_fused__to_copy_3         0.00%       0.000us         0.00%       0.000us       0.000us      31.879us         4.43%      31.879us       3.188us            10
                                 hipPointerGetAttribute         3.43%      18.408us         3.43%      18.408us       0.205us       0.000us         0.00%       0.000us       0.000us            90
                                  hipModuleLaunchKernel        47.18%     253.038us        47.18%     253.038us       5.061us       0.000us         0.00%       0.000us       0.000us            50
                            hipGetDevicePropertiesR0600         2.41%      12.919us         2.41%      12.919us       0.215us       0.000us         0.00%       0.000us       0.000us            60
                               hipExtModuleLaunchKernel        17.12%      91.819us        17.12%      91.819us       4.591us       0.000us         0.00%       0.000us       0.000us            20
                                   hipDeviceSynchronize        29.86%     160.129us        29.86%     160.129us      14.557us       0.000us         0.00%       0.000us       0.000us            11
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 536.313us
Self CUDA time total: 719.394us


========================================
Profiling IREE:
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
main$async_dispatch_0_attention_64xDx128xbf16_dispat...         0.00%       0.000us         0.00%       0.000us       0.000us       1.521ms       100.00%       1.521ms     152.132us            10
                                      hipCtxPushCurrent         0.36%      10.326us         0.36%      10.326us       0.172us       0.000us         0.00%       0.000us       0.000us            60
                                       hipCtxPopCurrent         0.22%       6.211us         0.22%       6.211us       0.104us       0.000us         0.00%       0.000us       0.000us            60
                                     hipStreamWaitEvent         0.92%      25.970us         0.92%      25.970us       2.597us       0.000us         0.00%       0.000us       0.000us            10
                                    hipEventSynchronize        95.21%       2.699ms        95.27%       2.701ms      96.469us       0.000us         0.00%       0.000us       0.000us            28
                                  hipModuleLaunchKernel         2.16%      61.131us         2.31%      65.357us       6.536us       0.000us         0.00%       0.000us       0.000us            10
                                   hipDeviceSynchronize         1.14%      32.247us         1.14%      32.247us       2.932us       0.000us         0.00%       0.000us       0.000us            11
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 2.835ms
Self CUDA time total: 1.521ms


========================================
Summary:
Total torch time: 0.07194 ms
Total IREE time: 0.15213 ms
Numeric error: 0.004910651520237259
```
