!dtype = f16
!Q     = tensor<16x4096x64xf16>
!K     = tensor<16x4096x64xf16>
!V     = tensor<16x4096x64xf16>
!O     = tensor<16x4096x64xf16>

#tuning = #iree_codegen.compilation_info<lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1,128,0,0,32]]>, translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute workgroup_size = [64, 4] subgroup_size = 64 ,{mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>, subgroup_m_count = 4, subgroup_n_count = 1> , llvm_func_attrs = { "amdgpu-waves-per-eu" = "2","denormal-fp-math-f32" = "preserve-sign" }}>>


#Q = affine_map<(b, m, n, k1, k2) -> (b, m, k1)>
#K = affine_map<(b, m, n, k1, k2) -> (b, k2, k1)>
#V = affine_map<(b, m, n, k1, k2) -> (b, k2, n)>
#S = affine_map<(b, m, n, k1, k2) -> ()>
#O = affine_map<(b, m, n, k1, k2) -> (b, m, n)>

func.func @main(%Q : !Q, %K : !K, %V : !V) -> !O {
  %scale = arith.constant 1.0 : !dtype
  %empty = tensor.empty() : !O
  %O = iree_linalg_ext.attention 
       { indexing_maps = [#Q, #K, #V, #S, #O]
         ,compilation_info = #tuning
       }
       ins(%Q, %K, %V, %scale : !Q, !K, !V, !dtype)
       outs(%empty : !O) -> !O
  return %O : !O
}
