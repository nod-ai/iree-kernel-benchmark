!dtype = f8E4M3FNUZ
!Q     = tensor<32x4096x128xf8E4M3FNUZ>
!K     = tensor<32x4096x128xf8E4M3FNUZ>
!V     = tensor<32x4096x128xf8E4M3FNUZ>
!O     = tensor<32x4096x128xf8E4M3FNUZ>



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
         
       }
       ins(%Q, %K, %V, %scale : !Q, !K, !V, !dtype)
       outs(%empty : !O) -> !O
  return %O : !O
}
