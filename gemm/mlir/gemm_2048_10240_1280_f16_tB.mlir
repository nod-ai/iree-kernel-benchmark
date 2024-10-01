#translation = #iree_codegen.translation_info<None workgroup_size = [128, 2, 1] subgroup_size = 64>
module attributes {transform.with_named_sequence} {
  stream.executable private @gemm {
    stream.executable.export public @gemm workgroups() -> (index, index, index) {
      %c32 = arith.constant 32 : index
      %c160 = arith.constant 160 : index
      %c1 = arith.constant 1 : index
      stream.return %c32, %c160, %c1 : index, index, index
    }
    builtin.module {
      func.func @gemm(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding) attributes {translation_info = #translation} {
        %c19 = arith.constant 19 : index
        %c18 = arith.constant 18 : index
        %c17 = arith.constant 17 : index
        %c3 = arith.constant 3 : index
        %c2 = arith.constant 2 : index
        %c16 = arith.constant 16 : index
        %c8 = arith.constant 8 : index
        %c4 = arith.constant 4 : index
        %c32 = arith.constant 32 : index
        %c64 = arith.constant 64 : index
        %c1 = arith.constant 1 : index
        %c40 = arith.constant 40 : index
        %c0 = arith.constant 0 : index
        %cst = arith.constant dense<0.000000e+00> : vector<4xf32>
        %workgroup_id_0 = stream.dispatch.workgroup.id[0] : index
        %workgroup_id_1 = stream.dispatch.workgroup.id[1] : index
        %thread_id_x = gpu.thread_id  x
        %thread_id_y = gpu.thread_id  y
        %alloc = memref.alloc() : memref<64x32xf16, #gpu.address_space<workgroup>>
        %alloc_0 = memref.alloc() : memref<64x32xf16, #gpu.address_space<workgroup>>
        %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<2048x1280xf16, strided<[1280, 1], offset: ?>>
        %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<10240x1280xf16, strided<[1280, 1], offset: ?>>
        %2 = arith.muli %workgroup_id_0, %c64 : index
        %3 = arith.muli %thread_id_y, %c32 : index
        %4 = arith.divsi %thread_id_x, %c4 : index
        %5 = arith.addi %4, %3 : index
        %6 = arith.remsi %5, %c64 : index
        %7 = arith.addi %6, %2 : index
        %8 = arith.remsi %thread_id_x, %c4 : index
        %9 = arith.muli %8, %c8 : index
        %10 = arith.divsi %thread_id_x, %c64 : index
        %11 = arith.muli %10, %c32 : index
        %12 = arith.remsi %thread_id_x, %c16 : index
        %13 = arith.addi %12, %11 : index
        %14 = arith.remsi %thread_id_x, %c64 : index
        %15 = arith.divsi %14, %c16 : index
        %16 = arith.muli %15, %c4 : index
        %17 = arith.addi %16, %c16 : index
        %18 = arith.addi %13, %c16 : index
        %19 = arith.muli %workgroup_id_1, %c64 : index
        %20 = arith.addi %6, %19 : index
        %21 = arith.addi %12, %3 : index
        %22 = arith.addi %21, %c16 : index
        %23:4 = scf.for %arg3 = %c0 to %c40 step %c1 iter_args(%arg4 = %cst, %arg5 = %cst, %arg6 = %cst, %arg7 = %cst) -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>) {
          %62 = arith.muli %arg3, %c32 : index
          %63 = arith.addi %62, %9 : index
          %64 = vector.load %0[%7, %63] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<8xf16>
          vector.store %64, %alloc[%6, %9] : memref<64x32xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          amdgpu.lds_barrier
          %65 = vector.load %alloc[%13, %16] : memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %66 = vector.load %alloc[%13, %17] : memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %67 = vector.load %alloc[%18, %16] : memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %68 = vector.load %alloc[%18, %17] : memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %69 = vector.load %1[%20, %63] : memref<10240x1280xf16, strided<[1280, 1], offset: ?>>, vector<8xf16>
          amdgpu.lds_barrier
          vector.store %69, %alloc_0[%6, %9] : memref<64x32xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          amdgpu.lds_barrier
          %70 = vector.load %alloc_0[%21, %16] : memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %71 = vector.load %alloc_0[%21, %17] : memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %72 = vector.load %alloc_0[%22, %16] : memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %73 = vector.load %alloc_0[%22, %17] : memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %74 = amdgpu.mfma %65 * %70 + %arg4 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %75 = amdgpu.mfma %66 * %71 + %74 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %76 = amdgpu.mfma %67 * %72 + %arg7 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %77 = amdgpu.mfma %68 * %73 + %76 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %78 = amdgpu.mfma %67 * %70 + %arg6 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %79 = amdgpu.mfma %68 * %71 + %78 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %80 = amdgpu.mfma %65 * %72 + %arg5 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %81 = amdgpu.mfma %66 * %73 + %80 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          scf.yield %75, %81, %79, %77 : vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>
        }
        %24 = vector.extract_strided_slice %23#0 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %25 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<2048x10240xf32, strided<[10240, 1], offset: ?>>
        %26 = arith.remsi %thread_id_x, %c64 : index
        %27 = arith.divsi %26, %c16 : index
        %28 = arith.muli %27, %c4 : index
        %29 = arith.divsi %thread_id_x, %c64 : index
        %30 = arith.muli %29, %c32 : index
        %31 = arith.muli %workgroup_id_0, %c64 : index
        %32 = arith.addi %31, %30 : index
        %33 = arith.addi %32, %28 : index
        %34 = arith.muli %thread_id_y, %c32 : index
        %35 = arith.muli %workgroup_id_1, %c64 : index
        %36 = arith.remsi %thread_id_x, %c16 : index
        %37 = arith.addi %36, %35 : index
        %38 = arith.addi %37, %34 : index
        vector.store %24, %25[%33, %38] : memref<2048x10240xf32, strided<[10240, 1], offset: ?>>, vector<1xf32>
        %39 = vector.extract_strided_slice %23#0 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %40 = arith.addi %33, %c1 : index
        vector.store %39, %25[%40, %38] : memref<2048x10240xf32, strided<[10240, 1], offset: ?>>, vector<1xf32>
        %41 = vector.extract_strided_slice %23#0 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %42 = arith.addi %33, %c2 : index
        vector.store %41, %25[%42, %38] : memref<2048x10240xf32, strided<[10240, 1], offset: ?>>, vector<1xf32>
        %43 = vector.extract_strided_slice %23#0 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %44 = arith.addi %33, %c3 : index
        vector.store %43, %25[%44, %38] : memref<2048x10240xf32, strided<[10240, 1], offset: ?>>, vector<1xf32>
        %45 = vector.extract_strided_slice %23#3 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %46 = arith.addi %33, %c16 : index
        %47 = arith.addi %38, %c16 : index
        vector.store %45, %25[%46, %47] : memref<2048x10240xf32, strided<[10240, 1], offset: ?>>, vector<1xf32>
        %48 = vector.extract_strided_slice %23#3 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %49 = arith.addi %33, %c17 : index
        vector.store %48, %25[%49, %47] : memref<2048x10240xf32, strided<[10240, 1], offset: ?>>, vector<1xf32>
        %50 = vector.extract_strided_slice %23#3 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %51 = arith.addi %33, %c18 : index
        vector.store %50, %25[%51, %47] : memref<2048x10240xf32, strided<[10240, 1], offset: ?>>, vector<1xf32>
        %52 = vector.extract_strided_slice %23#3 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %53 = arith.addi %33, %c19 : index
        vector.store %52, %25[%53, %47] : memref<2048x10240xf32, strided<[10240, 1], offset: ?>>, vector<1xf32>
        %54 = vector.extract_strided_slice %23#2 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %54, %25[%46, %38] : memref<2048x10240xf32, strided<[10240, 1], offset: ?>>, vector<1xf32>
        %55 = vector.extract_strided_slice %23#2 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %55, %25[%49, %38] : memref<2048x10240xf32, strided<[10240, 1], offset: ?>>, vector<1xf32>
        %56 = vector.extract_strided_slice %23#2 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %56, %25[%51, %38] : memref<2048x10240xf32, strided<[10240, 1], offset: ?>>, vector<1xf32>
        %57 = vector.extract_strided_slice %23#2 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %57, %25[%53, %38] : memref<2048x10240xf32, strided<[10240, 1], offset: ?>>, vector<1xf32>
        %58 = vector.extract_strided_slice %23#1 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %58, %25[%33, %47] : memref<2048x10240xf32, strided<[10240, 1], offset: ?>>, vector<1xf32>
        %59 = vector.extract_strided_slice %23#1 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %59, %25[%40, %47] : memref<2048x10240xf32, strided<[10240, 1], offset: ?>>, vector<1xf32>
        %60 = vector.extract_strided_slice %23#1 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %60, %25[%42, %47] : memref<2048x10240xf32, strided<[10240, 1], offset: ?>>, vector<1xf32>
        %61 = vector.extract_strided_slice %23#1 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %61, %25[%44, %47] : memref<2048x10240xf32, strided<[10240, 1], offset: ?>>, vector<1xf32>
        return
      }
    }
  }
  func.func @isolated_benchmark(%arg0: tensor<2048x1280xf16>, %arg1: tensor<10240x1280xf16>) -> tensor<2048x10240xf32> {
    %0 = flow.dispatch @gemm::@gemm(%arg0, %arg1) : (tensor<2048x1280xf16>, tensor<10240x1280xf16>) -> tensor<2048x10240xf32>
    return %0 : tensor<2048x10240xf32>
  }
}
