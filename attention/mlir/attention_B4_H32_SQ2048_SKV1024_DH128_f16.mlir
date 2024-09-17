
module {
    func.func @main_0(%295 : !torch.vtensor<[4,32,2048,128],f16>, %298 : !torch.vtensor<[4,32,1024,128],f16>, %301 : !torch.vtensor<[4,32,1024,128],f16>) -> !torch.vtensor<[4,32,2048,128],f16> {
        %false_371 = torch.constant.bool false
        %float0.000000e00 = torch.constant.float 0.000000e+00
        %none_372 = torch.constant.none
        %none_373 = torch.constant.none
        %282:2 = torch.operator "torch.aten._scaled_dot_product_flash_attention_for_cpu"(%295, %298, %301, %float0.000000e00, %false_371, %none_372, %none_373) : (!torch.vtensor<[4,32,2048,128],f16>, !torch.vtensor<[4,32,1024,128],f16>, !torch.vtensor<[4,32,1024,128],f16>, !torch.float, !torch.bool, !torch.none, !torch.none) -> (!torch.vtensor<[4,32,2048,128],f16>, !torch.vtensor<[4,32,2048], f32>)
        return %282#0 : !torch.vtensor<[4,32,2048,128],f16>
    }
} 
