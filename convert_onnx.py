import torch
import torchvision.models as models

model_path = "res34_fair_align_multi_4_20190809.pt"  
model = torch.load(model_path, map_location=torch.device('cpu'))
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

onnx_model_path = "resnet34.onnx"
torch.onnx.export(
    model,
    dummy_input,
    onnx_model_path,
    export_params=True,       
    opset_version=11,         
    do_constant_folding=True, 
    input_names=["input"],     
    output_names=["output"],   
    dynamic_axes={            
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
    }
)

print(f"Model has been converted to ONNX and saved at {onnx_model_path}")
