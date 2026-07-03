import torch
from facenet_pytorch import InceptionResnetV1
import os

def convert_to_onnx():
    print("Loading pre-trained InceptionResnetV1 model (vggface2)...")
    # Initialize the model and load pre-trained weights
    model = InceptionResnetV1(pretrained='vggface2').eval()
    
    # Ensure the models directory exists
    os.makedirs('models', exist_ok=True)
    onnx_path = os.path.join('models', 'facenet.onnx')
    
    print("Creating dummy input tensor...")
    # The expected input size is (batch_size, channels, height, width)
    # facenet-pytorch expects 160x160 aligned face images
    dummy_input = torch.randn(1, 3, 160, 160)
    
    print(f"Exporting model to {onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,        # store the trained parameter weights inside the model file
        opset_version=11,          # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=['input'],     # the model's input names
        output_names=['output'],   # the model's output names
        dynamic_axes={
            'input': {0: 'batch_size'},    # variable length axes
            'output': {0: 'batch_size'}
        }
    )
    
    print("Conversion completed successfully!")

if __name__ == "__main__":
    convert_to_onnx()
