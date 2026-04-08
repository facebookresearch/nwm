import argparse
import os
import tensorrt as trt

# 1. Setup the TensorRT Logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def compile_onnx_to_tensorrt(onnx_file_path, engine_file_path, workspace_gb=2):
    print("="*50)
    print("TensorRT Engine Compiler")
    print("="*50)
    print(f"Input ONNX:  {onnx_file_path}")
    print(f"Output Plan: {engine_file_path}")
    print(f"Workspace:   {workspace_gb} GB")
    print("-" * 50)

    if not os.path.exists(onnx_file_path):
        print(f"[ERROR] ONNX file not found at {onnx_file_path}")
        return False

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(engine_file_path)), exist_ok=True)

    # 2. Create the Builder, Network, and Parser
    builder = trt.Builder(TRT_LOGGER)
    # EXPLICIT_BATCH is required for modern TensorRT/ONNX parsing
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # 3. Parse the ONNX file
    print("Parsing ONNX model... (This might take a minute)")
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("[ERROR] Failed to parse the ONNX file. Details:")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False

    # 4. Configure Builder Optimization Profile & Precision
    config = builder.create_builder_config()
    
    # Allocate GPU memory strictly for the compilation process
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))

    # --- PRECISION FLAGS ---
    # Enable FP16 (Standard fast math)
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("[INFO] Enabled FP16 hardware optimizations.")
        
    # Enable BF16 (If supported by your TensorRT version and RTX 4060)
    try:
        config.set_flag(trt.BuilderFlag.BF16)
        print("[INFO] Enabled BF16 hardware optimizations.")
    except AttributeError:
        # Older TensorRT versions might not expose the BF16 Python flag directly
        pass 

    print("\n[START] Compiling TensorRT engine...")
    print(">>> Your GPU will max out. Do not close this terminal. <<<")
    
    # 5. Build and Serialize the Engine
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("[ERROR] Engine compilation failed. (Likely Out-of-Memory. Try lowering workspace_gb).")
        return False

    # 6. Save the physical .engine file to disk
    print("[SUCCESS] Compilation finished. Saving to disk...")
    with open(engine_file_path, 'wb') as f:
        f.write(serialized_engine)
        
    print(f"[DONE] Engine saved to: {engine_file_path}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile an ONNX model to a TensorRT Engine.")
    parser.add_argument("--onnx", type=str, default="./onnx_models/cdit.onnx", help="Path to the input ONNX file.")
    parser.add_argument("--engine", type=str, default="./onnx_models/cdit.engine", help="Path to save the output Engine file.")
    parser.add_argument("--workspace", type=int, default=2, help="Max VRAM workspace size in Gigabytes (default: 2).")
    
    args = parser.parse_args()
    
    compile_onnx_to_tensorrt(args.onnx, args.engine, args.workspace)
