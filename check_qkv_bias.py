#!/usr/bin/env python3
"""
检查UNI模型的 qkv_bias 配置
"""
import torch
import timm

def check_qkv_bias():
    print("=" * 80)
    print("检查 UNI 模型的 qkv_bias 配置")
    print("=" * 80)
    
    checkpoint_path = "hf-hub:MahmoodLab/uni"
    
    try:
        print(f"\n正在加载 UNI 模型...")
        timm_kwargs = {
            "dynamic_img_size": True,
            "num_classes": 0,
            "init_values": 1e-5
        }
        model = timm.create_model(checkpoint_path, pretrained=True, **timm_kwargs)
        print("✓ 模型加载成功！\n")
        
        # 检查第一个block的attention的qkv层
        if hasattr(model, 'blocks') and len(model.blocks) > 0:
            first_block = model.blocks[0]
            
            print("第一个 Transformer Block 的配置:")
            print(f"  Block类型: {type(first_block).__name__}")
            
            if hasattr(first_block, 'attn'):
                attn = first_block.attn
                print(f"  Attention类型: {type(attn).__name__}")
                
                if hasattr(attn, 'qkv'):
                    qkv_layer = attn.qkv
                    print(f"\n  QKV Layer:")
                    print(f"    - 类型: {type(qkv_layer).__name__}")
                    print(f"    - Weight shape: {qkv_layer.weight.shape}")
                    
                    # 检查是否有bias
                    if qkv_layer.bias is not None:
                        print(f"    - Bias shape: {qkv_layer.bias.shape}")
                        print(f"    - ✓ qkv_bias = True")
                    else:
                        print(f"    - Bias: None")
                        print(f"    - ✓ qkv_bias = False")
                
                # 检查proj层
                if hasattr(attn, 'proj'):
                    proj_layer = attn.proj
                    print(f"\n  Proj Layer:")
                    print(f"    - Weight shape: {proj_layer.weight.shape}")
                    if proj_layer.bias is not None:
                        print(f"    - Bias shape: {proj_layer.bias.shape}")
                        print(f"    - ✓ proj_bias = True")
                    else:
                        print(f"    - Bias: None")
                        print(f"    - ✓ proj_bias = False")
            
            # 检查MLP/FFN
            if hasattr(first_block, 'mlp'):
                mlp = first_block.mlp
                print(f"\n  MLP/FFN:")
                if hasattr(mlp, 'fc1'):
                    fc1 = mlp.fc1
                    print(f"    - FC1 Weight shape: {fc1.weight.shape}")
                    if fc1.bias is not None:
                        print(f"    - FC1 Bias shape: {fc1.bias.shape}")
                        print(f"    - ✓ ffn_bias = True")
                    else:
                        print(f"    - FC1 Bias: None")
                        print(f"    - ✓ ffn_bias = False")
        
        # 统计所有Linear层的bias使用情况
        print("\n" + "=" * 80)
        print("统计所有Linear层的bias使用情况:")
        print("=" * 80)
        
        total_linear = 0
        with_bias = 0
        without_bias = 0
        
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                total_linear += 1
                if module.bias is not None:
                    with_bias += 1
                    if 'qkv' in name:
                        print(f"  [QKV有bias] {name}: weight {module.weight.shape}, bias {module.bias.shape}")
                else:
                    without_bias += 1
                    if 'qkv' in name:
                        print(f"  [QKV无bias] {name}: weight {module.weight.shape}")
        
        print(f"\n总Linear层数: {total_linear}")
        print(f"  - 有bias: {with_bias} ({with_bias/total_linear*100:.1f}%)")
        print(f"  - 无bias: {without_bias} ({without_bias/total_linear*100:.1f}%)")
        
    except Exception as e:
        print(f"\n✗ 失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    check_qkv_bias()






