#!/usr/bin/env python3
"""
检查UNI预训练权重的配置
"""
import torch
import timm
from pathlib import Path

def check_uni_weights():
    print("=" * 80)
    print("检查 UNI 预训练权重配置")
    print("=" * 80)
    
    checkpoint_path = "hf-hub:MahmoodLab/uni"
    
    try:
        # 尝试加载UNI模型
        print(f"\n1. 正在加载模型: {checkpoint_path}")
        timm_kwargs = {
            "dynamic_img_size": True,
            "num_classes": 0,
            "init_values": 1e-5
        }
        model = timm.create_model(checkpoint_path, pretrained=True, **timm_kwargs)
        print("✓ 模型加载成功！")
        
        # 检查模型架构
        print("\n2. 模型架构信息:")
        print(f"   - 模型类型: {type(model).__name__}")
        print(f"   - Embed dim: {model.embed_dim if hasattr(model, 'embed_dim') else 'N/A'}")
        print(f"   - Num features: {model.num_features if hasattr(model, 'num_features') else 'N/A'}")
        print(f"   - Patch size: {model.patch_embed.patch_size if hasattr(model, 'patch_embed') else 'N/A'}")
        print(f"   - Num heads: {model.num_heads if hasattr(model, 'num_heads') else 'N/A'}")
        print(f"   - Depth (layers): {len(model.blocks) if hasattr(model, 'blocks') else 'N/A'}")
        
        # 检查register tokens
        print("\n3. Register Tokens 配置:")
        if hasattr(model, 'num_register_tokens'):
            print(f"   - num_register_tokens: {model.num_register_tokens}")
        else:
            print(f"   - num_register_tokens: 属性不存在")
            
        if hasattr(model, 'register_tokens'):
            if model.register_tokens is not None:
                print(f"   - register_tokens shape: {model.register_tokens.shape}")
            else:
                print(f"   - register_tokens: None (没有register tokens)")
        else:
            print(f"   - register_tokens: 属性不存在")
        
        # 检查关键参数
        print("\n4. 位置编码和Token配置:")
        if hasattr(model, 'cls_token'):
            print(f"   - cls_token shape: {model.cls_token.shape}")
        if hasattr(model, 'pos_embed'):
            print(f"   - pos_embed shape: {model.pos_embed.shape}")
            pos_embed_size = model.pos_embed.shape[1]
            print(f"   - pos_embed tokens数量: {pos_embed_size}")
            # 计算patch数量
            patch_tokens = pos_embed_size - 1  # 减去cls_token
            import math
            grid_size = int(math.sqrt(patch_tokens))
            print(f"   - Patch tokens: {patch_tokens} (grid: {grid_size}×{grid_size})")
        
        # 检查interpolate相关属性
        print("\n5. Interpolate 相关配置:")
        if hasattr(model, 'interpolate_antialias'):
            print(f"   - interpolate_antialias: {model.interpolate_antialias}")
        else:
            print(f"   - interpolate_antialias: 属性不存在")
            
        if hasattr(model, 'interpolate_offset'):
            print(f"   - interpolate_offset: {model.interpolate_offset}")
        else:
            print(f"   - interpolate_offset: 属性不存在")
        
        # 检查其他重要配置
        print("\n6. 其他配置:")
        if hasattr(model, 'qkv_bias'):
            print(f"   - qkv_bias: {model.qkv_bias}")
        if hasattr(model, 'proj_bias'):
            print(f"   - proj_bias: {model.proj_bias}")
        
        # 检查第一个block的配置
        if hasattr(model, 'blocks') and len(model.blocks) > 0:
            first_block = model.blocks[0]
            print(f"\n7. Block配置 (第一个block):")
            if hasattr(first_block, 'ls1'):
                print(f"   - 有 LayerScale (ls1)")
                if hasattr(first_block.ls1, 'gamma'):
                    print(f"   - ls1.gamma初始值: {first_block.ls1.gamma.data[0].item():.2e}")
            if hasattr(first_block.attn, 'qkv'):
                qkv = first_block.attn.qkv
                has_bias = qkv.bias is not None
                print(f"   - qkv has bias: {has_bias}")
        
        # 测试前向传播
        print("\n8. 测试前向传播:")
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            try:
                output = model(dummy_input)
                print(f"   ✓ 前向传播成功！")
                print(f"   - 输出shape: {output.shape}")
            except Exception as e:
                print(f"   ✗ 前向传播失败: {e}")
        
        # 检查state_dict的keys
        print("\n9. State Dict Keys (前20个):")
        state_dict = model.state_dict()
        keys = list(state_dict.keys())
        for i, key in enumerate(keys[:20]):
            shape = state_dict[key].shape
            print(f"   {i+1:2d}. {key:50s} {tuple(shape)}")
        print(f"   ... (总共 {len(keys)} 个参数)")
        
        # 检查是否有register相关的keys
        register_keys = [k for k in keys if 'regis' in k.lower()]
        if register_keys:
            print(f"\n   发现 register 相关的keys:")
            for k in register_keys:
                print(f"   - {k}: {state_dict[k].shape}")
        else:
            print(f"\n   ✓ 没有发现 register 相关的keys")
        
    except Exception as e:
        print(f"\n✗ 加载失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    check_uni_weights()

