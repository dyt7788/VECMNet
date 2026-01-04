# ---- 需要: pip install "ptflops==0.6.10"  (旧torch更稳) ----
    from ptflops import get_model_complexity_info
    import torch
    
    # 取一个 batch 示例
    sample_batch = next(iter(train_loader))
    images = sample_batch['images']            # [B, 3, H, W]  (half on cuda)
    caption_ids = sample_batch['caption_ids']  # [B, L]       (long)
    
    # 子批，避免显存/内存过大
    images = images[:1]
    caption_ids = caption_ids[:1]
    
    # 记录并在结束时恢复 base_model 的设备与精度
    base = model.base_model
    orig_device = next(base.parameters()).device
    orig_dtype  = next(base.parameters()).dtype
    
    try:
        # 1) 统计阶段：移到CPU，转float32，eval()
        base_cpu = base.cpu().float().eval()
    
        # 输入也转到CPU/float32（caption保持long）
        img_cpu = images.detach().cpu().float()
        cap_cpu = caption_ids.detach().cpu().long()
    
        # 2) 封装 wrapper：ptflops 仅支持单输入，这里固定 caption_ids
        class ModelWrapper(torch.nn.Module):
            def __init__(self, base_model, fixed_caps):
                super().__init__()
                self.model = base_model
                self.fixed_caps = fixed_caps
            def forward(self, x):
                # x: image tensor on CPU/float32
                return self.model(x, self.fixed_caps)
    
        wrapper = ModelWrapper(base_cpu, cap_cpu)
    
        # 3) 输入尺寸按当前图片张量自动取
        input_res = tuple(img_cpu.shape[1:])   # (3, H, W)
    
        # 4) 统计（关闭逐层打印，得到字符串形式）
        flops_str, params_str = get_model_complexity_info(
            wrapper, input_res,
            as_strings=True,
            print_per_layer_stat=False,
            verbose=False
        )
    
        # 可选：转成数字方便表格（'23.58 GMac' -> 23.58, '192.31 M' -> 192.31）
        def _to_num(s):
            try:
                return float(s.split()[0])
            except Exception:
                return None
        flops_g  = _to_num(flops_str)   # 单位 G
        params_m = _to_num(params_str)  # 单位 M
    
        # 输出：用 logger 或 print 都可
        try:
            logger.info(f"FLOPs (G): {flops_str} | Params (M): {params_str}")
        except NameError:
            print('='*60)
            print('Model Complexity Summary (ptflops on CPU/FP32):')
            print(f'Params: {params_str}')
            print(f'FLOPs : {flops_str}')
            print('='*60)
    
    finally:
        # 5) 还原 base_model 的设备与精度（回到原来的 half + cuda）
        base.to(orig_device)
        if orig_dtype == torch.float16:
            base.half()
        else:
            base.to(orig_dtype)
        base.train()  # 恢复训练态（若你当时在训练）