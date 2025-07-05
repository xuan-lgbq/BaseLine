import torch

def get_parameter_ordering(model):
    """获取模型参数的有序排列信息"""
    param_info = []
    total_params = 0
    
    for layer_idx, (name, param) in enumerate(model.named_parameters()):
        if param.requires_grad:
            param_count = param.numel()
            param_info.append({
                'layer_idx': layer_idx,
                'param_name': name,
                'start_idx': total_params,
                'end_idx': total_params + param_count,
                'param_count': param_count,
                'shape': param.shape,
                'is_weight': 'weight' in name,
                'is_bias': 'bias' in name
            })
            total_params += param_count
    
    return param_info, total_params

def flatten_model_parameters(model):
    """将模型参数展平为一维向量"""
    params = []
    for param in model.parameters():
        if param.requires_grad:
            params.append(param.view(-1))
    return torch.cat(params)

def unflatten_to_model_parameters(flat_params, model):
    """将一维向量重新组织为模型参数形状"""
    params_dict = {}
    start_idx = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count = param.numel()
            param_data = flat_params[start_idx:start_idx + param_count]
            params_dict[name] = param_data.view(param.shape)
            start_idx += param_count
    
    return params_dict

def group_parameters_by_layer(model):
    """按层分组参数"""
    layer_groups = {}
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            # 提取层名（去掉weight/bias后缀）
            layer_name = '.'.join(name.split('.')[:-1])
            
            if layer_name not in layer_groups:
                layer_groups[layer_name] = {}
            
            param_type = name.split('.')[-1]  # weight 或 bias
            layer_groups[layer_name][param_type] = param
    
    return layer_groups