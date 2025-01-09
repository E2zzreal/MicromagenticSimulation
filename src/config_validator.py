import logging

logger = logging.getLogger(__name__)

def validate_config(config):
    """验证配置文件的完整性和有效性"""
    try:
        required_fields = {
            'size': (list, 2),
            'grain_size': (float, None),
            'boundary_thickness': (float, None),
            'mesh_size': (list, 2),
            'materials': {
                'Nd2Fe14B': {
                    'saturation_magnetization': (float, None),
                    'exchange_constant': (float, None),
                    'anisotropy_constant': (float, None),
                    'damping_constant': (float, None)
                },
                'NdFe2': {
                    'saturation_magnetization': (float, None),
                    'exchange_constant': (float, None),
                    'anisotropy_constant': (float, None),
                    'damping_constant': (float, None)
                }
            },
            'temperature': {
                'enabled': (bool, None),
                'T': (float, None),
                'kB': (float, None),
                'dt': (float, None),
                'seed': (int, None)
            },
            'hysteresis': {
                'H_max': (float, None),
                'n_points': (int, None),
                'direction': (list, 3),
                'steady_tolerance': (float, None)
            }
        }
        
        _validate_structure(config, required_fields)
        _validate_values(config)
        
        logger.info("配置验证通过")
        return True
        
    except Exception as e:
        logger.error(f"配置验证失败: {str(e)}")
        raise

def _validate_structure(config, required):
    """验证配置结构"""
    for key, value in required.items():
        if key not in config:
            raise ValueError(f"缺少必要的配置项: {key}")
            
        if isinstance(value, dict):
            _validate_structure(config[key], value)
        else:
            expected_type, length = value
            # 尝试转换数值类型
            if expected_type in (int, float):
                try:
                    if isinstance(config[key], (list, tuple)):
                        config[key] = [expected_type(x) for x in config[key]]
                    else:
                        config[key] = expected_type(config[key])
                except (ValueError, TypeError):
                    raise TypeError(f"配置项 {key} 无法转换为 {expected_type}")
            
            if not isinstance(config[key], (expected_type, list)):
                raise TypeError(f"配置项 {key} 类型错误，应为 {expected_type}")
            if length and len(config[key]) != length:
                raise ValueError(f"配置项 {key} 长度应为 {length}")

def _validate_values(config):
    """验证配置值的有效性"""
    try:
        # 检查尺寸参数
        if any(x <= 0 for x in config['size']):
            raise ValueError("模拟尺寸必须大于0")
        if config['grain_size'] <= 0:
            raise ValueError("晶粒尺寸必须大于0")
        if config['boundary_thickness'] <= 0:
            raise ValueError("晶界厚度必须大于0")
        if any(x <= 0 for x in config['mesh_size']):
            raise ValueError("网格尺寸必须大于0")
            
        # 检查材料参数
        for material in ['Nd2Fe14B', 'NdFe2']:
            props = config['materials'][material]
            if props['saturation_magnetization'] <= 0:
                raise ValueError(f"{material}的饱和磁化强度必须大于0")
            if props['exchange_constant'] <= 0:
                raise ValueError(f"{material}的交换常数必须大于0")
            if props['damping_constant'] <= 0 or props['damping_constant'] >= 1:
                raise ValueError(f"{material}的阻尼常数必须在0到1之间")
                
        # 检查温度参数
        if config['temperature']['enabled']:
            if config['temperature']['T'] < 0:
                raise ValueError("温度必须大于等于0K")
            if config['temperature']['dt'] <= 0:
                raise ValueError("时间步长必须大于0")
                
    except Exception as e:
        logger.error(f"配置值验证失败: {str(e)}")
        raise 