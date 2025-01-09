import yaml
import logging
import os
from pathlib import Path
import numpy as np
from microstructure import MicrostructureGenerator
from material_properties import MaterialProperties
from simulation import MicroMagneticSimulation
from visualization import Visualizer
from config_validator import validate_config
import sys

def setup_directories():
    """创建必要的目录结构"""
    # 获取项目根目录（main.py所在目录的父目录）
    project_root = Path(__file__).parent.parent
    
    # 创建所需目录
    for dir_name in ['logs', 'results', 'config']:
        dir_path = project_root / dir_name
        dir_path.mkdir(exist_ok=True)
        
    return project_root

def setup_logging(project_root):
    """配置日志系统"""
    log_file = project_root / 'logs' / 'simulation.log'
    
    # 创建并配置处理器
    file_handler = logging.FileHandler(str(log_file))
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 为处理器设置格式化器
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 配置根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # 配置各模块的日志级别
    logging.getLogger('simulation').setLevel(logging.DEBUG)
    logging.getLogger('visualization').setLevel(logging.INFO)
    logging.getLogger('microstructure').setLevel(logging.INFO)
    # 设置matplotlib的日志级别为INFO，过滤掉DEBUG信息
    logging.getLogger('matplotlib').setLevel(logging.INFO)
    
    # 返回主模块的日志器
    logger = logging.getLogger(__name__)
    return logger

def load_config(project_root):
    """加载配置文件"""
    config_path = project_root / 'config' / 'config.yaml'
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"配置文件未找到: {config_path}")

def preprocess_config(config):
    """预处理配置，确保数值类型正确"""
    try:
        # 转换网格尺寸为整数
        config['mesh_size'] = [int(x) for x in config['mesh_size']]
        
        # 转换物理尺寸为浮点数
        config['size'] = [float(x) for x in config['size']]
        config['grain_size'] = float(config['grain_size'])
        config['boundary_thickness'] = float(config['boundary_thickness'])
        
        # 转换材料参数
        for material in ['Nd2Fe14B', 'NdFe2']:
            for key in ['saturation_magnetization', 'exchange_constant', 
                       'anisotropy_constant', 'damping_constant']:
                config['materials'][material][key] = float(
                    config['materials'][material][key]
                )
        
        # 转换温度参数
        if 'temperature' in config:
            config['temperature']['T'] = float(config['temperature']['T'])
            config['temperature']['kB'] = float(config['temperature']['kB'])
            config['temperature']['dt'] = float(config['temperature']['dt'])
            config['temperature']['seed'] = int(config['temperature']['seed'])
        
        # 转换磁滞回线参数
        if 'hysteresis' in config:
            config['hysteresis']['H_max'] = float(config['hysteresis']['H_max'])
            config['hysteresis']['n_points'] = int(config['hysteresis']['n_points'])
            config['hysteresis']['steady_tolerance'] = float(config['hysteresis']['steady_tolerance'])
            config['hysteresis']['direction'] = [float(x) for x in config['hysteresis']['direction']]
        
        # 转换可视化参数
        if 'visualization' in config and 'magnetization' in config['visualization']:
            config['visualization']['magnetization']['quiver_step'] = int(
                config['visualization']['magnetization']['quiver_step']
            )
            config['visualization']['magnetization']['animation_fps'] = int(
                config['visualization']['magnetization']['animation_fps']
            )
            config['visualization']['magnetization']['dpi'] = int(
                config['visualization']['magnetization']['dpi']
            )
        
        return config
        
    except (ValueError, TypeError) as e:
        logger.error(f"配置预处理失败: {str(e)}")
        raise

def main():
    try:
        # 设置项目目录
        project_root = setup_directories()
        
        # 设置日志
        logger = setup_logging(project_root)
        logger.info("日志系统初始化成功")
        
        # 加载配置
        config = load_config(project_root)
        logger.info("配置文件加载成功")
        
        # 预处理配置（先进行类型转换）
        config = preprocess_config(config)
        logger.info("配置预处理完成")
        
        # 验证配置（验证转换后的值）
        validate_config(config)
        logger.info("配置验证通过")
        
        # 生成微观结构
        generator = MicrostructureGenerator(config)
        structure = generator.generate_structure()
        logger.info("微观结构生成完成")
        
        # 初始化材料属性
        materials = MaterialProperties(config)
        
        # 初始化可视化工具
        visualizer = Visualizer(config)
        
        # 绘制微观结构
        visualizer.plot_microstructure(structure)
        logger.info("微观结构可视化完成")
        
        # 初始化微磁学模拟
        simulation = MicroMagneticSimulation(config, structure, materials)
        
        # 计算磁滞回线
        logger.info("开始计算磁滞回线...")
        H_max = config['hysteresis']['H_max']
        H_history, M_history = simulation.sweep_external_field(
            H_max,
            direction=config['hysteresis']['direction'],
            visualizer=visualizer
        )
        
        # 保存最终的静态磁滞回线图
        visualizer.plot_hysteresis(
            H_history,
            M_history,
            save_path='results/hysteresis_static.png'
        )
        
        logger.info("磁滞回线计算完成")
        
        # 输出磁化强度分布
        # 1. 保存最终状态的详细分布
        visualizer.plot_magnetization_distribution(
            simulation.M,
            time=config['time']['total'],
            save_path='results/magnetization_final.png'
        )
        
        # 2. 生成磁化强度演化动画
        visualizer.animate_magnetization(
            M_history,
            np.linspace(0, config['time']['total'], len(M_history)),
            save_path='results/magnetization_evolution.gif'
        )
        
    except Exception as e:
        logger.error(f"程序运行出错: {str(e)}")
        raise

if __name__ == "__main__":
    main() 