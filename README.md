# Nd2FeB永磁体微磁学模拟

本项目用于模拟烧结Nd2FeB永磁体的微磁学特性，采用Landau-Lifshitz-Gilbert (LLG)方程进行数值模拟。

## 主要功能
- 微结构生成：创建包含主相和晶界相的Nd2FeB微结构
- 材料属性计算：计算温度依赖的磁化强度、各向异性常数等
- 微磁学模拟：求解LLG方程，计算有效场（交换场、退磁场、各向异性场等）
- 可视化：提供磁化分布、磁滞回线等可视化功能

## 项目结构
- `config/`: 配置文件目录
- `src/`: 源代码目录
  - `config_validator.py`: 配置验证
  - `main.py`: 主程序
  - `material_properties.py`: 材料属性计算
  - `microstructure.py`: 微结构生成
  - `simulation.py`: 微磁学模拟
  - `visualization.py`: 可视化工具

## 使用方法
1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

2. 配置模拟参数：
   修改`config/config.yaml`文件中的参数

3. 运行模拟：
   ```bash
   python src/main.py
   ```

4. 查看结果：
   - 模拟结果保存在`results/`目录
   - 使用`visualization.py`中的工具进行可视化

## 示例
```python
from src.simulation import MicroMagneticSimulation
from src.microstructure import MicrostructureGenerator
from src.material_properties import MaterialProperties

# 初始化
config = load_config()
microstructure = MicrostructureGenerator(config).generate_structure()
material = MaterialProperties(config)

# 创建模拟器
sim = MicroMagneticSimulation(config, microstructure, material)

# 运行模拟
H_range = np.linspace(-2e6, 2e6, 100)  # 外场范围
M_history = sim.sweep_external_field(H_range, direction=[0, 0, 1])
