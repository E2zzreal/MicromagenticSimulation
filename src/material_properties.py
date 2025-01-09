import numpy as np
import logging

logger = logging.getLogger(__name__)

class MaterialProperties:
    def __init__(self, config):
        self.config = config
        self.Nd2Fe14B = config['materials']['Nd2Fe14B']
        self.NdFe2 = config['materials']['NdFe2']
        self.temperature = config.get('temperature', {})
        
    def get_local_properties(self, position, structure):
        """根据位置确定局部材料属性"""
        # 判断位置是否在晶界上
        if self._is_boundary(position, structure):
            return self.NdFe2
        return self.Nd2Fe14B
        
    def _is_boundary(self, position, structure):
        """判断给定位置是否位于晶界"""
        try:
            boundary_thickness = self.config['boundary_thickness']
            vor = structure
            
            # 找到最近的两个晶粒中心点
            distances = []
            for point in vor.points:
                dist = np.sqrt(np.sum((position - point)**2))
                distances.append(dist)
            
            # 获取最近的两个距离
            nearest_distances = sorted(distances)[:2]
            
            # 如果两个最近晶粒的距离差小于晶界厚度，则认为在晶界上
            if abs(nearest_distances[1] - nearest_distances[0]) < boundary_thickness:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"晶界判断时发生错误: {str(e)}")
            raise

    def get_exchange_constant(self):
        """获取交换常数"""
        A = float(self.Nd2Fe14B['exchange_constant'])
        if A <= 0:
            raise ValueError("交换常数必须大于0")
        return A

    def get_anisotropy_constant(self):
        """获取各向异性常数"""
        return self.Nd2Fe14B['anisotropy_constant']

    def get_saturation_magnetization(self):
        """获取饱和磁化强度"""
        Ms = float(self.Nd2Fe14B['saturation_magnetization'])
        if Ms <= 0:
            raise ValueError("饱和磁化强度必须大于0")
        return Ms

    def get_damping_constant(self):
        """获取阻尼常数"""
        return self.Nd2Fe14B['damping_constant']

    def get_temperature_dependent_Ms(self, T):
        """获取温度依赖的饱和磁化强度"""
        Ms_0 = self.Nd2Fe14B['saturation_magnetization']
        Tc = 585  # 居里温度（K）
        beta = 0.36  # 临界指数
        
        # Bloch定律
        if T >= Tc:
            return 0
        return Ms_0 * (1 - (T/Tc)**beta)**(1/3)
    
    def get_temperature_dependent_K1(self, T):
        """获取温度依赖的磁晶各向异性常数"""
        K1_0 = self.Nd2Fe14B['anisotropy_constant']
        Tc = 585
        n = 3  # 各向异性温度依赖指数
        
        if T >= Tc:
            return 0
        return K1_0 * (1 - T/Tc)**n 