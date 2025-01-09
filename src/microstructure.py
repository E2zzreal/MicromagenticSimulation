import numpy as np
from scipy.spatial import Voronoi
import logging

logger = logging.getLogger(__name__)

class MicrostructureGenerator:
    def __init__(self, config):
        self.config = config
        self.size = [float(config['size'][0]), float(config['size'][1])]
        self.grain_size = float(config['grain_size'])
        self.boundary_thickness = float(config['boundary_thickness'])
        self.logger = logging.getLogger('microstructure')

    def generate_structure(self):
        """生成二维微观结构"""
        try:
            # 计算所需的晶粒数量
            area = self.size[0] * self.size[1]
            grain_area = np.pi * (self.grain_size/2)**2
            n_grains = int(1.5 * area / grain_area)  # 1.5为补偿系数
            
            logger.info("开始生成微观结构...")
            
            # 在扩展区域内生成随机点
            margin = self.grain_size  # 添加边界余量
            extended_size = [self.size[0] + 2*margin, self.size[1] + 2*margin]
            logger.info("设置边界完成")
            
            # 生成随机点作为晶粒中心
            points = []
            inner_points = np.random.rand(n_grains, 2) * self.size
            points.extend(inner_points)
            logger.info("生成晶粒中心点完成")
            
            # 在边界区域添加镜像点以改善边界晶粒的形状
            logger.info("开始添加镜像点...")
            for x in inner_points:
                # 左右镜像
                if x[0] < margin:
                    points.append([x[0] - margin, x[1]])
                if x[0] > self.size[0] - margin:
                    points.append([x[0] + margin, x[1]])
                    
                # 上下镜像
                if x[1] < margin:
                    points.append([x[0], x[1] - margin])
                if x[1] > self.size[1] - margin:
                    points.append([x[0], x[1] + margin])
                    
                # 角落镜像
                if x[0] < margin and x[1] < margin:
                    points.append([x[0] - margin, x[1] - margin])
                if x[0] < margin and x[1] > self.size[1] - margin:
                    points.append([x[0] - margin, x[1] + margin])
                if x[0] > self.size[0] - margin and x[1] < margin:
                    points.append([x[0] + margin, x[1] - margin])
                if x[0] > self.size[0] - margin and x[1] > self.size[1] - margin:
                    points.append([x[0] + margin, x[1] + margin])
            
            points = np.array(points)
            logger.info("镜像点添加完成")
            
            # 使用Voronoi图生成晶粒结构
            vor = Voronoi(points)
            
            # 记录实际的晶粒数量
            n_actual_grains = len(inner_points)
            logger.info(f"成功生成微观结构，包含 {n_actual_grains} 个晶粒")
            
            # 保存额外信息以供后续使用
            vor.actual_size = self.size
            vor.inner_points = inner_points
            vor.grain_size = self.grain_size
            vor.boundary_thickness = self.boundary_thickness
            
            return vor
            
        except Exception as e:
            logger.error(f"生成微观结构时发生错误: {str(e)}")
            raise 