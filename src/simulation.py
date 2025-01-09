import numpy as np
import logging
from scipy.integrate import solve_ivp
import time
import psutil
from numba import jit
from multiprocessing import Pool

# 设置 Numba 日志级别为 WARNING，抑制 DEBUG 信息
from numba.core.errors import NumbaWarning
import warnings
warnings.simplefilter('ignore', category=NumbaWarning)

# 设置 numba logger 级别
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)

# 添加物理常数
gamma = 2.21e5  # 旋磁比，单位：m/(A·s)

logger = logging.getLogger(__name__)

class MicroMagneticSimulation:
    def __init__(self, config, microstructure, material_properties):
        """初始化微磁模拟"""
        self.config = config
        self.microstructure = microstructure
        self.material_properties = material_properties
        
        # 确保网格尺寸为整数
        self.mesh_size = [int(x) for x in config['mesh_size']]
        
        # 初始化磁化强度场
        Ms = self.material_properties.get_saturation_magnetization()
        self.M = np.zeros((self.mesh_size[0], self.mesh_size[1], 3))
        
        # 添加小的随机扰动
        theta = np.random.normal(0, 0.1, self.M.shape[:2])
        phi = np.random.uniform(0, 2*np.pi, self.M.shape[:2])
        
        # 确保初始磁化强度场的形状正确
        self.M[:,:,0] = Ms * np.sin(theta) * np.cos(phi)
        self.M[:,:,1] = Ms * np.sin(theta) * np.sin(phi)
        self.M[:,:,2] = Ms * np.cos(theta)
        
        # 验证初始化
        logger.debug(f"初始化完成:")
        logger.debug(f"- 网格尺寸: {self.mesh_size}")
        logger.debug(f"- 磁化强度场形状: {self.M.shape}")
        logger.debug(f"- 饱和磁化强度: {Ms:.2e}")
        
        # 初始化外加场
        self.H_external = np.zeros_like(self.M)
        
        # 添加温度相关参数
        self.temperature = config.get('temperature', {})
        if self.temperature.get('enabled', False):
            self.T = self.temperature['T']
            self.kB = self.temperature['kB']
            self.dt = self.temperature['dt']
            # 设置随机数种子
            np.random.seed(self.temperature.get('seed', 42))
        
    def calculate_effective_field(self):
        """计算有效场"""
        try:
            H_exchange = self._calculate_exchange_field()
            H_anisotropy = self._calculate_anisotropy_field()
            H_demagnetizing = self._calculate_demagnetizing_field()
            
            # 计算总有效场
            H_eff = H_exchange + H_anisotropy + H_demagnetizing + self.H_external
            return H_eff
            
        except Exception as e:
            logger.error(f"有效场计算失败: {str(e)}")
            raise
    
    def _calculate_exchange_field(self):
        """计算交换场"""
        try:
            H_ex = np.zeros_like(self.M)
            A = self.material_properties.get_exchange_constant()
            Ms = self.material_properties.get_saturation_magnetization()
            
            dx = self.config['size'][0] / self.mesh_size[0] * 1e-9
            dy = self.config['size'][1] / self.mesh_size[1] * 1e-9
            
            # 使用numba加速的核心计算
            H_ex = _calculate_exchange_field_kernel(self.M, dx, dy, A, Ms)
            
            return H_ex
            
        except Exception as e:
            logger.error(f"计算交换场时发生错误: {str(e)}")
            raise
    
    def _calculate_anisotropy_field(self):
        """计算各向异性场"""
        try:
            K1 = self.material_properties.get_anisotropy_constant()  # 4.3e6 J/m³
            Ms = self.material_properties.get_saturation_magnetization()  # 1.61e6 A/m
            
            # 易磁化轴
            easy_axis = np.array([0, 0, 1])
            
            # 归一化磁化强度
            m = self.M / Ms
            
            # 计算与易轴的投影
            cos_theta = np.sum(m * easy_axis, axis=2, keepdims=True)
            
            # 各向异性场计算公式：Ha = 2K1/(μ0*Ms) * cos(theta)
            mu0 = 4 * np.pi * 1e-7  # 真空磁导率
            H_anis = 2 * K1 / (mu0 * Ms) * cos_theta * easy_axis
            
            return H_anis
            
        except Exception as e:
            logger.error(f"计算各向异性场时发生错误: {str(e)}")
            raise
    
    def _calculate_demagnetizing_field(self):
        """计算退磁场（顺序处理，避免广播错误）"""
        try:
            # 准备频率空间网格
            kx = 2 * np.pi * np.fft.fftfreq(self.mesh_size[0])
            ky = 2 * np.pi * np.fft.fftfreq(self.mesh_size[1])
            kx, ky = np.meshgrid(kx, ky, indexing='ij')
            k_squared = kx**2 + ky**2
            k_squared[k_squared == 0] = 1e-10  # 避免除以零
            
            # 计算三个方向的退磁场分量
            H_demag = np.zeros_like(self.M)
            for i in range(3):
                M_component = self.M[:, :, i]
                M_fft = np.fft.fftn(M_component, axes=(0, 1))
                H_demag[:, :, i] = np.fft.ifftn(-M_fft * k_squared / (k_squared + 1e-10)).real
            
            return H_demag
            
        except Exception as e:
            logger.error(f"计算退磁场时发生错误: {str(e)}")
            raise
    
    def _calculate_external_field(self):
        """计算外加场"""
        # 从配置中获取外加场设置
        return np.zeros_like(self.M)  # 暂时返回零场
    
    def _calculate_thermal_field(self):
        """计算热场（朗之万随机场）"""
        try:
            Ms = self.material_properties.get_saturation_magnetization()
            alpha = self.material_properties.get_damping_constant()
            V = self._calculate_cell_volume()  # 计算单个网格单元体积
            
            # 计算热场强度
            D = np.sqrt((2 * self.kB * self.T * alpha) / 
                       (self.dt * gamma * Ms * V * (1 + alpha**2)))
            
            # 生成高斯白噪声
            H_thermal = np.random.normal(0, D, size=self.M.shape)
            
            logger.debug(f"热场强度: {np.mean(np.abs(H_thermal))} A/m")
            return H_thermal
            
        except Exception as e:
            logger.error(f"计算热场时发生错误: {str(e)}")
            raise
    
    def _calculate_cell_volume(self):
        """计算单个网格单元的体积"""
        dx = self.config['size'][0] / self.mesh_size[0]
        dy = self.config['size'][1] / self.mesh_size[1]
        # 假设z方向厚度为dx（二维模拟）
        return dx * dy * dx
    
    def solve_LLG(self, t_span, t_eval):
        """求解LLG方程"""
        try:
            def LLG(t, m):
                # 将一维数组重塑为三维数组
                m_reshaped = m.reshape(self.mesh_size[0], self.mesh_size[1], 3)
                self.M = m_reshaped  # 更新当前状态
                
                # 计算有效场
                H_eff = self.calculate_effective_field()
                
                # 计算LLG方程右端项
                prec_term = -gamma * np.cross(m_reshaped, H_eff)
                damp_term = -self.material_properties.get_damping_constant() * gamma * np.cross(
                    m_reshaped, np.cross(m_reshaped, H_eff)
                )
                
                # 更新进度（根据当前时间估算）
                if hasattr(self, '_last_t'):
                    if t - self._last_t >= 0.1 * (t_span[1] - t_span[0]):  # 每10%更新一次
                        progress = (t - t_span[0]) / (t_span[1] - t_span[0]) * 100
                        logger.info(f"LLG求解进度: {progress:.1f}%")
                        self._last_t = t
                else:
                    self._last_t = t
                
                return (prec_term + damp_term).reshape(-1)
            
            logger.info(f"开始求解LLG方程: {t_span[0]:.2e}s -> {t_span[1]:.2e}s")
            start_time = time.time()
            
            solution = solve_ivp(
                LLG,
                t_span,
                self.M.reshape(-1),
                t_eval=t_eval,
                method='RK45',
                rtol=1e-6,
                atol=1e-6,
                max_step=1e-11,
                vectorized=False
            )
            
            # 清理临时变量
            if hasattr(self, '_last_t'):
                delattr(self, '_last_t')
            
            # 计算性能统计
            elapsed_time = time.time() - start_time
            steps_per_second = solution.nfev / elapsed_time
            
            logger.info(f"LLG方程求解完成:")
            logger.info(f"- 耗时: {elapsed_time:.2f}s")
            logger.info(f"- 函数评估: {solution.nfev}次 ({steps_per_second:.1f}步/秒)")
            logger.info(f"- 实际时间步数: {len(solution.t)}")
            logger.info(f"- 求解器状态: {solution.message}")
            
            return solution
            
        except Exception as e:
            logger.error(f"求解LLG方程时发生错误: {str(e)}")
            raise
    
    def solve_LLG_stochastic(self, t_span, t_eval):
        """求解随机LLG方程（包含温度效应）"""
        try:
            M0 = self.M.reshape(-1)
            
            def SLLG(t, m):
                """随机LLG方程右端项"""
                m = m.reshape(self.M.shape)
                H_eff = self.calculate_effective_field()  # 包含热场
                
                # 计算LLG方程
                alpha = self.material_properties.get_damping_constant()
                gamma = 2.21e5  # 旋磁比
                
                dm_dt = -gamma * np.cross(m, H_eff) + \
                        alpha * np.cross(m, np.cross(m, H_eff))
                
                # 归一化以保持磁矩模长
                m_norm = np.linalg.norm(m, axis=2, keepdims=True)
                # 避免除零，同时保持维度一致
                mask = (m_norm > 1e-10)[..., 0]  # 移除多余的维度
                m[mask] = m[mask] / m_norm[mask]
                
                return dm_dt.reshape(-1)
            
            # 使用Heun方法（随机微分方程的二阶方法）
            solution = self._heun_solver(
                SLLG,
                t_span,
                M0,
                self.temperature['dt']
            )
            
            logger.info("随机LLG方程求解完成")
            return solution
            
        except Exception as e:
            logger.error(f"求解随机LLG方程时发生错误: {str(e)}")
            raise
    
    def _heun_solver(self, func, t_span, y0, dt):
        """Heun方法求解随机微分方程"""
        t = t_span[0]
        y = y0
        results = [(t, y.copy())]
        
        while t < t_span[1]:
            # 预测步
            k1 = func(t, y)
            y_pred = y + dt * k1
            
            # 校正步
            k2 = func(t + dt, y_pred)
            y_new = y + 0.5 * dt * (k1 + k2)
            
            t += dt
            y = y_new
            results.append((t, y.copy()))
        
        return np.array(results) 
    
    def sweep_external_field(self, H_range, direction, visualizer=None):
        """扫描外加场"""
        try:
            # 确保 direction 是 numpy 数组并进行归一化
            direction = np.array(direction, dtype=np.float64)
            direction = direction / np.linalg.norm(direction)
            
            H_history = []
            M_history = []
            
            # 定义场扫描步骤
            field_steps = [
                (np.linspace(0, H_range, 100), "增加外场至最大值"),
                (np.linspace(H_range, -H_range, 100), "降低外场至最小值"),
                (np.linspace(-H_range, H_range, 50), "恢复外场至最大值")
            ]
            
            # 计算总步数和初始化进度追踪
            total_steps = sum(len(H_values) for H_values, _ in field_steps)
            current_step = 0
            start_time = time.time()
            last_update_time = start_time
            update_interval = 2.0  # 每2秒更新一次进度
            
            logger.info("开始磁滞回线计算")
            logger.info(f"总步数: {total_steps}")
            
            for phase_idx, (H_values, phase_name) in enumerate(field_steps):
                phase_start_time = time.time()
                logger.info(f"\n阶段 {phase_idx + 1}/3: {phase_name}")
                
                for H_idx, H in enumerate(H_values):
                    # 计算外加场向量
                    H_vector = H * direction
                    self._set_external_field(H_vector)
                    
                    # 求解当前步
                    solution = self.solve_to_steady_state()
                    
                    # 记录结果
                    M_avg = np.mean(self.M, axis=(0,1))
                    H_history.append(H)
                    M_history.append(M_avg)
                    
                    # 更新进度
                    current_step += 1
                    current_time = time.time()
                    
                    # 每隔一定时间更新进度信息
                    if current_time - last_update_time >= update_interval:
                        # 计算进度
                        progress = current_step / total_steps
                        elapsed_time = current_time - start_time
                        phase_elapsed = current_time - phase_start_time
                        
                        # 估算剩余时间
                        steps_remaining = total_steps - current_step
                        time_per_step = elapsed_time / current_step
                        eta = steps_remaining * time_per_step
                        
                        # 计算当前阶段进度
                        phase_progress = (H_idx + 1) / len(H_values)
                        
                        # 输出进度信息
                        logger.info(
                            f"进度: {progress*100:.1f}% "
                            f"[{current_step}/{total_steps}] "
                            f"当前阶段: {phase_progress*100:.1f}% "
                            f"已用时间: {elapsed_time:.1f}s "
                            f"预计剩余: {eta:.1f}s"
                        )
                        
                        # 输出当前磁化状态
                        M_proj = np.dot(M_avg, direction)
                        logger.info(
                            f"外场: {H:.2e} A/m "
                            f"磁化强度: {M_proj:.2e} A/m"
                        )
                        
                        last_update_time = current_time
                
                # 输出阶段完成信息
                phase_time = time.time() - phase_start_time
                logger.info(
                    f"阶段 {phase_idx + 1} 完成 "
                    f"耗时: {phase_time:.1f}s "
                    f"平均每步: {phase_time/len(H_values):.3f}s"
                )
            
            # 输出总体完成信息
            total_time = time.time() - start_time
            logger.info("\n磁滞回线计算完成:")
            logger.info(f"- 总耗时: {total_time:.1f}s")
            logger.info(f"- 平均每步: {total_time/total_steps:.3f}s")
            logger.info(f"- 总步数: {total_steps}")
            
            return np.array(H_history), np.array(M_history)
            
        except Exception as e:
            logger.error(f"扫描外加场时发生错误: {str(e)}")
            raise
    
    def _set_external_field(self, H_ext):
        """设置外加场"""
        try:
            # 确保 H_ext 是 numpy 数组
            H_ext = np.array(H_ext, dtype=np.float64)
            
            # 检查维度
            if H_ext.shape != (3,):
                raise ValueError(f"外加场维度错误: {H_ext.shape}, 应为 (3,)")
            
            # 广播到整个网格
            self.H_external = np.broadcast_to(H_ext, self.M.shape)
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"外加场设置完成: {H_ext}, 模值: {np.linalg.norm(H_ext):.2e}")
            
        except Exception as e:
            logger.error(f"设置外加场时发生错误: {str(e)}")
            raise
    
    def solve_to_steady_state(self, max_time=5e-9, tolerance=1e-2):
        """求解直到达到稳态"""
        try:
            t_span = (0, max_time)
            
            # 记录初始状态
            Ms = self.material_properties.get_saturation_magnetization()
            initial_M_avg = np.mean(np.abs(self.M), axis=(0,1))
            logger.info(f"初始磁化状态: [{initial_M_avg[0]:.2e}, {initial_M_avg[1]:.2e}, {initial_M_avg[2]:.2e}] A/m")
            
            # 记录计算开始时间和内存使用
            start_time = time.time()
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # 求解LLG方程
            if self.temperature.get('enabled', False):
                solution = self.solve_LLG_stochastic(t_span, self.temperature['dt'])
            else:
                t_eval = np.linspace(0, max_time, 500)
                solution = self.solve_LLG(t_span, t_eval)
            
            # 检查稳态和磁矩守恒
            if len(solution.y.T) >= 3:
                # 检查稳态
                last_states = solution.y[:, -3:].reshape(3, *self.M.shape)
                state_diffs = [
                    np.max(np.abs(last_states[i+1] - last_states[i])) / Ms
                    for i in range(2)
                ]
                max_diff = max(state_diffs)
                avg_diff = sum(state_diffs) / len(state_diffs)
                
                # 检查磁矩守恒
                M_magnitude = np.linalg.norm(self.M, axis=2)
                magnitude_error = np.max(np.abs(M_magnitude - Ms)) / Ms
                
                # 计算性能指标
                elapsed_time = time.time() - start_time
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_used = current_memory - initial_memory
                
                # 输出关键信息
                logger.info("求解完成:")
                logger.info(f"- 稳态判据: {max_diff:.2e} (阈值: {tolerance:.2e})")
                logger.info(f"- 磁矩守恒: {magnitude_error:.2e}")
                logger.info(f"- 计算时间: {elapsed_time:.2f}s")
                logger.info(f"- 内存使用: {memory_used:.1f}MB")
                logger.info(f"- 函数评估: {solution.nfev}次")
                
                # 输出警告信息
                if max_diff > tolerance:
                    logger.warning(f"未达到稳态 (最大变化: {max_diff:.2e}, 平均变化: {avg_diff:.2e})")
                if magnitude_error > 1e-6:
                    logger.warning(f"磁矩模长误差较大: {magnitude_error:.2e}")
            
            # 更新磁化强度场
            self.M = solution.y[:, -1].reshape(self.M.shape)
            
            # 输出最终磁化状态
            final_M_avg = np.mean(np.abs(self.M), axis=(0,1))
            logger.info(f"最终磁化状态: [{final_M_avg[0]:.2e}, {final_M_avg[1]:.2e}, {final_M_avg[2]:.2e}] A/m")
            
            return solution
            
        except Exception as e:
            logger.error(f"求解稳态时发生错误: {str(e)}")
            raise 

    def _apply_periodic_boundary(self, field):
        """应用周期性边界条件"""
        field[0,:] = field[-2,:]  # 左边界
        field[-1,:] = field[1,:]  # 右边界
        field[:,0] = field[:,-2]  # 下边界
        field[:,-1] = field[:,1]  # 上边界
        return field

@jit(nopython=True)
def _calculate_exchange_field_kernel(M, dx, dy, A, Ms):
    """使用numba加速交换场计算"""
    H_ex = np.zeros_like(M)
    nx, ny, _ = M.shape
    
    # 使用周期性边界条件
    for i in range(nx):
        for j in range(ny):
            for d in range(3):
                # 计算x方向的二阶差分（周期性边界）
                ip1 = (i + 1) % nx
                im1 = (i - 1) % nx
                d2m_dx2 = (M[ip1,j,d] - 2*M[i,j,d] + M[im1,j,d]) / dx**2
                
                # 计算y方向的二阶差分（周期性边界）
                jp1 = (j + 1) % ny
                jm1 = (j - 1) % ny
                d2m_dy2 = (M[i,jp1,d] - 2*M[i,j,d] + M[i,jm1,d]) / dy**2
                
                # 交换场计算公式
                H_ex[i,j,d] = (2*A)/(Ms) * (d2m_dx2 + d2m_dy2)
    
    return H_ex 