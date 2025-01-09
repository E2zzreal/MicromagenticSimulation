import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import logging
import matplotlib as mpl

logger = logging.getLogger(__name__)

class Visualizer:
    def __init__(self, config):
        self.config = config
        # 使用默认样式
        plt.style.use('default')
        
        # 设置默认的绘图参数
        plt.rcParams.update({
            'figure.figsize': (10, 8),
            'axes.grid': True,
            'grid.linestyle': '--',
            'grid.alpha': 0.5,
            'font.size': 12,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'lines.linewidth': 2
        })
        
        # 移除所有字体设置
        # 初始化实时绘图窗口
        self.live_fig = None
        self.live_ax = None
        self.live_line = None
        self.live_point = None
        
    def plot_microstructure(self, structure):
        """绘制微观结构"""
        try:
            plt.figure(figsize=(10, 10))
            
            # 绘制Voronoi图
            vor = structure
            actual_size = vor.actual_size
            
            # 设置显示范围
            plt.xlim(0, actual_size[0])
            plt.ylim(0, actual_size[1])
            
            # 绘制晶界
            for simplex in vor.ridge_vertices:
                if simplex[0] >= 0 and simplex[1] >= 0:
                    vertices = vor.vertices[simplex]
                    # 只绘制在实际区域内的晶界
                    if (all(0 <= v[0] <= actual_size[0] for v in vertices) and
                        all(0 <= v[1] <= actual_size[1] for v in vertices)):
                        plt.plot(vertices[:, 0], vertices[:, 1], 'k-', linewidth=0.5)
            
            # 绘制晶粒中心点
            inner_points = vor.inner_points
            plt.plot(inner_points[:, 0], inner_points[:, 1], 'r.', markersize=3)
            
            # 使用英文标题和标签
            plt.title('NdFeB Permanent Magnet Microstructure')
            plt.xlabel('x (nm)')
            plt.ylabel('y (nm)')
            plt.axis('equal')
            
            # 添加比例尺
            scale_bar_length = vor.grain_size
            plt.plot([10, 10 + scale_bar_length], [10, 10], 'k-', linewidth=2)
            plt.text(10, 15, f'{int(scale_bar_length)} nm', ha='left', va='bottom')
            
            # 保存图像
            plt.savefig('results/microstructure.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Microstructure image saved")
            
        except Exception as e:
            logger.error(f"Error plotting microstructure: {str(e)}")
            raise
    
    def plot_magnetization(self, M):
        """绘制磁化强度分布"""
        try:
            fig, ax = plt.subplots(figsize=(10, 10))
            
            # 计算磁化强度大小
            M_magnitude = np.sqrt(np.sum(M**2, axis=2))
            
            # 绘制热图
            im = ax.imshow(M_magnitude.T, origin='lower', cmap='viridis')
            plt.colorbar(im, ax=ax, label='Magnetization (A/m)')
            
            # 添加磁化方向箭头
            step = 5  # 每隔5个点画一个箭头
            X, Y = np.meshgrid(range(M.shape[0]), range(M.shape[1]))
            ax.quiver(X[::step, ::step], Y[::step, ::step],
                     M[::step, ::step, 0], M[::step, ::step, 1],
                     angles='xy', scale_units='xy', scale=0.5)
            
            plt.title('Magnetization Distribution')
            plt.xlabel('x (Grid Points)')
            plt.ylabel('y (Grid Points)')
            
            plt.savefig('results/magnetization.png')
            plt.close()
            
            logger.info("Magnetization distribution image saved")
            
        except Exception as e:
            logger.error(f"Error plotting magnetization distribution: {str(e)}")
            raise 
    
    def animate_hysteresis(self, H_history, M_history, save_path=None):
        """创建磁滞回线动画"""
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            line, = ax.plot([], [], 'b-', lw=2)
            point, = ax.plot([], [], 'ro')
            
            # 设置图表
            ax.set_xlabel('External Field (A/m)')
            ax.set_ylabel('Magnetization (A/m)')
            ax.set_title('Hysteresis Loop')
            
            # 设置坐标轴范围
            ax.set_xlim(min(H_history)*1.1, max(H_history)*1.1)
            ax.set_ylim(min(M_history[:,0])*1.1, max(M_history[:,0])*1.1)
            
            # 添加网格
            ax.grid(True)
            
            def init():
                line.set_data([], [])
                point.set_data([], [])
                return line, point
            
            def update(frame):
                # 绘制到当前帧的轨迹
                line.set_data(H_history[:frame], M_history[:frame,0])
                # 绘制当前点
                point.set_data([H_history[frame]], [M_history[frame,0]])
                return line, point
            
            anim = FuncAnimation(fig, update, frames=len(H_history),
                               init_func=init, blit=True, interval=20)
            
            if save_path:
                anim.save(save_path, writer='pillow')
            else:
                plt.show()
                
            logger.info("磁滞回线动画生成完成")
            
        except Exception as e:
            logger.error(f"生成磁滞回线动画时发生错误: {str(e)}")
            raise
    
    def plot_hysteresis(self, H_history, M_history, save_path=None):
        """绘制静态磁滞回线"""
        try:
            plt.figure(figsize=(10, 8))
            plt.plot(H_history, M_history[:,0], 'b-', label='M-H curve')
            plt.xlabel('Applied Field H (A/m)')
            plt.ylabel('Magnetization M (A/m)')
            plt.title('Hysteresis Loop')
            plt.grid(True)
            plt.legend()
            
            if save_path:
                plt.savefig(save_path)
            else:
                plt.show()
                
            logger.info("Hysteresis loop image saved")
            
        except Exception as e:
            logger.error(f"Error plotting hysteresis loop: {str(e)}")
            raise 
    
    def plot_magnetization_distribution(self, M, time=None, save_path=None):
        """绘制详细的磁化强度分布"""
        try:
            fig = plt.figure(figsize=(15, 10))
            
            # 创建子图
            gs = plt.GridSpec(2, 3, figure=fig)
            ax1 = fig.add_subplot(gs[0, 0])  # 磁化强度大小
            ax2 = fig.add_subplot(gs[0, 1])  # x方向分量
            ax3 = fig.add_subplot(gs[0, 2])  # y方向分量
            ax4 = fig.add_subplot(gs[1, :])  # 矢量场图
            
            # 1. 磁化强度大小分布
            M_magnitude = np.sqrt(np.sum(M**2, axis=2))
            im1 = ax1.imshow(M_magnitude.T, origin='lower', cmap='viridis')
            plt.colorbar(im1, ax=ax1, label='|M| (A/m)')
            ax1.set_title('Magnetization Magnitude')
            
            # 2. x方向分量
            im2 = ax2.imshow(M[:,:,0].T, origin='lower', cmap='RdBu')
            plt.colorbar(im2, ax=ax2, label='Mx (A/m)')
            ax2.set_title('X Component')
            
            # 3. y方向分量
            im3 = ax3.imshow(M[:,:,1].T, origin='lower', cmap='RdBu')
            plt.colorbar(im3, ax=ax3, label='My (A/m)')
            ax3.set_title('Y Component')
            
            # 4. 矢量场图
            X, Y = np.meshgrid(range(M.shape[0]), range(M.shape[1]))
            step = 10  # 每隔10个点画一个箭头
            
            # 归一化矢量
            M_norm = M / np.linalg.norm(M, axis=2, keepdims=True)
            
            q = ax4.quiver(X[::step, ::step].T, Y[::step, ::step].T,
                          M_norm[::step, ::step, 0].T, M_norm[::step, ::step, 1].T,
                          M_magnitude[::step, ::step].T,
                          cmap='viridis',
                          angles='xy', scale_units='xy', scale=2)
            plt.colorbar(q, ax=ax4, label='|M| (A/m)')
            ax4.set_title('Magnetization Direction')
            
            # 设置总标题
            if time is not None:
                fig.suptitle(f'Magnetization Distribution (t = {time:.2e} s)', fontsize=16)
            else:
                fig.suptitle('Magnetization Distribution', fontsize=16)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
                
            logger.info(f"磁化强度分布图像已保存: {save_path}")
            
        except Exception as e:
            logger.error(f"绘制磁化强度分布时发生错误: {str(e)}")
            raise
    
    def animate_magnetization(self, M_history, times, save_path=None):
        """创建磁化强度演化动画"""
        try:
            fig = plt.figure(figsize=(15, 10))
            gs = plt.GridSpec(2, 3, figure=fig)
            
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[0, 2])
            ax4 = fig.add_subplot(gs[1, :])
            
            # 初始化图像对象
            M = M_history[0]
            M_magnitude = np.sqrt(np.sum(M**2, axis=2))
            
            im1 = ax1.imshow(M_magnitude.T, origin='lower', cmap='viridis')
            im2 = ax2.imshow(M[:,:,0].T, origin='lower', cmap='RdBu')
            im3 = ax3.imshow(M[:,:,1].T, origin='lower', cmap='RdBu')
            
            X, Y = np.meshgrid(range(M.shape[0]), range(M.shape[1]))
            step = 10
            quiver = ax4.quiver(X[::step, ::step].T, Y[::step, ::step].T,
                               M[::step, ::step, 0].T, M[::step, ::step, 1].T,
                               M_magnitude[::step, ::step].T)
            
            # 设置标题和标签
            ax1.set_title('Magnetization Magnitude')
            ax2.set_title('X Component')
            ax3.set_title('Y Component')
            ax4.set_title('Magnetization Direction')
            
            time_text = fig.text(0.5, 0.95, '', ha='center')
            
            def update(frame):
                M = M_history[frame]
                M_magnitude = np.sqrt(np.sum(M**2, axis=2))
                
                # 更新图像
                im1.set_array(M_magnitude.T)
                im2.set_array(M[:,:,0].T)
                im3.set_array(M[:,:,1].T)
                
                # 更新矢量场
                M_norm = M / np.linalg.norm(M, axis=2, keepdims=True)
                quiver.set_UVC(M_norm[::step, ::step, 0].T,
                              M_norm[::step, ::step, 1].T,
                              M_magnitude[::step, ::step].T)
                
                # 更新时间
                time_text.set_text(f't = {times[frame]:.2e} s')
                
                return im1, im2, im3, quiver, time_text
            
            anim = FuncAnimation(fig, update, frames=len(M_history),
                               interval=50, blit=True)
            
            if save_path:
                anim.save(save_path, writer='pillow', fps=20)
                plt.close()
            else:
                plt.show()
                
            logger.info("磁化强度演化动画已保存")
            
        except Exception as e:
            logger.error(f"生成磁化强度动画时发生错误: {str(e)}")
            raise 
    
    def setup_live_plot(self, H_range):
        """设置实时绘图"""
        self.live_fig, self.live_ax = plt.subplots()
        self.live_ax.set_xlabel('External Field (A/m)')  # 使用英文
        self.live_ax.set_ylabel('Magnetization (A/m)')   # 使用英文
        self.live_ax.set_title('Hysteresis Loop')        # 使用英文
    
    def update_live_plot(self, H_history, M_history):
        """更新实时绘图"""
        if self.live_line is None:
            return
            
        # 更新数据
        self.live_line.set_data(H_history, M_history[:,0])
        self.live_point.set_data([H_history[-1]], [M_history[-1,0]])
        
        # 刷新图表
        self.live_fig.canvas.draw()
        self.live_fig.canvas.flush_events()
    
    def close_live_plot(self):
        """关闭实时绘图窗口"""
        if self.live_fig is not None:
            plt.close(self.live_fig)
            self.live_fig = None
            self.live_ax = None
            self.live_line = None
            self.live_point = None 
    
    def setup_visualization(self):
        """设置可视化环境"""
        # 移除所有字体设置
        # plt.rcParams['font.family'] = ['sans-serif']
        # plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
        # plt.rcParams['axes.unicode_minus'] = False
        
        # 使用默认字体设置
        plt.style.use('seaborn')
        plt.rcParams['figure.figsize'] = [10, 6] 