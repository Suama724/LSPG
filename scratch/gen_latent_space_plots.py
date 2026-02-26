import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pickle
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from net.AE import load_model, encode_ela_feats

class ScatterDataset:
    @staticmethod
    def gen_latent_space_plots_with_clouds(dim, time, model_path, scaler_path, save_path):
        """
        生成潜空间可视化图：包含半透明原始点云、高亮中心点以及标签。
        """
        # 1. 路径准备
        feats_path = os.path.join(config.ARTIFACTS_DIR, 'datasets', f'dataset_{dim}D_{time}', f'results_{dim}D_{time}.pkl')
        pic_name = f'dataset_{dim}D_{time}.png'
        os.makedirs(save_path, exist_ok=True)

        # 2. 数据加载与预处理
        df_raw = pd.read_pickle(feats_path)
        features = np.array(df_raw['ela_feats'].to_list())
        labels = df_raw['meta_func_id'].values

        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        features_scaled = scaler.transform(features)

        # 3. 模型推理获取潜空间坐标
        device = 'cpu'
        ela_feats_num = features.shape[1]
        model = load_model(model_path, ela_feats_num, device)
        latent_points = encode_ela_feats(model, features_scaled, device)
        
        print(f"Latent points shape: {latent_points.shape}")

        # 4. 开始绘图
        plt.figure(figsize=(10, 8))
        
        unique_fids = np.unique(labels)
        # 使用 husl 调色板确保不同函数 ID 颜色区分明显
        colors = sns.color_palette("husl", len(unique_fids))

        for i, fid in enumerate(unique_fids):
            mask = (labels == fid)
            points_of_fid = latent_points[mask]
            
            if len(points_of_fid) == 0:
                continue

            # --- A. 绘制原始点云 (低透明度背景) ---
            plt.scatter(points_of_fid[:, 0], points_of_fid[:, 1], 
                        color=colors[i], 
                        s=15, 
                        alpha=0.15,      # 低透明度，展现分布密度
                        edgecolors='none', 
                        zorder=1) 
            
            # --- B. 计算并绘制中心点 (高亮度) ---
            centroid = points_of_fid.mean(axis=0)
            plt.scatter(centroid[0], centroid[1], 
                        color=colors[i], 
                        marker='o', 
                        s=180,           # 较大的尺寸
                        edgecolors='white', 
                        linewidth=1.5,   # 白色边框增强对比度
                        label=f'F{int(fid)} Center' if i < 5 else None, # 防止图例过长
                        zorder=10)
            
            # --- C. 绘制文字标签 ---
            # 添加微小的随机偏移，防止多个中心点完全重合时文字叠死
            text_pos = centroid + np.random.normal(0, 0.08, size=centroid.shape)
            plt.text(text_pos[0], text_pos[1], f'F{int(fid)}', 
                    fontsize=7, 
                    weight='bold', 
                    color='black',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor=colors[i], boxstyle='round,pad=0.2'),
                    zorder=11)

        # 5. 样式修饰
        plt.title(f'Latent Space Distribution ({dim}D, Time: {time})', fontsize=14)
        plt.xlabel('Latent Dim 1')
        plt.ylabel('Latent Dim 2')
        
        # 固定坐标轴范围以便不同图之间对比
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        
        # 强制比例一致，防止潜空间几何畸变
        plt.gca().set_aspect('equal', adjustable='box')
        
        plt.grid(True, linestyle='--', alpha=0.4, zorder=0)
        
        # 6. 保存
        save_file_path = os.path.join(save_path, pic_name)
        plt.savefig(save_file_path, dpi=300, bbox_inches='tight')
        plt.close() # 及时释放内存
        print(f"Success: Plot saved to {save_file_path}")

    @staticmethod
    def gen_latent_space_plots(dim, time, model_path, scaler_path, save_path):
        #dim = 10
        #time = "2026_02_05_101041"
        feats_path = os.path.join(config.ARTIFACTS_DIR, 'datasets', f'dataset_{dim}D_{time}', f'results_{dim}D_{time}.pkl')
        pic_name = f'dataset_{dim}D_{time}.png'

        
        os.makedirs(save_path, exist_ok=True)

        df_raw = pd.read_pickle(feats_path)
        features = np.array(df_raw['ela_feats'].to_list())
        labels = df_raw['meta_func_id'].values

        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        features_sccaled = scaler.transform(features)

        device = 'cpu'
        ela_feats_num = features.shape[1]

        model = load_model(model_path, ela_feats_num, device)

        latent_points = encode_ela_feats(model, features_sccaled, device)
        print(latent_points.shape)

        #jitter = np.random.normal(0, 0.02, size=latent_points.shape)
        #latent_points = latent_points + jitter

        plt.figure(figsize=(10, 8))

        unique_fids = np.unique(labels)
        colors = sns.color_palette("husl", len(unique_fids))

        for i, fid in enumerate(unique_fids):
            mask = (labels == fid)
            centroid = latent_points[mask].mean(axis=0)
            plt.scatter(centroid[0], centroid[1], 
                        color=colors[i], marker='o', s=150, 
                        zorder=10)
            
            text_pos = centroid + np.random.normal(0, 0.05, size=centroid.shape)
    
            plt.text(text_pos[0], text_pos[1], f'F{int(fid)}', 
                    fontsize=6, weight='bold', color='black',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor=colors[i], boxstyle='round,pad=0.3'),
                    zorder=11)        
            
        
        #scatter = plt.scatter(latent_points[:, 0], latent_points[:, 1],
        #                      c=labels, cmap='viridis', s=30, alpha=1 , edgecolors='none')
        
        #plt.colorbar(scatter, label='BBOB Function ID')
        plt.title(f'Latent Space')
        plt.xlabel('Latent Dim 1')
        plt.ylabel('Latent Dim 2')
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(True, linestyle='--', alpha=0.6)

        save_file_path = os.path.join(save_path, pic_name)
        plt.savefig(save_file_path, dpi=300, bbox_inches='tight')
        print("Success")

    @staticmethod
    def pipeline_gen_scatter_plots_from_generated_dataset():
        time_list = [
            '2026_02_08_142619',
            '2026_02_08_143219',
            '2026_02_08_144421',
            '2026_02_08_145148',
            '2026_02_08_145727',
            '2026_02_08_150230',
            '2026_02_08_150755',
            '2026_02_08_151308',
        ]

        dim = 30

        model_path = os.path.join(config.ARTIFACTS_DIR, 'models', 'model_40D_2026_02_08_182638', 'model', '2026_02_08_182638', 'autoencoder_best.pth')
        scaler_path = os.path.join(config.ARTIFACTS_DIR, 'models', 'model_40D_2026_02_08_182638', 'scaler', 'scaler.pkl')
        save_path = os.path.join(config.ARTIFACTS_DIR, 'scratch_plots')    
        ScatterDataset.gen_latent_space_plots_with_clouds(15, '2026_02_08_160734', model_path, scaler_path, save_path) 
        ScatterDataset.gen_latent_space_plots_with_clouds(20, '2026_02_05_141308', model_path, scaler_path, save_path) 
        ScatterDataset.gen_latent_space_plots_with_clouds(20, '2026_02_08_160754', model_path, scaler_path, save_path) 
        ScatterDataset.gen_latent_space_plots_with_clouds(25, '2026_02_08_160814', model_path, scaler_path, save_path)
        
        ScatterDataset.gen_latent_space_plots_with_clouds(30, '2026_02_08_142619', model_path, scaler_path, save_path) 
        ScatterDataset.gen_latent_space_plots_with_clouds(30, '2026_02_08_143219', model_path, scaler_path, save_path) 
        ScatterDataset.gen_latent_space_plots_with_clouds(30, '2026_02_08_143920', model_path, scaler_path, save_path) 
        ScatterDataset.gen_latent_space_plots_with_clouds(30, '2026_02_08_144421', model_path, scaler_path, save_path) 
        ScatterDataset.gen_latent_space_plots_with_clouds(30, '2026_02_08_145148', model_path, scaler_path, save_path) 
        ScatterDataset.gen_latent_space_plots_with_clouds(30, '2026_02_08_145727', model_path, scaler_path, save_path) 
        ScatterDataset.gen_latent_space_plots_with_clouds(30, '2026_02_08_150230', model_path, scaler_path, save_path) 
        ScatterDataset.gen_latent_space_plots_with_clouds(30, '2026_02_08_150755', model_path, scaler_path, save_path)
        ScatterDataset.gen_latent_space_plots_with_clouds(30, '2026_02_08_151308', model_path, scaler_path, save_path)

        ScatterDataset.gen_latent_space_plots_with_clouds(35, '2026_02_08_160845', model_path, scaler_path, save_path) 
        ScatterDataset.gen_latent_space_plots_with_clouds(40, '2026_02_08_153031', model_path, scaler_path, save_path)

        #for time in time_list:
            #ScatterDataset.gen_latent_space_plots_with_clouds(dim, time, model_path, scaler_path, save_path)

class ScatterSample:
    @staticmethod
    def _decorate_plot(save_path, npy_path, title):
        plt.title(title, fontsize=15)
        plt.xlabel('Latent Dim 1')
        plt.ylabel('Latent Dim 2')
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc='best')
        
        file_name = os.path.basename(npy_path).replace('.npy', '.png')
        '''
        if self.cfg.get('compare_with_dataset', False):
            file_name = "compare_" + file_name
        else:
            file_name = "simple_" + file_name
        '''    
        full_save_path = os.path.join(save_path, file_name)
        plt.savefig(full_save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved to: {full_save_path}")

    @staticmethod
    def gen_sample_plots_no_dataset(npy_path, save_path):
        print(f"Plotting (No Dataset): {npy_path}")
        sampled_points = np.load(npy_path)
        
        plt.figure(figsize=(10, 8))
        
        plt.scatter(sampled_points[:, 0], sampled_points[:, 1], 
                    c='red', marker='o', s=40, linewidth=1.5, edgecolors='black',
                    label='Sampled Points', zorder=10)
        
        ScatterSample._decorate_plot(save_path, npy_path, "Latent Space Sampling (No Background)")
    
    def gen_sample_plots_with_dataset(xx):
        raise NotImplementedError
    
    @staticmethod
    def pipline_gen_sample_from_npy():
        save_path = os.path.join(config.ARTIFACTS_DIR, 'latent_samples')
        os.makedirs(save_path, exist_ok=True)
        ScatterSample.gen_sample_plots_no_dataset(
            save_path=save_path,
            npy_path=os.path.join(config.ARTIFACTS_DIR, 'latent_samples', 'sampled_points_2026_02_08_092422.npy')
        )
        bg_latent_points = None
        bg_labels = None
        bg_loaded = False

                     
if __name__ == '__main__':
    ScatterDataset.pipeline_gen_scatter_plots_from_generated_dataset()
    #ScatterSample.pipline_gen_sample_from_npy()
    