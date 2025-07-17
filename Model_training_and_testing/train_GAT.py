import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import json
from datetime import datetime
import pickle
import time
warnings.filterwarnings('ignore')

# ==================== 基础配置 ====================
# 设置字体和画图参数
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 1200
plt.rcParams['figure.dpi'] = 1200

# 添加项目路径到系统路径
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

# 导入项目特定模块
from torch_geometric.loader import DataLoader
from Feature_extraction.utils import load_dataset
from .GAT import create_2layer_gat, create_3layer_gat, create_model

class GATLayerComparisonTrainer:
    """
    GAT层数对比训练器
    专门用于比较2层和3层GAT在不同距离阈值数据上的表现
    """
    
    def __init__(self, base_save_dir="4_Results_TR_RS", seed=42, use_class_weights=True, show_plots=True):
        """
        初始化GAT层数对比训练器
        
        参数:
            base_save_dir (str): 基础保存目录
            seed (int): 随机种子
            use_class_weights (bool): 是否使用类别权重
            show_plots (bool): 是否显示图表
        """
        self.base_save_dir = Path(base_save_dir)
        self.seed = seed
        self.use_class_weights = use_class_weights
        self.show_plots = show_plots
        
        # GAT层数配置
        self.gat_configs = {
            '2Layer_GAT': {'num_layers': 2, 'name': '2Layer_GAT'},
            '3Layer_GAT': {'num_layers': 3, 'name': '3Layer_GAT'}
        }
        
        # 距离阈值映射
        self.distance_mapping = {
            '4.0A': '4.0A',
            '8.0A': '8.0A', 
            '12.0A': '12.0A'
        }
        
        # 初始化设备和环境
        self.setup_seed()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🔬 GAT Layer Comparison Trainer Initialized")
        print(f"🖥️ Using device: {self.device}")
        print(f"📁 Base save directory: {self.base_save_dir}")
        print(f"⚖️ Using class weights: {self.use_class_weights}")
        print(f"📊 Show plots: {self.show_plots}")
    
    def setup_seed(self):
        """设置随机种子，确保实验结果可重复"""
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    def extract_distance_from_path(self, train_path):
        """从路径中提取距离阈值"""
        path_str = str(train_path)
        for distance_key in self.distance_mapping.keys():
            if distance_key in path_str:
                return distance_key
        return 'unknown'
    
    def create_directory_structure(self, train_paths):
        """
        创建GAT层数对比的目录结构
        
        参数:
            train_paths (list): 训练数据路径列表
            
        返回:
            all_dirs (dict): 所有目录结构的字典
        """
        all_dirs = {}
        
        for train_path in train_paths:
            distance = self.extract_distance_from_path(train_path)
            
            # 创建距离特定的主目录
            distance_dir = self.base_save_dir / f"GAT_Comparison_{distance}"
            distance_dir.mkdir(parents=True, exist_ok=True)
            
            # 为每种GAT配置创建子目录
            distance_structure = {}
            for gat_type, config in self.gat_configs.items():
                gat_name = config['name']
                gat_dir = distance_dir / gat_name
                
                distance_structure[gat_type] = {
                    'base_dir': gat_dir,
                    'models': gat_dir / "models",
                    'train_results': gat_dir / "train_results",
                    'training_plots': gat_dir / "training_plots"
                }
                
                # 创建所有必要的目录
                for dir_path in distance_structure[gat_type].values():
                    if isinstance(dir_path, Path):
                        dir_path.mkdir(parents=True, exist_ok=True)
            
            all_dirs[distance] = {
                'distance_dir': distance_dir,
                'gat_dirs': distance_structure
            }
            
            print(f"📁 Created directory structure for {distance}")
        
        return all_dirs
    
    def calculate_class_weights(self, labels):
        """计算类别权重以处理数据不平衡问题"""
        if not self.use_class_weights:
            return None
            
        unique_labels = np.unique(labels)
        class_weights = compute_class_weight('balanced', classes=unique_labels, y=labels)
        weight_dict = dict(zip(unique_labels, class_weights))
        
        print(f"⚖️ Class weights: Class 0: {weight_dict[0]:.3f}, Class 1: {weight_dict[1]:.3f}")
        
        return torch.FloatTensor(class_weights).to(self.device)
    
    def calculate_specificity(self, y_true, y_pred):
        """计算特异性 (Specificity)"""
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        return specificity
    
    def evaluate_model(self, model, data_loader, class_weights=None):
        """详细评估模型性能"""
        model.eval()
        all_preds = []
        all_probs = []
        all_labels = []
        total_loss = 0

        if self.use_class_weights and class_weights is not None:
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = torch.nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                
                # GAT模型的forward接口
                outputs, _ = model(batch.x, batch.edge_index, batch.batch)
                
                loss = criterion(outputs, batch.y)
                total_loss += loss.item()
                
                probs = torch.softmax(outputs, dim=1)[:, 1]
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, average='binary'),
            'recall': recall_score(all_labels, all_preds, average='binary'),
            'f1': f1_score(all_labels, all_preds, average='binary'),
            'auc': roc_auc_score(all_labels, all_probs),
            'mcc': matthews_corrcoef(all_labels, all_preds),
            'specificity': self.calculate_specificity(all_labels, all_preds)
        }
        
        return metrics
    
    def train_single_fold(self, gat_type, model_params, training_params, 
                         fold_train_data, fold_val_data, fold_num, gat_dirs, 
                         class_weights, distance):
        """训练单个fold的GAT模型"""
        gat_config = self.gat_configs[gat_type]
        num_layers = gat_config['num_layers']
        gat_name = gat_config['name']
        
        print(f"\n🚀 Training {gat_name} on {distance} data - Fold {fold_num}")
        
        # 创建模型
        model = create_model(
            model_params['node_feature_dim'], 
            num_layers=num_layers,
            **{k: v for k, v in model_params.items() if k != 'node_feature_dim'}
        ).to(self.device)
        
        # 优化器和调度器
        optimizer = torch.optim.AdamW(model.parameters(), 
                                     lr=training_params['lr'], 
                                     weight_decay=training_params.get('weight_decay', 0.01))
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-6, verbose=False
        )
        
        # 损失函数
        if self.use_class_weights and class_weights is not None:
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = torch.nn.CrossEntropyLoss()
        
        # 数据加载器
        train_loader = DataLoader(fold_train_data, batch_size=training_params['batch_size'], shuffle=True)
        val_loader = DataLoader(fold_val_data, batch_size=training_params['batch_size'], shuffle=False)
        
        # 训练状态
        best_val_auc = 0
        best_model_state = None
        best_epoch = 0
        best_metrics = None
        patience_counter = 0
        patience = 20
        
        # 训练历史
        training_history = {
            'train_losses': [],
            'val_losses': [],
            'train_accuracies': [],
            'val_accuracies': [],
            'val_aucs': []
        }
        
        # 主训练循环
        num_epochs = training_params['max_epochs']
        
        for epoch in range(num_epochs):
            # 训练阶段
            model.train()
            total_loss = 0
            train_preds = []
            train_labels = []
            
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                
                outputs, _ = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(outputs, batch.y)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                train_preds.extend(preds.cpu().numpy())
                train_labels.extend(batch.y.cpu().numpy())
            
            # 计算训练指标
            avg_train_loss = total_loss / len(train_loader)
            train_accuracy = accuracy_score(train_labels, train_preds)
            
            training_history['train_losses'].append(avg_train_loss)
            training_history['train_accuracies'].append(train_accuracy)
            
            # 验证阶段
            val_metrics = self.evaluate_model(model, val_loader, class_weights)
            val_loss = val_metrics['loss']
            val_accuracy = val_metrics['accuracy']
            val_auc = val_metrics['auc']
            
            training_history['val_losses'].append(val_loss)
            training_history['val_accuracies'].append(val_accuracy)
            training_history['val_aucs'].append(val_auc)
            
            # 更新学习率
            scheduler.step(val_auc)
            
            # 保存最佳模型
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch + 1
                best_model_state = model.state_dict().copy()
                best_metrics = val_metrics.copy()
                patience_counter = 0
                
                # 保存最佳模型
                best_model_path = gat_dirs[gat_type]['models'] / f'best_model_fold_{fold_num}.pth'
                torch.save({
                    'model_state_dict': best_model_state,
                    'fold': fold_num,
                    'best_val_auc': best_val_auc,
                    'best_epoch': best_epoch,
                    'best_val_metrics': best_metrics,
                    'model_params': model_params,
                    'training_params': training_params,
                    'num_layers': num_layers
                }, best_model_path)
                
            else:
                patience_counter += 1
            
            # 训练进度显示
            if (epoch + 1) % 5 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"  Epoch {epoch+1}/{num_epochs} - Val AUC: {val_auc:.4f} - Val ACC: {val_accuracy:.4f} - LR: {current_lr:.6f}")
            
            # 早停检查
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                print(f"  Best model from epoch {best_epoch} (AUC: {best_val_auc:.4f})")
                break
        
        # 保存训练历史
        history_df = pd.DataFrame({
            'epoch': range(1, len(training_history['train_losses']) + 1),
            'train_loss': training_history['train_losses'],
            'val_loss': training_history['val_losses'],
            'train_accuracy': training_history['train_accuracies'],
            'val_accuracy': training_history['val_accuracies'],
            'val_auc': training_history['val_aucs']
        })
        numeric_columns = history_df.select_dtypes(include=[np.number]).columns.drop('epoch')
        history_df[numeric_columns] = history_df[numeric_columns].round(4)
        
        history_df.to_csv(gat_dirs[gat_type]['train_results'] / f'training_history_fold_{fold_num}.csv', index=False)
        
        print(f"✅ {gat_name} Fold {fold_num} completed - Best AUC: {best_val_auc:.4f} at Epoch {best_epoch}")
        
        return best_metrics, training_history
    
    def train_distance_dataset(self, train_path, model_params, training_params, all_dirs, n_folds=5):
        """训练单个距离阈值数据集的所有GAT配置"""
        distance = self.extract_distance_from_path(train_path)
        
        print(f"\n{'='*80}")
        print(f"🧬 Training GAT Models on {distance} Dataset")
        print(f"📁 Data path: {train_path}")
        print(f"{'='*80}")
        
        # 🔍 检查已完成的训练
        distance_dirs = all_dirs[distance]
        gat_dirs = distance_dirs['gat_dirs']
        
        existing_results = {}
        for gat_type in self.gat_configs.keys():
            cv_summary_file = gat_dirs[gat_type]['train_results'] / 'cv_summary.json'
            if cv_summary_file.exists():
                try:
                    with open(cv_summary_file, 'r') as f:
                        existing_results[gat_type] = json.load(f)
                    print(f"✅ Found existing results for {self.gat_configs[gat_type]['name']}")
                except:
                    print(f"⚠️ Corrupted results file for {self.gat_configs[gat_type]['name']}, will retrain")
        
        # 如果所有GAT类型都已完成，跳过训练
        if len(existing_results) == len(self.gat_configs):
            print(f"🎯 All GAT models for {distance} already trained! Skipping...")
            
            # 重构已有结果
            distance_results = {}
            for gat_type, cv_summary in existing_results.items():
                distance_results[gat_type] = {
                    'cv_summary': cv_summary,
                    'fold_results': None,  
                    'training_history': None
                }
            
            return distance_results
        
        # 加载数据（只有在需要训练时才加载）
        train_data = load_dataset(train_path)
        print(f"✅ Loaded {len(train_data)} graphs")
        
        # 数据准备
        train_labels = np.array([graph.y.item() for graph in train_data])
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.seed)
        
        # 存储所有结果
        distance_results = {}
        
        # 训练每种GAT配置
        for gat_type in self.gat_configs.keys():
            # 🔍 检查单个GAT类型是否已完成
            if gat_type in existing_results:
                print(f"✅ {self.gat_configs[gat_type]['name']} already trained, using existing results")
                distance_results[gat_type] = {
                    'cv_summary': existing_results[gat_type],
                    'fold_results': None,
                    'training_history': None
                }
                continue
            
            print(f"\n{'='*60}")
            print(f"🔄 Training {self.gat_configs[gat_type]['name']} on {distance} data")
            print(f"{'='*60}")
            
            fold_results = []
            all_training_history = []
            
            # K折交叉验证
            for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(train_data)), train_labels)):
                print(f"\n🔄 Fold {fold + 1}/{n_folds}")
                
                # 数据划分
                fold_train_data = [train_data[i] for i in train_idx]
                fold_val_data = [train_data[i] for i in val_idx]
                
                fold_train_labels = train_labels[train_idx]
                fold_val_labels = train_labels[val_idx]
                
                # 打印数据分布
                print(f"Training: {len(fold_train_data)} samples (AVP: {sum(fold_train_labels)} - {sum(fold_train_labels)/len(fold_train_labels)*100:.1f}%)")
                print(f"Validation: {len(fold_val_data)} samples (AVP: {sum(fold_val_labels)} - {sum(fold_val_labels)/len(fold_val_labels)*100:.1f}%)")
                
                # 计算类别权重
                class_weights = self.calculate_class_weights(fold_train_labels)
                
                # 训练当前fold
                fold_metrics, fold_history = self.train_single_fold(
                    gat_type, model_params, training_params,
                    fold_train_data, fold_val_data, fold + 1, gat_dirs,
                    class_weights, distance
                )
                
                fold_results.append(fold_metrics)
                all_training_history.append(fold_history)
                
                # 打印fold结果
                print(f"Fold {fold + 1} Results:")
                for metric, value in fold_metrics.items():
                    print(f"  {metric}: {value:.4f}")
            
            # 保存交叉验证结果
            cv_df = pd.DataFrame(fold_results)
            cv_df['fold'] = range(1, len(fold_results) + 1)
            
            numeric_columns = cv_df.select_dtypes(include=[np.number]).columns
            cv_df[numeric_columns] = cv_df[numeric_columns].round(4)
            
            cv_df.to_csv(gat_dirs[gat_type]['train_results'] / 'cross_validation_results.csv', index=False)
            
            # 计算统计摘要
            cv_summary = {}
            for metric in fold_results[0].keys():
                values = [fold[metric] for fold in fold_results]
                cv_summary[metric] = {
                    'mean': round(np.mean(values), 4),
                    'std': round(np.std(values), 4)
                }
            
            # 保存统计摘要
            with open(gat_dirs[gat_type]['train_results'] / 'cv_summary.json', 'w') as f:
                json.dump(cv_summary, f, indent=2)
            
            # 生成训练曲线图
            self.plot_training_curves(gat_type, all_training_history, gat_dirs, distance)
            
            # 生成结果汇总图
            self.plot_results_summary(gat_type, fold_results, gat_dirs, distance)
            
            distance_results[gat_type] = {
                'fold_results': fold_results,
                'cv_summary': cv_summary,
                'training_history': all_training_history
            }
            
            print(f"✅ {self.gat_configs[gat_type]['name']} training completed!")
        
        # 生成距离特定的比较图
        self.plot_distance_comparison(distance_results, distance_dirs['distance_dir'], distance)
        
        return distance_results
    
    def plot_training_curves(self, gat_type, all_training_history, gat_dirs, distance):
        """绘制训练曲线"""
        if not all_training_history:
            return
        
        gat_name = self.gat_configs[gat_type]['name']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{gat_name} on {distance} - 5-Fold Training Curves', 
                     fontsize=18, fontweight='bold', y=0.96)
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        # 训练损失
        ax1 = axes[0, 0]
        for fold_idx, history in enumerate(all_training_history):
            epochs = range(1, len(history['train_losses']) + 1)
            ax1.plot(epochs, history['train_losses'], color=colors[fold_idx], 
                    linewidth=2, label=f'Fold {fold_idx + 1}', alpha=0.8)
        ax1.set_title('Training Loss', fontweight='bold', fontsize=14)
        ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Training Loss', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 验证损失
        ax2 = axes[0, 1]
        for fold_idx, history in enumerate(all_training_history):
            epochs = range(1, len(history['val_losses']) + 1)
            ax2.plot(epochs, history['val_losses'], color=colors[fold_idx], 
                    linewidth=2, label=f'Fold {fold_idx + 1}', alpha=0.8)
        ax2.set_title('Validation Loss', fontweight='bold', fontsize=14)
        ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 验证准确度
        ax3 = axes[1, 0]
        for fold_idx, history in enumerate(all_training_history):
            epochs = range(1, len(history['val_accuracies']) + 1)
            ax3.plot(epochs, history['val_accuracies'], color=colors[fold_idx], 
                    linewidth=2, label=f'Fold {fold_idx + 1}', alpha=0.8)
        ax3.set_title('Validation Accuracy', fontweight='bold', fontsize=14)
        ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Validation Accuracy', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # 验证AUC
        ax4 = axes[1, 1]
        for fold_idx, history in enumerate(all_training_history):
            epochs = range(1, len(history['val_aucs']) + 1)
            ax4.plot(epochs, history['val_aucs'], color=colors[fold_idx], 
                    linewidth=2, label=f'Fold {fold_idx + 1}', alpha=0.8)
        ax4.set_title('Validation AUC', fontweight='bold', fontsize=14)
        ax4.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Validation AUC', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        
        # 保存图表
        save_path = gat_dirs[gat_type]['training_plots'] / 'training_curves.png'
        plt.savefig(save_path, dpi=1200, bbox_inches='tight')
        
        if self.show_plots:
            plt.show()
        else:
            plt.close()
        
        print(f"📊 Training curves saved: {save_path}")
    
    def plot_results_summary(self, gat_type, fold_results, gat_dirs, distance):
        """绘制结果汇总图"""
        gat_name = self.gat_configs[gat_type]['name']
        
        metrics_mapping = {
            'accuracy': 'Accuracy',
            'recall': 'Sensitivity/SN/Recall', 
            'specificity': 'Specificity/SP',
            'f1': 'F1 Score',
            'mcc': 'MCC',
            'auc': 'AUC'
        }
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for i, (metric_key, metric_label) in enumerate(metrics_mapping.items()):
            values = [fold[metric_key] for fold in fold_results]
            folds = range(1, len(fold_results) + 1)
            
            axes[i].bar(folds, values, alpha=0.7, color=plt.cm.Set3(i))
            axes[i].axhline(y=np.mean(values), color='red', linestyle='--', linewidth=2, 
                           label=f'Mean: {np.mean(values):.4f}')
            
            axes[i].set_title(f'{metric_label}', fontweight='bold', fontsize=12)
            axes[i].set_xlabel('Fold', fontsize=10, fontweight='bold')
            axes[i].set_ylabel(metric_label, fontsize=10, fontweight='bold')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            
            for j, v in enumerate(values):
                axes[i].text(j + 1, v + max(values) * 0.01, f'{v:.3f}', 
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.suptitle(f'{gat_name} on {distance} - Cross-Validation Results', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = gat_dirs[gat_type]['training_plots'] / 'results_summary.png'
        plt.savefig(save_path, dpi=1200, bbox_inches='tight')
        
        if self.show_plots:
            plt.show()
        else:
            plt.close()
        
        print(f"📊 Results summary saved: {save_path}")
    
    def plot_distance_comparison(self, distance_results, distance_dir, distance):
        """绘制同一距离下2层和3层GAT的比较图"""
        metrics = ['accuracy', 'recall', 'specificity', 'f1', 'mcc', 'auc']
        metric_labels = ['Accuracy', 'Sensitivity/SN/Recall', 'Specificity/SP', 'F1 Score', 'MCC', 'AUC']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        gat_types = ['2layer_gat', '3layer_gat']
        gat_names = ['2-Layer GAT', '3-Layer GAT']
        colors = ['#1f77b4', '#ff7f0e']
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            means = []
            stds = []
            
            for gat_type in gat_types:
                if gat_type in distance_results:
                    cv_summary = distance_results[gat_type]['cv_summary']
                    means.append(cv_summary[metric]['mean'])
                    stds.append(cv_summary[metric]['std'])
                else:
                    means.append(0)
                    stds.append(0)
            
            x_pos = np.arange(len(gat_names))
            bars = axes[i].bar(x_pos, means, yerr=stds, alpha=0.7, color=colors, capsize=5)
            
            axes[i].set_title(f'{label}', fontweight='bold', fontsize=12)
            axes[i].set_xlabel('GAT Configuration', fontsize=10, fontweight='bold')
            axes[i].set_ylabel(label, fontsize=10, fontweight='bold')
            axes[i].set_xticks(x_pos)
            axes[i].set_xticklabels(gat_names)
            axes[i].grid(True, alpha=0.3)
            
            # 添加数值标签
            for j, (mean, std) in enumerate(zip(means, stds)):
                axes[i].text(j, mean + std + max(means) * 0.01, f'{mean:.3f}', 
                           ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # 标记更好的结果
            if len(means) == 2 and means[0] != means[1]:
                better_idx = 0 if means[0] > means[1] else 1
                bars[better_idx].set_edgecolor('green')
                bars[better_idx].set_linewidth(3)
        
        plt.suptitle(f'2-Layer vs 3-Layer GAT Comparison on {distance} Dataset', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = distance_dir / 'gat_layer_comparison.png'
        plt.savefig(save_path, dpi=1200, bbox_inches='tight')
        
        if self.show_plots:
            plt.show()
        else:
            plt.close()
        
        print(f"📊 GAT train results comparison saved: {save_path}")
    
    def generate_comprehensive_comparison(self, all_distance_results):
        """生成所有距离和层数的综合比较报告"""
        print(f"\n{'='*80}")
        print(f"📊 Generating Comprehensive Train Performance Comparison Report")
        print(f"{'='*80}")
        
        # 创建综合比较目录
        comprehensive_dir = self.base_save_dir / "Comprehensive_Train_Comparison"
        comprehensive_dir.mkdir(parents=True, exist_ok=True)
        
        # 准备数据
        comparison_data = []
        metrics = ['accuracy', 'recall', 'specificity', 'f1', 'mcc', 'auc']
        
        for distance, distance_results in all_distance_results.items():
            for gat_type, results in distance_results.items():
                row = {
                    'Distance': distance,
                    'GAT_Type': self.gat_configs[gat_type]['name'],
                    'Layers': self.gat_configs[gat_type]['num_layers']
                }
                
                cv_summary = results['cv_summary']
                for metric in metrics:
                    # 修改这里：确保4位小数精度
                    mean_val = round(cv_summary[metric]['mean'], 3)
                    std_val = round(cv_summary[metric]['std'], 3)
                    row[metric] = f"{mean_val:.3f} ± {std_val:.3f}"
                
                comparison_data.append(row)
        
        # 保存比较表格
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv(comprehensive_dir / 'comprehensive_train_comparison.csv', index=False)
        
        # 生成综合比较图
        self.plot_comprehensive_comparison(comparison_data, comprehensive_dir)
        
        # 生成最佳配置分析
        best_configs = self.analyze_best_configurations(comparison_data, comprehensive_dir)
        
        # 修改这里：保存完整报告时也使用格式化字符串
        report = {
            'comparison_data': comparison_data,
            'best_configurations': best_configs,
            'summary': {
                'total_experiments': len(comparison_data),
                'distance_thresholds': list(all_distance_results.keys()),
                'gat_configurations': list(self.gat_configs.keys())
            },
            # 添加格式化的最佳配置摘要
            'best_configurations_formatted': {}
        }
        
        # 为最佳配置添加格式化版本
        for metric_name, config in best_configs.items():
            if config:
                report['best_configurations_formatted'][metric_name] = {
                    'config': f"{config['gat_type']} on {config['distance']}",
                    'performance': f"{config['score']:.3f} ± {config['std']:.3f}"
                }
        
        # 为每个距离和GAT配置添加格式化版本
        report['formatted_results'] = {}
        for distance, distance_results in all_distance_results.items():
            report['formatted_results'][distance] = {}
            for gat_type, results in distance_results.items():
                gat_name = self.gat_configs[gat_type]['name']
                cv_summary = results['cv_summary']
                
                formatted_metrics = {}
                for metric in metrics:
                    mean_val = round(cv_summary[metric]['mean'], 3)
                    std_val = round(cv_summary[metric]['std'], 3)
                    formatted_metrics[metric] = f"{mean_val:.3f} ± {std_val:.3f}"
                
                report['formatted_results'][distance][gat_name] = formatted_metrics
        
        with open(comprehensive_dir / 'comprehensive_train_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Comprehensive comparison complete! Results saved to: {comprehensive_dir}")
        
        return report
    
    def plot_comprehensive_comparison(self, comparison_data, comprehensive_dir):
        """绘制综合比较图表"""
        metrics = ['accuracy', 'recall', 'specificity', 'f1', 'mcc', 'auc']
        metric_labels = ['Accuracy', 'Sensitivity/SN/Recall', 'Specificity/SP', 'F1 Score', 'MCC', 'AUC']
        
        # 创建大型综合比较图
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))
        axes = axes.ravel()
        
        distances = ['4.0A', '8.0A', '12.0A']
        gat_types = ['2Layer_GAT', '3Layer_GAT']
        
        # 设置颜色
        colors = {'2Layer_GAT': '#1f77b4', '3Layer_GAT': '#ff7f0e'}
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            # 准备数据 - 修改这部分
            data_2layer = []
            data_3layer = []
            std_2layer = []
            std_3layer = []
            
            for distance in distances:
                for row in comparison_data:
                    if row['Distance'] == distance:
                        # 从格式化字符串中提取数值
                        metric_str = row[metric]  # 格式: "0.8327 ± 0.0169"
                        parts = metric_str.split(' ± ')
                        mean_val = float(parts[0])
                        std_val = float(parts[1])
                        
                        if row['GAT_Type'] == '2Layer_GAT':
                            data_2layer.append(mean_val)
                            std_2layer.append(std_val)
                        elif row['GAT_Type'] == '3Layer_GAT':
                            data_3layer.append(mean_val)
                            std_3layer.append(std_val)
            
            # 绘制分组柱状图
            x = np.arange(len(distances))
            width = 0.35
            
            bars1 = axes[i].bar(x - width/2, data_2layer, width, yerr=std_2layer, 
                            label='2-Layer GAT', color=colors['2Layer_GAT'], alpha=0.8, capsize=5)
            bars2 = axes[i].bar(x + width/2, data_3layer, width, yerr=std_3layer,
                            label='3-Layer GAT', color=colors['3Layer_GAT'], alpha=0.8, capsize=5)
            
            axes[i].set_title(f'{label}', fontweight='bold', fontsize=14)
            axes[i].set_xlabel('Distance Threshold', fontsize=12, fontweight='bold')
            axes[i].set_ylabel(label, fontsize=12, fontweight='bold')
            axes[i].set_xticks(x)
            axes[i].set_xticklabels(distances)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            
            # 添加数值标签 - 修改这部分
            max_height = max(max(data_2layer) if data_2layer else [0], max(data_3layer) if data_3layer else [0])
            for j, (bar1, bar2) in enumerate(zip(bars1, bars2)):
                if j < len(data_2layer):
                    axes[i].text(bar1.get_x() + bar1.get_width()/2, 
                            data_2layer[j] + std_2layer[j] + max_height * 0.02, 
                            f'{data_2layer[j]:.3f}', ha='center', va='bottom', 
                            fontsize=9, fontweight='bold')
                if j < len(data_3layer):
                    axes[i].text(bar2.get_x() + bar2.get_width()/2, 
                            data_3layer[j] + std_3layer[j] + max_height * 0.02,
                            f'{data_3layer[j]:.3f}', ha='center', va='bottom', 
                            fontsize=9, fontweight='bold')
                
                # 标记更好的结果
                if j < len(data_2layer) and j < len(data_3layer):
                    if data_2layer[j] > data_3layer[j]:
                        bar1.set_edgecolor('green')
                        bar1.set_linewidth(2)
                    elif data_3layer[j] > data_2layer[j]:
                        bar2.set_edgecolor('green')
                        bar2.set_linewidth(2)
        
        plt.suptitle('Comprehensive GAT Train Performance Comparison Across Distance Thresholds', 
                    fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        save_path = comprehensive_dir / 'comprehensive_train_comparison.png'
        plt.savefig(save_path, dpi=1200, bbox_inches='tight')
        
        if self.show_plots:
            plt.show()
        else:
            plt.close()
        
        print(f"📊 Comprehensive comparison plot saved: {save_path}")
    
    def analyze_best_configurations(self, comparison_data, comprehensive_dir):
        """分析最佳配置"""
        metrics = ['accuracy', 'recall', 'specificity', 'f1', 'mcc', 'auc']
        metric_labels = ['Accuracy', 'Sensitivity/SN/Recall', 'Specificity/SP', 'F1 Score', 'MCC', 'AUC']
        
        best_configs = {}
        
        for metric, label in zip(metrics, metric_labels):
            best_score = 0
            best_config = None
            
            for row in comparison_data:
                # 从格式化字符串中提取分数 
                metric_str = row[metric] 
                parts = metric_str.split(' ± ')
                score = float(parts[0])
                std = float(parts[1])
                
                if score > best_score:
                    best_score = score
                    best_config = {
                        'distance': row['Distance'],
                        'gat_type': row['GAT_Type'],
                        'layers': row['Layers'],
                        'score': round(score, 3),
                        'std': round(std, 3)
                    }
            
            best_configs[label] = best_config
        
        # 打印最佳配置
        print(f"\n🏆 Best Configurations by Metric:")
        print("="*80)
        for metric_label, config in best_configs.items():
            if config:
                print(f"{metric_label}: {config['gat_type']} on {config['distance']} ({config['score']:.4f} ± {config['std']:.4f})")
        
        # 保存最佳配置摘要表格
        best_configs_df = pd.DataFrame([
            {
                'Metric': metric_label,
                'Best_Configuration': f"{config['gat_type']} on {config['distance']}",
                'Performance': f"{config['score']:.3f} ± {config['std']:.3f}",
                'Score': config['score'],
                'Std': config['std']
            }
            for metric_label, config in best_configs.items() if config
        ])
        
        best_configs_df.to_csv(comprehensive_dir / 'best_configurations.csv', index=False)
        
        return best_configs

def train_gat_layer_comparison(
    train_paths=[
        '3_Graph_Data/TR/TR_ESMC_4.0A.pkl',
        '3_Graph_Data/TR/TR_ESMC_8.0A.pkl', 
        '3_Graph_Data/TR/TR_ESMC_12.0A.pkl'
    ],
    base_save_dir="4_Results_TR_RS",
    random_seed=42,
    model_params=None,
    training_params=None,
    n_folds=5,
    use_class_weights=True,
    show_plots=True
):
    """
    GAT层数对比训练主函数
    
    参数:
        train_paths (list): 训练数据路径列表
        base_save_dir (str): 基础保存目录
        random_seed (int): 随机种子
        model_params (dict): 模型参数
        training_params (dict): 训练参数
        n_folds (int): 交叉验证折数
        use_class_weights (bool): 是否使用类别权重
        show_plots (bool): 是否显示图表
        
    返回:
        comprehensive_report (dict): 综合比较报告
    """
    
    print("🔬 GAT Layer Comparison Training System")
    print("="*80)
    print(f"📁 Training datasets:")
    for path in train_paths:
        print(f"  - {path}")
    print(f"📁 Results will be saved to: {base_save_dir}")
    
    # 初始化训练器
    trainer = GATLayerComparisonTrainer(
        base_save_dir=base_save_dir,
        seed=random_seed,
        use_class_weights=use_class_weights,
        show_plots=show_plots
    )
    
    # 检查第一个数据集获取输入维度
    first_data = load_dataset(train_paths[0])
    input_dim = first_data[0].x.shape[1]
    print(f"🔬 Input feature dimension: {input_dim}")
    
    # 设置默认参数
    if model_params is None:
        model_params = {
            'node_feature_dim': input_dim,
            'hidden_dim': 128,
            'output_dim': 2,
            'drop': 0.3,  # 针对GAT优化的dropout
            'heads': 8,   # 增加注意力头数
            'k': 0.7,
            'add_self_loops': True
        }
    else:
        model_params['node_feature_dim'] = input_dim
    
    if training_params is None:
        training_params = {
            'lr': 0.0001,
            'batch_size': 64,
            'max_epochs': 100,
            'weight_decay': 0.01
        }
    
    print(f"🤖 Model parameters: {model_params}")
    print(f"🏋️ Training parameters: {training_params}")
    
    # 创建目录结构
    all_dirs = trainer.create_directory_structure(train_paths)
    
    # 训练所有距离数据集
    all_distance_results = {}
    
    for train_path in train_paths:
        try:
            print(f"\n{'='*100}")
            print(f"🚀 Processing {train_path}")
            print(f"{'='*100}")
            
            distance_results = trainer.train_distance_dataset(
                train_path, model_params, training_params, all_dirs, n_folds
            )
            
            distance = trainer.extract_distance_from_path(train_path)
            all_distance_results[distance] = distance_results
            
            print(f"✅ {train_path} processing completed!")
            
        except Exception as e:
            print(f"❌ Error processing {train_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 生成综合比较报告
    if all_distance_results:
        comprehensive_report = trainer.generate_comprehensive_comparison(all_distance_results)
    else:
        comprehensive_report = None
        print("❌ No successful training results to compare")
    
    # 打印最终摘要
    print(f"\n{'='*80}")
    print(f"🎉 GAT Layer Comparison Training Complete!")
    print(f"{'='*80}")
    print(f"📁 Results saved to: {base_save_dir}")
    print(f"🧬 Processed datasets: {len(all_distance_results)}/{len(train_paths)}")
    
    for distance in all_distance_results.keys():
        print(f"  ✅ {distance} dataset")
    
    return comprehensive_report

# ==================== Jupyter Notebook 兼容函数 ====================
def run_gat_comparison_notebook():
    """专门为 Jupyter Notebook 设计的运行函数"""
    
    # 数据路径
    train_paths = [
        '3_Graph_Data/TR/TR_ESMC_4.0A.pkl',
        '3_Graph_Data/TR/TR_ESMC_8.0A.pkl', 
        '3_Graph_Data/TR/TR_ESMC_12.0A.pkl'
    ]
    
    # 运行GAT层数对比训练
    comprehensive_report = train_gat_layer_comparison(
        train_paths=train_paths,
        base_save_dir="4_Results_TR_RS",
        random_seed=42,
        model_params={
            'hidden_dim': 128,
            'output_dim': 2,
            'drop': 0.3,
            'heads': 8,
            'k': 0.7,
            'add_self_loops': True
        },
        training_params={
            'lr': 0.0001,
            'batch_size': 64,
            'max_epochs': 100,
            'weight_decay': 0.01
        },
        n_folds=5,
        use_class_weights=True,
        show_plots=True
    )
    
    return comprehensive_report

if __name__ == "__main__":
    # 直接运行GAT层数对比
    comprehensive_report = run_gat_comparison_notebook()