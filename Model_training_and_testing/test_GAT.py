import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, matthews_corrcoef, confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import json
warnings.filterwarnings('ignore')

# 设置字体和分辨率
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 1200
plt.rcParams['figure.dpi'] = 1200

# 添加项目路径
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

# 导入必要的模块
from torch_geometric.loader import DataLoader
from Feature_extraction.utils import load_dataset
from Model_training_and_testing.GAT import create_model

class GATModelTestEvaluator:
    """GAT模型测试评估器"""
    
    def __init__(self, model_name, distance, model_test_dir):
        self.model_name = model_name
        self.distance = distance
        self.model_test_dir = Path(model_test_dir)
        self.model_test_dir.mkdir(parents=True, exist_ok=True)
        
    def evaluate_and_visualize(self, y_true, y_pred, y_prob, test_data=None):
        """完整的评估和可视化"""
        print(f"\n{'='*80}")
        print(f"🎯 {self.model_name} on {self.distance} - Test Set Evaluation Results")
        print(f"{'='*80}")
        
        # 确保数据类型正确
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob)
        
        # 计算所有指标
        metrics = self.calculate_all_metrics(y_true, y_pred, y_prob)
        
        # 打印结果
        self.print_test_results(metrics, y_true, y_pred)
        
        # 保存结果
        self.save_results(y_true, y_pred, y_prob, metrics, test_data) 
               
        # 生成可视化
        self.create_comprehensive_plots(y_true, y_pred, y_prob, metrics)
        
        return metrics
    
    def calculate_all_metrics(self, y_true, y_pred, y_prob):
        """计算所有评估指标"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary'),
            'recall': recall_score(y_true, y_pred, average='binary'),
            'f1': f1_score(y_true, y_pred, average='binary'),
            'auc': roc_auc_score(y_true, y_prob),
            'mcc': matthews_corrcoef(y_true, y_pred)
        }
        
        # 混淆矩阵分解
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # 额外指标
        metrics.update({
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,
            'true_positive': int(tp),
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn)
        })
        
        return metrics
    
    def print_test_results(self, metrics, y_true, y_pred):
        """打印测试结果"""
        total_samples = len(y_true)
        positive_samples = np.sum(y_true == 1)
        negative_samples = np.sum(y_true == 0)
        
        print(f"📊 Dataset Statistics:")
        print(f"   Total samples: {total_samples}")
        print(f"   AVP samples: {positive_samples} ({positive_samples/total_samples*100:.1f}%)")
        print(f"   non_AVP samples: {negative_samples} ({negative_samples/total_samples*100:.1f}%)")
        
        pred_positive = np.sum(y_pred == 1)
        pred_negative = np.sum(y_pred == 0)
        
        print(f"\n🔮 Prediction Statistics:")
        print(f"   Predicted as AVP: {pred_positive} ({pred_positive/total_samples*100:.1f}%)")
        print(f"   Predicted as non_AVP: {pred_negative} ({pred_negative/total_samples*100:.1f}%)")
        
        print(f"\n📈 Performance Metrics:")
        print(f"   Accuracy:      {metrics['accuracy']:.3f}")
        print(f"   Specificity:   {metrics['specificity']:.3f}")
        print(f"   Sensitivity:   {metrics['sensitivity']:.3f}")
        print(f"   Recall:        {metrics['recall']:.3f}")
        print(f"   AUC:           {metrics['auc']:.3f}")
        print(f"   F1-Score:      {metrics['f1']:.3f}")
        print(f"   MCC:           {metrics['mcc']:.3f}")
        print(f"   Precision:     {metrics['precision']:.3f}")

    def create_comprehensive_plots(self, y_true, y_pred, y_prob, metrics):
        """创建综合可视化图表"""
        # 创建2x3子图布局
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 主标题
        fig.suptitle(f'{self.model_name} on {self.distance} Dataset\nTest Set Evaluation', 
                     fontsize=18, fontweight='bold', y=0.96, ha='center')
        
        # 定义子图标签
        subplot_labels = ['(A)', '(B)', '(C)', '(D)', '(E)', '(F)']
        
        # 第一行：混淆矩阵、归一化混淆矩阵、饼图
        self.plot_confusion_matrix_with_accuracy(y_true, y_pred, metrics['accuracy'], axes[0, 0])
        axes[0, 0].text(-0.15, 1.05, subplot_labels[0], transform=axes[0, 0].transAxes, 
                        fontsize=16, fontweight='bold', va='top', ha='right')
        
        self.plot_normalized_confusion_matrix(y_true, y_pred, axes[0, 1])
        axes[0, 1].text(-0.15, 1.05, subplot_labels[1], transform=axes[0, 1].transAxes, 
                        fontsize=16, fontweight='bold', va='top', ha='right')
        
        self.plot_confusion_pie_chart(y_true, y_pred, axes[0, 2])
        axes[0, 2].text(-0.15, 1.05, subplot_labels[2], transform=axes[0, 2].transAxes, 
                        fontsize=16, fontweight='bold', va='top', ha='right')
        
        # 第二行：概率分布、标签分布、ROC曲线
        self.plot_probability_distribution(y_true, y_prob, axes[1, 0])
        axes[1, 0].text(-0.15, 1.05, subplot_labels[3], transform=axes[1, 0].transAxes, 
                        fontsize=16, fontweight='bold', va='top', ha='right')
        
        self.plot_label_distribution(y_true, y_pred, axes[1, 1])
        axes[1, 1].text(-0.15, 1.05, subplot_labels[4], transform=axes[1, 1].transAxes, 
                        fontsize=16, fontweight='bold', va='top', ha='right')
        
        self.plot_roc_curve_in_subplot(y_true, y_prob, axes[1, 2])
        axes[1, 2].text(-0.15, 1.05, subplot_labels[5], transform=axes[1, 2].transAxes, 
                        fontsize=16, fontweight='bold', va='top', ha='right')
        
        # 调整布局
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        
        plt.savefig(self.model_test_dir / 'test_evaluation.png', 
                   dpi=1200, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix_with_accuracy(self, y_true, y_pred, accuracy, ax):
        """绘制带准确度的混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['non_AVP', 'AVP'],
                   yticklabels=['non_AVP', 'AVP'],
                   cbar_kws={'label': 'Count'},
                   annot_kws={'size': 14, 'weight': 'bold'})
        
        ax.set_title(f'Confusion Matrix (Counts)\nAccuracy: {accuracy:.3f}', 
                    fontweight='bold', fontsize=14)
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    
    def plot_normalized_confusion_matrix(self, y_true, y_pred, ax):
        """绘制归一化混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues', ax=ax,
                   xticklabels=['non_AVP', 'AVP'],
                   yticklabels=['non_AVP', 'AVP'],
                   cbar_kws={'label': 'Proportion'},
                   annot_kws={'size': 14, 'weight': 'bold'})
        
        ax.set_title('Confusion Matrix (Normalized)', fontweight='bold', fontsize=14)
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    
    def plot_confusion_pie_chart(self, y_true, y_pred, ax):
        """绘制混淆矩阵饼图"""
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        sizes = [tp, tn, fp, fn]
        labels = ['TP', 'TN', 'FP', 'FN']
        colors = ['#90EE90', '#87CEEB', '#FFB6C1', '#F0E68C']
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                         startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
        
        ax.set_title('Confusion Matrix Distribution', fontweight='bold', fontsize=14)
        
        # 添加图例
        legend = ax.legend(wedges, [f'{label}: {size}' for label, size in zip(labels, sizes)],
                          title="Counts", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        legend.get_title().set_fontweight('bold')
        for text in legend.get_texts():
            text.set_fontweight('bold')
    
    def plot_probability_distribution(self, y_true, y_prob, ax):
        """绘制预测概率分布"""
        y_true = np.array(y_true)
        y_prob = np.array(y_prob)
        
        avp_probs = y_prob[y_true == 1]
        non_avp_probs = y_prob[y_true == 0]
        
        if len(non_avp_probs) > 0:
            ax.hist(non_avp_probs, bins=30, alpha=0.6, label='True non_AVP', 
                   color='red', density=True, edgecolor='black', linewidth=0.5)
        
        if len(avp_probs) > 0:
            ax.hist(avp_probs, bins=30, alpha=0.6, label='True AVP', 
                   color='blue', density=True, edgecolor='black', linewidth=0.5)
        
        ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2,
                  label='Classification Threshold (0.5)')
        
        ax.set_xlabel('Prediction Probability', fontsize=12, fontweight='bold')
        ax.set_ylabel('Density', fontsize=12, fontweight='bold')
        ax.set_title('Probability Distribution', fontweight='bold', fontsize=14)
        legend = ax.legend()
        for text in legend.get_texts():
            text.set_fontweight('bold')
        ax.grid(True, alpha=0.3)
    
    def plot_label_distribution(self, y_true, y_pred, ax):
        """绘制标签分布对比"""
        true_avp = np.sum(y_true == 1)
        true_non_avp = np.sum(y_true == 0)
        pred_avp = np.sum(y_pred == 1)
        pred_non_avp = np.sum(y_pred == 0)
        
        labels = ['non_AVP', 'AVP']
        true_counts = [true_non_avp, true_avp]
        pred_counts = [pred_non_avp, pred_avp]
        
        x = np.arange(len(labels))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, true_counts, width, label='True Labels', 
                      color='steelblue', alpha=0.8, edgecolor='black', linewidth=1)
        bars2 = ax.bar(x + width/2, pred_counts, width, label='Predicted Labels', 
                      color='orange', alpha=0.8, edgecolor='black', linewidth=1)
        
        # 在柱子上添加数值标签
        for bar, value in zip(bars1, true_counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(true_counts + pred_counts)*0.01,
                   f'{int(value)}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        for bar, value in zip(bars2, pred_counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(true_counts + pred_counts)*0.01,
                   f'{int(value)}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax.set_xlabel('Labels', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title('Label Distribution Comparison', fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        legend = ax.legend()
        for text in legend.get_texts():
            text.set_fontweight('bold')
        ax.grid(True, alpha=0.3)
    
    def plot_roc_curve_in_subplot(self, y_true, y_prob, ax):
        """在子图中绘制ROC曲线"""
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        auc_score = roc_auc_score(y_true, y_prob)
        
        ax.plot(fpr, tpr, color='darkorange', linewidth=3, 
                label=f'ROC Curve (AUC = {auc_score:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', linewidth=2, linestyle='--', 
                label='Random Classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate (FPR)', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate (TPR)', fontsize=12, fontweight='bold')
        ax.set_title('ROC Curve', fontweight='bold', fontsize=14)
        legend = ax.legend(loc="lower right", fontsize=10)
        for text in legend.get_texts():
            text.set_fontweight('bold')
        ax.grid(True, alpha=0.3)
    
    def save_results(self, y_true, y_pred, y_prob, metrics, test_data=None):
        """保存详细结果 - 保存为3位有效数字"""
        # 构建结果DataFrame
        result_dict = {
            'sample_id': range(len(y_true)),
            'true_label': y_true,
            'predicted_label': y_pred,
            'prediction_probability': np.round(y_prob, 3),  # 3位小数
            'correct_prediction': (y_true == y_pred)
        }
        
        results_df = pd.DataFrame(result_dict)
        results_df.to_csv(self.model_test_dir / 'test_predictions.csv', index=False)
        
        # 保存指标 - 转换为3位有效数字
        metrics_3sf = {}
        for key, value in metrics.items():
            if isinstance(value, (int, np.integer)):
                metrics_3sf[key] = int(value)
            elif isinstance(value, (float, np.floating)):
                metrics_3sf[key] = round(float(value), 3)
            else:
                metrics_3sf[key] = value
        
        metrics_df = pd.DataFrame([metrics_3sf])
        metrics_df.to_csv(self.model_test_dir / 'test_metrics.csv', index=False)
        
        # 保存测试摘要 - 转换为3位有效数字
        test_summary = {
            'model_name': self.model_name,
            'distance': self.distance,
            'test_performance': metrics_3sf,
            'test_info': {
                'total_samples': len(y_true),
                'avp_samples': int(np.sum(y_true == 1)),
                'non_avp_samples': int(np.sum(y_true == 0))
            }
        }
        
        with open(self.model_test_dir / 'test_summary.json', 'w') as f:
            json.dump(test_summary, f, indent=2)
        
        print(f"\n💾 {self.model_name} test results saved to: {self.model_test_dir}")

class GATLayerComparisonTester:
    """GAT层数对比测试器 - 专门测试训练好的GAT模型"""
    
    def __init__(self, train_results_dir="4_Results_TR_RS", test_results_dir="4_Results_TS_RS", device='cuda'):
        self.train_results_dir = Path(train_results_dir)
        self.test_results_dir = Path(test_results_dir)
        self.test_results_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # GAT配置映射
        self.gat_configs = {
            '2Layer_GAT': {'num_layers': 2, 'name': '2-Layer GAT'},
            '3Layer_GAT': {'num_layers': 3, 'name': '3-Layer GAT'}
        }
        
        # 距离阈值映射
        self.distance_mapping = {
            '4.0A': '4.0A',
            '8.0A': '8.0A', 
            '12.0A': '12.0A'
        }
    
    def extract_distance_from_path(self, test_path):
        """从测试数据路径提取距离阈值"""
        path_str = str(test_path)
        for distance_key in self.distance_mapping.keys():
            if distance_key in path_str:
                return distance_key
        return 'unknown'
    
    def find_available_models(self):
        """查找可用的训练模型"""
        available_models = {}
        
        if not self.train_results_dir.exists():
            print(f"❌ Training results directory not found: {self.train_results_dir}")
            return available_models
        
        # 查找每个距离阈值下的GAT模型
        for distance in self.distance_mapping.keys():
            distance_dir = self.train_results_dir / f"GAT_Comparison_{distance}"
            if distance_dir.exists():
                distance_models = {}
                
                for gat_type in self.gat_configs.keys():
                    gat_dir = distance_dir / gat_type
                    models_dir = gat_dir / 'models'
                    cv_summary_file = gat_dir / 'train_results' / 'cv_summary.json'
                    
                    if models_dir.exists() and cv_summary_file.exists():
                        model_files = list(models_dir.glob("best_model_fold_*.pth"))
                        if model_files:
                            distance_models[gat_type] = {
                                'models_dir': models_dir,
                                'cv_summary_file': cv_summary_file,
                                'model_files': model_files
                            }
                            print(f"✅ Found {distance} {gat_type}: {len(model_files)} fold models")
                
                if distance_models:
                    available_models[distance] = distance_models
        
        return available_models
    
    def safe_torch_load(self, path, map_location=None):
        """安全的torch.load包装器"""
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=map_location)
    
    def test_single_gat_model(self, distance, gat_type, model_info, test_data):
        """测试单个GAT模型"""
        gat_name = self.gat_configs[gat_type]['name']
        print(f"\n{'='*80}")
        print(f"🧪 Testing {gat_name} on {distance} Dataset")
        print(f"{'='*80}")
        
        # 创建测试结果目录
        test_distance_dir = self.test_results_dir / f"GAT_Test_{distance}"
        test_model_dir = test_distance_dir / gat_type
        test_model_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取模型文件
        model_files = model_info['model_files']
        model_files.sort()
        
        print(f"📦 Found {len(model_files)} fold models")
        
        test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
        all_predictions = []
        
        # 对每个fold模型进行预测
        for i, model_path in enumerate(model_files):
            try:
                print(f"Loading model from fold {i+1}: {model_path.name}")
                
                checkpoint = self.safe_torch_load(model_path, map_location=self.device)
                
                # 从checkpoint中读取模型参数
                model_params = checkpoint['model_params']
                num_layers = checkpoint['num_layers']
                
                # 创建模型 - 使用你的GAT模型
                model = create_model(
                    model_params['node_feature_dim'], 
                    num_layers=num_layers,
                    **{k: v for k, v in model_params.items() if k != 'node_feature_dim'}
                ).to(self.device)
                
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                
                fold_probs = []
                with torch.no_grad():
                    for batch in test_loader:
                        batch = batch.to(self.device)
                        # GAT模型的forward接口
                        outputs, _ = model(batch.x, batch.edge_index, batch.batch)
                        probs = torch.softmax(outputs, dim=1)[:, 1]
                        fold_probs.extend(probs.cpu().numpy())
                
                all_predictions.append(fold_probs)
                print(f"✅ Successfully tested fold {i+1}")
                
            except Exception as e:
                print(f"❌ Error loading model from {model_path}: {e}")
                continue
        
        if not all_predictions:
            print(f"❌ No valid models could be loaded for {gat_name} on {distance}!")
            return None
        
        # 集成预测（平均概率）
        ensemble_probs = np.mean(all_predictions, axis=0)
        ensemble_preds = (ensemble_probs > 0.5).astype(int)
        
        # 真实标签
        true_labels = [graph.y.item() for graph in test_data]
        
        # 计算指标预览
        test_metrics_preview = {
            'accuracy': accuracy_score(true_labels, ensemble_preds),
            'recall': recall_score(true_labels, ensemble_preds, average='binary'),
            'f1': f1_score(true_labels, ensemble_preds, average='binary'),
            'auc': roc_auc_score(true_labels, ensemble_probs),
            'mcc': matthews_corrcoef(true_labels, ensemble_preds)
        }
        
        print(f"\n🎯 {gat_name} on {distance} Test Results Preview:")
        print(f"   Accuracy: {test_metrics_preview['accuracy']:.3f}")
        print(f"   AUC:      {test_metrics_preview['auc']:.3f}")
        print(f"   F1-Score: {test_metrics_preview['f1']:.3f}")
        print(f"   MCC:      {test_metrics_preview['mcc']:.3f}")
        
        # 创建GAT模型评估器并进行详细评估
        evaluator = GATModelTestEvaluator(gat_name, distance, test_model_dir)
        final_metrics = evaluator.evaluate_and_visualize(
            true_labels, ensemble_preds, ensemble_probs, test_data
        )
        
        print(f"✅ {gat_name} on {distance} testing completed!")
        
        return final_metrics
    
    def test_distance_dataset(self, test_path, available_models):
        """测试单个距离数据集的所有GAT模型"""
        distance = self.extract_distance_from_path(test_path)
        
        print(f"\n{'='*100}")
        print(f"🧬 Testing GAT Models on {distance} Test Dataset")
        print(f"📁 Test data path: {test_path}")
        print(f"{'='*100}")
        
        # 加载测试数据
        test_data = load_dataset(test_path)
        print(f"✅ Loaded {len(test_data)} test graphs")
        
        # 分析测试集分布
        test_labels = [graph.y.item() for graph in test_data]
        test_pos = sum(test_labels)
        test_neg = len(test_labels) - test_pos
        
        print(f"\n📊 Test Set Distribution:")
        print(f"   Total samples: {len(test_labels)}")
        print(f"   AVP samples: {test_pos} ({test_pos/len(test_labels)*100:.1f}%)")
        print(f"   non_AVP samples: {test_neg} ({test_neg/len(test_labels)*100:.1f}%)")
        
        # 检查是否有对应的训练模型
        if distance not in available_models:
            print(f"❌ No trained models found for {distance} dataset!")
            return {}
        
        distance_models = available_models[distance]
        distance_results = {}
        
        # 测试每种GAT配置
        for gat_type, model_info in distance_models.items():
            try:
                test_results = self.test_single_gat_model(
                    distance, gat_type, model_info, test_data
                )
                if test_results:
                    distance_results[gat_type] = test_results
            except Exception as e:
                print(f"❌ Error testing {gat_type} on {distance}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 生成距离特定的比较报告
        if len(distance_results) > 1:
            self.create_distance_comparison_report(distance, distance_results)
        
        return distance_results
    
    def plot_distance_comparison(self, comparison_data, comparison_dir, distance):
        """绘制距离特定的GAT模型比较图"""
        metrics = ['accuracy', 'sensitivity', 'specificity', 'f1', 'mcc', 'auc']
        metric_labels = ['Accuracy', 'Sensitivity', 'Specificity', 'F1 Score', 'MCC', 'AUC']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        gat_models = [row['GAT_Model'] for row in comparison_data]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(gat_models)]
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            values = [row[metric] for row in comparison_data]
            
            bars = axes[i].bar(gat_models, values, color=colors, alpha=0.8)
            axes[i].set_title(f'{label}', fontweight='bold', fontsize=14)
            axes[i].set_ylabel(label, fontsize=12)
            axes[i].grid(True, alpha=0.3)
            
            # 添加数值标签 - 3位有效数字
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, 
                            bar.get_height() + max(values) * 0.01,
                            f'{value:.3f}', ha='center', va='bottom', 
                            fontsize=10, fontweight='bold')
            
            # 标记最佳结果
            best_idx = values.index(max(values))
            bars[best_idx].set_edgecolor('green')
            bars[best_idx].set_linewidth(2)
        
        plt.suptitle(f'GAT Models Test Performance Comparison on {distance} Dataset', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        # 保存图片
        save_path = comparison_dir / f'{distance}_gat_test_comparison.png'
        plt.savefig(save_path, dpi=1200, bbox_inches='tight')
        plt.show()
        
        print(f"📊 {distance} comparison plot saved: {save_path}")
        
    def create_distance_comparison_report(self, distance, distance_results):
        """创建距离特定的GAT模型比较报告"""
        print(f"\n📊 Creating {distance} GAT Models Comparison Report")
        print("="*60)
        
        # 创建比较目录
        comparison_dir = self.test_results_dir / f"GAT_Test_{distance}"
        
        # 准备比较数据 - 转换为3位有效数字
        comparison_data = []
        metrics_keys = ['accuracy', 'sensitivity', 'specificity', 'f1', 'mcc', 'auc', 'precision']
        
        for gat_type, results in distance_results.items():
            gat_name = self.gat_configs[gat_type]['name']
            row = {'GAT_Model': gat_name, 'Layers': self.gat_configs[gat_type]['num_layers']}
            for metric in metrics_keys:
                if metric in results:
                    row[metric] = round(results[metric], 3)
                else:
                    row[metric] = 0.000
            comparison_data.append(row)
        
        # 保存比较表格
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv(comparison_dir / f'{distance}_gat_comparison.csv', index=False)
        
        # 生成距离比较图表
        if len(comparison_data) > 1:
            self.plot_distance_comparison(comparison_data, comparison_dir, distance)   
                 
        # 打印比较结果
        print(f"\n📊 {distance} GAT Models Test Comparison:")
        print("="*80)
        print(f"{'Model':<15} {'Layers':<8} {'Accuracy':<10} {'Sensitivity':<12} {'Specificity':<12} {'F1 Score':<10} {'MCC':<8} {'AUC':<8}")
        print("="*80)
        
        for row in comparison_data:
            print(f"{row['GAT_Model']:<15} {row['Layers']:<8} {row['accuracy']:<10.3f} {row['sensitivity']:<12.3f} {row['specificity']:<12.3f} {row['f1']:<10.3f} {row['mcc']:<8.3f} {row['auc']:<8.3f}")
        
        print("="*80)
        
        # 找出最佳模型
        best_models = {}
        for metric in metrics_keys:
            best_score = 0
            best_model = None
            for row in comparison_data:
                if row[metric] > best_score:
                    best_score = row[metric]
                    best_model = row['GAT_Model']
            if best_model:
                best_models[metric] = (best_model, best_score)
        
        print(f"\n🏆 Best {distance} GAT Models by Metric:")
        for metric, (model_name, score) in best_models.items():
            print(f"  {metric}: {model_name} ({score:.3f})")
        
        # 保存比较报告 - 转换为3位有效数字
        comparison_report = {
            'distance': distance,
            'test_comparison_table': comparison_data,
            'best_models': {k: (v[0], round(v[1], 3)) for k, v in best_models.items()},
            'total_models_tested': len(distance_results),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open(comparison_dir / f'{distance}_comparison_report.json', 'w') as f:
            json.dump(comparison_report, f, indent=2)
        
        print(f"\n✅ {distance} comparison report saved to: {comparison_dir}")
    
    def test_all_datasets(self, test_paths):
        """测试所有数据集的GAT模型"""
        print(f"\n🎭 GAT Layer Comparison Testing System")
        print(f"{'='*80}")
        
        # 查找可用模型
        available_models = self.find_available_models()
        
        if not available_models:
            print("❌ No trained GAT models found!")
            return {}
        
        print(f"\n🎯 Available trained models:")
        for distance, models in available_models.items():
            print(f"  {distance}: {list(models.keys())}")
        
        # 测试所有数据集
        all_test_results = {}
        
        for test_path in test_paths:
            try:
                distance_results = self.test_distance_dataset(test_path, available_models)
                if distance_results:
                    distance = self.extract_distance_from_path(test_path)
                    all_test_results[distance] = distance_results
            except Exception as e:
                print(f"❌ Error testing {test_path}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 生成综合比较报告
        if len(all_test_results) > 0:
            self.create_comprehensive_test_report(all_test_results)
        
        return all_test_results
    
    def create_comprehensive_test_report(self, all_test_results):
        """创建综合测试比较报告"""
        print(f"\n📊 Creating Comprehensive GAT Test Comparison Report")
        print("="*80)
        
        # 创建综合比较目录
        comprehensive_dir = self.test_results_dir / "Comprehensive_Test_Comparison"
        comprehensive_dir.mkdir(parents=True, exist_ok=True)
        
        # 准备综合数据 - 修改为与训练集一致的格式，转换为3位有效数字
        comprehensive_data = []
        metrics = ['accuracy', 'sensitivity', 'specificity', 'f1', 'mcc', 'auc']
        
        for distance, distance_results in all_test_results.items():
            for gat_type, results in distance_results.items():
                gat_name = self.gat_configs[gat_type]['name']
                row = {
                    'Distance': distance,
                    'GAT_Model': gat_name,
                    'Layers': self.gat_configs[gat_type]['num_layers']
                }
                
                for metric in metrics:
                    value = round(results.get(metric, 0), 3)
                    row[metric] = f"{value:.3f}"  
                comprehensive_data.append(row)
        
        # 保存综合比较表格
        comprehensive_df = pd.DataFrame(comprehensive_data)
        comprehensive_df.to_csv(comprehensive_dir / 'comprehensive_test_comparison.csv', index=False)
        
        # 生成综合比较图
        self.plot_comprehensive_test_comparison(comprehensive_data, comprehensive_dir)
        
        # 分析最佳配置
        best_overall_configs = self.analyze_best_test_configurations(comprehensive_data)
        
        # 修改：保存完整测试报告，格式与训练集一致，转换为3位有效数字
        test_report = {
            'comparison_data': comprehensive_data,
            'best_configurations': best_overall_configs,
            'summary': {
                'total_test_experiments': len(comprehensive_data),
                'tested_distances': list(all_test_results.keys()),
                'tested_gat_models': list(set([row['GAT_Model'] for row in comprehensive_data]))
            },
            # 添加格式化的最佳配置摘要
            'best_configurations_formatted': {},
            # 添加格式化的结果
            'formatted_results': {}
        }
        
        # 为最佳配置添加格式化版本
        for metric_name, config in best_overall_configs.items():
            if config:
                test_report['best_configurations_formatted'][metric_name] = {
                    'config': f"{config['gat_model']} on {config['distance']}",
                    'performance': f"{config['score']:.3f}"
                }
        
        # 为每个距离和GAT配置添加格式化版本
        for distance, distance_results in all_test_results.items():
            test_report['formatted_results'][distance] = {}
            for gat_type, results in distance_results.items():
                gat_name = self.gat_configs[gat_type]['name']
                
                formatted_metrics = {}
                for metric in metrics:
                    value = round(results.get(metric, 0), 3)
                    formatted_metrics[metric] = f"{value:.3f}"
                
                test_report['formatted_results'][distance][gat_name] = formatted_metrics
        
        # 保存JSON报告，使用UTF-8编码
        with open(comprehensive_dir / 'comprehensive_test_report.json', 'w', encoding='utf-8') as f:
            json.dump(test_report, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Comprehensive test comparison complete! Results saved to: {comprehensive_dir}")
        
        return test_report
    
    def plot_comprehensive_test_comparison(self, comprehensive_data, comprehensive_dir):
        """绘制综合测试比较图表"""
        metrics = ['accuracy', 'sensitivity', 'specificity', 'f1', 'mcc', 'auc']
        metric_labels = ['Accuracy', 'Sensitivity/SN/Recall', 'Specificity/SP', 'F1 Score', 'MCC', 'AUC']
        
        # 创建大型综合比较图
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))
        axes = axes.ravel()
        
        distances = ['4.0A', '8.0A', '12.0A']
        gat_models = ['2-Layer GAT', '3-Layer GAT']
        colors = {'2-Layer GAT': '#1f77b4', '3-Layer GAT': '#ff7f0e'}
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            # 准备数据 - 从格式化字符串中提取数值
            data_2layer = []
            data_3layer = []
            
            for distance in distances:
                val_2layer = 0
                val_3layer = 0
                
                for row in comprehensive_data:
                    if row['Distance'] == distance:
                        # 从格式化字符串中提取数值
                        metric_str = row[metric]  
                        parts = metric_str.split(' ± ')
                        mean_val = float(parts[0])
                        
                        if row['GAT_Model'] == '2-Layer GAT':
                            val_2layer = mean_val
                        elif row['GAT_Model'] == '3-Layer GAT':
                            val_3layer = mean_val
                
                data_2layer.append(val_2layer)
                data_3layer.append(val_3layer)
            
            # 绘制分组柱状图（测试集没有误差棒）
            x = np.arange(len(distances))
            width = 0.35
            
            bars1 = axes[i].bar(x - width/2, data_2layer, width, 
                            label='2-Layer GAT', color=colors['2-Layer GAT'], alpha=0.8)
            bars2 = axes[i].bar(x + width/2, data_3layer, width,
                            label='3-Layer GAT', color=colors['3-Layer GAT'], alpha=0.8)
            
            axes[i].set_title(f'{label}', fontweight='bold', fontsize=14)
            axes[i].set_xlabel('Distance Threshold', fontsize=12, fontweight='bold')
            axes[i].set_ylabel(label, fontsize=12, fontweight='bold')
            axes[i].set_xticks(x)
            axes[i].set_xticklabels(distances)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            
            # 添加数值标签 - 3位有效数字
            for j, (bar1, bar2, v1, v2) in enumerate(zip(bars1, bars2, data_2layer, data_3layer)):
                if v1 > 0:
                    axes[i].text(bar1.get_x() + bar1.get_width()/2, 
                            bar1.get_height() + max(data_2layer + data_3layer) * 0.01, 
                            f'{v1:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
                if v2 > 0:
                    axes[i].text(bar2.get_x() + bar2.get_width()/2, 
                            bar2.get_height() + max(data_2layer + data_3layer) * 0.01,
                            f'{v2:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
                
                # 标记更好的结果
                if v1 > v2 and v1 > 0:
                    bar1.set_edgecolor('green')
                    bar1.set_linewidth(2)
                elif v2 > v1 and v2 > 0:
                    bar2.set_edgecolor('green')
                    bar2.set_linewidth(2)
        
        # 添加主标题
        plt.suptitle('Comprehensive GAT Test Performance Comparison Across Distance Thresholds', 
                    fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        # 修改：保存图片
        save_path = comprehensive_dir / 'comprehensive_test_comparison.png'
        plt.savefig(save_path, dpi=1200, bbox_inches='tight')
        
        # 修改：控制是否显示图片
        plt.show()
        
        print(f"📊 Comprehensive test comparison plot saved: {save_path}")
    
    def analyze_best_test_configurations(self, comprehensive_data):
        """分析最佳测试配置"""
        metrics = ['accuracy', 'sensitivity', 'specificity', 'f1', 'mcc', 'auc']
        metric_labels = ['Accuracy', 'Sensitivity', 'Specificity', 'F1 Score', 'MCC', 'AUC']
        
        best_configs = {}
        
        for metric, label in zip(metrics, metric_labels):
            best_score = 0
            best_config = None
            
            for row in comprehensive_data:           
                metric_str = row[metric]  
                parts = metric_str.split(' ± ')
                score = float(parts[0])
                
                if score > best_score:
                    best_score = score
                    best_config = {
                        'distance': row['Distance'],
                        'gat_model': row['GAT_Model'],
                        'layers': row['Layers'],
                        'score': round(score, 3)
                    }
            
            best_configs[label] = best_config
        
        # 打印最佳配置
        print(f"\n🏆 Best Test Configurations by Metric:")
        print("="*80)
        for metric_label, config in best_configs.items():
            if config:
                print(f"{metric_label}: {config['gat_model']} on {config['distance']} ({config['score']:.3f})")
        
        return best_configs

def main_gat_testing(
    test_paths=[
        '3_Graph_Data/TS/TS_ESMC_4.0A.pkl',
        '3_Graph_Data/TS/TS_ESMC_8.0A.pkl',
        '3_Graph_Data/TS/TS_ESMC_12.0A.pkl'
    ],
    train_results_dir="4_Results_TR_RS"
):
    """
    主GAT测试函数 - 测试所有训练过的GAT模型
    
    参数:
        test_paths (list): 测试数据文件路径列表
        train_results_dir (str): 训练结果目录
    """
    print("🧪 GAT Layer Comparison Testing System")
    print("="*80)
    
    # 配置参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using device: {device}")
    
    print(f"📊 Test datasets:")
    for path in test_paths:
        print(f"  - {path}")
    
    print(f"📁 Training results directory: {train_results_dir}")
    
    # 检查训练结果目录
    if not Path(train_results_dir).exists():
        print(f"❌ Training results directory not found: {train_results_dir}")
        print("Please run the GAT training code first!")
        return None
    
    # 创建GAT测试器
    tester = GATLayerComparisonTester(
        train_results_dir=train_results_dir,
        test_results_dir="4_Results_TS_RS",
        device=device
    )
    
    # 执行测试
    try:
        all_test_results = tester.test_all_datasets(test_paths)
        
        if all_test_results:
            print(f"\n🎉 GAT testing completed successfully!")
            print(f"📊 Tested configurations:")
            for distance, results in all_test_results.items():
                for gat_type, metrics in results.items():
                    gat_name = tester.gat_configs[gat_type]['name']
                    print(f"  ✅ {gat_name} on {distance}: AUC = {metrics['auc']:.3f}")
        else:
            print(f"\n❌ No GAT models were successfully tested!")
        
        return all_test_results
        
    except Exception as e:
        print(f"❌ Error during GAT testing: {e}")
        import traceback
        traceback.print_exc()
        return None

# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 测试示例
    results = main_gat_testing(
        test_paths=[
            '3_Graph_Data/TS/TS_ESMC_4.0A.pkl',
            '3_Graph_Data/TS/TS_ESMC_8.0A.pkl',
            '3_Graph_Data/TS/TS_ESMC_12.0A.pkl'
        ],
        train_results_dir="4_Results_TR_RS"
    )
    
    if results:
        print("\n🎉 GAT multi-model testing completed successfully!")
        print(f"📊 Total configurations tested: {sum(len(r) for r in results.values())}")
    else:
        print("\n❌ GAT testing failed!")