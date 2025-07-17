import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from typing import Dict, Tuple
import torch
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from torch_geometric.nn import GATConv
import torch.nn.functional as F
import networkx as nx
from mpl_toolkits.axes_grid1 import make_axes_locatable

# 添加项目路径并设置Matplotlib字体
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12

# 导入所需模块
from Feature_extraction.graph_builder import SimpleGraphDatasetBuilder
from Feature_extraction.utils import load_dataset
from Model_training_and_testing.GAT import create_model


class DataProcessor:
    """数据处理器 - 负责处理CSV和FASTA输入"""
    @staticmethod
    def prepare_input_data(input_path: str, output_dir: Path) -> str:
        input_path = Path(input_path)
        output_csv = output_dir / "input_for_prediction.csv"
        if input_path.suffix.lower() == '.csv':
            if input_path.resolve() == output_csv.resolve() and output_csv.exists():
                print(f"⏭️ Using existing standardized CSV: {output_csv}")
                return str(output_csv)
            df = pd.read_csv(input_path)
            if 'Id' not in df.columns or 'Sequence' not in df.columns:
                raise ValueError("CSV must contain 'Id' and 'Sequence' columns.")
        elif input_path.suffix.lower() in ['.fasta', '.fa', '.fas']:
            df = DataProcessor._parse_fasta_file(input_path)
        else:
            raise ValueError(f"❌ Unsupported file format: {input_path.suffix}")

        if 'Label' not in df.columns:
            df['Label'] = 0

        df.to_csv(output_csv, index=False)
        print(f"✅ Input data standardized to: {output_csv}")
        return str(output_csv)

    @staticmethod
    def _parse_fasta_file(fasta_path: Path) -> pd.DataFrame:
        sequences = []
        with open(fasta_path, 'r') as f:
            current_id, current_seq = None, ""
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_id:
                        sequences.append({'Id': current_id, 'Sequence': current_seq.upper()})
                    current_id, current_seq = line[1:].split()[0], ""
                else:
                    current_seq += line
            if current_id:
                sequences.append({'Id': current_id, 'Sequence': current_seq.upper()})
        
        if not sequences:
            raise ValueError("❌ No valid sequences found in FASTA file.")
        return pd.DataFrame(sequences)


class GraphBuilder:
    """图构建器 - 负责构建分子图"""
    def __init__(self, config: Dict):
        self.config = config

    def build_graphs(self, standardized_csv_path: str, original_input_path: str) -> str:
        """
        构建或定位图数据文件。
        使用原始输入路径来确定数据集名称，以确保使用正确的、已存在的图文件。
        """
        print("🔧 Building molecular graphs...")
        # 🔥 修复：使用原始输入文件的名称来确定图数据路径
        dataset_name = Path(original_input_path).stem
        expected_graph_file = Path(f"3_Graph_Data/{dataset_name}/{dataset_name}_ESMC_4.0A.pkl")

        if expected_graph_file.exists():
            print(f"✅ Found and using existing graph file: {expected_graph_file}")
            return str(expected_graph_file)

        # 仅在找不到对应的图文件时，才进行构建
        print(f"⚠️ Graph file not found. Building new graphs for '{dataset_name}'...")
        builder = SimpleGraphDatasetBuilder(self.config)
        graphs, _ = builder.automated_pipeline(
            csv_file=standardized_csv_path, output_name=dataset_name, save_dataset=True
        )
        print(f"✅ Graph building completed: {len(graphs)} graphs created at {expected_graph_file}")
        return str(expected_graph_file)


class ModelLoader:
    """模型加载器 - 负责加载训练好的模型"""
    def __init__(self, model_dir: str, device: torch.device):
        self.model_dir = Path(model_dir)
        self.device = device

    def load_models(self) -> list:
        print(f"🤖 Loading GAT models from: {self.model_dir}")
        models_dir = self.model_dir / 'models'
        model_files = sorted(list(models_dir.glob("best_model_fold_*.pth")))
        if not model_files:
            raise FileNotFoundError(f"❌ No model files found in: {models_dir}")

        models = []
        model_info = None
        for model_path in model_files:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            if model_info is None:
                model_info = checkpoint['model_params']
            
            model = create_model(**model_info).to(self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            models.append(model)
        
        print(f"✅ Successfully loaded {len(models)} models for ensemble prediction.")
        return models


class GradCAMAnalyzer:
    """Grad-CAM分析器 - 负责模型可解释性分析"""
    def __init__(self, models: list, device: torch.device, output_dir: Path):
        self.models = models
        self.device = device
        self.output_dir = output_dir
        self.model_activations = None
        self.model_gradients = None
        
        # 创建individual_sample_plots目录
        self.individual_plots_dir = output_dir / "individual_sample_plots"
        self.individual_plots_dir.mkdir(parents=True, exist_ok=True)
        
        # 可视化参数
        self.NODE_SIZE = 1000
        self.FONT_SIZE = 26
        self.POSITION_FONT_SIZE = 20
        self.AVP_COLORMAP = 'Greens'
        self.NON_AVP_COLORMAP = 'Reds'
        self.TOP_K_AMINO_ACIDS = 5

    def _forward_hook(self, module, input, output):
        self.model_activations = output[0] if isinstance(output, tuple) else output

    def _backward_hook(self, module, grad_input, grad_output):
        self.model_gradients = grad_output[0]

    def compute_grad_cam(self, model, graph, target_class):
        """计算单个模型的Grad-CAM"""
        model.eval()
        model.zero_grad()
        
        # 找到最后一个GAT层
        last_gat_layer = None
        for module in model.modules():
            if isinstance(module, GATConv):
                last_gat_layer = module
        
        if last_gat_layer is None:
            print(f"⚠️ No GAT layer found in model")
            return np.zeros(graph.num_nodes)
        
        # 注册钩子
        f_handle = last_gat_layer.register_forward_hook(self._forward_hook)
        b_handle = last_gat_layer.register_full_backward_hook(self._backward_hook)
        
        try:
            # 前向传播
            output, _ = model(graph.x, graph.edge_index, graph.batch)
            
            # 反向传播
            output[0, target_class].backward()
            
            if self.model_gradients is None or self.model_activations is None:
                print(f"⚠️ Failed to capture gradients or activations")
                return np.zeros(graph.num_nodes)
            
            # 计算权重和CAM
            weights = torch.mean(self.model_gradients, dim=0)
            cam = F.relu(torch.sum(self.model_activations * weights, dim=1))
            
            # 归一化
            scores = cam.detach().cpu().numpy()
            
            # 确保返回的数组长度与图节点数一致
            if len(scores) != graph.num_nodes:
                print(f"⚠️ Score length ({len(scores)}) != graph nodes ({graph.num_nodes})")
                return np.zeros(graph.num_nodes)
            
            return scores / scores.max() if scores.max() > 0 else scores
            
        except Exception as e:
            print(f"⚠️ Error in compute_grad_cam: {e}")
            return np.zeros(graph.num_nodes)
            
        finally:
            f_handle.remove()
            b_handle.remove()
            self.model_activations = None
            self.model_gradients = None
            model.zero_grad()

    def run_ensemble_grad_cam(self, graph, target_class):
        """运行集成Grad-CAM"""
        all_scores = []
        
        for i, model in enumerate(self.models):
            scores = self.compute_grad_cam(model, graph, target_class)
            if len(scores) == graph.num_nodes:
                all_scores.append(scores)
        
        if not all_scores:
            print(f"⚠️ No valid Grad-CAM scores computed")
            return np.zeros(graph.num_nodes)
        
        # 计算平均重要性分数
        final_importance = np.mean(all_scores, axis=0)
        if final_importance.max() > 0:
            final_importance /= final_importance.max()
        
        return final_importance

    def get_top_k_amino_acids(self, importance, sequence, k=5):
        """获取Top K重要氨基酸的信息"""
        if len(importance) != len(sequence):
            print(f"⚠️ Importance length ({len(importance)}) != sequence length ({len(sequence)})")
            return []
        
        top_k_indices = np.argsort(importance)[-k:][::-1]
        top_k_info = [
            (sequence[i], i + 1, importance[i]) for i in top_k_indices
        ]
        return top_k_info

    def visualize_sample(self, seq_id, sequence, prob, pred, importance):
        """为单个样本生成并保存可视化图（仅保存，不显示）"""
        # 检查数据一致性
        if len(importance) != len(sequence):
            print(f"⚠️ Skipping {seq_id}: importance length ({len(importance)}) != sequence length ({len(sequence)})")
            return
        
        try:
            seq_len = len(sequence)
            fig_width = max(18, seq_len * 0.7)
            fig, ax = plt.subplots(1, 1, figsize=(fig_width, 6))
            
            # 准备标题
            top_k_info = self.get_top_k_amino_acids(importance, sequence, self.TOP_K_AMINO_ACIDS)
            pred_label = "AVP" if pred == 1 else "non-AVP"
            
            if top_k_info:
                top_k_text = ", ".join([f"{aa}({pos}): {score:.3f}" for aa, pos, score in top_k_info])
            else:
                top_k_text = "N/A"
            
            # 构建标题（只显示序列、ID和重要性节点）
            title_line1 = f"{sequence} (ID: {seq_id})"
            title_line2 = f"Predicted: {pred_label} (Prob: {prob:.3f})"
            title_line3 = f"Top Normalized Node Importance: {top_k_text}"
            
            full_title = "\n".join([title_line1, title_line2, title_line3])
            ax.set_title(full_title, fontsize=24, pad=40, loc='center')
            
            self._plot_linear_sequence(ax, sequence, importance, pred)
            
            # 绘制颜色条
            cmap = plt.cm.get_cmap(self.AVP_COLORMAP if pred == 1 else self.NON_AVP_COLORMAP)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="1.5%", pad=0.2)
            
            cbar = plt.colorbar(sm, cax=cax, shrink=0.8)
            cbar.set_label('Node Importance', fontsize=12, rotation=270, labelpad=20)
            
            ax.set_ylim(-1, 1)
            ax.axis('off')
            plt.tight_layout(rect=[0, 0, 1, 0.9])
            
            # 保存图片（不显示）
            save_path = self.individual_plots_dir / f"sample_{seq_id}_grad_cam.png"
            plt.savefig(save_path, dpi=50, bbox_inches='tight')
            plt.close(fig)  # 关闭图形释放内存
            
            print(f"🖼️ Grad-CAM visualization saved: {save_path}")
            
        except Exception as e:
            print(f"⚠️ Error visualizing sample {seq_id}: {e}")

    def _plot_linear_sequence(self, ax, sequence, importance, pred):
        """绘制线性序列可视化"""
        try:
            # 再次检查数据一致性
            if len(importance) != len(sequence):
                print(f"⚠️ Plotting error: importance length ({len(importance)}) != sequence length ({len(sequence)})")
                return
            
            G = nx.path_graph(len(sequence))
            pos = {i: (i, 0) for i in range(len(sequence))}
            cmap = plt.get_cmap(self.AVP_COLORMAP if pred == 1 else self.NON_AVP_COLORMAP)
            
            # 确保importance是numpy数组且长度正确
            importance = np.array(importance)
            if len(importance) != len(sequence):
                print(f"⚠️ Final check failed: importance length ({len(importance)}) != sequence length ({len(sequence)})")
                return
            
            nx.draw_networkx_nodes(G, pos, node_color=importance, cmap=cmap, vmin=0, vmax=1, 
                                 node_size=self.NODE_SIZE, edgecolors='black', ax=ax)
            nx.draw_networkx_edges(G, pos, edge_color='gray', width=1.5, ax=ax)
            nx.draw_networkx_labels(G, pos, labels=dict(enumerate(sequence)), 
                                  font_size=self.FONT_SIZE, font_color='white', font_weight='bold', ax=ax)
            nx.draw_networkx_labels(G, {i: (i, -0.35) for i in range(len(sequence))}, 
                                  labels={i: str(i+1) for i in range(len(sequence))}, 
                                  font_size=self.POSITION_FONT_SIZE, font_color='black', ax=ax)
        except Exception as e:
            print(f"⚠️ Error in _plot_linear_sequence: {e}")


class Predictor:
    """预测器 - 负责模型预测"""
    def __init__(self, models: list, device: torch.device):
        self.models = models
        self.device = device

    def predict_graphs(self, graph_file: str, batch_size: int) -> Tuple[pd.DataFrame, list]:
        print("🔮 Making predictions...")
        graphs = load_dataset(graph_file)
        
        # 🔥 关键修复：保持原始顺序，不要shuffle
        data_loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
        
        all_probs = []
        all_indices = []  # 记录原始索引
        
        with torch.no_grad():
            current_idx = 0
            for batch in data_loader:
                batch = batch.to(self.device)
                batch_size_actual = batch.batch.max().item() + 1
                
                # 计算集成预测
                batch_model_probs = []
                for model in self.models:
                    model_output = model(batch.x, batch.edge_index, batch.batch)[0]
                    model_probs = torch.softmax(model_output, dim=1)[:, 1]
                    batch_model_probs.append(model_probs)
                
                ensemble_probs = torch.stack(batch_model_probs).mean(dim=0)
                all_probs.extend(ensemble_probs.cpu().numpy())
                
                # 记录当前批次的索引
                for i in range(batch_size_actual):
                    all_indices.append(current_idx + i)
                current_idx += batch_size_actual

        # 🔥 关键修复：确保结果和图数据的对应关系
        results_data = []
        for i, (graph, prob) in enumerate(zip(graphs, all_probs)):
            seq_id = getattr(graph, 'seq_id', f'sample_{i}')
            sequence = getattr(graph, 'sequence', '')
            
            # 验证序列长度与图节点数的一致性
            if len(sequence) != graph.num_nodes:
                print(f"⚠️ Warning: {seq_id} - sequence length ({len(sequence)}) != graph nodes ({graph.num_nodes})")
                print(f"   Sequence: {sequence}")
                print(f"   Graph info: nodes={graph.num_nodes}, edges={graph.edge_index.shape[1]}")
            
            results_data.append({
                'Id': seq_id,
                'Sequence': sequence,
                'AVP_Probability': round(prob, 4),
                'Graph_Nodes': graph.num_nodes,  # 添加调试信息
                'Original_Index': i  # 添加原始索引
            })

        results_df = pd.DataFrame(results_data)
        results_df['Predicted_Label'] = (results_df['AVP_Probability'] > 0.5).astype(int)
        results_df['Predicted_Class'] = results_df['Predicted_Label'].map({0: 'non_AVP', 1: 'AVP'})
        
        # 🔥 关键修复：按概率排序但保持图数据的原始顺序对应
        # 先保存原始顺序
        original_order = results_df['Original_Index'].tolist()
        
        # 排序结果
        results_df_sorted = results_df.sort_values('AVP_Probability', ascending=False).reset_index(drop=True)
        
        # 重新排列图数据以匹配排序后的结果
        sorted_indices = results_df_sorted['Original_Index'].tolist()
        graphs_sorted = [graphs[i] for i in sorted_indices]
        
        # 移除调试列
        results_df_final = results_df_sorted.drop(['Graph_Nodes', 'Original_Index'], axis=1)
        
        print(f"✅ Prediction completed for {len(results_df_final)} samples")
        return results_df_final, graphs_sorted

class ResultManager:
    """结果管理器 - 负责保存结果和可视化"""
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def save_results(self, results_df: pd.DataFrame) -> Dict[str, str]:
        print("💾 Saving prediction results...")
        saved_files = {}

        # 1. 保存完整结果
        full_results_file = self.output_dir / "predictions_full.csv"
        results_df.to_csv(full_results_file, index=False)
        saved_files['full_results'] = str(full_results_file)
        print(f"📄 Full results saved to: {full_results_file}")

        # 2. 保存简化版结果
        simple_results = results_df[['Id', 'Predicted_Class', 'AVP_Probability']]
        simple_file = self.output_dir / "predictions_simple.csv"
        simple_results.to_csv(simple_file, index=False)
        saved_files['simple_results'] = str(simple_file)
        print(f"📋 Simple results saved to: {simple_file}")
        
        return saved_files

    def create_visualization(self, results_df: pd.DataFrame):
        print("📊 Creating visualization...")
        fig, axes = plt.subplots(1, 3, figsize=(21, 6))
        fig.suptitle('AVP Prediction Results Analysis', fontsize=18, fontweight='bold')

        # 图1: 预测类别分布 (饼图)
        class_counts = results_df['Predicted_Class'].value_counts()
        axes[0].pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', colors=['#ff7f7f', '#7fbf7f'], startangle=90)
        axes[0].set_title('Prediction Class Distribution', fontweight='bold')

        # 图2: AVP概率分布 (直方图)
        axes[1].hist(results_df['AVP_Probability'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1].axvline(x=0.5, color='red', linestyle='--', label='Decision Threshold')
        axes[1].set_xlabel('AVP Probability')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('AVP Probability Distribution', fontweight='bold')
        axes[1].legend()

        # 图3: 预测置信度分布 (条形图)
        prob = results_df['AVP_Probability']
        confidence_levels = {
            'High': (prob > 0.9) | (prob < 0.1),
            'Higher': ((prob >= 0.8) & (prob <= 0.9)) | ((prob >= 0.1) & (prob <= 0.2)),
            'Medium': ((prob >= 0.6) & (prob < 0.8)) | ((prob > 0.2) & (prob <= 0.4)),
            'Low': (prob > 0.4) & (prob < 0.6)
        }
        confidence_counts = {name: mask.sum() for name, mask in confidence_levels.items()}
        colors = {'High': 'green', 'Higher': 'blue', 'Medium': 'orange', 'Low': 'red'}
        
        bars = axes[2].bar(confidence_counts.keys(), confidence_counts.values(), color=colors.values(), alpha=0.8)
        for bar in bars:
            height = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width()/2., height, f'{height}\n({height/len(results_df)*100:.1f}%)', ha='center', va='bottom', fontweight='bold')
        
        axes[2].set_title('Prediction Confidence Distribution', fontweight='bold')
        axes[2].set_ylabel('Count')
        axes[2].set_ylim(0, max(confidence_counts.values()) * 1.15)
        
        legend_elements = [Patch(facecolor=color, label=f"{name} ({desc})") for (name, color), desc in zip(colors.items(), ['>0.9 or <0.1', '0.8-0.9 or 0.1-0.2', '0.6-0.8 or 0.2-0.4', '0.4-0.6'])]
        axes[2].legend(handles=legend_elements, loc='upper right', fontsize=9, title='Confidence Levels')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plot_file = self.output_dir / "prediction_analysis.png"
        plt.savefig(plot_file, dpi=600, bbox_inches='tight')
        plt.show()
        print(f"📈 Visualization saved to: {plot_file}")

    def print_summary(self, results_df: pd.DataFrame):
        print("\n📊 Prediction Summary")
        print("="*60)
        total = len(results_df)
        avp_count = (results_df['Predicted_Class'] == 'AVP').sum()
        print(f"Total sequences predicted: {total}")
        print(f"  - AVP: {avp_count} ({avp_count/total:.1%})")
        print(f"  - non-AVP: {total - avp_count} ({(total - avp_count)/total:.1%})")
        print("\n🔥 Top 5 AVP Predictions:")
        print(results_df[results_df['Predicted_Class'] == 'AVP'].head(5)[['Id', 'AVP_Probability']].to_string(index=False))


class EndToEndAVPPredictor:
    """端到端抗病毒肽预测器 - 主控制器"""
    def __init__(self, model_dir: str, device: str):
        print("🚀 Initializing End-to-End AVP Predictor...")
        self.device = torch.device('cuda' if device == 'auto' and torch.cuda.is_available() else 'cpu')
        print(f"🖥️ Using device: {self.device}")
        
        self.config = {
            'graph': {'distance_threshold': 4.0, 'use_esmc': True, 'use_aaindex': False},
            'data': {'output_base_dir': '3_Graph_Data'}
        }
        
        self.data_processor = DataProcessor()
        self.graph_builder = GraphBuilder(self.config)
        self.model_loader = ModelLoader(model_dir, self.device)
        self.models = self.model_loader.load_models()
        self.predictor = Predictor(self.models, self.device)
        print("✅ Predictor initialized successfully!")

    def predict(self, input_data: str, batch_size: int, create_visualization: bool, 
                grad_cam: bool = False) -> Tuple[pd.DataFrame, Dict[str, str]]:
        print("\n🚀 Starting End-to-End AVP Prediction...")
        output_dir = Path("5_Results_Predicted") / Path(input_data).stem
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n🔄 Step 1: Data Processing & Graph Building")
        csv_file = self.data_processor.prepare_input_data(input_data, output_dir)
        graph_file = self.graph_builder.build_graphs(csv_file, input_data)
        
        print("\n🔄 Step 2: Model Prediction")
        results_df, graphs = self.predictor.predict_graphs(graph_file, batch_size)
        
        print("\n🔄 Step 3: Results Processing")
        result_manager = ResultManager(output_dir)
        saved_files = result_manager.save_results(results_df)
        result_manager.print_summary(results_df)
        
        if create_visualization:
            print("\n🔄 Step 4: Creating Visualization")
            result_manager.create_visualization(results_df)
        
        # 新增：Grad-CAM可视化
        if grad_cam:
            print("\n🔄 Step 5: Generating Grad-CAM Visualizations")
            grad_cam_analyzer = GradCAMAnalyzer(self.models, self.device, output_dir)
            
            print(f"🎯 Generating Grad-CAM for all {len(results_df)} samples...")
            
            # 为每个预测样本生成Grad-CAM可视化
            successful_count = 0
            skipped_count = 0
            
            for idx, (_, row) in enumerate(results_df.iterrows()):
                graph = graphs[idx]
                seq_id = row['Id']
                sequence = row['Sequence']
                prob = row['AVP_Probability']
                pred = row['Predicted_Label']
                
                print(f"\n🔍 Processing {seq_id} ({idx+1}/{len(results_df)})")
                print(f"   Sequence length: {len(sequence)}, Graph nodes: {graph.num_nodes}")
                
                # 验证序列长度与图节点数是否一致
                if len(sequence) != graph.num_nodes:
                    print(f"⚠️ Skipping {seq_id}: sequence length ({len(sequence)}) != graph nodes ({graph.num_nodes})")
                    skipped_count += 1
                    continue
                
                try:
                    # 准备图数据
                    graph_gpu = graph.clone().to(self.device)
                    graph_gpu.batch = torch.zeros(graph_gpu.num_nodes, dtype=torch.long, device=self.device)
                    
                    # 计算Grad-CAM
                    importance = grad_cam_analyzer.run_ensemble_grad_cam(graph_gpu, pred)
                    
                    # 生成可视化（仅保存）
                    if len(importance) == len(sequence):
                        grad_cam_analyzer.visualize_sample(seq_id, sequence, prob, pred, importance)
                        successful_count += 1
                    else:
                        print(f"⚠️ Skipping visualization for {seq_id}: importance/sequence length mismatch")
                        skipped_count += 1
                        
                except Exception as e:
                    print(f"⚠️ Error processing {seq_id}: {e}")
                    skipped_count += 1
            
            print(f"\n📊 Grad-CAM Summary:")
            print(f"  ✅ Successfully processed: {successful_count}")
            print(f"  ⚠️ Skipped: {skipped_count}")
            print(f"  📁 Visualizations saved to: {grad_cam_analyzer.individual_plots_dir}")
            
        print("\n🎉 End-to-end prediction completed successfully!")
        return results_df, saved_files


def predict_avp(input_data: str, 
                model_dir: str = "4_Results_TR_RS/GAT_Comparison_4.0A/3Layer_GAT",
                batch_size: int = 64,
                create_visualization: bool = True,
                grad_cam: bool = False,
                device: str = 'auto') -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    主预测函数 - 简化接口
    
    Args:
        input_data: 输入数据路径（支持CSV和FASTA）
        model_dir: 训练好的GAT模型目录
        batch_size: 预测批处理大小
        create_visualization: 是否创建可视化
        grad_cam: 是否生成Grad-CAM可视化
        device: 计算设备
    
    Returns:
        results_df: 预测结果DataFrame
        saved_files: 保存的文件路径字典
    """
    try:
        predictor = EndToEndAVPPredictor(model_dir=model_dir, device=device)
        return predictor.predict(
            input_data=input_data,
            batch_size=batch_size,
            create_visualization=create_visualization,
            grad_cam=grad_cam
        )
    except Exception as e:
        print(f"\n❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(), {}


if __name__ == "__main__":
    try:
        results, files = predict_avp(
            input_data="1_Data/Processed_data_set/Final_merged_data_set/TS.csv",
            batch_size=256,
            grad_cam=True  # 启用Grad-CAM可视化
        )
        if not results.empty:
            print(f"\n✅ Prediction successful. Result files: {files}")
    except Exception as e:
        print(f"An error occurred in the main execution block: {e}")