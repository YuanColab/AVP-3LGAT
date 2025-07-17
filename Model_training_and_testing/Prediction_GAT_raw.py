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


class Predictor:
    """预测器 - 负责模型预测"""
    def __init__(self, models: list, device: torch.device):
        self.models = models
        self.device = device

    def predict_graphs(self, graph_file: str, batch_size: int) -> pd.DataFrame:
        print("🔮 Making predictions...")
        graphs = load_dataset(graph_file)
        data_loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
        
        all_probs = []
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                batch_model_probs = [torch.softmax(model(batch.x, batch.edge_index, batch.batch)[0], dim=1)[:, 1] for model in self.models]
                ensemble_probs = torch.stack(batch_model_probs).mean(dim=0)
                all_probs.extend(ensemble_probs.cpu().numpy())

        results_df = pd.DataFrame({
            'Id': [g.seq_id for g in graphs],
            'Sequence': [g.sequence for g in graphs],
            'AVP_Probability': np.round(all_probs, 4)
        })
        results_df['Predicted_Label'] = (results_df['AVP_Probability'] > 0.5).astype(int)
        results_df['Predicted_Class'] = results_df['Predicted_Label'].map({0: 'non_AVP', 1: 'AVP'})
        return results_df.sort_values('AVP_Probability', ascending=False).reset_index(drop=True)


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

    def predict(self, input_data: str, batch_size: int, create_visualization: bool) -> Tuple[pd.DataFrame, Dict[str, str]]:
        print("\n🚀 Starting End-to-End AVP Prediction...")
        output_dir = Path("5_Results_Predicted") / Path(input_data).stem
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n🔄 Step 1: Data Processing & Graph Building")
        csv_file = self.data_processor.prepare_input_data(input_data, output_dir)
        graph_file = self.graph_builder.build_graphs(csv_file, input_data)
        
        print("\n🔄 Step 2: Model Prediction")
        results_df = self.predictor.predict_graphs(graph_file, batch_size)
        
        print("\n🔄 Step 3: Results Processing")
        result_manager = ResultManager(output_dir)
        saved_files = result_manager.save_results(results_df)
        result_manager.print_summary(results_df)
        
        if create_visualization:
            print("\n🔄 Step 4: Creating Visualization")
            result_manager.create_visualization(results_df)
            
        print("\n🎉 End-to-end prediction completed successfully!")
        return results_df, saved_files


def predict_avp(input_data: str, 
                model_dir: str = "4_Results_TR_RS/GAT_Comparison_4.0A/3Layer_GAT",
                batch_size: int = 64,
                create_visualization: bool = True,
                device: str = 'auto') -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    主预测函数 - 简化接口
    
    Args:
        input_data: 输入数据路径（支持CSV和FASTA）
        model_dir: 训练好的GAT模型目录
        batch_size: 预测批处理大小
        create_visualization: 是否创建可视化
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
            create_visualization=create_visualization
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
            batch_size=256
        )
        if not results.empty:
            print(f"\n✅ Prediction successful. Result files: {files}")
    except Exception as e:
        print(f"An error occurred in the main execution block: {e}")