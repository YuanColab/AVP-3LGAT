"""
Utility functions for feature extraction
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any
from loguru import logger
import pickle
import torch
from collections import Counter
import numpy as np

def setup_dataset_logging(dataset_name: str, config: Dict[str, Any], level: str = "INFO"):
    """
    为特定数据集设置日志系统，保存到 2_Log/2.2_Feature_extraction/dataset_name/ 目录
    
    Args:
        dataset_name: 数据集名称（从文件名提取）
        config: 配置字典，用于生成特征后缀
        level: 日志级别
    
    Returns:
        log_file: 日志文件路径
    """
    logger.remove()  # 移除所有现有的handler
    
    # 创建日志目录结构：2_Log/2.2_Feature_extraction/dataset_name/
    log_base_dir = "2_Log/2.2_Feature_extraction"
    log_dataset_dir = os.path.join(log_base_dir, dataset_name)
    os.makedirs(log_dataset_dir, exist_ok=True)
    
    # 生成特征后缀
    feature_suffix = ""
    if config['graph']['use_esmc']:
        feature_suffix += "ESMC"
    if config['graph']['use_aaindex']:
        if feature_suffix:
            feature_suffix += "+"
        feature_suffix += "AAIndex"
    
    # 生成距离后缀
    distance_suffix = f"_{config['graph']['distance_threshold']}A"
    
    # 生成完整的日志文件名（不包含时间戳）
    log_filename = f"{dataset_name}_{feature_suffix}{distance_suffix}.log"
    log_file = os.path.join(log_dataset_dir, log_filename)
    
    # 添加控制台输出
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    logger.add(sys.stdout, format=log_format, level=level, colorize=True)
    
    # 添加文件输出
    file_format = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    logger.add(
        log_file,
        format=file_format,
        level="DEBUG",
        rotation="50 MB",
        retention="30 days"
    )
    
    logger.info(f"Dataset-specific logging initialized: {log_file}")
    return log_file

def check_file_exists_and_valid(file_path: str, min_size: int = 0) -> bool:
    """检查文件是否存在且有效"""
    if not os.path.exists(file_path):
        return False
    try:
        file_size = os.path.getsize(file_path)
        return file_size > min_size
    except:
        return False

def filter_existing_files(ids: List[str], sequences: List[str], output_dir: str, 
                         file_extension: str, min_size: int = 0) -> Tuple[List[str], List[str], List[str]]:
    """筛选已存在的文件，返回需要处理的序列"""
    unprocessed_ids = []
    unprocessed_sequences = []
    existing_files = []
    
    for seq_id, sequence in zip(ids, sequences):
        file_path = os.path.join(output_dir, f"{seq_id}.{file_extension}")
        if check_file_exists_and_valid(file_path, min_size):
            existing_files.append(seq_id)
        else:
            unprocessed_ids.append(seq_id)
            unprocessed_sequences.append(sequence)
    
    return unprocessed_ids, unprocessed_sequences, existing_files

def format_time(seconds: float) -> str:
    """格式化时间显示"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.1f}h"

def create_output_directory(csv_file: str, suffix: str) -> str:
    """基于输入文件创建输出目录"""
    csv_path = Path(csv_file)
    output_dir = csv_path.parent / f"{csv_path.stem}{suffix}"
    os.makedirs(output_dir, exist_ok=True)
    return str(output_dir)

def get_device():
    """获取可用设备"""
    if torch.cuda.is_available():
        device = 'cuda'
        logger.info(f"CUDA available: {torch.cuda.get_device_name()}")
    else:
        device = 'cpu'
        logger.info("Using CPU")
    return device

def load_dataset(file_path: str):
    """加载保存的数据集"""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def inspect_graph_data(graphs, index=0, detailed=True):
    """
    检查图数据的详细信息
    
    Args:
        graphs: 图数据列表
        index: 要检查的图的索引，默认为0（第一个）
        detailed: 是否显示详细信息
    """
    if index >= len(graphs):
        print(f"❌ Error: Index {index} out of range, dataset only has {len(graphs)} samples")
        return
    
    graph = graphs[index]
    
    print(f"🔍 Inspecting graph {index + 1}")
    print("=" * 60)
    
    # 1. 基本信息
    print(f"📊 Basic Information:")
    print(f"  Graph data type: {type(graph)}")
    print(f"  Label: {graph.y.item()}")
    print(f"  Label type: {type(graph.y)} | Shape: {graph.y.shape}")
    
    # 2. 节点特征信息
    print(f"\n🔗 Node Features (graph.x):")
    print(f"  Feature matrix shape: {graph.x.shape}")
    print(f"  Number of nodes: {graph.x.shape[0]}")
    print(f"  Feature dimension: {graph.x.shape[1]}")
    print(f"  Data type: {graph.x.dtype}")
    print(f"  Device: {graph.x.device}")
    
    # 3. 边索引信息
    print(f"\n🌐 Edge Index (graph.edge_index):")
    print(f"  Edge index shape: {graph.edge_index.shape}")
    print(f"  Number of edges: {graph.edge_index.shape[1]}")
    print(f"  Data type: {graph.edge_index.dtype}")
    print(f"  Device: {graph.edge_index.device}")
    
    # 4. 统计信息
    print(f"\n📈 Statistics:")
    print(f"  Node feature value range: [{graph.x.min().item():.6f}, {graph.x.max().item():.6f}]")
    print(f"  Node feature mean: {graph.x.mean().item():.6f}")
    print(f"  Node feature std: {graph.x.std().item():.6f}")
    print(f"  Edge index range: [{graph.edge_index.min().item()}, {graph.edge_index.max().item()}]")
    
    # 5. 检查图的连通性
    num_nodes = graph.x.shape[0]
    max_node_index = graph.edge_index.max().item()
    min_node_index = graph.edge_index.min().item()
    print(f"  Edge index validity: {'✅ Valid' if max_node_index < num_nodes and min_node_index >= 0 else '❌ Invalid'}")
    
    if detailed:
        print(f"\n🔍 Detailed Information:")
        
        # 显示前几个节点的特征
        print(f"  First 5 node features:")
        for i in range(min(5, graph.x.shape[0])):
            feat_str = str(graph.x[i][:10].tolist())  # 只显示前10个特征
            if graph.x.shape[1] > 10:
                feat_str = feat_str[:-1] + ", ...]"
            print(f"    Node {i}: {feat_str}")
        
        # 显示前几条边
        print(f"\n  First 10 edges:")
        for i in range(min(10, graph.edge_index.shape[1])):
            src = graph.edge_index[0, i].item()
            dst = graph.edge_index[1, i].item()
            print(f"    Edge {i}: {src} -> {dst}")
        
        # 度分布统计
        degrees = []
        for node in range(num_nodes):
            degree = (graph.edge_index[0] == node).sum().item() + (graph.edge_index[1] == node).sum().item()
            degrees.append(degree)
        
        degree_counter = Counter(degrees)
        print(f"\n  Degree distribution:")
        print(f"    Average degree: {sum(degrees) / len(degrees):.2f}")
        print(f"    Max degree: {max(degrees)}")
        print(f"    Min degree: {min(degrees)}")
        print(f"    Degree distribution (top 5): {dict(list(degree_counter.most_common(5)))}")

def check_anomalies(graphs, dataset_name="Dataset", sample_size=100):
    """检查数据中的异常值"""
    sample_graphs = graphs[:sample_size] if len(graphs) > sample_size else graphs
    
    node_counts = []
    edge_counts = []
    feature_dims = []
    labels = []
    
    for graph in sample_graphs:
        node_counts.append(graph.x.shape[0])
        edge_counts.append(graph.edge_index.shape[1])
        feature_dims.append(graph.x.shape[1])
        labels.append(graph.y.item())
        
        # 检查NaN或无穷大
        if torch.isnan(graph.x).any():
            print(f"⚠️  Found NaN values in node features")
        if torch.isinf(graph.x).any():
            print(f"⚠️  Found infinite values in node features")
    
    print(f"🔍 {dataset_name} anomaly check (checked {len(sample_graphs)} samples):")
    print(f"  Node count range: {min(node_counts)} - {max(node_counts)}")
    print(f"  Edge count range: {min(edge_counts)} - {max(edge_counts)}")
    print(f"  Feature dimensions: {set(feature_dims)}")
    print(f"  Label distribution: {dict(Counter(labels))}")
    
    # 检查是否有孤立节点
    isolated_nodes_count = []
    for graph in sample_graphs[:min(100, len(sample_graphs))]:
        edge_index = graph.edge_index
        num_nodes = graph.x.shape[0]
        connected_nodes = set(edge_index.flatten().tolist())
        isolated = num_nodes - len(connected_nodes)
        isolated_nodes_count.append(isolated)
    
    if max(isolated_nodes_count) > 0:
        print(f"  ⚠️  Found isolated nodes, count range: {min(isolated_nodes_count)} - {max(isolated_nodes_count)}")
    else:
        print(f"  ✅ No isolated nodes found")

def analyze_graph_structures(graphs, dataset_name="Dataset", sample_size=500):
    """分析图结构特征"""
    sample_graphs = graphs[:sample_size] if len(graphs) > sample_size else graphs
    
    node_counts = [g.x.shape[0] for g in sample_graphs]
    edge_counts = [g.edge_index.shape[1] for g in sample_graphs]
    edge_densities = [edges / (nodes * (nodes - 1)) if nodes > 1 else 0 for nodes, edges in zip(node_counts, edge_counts)]
    
    print(f"📊 {dataset_name} graph structure statistics (based on {len(sample_graphs)} samples):")
    print(f"  Node count - Mean: {np.mean(node_counts):.2f}, Std: {np.std(node_counts):.2f}")
    print(f"  Edge count - Mean: {np.mean(edge_counts):.2f}, Std: {np.std(edge_counts):.2f}")
    print(f"  Edge density - Mean: {np.mean(edge_densities):.4f}, Std: {np.std(edge_densities):.4f}")
    
    # 检查异常值
    node_q75, node_q25 = np.percentile(node_counts, [75, 25])
    node_iqr = node_q75 - node_q25
    node_outliers = [n for n in node_counts if n < (node_q25 - 1.5 * node_iqr) or n > (node_q75 + 1.5 * node_iqr)]
    
    if node_outliers:
        print(f"  Node count outliers: {len(node_outliers)} samples (range: {min(node_outliers)}-{max(node_outliers)})")
    else:
        print(f"  ✅ No node count outliers")

def compare_samples(graphs, dataset_name="Dataset", num_samples=5):
    """对比多个样本的基本信息"""
    print(f"📊 {dataset_name} multi-sample comparison (first {num_samples} samples)")
    print("="*80)
    
    for i in range(min(num_samples, len(graphs))):
        graph = graphs[i]
        print(f"Sample {i+1}: nodes={graph.x.shape[0]:3d}, edges={graph.edge_index.shape[1]:4d}, "
              f"label={graph.y.item()}, feature_dim={graph.x.shape[1]}")

def validate_file_paths(file_configs: List[Tuple[str, int]]) -> bool:
    """验证文件路径是否存在"""
    missing_files = []
    
    print("🔍 Validating input files...")
    for file_path, label in file_configs:
        if os.path.exists(file_path):
            print(f"  ✅ {file_path} (label: {label})")
        else:
            print(f"  ❌ {file_path} - File not found")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️  Found {len(missing_files)} missing files:")
        for file in missing_files:
            print(f"    - {file}")
        print("\nPlease ensure these files exist before running the program.")
        return False
    
    return True

def get_file_info(file_configs: List[Tuple[str, int]]) -> Dict[str, Any]:
    """获取文件信息"""
    file_info = {
        'total_files': len(file_configs),
        'file_types': {},
        'labels': [],
        'total_size': 0
    }
    
    for file_path, label in file_configs:
        if os.path.exists(file_path):
            # 文件类型
            ext = os.path.splitext(file_path)[1].lower()
            file_info['file_types'][ext] = file_info['file_types'].get(ext, 0) + 1
            
            # 标签
            file_info['labels'].append(label)
            
            # 文件大小
            try:
                file_info['total_size'] += os.path.getsize(file_path)
            except:
                pass
    
    file_info['label_distribution'] = dict(Counter(file_info['labels']))
    file_info['total_size_mb'] = file_info['total_size'] / (1024 * 1024)
    
    return file_info