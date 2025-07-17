"""
Simple Graph Dataset Builder - CSV Input Only (Optimized)
"""

import os
import time
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from Bio.PDB import PDBParser
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
from tqdm import tqdm
from loguru import logger
import pickle
import gc
from collections import Counter

# 导入处理器和工具函数
from Feature_extraction.aaindex import AAIndexProcessor
from Feature_extraction.esmc import ESMCEmbeddingProcessor
from Feature_extraction.esmfold import ESMFoldStructurePredictor
from Feature_extraction.utils import setup_dataset_logging, format_time


class SimpleGraphDatasetBuilder:
    """优化的图数据集构建器，专用于CSV文件输入"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化图数据集构建器
        
        Args:
            config: 包含所有图构建和特征提取配置的参数字典
        """
        # 设置默认参数
        default_config = {
            'graph': {
                'use_esmc': True,
                'use_aaindex': True,
                'distance_threshold': 8.0,
                'feature_fusion_strategy': 'concat'
            },
            'aaindex': {
                'aaindex_file': 'Feature_extraction/aaindex1.csv',
                'normalize': True,
                'normalization_method': 'fixed_range'
            },
            'esmc': {
                'device': 'auto',
                'model_name': 'esmc_600m',
                'batch_size': 5
            },
            'esmfold': {
                'device': 'auto',
                'batch_size': 5,
                'model_cache_dir': './model_cache'
            },
            'data': {
                'output_base_dir': '3_Graph_Data',
                'skip_existing': True
            }
        }
        
        # 合并参数
        self.config = self._merge_config(default_config, config)
        
        # 初始化组件
        self.parser = PDBParser(QUIET=True)
        self.processors = {}
        self.stats = self._init_stats()
        
        print("Graph builder initialized (strict mode - CSV input only)")
        print(f"Using features: ESMC={self.config['graph']['use_esmc']}, AAIndex={self.config['graph']['use_aaindex']}")
        print(f"Distance threshold: {self.config['graph']['distance_threshold']}Å")

    # ================================
    # 配置和初始化相关方法
    # ================================
    
    def _merge_config(self, default: Dict, user: Dict) -> Dict:
        """合并默认参数与用户参数"""
        result = default.copy()
        for key, value in user.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key].update(value)
            else:
                result[key] = value
        return result
    
    def _init_stats(self) -> Dict:
        """初始化统计信息"""
        return {
            'total_sequences': 0,
            'successful_graphs': 0,
            'failed_sequences': 0,
            'missing_pdb': 0,
            'missing_esmc': 0,
            'missing_aaindex': 0,
            'processing_errors': 0,
            'length_mismatch_errors': 0,
            'feature_dimensions': {},
            'esmc_extracted': 0,
            'esmfold_extracted': 0,
            'aaindex_extracted': 0
        }

    def setup_dataset_logging(self, dataset_name: str):
        """
        为特定数据集设置日志系统
        
        Args:
            dataset_name: 数据集名称（从文件名提取）
        
        Returns:
            log_file: 日志文件路径
        """
        return setup_dataset_logging(dataset_name, self.config)

    # ================================
    # 目录和处理器管理
    # ================================
    
    def get_output_directory(self, base_name: str) -> str:
        """
        生成输出目录名称
        
        Args:
            base_name: 基础名称
        
        Returns:
            输出目录路径
        """
        output_dir = os.path.join(
            self.config['data']['output_base_dir'],
            base_name
        )
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    
    def get_feature_directories(self, output_dir: str) -> Dict[str, str]:
        """生成特征提取目录"""
        return {
            'esmc_dir': os.path.join(output_dir, 'esmc_features'),
            'esmfold_dir': os.path.join(output_dir, 'esmfold_structures'),
            'aaindex_dir': os.path.join(output_dir, 'aaindex_features')
        }
    
    def get_processor(self, processor_type: str):
        """获取或创建处理器"""
        if processor_type not in self.processors:
            if processor_type == 'esmc':
                self.processors['esmc'] = ESMCEmbeddingProcessor(
                    device=self.config['esmc']['device']
                )
            elif processor_type == 'esmfold':
                self.processors['esmfold'] = ESMFoldStructurePredictor(
                    device=self.config['esmfold']['device'],
                    batch_size=self.config['esmfold']['batch_size']
                )
            elif processor_type == 'aaindex':
                self.processors['aaindex'] = AAIndexProcessor(
                    aaindex_file=self.config['aaindex']['aaindex_file']
                )
        
        return self.processors[processor_type]

    # ================================
    # 数据准备和验证
    # ================================
    
    def load_and_validate_csv(self, csv_file: str) -> pd.DataFrame:
        """
        加载和验证CSV文件
        
        Args:
            csv_file: CSV文件路径
        
        Returns:
            验证后的DataFrame
        """
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
        
        df = pd.read_csv(csv_file)
        
        # 验证必要的列
        required_columns = ['Id', 'Sequence']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"CSV file missing required columns: {missing_columns}")
        
        # 验证Label列
        if 'Label' not in df.columns:
            raise ValueError("CSV file missing Label column")
        
        # 移除空行和无效数据
        df = df.dropna(subset=['Id', 'Sequence', 'Label'])
        
        if df.empty:
            raise ValueError("No valid data found in CSV file")
        
        logger.info(f"Loaded {len(df)} sequences from {csv_file}")
        
        # 序列长度统计
        seq_lengths = df['Sequence'].str.len()
        logger.info(f"Sequence length range: {seq_lengths.min()}-{seq_lengths.max()}, average: {seq_lengths.mean():.1f}")
        
        # 标签分布
        label_counts = Counter(df['Label'])
        logger.info(f"Label distribution: {dict(label_counts)}")
        
        return df

    # ================================
    # 特征提取管理
    # ================================
    
    def extract_features_if_needed(self, csv_file: str, feature_dirs: Dict[str, str], 
                                 force_reprocess: bool = False) -> Dict[str, Any]:
        """根据需要提取特征"""
        extraction_stats = {}
        skip_existing = self.config['data']['skip_existing'] and not force_reprocess
        
        # 创建特征目录
        for dir_path in feature_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        # Extract ESMC features
        if self.config['graph']['use_esmc']:
            logger.info("Extracting ESMC embedding features...")
            processor = self.get_processor('esmc')
            result = processor.process_dataset(
                csv_file, feature_dirs['esmc_dir'], skip_existing=skip_existing
            )
            if isinstance(result, tuple):
                extraction_stats['esmc_success'] = result[0]
                extraction_stats['esmc_failed'] = result[1] if len(result) > 1 else 0
            else:
                extraction_stats['esmc_success'] = result
                extraction_stats['esmc_failed'] = 0
            
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        # Extract ESMFold structures
        logger.info("Extracting ESMFold structures...")
        processor = self.get_processor('esmfold')
        result = processor.process_dataset(
            csv_file, feature_dirs['esmfold_dir'], skip_existing=skip_existing
        )
        if isinstance(result, tuple):
            extraction_stats['esmfold_processed'] = result[0]
        else:
            extraction_stats['esmfold_processed'] = result
        
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Extract AAIndex features
        if self.config['graph']['use_aaindex']:
            logger.info("Extracting AAIndex features...")
            processor = self.get_processor('aaindex')
            result = processor.process_dataset(
                csv_file, feature_dirs['aaindex_dir'], 
                skip_existing=skip_existing,
                normalize=self.config['aaindex']['normalize'],
                normalization_method=self.config['aaindex']['normalization_method']
            )
            if isinstance(result, tuple):
                extraction_stats['aaindex_success'] = result[0]
                extraction_stats['aaindex_failed'] = result[1] if len(result) > 1 else 0
            else:
                extraction_stats['aaindex_success'] = result
                extraction_stats['aaindex_failed'] = 0
        
        return extraction_stats

    # ================================
    # 特征加载方法
    # ================================
    
    def load_esmc_embeddings(self, sequence_id: str, embeddings_folder: str) -> Optional[np.ndarray]:
        """加载ESMC嵌入"""
        try:
            embedding_file = os.path.join(embeddings_folder, f"{sequence_id}.pt")
            if not os.path.exists(embedding_file):
                return None
            
            embeddings = torch.load(embedding_file, map_location='cpu')
            if not isinstance(embeddings, torch.Tensor):
                return None
            
            embeddings = embeddings.numpy()
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            elif embeddings.ndim > 2:
                return None
            
            return embeddings.astype(np.float32)
        except Exception as e:
            logger.debug(f"Failed to load ESMC embeddings {sequence_id}: {e}")
            return None
    
    def load_aaindex_features(self, sequence_id: str, aaindex_folder: str) -> Optional[np.ndarray]:
        """加载AAIndex特征"""
        try:
            feature_file = os.path.join(aaindex_folder, f"{sequence_id}.npy")
            if not os.path.exists(feature_file):
                return None
            
            features = np.load(feature_file)
            return features.astype(np.float32)
        except Exception as e:
            logger.debug(f"Failed to load AAIndex features {sequence_id}: {e}")
            return None
    
    def extract_coordinates_from_pdb(self, pdb_file: str) -> Optional[np.ndarray]:
        """从PDB文件提取坐标"""
        try:
            if not os.path.exists(pdb_file):
                return None
                
            structure = self.parser.get_structure('protein', pdb_file)
            coordinates = []
            
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if 'CA' in residue:
                            coord = residue['CA'].get_coord()
                            coordinates.append(coord)
            
            if not coordinates:
                return None
                
            return np.array(coordinates, dtype=np.float32)
        except Exception as e:
            logger.debug(f"Failed to extract coordinates from PDB {pdb_file}: {e}")
            return None

    # ================================
    # 特征处理和图构建
    # ================================
    
    def combine_node_features(self, esmc_features: Optional[np.ndarray], 
                            aaindex_features: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """合并节点特征 - 严格模式，要求精确匹配"""
        features_list = []
        feature_info = []
        
        if self.config['graph']['use_esmc'] and esmc_features is not None:
            features_list.append(esmc_features)
            feature_info.append(f"ESMC({esmc_features.shape[1]})")
        
        if self.config['graph']['use_aaindex'] and aaindex_features is not None:
            features_list.append(aaindex_features)
            feature_info.append(f"AAIdx({aaindex_features.shape[1]})")
        
        if not features_list:
            logger.debug("No available features")
            return None
        
        # 检查所有特征序列长度是否一致
        lengths = [f.shape[0] for f in features_list]
        unique_lengths = set(lengths)
        
        if len(unique_lengths) > 1:
            # 长度不一致时抛出错误
            feature_length_info = dict(zip(feature_info, lengths))
            raise ValueError(
                f"Inconsistent feature lengths, strict match required: {feature_length_info}"
            )
        
        # 长度一致，直接使用
        min_length = lengths[0]
        logger.debug(f"All feature lengths consistent: {min_length}")
        
        # 拼接特征
        combined_features = np.concatenate(features_list, axis=1)
        
        # 存储特征维度信息
        feature_dim_key = "+".join(feature_info)
        self.stats['feature_dimensions'][feature_dim_key] = combined_features.shape[1]
        
        logger.debug(f"Feature combination complete: sequence_length={min_length}, feature_dim={combined_features.shape[1]}")
        
        return combined_features
    
    def compute_distance_matrix(self, coordinates: np.ndarray) -> np.ndarray:
        """计算距离矩阵"""
        diff = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff**2, axis=2))
        return distances
    
    def create_edges_from_distances(self, distance_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """从距离创建边"""
        adjacency = (distance_matrix < self.config['graph']['distance_threshold']) & (distance_matrix > 0)
        edge_indices = np.where(adjacency)
        
        if len(edge_indices[0]) == 0:
            return np.array([[], []], dtype=np.int64), np.array([])
        
        edge_index = np.array([edge_indices[0], edge_indices[1]], dtype=np.int64)
        edge_attr = distance_matrix[edge_indices].astype(np.float32)
        
        return edge_index, edge_attr
    
    def validate_and_align_data(self, seq_id: str, sequence: str, 
                               node_features: np.ndarray, 
                               coordinates: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """验证数据长度一致性 - 严格模式，不截断，不使用虚拟坐标"""
        original_length = len(sequence)
        feature_length = node_features.shape[0]
        coord_length = coordinates.shape[0]
        
        # 严格验证 - 所有长度必须精确匹配
        if feature_length != original_length:
            raise ValueError(
                f"{seq_id}: Feature length({feature_length}) does not match sequence length({original_length}), "
                f"strict correspondence required"
            )
        
        if coord_length != feature_length:
            raise ValueError(
                f"{seq_id}: PDB coordinate length({coord_length}) does not match feature length({feature_length}), "
                f"strict correspondence required"
            )
        
        if coord_length != original_length:
            raise ValueError(
                f"{seq_id}: PDB coordinate length({coord_length}) does not match sequence length({original_length}), "
                f"strict correspondence required"
            )
        
        # 最终验证：确保所有数据长度一致
        final_length = original_length
        assert node_features.shape[0] == final_length, f"{seq_id}: Node feature length validation failed"
        assert coordinates.shape[0] == final_length, f"{seq_id}: Coordinate length validation failed"
        
        logger.debug(f"{seq_id}: Data validation passed, strict match length={final_length}")
        
        return node_features, coordinates, final_length

    # ================================
    # 图创建核心方法
    # ================================
    
    def create_graph_from_data(self, csv_row: pd.Series, 
                              pdb_folder: str, 
                              esmc_folder: Optional[str] = None,
                              aaindex_folder: Optional[str] = None) -> Optional[Data]:
        """从数据创建图 - 严格模式，要求所有数据完整且匹配"""
        try:
            seq_id = str(csv_row['Id'])
            label = int(csv_row['Label'])
            sequence = csv_row.get('Sequence', '')
            original_sequence_length = len(sequence)
            
            if original_sequence_length == 0:
                raise ValueError(f"{seq_id}: Empty sequence")
            
            # 加载ESMC特征
            esmc_features = None
            if self.config['graph']['use_esmc'] and esmc_folder:
                esmc_features = self.load_esmc_embeddings(seq_id, esmc_folder)
                if esmc_features is None:
                    self.stats['missing_esmc'] += 1
                    raise ValueError(f"{seq_id}: ESMC features missing")
                else:
                    # 严格验证ESMC特征长度
                    if esmc_features.shape[0] != original_sequence_length:
                        raise ValueError(
                            f"{seq_id}: ESMC feature length({esmc_features.shape[0]}) "
                            f"does not match sequence length({original_sequence_length})"
                        )
            
            # 加载AAIndex特征
            aaindex_features = None
            if self.config['graph']['use_aaindex'] and aaindex_folder:
                aaindex_features = self.load_aaindex_features(seq_id, aaindex_folder)
                if aaindex_features is None:
                    self.stats['missing_aaindex'] += 1
                    # 尝试即时生成AAIndex特征
                    if sequence:
                        try:
                            processor = self.get_processor('aaindex')
                            aaindex_features = processor.extract_features_from_sequence(sequence)
                            if self.config['aaindex']['normalize']:
                                aaindex_features = processor.normalize_features(
                                    aaindex_features, 
                                    self.config['aaindex']['normalization_method']
                                )
                            logger.debug(f"{seq_id}: Generated AAIndex features on-the-fly")
                            
                            # 验证即时生成的特征长度
                            if aaindex_features.shape[0] != original_sequence_length:
                                raise ValueError(
                                    f"{seq_id}: On-the-fly generated AAIndex feature length({aaindex_features.shape[0]}) "
                                    f"does not match sequence length({original_sequence_length})"
                                )
                        except Exception as e:
                            logger.debug(f"{seq_id}: Failed to generate AAIndex features on-the-fly: {e}")
                            raise ValueError(f"{seq_id}: AAIndex features missing and cannot generate on-the-fly")
                else:
                    # 严格验证AAIndex特征长度
                    if aaindex_features.shape[0] != original_sequence_length:
                        raise ValueError(
                            f"{seq_id}: AAIndex feature length({aaindex_features.shape[0]}) "
                            f"does not match sequence length({original_sequence_length})"
                        )
            
            # 合并节点特征
            node_features = self.combine_node_features(esmc_features, aaindex_features)
            if node_features is None:
                raise ValueError(f"{seq_id}: Feature combination failed, no available features")
            
            # 加载坐标 - 严格要求存在
            pdb_file = os.path.join(pdb_folder, f"{seq_id}.pdb")
            coordinates = self.extract_coordinates_from_pdb(pdb_file)
            if coordinates is None:
                self.stats['missing_pdb'] += 1
                raise ValueError(f"{seq_id}: PDB file missing or no valid coordinates")
            
            # 严格验证和数据对齐
            node_features, coordinates, final_length = self.validate_and_align_data(
                seq_id, sequence, node_features, coordinates
            )
            
            # 最终验证
            if final_length == 0:
                raise ValueError(f"{seq_id}: Final length is 0")
            
            if final_length != original_sequence_length:
                raise ValueError(
                    f"{seq_id}: Final length({final_length}) does not match original sequence length({original_sequence_length})"
                )
            
            # 计算边
            distance_matrix = self.compute_distance_matrix(coordinates)
            edge_index, edge_attr = self.create_edges_from_distances(distance_matrix)
            
            # 创建图
            x = torch.tensor(node_features, dtype=torch.float32)
            edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
            edge_attr_tensor = torch.tensor(edge_attr.reshape(-1, 1), dtype=torch.float32) if edge_attr.size > 0 else torch.empty(0, 1)
            y = torch.tensor([label], dtype=torch.long)
            
            data = Data(
                x=x,
                edge_index=edge_index_tensor,
                edge_attr=edge_attr_tensor,
                y=y,
                seq_id=seq_id,
                sequence=sequence,
                original_length=original_sequence_length,
                final_length=final_length,
                length=final_length,
                num_nodes=final_length,
                num_edges=edge_index.shape[1] if edge_index.size > 0 else 0
            )
            
            self.stats['successful_graphs'] += 1
            return data
            
        except ValueError as ve:
            # 长度不匹配和其他验证错误
            if "does not match" in str(ve):
                self.stats['length_mismatch_errors'] += 1
            logger.error(f"Data validation failed {seq_id}: {ve}")
            self.stats['processing_errors'] += 1
            return None
            
        except Exception as e:
            # 其他意外错误
            logger.error(f"Graph creation failed {seq_id}: {e}")
            self.stats['processing_errors'] += 1
            return None

    # ================================
    # 批量图构建
    # ================================
    
    def build_graphs_from_csv(self, csv_file: str, 
                             pdb_folder: str,
                             esmc_folder: Optional[str] = None,
                             aaindex_folder: Optional[str] = None) -> List[Data]:
        """从CSV文件构建图 - 严格模式版本"""
        logger.info(f"Starting graph construction: {csv_file}")
        
        df = self.load_and_validate_csv(csv_file)
        self.stats['total_sequences'] = len(df)
        graphs = []
        start_time = time.time()
        
        # 生成特征配置字符串
        features = []
        if self.config['graph']['use_esmc']:
            features.append('ESMC')
        if self.config['graph']['use_aaindex']:
            features.append('AAIdx')
        feature_str = '+'.join(features) if features else 'None'
        
        progress_bar = tqdm(
            total=len(df), 
            desc=f"Building graphs-strict mode ({feature_str}, {self.config['graph']['distance_threshold']}Å)", 
            unit="graph",
            ncols=None,
            leave=True
        )
        
        for idx, row in df.iterrows():
            iter_start = time.time()
            graph = self.create_graph_from_data(row, pdb_folder, esmc_folder, aaindex_folder)
            
            if graph is not None:
                graphs.append(graph)
                success = True
            else:
                success = False
                self.stats['failed_sequences'] += 1
            
            # 计算进度统计
            iter_time = time.time() - iter_start
            elapsed_total = time.time() - start_time
            avg_time_per_graph = elapsed_total / (idx + 1)
            eta_seconds = avg_time_per_graph * (len(df) - idx - 1)
            
            progress_bar.update(1)
            progress_bar.set_postfix({
                'success': f"{len(graphs)}/{idx+1}",
                'time': f"{iter_time:.3f}s",
                'ETA': format_time(eta_seconds),
                'status': "✓" if success else "✗"
            })
        
        progress_bar.close()
        
        total_time = time.time() - start_time
        logger.info(f"Graph construction complete: {len(graphs)}/{len(df)} successful")
        logger.info(f"Total time: {format_time(total_time)}")
        
        # 详细统计报告
        if graphs:
            self._report_statistics()
        
        return graphs
    
    def _report_statistics(self):
        """报告详细统计信息"""
        logger.info("Detailed statistics:")
        logger.info(f"  Successful graphs: {self.stats['successful_graphs']}")
        logger.info(f"  Failed sequences: {self.stats['failed_sequences']}")
        logger.info(f"  Length mismatch errors: {self.stats['length_mismatch_errors']}")
        logger.info(f"  Missing PDB: {self.stats['missing_pdb']}")
        logger.info(f"  Missing ESMC: {self.stats['missing_esmc']}")
        logger.info(f"  Missing AAIndex: {self.stats['missing_aaindex']}")
        logger.info(f"  Processing errors: {self.stats['processing_errors']}")
        
        if self.stats['feature_dimensions']:
            logger.info(f"  Feature dimensions: {self.stats['feature_dimensions']}")

    # ================================
    # 验证方法
    # ================================
    
    def validate_graphs(self, graphs: List[Data]) -> Dict[str, Any]:
        """验证生成的图数据"""
        validation_stats = {
            'total_graphs': len(graphs),
            'node_count_range': {},
            'edge_count_range': {},
            'feature_dimension': {},
            'label_distribution': {},
            'anomalies': []
        }
        
        if not graphs:
            logger.warning("No graphs to validate")
            return validation_stats
        
        node_counts = [graph.num_nodes for graph in graphs]
        edge_counts = [graph.num_edges for graph in graphs]
        feature_dims = [graph.x.shape[1] for graph in graphs]
        labels = [graph.y.item() for graph in graphs]
        
        validation_stats['node_count_range'] = {
            'min': min(node_counts),
            'max': max(node_counts),
            'mean': np.mean(node_counts)
        }
        
        validation_stats['edge_count_range'] = {
            'min': min(edge_counts),
            'max': max(edge_counts),
            'mean': np.mean(edge_counts)
        }
        
        validation_stats['feature_dimension'] = {
            'unique_dims': list(set(feature_dims)),
            'most_common': max(set(feature_dims), key=feature_dims.count)
        }
        
        validation_stats['label_distribution'] = dict(Counter(labels))
        
        # 检查异常
        for i, graph in enumerate(graphs):
            if torch.isnan(graph.x).any():
                validation_stats['anomalies'].append(f"Graph {i}: NaN in node features")
            if torch.isinf(graph.x).any():
                validation_stats['anomalies'].append(f"Graph {i}: Inf in node features")
            if graph.num_edges == 0:
                validation_stats['anomalies'].append(f"Graph {i}: No edges")
        
        logger.info("Graph validation results:")
        logger.info(f"  Total graphs: {validation_stats['total_graphs']}")
        logger.info(f"  Node count range: {validation_stats['node_count_range']['min']}-{validation_stats['node_count_range']['max']}")
        logger.info(f"  Edge count range: {validation_stats['edge_count_range']['min']}-{validation_stats['edge_count_range']['max']}")
        logger.info(f"  Feature dimensions: {validation_stats['feature_dimension']['unique_dims']}")
        logger.info(f"  Label distribution: {validation_stats['label_distribution']}")
        
        if validation_stats['anomalies']:
            logger.warning(f"  Found {len(validation_stats['anomalies'])} anomalies")
            for anomaly in validation_stats['anomalies'][:5]:  # 只显示前5个
                logger.warning(f"    {anomaly}")
        else:
            logger.info("  ✅ No anomalies detected")
        
        return validation_stats

    # ================================
    # 主要管道方法
    # ================================
    
    def automated_pipeline(self, csv_file: str, 
                          output_name: str,
                          force_reprocess: bool = False, 
                          save_dataset: bool = True,
                          validate_output: bool = True) -> Tuple[List[Data], Dict[str, Any]]:
        """
        自动化管道：CSV输入 + 特征提取 + 图构建 + 验证
        
        Args:
            csv_file: CSV文件路径
            output_name: 输出数据集名称
            force_reprocess: 是否强制重新处理
            save_dataset: 是否保存数据集
            validate_output: 是否验证输出
        
        Returns:
            graphs: 构建的图列表
            stats: 处理统计信息
        """
        pipeline_start_time = time.time()
        
        # 1. 设置数据集特定的日志系统
        log_file = self.setup_dataset_logging(output_name)
        
        logger.info("="*80)
        logger.info("Automated CSV Graph Construction Pipeline (Strict Mode)")
        logger.info("="*80)
        logger.info(f"Log file: {log_file}")
        logger.info(f"Input CSV: {csv_file}")
        
        # 2. 验证输入文件并加载数据
        logger.info("Step 1: Loading and validating CSV data")
        logger.info("="*60)
        
        df = self.load_and_validate_csv(csv_file)
        
        # 3. 创建输出目录
        output_dir = self.get_output_directory(output_name)
        feature_dirs = self.get_feature_directories(output_dir)
        
        logger.info(f"Output directory: {output_dir}")
        
        # 显示配置
        features = []
        if self.config['graph']['use_esmc']:
            features.append('ESMC')
        if self.config['graph']['use_aaindex']:
            features.append('AAIndex')
        feature_config = '+'.join(features) if features else 'None'
        
        logger.info(f"Feature configuration: {feature_config}")
        logger.info(f"Distance threshold: {self.config['graph']['distance_threshold']}Å")
        
        # 4. 特征提取
        logger.info("Step 2: Extracting features")
        logger.info("="*60)
        
        extraction_stats = self.extract_features_if_needed(csv_file, feature_dirs, force_reprocess=force_reprocess)
        logger.info(f"Feature extraction results: {extraction_stats}")
        
        # 5. 构建图
        logger.info("Step 3: Building graphs from CSV")
        logger.info("="*60)
        
        graphs = self.build_graphs_from_csv(
            csv_file, 
            feature_dirs['esmfold_dir'],  # PDB文件夹
            feature_dirs['esmc_dir'],     # ESMC特征文件夹  
            feature_dirs['aaindex_dir']   # AAIndex特征文件夹
        )
        
        logger.info(f"Graph building complete: {len(graphs)} graphs created")
        
        # 6. 验证图
        if validate_output:
            logger.info("Step 4: Validating output graphs")
            logger.info("="*60)
            
            validation_stats = self.validate_graphs(graphs)
            logger.info(f"Graph validation results: {validation_stats}")
        else:
            validation_stats = {}
        
        # 7. 保存数据集
        if save_dataset:
            logger.info("Step 5: Saving dataset")
            logger.info("="*60)
            
            # 生成标准化文件名
            feature_suffix = ""
            if self.config['graph']['use_esmc']:
                feature_suffix += "ESMC"
            if self.config['graph']['use_aaindex']:
                if feature_suffix:
                    feature_suffix += "+"
                feature_suffix += "AAIndex"
            
            distance_suffix = f"_{self.config['graph']['distance_threshold']}A"
            output_file = os.path.join(output_dir, f"{output_name}_{feature_suffix}{distance_suffix}.pkl")
            
            # 使用pickle保存以保持一致性
            with open(output_file, 'wb') as f:
                pickle.dump(graphs, f)
            
            logger.info(f"Dataset saved: {output_file}")
        
        total_pipeline_time = time.time() - pipeline_start_time
        logger.info(f"Pipeline complete: {len(graphs)} graphs processed, total time {format_time(total_pipeline_time)}")
        
        # 构建返回的统计信息
        final_stats = {
            'total_sequences': self.stats['total_sequences'],
            'successful_graphs': len(graphs),
            'failed_sequences': self.stats['failed_sequences'],
            'extraction_stats': extraction_stats,
            'validation_stats': validation_stats,
            'total_time': total_pipeline_time,
            'feature_dimensions': self.stats['feature_dimensions']
        }
        
        return graphs, final_stats


# ================================
# 便利函数
# ================================

def build_graphs_from_files(file_configs: List[Tuple[str, Optional[int]]], 
                           output_name: str,
                           config: Dict[str, Any],
                           force_reprocess: bool = False,
                           save_dataset: bool = True,
                           validate_output: bool = True) -> Tuple[List[Data], Dict[str, Any]]:
    """
    从CSV文件构建图的便利函数
    
    Args:
        file_configs: 文件配置列表，现在只支持单个CSV文件 [(csv_file, None)]
        output_name: 输出数据集名称
        config: 配置字典
        force_reprocess: 是否强制重新处理
        save_dataset: 是否保存数据集
        validate_output: 是否验证输出
    
    Returns:
        graphs: 构建的图列表
        stats: 处理统计信息
    """
    if len(file_configs) != 1:
        raise ValueError("Only single CSV file input is supported")
    
    csv_file, _ = file_configs[0]  # 忽略label参数，使用CSV内的Label列
    
    builder = SimpleGraphDatasetBuilder(config)
    graphs, stats = builder.automated_pipeline(
        csv_file, output_name, 
        force_reprocess=force_reprocess, 
        save_dataset=save_dataset,
        validate_output=validate_output
    )
    
    return graphs, stats