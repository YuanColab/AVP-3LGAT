import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GATConv, LayerNorm, TopKPooling
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch.nn import Linear, Dropout, BatchNorm1d
import torch.nn as nn

# ==================== 基础模型类 ====================

class BaseGNNWithGradCAM(torch.nn.Module):
    """
    所有GNN模型的基类，包含通用功能和Grad-CAM支持
    """
    
    def __init__(self, node_feature_dim, hidden_dim=128, output_dim=2, 
                 drop=0.3, k=0.8, add_self_loops=True):
        super(BaseGNNWithGradCAM, self).__init__()
        
        # 基础参数
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.drop = drop
        self.k = k
        self.add_self_loops = add_self_loops
        
        # Grad-CAM相关属性
        self.gradients = None
        self.activations = None
        self.node_importance = None
        self.edge_importance = None
        
        # 输入投影层
        self.input_projection = Linear(node_feature_dim, hidden_dim)
        self.input_bn = BatchNorm1d(hidden_dim)
        
        # TopK池化层
        self.topk_pool = TopKPooling(hidden_dim, ratio=k)
        
        # 分类器 - 多尺度池化后的特征维度: hidden_dim * 4
        classifier_input_dim = hidden_dim * 4
        self.classifier = nn.Sequential(
            Linear(classifier_input_dim, hidden_dim * 2),
            BatchNorm1d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            Dropout(drop),
            Linear(hidden_dim * 2, hidden_dim),
            BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            Dropout(drop * 0.5),
            Linear(hidden_dim, output_dim)
        )
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """权重初始化"""
        for module in self.modules():
            if isinstance(module, Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def activations_hook(self, grad):
        """Grad-CAM梯度钩子"""
        self.gradients = grad
    
    def process_input(self, x):
        """处理输入特征"""
        x = self.input_projection(x)
        x = self.input_bn(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.drop, training=self.training)
        return x
    
    def apply_pooling(self, features, edge_index, batch):
        """应用多尺度池化"""
        # 保存激活值用于Grad-CAM
        self.activations = features
        if features.requires_grad:
            features.register_hook(self.activations_hook)
        
        # TopK池化
        selected_x, selected_edge_index, _, selected_batch, _, _ = self.topk_pool(
            features, edge_index, None, batch
        )
        
        # 多尺度全局池化
        pooled_max = global_max_pool(features, batch)
        pooled_mean = global_mean_pool(features, batch)
        pooled_add = global_add_pool(features, batch)
        pooled_topk = global_mean_pool(selected_x, selected_batch)
        
        # 拼接四种池化方式的结果
        graph_features = torch.cat([
            pooled_max, pooled_mean, pooled_add, pooled_topk
        ], dim=1)
        
        return graph_features
    
    def classify(self, graph_features):
        """分类"""
        return self.classifier(graph_features)
    
    # ========== Grad-CAM相关方法 ==========
    def get_activations_gradient(self):
        """获取激活值的梯度"""
        return self.gradients
    
    def get_activations(self):
        """获取激活值"""
        return self.activations
    
    def compute_node_importance(self, target_class_idx=1):
        """计算节点重要性分数"""
        if self.gradients is None or self.activations is None:
            return None
        
        alpha = torch.mean(self.gradients, dim=0)
        node_importance = []
        for i in range(self.activations.shape[0]):
            importance_score = F.relu(torch.sum(alpha * self.activations[i])).item()
            node_importance.append(importance_score)
        
        if max(node_importance) > 0:
            node_importance = [score/max(node_importance) for score in node_importance]
        
        self.node_importance = node_importance
        return node_importance
    
    def compute_edge_importance(self, edge_index, target_class_idx=1):
        """计算边重要性分数"""
        if self.gradients is None or self.activations is None:
            return None
        
        node_importance = self.compute_node_importance(target_class_idx)
        if node_importance is None:
            return None
        
        edge_importance = []
        for i in range(edge_index.size(1)):
            source_idx = edge_index[0, i].item()
            target_idx = edge_index[1, i].item()
            base_importance = node_importance[source_idx] * node_importance[target_idx]
            edge_importance.append(base_importance)
        
        self.edge_importance = edge_importance
        return edge_importance

# ==================== 可配置层数的GAT模型类 ====================

class ConfigurableGATModel(BaseGNNWithGradCAM):
    """可配置层数的GAT模型 - 支持2层和3层对比"""
    
    def __init__(self, node_feature_dim, hidden_dim=128, output_dim=2, 
                 drop=0.3, heads=4, k=0.8, add_self_loops=True, num_layers=3):
        super(ConfigurableGATModel, self).__init__(node_feature_dim, hidden_dim, output_dim, drop, k, add_self_loops)
        
        self.heads = heads
        self.num_layers = num_layers
        
        # 验证层数
        if num_layers not in [2, 3]:
            raise ValueError(f"当前只支持2层或3层GAT，得到: {num_layers}")
        
        # 动态构建GAT层
        self.gat_convs = nn.ModuleList()
        self.gat_norms = nn.ModuleList()
        
        for i in range(num_layers):
            if i == num_layers - 1:
                # 最后一层：输出hidden_dim，单头不拼接
                gat_conv = GATConv(hidden_dim, hidden_dim, heads=1, 
                                  dropout=drop, add_self_loops=add_self_loops, concat=False)
            else:
                # 前面的层：多头拼接
                gat_conv = GATConv(hidden_dim, hidden_dim // heads, heads=heads, 
                                  dropout=drop, add_self_loops=add_self_loops, concat=True)
            
            self.gat_convs.append(gat_conv)
            self.gat_norms.append(LayerNorm(hidden_dim))
    
    def forward(self, x, edge_index, batch, return_attention=False):
        x = self.process_input(x)
        
        all_attentions = []  # 新增：存储各层注意力权重
        
        for i in range(self.num_layers):
            if return_attention:
                # 修改：调用GATConv时启用return_attention_weights
                x, attn = self.gat_convs[i](x, edge_index, return_attention_weights=True)
                all_attentions.append(attn)  # 保存(edge_index, attention_values)
            else:
                x = self.gat_convs[i](x, edge_index)
            
            x = self.gat_norms[i](x, batch)
            x = F.relu(x)
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.drop, training=self.training)
        
        if return_attention:
            return all_attentions  # 修改：直接返回注意力权重列表
        else:
            features = x
            graph_features = self.apply_pooling(features, edge_index, batch)
            output = self.classify(graph_features)
            return output, features
    
    def get_model_info(self):
        return {
            'model_name': f'{self.num_layers}Layer_GAT',
            'node_feature_dim': self.node_feature_dim,
            'hidden_dim': self.hidden_dim,
            'heads': self.heads,
            'num_layers': self.num_layers,
            'total_parameters': sum(p.numel() for p in self.parameters())
        }

# ==================== 保持原有的固定层数模型（向后兼容） ====================

class SingleGATModel(BaseGNNWithGradCAM):
    """原始的3层GAT模型（保持向后兼容性）"""
    
    def __init__(self, node_feature_dim, hidden_dim=128, output_dim=2, 
                 drop=0.3, heads=4, k=0.8, add_self_loops=True):
        super(SingleGATModel, self).__init__(node_feature_dim, hidden_dim, output_dim, drop, k, add_self_loops)
        
        self.heads = heads
        
        # 固定3层GAT
        self.gat_conv1 = GATConv(hidden_dim, hidden_dim // heads, heads=heads, 
                                dropout=drop, add_self_loops=add_self_loops, concat=True)
        self.gat_norm1 = LayerNorm(hidden_dim)
        
        self.gat_conv2 = GATConv(hidden_dim, hidden_dim // heads, heads=heads, 
                                dropout=drop, add_self_loops=add_self_loops, concat=True)
        self.gat_norm2 = LayerNorm(hidden_dim)
        
        self.gat_conv3 = GATConv(hidden_dim, hidden_dim, heads=1, 
                                dropout=drop, add_self_loops=add_self_loops, concat=False)
        self.gat_norm3 = LayerNorm(hidden_dim)
    
    def forward(self, x, edge_index, batch, return_attention=False):
        x = self.process_input(x)
        
        if return_attention:
            # 修改：三处GATConv调用均启用return_attention_weights
            x, attn1 = self.gat_conv1(x, edge_index, return_attention_weights=True)
            x = self.gat_norm1(x, batch)
            x = F.relu(x)
            x = F.dropout(x, p=self.drop, training=self.training)
            
            x, attn2 = self.gat_conv2(x, edge_index, return_attention_weights=True)
            x = self.gat_norm2(x, batch)
            x = F.relu(x)
            x = F.dropout(x, p=self.drop, training=self.training)
            
            x, attn3 = self.gat_conv3(x, edge_index, return_attention_weights=True)
            x = self.gat_norm3(x, batch)
            features = F.relu(x)
            
            return [attn1, attn2, attn3]  # 修改：返回三层注意力权重
        else:
            # 固定3层GAT
            x = self.gat_conv1(x, edge_index)
            x = self.gat_norm1(x, batch)
            x = F.relu(x)
            x = F.dropout(x, p=self.drop, training=self.training)
            
            x = self.gat_conv2(x, edge_index)
            x = self.gat_norm2(x, batch)
            x = F.relu(x)
            x = F.dropout(x, p=self.drop, training=self.training)
            
            x = self.gat_conv3(x, edge_index)
            x = self.gat_norm3(x, batch)
            features = F.relu(x)
            
            # 池化和分类
            graph_features = self.apply_pooling(features, edge_index, batch)
            output = self.classify(graph_features)
            
            return output, features
    
    def get_model_info(self):
        return {
            'model_name': 'Single_GAT_3Layer',
            'node_feature_dim': self.node_feature_dim,
            'hidden_dim': self.hidden_dim,
            'heads': self.heads,
            'total_parameters': sum(p.numel() for p in self.parameters())
        }

# ==================== 模型工厂函数 ====================

def create_model(node_feature_dim, num_layers=3, **kwargs):
    """模型工厂函数 - 创建可配置层数的GAT模型"""
    
    # 默认参数
    default_params = {
        'hidden_dim': 128,
        'output_dim': 2,
        'drop': 0.3,
        'heads': 4,
        'k': 0.7,
        'add_self_loops': True,
        'num_layers': num_layers
    }
    
    # 更新参数
    default_params.update(kwargs)
    
    return ConfigurableGATModel(node_feature_dim, **default_params)

# ==================== 便利函数 ====================

def create_2layer_gat(node_feature_dim, **kwargs):
    """创建2层GAT模型"""
    return create_model(node_feature_dim, num_layers=2, **kwargs)

def create_3layer_gat(node_feature_dim, **kwargs):
    """创建3层GAT模型"""
    return create_model(node_feature_dim, num_layers=3, **kwargs)

# 为了保持向后兼容性，保留原始函数名
def create_hybrid_model(node_feature_dim, **kwargs):
    """
    创建GAT模型（保持向后兼容性，默认3层）
    """
    return create_model(node_feature_dim, num_layers=3, **kwargs)

# ==================== 层数对比测试函数 ====================

def compare_layer_configurations(node_feature_dim, **kwargs):
    """对比2层和3层GAT的配置信息"""
    print("🔬 GAT Layer Configuration Comparison")
    print("="*60)
    
    configs = [
        {'num_layers': 2, 'description': '2层GAT - 轻量高效'},
        {'num_layers': 3, 'description': '3层GAT - 深度表达'}
    ]
    
    results = {}
    
    for config in configs:
        num_layers = config['num_layers']
        description = config['description']
        
        print(f"\n📊 {description}")
        print("-" * 40)
        
        try:
            # 创建模型
            model = create_model(node_feature_dim, num_layers=num_layers, **kwargs)
            
            # 获取模型信息
            info = model.get_model_info()
            
            print(f"   🏗️  Model: {info['model_name']}")
            print(f"   📊  Parameters: {info['total_parameters']:,}")
            print(f"   🧠  Hidden dim: {info['hidden_dim']}")
            print(f"   👁️  Heads: {info['heads']}")
            print(f"   🔗  Layers: {info['num_layers']}")
            
            # 计算每样本参数数
            params_per_sample = info['total_parameters'] / 6810  # 你的训练集大小
            print(f"   📈  Params/Sample: {params_per_sample:.1f}")
            
            # 保存结果
            results[f'{num_layers}layer'] = info
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n💡 使用建议:")
    print(f"   🎯 2层GAT: 适合快速实验，训练速度快，不易过拟合")
    print(f"   🎯 3层GAT: 表达能力更强，可能捕获更复杂的模式")
    print(f"")
    print(f"🔧 调用方式:")
    print(f"   # 2层GAT")
    print(f"   model_2layer = create_2layer_gat(node_feature_dim)")
    print(f"   # 3层GAT") 
    print(f"   model_3layer = create_3layer_gat(node_feature_dim)")
    
    return results

# ==================== 使用示例和测试 ====================

if __name__ == "__main__":
    print("🚀 Configurable GAT Model Test")
    print("="*50)
    
    # 模拟参数
    node_feature_dim = 1152  # ESMC特征维度
    
    # 对比不同层数配置
    print("📊 Testing 2-layer vs 3-layer GAT:")
    results = compare_layer_configurations(
        node_feature_dim, 
        hidden_dim=128, 
        heads=8,
        drop=0.3
    )
    
    # 测试前向传播
    print(f"\n🧪 Forward Pass Test:")
    print("-" * 30)
    
    for num_layers in [2, 3]:
        try:
            model = create_model(node_feature_dim, num_layers=num_layers, 
                               hidden_dim=128, heads=4)
            
            # 模拟数据
            num_nodes = 50
            num_edges = 100
            x = torch.randn(num_nodes, node_feature_dim)
            edge_index = torch.randint(0, num_nodes, (2, num_edges))
            batch = torch.zeros(num_nodes, dtype=torch.long)
            batch[num_nodes//2:] = 1
            
            with torch.no_grad():
                output, features = model(x, edge_index, batch)
                print(f"   ✅ {num_layers}-layer: Output {output.shape}, Features {features.shape}")
                
        except Exception as e:
            print(f"   ❌ {num_layers}-layer error: {e}")
    
    print(f"\n✨ Available Functions:")
    print(f"   create_2layer_gat() - 创建2层GAT")
    print(f"   create_3layer_gat() - 创建3层GAT")
    print(f"   create_model(num_layers=N) - 创建N层GAT")
    print(f"   compare_layer_configurations() - 对比不同层数配置")