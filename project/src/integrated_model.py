import pickle
from typing import List, Dict
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
import time
import logging
import argparse

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 检查GPU是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# 全连接网络
class FullyConnectedModel(nn.Module):
    def __init__(self, input_dim=1):
        super().__init__()
        # 使用 LayerNorm 替代 BatchNorm
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.25),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ELU(alpha=1.0),

            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.SiLU(),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        # 确保输入是二维的 [batch_size, features]
        if x.dim() == 1:
            x = x.unsqueeze(1)
        return self.network(x)


# CNN网络
class CNNModel(nn.Module):
    def __init__(self, input_dim=1, max_time_steps=300):
        super().__init__()
        self.max_time_steps = max_time_steps
        
        # 使用1D CNN来处理时间序列数据
        self.cnn = nn.Sequential(
            # 第一个卷积块
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # 第二个卷积块
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # 第三个卷积块
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.25),
            
            # 全局平均池化
            nn.AdaptiveAvgPool1d(1)
        )
        
        # 用于将CNN输出映射到波动率值
        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            
            nn.Linear(32, 1)
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        # 确保输入是二维的 [batch_size, features]
        if x.dim() == 1:
            x = x.unsqueeze(1)
            
        # 将输入转换为序列形式 [batch_size, 1, sequence_length]
        batch_size = x.size(0)
        
        # 创建时间序列表示
        # 将输入的时间点扩展为一个小的时间窗口
        seq_length = min(10, self.max_time_steps)  # 使用10个时间点的窗口
        input_seq = torch.zeros(batch_size, 1, seq_length, device=x.device)
        
        # 将输入的时间点放在序列的中间位置
        mid_idx = seq_length // 2
        input_seq[:, 0, mid_idx] = x.squeeze(-1) / self.max_time_steps  # 归一化时间
        
        # 通过CNN处理序列
        cnn_out = self.cnn(input_seq)
        # 展平输出
        cnn_out = cnn_out.view(batch_size, -1)
        
        # 通过回归器得到最终的波动率预测
        output = self.regressor(cnn_out)
        
        return output


# LSTM网络
class LSTMModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 输出层
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
            nn.Dropout(dropout),
            
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
            
            nn.Linear(16, 1)
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_normal_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)

    def forward(self, x):
        # 确保输入是二维的 [batch_size, features]
        if x.dim() == 1:
            x = x.unsqueeze(1)
        
        # 将输入扩展为序列形式 [batch_size, seq_len, features]
        # 对于单时间点预测，我们创建一个长度为1的序列
        x = x.unsqueeze(1)  # [batch_size, 1, features]
        
        # LSTM前向传播
        lstm_out, _ = self.lstm(x)
        
        # 取最后一个时间步的输出
        lstm_out = lstm_out[:, -1, :]  # [batch_size, hidden_dim]
        
        # 通过回归器得到最终的波动率预测
        output = self.regressor(lstm_out)
        
        return output


def monte_pricing(options: List[Dict[str, float]], model: nn.Module,
                  initial_price: torch.Tensor, r: float, h: float, N: int) -> torch.Tensor:
    """使用并行化蒙特卡洛模拟计算多个期权的价格"""
    # 获取所有期权的到期时间
    Ts = torch.tensor([opt['T'] for opt in options], device=device, dtype=torch.float32)
    max_T = int(Ts.max().item())
    num_options = len(options)

    # 使用对偶变量法减少方差（Antithetic Variates）
    # 生成一半的随机数，然后生成其负值作为对偶变量
    half_N = N // 2
    rand_mat_half = torch.randn(half_N, max_T, device=device) * torch.sqrt(torch.tensor(h, device=device))
    rand_mat = torch.cat([rand_mat_half, -rand_mat_half], dim=0)  # (N, max_T)

    # 2. 创建时间索引 (max_T, 1)
    time_indices = torch.arange(max_T, device=device).float().view(-1, 1)

    # 3. 计算所有时间步的波动率 (max_T, 1)
    # 在训练模式下计算波动率以保留梯度
    sigma_vec = model(time_indices).squeeze(1)  # (max_T,)

    # 4. 生成价格路径 (N x max_T)
    # 使用列表收集路径避免原地操作
    path_list = [torch.full((N,), initial_price[-1].item(), device=device)]

    # 向量化路径生成 - 避免原地操作
    for t in range(max_T - 1):
        S_t = path_list[-1]
        S_next = S_t + r * S_t * h + sigma_vec[t] * S_t * rand_mat[:, t]
        path_list.append(S_next)

    # 将路径列表转换为张量
    S = torch.stack(path_list, dim=1)  # (N, max_T)

    # 5. 提取到期价格 (N x num_options)
    expiry_indices = (Ts - 1).long()  # 转换为0-based索引
    expiry_indices = torch.clamp(expiry_indices, 0, max_T - 1)
    S_T = S.gather(1, expiry_indices.unsqueeze(0).expand(N, -1))

    # 6. 计算期权价格 (N x num_options)
    Ks = torch.tensor([opt['K'] for opt in options], device=device, dtype=torch.float32)
    payoffs = torch.clamp_min(S_T - Ks, 0.0)
    discount_factors = torch.exp(-r * Ts * h)
    option_prices = discount_factors * payoffs

    # 7. 计算蒙特卡洛平均值
    return option_prices.mean(dim=0)


def get_model(model_type):
    """根据模型类型获取相应的模型实例"""
    if model_type == "fc":
        return FullyConnectedModel()
    elif model_type == "cnn":
        return CNNModel()
    elif model_type == "lstm":
        return LSTMModel()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def main(model_type="fc", epochs=500):
    """主训练函数"""
    # 数据导入
    initial_price = torch.tensor([100.0] * 4, device=device)
    
    # Load data
    with open('../data/options_data_train_jump.pkl', 'rb') as f:
        options_data_train = pickle.load(f)
    
    with open('../data/options_data_test_jump.pkl', 'rb') as f:
        options_data_test = pickle.load(f)
        
    with open('../data/sigma_data_jump.pkl', 'rb') as f:
        sigma_data = pickle.load(f)

    for opt in options_data_train[0:10]:
        print(opt)

    # 训练部分
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    model = get_model(model_type).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    r = 0.05
    h = 1 / 252
    N = 1000000

    train_loss_history = []
    learning_rate_history = []

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        min_lr=1e-5
    )

    max_grad_norm = 5.0

    # 根据模型类型设置日志文件名
    # 根据模型类型设置日志文件名
    log_file_map = {
        "fc": f"../output/training_log_{model_type}.txt",
        "cnn": f"../output/training_log_{model_type}.txt",
        "lstm": f"../output/training_log_{model_type}.txt"
    }
    
    model_name_map = {
        "fc": "Fully Connected",
        "cnn": "CNN",
        "lstm": "LSTM"
    }

    print(f"开始训练 {model_name_map[model_type]} 模型...")
    # 记录训练开始时间
    start_time = time.time()
    
    # 打开日志文件
    with open(log_file_map[model_type], 'w', encoding='utf-8') as log_file:
        log_file.write(f"=== {model_name_map[model_type]} Model Training Log ===\n")
        log_file.write(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()

            # 记录蒙特卡洛定价开始时间
            pricing_start_time = time.time()
            predicted_prices = monte_pricing(options_data_train, model, initial_price, r, h, N)
            pricing_time = time.time() - pricing_start_time

            # 直接在GPU上创建真实价格张量
            true_prices = torch.tensor([opt['O_0'] for opt in options_data_train],
                                       dtype=torch.float32,
                                       device=device)

            loss = torch.mean((predicted_prices - true_prices) ** 2)
            loss.backward()

            # 在反向传播后检查GPU内存使用情况
            if device.type == 'cuda':
                torch.cuda.synchronize()  # 确保所有CUDA操作完成
                
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step(loss)

            current_lr = optimizer.param_groups[0]['lr']
            train_loss_history.append(loss.item())
            learning_rate_history.append(current_lr)

            # 每100个epoch打印一次详细信息（LSTM模型每10个epoch打印一次）
            print_interval = 10 if model_type == "lstm" else 100
            if (epoch + 1) % print_interval == 0:
                epoch_time = time.time() - start_time
                log_message = (f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}, LR: {current_lr:.6f}\n'
                              f'  蒙特卡洛定价耗时: {pricing_time:.2f}秒\n'
                              f'  累计训练时间: {epoch_time:.2f}秒\n')
                print(log_message.strip())
                log_file.write(log_message)
                
                # 显示GPU内存使用情况
                if device.type == 'cuda':
                    torch.cuda.synchronize()  # 确保所有CUDA操作完成
                    gpu_allocated = torch.cuda.memory_allocated(0) / 1e6  # 改为MB单位
                    gpu_reserved = torch.cuda.memory_reserved(0) / 1e6    # 改为MB单位
                    gpu_message = f'  GPU内存使用: 已分配 {gpu_allocated:.2f} MB, 已保留 {gpu_reserved:.2f} MB\n'
                    print(gpu_message.strip())
                    log_file.write(gpu_message)

        # 记录训练结束时间
        total_training_time = time.time() - start_time
        final_message = (f"\n训练完成!\n"
                        f"最终损失: {loss.item():.6f}\n"
                        f"总训练时间: {total_training_time:.2f}秒\n"
                        f"平均每epoch耗时: {total_training_time/epochs:.2f}秒\n")
        print(final_message)
        log_file.write(final_message)
        
        # 记录GPU信息
        if device.type == 'cuda':
            torch.cuda.synchronize()  # 确保所有CUDA操作完成
            gpu_allocated = torch.cuda.memory_allocated(0) / 1e6  # 改为MB单位
            gpu_reserved = torch.cuda.memory_reserved(0) / 1e6    # 改为MB单位
            gpu_info = (f"GPU信息:\n"
                       f"  设备名称: {torch.cuda.get_device_name(0)}\n"
                       f"  GPU内存使用: 已分配 {gpu_allocated:.2f} MB, 已保留 {gpu_reserved:.2f} MB\n")
            log_file.write(gpu_info)

    # 绘制损失曲线
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    plt.plot(train_loss_history, 'b-', label='Training Loss')
    plt.title('Training Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(learning_rate_history, 'r-', label='Learning Rate')
    plt.title('Learning Rate History')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('./output/training_history.png')
    plt.show()

    # 显示预测结果
    print("\n期权价格比较:")
    for i, opt in enumerate(options_data_train[0:10]):
        pred_price = predicted_prices[i].item()
        true_price = opt['O_0']
        diff = abs(pred_price - true_price)
        diff_pct = diff / true_price * 100
        print(f"期权 {i + 1}:")
        print(f"  到期天数: {opt['T']}天, 行权价: {opt['K']:.2f}")
        print(f"  预测价格: {pred_price:.4f}, 真实价格: {true_price:.4f}")
        print(f"  绝对差异: {diff:.4f}, 相对差异: {diff_pct:.2f}%")
        print()

    # 保存模型权重
    # 根据模型类型设置模型保存路径
    model_save_map = {
        "fc": f"../models/option_pricing_model_weights_fake_jump.pth",
        "cnn": f"../models/option_pricing_model_weights_fake_jump_CNN.pth",
        "lstm": f"../models/option_pricing_model_weights_fake_jump_LSTM.pth"
    }
    
    torch.save(model.state_dict(), model_save_map[model_type])
    print(f"模型已保存到 {model_save_map[model_type]}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练期权定价模型')
    parser.add_argument('--model', type=str, default='fc', choices=['fc', 'cnn', 'lstm'],
                        help='模型类型: fc (全连接), cnn (卷积网络), lstm (长短期记忆网络)')
    parser.add_argument('--epochs', type=int, default=500, help='训练轮数')
    
    args = parser.parse_args()
    main(model_type=args.model, epochs=args.epochs)