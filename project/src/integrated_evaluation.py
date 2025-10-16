import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import torch.nn as nn
import os
import time
import argparse
import pickle

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 设置非交互式后端
plt.switch_backend('Agg')

# 设置英文字体以避免中文显示问题
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Bitstream Vera Sans']
plt.rcParams['axes.unicode_minus'] = False

# 检查GPU是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1e6:.2f} MB")


# 辅助函数：将数据移动到指定设备
def to_device(data, device):
    """将数据转移到指定设备"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    return data


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


def monte_pricing_fake_data(options, model, initial_price, r, h, N):
    """使用并行化蒙特卡洛模拟计算多个期权的价格(无梯度计算) - 用于假数据"""
    # 禁用梯度计算
    with torch.no_grad():
        # 获取所有期权的到期时间
        Ts = torch.tensor([opt['T'] for opt in options], device=device, dtype=torch.float32)
        max_T = int(Ts.max().item())
        num_options = len(options)

        # 1. 生成随机数矩阵 (N x max_T)
        rand_mat = torch.randn(N, max_T, device=device) * torch.sqrt(torch.tensor(h, device=device))

        # 2. 创建时间索引 (max_T, 1) - 确保在正确设备上
        time_indices = torch.arange(max_T, device=device).float().view(-1, 1)

        # 3. 计算所有时间步的波动率 (max_T, 1) - 使用模型评估模式
        model.eval()
        sigma_vec = model(time_indices).squeeze(1)  # (max_T,)

        # 4. 预分配路径张量并生成价格路径
        S = torch.zeros((N, max_T), device=device)
        S[:, 0] = initial_price[-1].item()  # 所有路径起始于初始价格

        # 生成路径 - 无梯度计算
        for t in range(max_T - 1):
            # 欧拉法更新路径
            S[:, t + 1] = S[:, t] + r * S[:, t] * h + sigma_vec[t] * S[:, t] * rand_mat[:, t]

        # 5. 提取到期价格 (N x num_options)
        # 计算到期日对应的索引(注意:Ts是天数)
        expiry_indices = (Ts - 1).long()  # 转换为0-based索引
        expiry_indices = torch.clamp(expiry_indices, 0, max_T - 1)

        # 使用torch.gather提取每个期权对应到期日的股价
        S_T = torch.gather(S, 1, expiry_indices.view(1, -1).expand(N, -1))

        # 6. 计算期权价格 (num_options,)
        Ks = torch.tensor([opt['K'] for opt in options], device=device, dtype=torch.float32)
        payoffs = torch.clamp_min(S_T - Ks, 0.0)
        discount_factors = torch.exp(-r * Ts * h)
        option_prices = discount_factors * payoffs.mean(dim=0)

        return option_prices.cpu()  # 返回CPU数据便于后续可视化


def monte_pricing_real_data(options, model, initial_price, r, h, N):
    """使用并行化蒙特卡洛模拟计算多个期权的价格(无梯度计算) - 用于真实数据"""
    # 禁用梯度计算
    with torch.no_grad():
        # 获取所有期权的到期时间
        Ts = torch.tensor([opt['T'] for opt in options], device=device, dtype=torch.float32)
        max_T = int(Ts.max().item())
        num_options = len(options)

        # 使用对偶变量法减少方差（Antithetic Variates）
        # 生成一半的随机数，然后生成其负值作为对偶变量
        half_N = N // 2
        rand_mat_half = torch.randn(half_N, max_T, device=device) * torch.sqrt(torch.tensor(h, device=device))
        rand_mat = torch.cat([rand_mat_half, -rand_mat_half], dim=0)  # (N, max_T)

        # 2. 创建时间索引 (max_T, 1) - 确保在正确设备上
        time_indices = torch.arange(max_T, device=device).float().view(-1, 1)

        # 3. 计算所有时间步的波动率 (max_T, 1) - 使用模型评估模式
        model.eval()
        sigma_vec = model(time_indices).squeeze(1)  # (max_T,)

        # 4. 预分配路径张量并生成价格路径
        S = torch.zeros((N, max_T), device=device)
        S[:, 0] = initial_price[-1].item()  # 所有路径起始于初始价格

        # 生成路径 - 无梯度计算
        for t in range(max_T - 1):
            # 欧拉法更新路径
            S[:, t + 1] = S[:, t] + r * S[:, t] * h + sigma_vec[t] * S[:, t] * rand_mat[:, t]

        # 5. 提取到期价格 (N x num_options)
        # 计算到期日对应的索引(注意:Ts是天数)
        expiry_indices = (Ts - 1).long()  # 转换为0-based索引
        expiry_indices = torch.clamp(expiry_indices, 0, max_T - 1)

        # 使用torch.gather提取每个期权对应到期日的股价
        S_T = torch.gather(S, 1, expiry_indices.view(1, -1).expand(N, -1))

        # 6. 计算期权价格 (num_options,)
        Ks = torch.tensor([opt['K'] for opt in options], device=device, dtype=torch.float32)
        payoffs = torch.clamp_min(S_T - Ks, 0.0)
        discount_factors = torch.exp(-r * Ts * h)
        option_prices = discount_factors * payoffs.mean(dim=0)

        return option_prices.cpu()  # 返回CPU数据便于后续可视化


def evaluate_pricing_accuracy(options, model, initial_price, r, h, N, is_real_data=False, model_type="fc", data_type="fake"):
    """评估期权定价准确性"""
    print("Starting Monte Carlo pricing...")
    # 记录推理开始时间
    inference_start_time = time.time()
    
    # 根据数据类型选择合适的蒙特卡洛定价函数
    if is_real_data:
        predicted_prices = monte_pricing_real_data(options, model, initial_price, r, h, N)
    else:
        predicted_prices = monte_pricing_fake_data(options, model, initial_price, r, h, N)
        
    # 记录推理结束时间
    inference_time = time.time() - inference_start_time
    print("Pricing completed.")
    print(f"Inference time: {inference_time:.2f} seconds")
    print(f"Average time per option: {inference_time/len(options):.4f} seconds")

    true_prices = torch.tensor([opt['O_0'] for opt in options],
                               dtype=torch.float32,
                               device=device).cpu()  # 转移到CPU便于可视化

    # 转换为NumPy数组
    pred_np = predicted_prices.numpy().flatten()
    true_np = true_prices.numpy().flatten()

    # 计算核心指标
    mae = np.mean(np.abs(pred_np - true_np))
    mse = np.mean((pred_np - true_np) ** 2)
    rmse = np.sqrt(mse)
    # 处理可能的除零情况
    with np.errstate(divide='ignore', invalid='ignore'):
        mape_values = np.abs((true_np - pred_np) / true_np) * 100
        mape = np.mean(mape_values[np.isfinite(mape_values)])  # 只计算有限值的平均值
    r = np.corrcoef(pred_np, true_np)[0, 1]
    r2 = r ** 2

    # 打印核心指标
    print("\nOption Pricing Model Evaluation")
    print("=" * 40)
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"Correlation Coefficient (R): {r:.6f}")
    print(f"Coefficient of Determination (R²): {r2:.6f}")
    print(f"Total Inference Time: {inference_time:.2f} seconds")
    print(f"Average Time per Option: {inference_time/len(options):.4f} seconds")
    if device.type == 'cuda':
        print(f"GPU Memory Usage: {torch.cuda.memory_allocated(0) / 1e6:.2f} MB")
    print("=" * 40)

    # 设置绘图样式
    sns.set_style("whitegrid")
    plt.figure(figsize=(16, 5))

    # 图表1：实际价格 vs 预测价格
    plt.subplot(1, 3, 1)
    plt.scatter(true_np, pred_np, alpha=0.6)
    plt.plot([true_np.min(), true_np.max()], [true_np.min(), true_np.max()], 'r--')
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted Prices")
    plt.grid(True)

    # 添加R²注解
    plt.text(0.05, 0.95, f"R² = {r2:.4f}", transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8))

    # 图表2：误差分布直方图
    plt.subplot(1, 3, 2)
    errors = pred_np - true_np
    sns.histplot(errors, kde=True, bins=30)
    plt.axvline(0, color='r', linestyle='--')
    plt.xlabel("Prediction Error (Pred - Actual)")
    plt.ylabel("Frequency")
    plt.title("Prediction Error Distribution")

    # 添加误差统计
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    plt.text(0.05, 0.95, f"Mean = {mean_error:.4f}\nStd Dev = {std_error:.4f}",
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8))

    # 图表3：绝对误差 vs 价格
    plt.subplot(1, 3, 3)
    abs_errors = np.abs(errors)
    plt.scatter(true_np, abs_errors, alpha=0.6)
    plt.xlabel("Actual Price")
    plt.ylabel("Absolute Error")
    plt.title("Absolute Error vs Actual Price")
    plt.grid(True)

    # 添加趋势线
    z = np.polyfit(true_np, abs_errors, 1)
    p = np.poly1d(z)
    xlim = plt.xlim()
    plt.plot(xlim, p(xlim), 'r--')

    plt.tight_layout()
    # 保存到项目output目录，并根据模型类型命名文件
    output_path = f'../output/evaluation_plot_{model_type}_{data_type}.png'
    plt.savefig(output_path)
    print(f"\nEvaluation plot saved as '{output_path}'")

    # 返回指标字典
    eval_results = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'R': r,
        'R²': r2
    }
    
    return eval_results


def evaluate_volatility_prediction(model, model_type="fc", data_type="fake", time_steps=100):
    """评估波动率预测准确性"""
    print("\nStarting volatility prediction...")
    predicted_volatility = []

    # 确保模型在正确设备上
    model.to(device)

    for t in range(0, time_steps):
        time_input = torch.tensor([t], dtype=torch.float32).to(device)

        # 使用torch.no_grad()禁用梯度计算
        with torch.no_grad():
            v_t = model(time_input.unsqueeze(0))
            predicted_volatility.append(v_t.item())  # 存储标量值
    print("Volatility prediction completed.")

    # 计算理论指数衰减曲线
    sigma_0 = 2
    theoretical_volatility = sigma_0 * np.exp(-0.02 * np.arange(0, time_steps))

    # 确保长度匹配
    if len(predicted_volatility) > len(theoretical_volatility):
        predicted_volatility = predicted_volatility[:len(theoretical_volatility)]
    elif len(theoretical_volatility) > len(predicted_volatility):
        theoretical_volatility = theoretical_volatility[:len(predicted_volatility)]

    # 转换为数组进行计算
    pred = np.array(predicted_volatility)
    true = theoretical_volatility

    # 计算核心指标
    mae = np.mean(np.abs(pred - true))
    mse = np.mean((pred - true) ** 2)
    rmse = np.sqrt(mse)
    # 处理可能的除零情况
    with np.errstate(divide='ignore', invalid='ignore'):
        mape_values = np.abs((true - pred) / true) * 100
        mape = np.mean(mape_values[np.isfinite(mape_values)])  # 只计算有限值的平均值
    r = np.corrcoef(pred, true)[0, 1]
    r2 = r ** 2

    # 计算特定波动率相关指标
    avg_volatility = np.mean(true)
    volatility_range = np.max(true) - np.min(true)
    peaks_error = np.mean(np.abs(pred - true)[np.argsort(true)[-5:]])  # Top 5 highest volatilities

    # 计算误差统计
    errors = pred - true
    abs_errors = np.abs(errors)
    skewness = stats.skew(errors)
    kurtosis = stats.kurtosis(errors)
    cumulative_error = np.cumsum(abs_errors)

    # 打印核心指标
    print("\nVolatility Forecasting Model Evaluation")
    print("=" * 60)
    print(f"Time Period: Day 0 to Day {len(pred) - 1}")
    print("=" * 60)
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"Correlation Coefficient (R): {r:.6f}")
    print(f"Coefficient of Determination (R²): {r2:.6f}")
    print("=" * 60)
    
    # 打印额外的波动率指标
    print(f"Average Volatility: {avg_volatility:.4f}")
    print(f"Volatility Range: {volatility_range:.4f}")
    print(f"Peak Volatility Error (Top 5): {peaks_error:.4f}")
    print("=" * 60)

    # 设置可视化
    sns.set_style("whitegrid")
    plt.figure(figsize=(16, 12))

    # 图表1：随时间变化的波动率比较
    plt.subplot(2, 2, 1)
    days = np.arange(len(true))
    plt.plot(days, true, 'b-', label='Theoretical Volatility', linewidth=2)
    plt.plot(days, pred, 'r--', label='Predicted Volatility', linewidth=1.5)
    plt.fill_between(days, true, pred, where=(pred > true), color='red', alpha=0.2, label='Overestimate')
    plt.fill_between(days, true, pred, where=(pred < true), color='blue', alpha=0.2, label='Underestimate')
    plt.xlabel("Time (Days)")
    plt.ylabel("Volatility (σ)")
    plt.title("Theoretical vs Predicted Volatility Over Time")
    plt.legend()
    plt.grid(True)

    # 添加R²注解
    plt.text(0.05, 0.95, f"R² = {r2:.4f}", transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8))

    # 图表2：误差散点图
    plt.subplot(2, 2, 2)
    plt.scatter(true, errors, c=days, cmap='viridis', alpha=0.7)
    plt.colorbar(label='Time (Days)')
    plt.axhline(0, color='black', linestyle='-')
    plt.xlabel("Theoretical Volatility")
    plt.ylabel("Prediction Error (Pred - Theoretical)")
    plt.title("Prediction Error Analysis")

    # 添加误差统计
    plt.text(0.05, 0.95, f"Mean Error = {np.mean(errors):.4f}\nStd Dev = {np.std(errors):.4f}",
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8))

    # 图表3：误差分布
    plt.subplot(2, 2, 3)
    sns.histplot(errors, kde=True, bins=30)
    plt.axvline(0, color='r', linestyle='--')
    plt.xlabel("Prediction Error (Pred - Theoretical)")
    plt.ylabel("Frequency")
    plt.title("Volatility Prediction Error Distribution")

    # 添加分布统计
    plt.text(0.65, 0.95, f"Skewness = {skewness:.2f}\nKurtosis = {kurtosis:.2f}",
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8))

    # 图表4：随时间变化的绝对误差
    plt.subplot(2, 2, 4)
    plt.plot(days, abs_errors, 'g-')
    plt.fill_between(days, abs_errors, 0, color='green', alpha=0.1)
    plt.xlabel("Time (Days)")
    plt.ylabel("Absolute Error")
    plt.title("Absolute Error Over Time")
    plt.grid(True)

    # 添加累积误差
    plt.text(0.05, 0.95, f"Cumulative Error = {cumulative_error[-1]:.4f}",
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    # 保存到项目output目录，并根据模型类型命名文件
    output_path = f'../output/volatility_evaluation_{model_type}_{data_type}.png'
    plt.savefig(output_path)
    print(f"\nEvaluation plot saved as '{output_path}'")
    
    # 返回指标字典
    volatility_results = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'R': r,
        'R²': r2,
        'Avg_Volatility': avg_volatility,
        'Volatility_Range': volatility_range,
        'Peak_Error': peaks_error,
        'Error_Skewness': skewness,
        'Error_Kurtosis': kurtosis,
        'Cumulative_Error': cumulative_error[-1]
    }
    
    return volatility_results


def main(model_type="fc", data_type="fake", evaluate_pricing=True, evaluate_volatility=True):
    """主函数"""
    # 根据模型类型获取相应的模型实例
    model = get_model(model_type).to(device)
    
    # 根据数据类型和模型类型加载权重
    if data_type == "fake":
        if model_type == "fc":
            weight_file = 'option_pricing_model_weights_fake_jump.pth'
        elif model_type == "cnn":
            weight_file = 'option_pricing_model_weights_fake_jump_CNN.pth'
        elif model_type == "lstm":
            weight_file = 'option_pricing_model_weights_fake_jump_LSTM.pth'
    else:  # real data
        weight_file = 'real_data_trained_model.pth'
    
    # 加载权重
    model.load_state_dict(torch.load(f'../models/{weight_file}', map_location=device))
    model.eval()  # 切换到评估模式

    if evaluate_pricing:
        # 根据数据类型导入测试数据
        if data_type == "fake":
            with open('../data/options_data_test_jump.pkl', 'rb') as f:
                options = pickle.load(f)
            with open('../data/sigma_data_jump.pkl', 'rb') as f:
                sigma = pickle.load(f)
            initial_price = torch.tensor([100.0] * 4, device=device)
            options = to_device(options, device)
            
            # 评估参数
            r = 0.05
            h = 1 / 252
            N = 1000000
            
            # 评估期权定价准确性
            eval_results = evaluate_pricing_accuracy(options, model, initial_price, r, h, N, is_real_data=False, model_type=model_type, data_type=data_type)
        else:  # real data
            with open('../data/real_options_data.pkl', 'rb') as f:
                options = pickle.load(f)
            with open('../data/real_initial_price.pkl', 'rb') as f:
                initial_price = pickle.load(f)
            initial_price = to_device(initial_price, device)
            options = to_device(options, device)
            options = options[30:]  # 跳过前30个数据点
            
            # 评估参数
            r = 0.0149
            h = 1 / 252
            N = 1000000
            
            # 评估期权定价准确性
            eval_results = evaluate_pricing_accuracy(options, model, initial_price, r, h, N, is_real_data=True, model_type=model_type, data_type=data_type)
        
        print(f"\nEvaluation Results for {model_type.upper()} model on {data_type} data:")
        for key, value in eval_results.items():
            print(f"{key}: {value:.6f}")

    if evaluate_volatility:
        # 评估波动率预测准确性
        volatility_results = evaluate_volatility_prediction(model, model_type=model_type, data_type=data_type, time_steps=100)
        
        print(f"\nVolatility Evaluation Results for {model_type.upper()} model:")
        for key, value in volatility_results.items():
            print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate option pricing model')
    parser.add_argument('--model', type=str, default='fc', choices=['fc', 'cnn', 'lstm'],
                        help='Model type: fc (Fully Connected), cnn (Convolutional Neural Network), lstm (Long Short-Term Memory)')
    parser.add_argument('--data', type=str, default='fake', choices=['fake', 'real'],
                        help='Data type: fake (synthetic data), real (real market data)')
    parser.add_argument('--pricing', action='store_true', default=True,
                        help='Whether to evaluate option pricing accuracy')
    parser.add_argument('--volatility', action='store_true', default=True,
                        help='Whether to evaluate volatility prediction accuracy')
    
    args = parser.parse_args()
    main(model_type=args.model, data_type=args.data, 
         evaluate_pricing=args.pricing, evaluate_volatility=args.volatility)