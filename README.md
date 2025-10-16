# code_finance_math
This code aims to simulate and compare the performance of three machine learning models—Fully Connected Neural Network (FC), Long Short-Term Memory (LSTM), and Convolutional Neural Network (CNN)—in predicting volatility based on synthetic (virtual) data.

# 期权定价模型项目

## 项目结构

```
project/
├── src/
│   ├── integrated_model.py              # 整合的模型训练脚本
│   ├── integrated_evaluation.py          # 整合的模型评估脚本
│   ├── integrated_performance_analysis.py # 整合的性能分析脚本
│   └── shap_analysis.py                 # SHAP模型解释性分析脚本
├── data/
│   ├── *.pkl                            # 原始数据文件
│   └── *.pth                            # 原始模型权重文件
├── models/
│   └── *.pth                            # 训练生成的模型权重文件
├── output/
│   ├── *.txt                            # 训练日志文件
│   ├── *.png                            # 训练和评估生成的图表
│   └── training_history.png             # 训练历史图表
└── README.md                           # 项目说明文档
```

## 目录说明

### src目录
存放源代码文件：
- `integrated_model.py`: 整合了全连接网络、CNN和LSTM三种模型的训练脚本
- `integrated_evaluation.py`: 整合了全连接网络、CNN和LSTM三种模型的评估脚本
- `integrated_performance_analysis.py`: 整合的性能分析脚本
- `shap_analysis.py`: 使用SHAP库对模型进行解释性分析的脚本

### data目录
存放原始数据和模型文件：
- `options_data_*.pkl`: 期权数据文件
- `sigma_data_*.pkl`: 波动率数据文件
- `real_options_data.pkl`: 真实期权数据
- `real_initial_price.pkl`: 真实初始价格数据
- 原始的模型权重文件

### models目录
存放训练过程中生成的模型权重文件：
- `option_pricing_model_weights_*.pth`: 各种模型训练后生成的权重文件

### output目录
存放训练和评估过程中生成的输出文件：
- `training_log_*.txt`: 训练日志文件
- `*.png`: 训练和评估生成的图表
- `training_history.png`: 训练历史图表

## 文件说明

### 训练脚本
- `integrated_model.py`: 整合了全连接网络、CNN和LSTM三种模型的训练脚本

### 评估脚本
- `integrated_evaluation.py`: 整合了全连接网络、CNN和LSTM三种模型的评估脚本

### 性能分析脚本
- `integrated_performance_analysis.py`: 整合的性能分析脚本

### 模型解释性分析脚本
- `shap_analysis.py`: 使用SHAP技术分析模型特征重要性的脚本

### 数据文件
- `options_data_*.pkl`: 期权数据文件
- `sigma_data_*.pkl`: 波动率数据文件
- `real_options_data.pkl`: 真实期权数据
- `real_initial_price.pkl`: 真实初始价格数据
- `option_pricing_model_weights_*.pth`: 各种模型的权重文件
- `real_data_trained_model.pth`: 真实数据训练的模型权重

## 使用方法

### 训练模型

```bash
#cd src
#在\project\src目录下执行

# 训练全连接模型
python integrated_model.py --model fc --epochs 500

# 训练CNN模型
python integrated_model.py --model cnn --epochs 500

# 训练LSTM模型
python integrated_model.py --model lstm --epochs 500
```

### 评估模型

```bash
#cd src
#在\project\src目录下执行

# 评估全连接模型（合成数据）
python integrated_evaluation.py --model fc --data fake --pricing --volatility

# 评估CNN模型（合成数据）
python integrated_evaluation.py --model cnn --data fake --pricing --volatility

# 评估LSTM模型（合成数据）
python integrated_evaluation.py --model lstm --data fake --pricing --volatility

# 评估全连接模型（真实数据）
#python integrated_evaluation.py --model fc --data real --pricing --volatility
```

### 分析性能

```bash
# 进入src目录执行性能分析脚本
cd src

# 分析所有模型的性能
python integrated_performance_analysis.py
```

### 模型解释性分析

```bash
# 在src目录下执行SHAP分析
cd src

# 对模型进行SHAP分析
python shap_analysis.py

# 或者指定自定义参数
python shap_analysis.py --model_path ../models/your_model.pth --background_samples 100 --test_samples 200 --output_dir ../output
```

## 模型说明

### 全连接网络 (FC)
- 简单的前馈神经网络
- 参数较少，计算速度快
- 适用于基础的波动率预测任务

### 卷积神经网络 (CNN)
- 使用1D卷积层提取时间序列特征
- 参数适中，适合模式识别
- 能够捕捉局部时间依赖性

### 长短期记忆网络 (LSTM)
- 循环神经网络结构
- 参数较多，适合时间依赖建模
- 能够处理长期依赖关系

## 评估指标

### 期权定价准确性
- MAE (Mean Absolute Error): 平均绝对误差
- MSE (Mean Squared Error): 均方误差
- RMSE (Root Mean Squared Error): 均方根误差
- MAPE (Mean Absolute Percentage Error): 平均绝对百分比误差
- R: 相关系数
- R²: 决定系数

### 波动率预测准确性
- MAE, MSE, RMSE, MAPE, R, R² 等相同指标
- 与理论指数衰减波动率曲线进行比较

## SHAP分析说明

SHAP (SHapley Additive exPlanations) 是一种用于解释机器学习模型预测结果的方法。通过SHAP分析，我们可以：

1. 了解每个特征对模型预测的贡献程度
2. 可视化特征重要性和影响方向
3. 分析特定样本的预测原因

### 生成的图表类型

- **SHAP摘要图**: 显示特征重要性分布和影响方向
- **SHAP依赖图**: 展示特征值与SHAP值之间的关系
- **SHAP力图**: 解释单个预测的具体贡献因素

### 经济解释

脚本还会提供SHAP值的经济解释，包括：
- 整体趋势分析（时间特征对预测的影响趋势）
- 分阶段影响分析（初期、中期、后期的影响差异）
- 关键时间点识别（最大正向和负向影响的时间点）
