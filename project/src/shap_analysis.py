import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import shap

# 设置中文字体，确保中文正常显示
plt.rcParams["font.sans-serif"] = [
    "SimHei", "WenQuanYi Micro Hei", "Heiti TC",
    "Arial Unicode MS", "DejaVu Sans"
]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
plt.rcParams["font.size"] = 10  # 全局字体大小调整


class SHAPExplainer:
    """使用SHAP工具解释期权定价模型的特征影响，生成标准SHAP可视化图表"""

    def __init__(self, model_path: str):
        """初始化解释器，加载模型并设置计算设备"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用计算设备: {self.device}")

        # 加载模型（带异常处理）
        try:
            self.model = self._load_model(model_path)
            self.model.eval()
            print(f"模型加载成功: {model_path}")
        except Exception as e:
            print(f"模型加载失败: {str(e)}", file=sys.stderr)
            raise

    def _load_model(self, model_path: str) -> torch.nn.Module:
        """加载模型结构并加载权重"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(current_dir)
        from integrated_model import get_model  # 导入模型定义
        model = get_model("fc").to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        return model

    def predict_wrapper(self, inputs: np.ndarray) -> np.ndarray:
        """SHAP兼容的预测包装函数，将numpy输入转换为模型输出"""
        # 处理空输入
        if inputs.size == 0:
            return np.array([])

        inputs_tensor = torch.tensor(inputs, dtype=torch.float32, device=self.device)
        if inputs_tensor.dim() == 1:
            inputs_tensor = inputs_tensor.unsqueeze(1)  # 确保二维输入

        with torch.no_grad():
            predictions = self.model(inputs_tensor)

        return predictions.cpu().numpy().flatten()

    def compute_shap_values(
            self,
            background_samples: int = 100,
            test_samples: int = 100
    ) -> tuple:
        """计算SHAP值，生成背景数据和测试数据"""
        print("\n开始计算SHAP值...")

        # 生成时间步数据（0-99），确保数据分布均匀
        background_data = np.linspace(0, 99, background_samples).reshape(-1, 1)
        test_data = np.linspace(0, 99, max(test_samples, 100)).reshape(-1, 1)

        # 使用kmeans减少背景数据样本以提高性能
        if background_samples > 100:
            background_data = shap.kmeans(background_data, 100).data
            print("背景数据已通过kmeans简化为100个样本以提高性能")

        # 初始化SHAP核解释器
        explainer = shap.KernelExplainer(self.predict_wrapper, background_data)
        shap_values = explainer.shap_values(test_data)

        # 确保SHAP值维度正确（样本数×特征数）
        if len(shap_values.shape) == 1:
            shap_values = shap_values.reshape(-1, 1)

        print("SHAP值计算完成")
        return shap_values, test_data, background_data, explainer

    def plot_shap_analysis(
            self,
            shap_values: np.ndarray,
            test_data: np.ndarray,
            explainer: shap.KernelExplainer,
            output_dir: str
    ) -> None:
        """生成标准SHAP图表：力图、摘要图、依赖图，优化显示效果"""
        print("\n绘制标准SHAP分析图...")
        os.makedirs(output_dir, exist_ok=True)
        feature_names = ["时间步"]  # 特征名称

        # 1. 标准SHAP摘要图（优化点显示）
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values=shap_values,
            features=test_data,
            feature_names=feature_names,
            show=False,
            plot_type="dot"
        )
        plt.title("SHAP摘要图 - 特征重要性与影响分布", fontsize=14)
        plt.xlabel("SHAP值（对模型输出的影响）", fontsize=12)
        plt.tight_layout()
        summary_path = os.path.join(output_dir, "shap_summary_plot.png")
        plt.savefig(summary_path, dpi=300, bbox_inches="tight")
        print(f"标准摘要图已保存: {summary_path}")
        plt.close()

        # 2. 标准SHAP依赖图（优化趋势线和点分布）
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            ind=0,  # 第0个特征（时间步）
            shap_values=shap_values,
            features=test_data,
            feature_names=feature_names,
            show=False
        )
        plt.title("SHAP依赖图 - 时间步与预测影响的关系", fontsize=14)
        plt.xlabel("时间步", fontsize=12)
        plt.ylabel("SHAP值（影响强度）", fontsize=12)
        plt.tight_layout()
        dep_path = os.path.join(output_dir, "shap_dependence_plot.png")
        plt.savefig(dep_path, dpi=300, bbox_inches="tight")
        print(f"标准依赖图已保存: {dep_path}")
        plt.close()

        # 3. 标准SHAP力图（核心优化：解决刻度过密和除零错误）
        base_value = explainer.expected_value  # 基线值（模型平均预测）
        shap_flat = shap_values.flatten()

        # 选择代表性时间点（确保索引唯一）
        max_influence_idx = np.argmax(np.abs(shap_flat))
        sample_indices = {0, max_influence_idx, len(test_data) - 1}
        sample_indices = sorted(list(sample_indices))
        print(f"将生成{len(sample_indices)}个力图，索引为: {sample_indices}")

        for idx in sample_indices:
            try:
                time_step = int(test_data[idx][0])
                current_shap = shap_values[idx][0]
                current_feature = test_data[idx][0]

                # 计算合理的x轴范围（解决刻度过密问题）
                # 基于基线值和SHAP值扩展范围，确保显示清晰
                total_effect = base_value + current_shap
                margin = max(0.01, abs(total_effect - base_value) * 2)  # 动态边缘
                x_min = min(base_value, total_effect) - margin
                x_max = max(base_value, total_effect) + margin

                # 生成力图（适配新版SHAP API）
                explanation = shap.Explanation(
                    values=np.array([current_shap]),
                    base_values=base_value,
                    data=np.array([current_feature]),
                    feature_names=feature_names
                )
                # 设置图表样式
                plt.figure(figsize=(10, 2))
                shap.plots.waterfall(explanation, show=False)
                # 确保x轴范围正确
                plt.xlim(x_min, x_max)
                plt.title(f"SHAP力图 - 时间步 {time_step} 案例", fontsize=14)
                plt.tight_layout()
                force_path = os.path.join(output_dir, f"shap_force_plot_{time_step}.png")
                plt.savefig(force_path, dpi=300, bbox_inches="tight")
                print(f"标准力图已保存: {force_path}")

            except Exception as e:
                print(f"生成力图时出错 (索引 {idx}, 时间步 {time_step}): {e}")
            finally:
                plt.close()

    def economic_interpretation(
            self,
            shap_values: np.ndarray,
            test_data: np.ndarray
    ) -> dict:
        """生成SHAP值的经济解释统计结果，增强鲁棒性"""
        print("\n=== SHAP值的经济解释 ===")
        shap_flat = shap_values.flatten()
        n_samples = len(shap_flat)
        if n_samples == 0:
            raise ValueError("SHAP值为空，无法进行经济解释")

        # 基本统计量（处理可能的异常值）
        stats = {
            "mean": np.nanmean(shap_flat),
            "std": np.nanstd(shap_flat) if n_samples > 1 else 0,
            "min": np.nanmin(shap_flat),
            "max": np.nanmax(shap_flat)
        }
        print("SHAP值统计:")
        for metric, value in stats.items():
            print(f"  {metric}: {value:.6f}")

        # 整体趋势（避免除零）
        if n_samples <= 1:
            trend = "无明显趋势（样本量不足）"
        else:
            diff_values = np.diff(shap_flat)
            mean_diff = np.nanmean(diff_values) if not np.isnan(diff_values).all() else 0
            trend = "上升" if mean_diff > 1e-8 else "下降"  # 增加微小阈值避免浮点误差
        print(f"\n整体趋势: 时间特征对预测的影响呈{trend}趋势")

        # 分阶段分析（确保阶段非空）
        stage1_end = max(1, int(n_samples * 1 / 3))
        stage2_end = max(stage1_end + 1, int(n_samples * 2 / 3))
        stages = {
            "初期(0-33天)": shap_flat[:stage1_end],
            "中期(33-66天)": shap_flat[stage1_end:stage2_end],
            "后期(66-100天)": shap_flat[stage2_end:]
        }
        print("分阶段平均影响:")
        for stage_name, stage_data in stages.items():
            if len(stage_data) > 0:
                stage_mean = np.nanmean(stage_data)
                print(f"  {stage_name}: {stage_mean:.6f}")

        # 关键时间点
        valid_mask = ~np.isnan(shap_flat)
        if np.any(valid_mask):
            max_idx = np.nanargmax(shap_flat)
            min_idx = np.nanargmin(shap_flat)
        else:
            max_idx = min_idx = 0

        key_points = {
            "最大正向影响": (int(test_data[max_idx][0]), shap_flat[max_idx]),
            "最大负向影响": (int(test_data[min_idx][0]), shap_flat[min_idx])
        }
        print("\n关键时间点:")
        for point_type, (time_step, value) in key_points.items():
            print(f"  {point_type}: 第{time_step}天，贡献值={value:.6f}")

        return {**stats, "trend": trend, "key_points": key_points}


def main():
    """主函数：解析参数并执行SHAP分析流程"""
    parser = argparse.ArgumentParser(description="生成标准SHAP图分析期权定价模型")
    parser.add_argument(
        "--model_path", type=str,
        default="../models/option_pricing_model_weights_fake_jump.pth",
        help="模型权重文件路径"
    )
    parser.add_argument(
        "--background_samples", type=int, default=100,
        help="背景数据样本量（用于计算基线）"
    )
    parser.add_argument(
        "--test_samples", type=int, default=200,  # 增加测试样本量，使图表更密集
        help="测试数据样本量（建议≥200）"
    )
    parser.add_argument(
        "--output_dir", type=str, default="../output",
        help="图表输出目录"
    )
    args = parser.parse_args()

    print("=== 标准SHAP分析工具 ===")

    try:
        # 初始化解释器
        explainer = SHAPExplainer(args.model_path)

        # 计算SHAP值
        shap_values, test_data, _, shap_explainer = explainer.compute_shap_values(
            args.background_samples, args.test_samples
        )

        # 生成标准SHAP图
        explainer.plot_shap_analysis(
            shap_values=shap_values,
            test_data=test_data,
            explainer=shap_explainer,
            output_dir=args.output_dir
        )

        # 生成经济解释
        explainer.economic_interpretation(shap_values, test_data)

        print("\n=== 分析完成，标准SHAP图已保存到指定目录 ===")

    except Exception as e:
        print(f"分析失败: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()