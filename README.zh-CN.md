# CN4DP：从疾病轨迹到影像表型

论文 **From Trajectories to Phenotypes: Disease Progression as Structural Priors for Multi-organ Imaging Representation Learning** 的研究代码。

[arXiv 论文](https://arxiv.org/abs/2605.11958) · [完整英文说明](README.md)

方法先用 Delphi 风格的生成式 Transformer 学习纵向疾病轨迹，再通过 InfoNCE 或 MSE 对齐，将轨迹信息迁移到器官级影像表型编码器。下游可以使用影像表型、历史轨迹，或者二者的拼接及交叉注意力融合，预测疾病风险和发病时间。

本公开版本只提供代码与示例配置。**不包含 UK Biobank 真实数据、受试者标识、队列导出、真实数据拟合的统计量、研究模型权重、notebook 输出或原仓库 Git 历史。** Demo 完全通过随机数生成，结果仅用于检查流程是否可运行，不代表论文实验结果。

## 快速运行

使用 Python 3.10 或以上版本，demo 可在 CPU 上运行。

```bash
git clone https://github.com/PrinceAnn/cn4dp.git
cd cn4dp
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/demo/run_demo.py
```

流程依次执行：合成数据生成 → 教师数据转换 → 轨迹教师训练 → IDP 编码器蒸馏 → IDP 预测 → 轨迹预测 → 融合预测 → 生成式基线评分。教师 demo 训练 20 步，其余主要模型训练 2 个 epoch；输出写入 `runs/demo/`。

可选的表型补全扩展：

```bash
python scripts/demo/run_demo.py --with-imputation
```

补全代码是额外的实验扩展，与论文主要预测流程分开。公开版未附带原实验报告和私有结果。

## 配置与数据

`configs/demo/` 提供蒸馏、IDP 从零训练、IDP 预训练后微调、轨迹预测、拼接融合、交叉注意力融合、生成式基线和补全示例。完整逐步命令见英文 README。

所有 demo 受试者、特征和疾病标签均为虚构。示例有五个器官组、20 个随机特征，使用 `SYN_TARGET` 等虚构疾病名称；`eid` 是生成器创建的整数连接键。

| 输入表 | 必需字段 |
| --- | --- |
| 表型 | `eid`、`organ__feature_name` 格式的数值特征，每人一行 |
| 基本信息 | `eid`、`birth_year`、`imaging_date` |
| 疾病首次发病年龄 | `eid`、每个疾病一列，数值为年龄（年），未记录事件留空 |
| 下游划分 | `eid`、`split`，取值为 `train` / `val` / `test` |
| 风险集 | `eid`、`disease`、`label`、`imaging_date`、`delta_time`，附匹配信息 |

使用经授权的研究数据时，应在本地将数据转换为以上通用格式。公开代码不提供特定队列的字段映射或原始导出脚本。真实数据、权重及运行输出应保持在 Git 忽略的本地目录中。

## 时间与队列划分

教师训练、蒸馏和下游队列必须互斥。Demo 已按这个原则生成；下游受试者先划分训练、验证和测试集合，再分别进行风险集采样。所有预测配置使用同一份 `split_csv`，避免同一受试者多次出现时跨集合。

蒸馏和下游只使用严格早于成像时点的诊断记录。标准化参数从训练受试者拟合，或复用独立蒸馏队列的参数。风险集病例必须在成像后首次发病，对照在病例时点必须已经成像且尚未发生目标事件。

通用风险集采样器没有随访结束或删失字段，假设受试者在相关病例时点仍被观察；使用真实队列时需要结合随访信息调整对照资格。仅有出生年份时，成像年龄也是近似值。Demo 不能替代论文完整的质量控制、疾病选择及多随机种子实验。

当前交叉注意力实现用**聚合后的 IDP 表征**查询轨迹 token；论文概念描述中的逐器官查询变体需要进一步适配，具体见融合脚本的 `CrossAttentionFusion`。

## 检查与引用

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python scripts/check_public_release.py
```

发布检查针对已跟踪文件，检查数据文件、权重、软链接、本机路径、队列字段格式和常见凭据。还应人工检查暂存差异。全新历史不会自动清除远程已有分支或曾上传的对象。

引用信息见英文 README 和 `CITATION.cff`。Delphi 上游代码保留其 MIT 许可；详见 `Delphi/LICENSE` 和 `THIRD_PARTY.md`，仓库未附带任何上游权重。
