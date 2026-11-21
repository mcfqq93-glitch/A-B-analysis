# 两周学习计划：完整掌握这个项目

目标：两周后，你可以**不看资料**完成三件事：
1. 用 5 分钟讲清楚项目的流程、方法和结论；
2. 回答 `interview_qa.md` 里 80% 以上的问题；
3. 在空白 notebook 里从零重写核心分析（清洗 → SRM → z 检验 → CI → 样本量）。

每天大约 2–3 小时。每天的最后一项都是"**合上电脑，口头讲一遍**"，这是检验是否真正掌握的最好方法。

---

## 第一周：数据与核心检验

| 天 | 学习内容 | 动手任务 | 掌握标准 |
|---|---|---|---|
| 1 | 指南第 1–2 节；跑通环境 | 按 README 安装、运行全部 notebook 与 `pytest` | 能解释 p 值、α、β、功效，并说出 p 值**不是**什么 |
| 2 | 指南第 3 节；notebook 01 | 自己写代码找出错配行和重复用户，不看 `clean_ab_data()` | 能说出删除错配行的理由，以及什么时候应改用 ITT |
| 3 | SRM；协变量平衡 | 手算 SRM 的 χ²（用计算器）；把对照组人数改成 144,000 看 p 值怎么变 | 能说出 SRM 的 3 个常见原因，以及为什么阈值是 0.001 |
| 4 | 指南第 4 节；notebook 02 第 2 部分 | 读 `sample_size_per_group()`，用 `n ≈ 16σ²/δ²` 心算 3 个场景再对照 | 能现场估算样本量；能说明为什么不做事后功效 |
| 5 | 指南第 5–6 节；notebook 02 第 3–4 部分 | **不调库**，用 numpy 手写 z 检验，结果与 statsmodels 对齐 | 能写出 z 的公式，说明合并 vs 非合并 SE 的区别；知道 χ² = z² |
| 6 | 指南第 7 节；notebook 02 第 5 部分 | 手写绝对差 CI；推导一次相对提升的 Delta 方法方差 | 能用业务语言解读 [−0.39, +0.08] 这个区间 |
| 7 | 复习 + 第 12 节决策 | 在空白 notebook 里从零重写 01 + 02 的核心部分（限时 90 分钟） | 能用 1 分钟口述项目（Q1），并回答 Q2–Q4 |

## 第二周：进阶方法与表达

| 天 | 学习内容 | 动手任务 | 掌握标准 |
|---|---|---|---|
| 8 | 指南第 8 节；notebook 02 第 7 部分 | 用"真正的"有放回抽样（`rng.choice`）写一个慢版 Bootstrap，与快速版比较 | 能说清 Bootstrap 和置换检验的区别（见指南表格） |
| 9 | 指南第 9 节；notebook 03 Part A | 在纸上推导 Var(Y − θX) 的最小值；把模拟的相关系数调低再运行，看方差缩减变化 | 能讲出 CUPED 的公式、无偏原因、方差缩减比例 1 − ρ² |
| 10 | 指南第 10 节；notebook 03 Part B | 手写 Holm 和 BH（不看 `adjust_pvalues()`）；用 10 个 p 值测试 | 能说明 FWER 和 FDR 的区别，以及为什么要用交互作用检验 |
| 11 | 指南第 11 节；notebook 03 Part C | 修改 `simulate_peeking()` 的偷看次数（5、50 次），画出假阳性率曲线 | 能解释偷看为什么导致假阳性，并说出 2 种解决方法 |
| 12 | Streamlit 计算器 | 运行 app，读懂代码；**自己加一个功能**（建议：根据日流量和实验天数反推 MDE） | 能演示 app，并说明它解决了什么业务问题 |
| 13 | 面试题第一、二、三部分 | 每题遮住答案口述，录音回听；标记答不好的题 | 第一部分全部流利；第二、三部分 80% |
| 14 | 面试题第四部分 + 开放题框架 | 模拟面试：随机抽 10 题；用框架回答"设计一个实验评估新的推荐算法" | 能在 5 分钟内讲完整个项目并应对追问 |

---

## 推荐的扩展练习（选做，可以继续丰富作品集）

1. **贝叶斯 A/B 测试**：用 Beta(1,1) 先验，计算 P(p_t > p_c) 和期望损失，与频率学派结论比较。
2. **成组序贯检验**：实现 O'Brien–Fleming 边界，在偷看模拟中验证假阳性率回到 5%。
3. **比率指标的 Delta 方法**：模拟"用户随机化、按浏览次数算点击率"的场景，比较朴素方差与 Delta 方法方差。
4. **换一个数据集复现**：例如 Kaggle 上的 Cookie Cats 手游 A/B 数据（留存率）或 Marketing A/B 数据，套用本项目的函数库。

## 参考资料

- Kohavi, Tang, Xu. *Trustworthy Online Controlled Experiments* (2020)：A/B 测试领域最权威的书，第 3、17、21 章尤其相关。
- Deng et al. *Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data* (WSDM 2013)：CUPED 原始论文。
- Fabijan et al. *Diagnosing Sample Ratio Mismatch in Online Controlled Experiments* (KDD 2019)：SRM 的系统讨论。
- Johari et al. *Peeking at A/B Tests* (KDD 2017)：偷看问题与始终有效的 p 值。
