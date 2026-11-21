"""A/B 测试计算器：样本量规划 + 结果显著性分析。

运行:  streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from abtest.stats import (minimum_detectable_effect, power_two_proportions,  # noqa: E402
                          sample_size_per_group, srm_check, two_proportion_ztest)
from abtest.plotting import ACCENT, CONTROL, MUTED, TREATMENT, set_style  # noqa: E402

set_style()
st.set_page_config(page_title="A/B 测试计算器", page_icon="📊", layout="wide")
st.title("📊 A/B 测试计算器")
st.caption("转化率类指标（0/1）的样本量规划与显著性分析。公式与推导见项目 notebooks 与 docs/learning_guide.md。")

tab_plan, tab_result = st.tabs(["① 实验前：样本量规划", "② 实验后：结果分析"])

# ---------------------------------------------------------------------------
with tab_plan:
    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.subheader("输入")
        baseline = st.number_input("基线转化率（%）", 0.1, 99.0, 12.0, 0.1) / 100
        mde_type = st.radio("MDE 类型", ["相对提升", "绝对提升（百分点）"], horizontal=True)
        if mde_type == "相对提升":
            mde_rel = st.number_input("最小可检测相对提升（%）", 0.1, 500.0, 5.0, 0.5) / 100
            mde_abs = baseline * mde_rel
        else:
            mde_abs = st.number_input("最小可检测绝对提升（百分点）", 0.01, 50.0, 0.6, 0.05) / 100
            mde_rel = mde_abs / baseline
        alpha = st.select_slider("显著性水平 α", [0.01, 0.05, 0.10], value=0.05)
        power = st.select_slider("统计功效 1-β", [0.7, 0.8, 0.85, 0.9, 0.95], value=0.8)
        ratio = st.number_input("实验组 : 对照组 人数比", 0.1, 10.0, 1.0, 0.1)
        daily = st.number_input("每日进入实验的总用户数", 1, 10_000_000, 12_600, 100)

    with c2:
        st.subheader("结果")
        if not 0 < baseline + mde_abs < 1:
            st.error("基线 + MDE 必须在 0% 到 100% 之间。")
        else:
            n_c = sample_size_per_group(baseline, mde_abs, alpha, power, ratio)
            n_t = int(np.ceil(n_c * ratio))
            total = n_c + n_t
            days = total / daily
            m1, m2, m3 = st.columns(3)
            m1.metric("对照组样本量", f"{n_c:,}")
            m2.metric("实验组样本量", f"{n_t:,}")
            m3.metric("预计天数", f"{days:.1f} 天")
            st.write(f"检测 **{baseline:.2%} → {baseline + mde_abs:.2%}**"
                     f"（绝对 {mde_abs*100:+.2f}pp，相对 {mde_rel:+.1%}）。")
            if days < 7:
                st.info("建议至少运行 7 天（一个完整的周），以覆盖工作日 / 周末的行为差异。")
            if days > 60:
                st.warning("所需时间过长。可考虑：增大 MDE、使用 CUPED 等方差缩减方法、或换用更敏感的指标。")

            fig, ax = plt.subplots(figsize=(7, 3.6))
            grid = np.linspace(max(baseline * 0.002, 1e-4), max(mde_abs * 2.5, 1e-3), 200)
            grid = grid[baseline + grid < 1]
            ax.plot(grid * 100, [power_two_proportions(baseline, d, n_c, alpha, ratio) for d in grid],
                    color=CONTROL, label=f"n_control = {n_c:,}")
            ax.axhline(power, color=MUTED, ls="--", lw=1)
            ax.axvline(mde_abs * 100, color=TREATMENT, ls=":", lw=1.5, label="your MDE")
            ax.set_xlabel("True absolute lift (pp)")
            ax.set_ylabel("Power")
            ax.set_title("Power curve at the planned sample size")
            ax.legend(loc="lower right")
            st.pyplot(fig)

    with st.expander("📖 这些参数是什么意思？"):
        st.markdown(r"""
- **基线转化率**：对照组（现有版本）的转化率，通常用历史数据估计。
- **MDE（最小可检测效应）**：业务上值得检测的最小提升。样本量与 MDE² 成反比：MDE 减半，样本量约 ×4。
- **α**：第一类错误率（假阳性），没有效果却判断为有效果的概率。
- **功效 1-β**：真实效果等于 MDE 时能检测出来的概率。
- 公式：$n = \dfrac{\left(z_{1-\alpha/2}\sqrt{\bar p\bar q(1+1/k)} + z_{1-\beta}\sqrt{p_1q_1+p_2q_2/k}\right)^2}{\delta^2}$，其中 $k$ 为实验组/对照组人数比。
""")

# ---------------------------------------------------------------------------
with tab_result:
    st.caption("默认值为本项目 Udacity 数据集清洗后的真实结果。")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**对照组 Control**")
        n_c = st.number_input("用户数", 1, 10**9, 145_274, key="nc")
        x_c = st.number_input("转化数", 0, 10**9, 17_489, key="xc")
    with c2:
        st.markdown("**实验组 Treatment**")
        n_t = st.number_input("用户数", 1, 10**9, 145_310, key="nt")
        x_t = st.number_input("转化数", 0, 10**9, 17_264, key="xt")
    c3, c4, c5 = st.columns(3)
    alpha_r = c3.select_slider("显著性水平 α ", [0.01, 0.05, 0.10], value=0.05)
    alt_label = c4.selectbox("备择假设", ["双侧：p_t ≠ p_c", "单侧：p_t > p_c", "单侧：p_t < p_c"])
    expected_share = c5.number_input("设计的实验组流量占比（%）", 1.0, 99.0, 50.0, 1.0) / 100
    alt = {"双侧": "two-sided", "单侧：p_t >": "larger", "单侧：p_t <": "smaller"}
    alternative = next(v for k, v in alt.items() if alt_label.startswith(k))

    if x_c > n_c or x_t > n_t or x_c == 0 or x_t == 0:
        st.error("转化数必须大于 0 且不超过用户数。")
        st.stop()

    srm = srm_check([n_c, n_t], [1 - expected_share, expected_share])
    if srm.srm_detected:
        st.error(f"⚠️ 检测到样本比例失衡（SRM，p = {srm.p_value:.2e}）。请先排查分流与埋点问题，"
                 "在找到原因之前，下面的效果估计不可信。")
    else:
        st.success(f"✅ SRM 检查通过（p = {srm.p_value:.3f}），分流比例与设计一致。")

    r = two_proportion_ztest(x_c, n_c, x_t, n_t, alpha=alpha_r, alternative=alternative)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("对照组转化率", f"{r.p_control:.3%}")
    m2.metric("实验组转化率", f"{r.p_treatment:.3%}", f"{r.abs_diff*100:+.3f} pp")
    m3.metric("相对提升", f"{r.rel_lift:+.2%}")
    m4.metric("p 值", f"{r.p_value:.4f}")

    conf = 1 - alpha_r
    st.write(f"**绝对差 {conf:.0%} 置信区间**：[{r.ci_abs[0]*100:+.3f}, {r.ci_abs[1]*100:+.3f}] 个百分点　"
             f"**相对提升 {conf:.0%} 置信区间**：[{r.ci_rel[0]:+.2%}, {r.ci_rel[1]:+.2%}]　（z = {r.z:.3f}）")

    mde_now = minimum_detectable_effect(r.p_control, n_c, alpha_r, 0.8, n_t / n_c)
    if r.p_value < alpha_r:
        direction = "高于" if r.abs_diff > 0 else "低于"
        st.success(f"**统计显著**：实验组转化率{direction}对照组。下一步请判断效果大小（看置信区间）是否达到业务上值得上线的门槛。")
    else:
        st.warning(f"**统计不显著**：没有足够证据表明两组不同。按当前样本量，80% 功效下能检测的最小效果约为 "
                   f"{mde_now*100:.3f} 个百分点（相对 {mde_now/r.p_control:.1%}）。"
                   "若置信区间已排除了业务上有意义的提升，可以结束实验；否则需要更多样本。")

    fig, ax = plt.subplots(figsize=(8, 2.4))
    lo, hi = np.array(r.ci_abs) * 100
    ax.errorbar(r.abs_diff * 100, 0, xerr=[[r.abs_diff * 100 - lo], [hi - r.abs_diff * 100]],
                fmt="o", color=TREATMENT, capsize=6, ms=8)
    ax.axvline(0, color=MUTED, lw=1)
    span = max(abs(lo), abs(hi)) * 1.4
    ax.set_xlim(-span, span)
    ax.set_yticks([])
    ax.set_xlabel("Treatment - Control (pp)")
    ax.set_title(f"Difference with {conf:.0%} CI")
    st.pyplot(fig)

    with st.expander("📖 如何解读？"):
        st.markdown("""
- **p 值**：假如两组真实转化率相同，观察到至少这么大差异的概率。它**不是**"实验组无效的概率"。
- **置信区间包含 0** ⇔ 双侧检验不显著。区间的上下界告诉你效果"最好 / 最坏"可能有多大，比 p 值更有决策价值。
- **SRM**：实际分流比例与设计比例差异过大（p < 0.001）时，说明实验系统有问题，结果不可信。
- 不要在实验中途反复查看 p 值并"显著就停"，这会大幅提高假阳性率（见 notebook 03）。
""")
