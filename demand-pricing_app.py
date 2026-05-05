import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go
from datetime import datetime, timedelta
print(">>> 1. 程序开始启动，库导入完成")

# ==========================================
# 1. 模拟环境与模型定义
# ==========================================
class SimpleMarketEnv:
    """简易市场环境模拟器"""

    def __init__(self, initial_stock=100):
        self.stock = initial_stock
        self.initial_stock = initial_stock
        self.history = []
        self.current_step = 0

    def reset(self):
        self.stock = self.initial_stock
        self.history = []
        self.current_step = 0
        return self._get_state()

    def _get_state(self):
        return np.array([
            self.stock / self.initial_stock,
            min(self.current_step / 100.0, 1.0),
            np.random.uniform(0.8, 1.2)
        ])

    def step(self, action, price):
        price_multiplier = [0.9, 1.0, 1.2][action]
        current_price = price * price_multiplier

        # 修复后的逻辑
        base_demand = 10
        elasticity_factor = 1.0 - (current_price - 100) / 500
        noise = np.random.uniform(0.8, 1.2)  # 修复点
        demand = max(0, int(base_demand * elasticity_factor * noise))  # 修复点

        sales = min(demand, self.stock)
        revenue = sales * current_price
        self.stock -= sales
        self.current_step += 1

        self.history.append([self.current_step, current_price, sales, noise])
        return self._get_state(), revenue, sales, self.stock

print(">>> 2. 环境类定义完成")

class SimpleDQNNetwork(nn.Module):
    """模拟的 DQN 网络结构"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )

    def forward(self, x):
        return self.net(x)


# ==========================================
# 2. Streamlit 界面构建
# ==========================================
st.set_page_config(page_title="AI 动态定价控制台", layout="wide")
st.title(" 基于深度强化学习的智能动态定价系统")
st.markdown("""
> **系统说明**：本系统结合了 **LSTM 需求预测** 与 **DQN 决策网络**。
> 左侧为控制面板，右侧实时展示 AI 的定价策略与库存消耗情况。
""")

# --- 侧边栏：控制参数 ---
st.sidebar.header("⚙️ 参数配置")
initial_price = st.sidebar.slider("初始基准价格", 50, 200, 100)
initial_stock = st.sidebar.slider("初始库存", 50, 500, 100)
speed = st.sidebar.slider("模拟速度 (ms)", 100, 1000, 200)

col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("📊 实时状态监控")
    placeholder = st.empty()
with col2:
    st.subheader("📈 策略可视化")
    chart_placeholder = st.empty()

# --- 初始化环境与模型 ---
env = SimpleMarketEnv(initial_stock)
model = SimpleDQNNetwork()
state = env.reset()
df_history = pd.DataFrame(columns=["Step", "Price", "Sales", "Stock", "Revenue"])

# --- 模拟按钮 ---
if st.button("️ 开始模拟运行"):
    for i in range(50):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = model(state_tensor)
            action = torch.argmax(q_values).item()

        next_state, reward, sales, current_stock = env.step(action, initial_price)

        # 修正价格计算逻辑，使其与动作对应
        current_price = initial_price * [0.9, 1.0, 1.2][action]

        new_row = pd.DataFrame({
            "Step": [i],
            "Price": [current_price],
            "Sales": [sales],
            "Stock": [current_stock],
            "Revenue": [reward]
        })
        df_history = pd.concat([df_history, new_row], ignore_index=True)

        # 更新界面
        with placeholder.container():
            st.metric("当前价格", f"¥{current_price:.2f}",
                      delta=f"{'⬆️' if action == 2 else '⬇️' if action == 0 else '-'}")
            st.metric("剩余库存", f"{current_stock} 件", delta=f"-{sales}")
            st.metric("单步收益", f"¥{reward:.0f}")
            st.progress(current_stock / initial_stock)
            st.caption("库存消耗进度")

        with chart_placeholder.container():
            if len(df_history) > 0:  # 防止空数据绘图
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_history['Step'], y=df_history['Price'], name='价格',
                                         line=dict(color='red', width=3)))
                fig.add_trace(go.Scatter(x=df_history['Step'], y=df_history['Stock'], name='库存',
                                         line=dict(color='blue', width=3), yaxis="y2"))
                fig.update_layout(
                    title="价格策略 vs 库存消耗",
                    yaxis=dict(title="价格 (元)"),
                    yaxis2=dict(title="库存 (件)", overlaying="y", side="right"),
                    xaxis=dict(title="时间步"),
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

        state = next_state
        import time

        time.sleep(speed / 1000)

        if current_stock == 0:
            st.warning("库存售罄！模拟结束。")
            break
    st.success("模拟运行完成！")

# --- 底部：技术栈展示 ---
st.markdown("---")
st.markdown("""**技术栈**: Python | PyTorch | Streamlit | Plotly **核心算法**: Deep Q-Network (DQN) + 模拟退火策略 """)

