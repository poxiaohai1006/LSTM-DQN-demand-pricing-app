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

# --- 新增：简单的训练逻辑 (为了演示，实际应用需要更复杂的训练) ---
st.write("🧠 **系统正在思考...** (正在进行 500 轮模拟训练以寻找最优策略)")
progress_bar = st.progress(0)

# 简单的训练参数
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

for episode in range(500):
    state = env.reset()
    # 模拟退火：随着训练进行，减少随机探索
    epsilon = max(0.01, 0.5 * (1 - episode / 500))

    for _ in range(20):  # 每次模拟20步
        # 1. 选择动作 (Epsilon-Greedy)
        if np.random.rand() < epsilon:
            action = np.random.choice(3)  # 随机探索
        else:
            with torch.no_grad():
                q_vals = model(torch.FloatTensor(state).unsqueeze(0))
                action = torch.argmax(q_vals).item()  # 利用策略

        # 2. 执行动作
        next_state, reward, _, _ = env.step(action, initial_price)

        # 3. 学习 (简化版：只学习当前步)
        with torch.no_grad():
            target = torch.FloatTensor([reward])
        pred = model(torch.FloatTensor(state).unsqueeze(0))[0, action]

        loss = loss_fn(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state

    progress_bar.progress((episode + 1) / 500)

st.success(" 训练完成！正在使用最优策略进行演示...")

# --- 模拟按钮 ---
if st.button("️ 开始模拟运行 (展示最优决策)"):
    log_placeholder = st.empty()
    log_container = []

    # 重置环境
    state = env.reset()
    df_history = pd.DataFrame(columns=["Step", "Price", "Sales", "Stock", "Revenue"])

    for i in range(50):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = model(state_tensor)
            # 关键变化：现在我们相信模型，不再随机探索
            action = torch.argmax(q_values).item()

            # ... (后续的绘图和日志逻辑保持不变) ...
            price_multiplier = [0.9, 1.0, 1.2][action]
            current_price = initial_price * price_multiplier
            next_state, reward, sales, current_stock = env.step(action, initial_price)

            # --- 📝 构建决策日志 ---
            action_map = {0: "📉 降价", 1: "⏸️ 持平", 2: "📈 涨价"}
            log_entry = f"**Step {i + 1:02d}**: 价格={current_price:6.1f} | 动作={action_map[action]} | 售出={sales:2d} | 收益=¥{reward:6.0f}"
            log_container.append(log_entry)

            # ... (更新界面和图表的代码) ...

            # 更新日志显示
            with log_placeholder.container():
                st.markdown("#### 📋 最优决策日志 (基于训练)")
                for log in log_container[-5:]:
                    st.text(log)

            # --- 新增：构建可视化图表 ---
            # 1. 准备历史数据
            if len(env.history) > 0:
                # 将 env.history 转换为 DataFrame
                columns = ['Step', 'Price', 'Sales', 'Noise']
                df_history = pd.DataFrame(env.history, columns=columns)

                # 2. 创建双轴图表
                fig = go.Figure()

                # 轨迹 1: 价格走势 (红色实线)
                fig.add_trace(go.Scatter(
                    x=df_history['Step'],
                    y=df_history['Price'],
                    name='价格走势',
                    mode='lines+markers',
                    line=dict(color='firebrick', width=3),
                    marker=dict(size=6)
                ))

                # 轨迹 2: 库存消耗 (蓝色虚线)
                # 注意：库存需要根据初始库存和销量累计计算
                initial_stock_current = initial_stock
                df_history['Stock'] = initial_stock_current - df_history['Sales'].cumsum()

                fig.add_trace(go.Scatter(
                    x=df_history['Step'],
                    y=df_history['Stock'],
                    name='库存剩余',
                    mode='lines+markers',
                    line=dict(color='royalblue', width=3, dash='dash'),
                    marker=dict(size=6)
                ))

                # 3. 美化布局
                fig.update_layout(
                    title="📊 AI 动态定价与库存消耗实时监控",
                    xaxis_title="时间步 (Step)",
                    yaxis_title="金额 / 库存数量",
                    hovermode="x unified",
                    template="plotly_white",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    height=400
                )

                # 4. 更新界面
                with chart_placeholder.container():
                    st.plotly_chart(fig, use_container_width=True)

            state = next_state
            import time

            time.sleep(speed / 1000)

            if current_stock == 0:
                st.warning("库存售罄！模拟结束。")
                break
