import streamlit as st
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ==========================================
# PAGE CONFIGURATION & CUSTOM STYLING
# ==========================================
st.set_page_config(page_title="TAR UMT AI Food Delivery", page_icon="🤖", layout="wide")

st.markdown("""
<style>
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }
    div.stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #0d9488 0%, #06b6d4 100%);
        color: white;
        font-weight: 700;
        border-radius: 10px;
        padding: 0.65rem 1rem;
        border: none;
    }
    div.stButton > button:hover {
        background: linear-gradient(90deg, #0f766e 0%, #0891b2 100%);
        color: white;
    }
    .hud-card {
        background: rgba(30, 41, 59, 0.85);
        border: 1px solid #334155;
        border-left: 5px solid #06b6d4;
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. CAMPUS DELIVERY ENVIRONMENT (36 NODES)
# ==========================================
class CampusDeliveryEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.all_nodes = list(range(36))
        
        self.node_names = {
            0: "Red Bricks", 1: "DK4", 2: "Rimba", 3: "Block C",
            4: "Yum Yum", 5: "Arena", 6: "Block M", 7: "Library",
            8: "Roof Top D", 9: "Hostel", 10: "Block Q (West Slope)", 11: "DK W",
            12: "Carpark C", 13: "Roof Top C", 14: "DKABE", 15: "Roof Top E",
            16: "Clubhouse", 17: "Block W", 18: "Block V", 19: "DTAR",
            20: "Swimming Pool", 21: "Sport Complex", 22: "FM (CITC)", 23: "Roof Top A",
            24: "Bus Stop A", 25: "Carpark A", 26: "DSA", 27: "Basketball Court",
            28: "Roof Top B", 29: "Car Park B", 30: "Casaurina Cafe", 31: "East Slope",
            32: "Bus Stop B", 33: "Roof Top F", 34: "Roof Top G", 35: "Carpark D"
        }
        
        # 2D Map (X, Y) Coordinates matching hand-drawn campus layout (0-100 scale)
        self.node_coords = {
            0: (6, 48),   1: (15, 48),  2: (21, 38),  3: (32, 38),
            4: (43, 38),  5: (58, 25),  6: (43, 47),  7: (43, 56),
            8: (60, 90),  9: (80, 92),  10: (22, 80), 11: (35, 83),
            12: (71, 25), 13: (69, 34), 14: (73, 43), 15: (76, 68),
            16: (64, 57), 17: (67, 50), 18: (67, 43), 19: (63, 37),
            20: (63, 64), 21: (63, 72), 22: (15, 23), 23: (12, 12),
            24: (25, 9),  25: (26, 23), 26: (35, 24), 27: (45, 16),
            28: (51, 33), 29: (54, 56), 30: (94, 61), 31: (93, 52),
            32: (87, 45), 33: (82, 52), 34: (81, 41), 35: (84, 76)
        }
        
        self.graph = {
            0: [1], 1: [0, 2, 10], 2: [1, 3, 26], 3: [2, 4],
            4: [3, 6, 26, 27, 28], 5: [12, 13, 19, 27, 28], 6: [4, 7],
            7: [6, 8, 29], 8: [7, 9, 11, 15, 21, 35], 9: [8, 15, 35],
            10: [1, 11], 11: [10, 8], 12: [5, 14], 13: [5, 14],
            14: [12, 13, 15, 33, 34], 15: [8, 9, 14, 33], 16: [17, 20, 29],
            17: [16, 18], 18: [17, 19], 19: [18, 5], 20: [16, 21],
            21: [20, 8], 22: [23, 25], 23: [22, 24], 24: [23, 26, 27],
            25: [22, 26], 26: [2, 4, 24, 25, 27], 27: [4, 5, 24, 26],
            28: [4, 5], 29: [7, 16], 30: [31, 35], 31: [30, 32, 33],
            32: [31, 34], 33: [14, 15, 31], 34: [14, 32], 35: [8, 9, 30]
        }
        
        self.start_locations = [0, 4, 16, 22, 30]
        self.target_destinations = [5, 9, 2] # Arena, Hostel, Rimba
        
        self.food_menus = {
            0: ["Nasi Lemak", "Roti Canai", "Teh Tarik"],
            4: ["Chicken Rice", "Wantan Mee", "Mixed Rice"],
            16: ["Club Sandwich", "Burger & Fries", "Iced Lemon Tea"],
            22: ["Japanese Bento", "Onigiri", "Oden"],
            30: ["Western Chicken Chop", "Fried Kuey Teow", "Iced Coffee"]
        }
        
        self.uncovered_paths = [
            frozenset({1, 10}), frozenset({10, 11}),
            frozenset({27, 5}),
            frozenset({5, 12}), frozenset({12, 14}),
            frozenset({22, 25}), frozenset({25, 26}),
            frozenset({7, 29}), frozenset({29, 16}),
            frozenset({30, 35}), frozenset({35, 9}), frozenset({35, 8})
        ]
        
        self.pedestrian_zones = {0, 1, 4, 5, 16, 20, 21, 24, 27, 28, 30, 31, 32}
        self.car_zones = {3, 4, 5, 8, 10, 12, 21, 23, 25, 26, 27, 28, 29, 33, 34, 35}
        
        self.longer_paths = {
            frozenset({23, 24}): -3, frozenset({24, 27}): -2,
            frozenset({1, 10}): -2,  frozenset({7, 8}): -2,
            frozenset({14, 15}): -2, frozenset({30, 35}): -2
        }

        self.action_space = spaces.Discrete(37)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(36), spaces.Discrete(36), spaces.Discrete(2), spaces.Discrete(3)
        ))
        
        self.current_location = 0
        self.target_destination = 5
        self.current_weather = 0
        self.current_traffic = 0
        self.selected_food = "Nasi Lemak"

    def _get_obs(self):
        return (self.current_location, self.target_destination, self.current_weather, self.current_traffic)

    def _get_info(self):
        traffic_map = {0: "Clear 🟢", 1: "Pedestrian Ahead 🚶", 2: "Car Ahead 🚗"}
        return {
            "location_str": self.node_names[self.current_location],
            "target_str": self.node_names[self.target_destination],
            "weather_str": "Sunny ☀️" if self.current_weather == 0 else "Rainy 🌧️",
            "traffic_str": traffic_map[self.current_traffic],
            "food_item": self.selected_food
        }

    def _scan_traffic(self):
        obstacle_prob = 0.25 if self.current_weather == 0 else 0.50
        if random.random() < obstacle_prob:
            in_ped = self.current_location in self.pedestrian_zones
            in_car = self.current_location in self.car_zones
            if in_ped and in_car:
                return random.choice([1, 2])
            elif in_car:
                return 2
            elif in_ped:
                return 1
        return 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options is not None:
            self.target_destination = options.get("target", random.choice(self.target_destinations))
            self.current_location = options.get("start", random.choice(self.start_locations))
            self.current_weather = options.get("weather", random.choice([0, 1]))
            self.selected_food = options.get("food", random.choice(self.food_menus[self.current_location]))
        else:
            self.target_destination = random.choice(self.target_destinations)
            self.current_location = random.choice(self.start_locations)
            self.current_weather = random.choice([0, 1])
            self.selected_food = random.choice(self.food_menus[self.current_location])
            
        self.current_traffic = self._scan_traffic()
        return self._get_obs(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False
        truncated = False
        
        if action == 36: # Wait Action
            reward = -2
            self.current_traffic = self._scan_traffic()
            return self._get_obs(), reward, terminated, truncated, self._get_info()
            
        if action not in self.graph[self.current_location]:
            reward = -5
            return self._get_obs(), reward, terminated, truncated, self._get_info()

        if self.current_traffic == 1:
            reward = -25
            terminated = True
            return self._get_obs(), reward, terminated, truncated, self._get_info()
            
        if self.current_traffic == 2:
            reward = -50
            terminated = True
            return self._get_obs(), reward, terminated, truncated, self._get_info()

        path_edge = frozenset({self.current_location, action})
        reward += self.longer_paths.get(path_edge, -1)
        
        if self.current_weather == 1 and path_edge in self.uncovered_paths:
            reward -= 15
            
        self.current_location = int(action)
        
        if self.current_location == self.target_destination:
            reward += 100
            terminated = True
        else:
            self.current_traffic = self._scan_traffic()
            
        return self._get_obs(), reward, terminated, truncated, self._get_info()


# ==========================================
# 2. CACHED Q-LEARNING TRAINING
# ==========================================
@st.cache_resource
def load_and_train_agent():
    env = CampusDeliveryEnv()
    q_table = np.zeros((36, 36, 2, 3, 37))
    
    total_episodes = 25000
    learning_rate = 0.15
    discount_factor = 0.99
    epsilon = 1.0
    epsilon_decay = 0.9998
    min_epsilon = 0.05
    
    for _ in range(total_episodes):
        state, _ = env.reset()
        for _ in range(70):
            if random.random() < epsilon:
                if random.random() < 0.75:
                    action = random.choice(env.graph[env.current_location] + [36])
                else:
                    action = env.action_space.sample()
            else:
                action = int(np.argmax(q_table[state]))
                
            next_state, reward, terminated, _, _ = env.step(action)
            old_val = q_table[state][action]
            next_max = np.max(q_table[next_state])
            q_table[state][action] = old_val + learning_rate * (reward + discount_factor * next_max - old_val)
            state = next_state
            if terminated:
                break
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
    return env, q_table


# ==========================================
# 3. 2D CAMPUS MAP RENDERER (WITH TOP-LEFT HUD)
# ==========================================
def render_campus_map(env, visited_nodes, current_node, target_node, score, step_num, status_text):
    is_rainy = (env.current_weather == 1)
    bg_color = "#0b132b" if is_rainy else "#0f172a"
    
    fig, ax = plt.subplots(figsize=(13, 8.2), facecolor=bg_color)
    ax.set_facecolor(bg_color)
    
    # 1. Draw all road connections (Covered vs Uncovered Yellow Paths)
    drawn_edges = set()
    for u, neighbors in env.graph.items():
        for v in neighbors:
            edge = frozenset({u, v})
            if edge not in drawn_edges:
                drawn_edges.add(edge)
                x1, y1 = env.node_coords[u]
                x2, y2 = env.node_coords[v]
                if edge in env.uncovered_paths:
                    ax.plot([x1, x2], [y1, y2], color="#facc15", linestyle="--", linewidth=2.6, alpha=0.85, zorder=1)
                else:
                    ax.plot([x1, x2], [y1, y2], color="#475569", linestyle="-", linewidth=2.2, alpha=0.8, zorder=1)

    # 2. Draw AI Robot's traveled route trail
    if len(visited_nodes) > 1:
        for i in range(len(visited_nodes) - 1):
            u, v = visited_nodes[i], visited_nodes[i + 1]
            x1, y1 = env.node_coords[u]
            x2, y2 = env.node_coords[v]
            ax.plot([x1, x2], [y1, y2], color="#06b6d4", linewidth=5.0, alpha=0.95, zorder=3)

    # 3. Draw all 36 Campus Nodes & Hazard Badges
    for node_id, (x, y) in env.node_coords.items():
        if node_id == current_node:
            color, size, edge_c, z = "#22d3ee", 260, "#ffffff", 6
        elif node_id == target_node:
            color, size, edge_c, z = "#10b981", 230, "#ffffff", 5
        elif node_id in env.start_locations:
            color, size, edge_c, z = "#f97316", 150, "#fed7aa", 4
        elif node_id in visited_nodes:
            color, size, edge_c, z = "#0ea5e9", 120, "#bae6fd", 4
        else:
            color, size, edge_c, z = "#334155", 85, "#94a3b8", 2
            
        ax.scatter(x, y, s=size, c=color, edgecolors=edge_c, linewidths=1.8, zorder=z)
        
        # Add subtle hazard indicator tags (P = Pedestrian zone, C = Car zone)
        hazard_tag = ""
        if node_id in env.pedestrian_zones and node_id in env.car_zones:
            hazard_tag = " [P/C]"
        elif node_id in env.car_zones:
            hazard_tag = " [C]"
        elif node_id in env.pedestrian_zones:
            hazard_tag = " [P]"
            
        label_color = "#ffffff" if node_id in [current_node, target_node] else "#cbd5e1"
        font_w = "bold" if node_id in [current_node, target_node] else "normal"
        ax.text(x, y + 2.2, f"{env.node_names[node_id]}{hazard_tag}", fontsize=7.5,
                ha="center", va="bottom", color=label_color, fontweight=font_w,
                bbox=dict(boxstyle="round,pad=0.18", fc="#1e293b", ec="none", alpha=0.75), zorder=7)

    # 4. Highlight Active Obstacle at Current Robot Position
    rx, ry = env.node_coords[current_node]
    if env.current_traffic == 1:
        ax.text(rx, ry - 4.2, "⚠️ PEDESTRIAN BLOCKING (-25)", fontsize=9, fontweight="bold",
                color="#fef08a", ha="center", bbox=dict(boxstyle="round,pad=0.3", fc="#b45309", ec="#facc15"), zorder=8)
    elif env.current_traffic == 2:
        ax.text(rx, ry - 4.2, "🚨 CAR BLOCKING (-50)", fontsize=9, fontweight="bold",
                color="#fecaca", ha="center", bbox=dict(boxstyle="round,pad=0.3", fc="#991b1b", ec="#f87171"), zorder=8)

    # 5. TOP-LEFT LIVE SCORE & TELEMETRY HUD BOX
    weather_banner = "RAINY WEATHER (Uncovered = -15 pts)" if is_rainy else "SUNNY WEATHER (All Paths Clear)"
    traffic_banner = {0: "CLEAR ROAD", 1: "PEDESTRIAN AHEAD (WAIT!)", 2: "CAR AHEAD (WAIT!)"}[env.current_traffic]
    
    hud_text = (
        f"LIVE DELIVERY HUD\n"
        f"────────────────────────\n"
        f"Score       : {score} pts\n"
        f"Steps Taken : {step_num}\n"
        f"Weather     : {weather_banner}\n"
        f"Road Sensor : {traffic_banner}\n"
        f"Current Loc : {env.node_names[current_node]}\n"
        f"Target Dest : {env.node_names[target_node]}\n"
        f"Order Item  : {env.selected_food}\n"
        f"Status      : {status_text}"
    )
    ax.text(1.5, 98.5, hud_text, fontsize=9, color="#f8fafc", family="monospace",
            va="top", ha="left", zorder=10,
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#1e293b", edgecolor="#06b6d4", linewidth=2, alpha=0.92))

    # 6. Map Legend in Bottom-Right
    legend_elements = [
        patches.Patch(facecolor="#22d3ee", edgecolor="white", label="AI Robot Position"),
        patches.Patch(facecolor="#10b981", edgecolor="white", label="Target Destination"),
        patches.Patch(facecolor="#f97316", edgecolor="white", label="Start Restaurant"),
        plt.Line2D([0], [0], color="#06b6d4", lw=4, label="AI Traveled Path"),
        plt.Line2D([0], [0], color="#475569", lw=2, label="Covered Walkway (Roof)"),
        plt.Line2D([0], [0], color="#facc15", lw=2, ls="--", label="Uncovered Path (No Roof)")
    ]
    ax.legend(handles=legend_elements, loc="lower right", facecolor="#1e293b", edgecolor="#475569",
              labelcolor="white", fontsize=8)

    ax.set_xlim(0, 102)
    ax.set_ylim(2, 103)
    ax.axis("off")
    plt.tight_layout()
    return fig


# ==========================================
# 4. STREAMLIT APPLICATION FLOW
# ==========================================
env, q_table = load_and_train_agent()

if "page" not in st.session_state:
    st.session_state.page = "order"

# ------------------------------------------
# VIEW 1: CUSTOMER ORDER SELECTION SCREEN
# ------------------------------------------
if st.session_state.page == "order":
    st.title("🍔 TAR UMT Autonomous Campus Food Delivery")
    st.markdown("Select your delivery destination, restaurant, and meal below. Once confirmed, the screen will automatically switch to the **2D Live Robot Tracking Map**.")
    
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.subheader("📋 Step 1 – Configure Your Order")
        
        target_options = {env.node_names[nid]: nid for nid in env.target_destinations}
        selected_target_name = st.selectbox("🎯 1. Choose Delivery Destination:", list(target_options.keys()))
        chosen_target = target_options[selected_target_name]
        
        start_options = {env.node_names[nid]: nid for nid in env.start_locations}
        selected_start_name = st.selectbox("🏪 2. Choose Restaurant (Start Location):", list(start_options.keys()))
        chosen_start = start_options[selected_start_name]
        
        available_Menu = env.food_menus[chosen_start]
        chosen_food = st.selectbox(f"🍱 3. Select Food from {selected_start_name}:", available_Menu)
        
        weather_mode = st.radio(
            "🌦️ 4. Weather Condition (for testing or auto-random):",
            ["🎲 Auto Random Weather", "☀️ Force Sunny", "🌧️ Force Rainy"],
            horizontal=True
        )
        
        st.write("")
        if st.button("🚀 Confirm Order & Launch Live 2D Tracking"):
            if weather_mode == "☀️ Force Sunny":
                w_val = 0
            elif weather_mode == "🌧️ Force Rainy":
                w_val = 1
            else:
                w_val = random.choice([0, 1])
                
            st.session_state.order_config = {
                "start": chosen_start,
                "target": chosen_target,
                "food": chosen_food,
                "weather": w_val
            }
            st.session_state.page = "tracking"
            st.rerun()

    with col2:
        st.subheader("🗺️ Campus Map Preview")
        env.current_weather = 0
        env.current_traffic = 0
        env.selected_food = chosen_food
        preview_fig = render_campus_map(
            env, visited_nodes=[chosen_start], current_node=chosen_start,
            target_node=chosen_target, score=0, step_num=0, status_text="Waiting for Order"
        )
        st.pyplot(preview_fig)
        plt.close(preview_fig)

# ------------------------------------------
# VIEW 2: 2D LIVE TRACKING MAP & SIMULATION
# ------------------------------------------
elif st.session_state.page == "tracking":
    cfg = st.session_state.order_config
    state, info = env.reset(options=cfg)
    
    top_col1, top_col2 = st.columns([4, 1])
    with top_col1:
        st.subheader(f"📡 Live 2D Robot Tracking: {info['location_str']} ➔ {info['target_str']} ({info['food_item']})")
    with top_col2:
        if st.button("🔄 New Order"):
            st.session_state.page = "order"
            st.rerun()
            
    hud_placeholder = st.empty()
    map_placeholder = st.empty()
    log_expander = st.expander("📜 View Step-by-Step AI Telemetry Log", expanded=True)
    
    total_score = 0
    step_num = 0
    terminated = False
    visited_nodes = [env.current_location]
    logs = []
    
    # Render Initial Frame
    fig = render_campus_map(env, visited_nodes, env.current_location, env.target_destination,
                            total_score, step_num, "Order Dispatched")
    map_placeholder.pyplot(fig)
    plt.close(fig)
    time.sleep(0.8)
    
    # Run Autonomous Q-Learning Loop
    while not terminated and step_num < 45:
        step_num += 1
        action = int(np.argmax(q_table[state]))
        
        if action == 36:
            obstacle_name = "Pedestrian 🚶" if env.current_traffic == 1 else "Car 🚗"
            status_msg = f"WAITING for {obstacle_name} (-2 pts)"
            next_state, reward, terminated, _, info = env.step(action)
            logs.append(f"Step {step_num}: ⚠️ {obstacle_name} detected at **{info['location_str']}**! AI chooses **WAIT** ({reward} pts).")
        else:
            next_state, reward, terminated, _, info = env.step(action)
            visited_nodes.append(env.current_location)
            status_msg = f"Moved to {info['location_str']} ({reward:+} pts)"
            logs.append(f"Step {step_num}: 🤖 Moved to **{info['location_str']}** | Step Reward: `{reward} pts` | Sensor: {info['traffic_str']}")
            
        total_score += reward
        state = next_state
        
        # Update Top Metric Cards
        with hud_placeholder.container():
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("🏆 Current Score", f"{total_score} pts", f"{reward:+} pts")
            m2.metric("👣 Steps Taken", f"{step_num} steps")
            m3.metric("🌦️ Live Weather", info["weather_str"])
            m4.metric("🚦 Road Sensor", info["traffic_str"])
            
        # Update 2D Tracking Map
        fig = render_campus_map(env, visited_nodes, env.current_location, env.target_destination,
                                total_score, step_num, status_msg)
        map_placeholder.pyplot(fig)
        plt.close(fig)
        
        with log_expander:
            st.markdown("<br>".join(logs[-8:]), unsafe_allow_html=True)
            
        time.sleep(0.65)
        
    if terminated and reward > 0:
        st.success(f"🎉 **DELIVERY SUCCESSFUL!** Delivered **{info['food_item']}** to **{info['target_str']}** in {step_num} steps! Final Score: **{total_score} pts**")
        st.balloons()
    else:
        st.error(f"💥 **DELIVERY TERMINATED!** Final Score: **{total_score} pts**")