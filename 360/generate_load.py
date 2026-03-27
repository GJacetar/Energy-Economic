import pandas as pd
import numpy as np


def generate_peak_normalized_loads(base_load_kw=1000):
    print("2025标幺化峰值负荷")

    hours = pd.date_range(start="2025-01-01 00:00:00", periods=8760, freq="h")
    df = pd.DataFrame({'Datetime': hours})

    h = hours.hour
    wd = hours.weekday  # 0=周一, 6=周日
    m = hours.month

    # 基础随机波动噪音 (±2%)
    np.random.seed(42)
    noise = np.random.normal(0, 0.02, 8760)

    # ==========================================
    # 场景 A：重工业
    # ==========================================
    heavy_profile = np.array([
        0.95, 0.94, 0.94, 0.95, 0.96, 0.95, 0.94, 0.95,
        0.98, 0.99, 1.00, 0.98, 0.95, 0.97, 0.99, 1.00,
        0.98, 0.97, 0.96, 0.98, 0.99, 0.97, 0.96, 0.95
    ])
    load_A = heavy_profile[h] * (1 + noise)

    # ==========================================
    # 场景 B：轻工业
    # ==========================================
    light_profile = np.array([
        0.15, 0.15, 0.15, 0.15, 0.15, 0.20, 0.40, 0.70,
        0.95, 1.00, 0.95, 0.90, 0.60, 0.85, 0.95, 0.95,
        0.90, 0.70, 0.40, 0.25, 0.20, 0.15, 0.15, 0.15
    ])
    load_B = light_profile[h] * 1.0
    load_B[wd >= 5] = load_B[wd >= 5] * 0.3  # 周末放假降至30%
    load_B = load_B * (1 + noise)

    # ==========================================
    # 场景 C：大型商业综合体
    # ==========================================
    commercial_profile = np.array([
        0.10, 0.10, 0.10, 0.10, 0.10, 0.12, 0.20, 0.40,
        0.70, 0.90, 1.00, 1.00, 0.95, 0.95, 1.00, 0.95,
        0.90, 0.80, 0.70, 0.60, 0.50, 0.30, 0.15, 0.10
    ])
    load_C = commercial_profile[h] * 1.0
    load_C[wd >= 5] = load_C[wd >= 5] * 1.1  # 周末客流上浮10%

    summer_mask = (m >= 6) & (m <= 9)
    winter_mask = (m == 12) | (m <= 2)
    load_C[summer_mask] = load_C[summer_mask] * np.where((h[summer_mask] >= 11) & (h[summer_mask] <= 16), 1.4, 1.1)
    load_C[winter_mask] = load_C[winter_mask] * np.where(
        ((h[winter_mask] >= 8) & (h[winter_mask] <= 10)) | ((h[winter_mask] >= 17) & (h[winter_mask] <= 19)), 1.3, 1.1)
    load_C = load_C * (1 + noise)


    load_A = (load_A / np.max(load_A)) * base_load_kw
    load_B = (load_B / np.max(load_B)) * base_load_kw
    load_C = (load_C / np.max(load_C)) * base_load_kw

    # 格式化输出
    df['Heavy_Industry_kW'] = np.clip(load_A, 0, None)
    df['Daytime_Mfg_kW'] = np.clip(load_B, 0, None)
    df['Commercial_kW'] = np.clip(load_C, 0, None)

    df['Datetime'] = df['Datetime'].dt.tz_localize(None)
    df.to_csv("Typical_Load_Profiles_8760h.csv", index=False, encoding='utf-8-sig')



if __name__ == "__main__":
    generate_peak_normalized_loads()