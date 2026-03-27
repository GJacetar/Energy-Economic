import pandas as pd
import numpy as np
import numpy_financial as npf
import os
import re

SYSTEM_COST_PER_W = 3.2
OPEX_PER_KW_YEAR = 50
DEGRADATION_RATE = 0.004
LIFETIME_YEARS = 25
DISCOUNT_RATE = 0.08
FEED_IN_TARIFF = 0.35

# 国标物理参数
PV_DENSITY_KW_PER_SQM = 0.12

def safe_float(val):
    if pd.isna(val) or str(val).strip() in ['无', 'nan', '', 'None']:
        return None
    try:
        clean_val = re.sub(r'[^\d.]', '', str(val))
        return float(clean_val) if clean_val else None
    except:
        return None


def parse_time_rules(time_str):
    if pd.isna(time_str) or str(time_str).strip() in ['无', 'nan', '']:
        return []
    hours = set()
    clean_str = str(time_str).replace('、', ',')
    ranges = clean_str.split(',')
    for r in ranges:
        if '-' not in r: continue
        parts = r.split('-')
        if len(parts) != 2: continue
        start_part, end_part = parts[0], parts[1]
        match_start = re.search(r'(\d{1,2}):', start_part)
        match_end = re.search(r'(\d{1,2}):', end_part)
        if not match_start or not match_end: continue
        start_h = int(match_start.group(1))
        if start_h >= 24: start_h = 0
        end_h = int(match_end.group(1))
        if '次日' in end_part or end_h < start_h:
            hours.update(range(start_h, 24))
            hours.update(range(0, end_h))
        else:
            if end_h == 24: end_h = 24
            hours.update(range(start_h, end_h))
    return list(hours)

def build_8760_price_array(city_grid, df_price):
    df_grid = df_price[df_price['省份'] == city_grid].copy()
    if df_grid.empty: return None
    df_grid['month_num'] = df_grid['执行月份'].str.extract(r'(\d+)月').astype(int)
    df_grid = df_grid.sort_values('month_num')
    price_8760 = np.zeros(8760)
    dates = pd.date_range('2025-01-01', '2025-12-31 23:00:00', freq='h')
    for _, row in df_grid.iterrows():
        month = int(row['month_num'])
        month_mask = dates.month == month
        month_hours = dates[month_mask].hour
        base_price = safe_float(row['平段电价(元)'])
        if base_price is None: base_price = 0.5
        price_array = np.full(month_mask.sum(), base_price)
        v_h = parse_time_rules(row['低谷时段'])
        p_h = parse_time_rules(row['高峰时段'])
        d_h = parse_time_rules(row['深谷时段'])
        c_h = parse_time_rules(row['尖峰时段'])
        v_p = safe_float(row['低谷电价(元)'])
        p_p = safe_float(row['高峰电价(元)'])
        c_p = safe_float(row['尖峰电价(元)'])
        d_p = safe_float(row['深谷电价(元)'])
        if v_h and v_p is not None: price_array[np.isin(month_hours, v_h)] = v_p
        if p_h and p_p is not None: price_array[np.isin(month_hours, p_h)] = p_p
        if c_h and c_p is not None: price_array[np.isin(month_hours, c_h)] = c_p
        if d_h and d_p is not None: price_array[np.isin(month_hours, d_h)] = d_p
        price_8760[month_mask] = price_array
    return price_8760

def calc_payback(cash_flows):
    cum_cf = np.cumsum(cash_flows)
    if cum_cf[-1] < 0: return 99.9
    for i in range(1, len(cum_cf)):
        if cum_cf[i] >= 0 and cum_cf[i - 1] < 0:
            fraction = abs(cum_cf[i - 1]) / cash_flows[i]
            return round(i - 1 + fraction, 2)
    return 99.9


def run_economic_simulation():

    try:
        df_cities = pd.read_csv("City_LatLon_Grid_Mapping_Gaode.csv")
        try:
            df_price = pd.read_csv("电价数据.csv", encoding='utf-8').dropna(subset=['省份', '执行月份'])
        except:
            df_price = pd.read_csv("电价数据.csv", encoding='gbk').dropna(subset=['省份', '执行月份'])
        df_load = pd.read_csv("Typical_Load_Profiles_8760h.csv")
    except Exception as e:
        print(f"基础数据读取失败: {e}")
        return

    df_price['省份'] = df_price['省份'].str.strip()
    results = []
    load_types = ['Heavy_Industry_kW', 'Daytime_Mfg_kW', 'Commercial_kW']

    for _, city_row in df_cities.iterrows():
        prov, city, grid = city_row['Province'], city_row['City'], str(city_row['Grid_Region']).strip()
        pv_file = f"PV_Profiles_NASA_2025/PV_2025_8760h_{prov}_{city}.csv"
        if not os.path.exists(pv_file): continue

        df_pv = pd.read_csv(pv_file)
        pv_output_8760 = df_pv['PV_Output_kW'].values
        price_8760 = build_8760_price_array(grid, df_price)
        if price_8760 is None: continue

        for load_type in load_types:
            base_load_8760 = df_load[load_type].values

            # 蒙特卡洛随机采样：每种场景10次独立演化
            for mc_iter in range(10):
                if 'Heavy' in load_type:
                    peak_load_kw = np.random.uniform(10000, 30000)
                    pv_to_load_ratio = np.random.uniform(0.05, 0.20)
                    usable_ratio = np.random.uniform(0.70, 0.85)
                elif 'Commercial' in load_type:
                    peak_load_kw = np.random.uniform(2000, 6000)
                    pv_to_load_ratio = np.random.uniform(0.15, 0.40)
                    usable_ratio = np.random.uniform(0.20, 0.35)
                else:  # Daytime_Mfg (轻工业/物流)
                    peak_load_kw = np.random.uniform(2000, 5000)
                    pv_to_load_ratio = np.random.uniform(0.50, 0.80)
                    usable_ratio = np.random.uniform(0.75, 0.90)

                actual_pv_capacity_kw = peak_load_kw * pv_to_load_ratio
                roof_area_sqm = actual_pv_capacity_kw / (PV_DENSITY_KW_PER_SQM * usable_ratio)

                actual_load_8760 = (base_load_8760 / np.max(base_load_8760)) * peak_load_kw
                actual_pv_output_8760 = pv_output_8760 * (actual_pv_capacity_kw / 1000.0)

                self_consume_kw = np.minimum(actual_pv_output_8760, actual_load_8760)
                feed_in_kw = np.maximum(actual_pv_output_8760 - actual_load_8760, 0)

                revenue_yr1 = np.sum(self_consume_kw * price_8760) + np.sum(feed_in_kw * FEED_IN_TARIFF)

                capex = -(actual_pv_capacity_kw * 1000 * SYSTEM_COST_PER_W)
                opex = actual_pv_capacity_kw * OPEX_PER_KW_YEAR

                cash_flows = [capex]
                for yr in range(LIFETIME_YEARS):
                    cash_flows.append(revenue_yr1 * ((1 - DEGRADATION_RATE) ** yr) - opex)

                try:
                    irr_val = npf.irr(cash_flows)
                    irr = round(irr_val * 100, 2) if not pd.isna(irr_val) else 0.0
                except:
                    irr = 0.0

                npv = round(npf.npv(DISCOUNT_RATE, cash_flows), 2)

                base_annual_yield = np.sum(pv_output_8760)

                results.append({
                    'Province': prov,
                    'City': city,
                    'Grid_Region': grid,
                    'Load_Type': load_type.replace('_kW', ''),
                    'Lat': city_row['Lat'],
                    'Lon': city_row['Lon'],
                    'Roof_Area_sqm': round(roof_area_sqm, 2),
                    'Usable_Ratio': round(usable_ratio, 4),
                    'PV_Capacity_kW': round(actual_pv_capacity_kw, 2),
                    'Peak_Load_kW': round(peak_load_kw, 2),
                    'PV_to_Load_Ratio': round(pv_to_load_ratio, 4),
                    'Annual_PV_Yield_kWh': round(base_annual_yield, 2),
                    'Self_Consume_Rate_%': round((np.sum(self_consume_kw) / np.sum(actual_pv_output_8760)) * 100,
                                                 2) if np.sum(actual_pv_output_8760) > 0 else 0,
                    'Peak_Valley_Spread': round(np.max(price_8760) - np.min(price_8760), 4),
                    'Payback_Years': calc_payback(cash_flows),
                    'IRR_%': irr,
                    'NPV_Yuan': npv
                })

    df_results = pd.DataFrame(results)
    df_results.to_csv("Final_IRR_Dataset_for_AI_MC.csv", index=False, encoding='utf-8-sig')
    print(
        f"生成 {len(df_results)} 条包含底层微观数据集，已保存为 Final_IRR_Dataset_for_AI_MC.csv")


if __name__ == "__main__":
    run_economic_simulation()