import pandas as pd
import numpy as np
import requests
import pvlib
import os
import time

INPUT_CSV = "City_LatLon_Grid_Mapping_Gaode.csv"
OUTPUT_DIR = "../PV_Profiles_NASA_2025"
PEAK_KW = 1000

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def fetch_nasa_and_simulate_pv(lat, lon, target_year=2025):
    # 1. 跨年请求，覆盖时差真空期
    start_date = f"{target_year - 1}1231"
    end_date = f"{target_year}1231"

    url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
    params = {
        "parameters": "ALLSKY_SFC_SW_DWN,T2M,WS10M",
        "community": "RE",
        "longitude": lon,
        "latitude": lat,
        "start": start_date,
        "end": end_date,
        "format": "JSON",
        "time-standard": "UTC"
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    df = pd.DataFrame(data['properties']['parameter'])

    # 2. 时区转化：UTC -> 北京时间
    df.index = pd.to_datetime(df.index, format='%Y%m%d%H')
    df.index = df.index.tz_localize('UTC').tz_convert('Asia/Shanghai')
    df = df[~df.index.duplicated(keep='first')]

    df.rename(columns={
        'ALLSKY_SFC_SW_DWN': 'ghi',
        'T2M': 'temp_air',
        'WS10M': 'wind_speed'
    }, inplace=True)

    # 3. 气象缺失值处理
    df = df.replace(-999.0, np.nan)
    df['temp_air'] = df['temp_air'].interpolate(method='linear', limit_direction='both')
    df['wind_speed'] = df['wind_speed'].interpolate(method='linear', limit_direction='both')
    df['ghi'] = df['ghi'].fillna(0)

    # 4. 辐射分离
    solpos = pvlib.solarposition.get_solarposition(df.index, lat, lon)
    dni_dhi = pvlib.irradiance.erbs(df['ghi'], solpos['apparent_zenith'], df.index)
    df['dni'] = dni_dhi['dni'].fillna(0)
    df['dhi'] = dni_dhi['dhi'].fillna(0)

    # 5. 配置 PVWatts 系统与温度模型
    temp_model_params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']

    system = pvlib.pvsystem.PVSystem(
        surface_tilt=lat,
        surface_azimuth=180,
        module_parameters={'pdc0': PEAK_KW * 1000, 'gamma_pdc': -0.004},
        inverter_parameters={'pdc0': PEAK_KW * 1000 / 0.96},
        temperature_model_parameters=temp_model_params
    )

    location = pvlib.location.Location(lat, lon, tz='Asia/Shanghai')

    mc = pvlib.modelchain.ModelChain(
        system,
        location,
        dc_model='pvwatts',
        ac_model='pvwatts',
        aoi_model='no_loss',
        spectral_model='no_loss'
    )

    mc.run_model(df)
    pv_power_kw = mc.results.ac / 1000

    df_out = pd.DataFrame({
        'Datetime': df.index,
        'PV_Output_kW': pv_power_kw.values
    })

    # 抹平逆变器夜间微弱负荷
    df_out['PV_Output_kW'] = df_out['PV_Output_kW'].clip(lower=0)

    # 生成 2025 全年标准 8760 小时脚手架
    start_time = pd.Timestamp(f'{target_year}-01-01 00:00:00', tz='Asia/Shanghai')
    end_time = pd.Timestamp(f'{target_year}-12-31 23:00:00', tz='Asia/Shanghai')
    standard_8760_idx = pd.date_range(start=start_time, end=end_time, freq='h')

    # 将实际跑出的数据放入脚手架
    df_out = df_out.set_index('Datetime')
    df_out = df_out.reindex(standard_8760_idx).fillna(0)
    df_out.index.name = 'Datetime'
    df_out = df_out.reset_index()

    # 与电价表匹配
    df_out['Datetime'] = df_out['Datetime'].dt.tz_localize(None)

    # 验证
    if len(df_out) != 8760:
        raise ValueError(f"错误: 应得到 8760 行，实际生成了 {len(df_out)} 行！")

    return df_out


def batch_download_nasa_pv():
    df_cities = pd.read_csv(INPUT_CSV)
    total_cities = len(df_cities)


    success_count = 0

    for index, row in df_cities.iterrows():
        prov = row['Province']
        city = row['City']
        lat = row['Lat']
        lon = row['Lon']

        file_name = f"PV_2025_8760h_{prov}_{city}.csv"
        file_path = os.path.join(OUTPUT_DIR, file_name)

        if os.path.exists(file_path):
            print(f"[{index + 1}/{total_cities}] {prov}-{city} 已存在，跳过。")
            success_count += 1
            continue

        print(f"⏳ [{index + 1}/{total_cities}] 请求 NASA: {prov}-{city}...", end=" ", flush=True)

        success = False
        retry_limit = 10
        retries = 0

        while not success and retries < retry_limit:
            try:
                df_pv = fetch_nasa_and_simulate_pv(lat, lon, target_year=2025)
                df_pv.to_csv(file_path, index=False)

                print("成功！")
                success = True
                success_count += 1
                time.sleep(1.2)  # NASA限流保护

            except requests.exceptions.RequestException as e:
                retries += 1
                print(f"\n   [网络波动] ({retries}/{retry_limit}) 等待 5 秒重试...")
                time.sleep(5)
            except ValueError as ve:
                retries += 1
                print(f"\n   [丢包异常] {ve} ({retries}/{retry_limit}) 重新抓取中...")
                time.sleep(3)
            except Exception as e:
                retries += 1
                print(f"\n   [运算异常] {e} ({retries}/{retry_limit}) 等待 3 秒重试...")
                time.sleep(3)

    print(f"\n成功获取 {success_count}/{total_cities} 个城市。")


if __name__ == "__main__":
    batch_download_nasa_pv()