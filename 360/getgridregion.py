import requests
import pandas as pd
import time


# 1. 中国电网映射字典
def get_grid_region(province, city):
    prov_clean = province.replace('省', '').replace('市', '').replace('维吾尔自治区', '').replace('回族自治区',
                                                                                                  '').replace(
        '壮族自治区', '').replace('自治区', '')

    if prov_clean == '河北':
        if city in ['张家口市', '承德市', '唐山市', '秦皇岛市', '廊坊市']:
            return '冀北'
        else:
            return '国网河北省电力有限公司'

    elif prov_clean == '内蒙古':
        if city in ['呼伦贝尔市', '兴安盟', '通辽市', '赤峰市']:
            return '蒙东'
        else:
            return '蒙西'

    elif prov_clean == '广东':
        if city in ['广州市', '珠海市', '佛山市', '中山市', '东莞市', '深圳市']:
            return '广东(珠三角五市)'
        elif city == '惠州市':
            return '广东(惠州市)'
        elif city == '江门市':
            return '广东(江门市)'
        elif city in ['汕头市', '潮州市', '揭阳市', '汕尾市', '阳江市', '湛江市', '茂名市', '肇庆市']:
            return '广东(东西两翼地区)'
        elif city in ['云浮市', '河源市', '梅州市', '韶关市', '清远市']:
            return '广东(粤北山区)'
        else:
            return '广东(珠三角五市)'

    else:
        if prov_clean in ['北京', '上海', '天津', '重庆']:
            return prov_clean
        return prov_clean


# 2. 调用高德 API 获取全国城市与经纬度（加入无损重试机制）
def generate_city_mapping(api_key):
    print("通过高德地图 API，拉取全国行政区划与经纬度...")

    url = f"https://restapi.amap.com/v3/config/district?keywords=中国&subdistrict=2&key={api_key}"

    res = None
    success = False  # 成功标志位

    # 若网络/并发报错，原地等待后重试
    while not success:
        try:
            print("重新发送请求...", end=" ", flush=True)
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            res = response.json()

            if res.get('status') == '1':
                print("请求成功并解析数据...")
                success = True
            else:
                info = res.get('info')
                infocode = res.get('infocode')
                if infocode == '10003':
                    print(f"\n   [API] {info}，等待3秒重试")
                    time.sleep(3)
                else:
                    print(f"\n   [错误] {info}。Key 配置错误。")
                    return None

        except requests.exceptions.RequestException as e:
            # 捕获网络断开、超时等错误
            print(f"\n    网络异常 ({e})，等待3秒重试")
            time.sleep(3)
        except Exception as e:
            # 捕获 JSON 解析等其他异常
            print(f"\n   异常 ({e})，等待3重试")
            time.sleep(3)

    # ================= 数据解析模块 =================
    city_data = []
    provinces = res['districts'][0]['districts']

    for prov in provinces:
        prov_name = prov['name']
        if prov_name in ['香港特别行政区', '澳门特别行政区', '台湾省']:
            continue

        cities = prov['districts']

        if not cities or prov_name in ['北京市', '天津市', '上海市', '重庆市']:
            grid = get_grid_region(prov_name, prov_name)
            lon, lat = prov['center'].split(',')
            city_data.append({
                'Province': prov_name,
                'City': prov_name,
                'Lon': float(lon),
                'Lat': float(lat),
                'Grid_Region': grid
            })
        else:
            for city in cities:
                city_name = city['name']
                grid = get_grid_region(prov_name, city_name)
                lon, lat = city['center'].split(',')
                city_data.append({
                    'Province': prov_name,
                    'City': city_name,
                    'Lon': float(lon),
                    'Lat': float(lat),
                    'Grid_Region': grid
                })

    df_cities = pd.DataFrame(city_data)
    df_cities = df_cities[df_cities['City'].str.contains('市|州|盟|地区')]

    df_cities.to_csv("City_LatLon_Grid_Mapping_Gaode.csv", index=False, encoding='utf-8-sig')
    print(f"成功获取 {len(df_cities)} 个地级行政区坐标。")
    print("结果已保存至 City_LatLon_Grid_Mapping_Gaode.csv")

    return df_cities

# 高德 API Key

GAODE_API_KEY = "fb2971c8704085444d0ceeaaee639005"

if GAODE_API_KEY != "高德Web服务Key":
    df_result = generate_city_mapping(GAODE_API_KEY)
    if df_result is not None:
        print("\n======")
        print(df_result.head())
else:
    print("错误")