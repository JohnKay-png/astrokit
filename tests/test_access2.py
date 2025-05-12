import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.time import Time, TimeDelta
from astropy.coordinates import EarthLocation, AltAz, GCRS, CartesianRepresentation, ITRS
from astrokit.bodies import Earth
from astrokit.twobody import Orbit
from astrokit.twobody.elements import ClassicalElements
from astrokit.twobody.propagation import CowellPropagator
from astrokit.core.perturbations import J2_perturbation
import pandas as pd
from enum import Enum, auto
import matplotlib.ticker as mticker
from numba import jit
# 定义轨道模型枚举类型
class OrbitModel(Enum):
    TWOBODY = auto()  # 二体模型
    J2 = auto()       # J2摄动模型
    J3 = auto()       # J3摄动模型

# 定义积分器类型
class IntegratorType(Enum):
    COWELL = auto()   # Cowell积分器

# J3摄动模型（补充J2模型）
def J3_perturbation(t, state, mu, J2, J3, R):
    """J3摄动加速度计算函数"""
    r = state[:3]
    r_norm = np.linalg.norm(r)
    z = r[2]
    
    # J2项
    factor_J2 = -3/2 * J2 * mu * R**2 / r_norm**5
    ax_J2 = factor_J2 * r[0] * (5 * z**2 / r_norm**2 - 1)
    ay_J2 = factor_J2 * r[1] * (5 * z**2 / r_norm**2 - 1)
    az_J2 = factor_J2 * z * (5 * z**2 / r_norm**2 - 3)
    
    # J3项
    factor_J3 = -5/2 * J3 * mu * R**3 / r_norm**7
    ax_J3 = factor_J3 * r[0] * z * (7 * z**2 / r_norm**2 - 3)
    ay_J3 = factor_J3 * r[1] * z * (7 * z**2 / r_norm**2 - 3)
    az_J3 = factor_J3 * (3 * z**2 * (7 * z**2 / r_norm**2 - 1) / r_norm**2 - 6 * z**2 / r_norm**2 + 1)
    
    return np.array([0, 0, 0, ax_J2 + ax_J3, ay_J2 + ay_J3, az_J2 + az_J3])
@jit(nopython=True)
def is_line_of_sight_blocked(station_pos, sat_pos, a=6378.137, f=1/298.257223563):
    """
    判断星站连线是否被地球椭球遮挡。
    station_pos, sat_pos: 地心坐标系中的位置向量 (km)
    a: 地球赤道半径 (km)
    f: 地球扁率
    返回: True 表示被遮挡（不可见），False 表示未被遮挡（可能可见）
    """
    b = a * (1 - f)  # 极半径
    # 星站连线方向向量
    d = sat_pos - station_pos
    # 起点（地面站位置）
    p = station_pos
    
    # 椭球方程系数 (x/a)^2 + (y/a)^2 + (z/b)^2 = 1
    aa = a * a
    bb = b * b
    
    # 二次方程系数：At^2 + Bt + C = 0
    A = (d[0]**2 + d[1]**2) / aa + d[2]**2 / bb
    B = 2 * (p[0]*d[0] + p[1]*d[1]) / aa + 2 * p[2]*d[2] / bb
    C = (p[0]**2 + p[1]**2) / aa + p[2]**2 / bb - 1
    
    # 判别式
    delta = B**2 - 4 * A * C
    
    if delta < 0:
        return False  # 无交点，可能可见
    
    # 求解 t
    t1 = (-B + np.sqrt(delta)) / (2 * A)
    t2 = (-B - np.sqrt(delta)) / (2 * A)
    
    # 检查交点是否在星站连线段内 (0 < t < 1)
    if (0 < t1 < 1) or (0 < t2 < 1):
        return True  # 有交点，被遮挡
    return False
class AdvancedVisibilityCalculator:
    """高级可视性计算器，支持多种轨道模型和积分器"""
    
    def __init__(self, min_elevation=10.0):
        """
        初始化可视性计算器
        
        参数:
        min_elevation : float
            最小仰角(度)，低于此仰角的卫星被认为不可见
        """
        self.min_elevation = min_elevation * u.deg
        
    def get_propagator(self, model_type, integrator_type):
        """
        获取适合指定轨道模型和积分器类型的传播器
        
        参数:
        model_type : OrbitModel
            轨道模型类型
        integrator_type : IntegratorType
            积分器类型
            
        返回:
        传播器对象
        """
        # 地球参数
        J2 = Earth.J2.value
        J3 = 0.00000162 # 地球J3系数
        R = Earth.R.to(u.km).value
        mu = Earth.k.value
        
        if model_type == OrbitModel.TWOBODY:
            # 二体模型不需要额外摄动
            perturbation_function = lambda t, state: np.zeros(6)
        elif model_type == OrbitModel.J2:
            # J2摄动模型
            perturbation_function = lambda t, state: J2_perturbation(t, state, mu, J2, R)
        elif model_type == OrbitModel.J3:
            # J3摄动模型(包含J2)
            perturbation_function = lambda t, state: J3_perturbation(t, state, mu, J2, J3, R)
        else:
            raise ValueError("不支持的轨道模型类型")
            
        if integrator_type == IntegratorType.COWELL:
            return CowellPropagator(f=perturbation_function)
        else:
            raise ValueError("不支持的积分器类型")
    @jit(nopython=True)
    def is_line_of_sight_blocked(station_pos, sat_pos, a=6378.137, f=1/298.257223563):
        """
        判断星站连线是否被地球椭球遮挡。
        station_pos, sat_pos: 地心坐标系中的位置向量 (km)
        a: 地球赤道半径 (km)
        f: 地球扁率
        返回: True 表示被遮挡（不可见），False 表示未被遮挡（可能可见）
        """
        b = a * (1 - f)  # 极半径
        # 星站连线方向向量
        d = sat_pos - station_pos
        # 起点（地面站位置）
        p = station_pos
        
        # 椭球方程系数 (x/a)^2 + (y/a)^2 + (z/b)^2 = 1
        aa = a * a
        bb = b * b
        
        # 二次方程系数：At^2 + Bt + C = 0
        A = (d[0]**2 + d[1]**2) / aa + d[2]**2 / bb
        B = 2 * (p[0]*d[0] + p[1]*d[1]) / aa + 2 * p[2]*d[2] / bb
        C = (p[0]**2 + p[1]**2) / aa + p[2]**2 / bb - 1
        
        # 判别式
        delta = B**2 - 4 * A * C
        
        if delta < 0:
            return False  # 无交点，可能可见
        
        # 求解 t
        t1 = (-B + np.sqrt(delta)) / (2 * A)
        t2 = (-B - np.sqrt(delta)) / (2 * A)
        
        # 检查交点是否在星站连线段内 (0 < t < 1)
        if (0 < t1 < 1) or (0 < t2 < 1):
            return True  # 有交点，被遮挡
        return False        
    def calculate_visibility_windows(self, satellites, ground_stations, start_time, end_time, step=30*u.s, model_type=OrbitModel.J2, integrator_type=IntegratorType.COWELL):
        propagator = self.get_propagator(model_type, integrator_type)
        time_range = np.arange(0, (end_time - start_time).to(u.s).value, step.to(u.s).value)
        times = start_time + time_range * u.s
        
        results = {}
        
        for station in ground_stations:
            station_name = station['name']
            results[station_name] = {}
            location = EarthLocation(lon=station['longitude'] * u.deg, lat=station['latitude'] * u.deg, height=station['altitude'] * u.m)
            
            for sat_name, sat_orbit in satellites.items():
                # 传播轨道并获取位置/速度向量
                sat_positions = []
                sat_velocities = []
                for dt in time_range * u.s:
                    propagated_orbit = sat_orbit.propagate(dt, propagator=propagator)
                    sat_positions.append(propagated_orbit.rv[0].value)
                    sat_velocities.append(propagated_orbit.rv[1].value)
                
                visibility_data = []
                frame = AltAz(obstime=times, location=location)
                
                for i, (t, r, v) in enumerate(zip(times, sat_positions, sat_velocities)):
                    # Get station position in ITRS
                    station_itrs = location.get_itrs(obstime=t)
                    
                    # Satellite position in ITRS
                    sat_cart = CartesianRepresentation(r[0], r[1], r[2], unit=u.km)
                    sat_gcrs = GCRS(sat_cart, obstime=t)
                    sat_itrs = sat_gcrs.transform_to(ITRS(obstime=t))
                    
                    # Relative position vector (station -> satellite)
                    rel_pos = sat_itrs.cartesian.xyz - station_itrs.cartesian.xyz
                    distance = np.linalg.norm(rel_pos) * u.km
                    
                    # 正确的仰角计算（使用astropy内置转换）
                    # 将卫星位置转换到地面站为中心的AltAz坐标系
                    sat_altaz = sat_gcrs.transform_to(AltAz(obstime=t, location=location))
                    elevation = sat_altaz.alt
                    azimuth = sat_altaz.az
                    distance = sat_altaz.distance
                    
                    # Calculate azimuth using astropy's built-in transformation
                    azimuth = rel_altaz.az
                    latitude, longitude = self._calculate_subsatellite_point(r * u.km)
                    velocity = np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)  # 手动计算速度大小
                    
                    is_visible = elevation > self.min_elevation
                    visibility_data.append({
                        'time': t,
                        'elevation': elevation,
                        'azimuth': azimuth,
                        'distance': distance,
                        'is_visible': is_visible,
                        'latitude': latitude,
                        'longitude': longitude,
                        'velocity': velocity
                    })
                
                windows = self._extract_visibility_windows(visibility_data)
                results[station_name][sat_name] = {'data': visibility_data, 'windows': windows}
        
        return results
    @jit(nopython=True)
    def calculate_subsatellite_point(x, y, z, a=6378.137, f=1/298.257223563):
        b = a * (1 - f)  # 极半径
        p = np.sqrt(x**2 + y**2)
        e = np.sqrt((a**2 - b**2) / a**2)
        lat = np.arctan2(z, p * (1 - e**2))
        
        for _ in range(5):
            N = a / np.sqrt(1 - e**2 * np.sin(lat)**2)
            h = p / np.cos(lat) - N
            lat = np.arctan2(z, p * (1 - e**2 * N / (N + h)))
        
        lon = np.arctan2(y, x)
        return lat * 180 / np.pi, lon * 180 / np.pi
    def _calculate_subsatellite_point(self, r_vec):
        x, y, z = r_vec.value
        lat, lon = calculate_subsatellite_point(x, y, z)
        return lat * u.deg, lon * u.deg
    
    def _extract_visibility_windows(self, visibility_data):
        """从可见性数据中提取连续的可见窗口"""
        windows = []
        in_window = False
        window_start = None
        
        for point in visibility_data:
            if point['is_visible'] and not in_window:
                # 开始一个新窗口
                in_window = True
                window_start = point
            elif not point['is_visible'] and in_window:
                # 结束当前窗口
                in_window = False
                windows.append({
                    'start_time': window_start['time'],
                    'end_time': point['time'],
                    'duration': (point['time'] - window_start['time']).to(u.minute),
                    'max_elevation': max([p['elevation'] for p in visibility_data
                                         if window_start['time'] <= p['time'] <= point['time']])
                })
        
        # 处理可能在结束时仍处于可见状态的情况
        if in_window:
            windows.append({
                'start_time': window_start['time'],
                'end_time': visibility_data[-1]['time'],
                'duration': (visibility_data[-1]['time'] - window_start['time']).to(u.minute),
                'max_elevation': max([p['elevation'] for p in visibility_data
                                     if window_start['time'] <= p['time'] <= visibility_data[-1]['time']])
            })
            
        return windows
    
    def plot_visibility(self, results, station_name, sat_name):
        """绘制可见性图表，仅在有可见窗口时绘图"""
        data = results[station_name][sat_name]['data']
        windows = results[station_name][sat_name]['windows']
        
        if not windows:
            print(f"{station_name} 对 {sat_name} 无可见窗口，不生成图表")
            return
        
        times = [(d['time'] - data[0]['time']).to(u.hour).value for d in data]
        elevations = [d['elevation'].to(u.deg).value for d in data]
        azimuths = [d['azimuth'].to(u.deg).value for d in data]
        distances = [d['distance'].to(u.km).value for d in data]
        is_visible = [d['is_visible'] for d in data]
        
        visible_times = [t if v else np.nan for t, v in zip(times, is_visible)]
        visible_elevations = [e if v else np.nan for e, v in zip(elevations, is_visible)]
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 15))
        
        axes[0].plot(times, elevations, 'b-', alpha=0.5, label='全部数据')
        axes[0].plot(visible_times, visible_elevations, 'r-', linewidth=2, label='可见段')
        axes[0].axhline(y=self.min_elevation.value, color='g', linestyle='--', label=f'最小仰角 ({self.min_elevation.value}°)')
        axes[0].set_xlabel('时间 (小时)')
        axes[0].set_ylabel('仰角 (度)')
        axes[0].set_title(f'{station_name} 观测 {sat_name} 的仰角变化')
        axes[0].grid(True)
        axes[0].legend()
        
        axes[1].plot(times, azimuths, 'b-')
        axes[1].set_xlabel('时间 (小时)')
        axes[1].set_ylabel('方位角 (度)')
        axes[1].set_title(f'{station_name} 观测 {sat_name} 的方位角变化')
        axes[1].grid(True)
        
        axes[2].plot(times, distances, 'g-')
        axes[2].set_xlabel('时间 (小时)')
        axes[2].set_ylabel('距离 (km)')
        axes[2].set_title(f'{station_name} 观测 {sat_name} 的距离变化')
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def plot_ground_track(self, results, sat_name, ground_stations, show_visibility=True):
        """绘制卫星地面轨迹"""
        # 提取星下点数据
        data = list(results.values())[0][sat_name]['data']  # 使用第一个地面站的数据
        
        lats = [d['latitude'].value for d in data]
        lons = [d['longitude'].value for d in data]
        is_visible = []
        
        if show_visibility:
            # 计算在任意地面站可见的点
            for i in range(len(data)):
                visible = False
                for station in results.values():
                    if station[sat_name]['data'][i]['is_visible']:
                        visible = True
                        break
                is_visible.append(visible)
        
        # 创建地图
        plt.figure(figsize=(12, 8))
        ax = plt.axes(projection='mollweide')
        
        # 转换经度从度到弧度
        lons_rad = np.array(lons) * np.pi / 180
        lats_rad = np.array(lats) * np.pi / 180
        
        # 绘制所有点
        ax.scatter(lons_rad, lats_rad, s=5, c='blue', label='地面轨迹')
        
        if show_visibility:
            # 绘制可见点
            visible_lons = [lon for lon, vis in zip(lons_rad, is_visible) if vis]
            visible_lats = [lat for lat, vis in zip(lats_rad, is_visible) if vis]
            ax.scatter(visible_lons, visible_lats, s=10, c='red', label='可见段')
            
            # 标记地面站位置
            for station_name, station_data in results.items():
                for gs in ground_stations:
                    if gs['name'] == station_name:
                        station_lon = gs['longitude'] * np.pi / 180
                        station_lat = gs['latitude'] * np.pi / 180
                        ax.scatter(station_lon, station_lat, s=100, c='green', marker='^', label=f'{station_name}')
                        break
        
        # 设置网格和标题
        ax.grid(True)
        ax.set_title(f'{sat_name} 地面轨迹')
        
        # 设置刻度标签为度
        lon_formatter = mticker.FuncFormatter(lambda x, pos: f'{x*180/np.pi:.0f}°')
        lat_formatter = mticker.FuncFormatter(lambda y, pos: f'{y*180/np.pi:.0f}°')
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    def print_visibility_report(self, results, model_type, integrator_type):
        """打印可见性报告"""
        print(f"\n轨道模型: {model_type.name}, 积分器: {integrator_type.name}")
        for station_name, satellites in results.items():
            print(f"\n地面站: {station_name}")
            for sat_name, data in satellites.items():
                print(f"\n  卫星: {sat_name}")
                if not data['windows']:
                    print("  在指定时间范围内无可见窗口")
                    continue
                    
                print(f"  可见窗口数量: {len(data['windows'])}")
                for i, window in enumerate(data['windows'], 1):
                    print(f"  窗口 {i}:")
                    print(f"    开始时间: {window['start_time'].iso}")
                    print(f"    结束时间: {window['end_time'].iso}")
                    print(f"    持续时间: {window['duration'].value:.2f} 分钟")
                    print(f"    最大仰角: {window['max_elevation'].value:.2f} 度")
                
    def compare_models(self, satellites, ground_stations, start_time, end_time, step=30*u.s):
        """比较不同轨道模型的可视窗口差异"""
        results = {}
        
        # 使用不同的轨道模型和积分器组合进行计算
        for model_type in OrbitModel:
            for integrator_type in IntegratorType:
                model_results = self.calculate_visibility_windows(
                    satellites, 
                    ground_stations,
                    start_time,
                    end_time,
                    step=step,
                    model_type=model_type,
                    integrator_type=integrator_type
                )
                
                # 存储结果
                key = (model_type, integrator_type)
                results[key] = model_results
                
                # 打印可视性报告
                self.print_visibility_report(model_results, model_type, integrator_type)
                
        return results

    def analyze_visibility_differences(self, comparison_results):
                # 打印可视性报告
            self.print_visibility_report(model_results, model_type, integrator_type)
                
            return results

    def analyze_visibility_differences(self, comparison_results):
        """分析不同轨道模型和积分器的可视窗口差异"""
        baseline_key = (OrbitModel.TWOBODY, IntegratorType.COWELL)  # 使用二体模型作为基准
        baseline = comparison_results[baseline_key]
        
        differences = {}
        
        # 对每个组合进行比较
        for key, results in comparison_results.items():
            if key == baseline_key:
                continue
                
            model_type, integrator_type = key
            model_name = f"{model_type.name}_{integrator_type.name}"
            differences[model_name] = {}
            
            # 对每个地面站进行比较
            for station_name in results:
                differences[model_name][station_name] = {}
                
                # 对每颗卫星进行比较
                for sat_name in results[station_name]:
                    base_windows = baseline[station_name][sat_name]['windows']
                    curr_windows = results[station_name][sat_name]['windows']
                    
                    # 计算差异
                    time_diffs = []
                    elev_diffs = []
                    
                    # 只有在窗口数量相同的情况下才比较
                    if len(base_windows) == len(curr_windows):
                        for i in range(len(base_windows)):
                            base_win = base_windows[i]
                            curr_win = curr_windows[i]
                            
                            # 计算开始时间差异（秒）
                            start_diff = (curr_win['start_time'] - base_win['start_time']).to(u.s).value
                            
                            # 计算结束时间差异（秒）
                            end_diff = (curr_win['end_time'] - base_win['end_time']).to(u.s).value
                            
                            # 计算持续时间差异（秒）
                            duration_diff = (curr_win['duration'] - base_win['duration']).to(u.s).value
                            
                            # 计算最大仰角差异（度）
                            max_elev_diff = (curr_win['max_elevation'] - base_win['max_elevation']).value
                            
                            time_diffs.append({
                                'window': i+1,
                                'start_diff': start_diff,
                                'end_diff': end_diff,
                                'duration_diff': duration_diff
                            })
                            
                            elev_diffs.append(max_elev_diff)
                    else:
                        # 窗口数量不同，记录差异
                        window_diff = len(curr_windows) - len(base_windows)
                        time_diffs = f"窗口数量差异: {window_diff}"
                        elev_diffs = "不适用"
                        
                    differences[model_name][station_name][sat_name] = {
                        'time_differences': time_diffs,
                        'elevation_differences': elev_diffs
                    }
        
        # 打印分析结果
        print("\n不同轨道模型和积分器的可视窗口差异分析:")
        for model_name, stations in differences.items():
            print(f"\n模型组合: {model_name} (与二体模型比较)")
            for station_name, satellites in stations.items():
                print(f"\n  地面站: {station_name}")
                for sat_name, diffs in satellites.items():
                    print(f"\n    卫星: {sat_name}")
                    
                    if isinstance(diffs['time_differences'], str):
                        print(f"    {diffs['time_differences']}")
                    else:
                        print("    时间差异(秒):")
                        for diff in diffs['time_differences']:
                            print(f"      窗口 {diff['window']}:")
                            print(f"        开始时间差异: {diff['start_diff']:.2f}秒")
                            print(f"        结束时间差异: {diff['end_diff']:.2f}秒")
                            print(f"        持续时间差异: {diff['duration_diff']:.2f}秒")
        
        return differences
def main():
    # 设置matplotlib支持中文
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong']  # 按优先级尝试多种中文字体
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.rcParams['font.size'] = 12  # 设置适当的字体大小
    
    # 检查字体是否正确加载
    from matplotlib.font_manager import FontProperties
    try:
        FontProperties(family='SimHei')
        print("中文字体加载成功")
    except:
        print("警告: 中文字体加载可能有问题，图表中文可能显示不正确")
    # 定义地面站
    ground_stations = [

    {'name': '北京站', 'longitude': 116.3, 'latitude': 39.9, 'altitude': 50.0},

    {'name': '上海站', 'longitude': 121.4, 'latitude': 31.2, 'altitude': 20.0},

    {'name': '西昌站', 'longitude': 102.0, 'latitude': 28.0, 'altitude': 1800.0}

    ]
    # 定义卫星轨道
    # Create ClassicalElements with Earth reference first
    leo_elements = ClassicalElements(
        Earth,
        a=6886.0 * u.km,
        ecc=0.000 * u.one,
        inc=55.6 * u.deg,
        raan=0 * u.deg,
        argp=0 * u.deg,
        nu=0 * u.deg
    )
    meo_elements = ClassicalElements(
        Earth,
        a=20000 * u.km,
        ecc=0.01 * u.one,
        inc=45.0 * u.deg,
        raan=120 * u.deg,
        argp=30 * u.deg,
        nu=0 * u.deg
    )
    geo_elements = ClassicalElements(
        Earth,
        a=42164 * u.km,
        ecc=0.0001 * u.one,
        inc=0.1 * u.deg,
        raan=75 * u.deg,
        argp=0 * u.deg,
        nu=0 * u.deg
    )

    satellites = {
        'LEO卫星': Orbit(Earth, leo_elements),
        'MEO卫星': Orbit(Earth, meo_elements),
        'GEO卫星': Orbit(Earth, geo_elements)
    }

    # 定义计算时间范围
    start_time = Time.now()
    end_time = start_time + 0.2 * u.day  # 计算未来一天的可见性

    # 创建高级可视性计算器
    calculator = AdvancedVisibilityCalculator(min_elevation=10.0)

    # ====== 测试单一模型 ======
    print("使用J2模型和Cowell积分器计算可见性窗口...")
    results = calculator.calculate_visibility_windows(
        satellites,
        ground_stations,
        start_time,
        end_time,
        step=1 * u.min,  # 1分钟步长
        model_type=OrbitModel.J2,
        integrator_type=IntegratorType.COWELL
    )

    # 打印报告
    calculator.print_visibility_report(results, OrbitModel.J2, IntegratorType.COWELL)

    # 绘制北京站观测LEO卫星的可见性图表
    calculator.plot_visibility(results, '北京站', 'LEO卫星')

    # 绘制卫星地面轨迹
    calculator.plot_ground_track(results, 'LEO卫星', ground_stations, show_visibility=True)

    # ====== 比较不同模型 ======
    print("\n\n比较不同轨道模型的可视窗口差异...")
    # 使用更小的步长，提高精度
    comparison_results = calculator.compare_models(
        satellites,
        ground_stations,
        start_time,
        start_time + 6 * u.hour,  # 缩短比较时间范围以加快计算
        step=30 * u.s
    )

    # 分析差异
    differences = calculator.analyze_visibility_differences(comparison_results)

    print("\n测试完成！")

if __name__ == '__main__':

    main()
