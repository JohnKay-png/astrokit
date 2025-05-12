import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.time import Time, TimeDelta
from astropy.coordinates import EarthLocation, AltAz, GCRS
from astrokit.bodies import Earth
from astrokit.twobody import Orbit
from astrokit.twobody.elements import ClassicalElements
import pandas as pd

class VisibilityCalculator:
    """计算地面站对卫星的可视性"""
    
    def __init__(self, min_elevation=10.0):
        """
        初始化可视性计算器
        
        参数:
            min_elevation : float
                最小仰角(度)，低于此仰角的卫星被认为不可见
        """
        self.min_elevation = min_elevation * u.deg
        
    def calculate_visibility_windows(self, satellites, ground_stations, 
                                    start_time, end_time, step=30*u.s):
        """
        计算一组地面站对一组卫星的可视窗口
        
        参数:
            satellites : list of Orbit
                卫星轨道对象列表
            ground_stations : list of dict
                地面站信息列表，每个站点为一个字典，包含名称、经度、纬度和高度
            start_time : Time
                开始时间
            end_time : Time
                结束时间
            step : Quantity
                时间步长
                
        返回:
            dict: 包含每个地面站对每颗卫星的可视窗口和参数
        """
        # 创建时间序列
        time_range = np.arange(0, (end_time - start_time).to(u.s).value, step.to(u.s).value)
        times = start_time + time_range * u.s
        
        results = {}
        
        # 遍历每个地面站
        for station in ground_stations:
            station_name = station['name']
            results[station_name] = {}
            
            # 创建地面站位置对象
            location = EarthLocation(
                lon=station['longitude'] * u.deg,
                lat=station['latitude'] * u.deg,
                height=station['altitude'] * u.m
            )
            
            # 遍历每颗卫星
            for sat_name, sat_orbit in satellites.items():
                # 存储该卫星的可见性数据
                visibility_data = []
                
                # 存储卫星位置
                sat_positions = []
                
                # 遍历每个时间点
                for t in times:
                    # 传播轨道并获取位置向量
                    propagated_orbit = sat_orbit.propagate(t - start_time)
                    r = propagated_orbit.rv[0]  # 获取位置向量
                    sat_positions.append(r)
                    
                    # 转换为地平坐标系(站心坐标系)
                    from astropy.coordinates import CartesianRepresentation
                    frame = AltAz(obstime=t, location=location)
                    cart_rep = CartesianRepresentation(r[0], r[1], r[2])
                    sat_altaz = GCRS(cart_rep, obstime=t).transform_to(frame)
                    
                    elevation = sat_altaz.alt
                    azimuth = sat_altaz.az
                    
                    # 计算站-星距离
                    distance = np.linalg.norm(r) - Earth.R
                    
                    # 判断可见性 (高于最小仰角)
                    is_visible = elevation > self.min_elevation
                    
                    # 存储当前时间点的可见性数据
                    visibility_data.append({
                        'time': t,
                        'elevation': elevation,
                        'azimuth': azimuth,
                        'distance': distance,
                        'is_visible': is_visible
                    })
                
                # 处理可见窗口
                windows = self._extract_visibility_windows(visibility_data)
                results[station_name][sat_name] = {
                    'data': visibility_data,
                    'windows': windows
                }
                
        return results
    
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
        """绘制可见性图表"""
        data = results[station_name][sat_name]['data']
        
        # 提取数据
        times = [(d['time'] - data[0]['time']).to(u.hour).value for d in data]
        elevations = [d['elevation'].to(u.deg).value for d in data]
        azimuths = [d['azimuth'].to(u.deg).value for d in data]
        is_visible = [d['is_visible'] for d in data]
        
        # 创建可见性掩码
        visible_times = [t if v else np.nan for t, v in zip(times, is_visible)]
        visible_elevations = [e if v else np.nan for e, v in zip(elevations, is_visible)]
        
        # 绘图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 仰角随时间变化
        ax1.plot(times, elevations, 'b-', alpha=0.5, label='全部数据')
        ax1.plot(visible_times, visible_elevations, 'r-', linewidth=2, label='可见段')
        ax1.axhline(y=self.min_elevation.value, color='g', linestyle='--', label=f'最小仰角 ({self.min_elevation.value}°)')
        ax1.set_xlabel('时间 (小时)')
        ax1.set_ylabel('仰角 (度)')
        ax1.set_title(f'{station_name} 观测 {sat_name} 的仰角变化')
        ax1.grid(True)
        ax1.legend()
        
        # 方位角随时间变化
        ax2.plot(times, azimuths, 'b-')
        ax2.set_xlabel('时间 (小时)')
        ax2.set_ylabel('方位角 (度)')
        ax2.set_title(f'{station_name} 观测 {sat_name} 的方位角变化')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def print_visibility_report(self, results):
        """打印可见性报告"""
        for station_name, satellites in results.items():
            print(f"\n地面站: {station_name}")
            for sat_name, data in satellites.items():
                print(f"\n  卫星: {sat_name}")
                if not data['windows']:
                    print("    在指定时间范围内无可见窗口")
                    continue
                    
                print(f"    可见窗口数量: {len(data['windows'])}")
                for i, window in enumerate(data['windows'], 1):
                    print(f"    窗口 {i}:")
                    print(f"      开始时间: {window['start_time'].iso}")
                    print(f"      结束时间: {window['end_time'].iso}")
                    print(f"      持续时间: {window['duration'].value:.2f} 分钟")
                    print(f"      最大仰角: {window['max_elevation'].value:.2f} 度")

# 使用示例
def main():
    # 定义地面站
    ground_stations = [
        {'name': '北京站', 'longitude': 116.3, 'latitude': 39.9, 'altitude': 50.0},
        {'name': '上海站', 'longitude': 121.4, 'latitude': 31.2, 'altitude': 20.0}
    ]
    
    # 定义卫星轨道
    # 创建一颗LEO卫星和一颗GEO卫星
    satellites = {
        'LEO卫星': Orbit(
            Earth,
            ClassicalElements(
                a=7000 * u.km,
                ecc=0.001 * u.one,
                inc=51.6 * u.deg,
                raan=20 * u.deg,
                argp=0 * u.deg,
                nu=0 * u.deg
            )
        ),
        'GEO卫星': Orbit(
            Earth,
            ClassicalElements(
                a=42164 * u.km,
                ecc=0.0001 * u.one,
                inc=0.1 * u.deg,
                raan=75 * u.deg,
                argp=0 * u.deg,
                nu=0 * u.deg
            )
        )
    }
    
    # 定义计算时间范围
    start_time = Time.now()
    end_time = start_time + 1 * u.day  # 计算未来一天的可见性
    
    # 创建可见性计算器
    calculator = VisibilityCalculator(min_elevation=15.0)  # 最小仰角15度
    
    # 计算可见性
    results = calculator.calculate_visibility_windows(
        satellites,
        ground_stations,
        start_time,
        end_time,
        step=1 * u.min  # 1分钟步长
    )
    
    # 打印报告
    calculator.print_visibility_report(results)
    
    # 绘制北京站观测LEO卫星的可见性图表
    calculator.plot_visibility(results, '北京站', 'LEO卫星')

if __name__ == "__main__":
    main()
