import numpy as np
import random
import math
from scipy.stats import truncnorm
from enum import Enum

# 行星类型枚举
class PlanetType(Enum):
    TERRESTRIAL = "Terrestrial"  # 类地行星
    SUPER_EARTH = "Super-Earth"  # 超级地球
    SUB_NEPTUNE = "Sub-Neptune"  # 亚海王星
    ICE_GIANT = "Ice Giant"     # 冰巨星
    GAS_GIANT = "Gas Giant"     # 气态巨行星
    MINI_NEPTUNE = "Mini-Neptune"  # 迷你海王星
    OCEAN_WORLD = "Ocean World"  # 海洋世界
    LAVA_WORLD = "Lava World"   # 熔岩世界
    HELIUM_PLANET = "Helium Planet"  # 氦行星
    CHTHONIAN = "Chthonian"     # 克托尼行星
    CARBON_PLANET = "Carbon Planet"  # 碳行星
    IRON_PLANET = "Iron Planet"  # 铁行星
    DESERT_PLANET = "Desert Planet"  # 沙漠行星
    SNOWBALL_PLANET = "Snowball Planet"  # 雪球行星

# 恒星光谱类型
class StarType(Enum):
    O = "O"  # 蓝巨星
    B = "B"  # 蓝白星
    A = "A"  # 白星
    F = "F"  # 黄白星
    G = "G"  # 黄矮星 (类太阳)
    K = "K"  # 橙矮星
    M = "M"  # 红矮星
    WHITE_DWARF = "White Dwarf"  # 白矮星
    NEUTRON_STAR = "Neutron Star"  # 中子星
    BLACK_HOLE = "Black Hole"  # 黑洞

# 大气成分类
class AtmosphereComposition:
    def __init__(self):
        self.components = {}
        
    def add_component(self, gas, percentage):
        """添加大气成分"""
        self.components[gas] = percentage
        
    def normalize(self):
        """标准化大气成分百分比"""
        total = sum(self.components.values())
        for gas in self.components:
            self.components[gas] = round(self.components[gas] / total * 100, 2)
            
    def __str__(self):
        return ", ".join([f"{gas}: {perc}%" for gas, perc in self.components.items()])

# 恒星生成函数
def generate_star(weights=None, specified_type=None):
    """
    生成一颗恒星，包含详细的物理参数
    可以指定恒星光谱类型，包括O,B,A,F,G,K,M,白矮星,中子星,黑洞
    
    参数:
    weights (dict): 恒星类型的权重因子
    specified_type (str or StarType): 指定的恒星类型，如'O'、'G'、'White Dwarf'等
    """
    if weights is None:
        weights = {}
    
    # 所有可能的恒星类型
    star_types = [
        StarType.O, StarType.B, StarType.A, StarType.F, 
        StarType.G, StarType.K, StarType.M,
        StarType.WHITE_DWARF, StarType.NEUTRON_STAR, StarType.BLACK_HOLE
    ]
    
    # 基础概率 (基于宇宙丰度)
    base_probs = {
        StarType.O: 0.00003,
        StarType.B: 0.13,
        StarType.A: 0.6,
        StarType.F: 3.0,
        StarType.G: 7.6,
        StarType.K: 12.1,
        StarType.M: 76.5,
        StarType.WHITE_DWARF: 8.0,
        StarType.NEUTRON_STAR: 0.7,
        StarType.BLACK_HOLE: 0.1
    }
    
    # 如果指定了恒星类型
    if specified_type is not None:
        # 确保指定类型是有效的
        if isinstance(specified_type, str):
            # 尝试匹配已知类型
            found = False
            for st in star_types:
                if specified_type.lower() == st.value.lower() or specified_type.lower() == st.name.lower():
                    star_type = st
                    found = True
                    break
            
            if not found:
                # 尝试简化匹配
                if specified_type.lower() in ['o', 'b', 'a', 'f', 'g', 'k', 'm']:
                    star_type = StarType[specified_type.upper()]
                elif specified_type.lower() in ['wd', 'white dwarf']:
                    star_type = StarType.WHITE_DWARF
                elif specified_type.lower() in ['ns', 'neutron star', 'pulsar']:
                    star_type = StarType.NEUTRON_STAR
                elif specified_type.lower() in ['bh', 'black hole']:
                    star_type = StarType.BLACK_HOLE
                else:
                    raise ValueError(f"未知的恒星类型: {specified_type}")
        elif isinstance(specified_type, StarType):
            star_type = specified_type
        else:
            raise TypeError("指定的恒星类型必须是字符串或StarType枚举")
        
        # 直接使用指定类型
        selected_type = star_type
    else:
        # 应用权重
        total_prob = sum(base_probs.values())
        weighted_probs = []
        type_list = []
        
        for st in star_types:
            # 获取基础概率
            base_prob = base_probs.get(st, 0.00001)  # 默认极小值
            
            # 应用权重
            weight = weights.get(st.value, 1.0)
            if st in weights:
                weight = weights[st]
            elif st.value in weights:
                weight = weights[st.value]
            
            weighted_prob = base_prob * weight
            weighted_probs.append(weighted_prob)
            type_list.append(st)
        
        # 归一化
        total_weighted = sum(weighted_probs)
        if total_weighted <= 0:
            raise ValueError("权重设置导致所有概率为零")
        
        norm_probs = [p / total_weighted for p in weighted_probs]
        
        # 随机选择恒星类型
        selected_type = np.random.choice(type_list, p=norm_probs)
    
    # 恒星物理参数范围 (质量单位: 太阳质量)
    star_params = {
        StarType.O: {
            'mass_range': (16, 150), 
            'temp_range': (30000, 50000), 
            'luminosity_range': (3e4, 1e6), 
            'radius_range': (6.6, 20),
            'lifetime_range': (1e6, 1e7), 
            'color': 'blue'
        },
        StarType.B: {
            'mass_range': (2.1, 16), 
            'temp_range': (10000, 30000), 
            'luminosity_range': (25, 3e4), 
            'radius_range': (1.8, 6.6),
            'lifetime_range': (1e7, 1e8), 
            'color': 'blue-white'
        },
        StarType.A: {
            'mass_range': (1.4, 2.1), 
            'temp_range': (7500, 10000), 
            'luminosity_range': (5, 25), 
            'radius_range': (1.4, 1.8),
            'lifetime_range': (5e8, 1e9), 
            'color': 'white'
        },
        StarType.F: {
            'mass_range': (1.04, 1.4), 
            'temp_range': (6000, 7500), 
            'luminosity_range': (1.5, 5), 
            'radius_range': (1.15, 1.4),
            'lifetime_range': (2e9, 5e9), 
            'color': 'yellow-white'
        },
        StarType.G: {
            'mass_range': (0.8, 1.04), 
            'temp_range': (5200, 6000), 
            'luminosity_range': (0.6, 1.5), 
            'radius_range': (0.96, 1.15),
            'lifetime_range': (7e9, 15e9), 
            'color': 'yellow'
        },
        StarType.K: {
            'mass_range': (0.45, 0.8), 
            'temp_range': (3700, 5200), 
            'luminosity_range': (0.08, 0.6), 
            'radius_range': (0.7, 0.96),
            'lifetime_range': (15e9, 30e9), 
            'color': 'orange'
        },
        StarType.M: {
            'mass_range': (0.08, 0.45), 
            'temp_range': (2400, 3700), 
            'luminosity_range': (0.0001, 0.08), 
            'radius_range': (0.1, 0.7),
            'lifetime_range': (50e9, 10e12), 
            'color': 'red'
        },
        StarType.WHITE_DWARF: {
            'mass_range': (0.4, 1.4),  # 钱德拉塞卡极限以下
            'temp_range': (5000, 150000),  # 白矮星温度范围广
            'luminosity_range': (0.0001, 0.1),  # 低光度
            'radius_range': (0.008, 0.02),  # 地球大小
            'lifetime_range': (10e9, 100e9),  # 极长寿命
            'color': 'white'
        },
        StarType.NEUTRON_STAR: {
            'mass_range': (1.4, 2.16),  # 奥本海默极限以下
            'temp_range': (1e5, 1e6),  # 极高温度
            'luminosity_range': (0.1, 100),  # 可能高光度
            'radius_range': (10, 15),  # 公里级，转换为太阳半径单位
            'lifetime_range': (1e6, 1e10),  # 长寿命
            'color': 'blue-white'
        },
        StarType.BLACK_HOLE: {
            'mass_range': (3, 50),  # 恒星黑洞
            'temp_range': (0, 1e-5),  # 极低温度（霍金辐射）
            'luminosity_range': (0, 0),  # 无光度
            'radius_range': (0, 0),  # 事件视界半径
            'lifetime_range': (1e30, 1e100),  # 几乎无限寿命
            'color': 'black'
        }
    }
    
    params = star_params[selected_type]
    
    # 生成物理参数 - 使用截断正态分布模拟真实分布
    def truncated_normal(mean, std, low, high):
        a, b = (low - mean) / std, (high - mean) / std
        return truncnorm.rvs(a, b, loc=mean, scale=std)
    
    # 质量 - 遵循Salpeter初始质量函数
    mass_mean = np.mean(params['mass_range'])
    mass_std = (params['mass_range'][1] - params['mass_range'][0]) / 4
    mass = round(truncated_normal(mass_mean, mass_std, *params['mass_range']), 4)
    
    # 特殊恒星类型的调整
    if selected_type == StarType.WHITE_DWARF:
        # 白矮星质量-半径关系 (近似)
        radius = round(0.01 * (1.44 / mass) ** (1/3), 5)  # 太阳半径
        # 温度独立于质量
        temperature = random.uniform(*params['temp_range'])
        # 光度与温度相关
        luminosity = round(0.01 * (temperature/5778)**4 * (radius)**2, 6)
    
    elif selected_type == StarType.NEUTRON_STAR:
        # 中子星半径几乎恒定 (约10公里)
        radius = round(0.000014, 6)  # 10公里 / 太阳半径(695,700km) ≈ 1.4e-5
        # 温度独立于质量
        temperature = random.uniform(*params['temp_range'])
        # 光度计算
        luminosity = round(4 * math.pi * (radius * 6.957e8)**2 * 5.67e-8 * temperature**4 / 3.828e26, 6)
    
    elif selected_type == StarType.BLACK_HOLE:
        # 黑洞没有传统意义上的半径和光度
        radius = 0
        temperature = 0
        luminosity = 0
        # 史瓦西半径 (单位: 公里)
        schwarzschild_radius = 2.95 * mass  # 太阳质量 * 2.95公里/太阳质量
    
    else:
        # 主序星的质量-光度关系
        if mass > 0.43:
            luminosity = round(1.0 * (mass ** 3.5), 4)  # 以太阳光度为单位
        else:  # 低质量恒星
            luminosity = round(0.23 * (mass ** 2.3), 6)
        
        # 温度 - 使用质量-温度关系
        # 近似公式: T ≈ 5778 * M^0.5 (适用于G型星附近的简化)
        temperature = round(5778 * (mass ** 0.5), 1)
        
        # 半径 - 使用质量-半径关系
        # 近似公式: R ≈ R☉ * M^0.8 (适用于主序星)
        radius = round(1.0 * (mass ** 0.8), 4)
    
    # 年龄 - 基于恒星类型
    if selected_type in [StarType.O, StarType.B, StarType.A]:
        age = round(random.uniform(0.001, 0.1), 4)  # 年轻恒星 (百万年)
    elif selected_type in [StarType.F, StarType.G]:
        age = round(random.uniform(0.5, 10), 2)  # 中年恒星 (十亿年)
    elif selected_type in [StarType.K, StarType.M]:
        age = round(random.uniform(1, 13.8), 2)  # 古老恒星 (十亿年)
    else:  # 致密天体
        age = round(random.uniform(1, 13.8), 2)  # 古老
    
    # 金属丰度 [Fe/H]
    if selected_type in [StarType.O, StarType.B, StarType.A]:
        metallicity = round(random.uniform(-0.2, 0.3), 3)
    elif selected_type in [StarType.F, StarType.G]:
        metallicity = round(random.uniform(-0.5, 0.4), 3)
    elif selected_type in [StarType.K, StarType.M]:
        metallicity = round(random.uniform(-1.0, 0.2), 3)
    else:  # 致密天体
        metallicity = round(random.uniform(-0.5, 0.5), 3)
    
    # 恒星演化阶段
    # 对于特殊类型，直接指定
    if selected_type == StarType.WHITE_DWARF:
        evolution = "White Dwarf"
    elif selected_type == StarType.NEUTRON_STAR:
        evolution = "Neutron Star"
    elif selected_type == StarType.BLACK_HOLE:
        evolution = "Black Hole"
    else:
        # 主序星的演化阶段
        lifetime = 10 * (mass ** -2.5)  # 恒星寿命 (十亿年)
        age_fraction = age / lifetime
        
        if age_fraction < 0.1:
            evolution = "Pre-main sequence"
        elif age_fraction < 0.9:
            evolution = "Main sequence"
        elif age_fraction < 0.95:
            evolution = "Subgiant"
        elif mass < 8:  # 低质量恒星演化
            if age_fraction < 0.99:
                evolution = "Giant"
            else:
                evolution = "White Dwarf"
        else:  # 大质量恒星演化
            if random.random() < 0.7:
                evolution = "Neutron Star"
            else:
                evolution = "Black Hole"
    
    # 磁场强度 (高斯)
    if selected_type in [StarType.M, StarType.K]:
        magnetic_field = round(10 ** random.uniform(2, 4), 1)  # 红矮星有强磁场
    elif selected_type == StarType.NEUTRON_STAR:
        magnetic_field = round(10 ** random.uniform(10, 15), 1)  # 中子星超强磁场
    elif selected_type == StarType.WHITE_DWARF:
        magnetic_field = round(10 ** random.uniform(6, 9), 1)  # 白矮星强磁场
    else:
        magnetic_field = round(10 ** random.uniform(0, 2), 1)
    
    # 自转周期 (天)
    if selected_type in [StarType.O, StarType.B]:
        rotation_period = round(random.uniform(0.5, 3), 2)
    elif selected_type == StarType.NEUTRON_STAR:
        rotation_period = round(random.uniform(0.001, 10), 6)  # 毫秒到秒级
    else:
        rotation_period = round(random.uniform(10, 40), 1)
    
    # 安全计算平方根函数
    def safe_sqrt(value):
        if value <= 0:
            return 0
        return math.sqrt(value)
    
    # 计算宜居带 (仅适用于主序星)
    if selected_type in [StarType.O, StarType.B, StarType.A, StarType.F, StarType.G, StarType.K, StarType.M]:
        # 保守宜居带: Kopparapu et al. (2013)
        T_eff = temperature
        S_eff_sun = 1.0  # 太阳常数
        
        # 内边界 (失控温室效应)
        inner_coeff = max(0.0001, 4.190e-8 * T_eff**2 - 2.139e-4 * T_eff + 1.268)
        a_inner = safe_sqrt(luminosity / (S_eff_sun * inner_coeff))
        
        # 外边界 (最大温室效应)
        outer_coeff = max(0.0001, 6.190e-9 * T_eff**2 - 1.319e-4 * T_eff + 0.2341)
        a_outer = safe_sqrt(luminosity / (S_eff_sun * outer_coeff))
        
        # 雪线位置 (水冰线)
        snow_line = 2.7 * math.sqrt(luminosity)
    else:
        # 致密天体没有宜居带
        a_inner = 0
        a_outer = 0
        snow_line = 0
    
    # 黑洞的特殊属性
    if selected_type == StarType.BLACK_HOLE:
        schwarzschild_radius = round(2.95 * mass, 2)  # 公里
    else:
        schwarzschild_radius = 0
    
    return {
        'spectral_type': selected_type.value,
        'mass': mass,  # 太阳质量
        'luminosity': luminosity,  # 太阳光度
        'effective_temperature': temperature,  # K
        'radius': radius,  # 太阳半径
        'age': age,  # 十亿年
        'metallicity': metallicity,  # [Fe/H]
        'evolution_stage': evolution,
        'magnetic_field': magnetic_field,  # 高斯
        'rotation_period': rotation_period,  # 天 (中子星为秒)
        'habitable_zone_inner': round(a_inner, 3),  # AU
        'habitable_zone_outer': round(a_outer, 3),  # AU
        'snow_line': round(snow_line, 3),  # AU
        'color': params['color'],
        'schwarzschild_radius': schwarzschild_radius  # 黑洞的史瓦西半径 (公里)
    }

# 行星生成函数
def generate_planet(star, position_index, total_planets, weights=None):
    """
    生成一颗行星，包含详细的物理和轨道参数
    """
    if weights is None:
        weights = {}
    
    # 确定行星位置类型
    if position_index == 0:
        position_type = "Inner"
    elif position_index == total_planets - 1:
        position_type = "Outer"
    else:
        position_type = "Middle"
    
    # 轨道半长轴 (AU) - 遵循指数分布
    # 内边界: 潮汐撕裂半径 R_tidal = 2.9 * R_star * (M_star/M_planet)^{1/3}
    # 使用简化内边界: 0.02 AU
    inner_bound = 0.02
    outer_bound = 1000
    
    # 基于位置索引的对数分布
    log_min = math.log10(inner_bound)
    log_max = math.log10(outer_bound)
    log_step = (log_max - log_min) / (total_planets + 1)
    log_sma = log_min + (position_index + 1) * log_step + random.uniform(-0.2, 0.2) * log_step
    sma = round(10 ** log_sma, 5)
    
    # 偏心率 - 遵循Rayleigh分布
    # 近距离行星偏心率较小，远距离较大
    if sma < 0.1:
        eccentricity = round(np.random.rayleigh(0.01), 4)
    elif sma < 1.0:
        eccentricity = round(np.random.rayleigh(0.05), 4)
    elif sma < 10.0:
        eccentricity = round(np.random.rayleigh(0.1), 4)
    else:
        eccentricity = round(np.random.rayleigh(0.2), 4)
    
    # 倾角 - 遵循Fisher分布 (球面均匀分布)
    inclination = round(np.degrees(np.arccos(1 - 2 * random.random())), 2)  # 0-180度
    
    # 轨道参数
    ascending_node = round(random.uniform(0, 360), 2)  # 升交点黄经 (度)
    arg_periapsis = round(random.uniform(0, 360), 2)   # 近心点幅角 (度)
    mean_anomaly = round(random.uniform(0, 360), 2)    # 平近点角 (度)
    
    # 确定行星是否在宜居带
    in_habitable_zone = (sma >= star['habitable_zone_inner'] and 
                         sma <= star['habitable_zone_outer'])
    
    # 生成行星类型 (考虑位置、恒星参数和权重)
    planet_type = generate_planet_type(sma, star, weights, in_habitable_zone)
    
    # 行星物理参数
    planet_props = generate_planet_properties(planet_type, sma, star)
    
    # 平衡温度 (考虑反照率和温室效应)
    albedo = planet_props['albedo']
    # 基本公式: T_eq = T_star * (R_star / (2 * a))^{0.5} * (1 - A)^{0.25}
    T_star = star['effective_temperature']
    R_star = star['radius'] * 6.957e8  # 太阳半径转米
    a_m = sma * 1.496e11  # AU转米
    T_eq = T_star * math.sqrt(R_star / (2 * a_m)) * ((1 - albedo) ** 0.25)
    # 应用温室效应修正
    if isinstance(planet_props['atmosphere_pressure'], (int, float)) and planet_props['atmosphere_pressure'] > 0.1:
        greenhouse_factor = 1.0 + 0.1 * math.log10(planet_props['atmosphere_pressure'] + 1)
        T_surface = T_eq * greenhouse_factor
    else:
        T_surface = T_eq
    
    # 生成卫星系统
    moons = generate_moon_system(planet_props['mass'], sma, star)
    
    # 环系统
    rings = generate_ring_system(planet_props['mass'], sma, star)
    
    # 轴向倾斜 (自转轴倾角)
    obliquity = round(random.uniform(0, 180), 1)  # 度
    
    # 自转周期
    if planet_props['mass'] < 0.1:  # 小质量天体
        rotation_period = round(random.uniform(0.1, 100), 2)
    else:
        # 潮汐锁定概率 (靠近恒星的小行星)
        if sma < 0.1 and planet_props['mass'] < 10:
            rotation_period = round(2 * math.pi * math.sqrt(sma**3 / star['mass']), 3)  # 同步自转
        else:
            rotation_period = round(random.uniform(0.2, 100), 2)
    
    return {
        'type': planet_type.value,
        'mass': planet_props['mass'],  # 地球质量
        'radius': planet_props['radius'],  # 地球半径
        'density': planet_props['density'],  # g/cm³
        'surface_gravity': planet_props['surface_gravity'],  # m/s²
        'albedo': albedo,
        'semi_major_axis': sma,  # AU
        'eccentricity': eccentricity,
        'inclination': inclination,  # 度
        'ascending_node': ascending_node,  # 度
        'arg_periapsis': arg_periapsis,  # 度
        'mean_anomaly': mean_anomaly,  # 度
        'orbital_period': round(math.sqrt(sma**3 / star['mass']), 5),  # 年
        'in_habitable_zone': in_habitable_zone,
        'equilibrium_temperature': round(T_eq, 1),  # K
        'surface_temperature': round(T_surface, 1),  # K
        'atmosphere_pressure': planet_props['atmosphere_pressure'],  # bar
        'atmosphere_composition': str(planet_props['atmosphere']),
        'surface_composition': planet_props['surface_composition'],
        'tectonic_activity': planet_props['tectonic_activity'],
        'magnetic_field': planet_props['magnetic_field'],  # 高斯
        'rotation_period': rotation_period,  # 天
        'obliquity': obliquity,  # 度
        'moons': moons,
        'rings': rings,
        'position_type': position_type
    }

def generate_planet_type(sma, star, weights, in_habitable_zone):
    """
    基于详细规则生成行星类型
    """
    # 行星类型概率分布 (基于开普勒数据)
    type_probs = {
        PlanetType.TERRESTRIAL: 0.05,
        PlanetType.SUPER_EARTH: 0.20,
        PlanetType.SUB_NEPTUNE: 0.35,
        PlanetType.MINI_NEPTUNE: 0.15,
        PlanetType.ICE_GIANT: 0.10,
        PlanetType.GAS_GIANT: 0.05,
        PlanetType.OCEAN_WORLD: 0.03,
        PlanetType.LAVA_WORLD: 0.02,
        PlanetType.HELIUM_PLANET: 0.01,
        PlanetType.CHTHONIAN: 0.01,
        PlanetType.CARBON_PLANET: 0.01,
        PlanetType.IRON_PLANET: 0.01,
        PlanetType.DESERT_PLANET: 0.005,
        PlanetType.SNOWBALL_PLANET: 0.005
    }
    
    # 根据轨道距离调整概率
    snow_line = star['snow_line']
    
    if sma < 0.05:  # 极近轨道
        type_probs[PlanetType.LAVA_WORLD] *= 5
        type_probs[PlanetType.CHTHONIAN] *= 3
        type_probs[PlanetType.IRON_PLANET] *= 2
        type_probs[PlanetType.SUB_NEPTUNE] *= 0.2
        type_probs[PlanetType.ICE_GIANT] = 0
    
    elif sma < star['habitable_zone_inner']:  # 内系统
        type_probs[PlanetType.TERRESTRIAL] *= 2
        type_probs[PlanetType.SUPER_EARTH] *= 1.5
        type_probs[PlanetType.LAVA_WORLD] *= 2
        type_probs[PlanetType.IRON_PLANET] *= 1.5
        type_probs[PlanetType.SUB_NEPTUNE] *= 0.8
        type_probs[PlanetType.ICE_GIANT] = 0
    
    elif in_habitable_zone:  # 宜居带
        type_probs[PlanetType.TERRESTRIAL] *= 4
        type_probs[PlanetType.OCEAN_WORLD] *= 5
        type_probs[PlanetType.SUPER_EARTH] *= 2
        type_probs[PlanetType.DESERT_PLANET] *= 3
        type_probs[PlanetType.SNOWBALL_PLANET] *= 2
        type_probs[PlanetType.SUB_NEPTUNE] *= 0.5
        type_probs[PlanetType.ICE_GIANT] = 0
    
    elif sma < snow_line:  # 雪线内
        type_probs[PlanetType.SUB_NEPTUNE] *= 1.5
        type_probs[PlanetType.MINI_NEPTUNE] *= 1.5
        type_probs[PlanetType.SUPER_EARTH] *= 1.2
        type_probs[PlanetType.GAS_GIANT] *= 0.5
    
    elif sma < 5 * snow_line:  # 冰巨行星带
        type_probs[PlanetType.ICE_GIANT] *= 3
        type_probs[PlanetType.GAS_GIANT] *= 1.5
        type_probs[PlanetType.SUB_NEPTUNE] *= 0.5
        type_probs[PlanetType.OCEAN_WORLD] *= 0.1
    
    else:  # 远外系统
        type_probs[PlanetType.ICE_GIANT] *= 2
        type_probs[PlanetType.GAS_GIANT] *= 1.2
        type_probs[PlanetType.SNOWBALL_PLANET] *= 3
        type_probs[PlanetType.HELIUM_PLANET] *= 2
    
    # 应用用户权重
    for ptype in type_probs:
        if ptype in weights:
            type_probs[ptype] *= weights[ptype]
    
    # 归一化概率
    total = sum(type_probs.values())
    norm_probs = [p / total for p in type_probs.values()]
    
    # 随机选择行星类型
    return np.random.choice(list(type_probs.keys()), p=norm_probs)

def generate_planet_properties(planet_type, sma, star):
    """
    生成行星的详细物理属性
    """
    # 质量-半径关系参数
    # 使用分段幂律关系
    if planet_type == PlanetType.TERRESTRIAL:
        # 岩石行星: R ∝ M^0.27 (0.1-1 M⊕)
        mass = round(10 ** random.uniform(-1, 0), 3)  # 0.1-1 M⊕
        radius = round(1.0 * (mass ** 0.27), 3)
        density = round(random.uniform(3.5, 5.5), 2)
        albedo = random.uniform(0.1, 0.3)
        atmosphere = AtmosphereComposition()
        atmosphere.add_component("CO2", random.uniform(50, 95))
        atmosphere.add_component("N2", random.uniform(5, 40))
        atmosphere.add_component("Ar", random.uniform(0.1, 2))
        atmosphere.normalize()
        surface_composition = "硅酸盐岩石"
        tectonic_activity = random.choice(["Low", "Moderate", "High"])
    
    elif planet_type == PlanetType.SUPER_EARTH:
        # 超级地球: R ∝ M^0.5 (1-10 M⊕)
        mass = round(10 ** random.uniform(0, 1), 3)  # 1-10 M⊕
        radius = round(1.0 * (mass ** 0.5), 3)
        density = round(random.uniform(3.0, 5.0), 2)
        albedo = random.uniform(0.2, 0.4)
        atmosphere = AtmosphereComposition()
        if random.random() < 0.7:  # 大部分有厚大气
            atmosphere.add_component("H2", random.uniform(20, 70))
            atmosphere.add_component("He", random.uniform(10, 30))
            atmosphere.add_component("H2O", random.uniform(5, 20))
        else:
            atmosphere.add_component("CO2", random.uniform(60, 90))
            atmosphere.add_component("N2", random.uniform(10, 30))
        atmosphere.normalize()
        surface_composition = "硅酸盐岩石，可能含水层"
        tectonic_activity = random.choice(["Moderate", "High"])
    
    elif planet_type == PlanetType.SUB_NEPTUNE:
        # 亚海王星: R ∝ M^0.18 (1-10 M⊕)
        mass = round(10 ** random.uniform(0, 1), 3)  # 1-10 M⊕
        radius = round(1.0 * (mass ** 0.18), 3)
        density = round(random.uniform(1.0, 3.0), 2)
        albedo = random.uniform(0.3, 0.5)
        atmosphere = AtmosphereComposition()
        atmosphere.add_component("H2", random.uniform(70, 90))
        atmosphere.add_component("He", random.uniform(10, 25))
        atmosphere.add_component("CH4", random.uniform(0.1, 3))
        atmosphere.normalize()
        surface_composition = "无固体表面，深水层或超临界流体"
        tectonic_activity = "None"
    
    elif planet_type == PlanetType.MINI_NEPTUNE:
        # 迷你海王星: R ∝ M^0.12 (2-10 M⊕)
        mass = round(10 ** random.uniform(0.3, 1), 3)  # 2-10 M⊕
        radius = round(1.5 * (mass ** 0.12), 3)
        density = round(random.uniform(0.5, 1.5), 2)
        albedo = random.uniform(0.4, 0.6)
        atmosphere = AtmosphereComposition()
        atmosphere.add_component("H2", random.uniform(80, 95))
        atmosphere.add_component("He", random.uniform(5, 15))
        atmosphere.add_component("H2O", random.uniform(1, 5))
        atmosphere.normalize()
        surface_composition = "无固体表面，深水层或超临界流体"
        tectonic_activity = "None"
    
    elif planet_type == PlanetType.ICE_GIANT:
        # 冰巨星: R ∝ M^0.59 (10-50 M⊕)
        mass = round(10 ** random.uniform(1, 1.7), 3)  # 10-50 M⊕
        radius = round(3.5 * (mass ** 0.59), 3)
        density = round(random.uniform(1.0, 1.7), 2)
        albedo = random.uniform(0.5, 0.7)
        atmosphere = AtmosphereComposition()
        atmosphere.add_component("H2", random.uniform(70, 85))
        atmosphere.add_component("He", random.uniform(15, 25))
        atmosphere.add_component("CH4", random.uniform(1, 3))
        atmosphere.add_component("NH3", random.uniform(0.1, 1))
        atmosphere.normalize()
        surface_composition = "无固体表面，液态金属氢层"
        tectonic_activity = "None"
    
    elif planet_type == PlanetType.GAS_GIANT:
        # 气态巨行星: R ∝ M^0.0 (50-5000 M⊕)
        mass = round(10 ** random.uniform(1.7, 3.7), 3)  # 50-5000 M⊕
        radius = round(10.0 * (mass ** 0.0), 3)  # 半径基本不变
        if mass > 300:  # 质量过大导致收缩
            radius = round(radius * (300/mass)**0.1, 3)
        density = round(random.uniform(0.2, 1.5), 2)
        albedo = random.uniform(0.3, 0.5)
        atmosphere = AtmosphereComposition()
        atmosphere.add_component("H2", random.uniform(85, 93))
        atmosphere.add_component("He", random.uniform(7, 15))
        atmosphere.add_component("CH4", random.uniform(0.1, 0.5))
        atmosphere.add_component("NH3", random.uniform(0.05, 0.2))
        atmosphere.normalize()
        surface_composition = "无固体表面，液态金属氢层"
        tectonic_activity = "None"
    
    # 添加缺失的行星类型定义
    elif planet_type == PlanetType.OCEAN_WORLD:
        # 海洋世界: 类似超级地球但表面完全被海洋覆盖
        mass = round(10 ** random.uniform(0.5, 1.2), 3)  # 3-16 M⊕
        radius = round(1.2 * (mass ** 0.5), 3)
        density = round(random.uniform(2.0, 3.0), 2)
        albedo = random.uniform(0.2, 0.4)
        atmosphere = AtmosphereComposition()
        atmosphere.add_component("H2O", random.uniform(70, 95))
        atmosphere.add_component("N2", random.uniform(5, 20))
        atmosphere.add_component("CO2", random.uniform(1, 10))
        atmosphere.normalize()
        surface_composition = "全球海洋覆盖，无陆地"
        tectonic_activity = "Moderate"
    
    elif planet_type == PlanetType.LAVA_WORLD:
        # 熔岩世界: 高温岩石行星
        mass = round(10 ** random.uniform(-0.5, 0.5), 3)  # 0.3-3 M⊕
        radius = round(1.0 * (mass ** 0.3), 3)
        density = round(random.uniform(4.0, 5.5), 2)
        albedo = random.uniform(0.05, 0.15)
        atmosphere = AtmosphereComposition()
        atmosphere.add_component("CO2", random.uniform(80, 95))
        atmosphere.add_component("SO2", random.uniform(5, 15))
        atmosphere.add_component("N2", random.uniform(1, 5))
        atmosphere.normalize()
        surface_composition = "熔岩表面，火山活动频繁"
        tectonic_activity = "High"
    
    elif planet_type == PlanetType.HELIUM_PLANET:
        # 氦行星: 主要由氦组成的气态行星
        mass = round(10 ** random.uniform(1.0, 2.0), 3)  # 10-100 M⊕
        radius = round(3.0 * (mass ** 0.5), 3)
        density = round(random.uniform(0.5, 1.0), 2)
        albedo = random.uniform(0.4, 0.6)
        atmosphere = AtmosphereComposition()
        atmosphere.add_component("He", random.uniform(90, 99))
        atmosphere.add_component("H2", random.uniform(1, 10))
        atmosphere.normalize()
        surface_composition = "无固体表面，氦大气层"
        tectonic_activity = "None"
    
    elif planet_type == PlanetType.CHTHONIAN:
        # 克托尼行星: 气态巨行星的大气被剥离后留下的核心
        mass = round(10 ** random.uniform(1.0, 1.7), 3)  # 10-50 M⊕
        radius = round(1.5 * (mass ** 0.3), 3)
        density = round(random.uniform(5.0, 8.0), 2)
        albedo = random.uniform(0.1, 0.3)
        atmosphere = AtmosphereComposition()
        atmosphere.add_component("H2", random.uniform(10, 40))
        atmosphere.add_component("He", random.uniform(5, 20))
        atmosphere.add_component("CO2", random.uniform(30, 60))
        atmosphere.normalize()
        surface_composition = "裸露的岩石和金属核心"
        tectonic_activity = "Low"
    
    elif planet_type == PlanetType.CARBON_PLANET:
        # 碳行星: 富含碳的行星
        mass = round(10 ** random.uniform(-0.5, 0.5), 3)  # 0.3-3 M⊕
        radius = round(0.9 * (mass ** 0.3), 3)
        density = round(random.uniform(3.0, 4.0), 2)
        albedo = random.uniform(0.05, 0.15)
        atmosphere = AtmosphereComposition()
        atmosphere.add_component("CO", random.uniform(50, 80))
        atmosphere.add_component("CO2", random.uniform(10, 30))
        atmosphere.add_component("CH4", random.uniform(5, 15))
        atmosphere.normalize()
        surface_composition = "石墨和金刚石表面"
        tectonic_activity = "Moderate"
    
    elif planet_type == PlanetType.IRON_PLANET:
        # 铁行星: 主要由铁构成的行星
        mass = round(10 ** random.uniform(-0.5, 0.5), 3)  # 0.3-3 M⊕
        radius = round(0.8 * (mass ** 0.3), 3)
        density = round(random.uniform(7.0, 8.0), 2)
        albedo = random.uniform(0.1, 0.2)
        atmosphere = AtmosphereComposition()
        atmosphere.add_component("CO2", random.uniform(70, 90))
        atmosphere.add_component("SO2", random.uniform(5, 20))
        atmosphere.add_component("N2", random.uniform(1, 10))
        atmosphere.normalize()
        surface_composition = "铁和镍构成的表面"
        tectonic_activity = "Low"
    
    elif planet_type == PlanetType.DESERT_PLANET:
        # 沙漠行星: 干旱的岩石行星
        mass = round(10 ** random.uniform(-0.5, 0.5), 3)  # 0.3-3 M⊕
        radius = round(1.0 * (mass ** 0.3), 3)
        density = round(random.uniform(3.5, 4.5), 2)
        albedo = random.uniform(0.25, 0.35)
        atmosphere = AtmosphereComposition()
        atmosphere.add_component("CO2", random.uniform(60, 80))
        atmosphere.add_component("N2", random.uniform(15, 30))
        atmosphere.add_component("Ar", random.uniform(1, 5))
        atmosphere.normalize()
        surface_composition = "沙漠和岩石表面"
        tectonic_activity = "Low"
    
    elif planet_type == PlanetType.SNOWBALL_PLANET:
        # 雪球行星: 完全被冰覆盖的行星
        mass = round(10 ** random.uniform(-0.5, 0.5), 3)  # 0.3-3 M⊕
        radius = round(1.1 * (mass ** 0.3), 3)
        density = round(random.uniform(2.5, 3.5), 2)
        albedo = random.uniform(0.6, 0.8)
        atmosphere = AtmosphereComposition()
        atmosphere.add_component("N2", random.uniform(70, 90))
        atmosphere.add_component("O2", random.uniform(5, 20))
        atmosphere.add_component("CO2", random.uniform(1, 5))
        atmosphere.normalize()
        surface_composition = "全球冰盖覆盖"
        tectonic_activity = "None"
    
    else:
        # 默认处理：如果遇到未知类型，使用类地行星的参数
        mass = round(10 ** random.uniform(-1, 0), 3)  # 0.1-1 M⊕
        radius = round(1.0 * (mass ** 0.27), 3)
        density = round(random.uniform(3.5, 5.5), 2)
        albedo = random.uniform(0.1, 0.3)
        atmosphere = AtmosphereComposition()
        atmosphere.add_component("CO2", random.uniform(50, 95))
        atmosphere.add_component("N2", random.uniform(5, 40))
        atmosphere.add_component("Ar", random.uniform(0.1, 2))
        atmosphere.normalize()
        surface_composition = "硅酸盐岩石"
        tectonic_activity = random.choice(["Low", "Moderate", "High"])
    
    # 表面重力 (m/s²)
    # g = G * M / R²
    M_kg = mass * 5.972e24  # 地球质量转kg
    R_m = radius * 6.371e6  # 地球半径转m
    G = 6.67430e-11  # 引力常数
    surface_gravity = round(G * M_kg / (R_m ** 2), 2)
    
    # 大气压 (bar)
    if planet_type in [PlanetType.TERRESTRIAL, PlanetType.SUPER_EARTH, PlanetType.OCEAN_WORLD,
                      PlanetType.LAVA_WORLD, PlanetType.DESERT_PLANET, PlanetType.SNOWBALL_PLANET,
                      PlanetType.CHTHONIAN, PlanetType.CARBON_PLANET, PlanetType.IRON_PLANET]:
        atmosphere_pressure = round(10 ** random.uniform(-1, 2), 3)  # 0.1-100 bar
    elif planet_type in [PlanetType.SUB_NEPTUNE, PlanetType.MINI_NEPTUNE]:
        atmosphere_pressure = round(10 ** random.uniform(2, 5), 3)  # 100-100,000 bar
    else:  # 气态行星
        atmosphere_pressure = "N/A (深大气层)"
    
    # 磁场 (高斯)
    # 基于发电机理论: 需要熔融核心和快速自转
    if planet_type in [PlanetType.TERRESTRIAL, PlanetType.SUPER_EARTH, PlanetType.OCEAN_WORLD,
                      PlanetType.LAVA_WORLD, PlanetType.DESERT_PLANET, PlanetType.SNOWBALL_PLANET,
                      PlanetType.CHTHONIAN, PlanetType.CARBON_PLANET, PlanetType.IRON_PLANET] and mass > 0.5:
        magnetic_field = round(random.uniform(0.1, 1.0), 3)  # 地球磁场的0.1-1倍
    else:
        magnetic_field = 0.0
    
    return {
        'mass': mass,
        'radius': radius,
        'density': density,
        'surface_gravity': surface_gravity,
        'albedo': albedo,
        'atmosphere': atmosphere,
        'atmosphere_pressure': atmosphere_pressure,
        'surface_composition': surface_composition,
        'tectonic_activity': tectonic_activity,
        'magnetic_field': magnetic_field
    }

def generate_moon_system(planet_mass, sma, star):
    """
    生成行星的卫星系统
    """
    # 希尔球半径 (行星引力主导的区域)
    R_hill = sma * (planet_mass / (3 * star['mass'])) ** (1/3)
    
    # 最大稳定卫星轨道距离 (约1/3希尔球半径)
    max_moon_distance = R_hill / 3
    
    # 可能的卫星数量
    if planet_mass < 0.1:  # 小行星
        num_moons = 0
    elif planet_mass < 1:  # 火星大小
        num_moons = random.randint(0, 2)
    elif planet_mass < 10:  # 超级地球
        num_moons = random.randint(0, 3)
    elif planet_mass < 50:  # 海王星大小
        num_moons = random.randint(2, 8)
    else:  # 气态巨行星
        num_moons = random.randint(4, 20)
    
    moons = []
    for i in range(num_moons):
        # 卫星轨道距离 (行星半径倍数)
        distance = round(random.uniform(2, max_moon_distance * 0.9), 4)
        
        # 卫星大小 (相对)
        max_moon_size = planet_mass * 0.0001  # 月球质量约为地球的0.0123
        size = round(random.uniform(0.0001, max_moon_size), 6)
        
        # 卫星类型
        if size < 0.001:
            moon_type = "Small irregular"
        elif size < 0.01:
            moon_type = "Medium regular"
        else:
            moon_type = "Large spherical"
        
        moons.append({
            'distance': distance,  # 行星半径倍数
            'size': size,  # 地球质量
            'type': moon_type
        })
    
    return moons

def generate_ring_system(planet_mass, sma, star):
    """
    生成行星环系统
    """
    # 环存在的条件
    if planet_mass < 1 or sma < star['snow_line'] * 0.5:
        return None
    
    # 环存在概率
    ring_prob = 0.3
    if planet_mass > 10:  # 气态巨行星
        ring_prob = 0.7
    if random.random() > ring_prob:
        return None
    
    # 环参数
    inner_edge = round(random.uniform(1.5, 2.0), 2)  # 行星半径倍数
    outer_edge = round(random.uniform(2.0, 5.0), 2)
    thickness = round(random.uniform(0.01, 0.1), 3)  # 行星半径倍数
    
    # 环组成
    if sma < star['snow_line']:
        composition = "Rocky particles"
    else:
        composition = "Ice particles"
    
    return {
        'inner_edge': inner_edge,
        'outer_edge': outer_edge,
        'thickness': thickness,
        'composition': composition
    }

def generate_planetary_system(star, num_weights=None, type_weights=None, hz_weight=1.0):
    """
    生成完整的行星系统
    """
    if num_weights is None:
        num_weights = {}
    if type_weights is None:
        type_weights = {}
    
    # 行星数量分布 - 基于开普勒数据
    # 每颗恒星平均行星数: 1.6 (Mulders et al. 2018)
    num_options = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    base_probs = [0.2, 0.15, 0.15, 0.15, 0.12, 0.1, 0.06, 0.04, 0.03]
    
    # 应用权重
    weighted_probs = [base_probs[i] * num_weights.get(num_options[i], 1.0) 
                     for i in range(len(num_options))]
    total = sum(weighted_probs)
    norm_probs = [p/total for p in weighted_probs]
    
    # 随机选择行星数量
    num_planets = np.random.choice(num_options, p=norm_probs)
    
    # 生成行星
    planets = []
    for i in range(num_planets):
        planet = generate_planet(star, i, num_planets, type_weights)
        planets.append(planet)
    
    # 生成小行星带
    asteroid_belt = generate_asteroid_belt(planets, star)
    
    # 生成柯伊伯带
    kuiper_belt = generate_kuiper_belt(planets, star)
    
    # 生成碎片盘
    debris_disk = generate_debris_disk(star)
    
    # 系统特征
    system_features = {
        'asteroid_belt': asteroid_belt,
        'kuiper_belt': kuiper_belt,
        'debris_disk': debris_disk
    }
    
    return planets, system_features

def generate_asteroid_belt(planets, star):
    """
    生成小行星带
    """
    if len(planets) < 2:
        return None
    
    # 寻找类地行星和气态巨行星之间的间隙
    terrestrial_planets = [p for p in planets if p['type'] in ['Terrestrial', 'Super-Earth']]
    giant_planets = [p for p in planets if p['type'] in ['Ice Giant', 'Gas Giant']]
    
    if not terrestrial_planets or not giant_planets:
        return None
    
    max_terrestrial = max(p['semi_major_axis'] for p in terrestrial_planets)
    min_giant = min(p['semi_major_axis'] for p in giant_planets)
    
    if min_giant > max_terrestrial * 1.5:
        inner_edge = max_terrestrial * 1.1
        outer_edge = min_giant * 0.9
        width = outer_edge - inner_edge
        
        # 雪线位置
        snow_line = star['snow_line']
        composition = "Rocky" if inner_edge < snow_line else "Icy"
        
        return {
            'inner_edge': round(inner_edge, 3),
            'outer_edge': round(outer_edge, 3),
            'width': round(width, 3),
            'estimated_objects': random.randint(1000, 1000000),
            'composition': composition
        }
    
    return None

def generate_kuiper_belt(planets, star):
    """
    生成柯伊伯带
    """
    if not planets:
        return None
    
    # 找到最外层的行星
    outer_planet = max(planets, key=lambda p: p['semi_major_axis'])
    
    if outer_planet['semi_major_axis'] < 10:
        return None
    
    inner_edge = outer_planet['semi_major_axis'] * 1.5
    outer_edge = inner_edge * random.uniform(2, 10)
    
    return {
        'inner_edge': round(inner_edge, 3),
        'outer_edge': round(outer_edge, 3),
        'width': round(outer_edge - inner_edge, 3),
        'estimated_objects': random.randint(10000, 10000000),
        'composition': "Icy bodies"
    }

def generate_debris_disk(star):
    """
    生成碎片盘
    """
    # 年轻恒星更可能有碎片盘
    if star['age'] < 0.1:  # 年轻恒星
        prob = 0.8
    elif star['age'] < 1:  # 中年恒星
        prob = 0.3
    else:  # 古老恒星
        prob = 0.05
    
    if random.random() > prob:
        return None
    
    inner_edge = random.uniform(10, 50)
    outer_edge = inner_edge * random.uniform(2, 10)
    
    return {
        'inner_edge': round(inner_edge, 3),
        'outer_edge': round(outer_edge, 3),
        'width': round(outer_edge - inner_edge, 3),
        'temperature': round(random.uniform(30, 150), 1)  # K
    }

# 恒星系生成函数
def generate_stellar_system(
    star_type_weights=None,
    specified_star_type=None,
    num_planets_weights=None,
    planet_type_weights=None,
    habitable_zone_weight=1.0,
    asteroid_belt_weight=1.0,
    kuiper_belt_weight=1.0,
    debris_disk_weight=1.0
):
    """
    生成完整的恒星系
    可以指定恒星光谱类型
    
    参数:
    specified_star_type (str or StarType): 指定的恒星类型
    """
    # 1. 生成恒星
    star = generate_star(weights=star_type_weights, specified_type=specified_star_type)
    
    # 2. 对于致密天体，可能没有行星系统
    if star['spectral_type'] in ['White Dwarf', 'Neutron Star', 'Black Hole']:
        # 白矮星可能有一个残存的行星系统
        if star['spectral_type'] == 'White Dwarf' and random.random() < 0.3:
            planets, system_features = generate_planetary_system(
                star, 
                num_planets_weights,
                planet_type_weights,
                habitable_zone_weight
            )
        else:
            planets = []
            system_features = {
                'asteroid_belt': None,
                'kuiper_belt': None,
                'debris_disk': None
            }
    else:
        # 生成行星系统
        planets, system_features = generate_planetary_system(
            star, 
            num_planets_weights,
            planet_type_weights,
            habitable_zone_weight
        )
    
    # 3. 应用系统特征权重
    if system_features['asteroid_belt'] is not None and random.random() > asteroid_belt_weight:
        system_features['asteroid_belt'] = None
    if system_features['kuiper_belt'] is not None and random.random() > kuiper_belt_weight:
        system_features['kuiper_belt'] = None
    if system_features['debris_disk'] is not None and random.random() > debris_disk_weight:
        system_features['debris_disk'] = None
    
    # 4. 组装结果
    stellar_system = {
        'star': star,
        'planets': planets,
        'system_features': system_features
    }
    
    return stellar_system

# 示例用法
if __name__ == "__main__":
    # 生成一个类太阳恒星系
    solar_system = generate_stellar_system(
        specified_star_type="G"  # 指定G型星 (类太阳)
    )
    
    # 生成一个白矮星星系
    white_dwarf_system = generate_stellar_system(
        specified_star_type="White Dwarf"
    )
    
    # 生成一个黑洞系统
    black_hole_system = generate_stellar_system(
        specified_star_type="Black Hole"
    )
    
    # 打印类太阳系统的信息
    star = solar_system['star']
    print(f"\n类太阳恒星系:")
    print(f"恒星类型: {star['spectral_type']}")
    print(f"质量: {star['mass']} M☉, 光度: {star['luminosity']} L☉")
    print(f"温度: {star['effective_temperature']} K, 半径: {star['radius']} R☉")
    print(f"宜居带: {star['habitable_zone_inner']}-{star['habitable_zone_outer']} AU")
    print(f"行星数量: {len(solar_system['planets'])}")
    
    for i, planet in enumerate(solar_system['planets']):
        hz_flag = " (宜居带)" if planet['in_habitable_zone'] else ""
        print(f"行星 {i+1}: {planet['type']}{hz_flag}")
        print(f"  质量: {planet['mass']} M⊕, 半径: {planet['radius']} R⊕")
        print(f"  轨道: {planet['semi_major_axis']} AU")
    
    # 打印白矮星系统的信息
    star = white_dwarf_system['star']
    print(f"\n白矮星星系:")
    print(f"恒星类型: {star['spectral_type']}")
    print(f"质量: {star['mass']} M☉, 光度: {star['luminosity']} L☉")
    print(f"温度: {star['effective_temperature']} K, 半径: {star['radius']} R☉")
    print(f"行星数量: {len(white_dwarf_system['planets'])}")
    
    # 打印黑洞系统的信息
    star = black_hole_system['star']
    print(f"\n黑洞系统:")
    print(f"类型: {star['spectral_type']}")
    print(f"质量: {star['mass']} M☉, 史瓦西半径: {star['schwarzschild_radius']} 公里")
    print(f"行星数量: {len(black_hole_system['planets'])}")
