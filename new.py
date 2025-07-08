import json
import os
import pygame
import random
import sys
import time
import numpy as np

# 初始化pygame
pygame.init()

# 获取桌面分辨率（已注释）
desktop_sizes = pygame.display.get_desktop_sizes()

# 设置游戏窗口尺寸
WIDTH, HEIGHT = 1280, 720  # 使用固定分辨率替代桌面尺寸 #desktop_sizes[0]
# 设置字体路径
Deng_path = r'C:\Windows\Fonts\Deng.ttf'
# 创建时钟对象控制帧率
clock = pygame.time.Clock()
# 创建游戏窗口
screen = pygame.display.set_mode((WIDTH, HEIGHT))

# 建筑物分类字典
BUILDINGS = {
    "运输": ["传送带", "星际传输枢纽"],
    "储存": ["储液罐", "储物仓"],
    "电力": ["地热发电机", "太阳能电池板", "核聚变发电机", "风力发电机"],
    "生产": ["组装机", "熔炉"],
    "采矿": ["采矿机", "液体抽取器"],
    "研究": ["研究站"],
    "景观": ["水", "沙", "土"],
    "矿物": ["煤矿", "石油", "石矿", "硅矿", "钛矿", "铁矿", "铜矿"]
}

# 颜色定义字典
COLORS = {
    "bar": (220, 250, 250, 50),  # 半透明状态条
    "grid": (50, 50, 50),         # 网格线颜色
    "player": (224, 255, 255),    # 玩家颜色
    "text_cap": (120, 120, 120),  # 文字标题颜色
    "text_active": (255, 255, 0), # 激活文字颜色
    "input_border_inactive": (100, 180, 255),  # 输入框非激活边框
    "input_border_active": (70, 70, 100),      # 输入框激活边框
    "input_bg": (25, 25, 40),     # 输入框背景
    "caption": (200, 200, 200),   # 说明文字
    "button": (120, 190, 255),    # 按钮颜色
    "button_hover": (95, 255, 255),  # 按钮悬停颜色
    "button_bg": (20, 200, 200),  # 按钮背景
    "bg": (0, 0, 0)               # 背景颜色
}

# 创建不同大小的字体字典
FONTS = {
    f'Deng{size}': pygame.font.Font(Deng_path, size) for size in range(1, 61)
}

# 物品分类
GOODS = {
    'building': {"传送带", "星际传输枢纽", "储液罐", "储物仓", "地热发电机", "太阳能电池板", 
                 "核聚变发电机", "风力发电机", "组装机", "熔炉", "采矿机", "液体抽取器", "研究站"},
    'good': {"煤炭", "铁原矿", "铜原矿", "石原矿", "硅石", "钛原矿", "石油", "氢", "水", "硫酸",
             "石墨", "燃烧单元", "铁板", "磁铁", "铜板", "砖", "玻璃", "高纯硅版", "钛板", "精炼油",
             "金刚石", "齿轮", "钢材", "单晶硅", "磁线圈", "电路板", "集成电路", ""}
}

# 加载所有建筑物图片
IMGS = {}
for label, names in BUILDINGS.items():
    for name in names:
        try:
            IMGS[f'{label}_{name}'] = pygame.image.load(f'./pictures/{name}.png')
        except pygame.error as e:
            print(f"Error loading image {'./pictures/{name}.png'}: {e}")

with open('tech.json', 'r', encoding='utf-8') as f:
    TECHS = json.load(f)

class PerlinNoiseGenerator:
    """柏林噪声生成器类，用于生成，正态分布（？）"""
    def __init__(self, seed=None):
        # 设置随机种子
        self.seed = seed if seed is not None else random.randint(0, 100000000)
        random.seed(self.seed)
        # 创建并打乱排列表
        self.permutation = list(range(256))
        random.shuffle(self.permutation)
        # 双倍排列表用于快速查找
        self.p = self.permutation + self.permutation
    
    def fade(self, t):
        """平滑曲线函数，使噪声更自然"""
        return t * t * t * (t * (t * 6 - 15) + 10)
    
    def lerp(self, a, b, x):
        """线性插值函数"""
        return a + x * (b - a)
    
    def grad(self, hash, x, y, z):
        """梯度函数，生成随机梯度向量"""
        h = hash & 15
        u = x if h < 8 else y
        v = y if h < 4 else (x if h == 12 or h == 14 else z)
        return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)
    
    def noise(self, x, y, z=0):
        """生成柏林噪声"""
        # 计算网格单元坐标
        X = int(np.floor(x)) & 255
        Y = int(np.floor(y)) & 255
        Z = int(np.floor(z)) & 255
        
        # 计算在单元内的相对位置
        x -= np.floor(x)
        y -= np.floor(y)
        z -= np.floor(z)
        
        # 计算平滑曲线值
        u = self.fade(x)
        v = self.fade(y)
        w = self.fade(z)
        
        # 哈希坐标计算
        A = self.p[X] + Y
        AA = self.p[A] + Z
        AB = self.p[A + 1] + Z
        B = self.p[X + 1] + Y
        BA = self.p[B] + Z
        BB = self.p[B + 1] + Z
        
        # 混合所有梯度贡献值
        return self.lerp(
            self.lerp(
                self.lerp(self.grad(self.p[AA], x, y, z), 
                self.grad(self.p[BA], x - 1, y, z), u),
                self.lerp(self.grad(self.p[AB], x, y - 1, z), 
                self.grad(self.p[BB], x - 1, y - 1, z), u),
                v
            ),
            self.lerp(
                self.lerp(self.grad(self.p[AA + 1], x, y, z - 1), 
                self.grad(self.p[BA + 1], x - 1, y, z - 1), u),
                self.lerp(self.grad(self.p[AB + 1], x, y - 1, z - 1), 
                self.grad(self.p[BB + 1], x - 1, y - 1, z - 1), u),
                v
            ),
            w
        )

class GameState:
    """游戏状态枚举类"""
    build, start, save, setting, tech, playing, menu = range(7)
    transport, storage, electricity, product, mining, research = range(10, 16)

class PlanetMap:
    """行星地图类，管理游戏世界"""
    def __init__(self, size: tuple):
        # 地图尺寸 (宽, 高)
        self.size = size
        # 地形类型对应的图片
        self.terrain_to_image = (
            IMGS['景观_水'],  # 水
            IMGS['景观_沙'],  # 沙
            IMGS['景观_土'],  # 土
            IMGS['矿物_煤矿'],  # 煤矿
            IMGS['矿物_石油'],  # 石油
            IMGS['矿物_石矿'],  # 石矿
            IMGS['矿物_硅矿'],  # 硅矿
            IMGS['矿物_钛矿'],  # 钛矿
            IMGS['矿物_铁矿'],  # 铁矿
            IMGS['矿物_铜矿'],  # 铜矿
            None
        )
        # 随机生成地图种子
        self.seed = random.randint(0, 100000000)
        # 玩家足迹地图
        self.player_map = np.zeros(shape=size, dtype=bool)
        # 基础地形地图（使用柏林噪声生成）
        self.base_map = self.generate_perlin_terrain(
            size=self.size[0], 
            terrain_ratios={0: 0.5, 1: 0.2, 2:0.3},
            terrain_to_image=self.terrain_to_image,
            seed=self.seed
        )
        # 资源地图（使用柏林噪声生成） 矿物概率有问题，懒得调了，就这样吧
        self.iron_map = self.generate_perlin_terrain(
            size=self.size[0],
            terrain_ratios={10:0.6, 3: 0.01, 4: 0.05, 5: 0.01, 6: 0.01, 7: 0.025, 8: 0.05, 9: 0.045},
            terrain_to_image=self.terrain_to_image,
            seed=self.seed
        )
        # 建筑地图
        self.building_map = np.zeros(shape=size, dtype=list)
        # 地图图层集合
        self.map = [self.player_map, self.base_map, self.building_map, self.iron_map]
        # 视图区域 [x, y, width, height]
        self.view = [0, 0, WIDTH, HEIGHT]
        # 单元格大小（像素）
        self.cell_size = 20
        self.save_message = None
        self.save_message_time = 0
        # 添加移动状态记录
        self.moving_up = False
        self.moving_down = False
        self.moving_left = False
        self.moving_right = False
    
    def draw(self):
        """绘制地图到屏幕"""
        # 计算起始绘制位置（网格对齐）
        start_x = self.view[0] // self.cell_size * self.cell_size
        start_y = self.view[1] // self.cell_size * self.cell_size       
        # 计算结束位置（不超过地图边界）
        end_x = min(self.view[0] + self.view[2], self.size[0] * self.cell_size)
        end_y = min(self.view[1] + self.view[3], self.size[1] * self.cell_size)
        
        # 遍历视口内的所有单元格
        for x in range(start_x, end_x, self.cell_size):
            for y in range(start_y, end_y, self.cell_size):
                # 计算屏幕坐标
                screen_x = x - self.view[0]
                screen_y = y - self.view[1]
                # 创建单元格矩形
                rect = pygame.Rect(screen_x, screen_y, self.cell_size, self.cell_size)
                # 计算网格坐标
                grid_x = x // self.cell_size
                grid_y = y // self.cell_size
                
                # 检查是否在地图范围内
                if 0 <= grid_x < self.size[0] and 0 <= grid_y < self.size[1]:
                    # 获取地形图片
                    terrain_img = self.base_map[grid_y, grid_x]
                    # 绘制图片
                    screen.blit(terrain_img, (screen_x, screen_y))

                    
                    # 获取矿物图片
                    iron_img = self.iron_map[grid_y, grid_x]
                    # 绘制图片
                    if iron_img:
                        screen.blit(iron_img, (screen_x, screen_y))
                
                # 绘制网格线
                pygame.draw.rect(screen, COLORS["grid"], rect, 1)

    def generate_perlin_terrain(self, size: int, terrain_ratios: dict,
                              terrain_to_image: list, seed:int,
                              octaves=6, persistence=0.5):
        """使用柏林噪声生成地形"""
        # 设置随机种子
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # 初始化地形数组
        terrain = np.zeros((size, size), dtype=float)
        generator = PerlinNoiseGenerator(seed)
        
        # 噪声生成参数
        max_amplitude = 0
        amplitude = 1.0
        frequency = 1.0 / size
        
        # 多倍频叠加生成更自然的地形
        for _ in range(octaves):
            for y in range(size):
                for x in range(size):
                    nx = x * frequency
                    ny = y * frequency
                    # 累加噪声值
                    terrain[y, x] += generator.noise(nx, ny) * amplitude
            max_amplitude += amplitude
            amplitude *= persistence  # 振幅递减
            frequency *= 2            # 频率递增
        
        # 归一化噪声值到[0,1]范围
        terrain = (terrain / max_amplitude + 1) / 2
        
        # 根据阈值划分地形类型
        result = np.zeros(shape=(size, size), dtype=np.uint8)
        # 计算累积阈值
        thresholds = []
        cumulative = 0.0
        terrain_types = sorted(terrain_ratios.keys())

        for t in terrain_types:
            cumulative += terrain_ratios[t]
            thresholds.append(cumulative)

        # 分配地形类型
        for i in range(size):
            for j in range(size):
                value = terrain[i, j]
                for idx, th in enumerate(thresholds):
                    if value <= th:
                        result[i, j] = terrain_types[idx]
                        break
                else:
                    result[i, j] = terrain_types[-1]
        
        # 转换为图片数组
        img_list = [terrain_to_image[int(result[i, j])] for i in range(size) for j in range(size)]
        img_array = np.array(img_list, dtype=object).reshape((size, size))
        return img_array
    
    def handle_event(self, event):
        """处理地图视图移动事件"""
        if event.type == pygame.KEYDOWN:
            if event.mod == pygame.KMOD_LCTRL:
                if event.key == pygame.K_w or event.key == pygame.K_UP:
                    self.moving_up = True
                if event.key == pygame.K_s or event.key == pygame.K_DOWN:
                    self.moving_down = True
                if event.key == pygame.K_a or event.key == pygame.K_LEFT:
                    self.moving_left = True
                if event.key == pygame.K_d or event.key == pygame.K_RIGHT:
                    self.moving_right = True
        
        if event.type == pygame.KEYUP:
            if event.mod == pygame.KMOD_LCTRL:
                if event.key == pygame.K_w or event.key == pygame.K_UP:
                    self.moving_up = False
                if event.key == pygame.K_s or event.key == pygame.K_DOWN:
                    self.moving_down = False
                if event.key == pygame.K_a or event.key == pygame.K_LEFT:
                    self.moving_left = False
                if event.key == pygame.K_d or event.key == pygame.K_RIGHT:
                    self.moving_right = False
    
    def update(self):
        """更新地图视图位置"""
        move_amount = 10
        if self.moving_up and self.view[1] >= move_amount:
            self.view[1] -= move_amount
        if self.moving_down and self.view[1] <= HEIGHT-move_amount:
            self.view[1] += move_amount
        if self.moving_left and self.view[0] >= move_amount:
            self.view[0] -= move_amount
        if self.moving_right and self.view[0] <= WIDTH-move_amount:
            self.view[0] += move_amount
    
    def save_map_as_image(self):
        """将整个地图保存为图片文件"""
        try:
            # 创建保存目录
            save_dir = "saved_maps"
            os.makedirs(save_dir, exist_ok=True)
            
            # 生成文件名（带时间戳）
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{save_dir}/map_{timestamp}.png"
            
            # 创建全尺寸地图Surface
            map_width = self.size[0] * self.cell_size
            map_height = self.size[1] * self.cell_size
            map_surface = pygame.Surface((map_width, map_height))
            
            # 绘制整个地图
            for x in range(0, map_width, self.cell_size):
                for y in range(0, map_height, self.cell_size):
                    # 计算网格坐标
                    grid_x = x // self.cell_size
                    grid_y = y // self.cell_size
                    
                    # 确保在地图范围内
                    if 0 <= grid_x < self.size[0] and 0 <= grid_y < self.size[1]:
                        # 绘制地形
                        terrain_img = self.base_map[grid_y, grid_x]
                        map_surface.blit(terrain_img, (x, y))
                        
                        # 绘制资源
                        iron_img = self.iron_map[grid_y, grid_x]
                        if iron_img is not None:  # 确保资源图片存在
                            map_surface.blit(iron_img, (x, y))
                    
                    # 绘制网格线
                    rect = pygame.Rect(x, y, self.cell_size, self.cell_size)
                    pygame.draw.rect(map_surface, COLORS["grid"], rect, 1)
            
            # 保存为PNG文件
            pygame.image.save(map_surface, filename)
            
            # 设置保存成功消息
            self.save_message = f"地图已保存到: {filename}"
            self.save_message_time = time.time()
            return True
        except Exception as e:
            print(f"保存地图失败: {e}")
            self.save_message = f"保存失败: {str(e)}"
            self.save_message_time = time.time()
            return False

class Player:
    """玩家角色类"""
    def __init__(self, color=(224, 255, 255)):
        self.color = color  # 玩家颜色
        self.x = 0          # X坐标
        self.y = 0          # Y坐标
        self.size = 20      # 尺寸
        self.speed = 10     # 移动速度
        # 添加移动状态记录
        self.moving_up = False
        self.moving_down = False
        self.moving_left = False
        self.moving_right = False
        self.tech = []
        
    def change(self, color):
        """改变玩家颜色"""
        if self.color != color:
            self.color = color
    
    def draw(self, view):
        """在屏幕上绘制玩家"""
        # 计算屏幕坐标
        screen_x = self.x - view[0]
        screen_y = self.y - view[1]
        # 确保在视图范围内
        if (0 <= screen_x <= view[2] and 0 <= screen_y <= view[3]):
            rect = pygame.Rect(screen_x, screen_y, self.size, self.size)
            pygame.draw.rect(screen, self.color, rect)
    
    def handle_event(self, event):
        """处理玩家移动事件"""
        if event.type == pygame.KEYDOWN:
            if event.mod == pygame.KMOD_NONE:
                if event.key == pygame.K_w or event.key == pygame.K_UP:
                    self.moving_up = True
                if event.key == pygame.K_s or event.key == pygame.K_DOWN:
                    self.moving_down = True
                if event.key == pygame.K_a or event.key == pygame.K_LEFT:
                    self.moving_left = True
                if event.key == pygame.K_d or event.key == pygame.K_RIGHT:
                    self.moving_right = True
        
        if event.type == pygame.KEYUP:
            if event.mod == pygame.KMOD_NONE:
                if event.key == pygame.K_w or event.key == pygame.K_UP:
                    self.moving_up = False
                if event.key == pygame.K_s or event.key == pygame.K_DOWN:
                    self.moving_down = False
                if event.key == pygame.K_a or event.key == pygame.K_LEFT:
                    self.moving_left = False
                if event.key == pygame.K_d or event.key == pygame.K_RIGHT:
                    self.moving_right = False
    
    def update(self):
        """更新玩家位置"""
        if self.moving_up and self.y >= self.speed:
            self.y -= self.speed
        if self.moving_down and self.y <= HEIGHT-self.speed:
            self.y += self.speed
        if self.moving_left and self.x >= self.speed:
            self.x -= self.speed
        if self.moving_right and self.x <= WIDTH-self.speed:
            self.x += self.speed

class Button:
    """按钮UI类"""
    def __init__(self, font, text, center=(0, 0), pos=(0, 0)):
        self.font = font        # 按钮字体
        self.text = text        # 按钮文本
        self.center = center    # 中心位置（优先级高）
        self.pos = pos          # 左上角位置
        self.ishorved = False   # 鼠标悬停状态
        self.rect = None        # 按钮矩形区域
    
    def draw(self):
        """绘制按钮"""
        # 根据悬停状态选择颜色
        self.color = COLORS["button_bg"] if not self.ishorved else COLORS["button_hover"]
        # 渲染文本
        text_surface = self.font.render(self.text, True, self.color, COLORS['bg'])
        size = self.font.size(self.text)
        
        # 确定按钮位置
        if self.center != (0, 0):
            self.rect = text_surface.get_rect(center=self.center)
        else:
            self.rect = pygame.Rect(self.pos, size)
        
        # 绘制到屏幕
        screen.blit(text_surface, self.rect)

    def update(self, mouse_pos):
        """更新按钮悬停状态"""
        if self.rect:
            self.ishorved = self.rect.collidepoint(mouse_pos)
    
    def collided(self, event):
        """检查按钮点击事件"""
        if self.ishorved:
            return self.text

class Building:
    """建筑类"""
    def __init__(self, type_: str, name: str, pos: tuple):
        self.type = BUILDINGS[type_]       # 建筑类别
        self.name = BUILDINGS[type_][name] # 建筑名称
        self.img = IMGS[f'{self.type}_{self.name}']  # 建筑图片
        self.rect = pygame.rect.Rect(pos, (20, 20))  # 建筑矩形区域
    
    def draw(self, view):
        """绘制建筑（未实现）"""
        screen_x = self.x - view[0]
        screen_y = self.y - view[1]
        if (0 <= screen_x <= view[2] and 0 <= screen_y <= view[3]):
            rect = pygame.Rect(screen_x, screen_y, self.size, self.size)
            pygame.draw.rect(screen, self.color, rect)

# 创建各游戏界面的按钮 =====================================

# 开始界面按钮
start_buttons = [
    Button(FONTS['Deng40'], ["新游戏", "存档", "设置", "退出"][i], 
    pos=(WIDTH//10, HEIGHT//8*(i+1))) for i in range(4)
]

# 存档界面按钮
save_buttons = [
    Button(FONTS['Deng20'], "返回", center=(WIDTH//10*9, HEIGHT//20))
]

# 设置界面按钮
setting_buttons = [
    Button(FONTS['Deng20'], "返回", center=(WIDTH//10*9, HEIGHT//20*19))
]

# 菜单界面按钮
menu_buttons = [
    Button(FONTS['Deng40'], ["继续", "读档", "存档", "设置", "退出"][i], 
    center=(WIDTH//2, HEIGHT//8*(i+2))) for i in range(5)
]

# 游戏界面底部建筑类别按钮
playing_buttons = [
    Button(FONTS['Deng30'], ["运输", "储存", "电力", "生产", "采矿", "研究"][i], 
    center=(WIDTH//(i+1), HEIGHT//10*9)) for i in range(6)
]

playing_buttons_transport = [
    Button(FONTS['Deng30'], BUILDINGS[["运输", "储存", "电力", "生产", "采矿", "研究"][i]], 
    center=(WIDTH//(i+1), HEIGHT//5*4)) for i in range(6)
]

playing_buttons_storage = [
    Button(FONTS['Deng30'], BUILDINGS[["运输", "储存", "电力", "生产", "采矿", "研究"][i]], 
    center=(WIDTH//(i+1), HEIGHT//5*4)) for i in range(6)
]

playing_buttons_electricity = [
    Button(FONTS['Deng30'], BUILDINGS[["运输", "储存", "电力", "生产", "采矿", "研究"][i]], 
    center=(WIDTH//(i+1), HEIGHT//5*4)) for i in range(6)
]

playing_buttons_product = [
    Button(FONTS['Deng30'], BUILDINGS[["运输", "储存", "电力", "生产", "采矿", "研究"][i]], 
    center=(WIDTH//(i+1), HEIGHT//5*4)) for i in range(6)
]

playing_buttons_mining = [
    Button(FONTS['Deng30'], BUILDINGS[["运输", "储存", "电力", "生产", "采矿", "研究"][i]], 
    center=(WIDTH//(i+1), HEIGHT//5*4)) for i in range(6)
]

playing_buttons_research = [
    Button(FONTS['Deng30'], BUILDINGS[["运输", "储存", "电力", "生产", "采矿", "研究"][i]], 
    center=(WIDTH//(i+1), HEIGHT//5*4)) for i in range(6)
]

# 状态对应的按钮字典
state_buttons = {
    GameState.build: playing_buttons,
    GameState.start: start_buttons,
    GameState.save: save_buttons,
    GameState.setting: setting_buttons,
    GameState.menu: menu_buttons,
    GameState.playing: playing_buttons,
    GameState.tech:None,
    
    GameState.transport: playing_buttons_transport,
    GameState.storage: playing_buttons_storage,
    GameState.electricity: playing_buttons_electricity,
    GameState.product: playing_buttons_product,
    GameState.mining: playing_buttons_mining,
    GameState.research: playing_buttons_research
}

# 游戏主循环变量初始化 =====================================
current_state = GameState.start  # 当前游戏状态
past_state = None                # 先前游戏状态
pm = None                        # 行星地图实例
player = None                    # 玩家实例
running = True                   # 游戏运行标志

# 主游戏循环
while running:
    action = []  # 存储本帧发生的动作
    
    # 事件处理
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pm.save_map_as_image()
            running = False  # 退出游戏
        
        # 鼠标移动事件：更新按钮悬停状态
        if event.type == pygame.MOUSEMOTION:
            if current_state != GameState.playing:  # 游戏状态不处理按钮
                for b in state_buttons[current_state]:
                    b.update(event.pos)
        
        # 鼠标点击事件：检测按钮点击
        if event.type == pygame.MOUSEBUTTONDOWN:
            if current_state == GameState.playing:
                pass  # 游戏中可能有其他点击逻辑
            else:
                for b in state_buttons[current_state]:
                    clicked = b.collided(event)
                    if clicked:
                        action.append(clicked)
        
        # 键盘事件
        if event.type == pygame.KEYDOWN:
            # ESC键处理
            if event.key == pygame.K_ESCAPE:
                if current_state in (GameState.playing, GameState.build):
                    # 游戏中按ESC打开菜单
                    current_state = GameState.menu
                    past_state = GameState.playing
                elif current_state in (GameState.menu, GameState.tech):
                    current_state = GameState.playing
                    past_state = None
                else:
                    running = False  # 非游戏状态退出
            if event.key == pygame.K_b:
                if current_state == GameState.build and past_state == GameState.playing:
                    current_state = GameState.playing
                    past_state = None
                elif current_state == GameState.playing:
                    current_state = GameState.build
                    past_state = GameState.playing
            if event.key == pygame.K_f:
                pass
            if event.key == pygame.K_t:
                if current_state == GameState.tech and past_state == GameState.playing:
                    current_state = GameState.playing
                    past_state = None
                elif current_state == GameState.playing:
                    current_state = GameState.tech
                    past_state = GameState.playing

        if event.type in (pygame.KEYDOWN, pygame.KEYUP):
                # 其他按键传递给地图和玩家
                if pm:
                    pm.handle_event(event)
                if player:
                    player.handle_event(event)
    
    # 处理按钮动作
    if action:
        if "退出" in action:
            running = False
        if "设置" in action:
            current_state = GameState.setting
        if "存档" in action:
            current_state = GameState.save
        if "继续" in action:
            current_state = GameState.playing
        if "读档" in action:
            pass  # 待实现
        if "新游戏" in action:
            map_size = (100, 100)
            pm = PlanetMap(map_size)  # 创建新地图
            player = Player()         # 创建玩家
            current_state = GameState.playing  # 进入游戏状态
        if "返回" in action:
            # 返回上一状态
            if not past_state:
                current_state = GameState.start
            else:
                current_state = GameState.playing
    
    # 更新游戏状态
    if current_state in (GameState.playing, GameState.build):
        if pm:
            pm.update()
        if player:
            player.update()
    
    # 渲染 =============================================
    screen.fill(COLORS["bg"])  # 填充背景色
    
    # 绘制当前状态的UI
    if state_buttons[current_state]:
        # 绘制按钮
        for b in state_buttons[current_state]:
            b.draw()
    if current_state in (GameState.build, GameState.playing):
        # 游戏状态：绘制地图和玩家
        if pm:
            pm.draw()
        if player and pm:
            player.draw(pm.view)
            # 绘制底部状态条
            pygame.draw.rect(screen, COLORS["bar"], 
                            ((0, HEIGHT//11*10), (WIDTH, HEIGHT//11)))
    
    # 显示调试信息
    state_names = ["建造", "开始", "存档", "设置", "科技", "游戏中", "菜单"]
    state_text = FONTS['Deng20'].render(
        f"状态: {state_names[current_state]}", True, (255, 255, 255))
    screen.blit(state_text, (10, 30))
    
    # 显示玩家和视图坐标
    if player and pm:
        pos_text = FONTS['Deng20'].render(
            f'角色：{player.x},{player.y} 视图：{pm.view[0]},{pm.view[1]}', 
            True, (255, 255, 255))
        screen.blit(pos_text, (10, 10))
    
    # 刷新屏幕
    pygame.display.flip()
    clock.tick(60)  # 60 FPS

# 退出游戏
pygame.quit()
sys.exit()
