import time
import pygame
import string
import sys
import os
import numpy as np

import creation as c_m

pygame.init()
font_cap = pygame.font.Font(r'C:\Windows\Fonts\Deng.ttf', 60)
font_start_text = pygame.font.Font(r'C:\Windows\Fonts\Deng.ttf', 30)
font_text = pygame.font.Font(r'C:\Windows\Fonts\Deng.ttf', 10)
WIDTH, HEIGHT = 1280, 720 #pygame.display.get_desktop_sizes()[0]
planet_size = (100, 100)
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

BUILDINGS = {
    "运输": ["传送带", "星际传输枢纽"],
    "储存": ["储液罐", "储物仓",],
    "电力": ["地热发电机", "太阳能电池板", "核聚变发电机", "风力发电机"],
    "生产": ["组装机", "熔炉"],
    "采矿": ["采矿机", "液体抽取器"],
    "研究": ["研究站"]
    }
LANDS = {
    "景观": ["水", "沙", "土"],
    "矿物": ["煤矿", "石油", "石矿", "硅矿", "钛矿", "铁矿", "铜矿"]
    }
STARS = {
    9: "白矮星",
    8: "中子星",
    7: "黑洞",
    10: "O",
    11: "B",
    12: "A",
    13: "F",
    14: "G",
    15: "K",
    16: "M"}
COLORS = {
    "bar":(220, 250, 250, 50),
    "grid":(50, 50, 50),
    "player":(224, 255, 255),
    "text_cap":(120, 120, 120),
    "text_active":(255, 255, 0),
    "input_border_inactive":(100, 180, 255),
    "input_border_active":(70, 70, 100),
    "input_bg":(25, 25, 40),
    "caption":(200, 200, 200),
    "button":(120, 190, 255),
    "button_hover":(20, 200, 200),
    "button_active":(200, 200, 20),
    "button_bg":(95, 255, 255),
    "bg":(0, 0, 0)
    }
SUFFIXES = [
    'on', 'us', 'ia', 'is', 'es', 'ix', 'ax', 'or', 'a', 'ra', 'es', 
    'on', 'um', 'en', 'os', 'an', 'ra', 'as', 'or', 'ion', 'ara', 
    'ope', 'eus', 'ium', 'ara', 'elea', 'ope', 'eon', 'ara', 'ides'
]#恒星系后缀
IMGS = {}
for category, items in BUILDINGS.items():
    IMGS[category] = []  # 为每个类别创建图片列表
    for item in items:
        try:
            img = pygame.image.load(f"./pictures/{item}.png")
            IMGS[category].append(img)
        except:
            print(f"警告: 无法加载 {item} 图片")

class CreateMap:
    def __init__(self):
        self.size = 0
        self.seed = 0
        np.random.seed(self.seed)
        self.type_grid = np.zeros(shape=(self.size*2+1, self.size*2+1, self.size*2+1), dtype=int)
        self.name_grid = np.empty(shape=(self.size*2+1, self.size*2+1, self.size*2+1), dtype=object)
        self.name_grid.fill('')
        self.Map = {}

    def Create(self, t=0):
        name_length = np.random.randint(5, 10)
        if np.random.random() < 0.3:
            suffix = np.random.choice([s for s in SUFFIXES if len(s)==2])
        else:
            suffix = np.random.choice(SUFFIXES)
        main_length = name_length - len(suffix)
        main_part = ''.join(
            np.random.choice(
                list(string.ascii_lowercase), 
                main_length
            )
        )
        name = main_part + suffix
        if name[0] in 'aeiou':
            name = np.random.choice(list('bcdfghjklmnpqrstvwxyz')) + name[1:]
        
        x = np.random.randint(0, self.size*2+1)
        y = np.random.randint(0, self.size*2+1)
        z = np.random.randint(0, self.size*2+1)
        if t == 0:
            t = np.random.choice([10, 11, 12, 13, 14, 15, 16], p=[0.005, 0.015, 0.03, 0.05, 0.1, 0.3, 0.5])
        return x, y, z, t, name

    def CreateMap(self):
        _, _, _, center_type, center_name = self.Create(t=14)
        self.type_grid[self.size, self.size, self.size] = center_type
        self.name_grid[self.size, self.size, self.size] = center_name  
        m = [(self.size, self.size, self.size)]
        l = len(m)
        while l < self.size:
            if l == 1:
                x, y, z, t, n = self.Create(t=9)
            elif l == 2:
                x, y, z, t, n = self.Create(t=7)
            elif l//32*32 == l:
                x, y, z, t, n = self.Create(t=8)
            else:
                x, y, z, t, n = self.Create()
            if (x, y, z) not in m:
                n = n.capitalize()
                self.type_grid[x, y, z] = t
                self.name_grid[x, y, z] = n
                m.append((x, y, z))
                l = len(m)

    def Rebuild(self, size, seed):
        if self.size != int(size) and int(size) >= 32 and int(size) <= 64:
            self.size = int(size)
            self.type_grid = np.zeros(shape=(self.size*2+1, self.size*2+1, self.size*2+1), dtype=int)
            self.name_grid = np.empty(shape=(self.size*2+1, self.size*2+1, self.size*2+1), dtype=object)
            self.name_grid.fill('')
        if self.seed != int(seed):
            self.seed = int(seed)
            np.random.seed(self.seed)

    def Save(self):
        self.MAP = {}
        for x in range(self.type_grid.shape[0]):
            for y in range(self.type_grid.shape[1]):
                for z in range(self.type_grid.shape[2]):
                    if self.type_grid[x, y, z] != 0:
                        self.MAP[(x, y, z)] = [self.type_grid[x, y, z], self.name_grid[x, y, z]]

class InputBox:
    def __init__(self, x, y, width, height, max_len):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = COLORS['input_border_inactive']
        self.text = ''
        self.text_surface = font_start_text.render('', True, COLORS['text_cap'])
        self.active = False
        self.cursor_visible = True
        self.cursor_timer = time.time()
        self.cursor_position = 0
        self.text_surface_width = 0
        self.max_len = max_len

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.active = True
                self.color = COLORS['input_border_active']
            else:
                self.active = False
                self.color = COLORS['input_border_inactive']

        if event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_DELETE or event.key == pygame.K_BACKSPACE:
                if self.cursor_position > 0:
                    self.text = self.text[:self.cursor_position-1] + self.text[self.cursor_position:]
                    self.cursor_position -= 1
            elif event.key == pygame.K_RETURN:
                return self.text
            elif event.key == pygame.K_LEFT:
                if self.cursor_position > 0:
                    self.cursor_position -= 1
            elif event.key == pygame.K_RIGHT:
                if self.cursor_position < len(self.text):
                    self.cursor_position += 1
            else:
                if event.unicode.isdigit() and len(self.text) < self.max_len:
                    self.text = self.text[:self.cursor_position] + event.unicode + self.text[self.cursor_position:]
                    self.cursor_position += 1

            self.text_surface = font_text.render(self.text, True, COLORS['text_cap'])
            self.text_surface_width = self.text_surface.get_width()
            self.cursor_timer = time.time()
            self.cursor_visible = True
        return None
    
    def update(self):
        if time.time()-self.cursor_timer > 0.5:
            self.cursor_visible = not self.cursor_visible
            self.cursor_timer = time.time()

    def draw(self):
        pygame.draw.rect(screen, COLORS['input_bg'], self.rect)
        pygame.draw.rect(screen, self.color, self.rect, 2)
        screen.blit(self.text_surface, (self.rect.x+5, self.rect.y+5))
        if self.active and self.cursor_visible:
            cursor_x = self.rect.x + 5 + font_text.size(self.text[:self.cursor_position])[0]
            pygame.draw.line(screen, COLORS['text_active'], 
                            (cursor_x, self.rect.y + 5),
                            (cursor_x, self.rect.y + self.rect.height - 5), 2)
        
class Button:
    def __init__(self, x, y, width, height, text, action=True):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.action = action
        self.hovered = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.hovered and self.action:
                return self.text
        return None

    def draw(self, font):
        color = COLORS['button_hover'] if self.hovered else COLORS['button_bg']
        pygame.draw.rect(screen, color ,self.rect, border_radius=5)
        pygame.draw.rect(screen, COLORS['button'], self.rect, 2, border_radius=5)
        text_surface = font.render(self.text, True, COLORS['text_cap'])
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)

class Player:
    def __init__(self, color=COLORS['player']):
        self.rect = pygame.rect.Rect((0, 0), (20, 20))
        self.color = color

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w or event.key == pygame.K_UP:
                self.rect.y -= 10
            if event.key == pygame.K_s or event.key == pygame.K_DOWN:
                self.rect.y += 10
            if event.key == pygame.K_a or event.key == pygame.K_LEFT:
                self.rect.x -= 10
            if event.key == pygame.K_d or event.key == pygame.K_RIGHT:
                self.rect.x += 10
            self.rect.x = max(0, min(self.rect.x, WIDTH - self.rect.width))
            self.rect.y = max(0, min(self.rect.y, HEIGHT - self.rect.height))

    def draw(self, rect):
        pygame.draw.rect(screen, self.color, rect)

class Camera:
    def __init__(self, map_size):
        map_width, map_height = map_size[0]*20, map_size[1]*20
        self.rect = pygame.rect.Rect((0, 0), (WIDTH, HEIGHT))
        self.map_width = map_width
        self.map_height = map_height
        self.target = None
        self.speed = 20
        self.zoom = 1.0
        
    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if (event.key == pygame.K_w or event.key == pygame.K_UP) and event.mod == pygame.KMOD_CTRL:
                self.rect.y -= self.speed
            elif (event.key == pygame.K_s or event.key == pygame.K_DOWN) and event.mod == pygame.KMOD_CTRL:
                self.rect.y += self.speed
            elif (event.key == pygame.K_a or event.key == pygame.K_LEFT) and event.mod == pygame.KMOD_CTRL:
                self.rect.x -= self.speed
            elif (event.key == pygame.K_d or event.key == pygame.K_RIGHT) and event.mod == pygame.KMOD_CTRL:
                self.rect.x += self.speed
            elif event.key == pygame.K_PAGEUP:
                self.zoom = min(2.0, self.zoom + 0.1)
            elif event.key == pygame.K_PAGEDOWN:
                self.zoom = max(0.5, self.zoom - 0.1)
            self.rect.x = max(0, min(self.rect.x, self.map_width - self.rect.width))
            self.rect.y = max(0, min(self.rect.y, self.map_height - self.rect.height))
    
    def update(self, target=None):
        """更新相机位置，跟随目标（如果设置了目标）"""
        if target:
            self.rect.x = target.rect.x - WIDTH // 2
            self.rect.y = target.rect.y - HEIGHT // 2
            self.rect.x = max(0, min(self.rect.x, self.map_width - self.rect.width))
            self.rect.y = max(0, min(self.rect.y, self.map_height - self.rect.height))
    
    def apply(self, entity):
        """应用相机偏移到实体，返回屏幕坐标"""
        scaled_rect = entity.rect.copy()
        scaled_rect.x = (scaled_rect.x - self.rect.x) * self.zoom
        scaled_rect.y = (scaled_rect.y - self.rect.y) * self.zoom
        scaled_rect.width *= self.zoom
        scaled_rect.height *= self.zoom
        return scaled_rect
    
    def apply_point(self, point):
        """应用相机偏移到点，返回屏幕坐标"""
        return (point[0] - self.rect.x, point[1] - self.rect.y)

screen.fill(COLORS['bg'])
g_m = 'start'    # 游戏状态（开始菜单 start，创建游戏 new，存档选择 save，设置菜单 setting，游戏中 playing）
past_mode = None
buttons = []
inputboxes = []
galaxy = None
past_planet_map = np.zeros(shape=planet_size)
now_planet_map = np.zeros(shape=planet_size)

def draw_start():
    global WIDTH, HEIGHT
    text = font_cap.render('？？？', True, COLORS['caption'])
    text_cap_rect = text.get_rect(center=(WIDTH//2, HEIGHT//5))
    screen.blit(text, text_cap_rect)
    buttons = []
    for i in range(1, 6):
        t = ['新游戏', '继续游戏', '存档', '设置', '退出'][i-1]
        a = Button(WIDTH//8, HEIGHT//6*i, 200, 60, t)
        buttons.append(a)
    return buttons, []

def draw_new():
    global WIDTH, HEIGHT, past_mode
    past_mode = 'new'
    buttons = []
    inputboxes = []
    for i in range(1, 3):
        t = ['星系规模', '星系种子'][i-1]
        a = Button(WIDTH//2-200, HEIGHT//10*i, 200, 40, t, action=False)
        b = InputBox(WIDTH//2, HEIGHT//10*i, 400, 40, i*2**i)
        buttons.append(a)
        inputboxes.append(b)
    for i in range(1, 3):
        t = ['开始游戏', '返回'][i-1]
        a = Button(WIDTH//3*i, HEIGHT//8*7, 200, 60, t)
        buttons.append(a)
    return buttons, inputboxes

def draw_save():
    global WIDTH, HEIGHT, past_mode
    past_mode = 'save'
    buttons = []
    files = os.listdir('./save')
    for i in range(1, len(files)+1):
        t = files[i-1]
        a = Button(WIDTH//8, HEIGHT//6*i, 200, 50, t)
        buttons.append(a)
    a = Button(WIDTH//10, HEIGHT//10*9, 100, 60, '返回')
    buttons.append(a)
    return buttons, []

def draw_setting():
    global WIDTH, HEIGHT, past_mode
    past_mode = 'setting'
    buttons = []
    for i in range(1, 4):
        t = ['显示', '声音', '游戏'][i-1]
        a = Button(WIDTH//8*i, HEIGHT//10, 200, 50, t)
        buttons.append(a)
    a = Button(WIDTH//10, HEIGHT//10*9, 100, 60, '返回')
    buttons.append(a)
    return buttons, []

def draw_playing():
    global WIDTH, HEIGHT, COLORS
    buttons = []
    for i in range(1, 7):
        t = ['运输', '储存', '电力', '生产', '采矿', '研究'][i-1]
        a = Button(WIDTH//8*i, HEIGHT//5*4, 100, 60, t)
        buttons.append(a)
    build_bar_height = HEIGHT // 10 * 3
    build_bar_rect = pygame.Rect((0, HEIGHT-build_bar_height), (WIDTH, build_bar_height))
    build_bar_surface = pygame.Surface((WIDTH, build_bar_height), pygame.SRCALPHA)
    build_bar_surface.fill(COLORS['bar'])
    return buttons, [build_bar_surface, build_bar_rect]

def draw_grid():
    global WIDTH, HEIGHT
    start_x = -(camera.rect.x % 20)
    start_y = -(camera.rect.y % 20)
    for x in range(0, WIDTH, 20):
        pygame.draw.line(screen, COLORS['grid'], (x, 0), (x, HEIGHT))
    for y in range(0, HEIGHT, 20):
        pygame.draw.line(screen, COLORS['grid'], (0, y), (WIDTH, y))

def save(filename, player, camera, seed, size, Map):
    save_dir = './save/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(save_dir, str(filename))
    player_data = f"{player.rect.x},{player.rect.y},{player.rect.width},{player.rect.height}\n"
    color_data = f"{player.color[0]},{player.color[1]},{player.color[2]}\n"
    camera_data = f"{camera.rect.x},{camera.rect.y},{camera.rect.width},{camera.rect.height},{camera.zoom}\n"
    seed_data = f"{seed}\n"
    size_data = f"{size}\n"
    map_data = ""
    for pos, info in Map.items():
        x, y, z = pos
        t, name = info
        map_data += f"{x},{y},{z},{t},{name}\n"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(player_data)
        f.write(color_data)
        f.write(camera_data)
        f.write(seed_data)
        f.write(size_data)
        f.write(map_data)

def load(filename, player, camera):
    filename = './save/' + filename
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            player_data = lines[0].strip().split(',')
            player.rect.x = int(player_data[0])
            player.rect.y = int(player_data[1])
            player.rect.width = int(player_data[2])
            player.rect.height = int(player_data[3])
            color_data = lines[1].strip().split(',')
            player.color = (int(color_data[0]), int(color_data[1]), int(color_data[2]))
            camera_data = lines[2].strip().split(',')
            camera.rect.x = int(camera_data[0])
            camera.rect.y = int(camera_data[1])
            camera.rect.width = int(camera_data[2])
            camera.rect.height = int(camera_data[3])
            camera.zoom = float(camera_data[4])
            seed = int(lines[3].strip())
            size = int(lines[4].strip())
            Map = {}
            for line in lines[5:]:
                data = line.strip().split(',')
                if len(data) < 5:
                    continue
                x, y, z = int(data[0]), int(data[1]), int(data[2])
                t = int(data[3])
                name = ','.join(data[4:])  # 名字可能包含逗号
                Map[(x, y, z)] = [t, name]
            return seed, size, Map
    except Exception as e:
        print(f"加载失败: {e}")
        return None, None, None

buttons, inputboxes = draw_start()
bar_buttons = []
bar_imgs = []

running = True
while running:
    screen.fill(COLORS['bg'])
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            if g_m == 'playing':
                if galaxy:
                    save('auto save (past)', player, camera, galaxy.seed, galaxy.size, galaxy.MAP)
                running = False
            else:
                running = False

        if g_m == 'playing':
            player.handle_event(event)
            camera.handle_event(event)

        for ib in inputboxes:
            ib.handle_event(event)

        action_taken = False
        for b in buttons:
            action = b.handle_event(event)
            if action:
                action_taken = True
                if g_m == 'playing':
                    bar_buttons = []
                    bar_imgs = []
                    texts = BUILDINGS[action]
                    for text in texts:
                        i = texts.index(text) + 1
                        a = Button(WIDTH//8*i, HEIGHT//16*15, 100, 30, text)
                        img = IMGS[action][i-1]
                        img_rect = img.get_rect(center=(WIDTH//8*i+40, HEIGHT//16*15-40))
                        bar_imgs.append((img, img_rect))
                        bar_buttons.append(a)
                        
                if g_m == 'start':
                    if action == '新游戏':
                        g_m = 'new'
                    elif action == '继续游戏':
                        try:
                            player = Player()
                            camera = Camera(planet_size)
                            seed_value, size_value, Map = load('_last_exit_', player, camera)
                            galaxy = CreateMap()
                            galaxy.seed = seed_value
                            galaxy.size = size_value
                            galaxy.MAP = Map
                            g_m = 'playing'
                        except:
                            print('please start game first')
                    elif action == '存档':
                        g_m = 'save'
                    elif action == '设置':
                        g_m = 'setting'
                    elif action == '退出':
                        running = False
                        
                elif g_m == 'new':
                    if action == '开始游戏':
                        seed_value = inputboxes[1].text
                        size_value = inputboxes[0].text
                        if seed_value and size_value:
                            if galaxy is None:
                                galaxy = CreateMap()
                            galaxy.Rebuild(size_value, seed_value)
                            galaxy.CreateMap()
                            galaxy.Save()
                            player = Player()
                            camera = Camera(planet_size)
                            g_m = 'playing'
                            inputboxes = []
                    elif action == '返回':
                        g_m = 'start'

                elif g_m == 'save':
                    if action == '显示':
                        pass
                    elif action == '声音':
                        pass
                    elif action == '游戏':
                        pass
                    elif action == '返回':
                        g_m = 'start'

                elif g_m == 'setting':
                    if action == '':
                        pass
                    elif action == '返回':
                        g_m = 'start'

                elif g_m == 'con':
                    pass

                if action_taken:
                    if g_m == 'start':
                        buttons, inputboxes = draw_start()
                    elif g_m == 'new':
                        buttons, inputboxes = draw_new()
                        if galaxy is None:
                            galaxy = CreateMap()
                    elif g_m == 'save':
                        buttons, inputboxes = draw_save()
                    elif g_m == 'setting':
                        buttons, inputboxes = draw_setting()
                    elif g_m == 'playing':
                        buttons, bar_list = draw_playing()

    if g_m == 'playing':
        camera.update(player)
    for ib in inputboxes:
        ib.update()
    
    if g_m == 'start':
        text = font_cap.render('？？？', True, COLORS['caption'])
        text_cap_rect = text.get_rect(center=(WIDTH//2, HEIGHT//5))
        screen.blit(text, text_cap_rect)
    elif g_m == 'playing':
        draw_grid()
        player_rect = camera.apply(player)
        player.draw(player_rect)
        screen.blit(bar_list[0], bar_list[1])
        for bb in bar_buttons:
            bb.draw(font_text)
        for bi in bar_imgs:
            screen.blit(bi[0], bi[1])
    for ib in inputboxes:
        ib.draw()
    for b in buttons:
        b.draw(font_start_text)
                            
    pygame.display.update()
    clock.tick(60)

pygame.quit()
sys.exit()
