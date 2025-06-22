import os
import pygame
import random
import string
import sys
import time
import numpy as np

pygame.init()

desktop_sizes = pygame.display.get_desktop_sizes()

WIDTH, HEIGHT = 1280, 720 #desktop_sizes[0]
Deng_path = r'C:\Windows\Fonts\Deng.ttf'
clock = pygame.time.Clock()
screen = pygame.display.set_mode((WIDTH, HEIGHT))

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
    "button_hover":(95, 255, 255),
    "button_bg":(20, 200, 200),
    "bg":(0, 0, 0)
    }
FONTS = {
    f'Deng{size}':pygame.font.Font(Deng_path, size) for size in range(1, 61)
    }
GOODS = {
    'building':{"传送带", "星际传输枢纽",
                "储液罐", "储物仓",
                "地热发电机", "太阳能电池板", "核聚变发电机", "风力发电机",
                "组装机", "熔炉",
                "采矿机", "液体抽取器",
                "研究站"},
    'good':{"煤炭", "铁原矿", "铜原矿", "石原矿", "硅石", "钛原矿", "石油", "氢", "水", "硫酸",
            "石墨", "燃烧单元", "铁板", "磁铁", "铜板", "砖", "玻璃", "高纯硅版", "钛板", "精炼油",
            "金刚石", "齿轮", "钢材", "单晶硅",
            "磁线圈", "电路板", "集成电路", ""}
    }
IMGS = {f'{label}_{name}':pygame.image.load(f'./pictures/{name}.png') for label, names in BUILDINGS.items() for name in names}

class GameState:
    start, save, setting, playing, menu = range(5)

class PlanetMap:
    def __init__(self, size):
        self.player_map = np.zeros(shape=size, dtype=bool)
        self.base_map = np.zeros(shape=size, dtype=int)
        self.iron_map = np.zeros(shape=size, dtype=list)
        self.building_map = np.zeros(shape=size, dtype=list)
        self.map = [self.player_map, self.base_map, self.building_map,  self.iron_map]
        self.view = [0, 0, WIDTH, HEIGHT]
        self.cell_size = 20
        self.size = size
    
    def draw(self):
        start_x = self.view[0] // self.cell_size * self.cell_size
        start_y = self.view[1] // self.cell_size * self.cell_size       
        end_x = min(self.view[0] + self.view[2], self.size[0] * self.cell_size)
        end_y = min(self.view[1] + self.view[3], self.size[1] * self.cell_size)
        
        for x in range(start_x, end_x, self.cell_size):
            for y in range(start_y, end_y, self.cell_size):
                screen_x = x - self.view[0]
                screen_y = y - self.view[1]
                rect = pygame.Rect(screen_x, screen_y, self.cell_size, self.cell_size)
                pygame.draw.rect(screen, COLORS["grid"], rect, 1)
    
    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.mod == pygame.KMOD_LCTRL:
                if event.key == pygame.K_w or event.key == pygame.K_UP:
                    if self.view[1] >= 10:
                        self.view[1] -= 10
                if event.key == pygame.K_s or event.key == pygame.K_DOWN:
                    if self.view[1] <= HEIGHT-10:
                        self.view[1] += 10
                if event.key == pygame.K_a or event.key == pygame.K_LEFT:
                    if self.view[0] >= 10:
                        self.view[0] -= 10
                if event.key == pygame.K_d or event.key == pygame.K_RIGHT:
                    if self.view[0] <= WIDTH-10:
                        self.view[0] += 10

class Player:
    def __init__(self, color=(224, 255, 255)):
        self.color = color
        self.x = 0
        self.y = 0
        self.size = 20
        self.speed = 10
    
    def change(self, color):
        if self.color != color:
            self.color = color
    
    def draw(self, view):
        screen_x = self.x - view[0]
        screen_y = self.y - view[1]
        if (0 <= screen_x <= view[2] and 0 <= screen_y <= view[3]):
            rect = pygame.Rect(screen_x, screen_y, self.size, self.size)
            pygame.draw.rect(screen, self.color, rect)
    
    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.mod == pygame.KMOD_NONE:
                if event.key == pygame.K_w or event.key == pygame.K_UP:
                    if self.y >= self.speed:
                        self.y -= self.speed
                if event.key == pygame.K_s or event.key == pygame.K_DOWN:
                    if self.y <= HEIGHT-self.speed:
                        self.y += self.speed
                if event.key == pygame.K_a or event.key == pygame.K_LEFT:
                    if self.x >= self.speed:
                        self.x -= self.speed
                if event.key == pygame.K_d or event.key == pygame.K_RIGHT:
                    if self.x <= WIDTH-self.speed:
                        self.x += self.speed

class Button:
    def __init__(self, font, text, center=(0, 0), pos=(0, 0)):
        self.font = font
        self.text = text
        self.center:tuple = center
        self.pos:tuple = pos
        self.ishorved = False
        self.rect = None
    
    def draw(self):
        self.color = COLORS["button_bg"] if not self.ishorved else COLORS["button_hover"]
        text_surface = self.font.render(self.text, True, self.color, COLORS['bg'])
        size:tuple = self.font.size(self.text)
        self.rect = pygame.rect.Rect(self.pos, size)
        if self.center != (0, 0):
            self.rect = text_surface.get_rect(center=self.center)
        else:
            self.rect = pygame.Rect(self.pos, size)
        screen.blit(text_surface, self.rect)

    def update(self, mouse_pos):
        if self.rect:
            self.ishorved = self.rect.collidepoint(mouse_pos)
    
    def collided(self, event):
        if self.ishorved:
            return self.text

class Building:
    def __init__(self, type_:str, name:str, pos:tuple):
        self.type = BUILDINGS[type_]
        self.name = BUILDINGS[type_][name]
        self.img = IMGS[f'{self.type}_{self.name}']
        self.rect = pygame.rect.Rect(pos, (20, 20))

start_buttons = [Button(FONTS['Deng40'], ["新游戏", "存档", "设置", "退出"][i], pos=(WIDTH//10, HEIGHT//8*(i+1))) for i in range(4)]
#new_buttons = [Button(FONTS['Deng40'], ["开始", "返回"][i], pos=(WIDTH//4*(i+1), HEIGHT//16*15)) for i in range(2)]
save_buttons = [
    Button(FONTS['Deng20'], "返回", center=(WIDTH//10*9, HEIGHT//20))]
setting_buttons = [
    Button(FONTS['Deng20'], "返回", center=(WIDTH//10*9, HEIGHT//20*19))]
menu_buttons = [Button(FONTS['Deng40'], ["继续", "读档", "存档", "设置", "退出"][i], center=(WIDTH//2, HEIGHT//8*(i+2))) for i in range(5)]
state_buttons = {
    GameState.start: start_buttons,
    GameState.save: save_buttons,
    GameState.setting: setting_buttons,
    GameState.menu: menu_buttons
}

current_state = GameState.start
past_state = None
pm = None
player = None
running = True
while running:
    action = []
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEMOTION:
            if current_state != 3:
                for b in state_buttons[current_state]:
                    b.update(event.pos)
        if event.type == pygame.MOUSEBUTTONDOWN:
            if current_state != 3:
                for b in state_buttons[current_state]:
                    action.append(b.collided(event.pos))
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                if current_state == GameState.playing:
                    current_state = GameState.menu
                    past_state = GameState.playing
                else:
                    running = False
            else:
                if pm:
                    pm.handle_event(event)
                if player:
                    player.handle_event(event)
    
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
            pass
        if "新游戏" in action:
            map_size = (100, 100)
            pm = PlanetMap(map_size)
            player = Player()
            current_state = GameState.playing
        if "返回" in action:
            if not past_state:
                current_state = GameState.start
            else:
                current_state = GameState.playing

    screen.fill(COLORS["bg"])
    if current_state != 3:
        for b in state_buttons[current_state]:
            b.draw()
    else:
        if pm:
            pm.draw()
        if player and pm:
            player.draw(pm.view)
    
    state_names = ["开始", "存档", "设置", "游戏中", "菜单"]
    state_text = FONTS['Deng20'].render(f"状态: {state_names[current_state]}", True, (255, 255, 255))
    screen.blit(state_text, (10, 10))
    if player and pm:
        pos_text = FONTS['Deng20'].render(f'角色：{player.x},{player.y} 视图：{pm.view[0]},{pm.view[1]}', True, (255, 255, 255))
        screen.blit(pos_text, (10, 30))
    
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()
