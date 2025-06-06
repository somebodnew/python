import random
import time
import PySimpleGUI as sg
from enum import Enum, auto
from typing import List, Tuple, Dict, Optional, Any
from copy import deepcopy

class TimeOfDay(Enum):
    MORNING = auto()
    DAY = auto()
    EVENING = auto()
    NIGHT = auto()

class EcosystemMeta(type):
    
    _registry: Dict[str, Dict[str, type]] = {
        'plants': {},
        'animals': {}
    }

    def __new__(cls, name, bases, namespace):
        new_class = super().__new__(cls, name, bases, namespace)
        
        if 'Plant' in [b.__name__ for b in bases] and name != 'Plant':
            cls._registry['plants'][name] = new_class
            cls._configure_plant(new_class, namespace.get('behavior', {}))
        
        elif 'Animal' in [b.__name__ for b in bases] and name != 'Animal':
            cls._registry['animals'][name] = new_class
            cls._configure_animal(new_class, namespace.get('behavior', {}))
        
        return new_class

    @classmethod
    def _configure_plant(cls, target_class: type, behavior: dict):
        def spread(self):
            if not self.is_active(self.world.time_of_day):
                return

            neighbors = self.world.get_neighbors(self.x, self.y)
            for nx, ny in random.sample(neighbors, k=len(neighbors)):
                target = self.world.grid[ny][nx]

                if target is None and random.random() < behavior.get('spread_chance', 0.3):
                    if self.world.add_entity(self.__class__(), nx, ny):
                        self.energy -= 10
                elif (isinstance(target, Plant) and 
                      self.is_active(self.world.time_of_day) and 
                      not target.is_active(self.world.time_of_day) and 
                      random.random() < behavior.get('competitive_chance', 0.5)):
                    target.die()

        target_class.spread = spread
        
        def is_active(self, time_of_day: TimeOfDay):
            self.active = time_of_day in behavior.get('active_times', [])
            return self.active
        target_class.is_active = is_active
        
        return target_class

    @classmethod
    def _configure_animal(cls, target_class: type, behavior: dict):
        def move(self):
            if self.is_sleeping():
                return

            speed = behavior.get('base_speed', 1.0)
            if self.hunger > behavior.get('hunger_threshold', 50):
                speed *= behavior.get('hungry_speed_mod', 0.8)

            possible_moves = [
                (x, y) for x, y in self.world.get_neighbors(self.x, self.y)
                if self.world.grid[y][x] is None
            ]
            
            if possible_moves and random.random() < speed:
                nx, ny = random.choice(possible_moves)
                self.world.move_entity(self, nx, ny)

        target_class.move = move
        
        def eat(self):
            if self.is_sleeping() or self.hunger < 20:
                return

            for food_type in behavior.get('food_types', []):
                for x, y in self.world.get_neighbors(self.x, self.y):
                    target = self.world.grid[y][x]
                    if isinstance(target, food_type):
                        nutrition = behavior.get('nutrition', 30)
                        self.hunger = max(0, self.hunger - nutrition)
                        self.energy += behavior.get('energy_gain', 20)
                        target.die()
                        return

        target_class.eat = eat
        
        def is_sleeping(self):
            return self.world.time_of_day in behavior.get('sleep_time', [])
        target_class.is_sleeping = is_sleeping
        
        # Инициализация радиуса обзора
        original_init = target_class.__init__
        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self.vision_radius = behavior.get('vision_radius', 1)
        target_class.__init__ = new_init
        
        return target_class

# Базовый класс растений
class Plant(metaclass=EcosystemMeta):
    def __init__(self):
        self.x = 0
        self.y = 0
        self.world = None
        self.energy = 100
        self.active = False

    def die(self):
        self.world.remove_entity(self)

# Базовый класс животных
class Animal(metaclass=EcosystemMeta):
    def __init__(self):
        self.x = 0
        self.y = 0
        self.world = None
        self.energy = 100
        self.hunger = 0
        self.vision_radius = 1  # По умолчанию

    def reproduce(self):
        if self.energy < 80 or random.random() > 0.1:
            return

        for x, y in self.world.get_neighbors(self.x, self.y):
            if self.world.grid[y][x] is None:
                newborn = self.__class__()
                if self.world.add_entity(newborn, x, y):
                    self.energy -= 40
                    return

    def die(self):
        self.world.remove_entity(self)

# Конкретные классы растений
class Lumiere(Plant):
    behavior = {
        'active_times': [TimeOfDay.DAY],
        'spread_chance': 0.4,
        'competitive_chance': 0.6
    }

class Obscurite(Plant):
    behavior = {
        'active_times': [TimeOfDay.NIGHT],
        'spread_chance': 0.5,
        'competitive_chance': 0.7
    }

class Demi(Plant):
    behavior = {
        'active_times': [TimeOfDay.MORNING, TimeOfDay.EVENING],
        'spread_chance': 0.3,
        'competitive_chance': 0.5
    }

# Конкретные классы животных
class Pauvre(Animal):
    behavior = {
        'food_types': [Lumiere],
        'sleep_time': [TimeOfDay.NIGHT],
        'base_speed': 1.0,
        'hungry_speed_mod': 0.7,
        'hunger_threshold': 60,
        'nutrition': 40,
        'energy_gain': 25,
        'vision_radius': 2 
    }

class Malheureux(Animal):
    behavior = {
        'food_types': [Demi, Obscurite, Pauvre],
        'sleep_time': [TimeOfDay.DAY],
        'base_speed': 0.9,
        'hungry_speed_mod': 0.6,
        'nutrition': 50,
        'energy_gain': 30,
        'vision_radius': 3
    }

class World:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.grid = [[None for _ in range(width)] for _ in range(height)]
        self.hour = 0 
        self.time_of_day = self._calculate_time_of_day()
        self.entities = []
        self.snapshots = {} 

    def _calculate_time_of_day(self):
        if 6 <= self.hour < 12:
            return TimeOfDay.MORNING
        elif 12 <= self.hour < 18:
            return TimeOfDay.DAY
        elif 18 <= self.hour < 24:
            return TimeOfDay.EVENING
        else:
            return TimeOfDay.NIGHT

    def make_snapshot(self):
        snapshot = {
            'hour': self.hour,
            'time_of_day': self.time_of_day,
            'entities': [],
            'grid': [[None for _ in range(self.width)] for _ in range(self.height)]
        }
        
        for entity in self.entities:
            ent_data = {
                'class': entity.__class__.__name__,
                'x': entity.x,
                'y': entity.y,
                'energy': entity.energy
            }
            
            if isinstance(entity, Animal):
                ent_data.update({
                    'hunger': entity.hunger,
                    'vision_radius': entity.vision_radius
                })
            
            snapshot['entities'].append(ent_data)
            
            # Сохраняем положение в сетке
            snapshot['grid'][entity.y][entity.x] = ent_data
        
        self.snapshots[self.hour] = snapshot
        return snapshot

    def restore_snapshot(self, hour: int):
        if hour not in self.snapshots:
            return False
            
        snapshot = self.snapshots[hour]
        self.hour = snapshot['hour']
        self.time_of_day = snapshot['time_of_day']
        
        self.grid = [[None for _ in range(self.width)] for _ in range(self.height)]
        self.entities = []
        
        class_to_entity = {
            'Lumiere': Lumiere,
            'Obscurite': Obscurite,
            'Demi': Demi,
            'Pauvre': Pauvre,
            'Malheureux': Malheureux,
            }

        for ent_data in snapshot['entities']:
            class_name = ent_data['class']
            
            
            if class_name in class_to_entity:
                entity = class_to_entity[class_name]()
                
            entity.x = ent_data['x']
            entity.y = ent_data['y']
            entity.energy = ent_data['energy']
            entity.world = self
            
            if isinstance(entity, Animal):
                entity.hunger = ent_data['hunger']
                entity.vision_radius = ent_data.get('vision_radius', 1)
            
            self.grid[entity.y][entity.x] = entity
            self.entities.append(entity)
        
        return True

    def add_entity(self, entity, x: int, y: int) -> bool:
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False

        if self.grid[y][x] is not None:
            return False

        entity.x = x
        entity.y = y
        entity.world = self
        self.grid[y][x] = entity
        self.entities.append(entity)
        return True

    def remove_entity(self, entity):
        if entity in self.entities:
            self.entities.remove(entity)
        if 0 <= entity.x < self.width and 0 <= entity.y < self.height:
            self.grid[entity.y][entity.x] = None

    def move_entity(self, entity, new_x: int, new_y: int):
        self.grid[entity.y][entity.x] = None
        entity.x = new_x
        entity.y = new_y
        self.grid[new_y][new_x] = entity

    def get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        return [
            (x + dx, y + dy)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            if (dx != 0 or dy != 0) 
            and 0 <= x + dx < self.width 
            and 0 <= y + dy < self.height
        ]

    def update_time(self):
        self.hour = (self.hour + 1) % 24
        self.time_of_day = self._calculate_time_of_day()
        self.make_snapshot()

    def step(self):
        self.update_time()
        
        for entity in self.entities.copy():
            if isinstance(entity, Plant):
                entity.spread()

            elif isinstance(entity, Animal):
                entity.move()
                entity.eat()
                entity.reproduce()
                
                entity.hunger += 5
                if entity.hunger > 100:
                    entity.die()
                elif entity.energy <= 0:
                    entity.die()

def get_stats(world: World) -> str:
    counts = {
        'Lumiere': 0,
        'Obscurite': 0,
        'Demi': 0,
        'Pauvre': 0,
        'Malheureux': 0
    }
    vision_radii = []
    
    for entity in world.entities:
        if isinstance(entity, Lumiere):
            counts['Lumiere'] += 1
        elif isinstance(entity, Obscurite):
            counts['Obscurite'] += 1
        elif isinstance(entity, Demi):
            counts['Demi'] += 1
        elif isinstance(entity, Pauvre):
            counts['Pauvre'] += 1
            vision_radii.append(entity.vision_radius)
        elif isinstance(entity, Malheureux):
            counts['Malheureux'] += 1
            vision_radii.append(entity.vision_radius)
    
    avg_vision = sum(vision_radii) / len(vision_radii) if vision_radii else 0
    
    stats_text = (
        f"Lumiere: {counts['Lumiere']}\n"
        f"Obscurite: {counts['Obscurite']}\n"
        f"Demi: {counts['Demi']}\n"
        f"Pauvre: {counts['Pauvre']}\n"
        f"Malheureux: {counts['Malheureux']}\n"
        f"Средний радиус обзора: {avg_vision:.2f}\n"
        f"Текущий час: {world.hour}:00"
    )
    
    return stats_text

def draw_world(world: World, graph_elem, selected_animal: Optional[Animal] = None):
    graph_elem.erase()
    
    colors = {
        'Lumiere': '#FFFF00',    # желтый
        'Obscurite': '#0000FF',  # синий
        'Demi': '#808080',       # серый
        'Pauvre': '#FFFF00',     # желтый
        'Malheureux': '#800080'  # фиолетовый
    }
    
    for y in range(world.height):
        for x in range(world.width):
            entity = world.grid[y][x]
            graph_elem.draw_rectangle(
                        (x, y), (x+1, y+1)
                    )
            if entity:
                class_name = entity.__class__.__name__
                if class_name in ['Lumiere', 'Obscurite', 'Demi']:
                    graph_elem.draw_rectangle(
                        (x, y), (x+1, y+1), 
                        fill_color=colors[class_name]
                    )
                elif class_name == 'Pauvre':
                    graph_elem.draw_circle(
                        (x+0.5, y+0.5), 0.2, 
                        fill_color=colors[class_name]
                    )
                elif class_name == 'Malheureux':
                    graph_elem.draw_circle(
                        (x+0.5, y+0.5), 0.2, 
                        fill_color=colors[class_name]
                    )
    
    if selected_animal:
        graph_elem.draw_circle(
            (selected_animal.x + 0.5, selected_animal.y + 0.5),
            selected_animal.vision_radius,
            line_color='#888888',
            line_width=2
        )

def reset_world(width: int, height: int) -> World:
    world = World(width, height)
    
    for _ in range(20):
        x, y = random.randint(0, width-1), random.randint(0, height-1)
        plant_type = random.choice([Lumiere, Obscurite, Demi])
        world.add_entity(plant_type(), x, y)
    
    for _ in range(5):
        x, y = random.randint(0, width-1), random.randint(0, height-1)
        world.add_entity(Pauvre(), x, y)
    
    for _ in range(3):
        x, y = random.randint(0, width-1), random.randint(0, height-1)
        world.add_entity(Malheureux(), x, y)
    
    world.make_snapshot()
    return world

def main():
    WORLD_WIDTH = 15
    WORLD_HEIGHT = 20
    
    world = reset_world(WORLD_WIDTH, WORLD_HEIGHT)
    sg.theme('LightBrown4')
    layout = [
    [
        sg.Column([  
            [
                sg.Text('Время:'), 
                sg.Slider(
                    range=(0, 23), 
                    default_value=world.hour, 
                    orientation='h', 
                    size=(40, 15),
                    key='-HOUR-', 
                    enable_events=True,
                    resolution=1,
                    tick_interval=4
                )
            ],
            [
                sg.Button('Старт/Пауза', key='-RUN-'),
                sg.Button('Шаг', key='-STEP-'),
                sg.Button('Сброс', key='-RESET-'),
                sg.Text('Скорость:'),
                sg.Slider(
                    range=(1, 10), 
                    default_value=1, 
                    orientation='h', 
                    size=(15, 15),
                    key='-SPEED-', 
                    enable_events=True
                )
            ],
            [
                sg.Graph(
                    canvas_size=(600, 400),
                    graph_bottom_left=(0, 0),
                    graph_top_right=(WORLD_WIDTH, WORLD_HEIGHT),
                    key='-MAP-',
                    enable_events=True,
                    background_color='white'
                )
            ]
        ]),
        sg.Column([  
            [
                sg.Multiline(
                    size=(30, 25), 
                    key='-STATS-', 
                    disabled=True,
                    autoscroll=True
                )
            ]
        ])
    ]
]
    
    window = sg.Window(
        'Симуляция Экосистемы', 
        layout, 
        finalize=True,
        resizable=True
    )
    
    map_graph = window['-MAP-']
    stats_elem = window['-STATS-']
    hour_slider = window['-HOUR-']
    
    running = False
    selected_animal = None
    last_update_time = time.time()
    simulation_speed = 5  
    
    draw_world(world, map_graph, selected_animal)
    stats_elem.update(get_stats(world))
    
    while True:
        event, values = window.read(timeout=100)
        
        if event == sg.WIN_CLOSED:
            break
            
        if event == '-RUN-':
            running = not running
            
        if event == '-STEP-':
            world.step()
            hour_slider.update(world.hour)
            draw_world(world, map_graph, selected_animal)
            stats_elem.update(get_stats(world))
            
        if event == '-RESET-':
            world = reset_world(WORLD_WIDTH, WORLD_HEIGHT)
            selected_animal = None
            hour_slider.update(world.hour)
            draw_world(world, map_graph, selected_animal)
            stats_elem.update(get_stats(world))
            
        if event == '-HOUR-':
            new_hour = int(values['-HOUR-'])
            if new_hour != world.hour:
                if world.restore_snapshot(new_hour):
                    world.hour = new_hour
                    world.time_of_day = world._calculate_time_of_day()
                    draw_world(world, map_graph, selected_animal)
                    stats_elem.update(get_stats(world))
                else:
                    hour_slider.update(world.hour)
        
        if event == '-SPEED-':
            simulation_speed = int(values['-SPEED-'])
        
        if event == '-MAP-':
            mouse_pos = values['-MAP-']
            if mouse_pos != (None, None):
                x, y = mouse_pos
                selected_animal = None
                
                for entity in world.entities:
                    if isinstance(entity, Animal):
                        if (entity.x <= x <= entity.x + 1 and 
                            entity.y <= y <= entity.y + 1):
                            selected_animal = entity
                            break
                
                draw_world(world, map_graph, selected_animal)
        
        current_time = time.time()
        if running and current_time - last_update_time > 1.0 / simulation_speed:
            world.step()
            hour_slider.update(world.hour)
            draw_world(world, map_graph, selected_animal)
            stats_elem.update(get_stats(world))
            last_update_time = current_time
    
    window.close()

if __name__ == "__main__":
    main()