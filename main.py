import sys
import os
import math
import numpy as np
import pyaudio
import pyglet
from pyglet import clock
from pyglet.window import Window
from OpenGL.GL import *
from OpenGL.GLU import *

# =============================================================================
# Utility functions
# =============================================================================
def hsv_to_rgb(h, s, v):
    """Convert HSV color to RGB"""
    if s == 0.0: return (v, v, v)
    i = int(h*6.0)
    f = (h*6.0) - i
    p = v*(1.0 - s)
    q = v*(1.0 - s*f)
    t = v*(1.0 - s*(1.0-f))
    i %= 6
    return [
        (v, t, p),
        (q, v, p),
        (p, v, t),
        (p, q, v),
        (t, p, v),
        (v, p, q),
    ][i]

# =============================================================================
# Audio settings and stream initialization
# =============================================================================
CHUNK = 1024
RATE = 44100

p = pyaudio.PyAudio()

def get_loopback_device_index():
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        name = dev.get("name", "").lower()
        if "loopback" in name or "wasapi" in name:
            return i
    return None

device_index = get_loopback_device_index()
if device_index is None:
    print("Loopback device not found. Using default input device.")
else:
    print("Using loopback device:", p.get_device_info_by_index(device_index).get("name"))

try:
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    input_device_index=device_index,
                    **({"as_loopback": True} if device_index is not None else {}))
except Exception as e:
    print("Error opening audio stream:", e)
    sys.exit(1)

# =============================================================================
# Dodecahedron geometry setup
# =============================================================================
phi = (1 + math.sqrt(5)) / 2
dodecahedron_vertices = []

# Add cube permutations (±1, ±1, ±1)
for x in (-1, 1):
    for y in (-1, 1):
        for z in (-1, 1):
            dodecahedron_vertices.append([x, y, z])

# Add permutations of (0, ±1/phi, ±phi) and rotations
for i in range(3):
    for a in (-1/phi, 1/phi):
        for b in (-phi, phi):
            if i == 0:
                v = [0, a, b]
            elif i == 1:
                v = [a, b, 0]
            else:
                v = [b, 0, a]
            dodecahedron_vertices.append(v)

# Calculate edges based on distance
edge_length_sq = (2 / phi) ** 2
dodecahedron_edges = []
for i in range(len(dodecahedron_vertices)):
    for j in range(i+1, len(dodecahedron_vertices)):
        dx = dodecahedron_vertices[i][0] - dodecahedron_vertices[j][0]
        dy = dodecahedron_vertices[i][1] - dodecahedron_vertices[j][1]
        dz = dodecahedron_vertices[i][2] - dodecahedron_vertices[j][2]
        if abs(dx*dx + dy*dy + dz*dz - edge_length_sq) < 1e-6:
            dodecahedron_edges.append((i, j))






# =============================================================================
# Object parameters
# =============================================================================
cubes = [
    {"type": "cube", "base_size": 5.0, "angle": 0.0, "base_speed": 2.0, 
     "color": (1.0, 1.0, 1.0, 0.8)},
    {"type": "cube", "base_size": 3.5, "angle": 0.0, "base_speed": 2.0, 
     "color": (1.0, 1.0, 1.0, 0.8)},
    {"type": "dodecahedron", "base_size": 2.0, "angle": 0.0, "base_speed": 2.0,
     "color": (1.0, 1.0, 1.0, 0.8), "color_phase": 0.0, "color_speed": 0.3}
]


# =============================================================================
# Window configuration
# =============================================================================
config = pyglet.gl.Config(
    double_buffer=True,
    sample_buffers=1,
    samples=4,
    alpha_size=8,
    depth_size=24
)
window = Window(
    width=300,
    height=300,
    config=config,
    caption="Audio Reactive Wireframes",
    resizable=False,  # Окно не изменяемого размера
    style=Window.WINDOW_STYLE_BORDERLESS
)


# =============================================================================
# Windows 10+ Blur Effect (Alternative Approach)
# =============================================================================
if os.name == 'nt':
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL('user32')
    dwmapi = ctypes.WinDLL('dwmapi')

    hwnd = window._hwnd

    # Включаем режим acrylic blur через DWM
    class DWM_BLURBEHIND(ctypes.Structure):
        _fields_ = [
            ("dwFlags", wintypes.DWORD),
            ("fEnable", wintypes.BOOL),
            ("hRgnBlur", wintypes.HRGN),
            ("fTransitionOnMaximized", wintypes.BOOL)
        ]

    class ACCENT_POLICY(ctypes.Structure):
        _fields_ = [
            ("AccentState", ctypes.c_uint),
            ("AccentFlags", ctypes.c_uint),
            ("GradientColor", ctypes.c_uint),
            ("AnimationId", ctypes.c_uint)
        ]

    class WINDOWCOMPOSITIONATTRIBUTEDATA(ctypes.Structure):
        _fields_ = [
            ("Attribute", ctypes.c_int),
            ("Data", ctypes.POINTER(ctypes.c_byte)),
            ("SizeOfData", ctypes.c_size_t)
        ]

    try:
        # Настраиваем размытие
        blur_behind = DWM_BLURBEHIND()
        blur_behind.dwFlags = 0x00000003  # DWM_BB_ENABLE | DWM_BB_BLURREGION
        blur_behind.fEnable = True
        blur_behind.hRgnBlur = ctypes.windll.gdi32.CreateRectRgn(0, 0, -1, -1)
        
        dwmapi.DwmEnableBlurBehindWindow(hwnd, ctypes.byref(blur_behind))

        # Настраиваем кликабельную область
        margins = wintypes.MARGINS()
        margins.cxLeftWidth = -1
        margins.cxRightWidth = -1
        margins.cyTopHeight = -1
        margins.cyBottomHeight = -1
        dwmapi.DwmExtendFrameIntoClientArea(hwnd, ctypes.byref(margins))

        # Настраиваем прозрачность
        accent = ACCENT_POLICY()
        accent.AccentState = 3  # ACCENT_ENABLE_ACRYLICBLURBEHIND
        accent.GradientColor = 0x00FFFFFF  # Белый цвет с альфа=0

        data = WINDOWCOMPOSITIONATTRIBUTEDATA()
        data.Attribute = 19  # WCA_ACCENT_POLICY
        data.SizeOfData = ctypes.sizeof(accent)
        data.Data = ctypes.cast(ctypes.pointer(accent), ctypes.POINTER(ctypes.c_byte))

        ctypes.windll.user32.SetWindowCompositionAttribute(hwnd, ctypes.byref(data))

    except Exception as e:
        print("DWM composition error:", e)

# =============================================================================
# Window Dragging with WinAPI (Replace existing mouse handling)
# =============================================================================
import ctypes
import ctypes.wintypes

#region WinAPI Functions
GetCursorPos = ctypes.windll.user32.GetCursorPos
GetCursorPos.argtypes = [ctypes.POINTER(ctypes.wintypes.POINT)]
SetWindowPos = ctypes.windll.user32.SetWindowPos
GetWindowRect = ctypes.windll.user32.GetWindowRect
ScreenToClient = ctypes.windll.user32.ScreenToClient
ClientToScreen = ctypes.windll.user32.ClientToScreen

def get_cursor_pos():
    """Возвращает текущие экранные координаты курсора"""
    point = ctypes.wintypes.POINT()
    GetCursorPos(ctypes.byref(point))
    return (point.x, point.y)

def move_window(x, y):
    """Перемещает окно в указанные экранные координаты"""
    SetWindowPos(window._hwnd, None, x, y, 0, 0, 0x0001)
#endregion

class WindowDragger:
    """Класс для управления перетаскиванием окна"""
    def __init__(self):
        self.dragging = False
        self.start_cursor_pos = (0, 0)
        self.start_window_pos = (0, 0)
        
    def start_drag(self, x, y):
        """Начать перетаскивание при нажатии мыши"""
        self.dragging = True
        self.start_cursor_pos = get_cursor_pos()
        
        # Получаем текущую позицию окна
        rect = ctypes.wintypes.RECT()
        GetWindowRect(window._hwnd, ctypes.byref(rect))
        self.start_window_pos = (rect.left, rect.top)
    
    def update_drag(self):
        """Обновить позицию окна при перемещении мыши"""
        if self.dragging:
            current_cursor_pos = get_cursor_pos()
            
            # Вычисляем смещение
            dx = current_cursor_pos[0] - self.start_cursor_pos[0]
            dy = current_cursor_pos[1] - self.start_cursor_pos[1]
            
            # Новая позиция окна
            new_x = self.start_window_pos[0] + dx
            new_y = self.start_window_pos[1] + dy
            
            move_window(new_x, new_y)

# Инициализация перетаскивания окна
window_dragger = WindowDragger()

@window.event
def on_mouse_press(x, y, button, modifiers):
    if button == pyglet.window.mouse.LEFT:
        window_dragger.start_drag(x, y)

@window.event
def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
    if buttons & pyglet.window.mouse.LEFT:
        window_dragger.update_drag()

@window.event
def on_mouse_release(x, y, button, modifiers):
    window_dragger.dragging = False

# =============================================================================
# OpenGL initialization
# =============================================================================
def init_gl():
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_LINE_SMOOTH)
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
    glLineWidth(1.5)

init_gl()

def setup_projection():
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60.0, float(window.width)/window.height, 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    # Центрируем камеру и добавляем небольшое смещение по Y вверх
    glTranslatef(0.0, 0.3, -12.0)  # 0.3 - поднимает камеру, фигуры опускаются вниз

# =============================================================================
# Drawing functions
# =============================================================================
def draw_wire_cube(size, color):
    hs = size / 2.0
    vertices = [
        [-hs, -hs, -hs], [ hs, -hs, -hs], [ hs,  hs, -hs], [-hs,  hs, -hs],
        [-hs, -hs,  hs], [ hs, -hs,  hs], [ hs,  hs,  hs], [-hs,  hs,  hs]
    ]
    edges = [
        (0,1), (1,2), (2,3), (3,0),
        (4,5), (5,6), (6,7), (7,4),
        (0,4), (1,5), (2,6), (3,7)
    ]
    glColor4f(*color)
    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glVertex3f(*vertices[vertex])
    glEnd()

def draw_wire_dodecahedron(size, color):
    hs = size / 2
    scale = hs / phi
    scaled_vertices = [[v[0]*scale, v[1]*scale, v[2]*scale] for v in dodecahedron_vertices]
    
    glColor4f(*color)
    glBegin(GL_LINES)
    for edge in dodecahedron_edges:
        for vertex_index in edge:
            v = scaled_vertices[vertex_index]
            glVertex3f(v[0], v[1], v[2])
    glEnd()

# =============================================================================
# Audio processing
# =============================================================================
def process_audio_data(data):
    audio_data = np.frombuffer(data, dtype=np.int16)
    window_func = np.hanning(len(audio_data))
    fft_data = np.fft.rfft(audio_data * window_func)
    fft_magnitude = np.abs(fft_data)
    
    low_bins = fft_magnitude[1:8]
    mid_bins = fft_magnitude[8:46]
    high_bins = fft_magnitude[46:232] if len(fft_magnitude) > 232 else fft_magnitude[46:]
    
    return (
        np.average(low_bins) if low_bins.size else 0,
        np.average(mid_bins) if mid_bins.size else 0,
        np.average(high_bins) if high_bins.size else 0
    )

# =============================================================================
# Animation system
# =============================================================================
BLINK_SPEED_MULTIPLIER = 0.01  # Уменьшаем скорость мерцания
BASE_BLINK_SPEED = 0.2
SMOOTHING_FACTOR = 0.12       # Увеличиваем сглаживание
AMP_HISTORY_LENGTH = 8        # Увеличиваем историю для сглаживания

amp_history = [[0.0] * AMP_HISTORY_LENGTH for _ in range(3)]

def update(dt):
    try:
        data = stream.read(CHUNK, exception_on_overflow=False)
    except Exception as e:
        print("Audio input error:", e)
        return

    low, mid, high = (min(v / 80000.0, 2.5) for v in process_audio_data(data))
    
    # Комбинированная амплитуда для додекаэдра
    combined_amp = (low + mid + high) / 3.0

    for i, amp in enumerate([low, mid, combined_amp]):  # Для додекаэдра используем combined_amp
        # Обновляем историю амплитуды
        amp_history[i].pop(0)
        amp_history[i].append(amp)
        
        # Сглаженная амплитуда с учетом большей истории
        smoothed_amp = sum(amp_history[i]) / AMP_HISTORY_LENGTH
        amp_clean = max(smoothed_amp - 0.01, 0.0) * 1.4  # Уменьшаем порог

        cubes[i]["angle"] += (cubes[i]["base_speed"] + 30 * amp_clean) * dt
        cubes[i]["scale"] = cubes[i]["base_size"] * (1.0 + 0.12 * amp_clean)  # Увеличиваем масштабирование

        if cubes[i]['type'] == 'dodecahedron':
            cubes[i]['color_phase'] = (cubes[i]['color_phase'] + dt * cubes[i]['color_speed']) % 1.0
            target_rgb = hsv_to_rgb(cubes[i]['color_phase'], 0.9, 1.0)  # Уменьшаем насыщенность
            
            # Всегда используем цветовую анимацию, даже при низкой амплитуде
            blink_speed = BASE_BLINK_SPEED + (BLINK_SPEED_MULTIPLIER * amp_clean)
            blink = (math.sin(cubes[i]["angle"] * blink_speed) * 0.5 + 0.5)  # Мягкое мерцание
            
            # Базовый цвет с плавными переходами
            base_color = [
                target_rgb[0] * (0.7 + 0.3 * amp_clean),
                target_rgb[1] * (0.7 + 0.3 * amp_clean),
                target_rgb[2] * (0.7 + 0.3 * amp_clean),
                0.8 + 0.2 * amp_clean
            ]
            
            # Плавное смешение с предыдущим цветом
            cr, cg, cb, ca = cubes[i]['color']
            new_r = cr + (base_color[0] - cr) * 0.15
            new_g = cg + (base_color[1] - cg) * 0.15
            new_b = cb + (base_color[2] - cb) * 0.15
            new_a = ca + (base_color[3] - ca) * 0.1
            
            cubes[i]['color'] = (new_r, new_g, new_b, new_a)
        else:
            # Обработка кубов остается без изменений
            if amp_clean > 0.15:
                blink_speed = BASE_BLINK_SPEED + (BLINK_SPEED_MULTIPLIER * amp_clean)
                blink = (math.sin(cubes[i]["angle"] * blink_speed) + 1) / 2
                cubes[i]["color"] = (blink, blink, blink, 0.7 + 0.2 * blink)
            else:
                current_color = cubes[i]["color"][0]
                target_color = 1.0 - (amp_clean * 0.6)
                new_color = current_color + (target_color - current_color) * SMOOTHING_FACTOR
                cubes[i]["color"] = (new_color, new_color, new_color, 0.8)

# =============================================================================
# Main rendering loop
# =============================================================================

@window.event
def on_draw():
    window.clear()
    setup_projection()
    for cube in cubes:
        glPushMatrix()
        glRotatef(cube["angle"], 1, 1, 1)
        if cube["type"] == "cube":
            draw_wire_cube(cube.get("scale", cube["base_size"]), cube["color"])
        else:
            draw_wire_dodecahedron(cube.get("scale", cube["base_size"]), cube["color"])
        glPopMatrix()

clock.schedule_interval(update, 1/60.0)


# =============================================================================
# Application execution
# =============================================================================
pyglet.app.run()

# =============================================================================
# Cleanup
# =============================================================================
stream.stop_stream()
stream.close()
p.terminate()


