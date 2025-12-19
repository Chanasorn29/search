import heapq
import os
import matplotlib.pyplot as plt
from matplotlib import colors, animation
from collections import deque

# ==========================================
# 1. การจัดการแผนที่ (Map Loader)
# ==========================================
class GridMap:
    def __init__(self, filename):
        self.load_map(filename)

    def load_map(self, filename):
        with open(filename, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
        
        self.height = len(lines)
        self.width = len(lines[0])
        self.obstacles = set()
        for y, row in enumerate(lines):
            for x, char in enumerate(row):
                if char == '#': self.obstacles.add((x, y))
                elif char == 'S': self.start = (x, y)
                elif char == 'G': self.goal = (x, y)

    def get_neighbors(self, state):
        x, y = state
        moves = [('UP', (x, y-1)), ('DOWN', (x, y+1)), ('LEFT', (x-1, y)), ('RIGHT', (x+1, y))]
        return [(a, s) for a, s in moves if 0 <= s[0] < self.width and 0 <= s[1] < self.height and s not in self.obstacles]

# ==========================================
# 2. อัลกอริทึม (ส่งค่า History ออกมาวาด Video)
# ==========================================
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def search_algorithm(grid_map, method='bfs'):
    start = grid_map.start
    goal = grid_map.goal
    
    # สนับสนุน 3 อัลกอริทึม
    if method == 'bfs':
        frontier = deque([(start, [])])
    elif method == 'dfs':
        frontier = [(start, [])]
    else: # a_star
        frontier = [(0, start, [])]

    explored = set()
    history = [] # เก็บสถานะแต่ละ Step เพื่อทำ Video

    while frontier:
        if method == 'bfs': (curr, path) = frontier.popleft()
        elif method == 'dfs': (curr, path) = frontier.pop()
        else: _, curr, path = heapq.heappop(frontier)

        if curr in explored: continue
        explored.add(curr)
        history.append(list(explored)) # เก็บประวัติการค้นหา

        if curr == goal:
            return path + [curr], history

        for _, next_state in grid_map.get_neighbors(curr):
            if next_state not in explored:
                new_path = path + [curr]
                if method == 'bfs' or method == 'dfs':
                    frontier.append((next_state, new_path))
                else: # a_star
                    priority = len(new_path) + heuristic(next_state, goal)
                    heapq.heappush(frontier, (priority, next_state, new_path))
    return None, history

# ==========================================
# 3. การสร้างวีดีโอ (Video Generator)
# ==========================================
def create_video(grid_map, path, history, title):
    folder = "results"
    os.makedirs(folder, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 0:ถนัด, 1:กำแพง, 2:เริ่ม, 3:จบ, 4:สำรวจแล้ว, 5:เส้นทาง
    cmap = colors.ListedColormap(['white', 'black', 'green', 'red', 'lightgray', 'cyan'])
    norm = colors.BoundaryNorm([0,1,2,3,4,5,6], cmap.N)

    def update(frame):
        ax.clear()
        data = [[0 for _ in range(grid_map.width)] for _ in range(grid_map.height)]
        for (x, y) in grid_map.obstacles: data[y][x] = 1
        
        # วาดพื้นที่ที่ถูกสำรวจ (Explored)
        for (x, y) in history[min(frame, len(history)-1)]: data[y][x] = 4
        
        # ถ้าค้นหาเสร็จแล้ว ให้วาดเส้นทางสุดท้าย (Path)
        if frame >= len(history) and path:
            for (x, y) in path: data[y][x] = 5

        sx, sy = grid_map.start; gx, gy = grid_map.goal
        data[sy][sx] = 2; data[gy][gx] = 3

        ax.imshow(data, cmap=cmap, norm=norm)
        ax.set_title(f"{title} - Step: {frame}")
        ax.axis('off')

    # สร้าง Animation
    total_frames = len(history) + (20 if path else 0) # แถมเฟรมตอนจบ 20 เฟรม
    anim = animation.FuncAnimation(fig, update, frames=total_frames, interval=50)
    
    filename = os.path.join(folder, f"{title}.mp4")
    try:
        anim.save(filename, writer='ffmpeg') # ต้องติดตั้ง ffmpeg ในเครื่อง
        print(f"✅ บันทึกวีดีโอสำเร็จ: {filename}")
    except:
        filename = os.path.join(folder, f"{title}.gif")
        anim.save(filename, writer='pillow')
        print(f"✅ บันทึกเป็น GIF แทน (ไม่พบ ffmpeg): {filename}")
    
    plt.close()

# ==========================================
# 4. ส่วนรันโปรแกรม
# ==========================================
if __name__ == "__main__":
    print("--- Maze AI Video Generator ---")
    print("1: Small (10x10)\n2: Medium (20x10)\n3: Large (30x30)")
    choice = input("เลือกขนาดแผนที่ (1-3): ")
    
    map_files = {"1": "maps/small.txt", "2": "maps/medium.txt", "3": "maps/large.txt"}
    target_map = map_files.get(choice, "maps/small.txt")

    if not os.path.exists(target_map):
        print(f"X ไม่พบไฟล์ {target_map} กรุณาสร้างไฟล์ก่อน!")
    else:
        my_map = GridMap(target_map)
        methods = ['bfs', 'dfs', 'a_star']
        
        for m in methods:
            print(f"กำลังประมวลผล {m}...")
            path, history = search_algorithm(my_map, method=m)
            create_video(my_map, path, history, f"{m.upper()}_{choice}")