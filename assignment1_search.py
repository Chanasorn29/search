import heapq
import matplotlib.pyplot as plt
from collections import deque
from matplotlib import colors
import os

# ==========================================
# 1. Data Structures (โครงสร้างข้อมูล)
# ==========================================

class Node:
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state = state          # (x, y)
        self.parent = parent        # Node ก่อนหน้า
        self.action = action        # Action ที่ทำ (UP, DOWN, etc.)
        self.path_cost = path_cost  # g(n)

    # สำหรับ Priority Queue ใน A*
    def __lt__(self, other):
        return self.path_cost < other.path_cost

class GridMap:
    def __init__(self, width, height, obstacles, start, goal):
        self.width = width
        self.height = height
        self.obstacles = obstacles
        self.start = start
        self.goal = goal

    def get_neighbors(self, state):
        x, y = state
        moves = [('UP', (x, y-1)), ('DOWN', (x, y+1)), 
                 ('LEFT', (x-1, y)), ('RIGHT', (x+1, y))]
        results = []
        for action, (nx, ny) in moves:
            if 0 <= nx < self.width and 0 <= ny < self.height:
                if (nx, ny) not in self.obstacles:
                    results.append((action, (nx, ny)))
        return results

# ==========================================
# 2. Helper Functions (ฟังก์ชันช่วยทำงาน)
# ==========================================

def get_solution_path(node):
    """ย้อนรอยจากเป้าหมายกลับไปจุดเริ่มต้น"""
    path = []
    while node:
        path.append(node.state)
        node = node.parent
    return path[::-1]

def heuristic_manhattan(node_state, goal_state):
    """คำนวณระยะทางแบบ Manhattan |x1-x2| + |y1-y2|"""
    return abs(node_state[0] - goal_state[0]) + abs(node_state[1] - goal_state[1])

def visualize_path(grid_map, path, title):
    """วาดกราฟและบันทึกเป็นไฟล์รูปภาพลงในโฟลเดอร์"""
    print(f"\n--- {title} ---")
    if not path:
        print("No path found!")
        return

    # --- ส่วนที่ 1: กำหนดชื่อโฟลเดอร์ที่จะเก็บรูป ---
    folder_name = "search_results"  # ตั้งชื่อโฟลเดอร์ตรงนี้
    
    # สร้างโฟลเดอร์ถ้ายังไม่มี (ถ้ามีแล้วก็ข้ามไป)
    os.makedirs(folder_name, exist_ok=True) 
    
    # ----------------------------------------

    # เตรียมข้อมูลสำหรับวาด (เหมือนเดิม)
    data = [[0 for _ in range(grid_map.width)] for _ in range(grid_map.height)]
    for (x, y) in grid_map.obstacles: data[y][x] = 1
    for (x, y) in path:               data[y][x] = 4
    
    sx, sy = grid_map.start
    gx, gy = grid_map.goal
    data[sy][sx] = 2
    data[gy][gx] = 3

    cmap = colors.ListedColormap(['white', 'black', 'green', 'red', 'cyan'])
    bounds = [0, 1, 2, 3, 4, 5]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(grid_map.width / 2, grid_map.height / 2))
    ax.imshow(data, cmap=cmap, norm=norm)
    
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1, alpha=0.3)
    ax.set_xticks([x - 0.5 for x in range(grid_map.width + 1)])
    ax.set_yticks([y - 0.5 for y in range(grid_map.height + 1)])
    ax.set_xticklabels([]); ax.set_yticklabels([])
    
    ax.set_title(f"{title} (Cost: {len(path)-1})", fontsize=14, weight='bold')
    plt.tight_layout()
    
    # --- ส่วนที่ 2: บันทึกไฟล์ลงในโฟลเดอร์ ---
    # รวมชื่อโฟลเดอร์กับชื่อไฟล์เข้าด้วยกัน (เช่น search_results/BFS_Maze.png)
    filename = os.path.join(folder_name, f"{title}.png")
    
    plt.savefig(filename)
    print(f"✅ บันทึกรูปภาพสำเร็จ: {filename}")
    plt.close()

# ==========================================
# 3. Search Algorithms (สมองของ AI)
# ==========================================

def bfs(grid_map):
    start_node = Node(state=grid_map.start, path_cost=0)
    frontier = deque([start_node])
    explored = set()
    nodes_expanded = 0

    while frontier:
        current_node = frontier.popleft()
        nodes_expanded += 1

        if current_node.state == grid_map.goal:
            return get_solution_path(current_node), nodes_expanded

        explored.add(current_node.state)

        for action, neighbor_state in grid_map.get_neighbors(current_node.state):
            if neighbor_state not in explored and neighbor_state not in [n.state for n in frontier]:
                child = Node(neighbor_state, current_node, action, current_node.path_cost + 1)
                frontier.append(child)
    
    return None, nodes_expanded

def dfs(grid_map):
    start_node = Node(state=grid_map.start, path_cost=0)
    frontier = [start_node] # Stack
    explored = set()
    nodes_expanded = 0

    while frontier:
        current_node = frontier.pop()
        nodes_expanded += 1

        if current_node.state == grid_map.goal:
            return get_solution_path(current_node), nodes_expanded

        if current_node.state not in explored:
            explored.add(current_node.state)
            for action, neighbor_state in grid_map.get_neighbors(current_node.state):
                if neighbor_state not in explored:
                    child = Node(neighbor_state, current_node, action, current_node.path_cost + 1)
                    frontier.append(child)
    
    return None, nodes_expanded

def a_star(grid_map):
    start_node = Node(state=grid_map.start, path_cost=0)
    frontier = []
    count = 0
    # Priority Queue: (f_score, count, node)
    heapq.heappush(frontier, (0 + heuristic_manhattan(start_node.state, grid_map.goal), count, start_node))
    
    explored = set()
    nodes_expanded = 0
    g_costs = {start_node.state: 0}

    while frontier:
        _, _, current_node = heapq.heappop(frontier)
        nodes_expanded += 1

        if current_node.state == grid_map.goal:
            return get_solution_path(current_node), nodes_expanded

        explored.add(current_node.state)

        for action, neighbor_state in grid_map.get_neighbors(current_node.state):
            new_g = current_node.path_cost + 1
            
            if neighbor_state not in g_costs or new_g < g_costs[neighbor_state]:
                g_costs[neighbor_state] = new_g
                f_score = new_g + heuristic_manhattan(neighbor_state, grid_map.goal)
                child = Node(neighbor_state, current_node, action, new_g)
                count += 1
                heapq.heappush(frontier, (f_score, count, child))
                
    return None, nodes_expanded

# ==========================================
# 4. Main Execution (ส่วนทดสอบโปรแกรม)
# ==========================================

if __name__ == "__main__":
    # --- กำหนดแผนที่ (แก้ไขตรงนี้ได้เลย) ---
    maze_layout = [
        "S..#..........#.....",
        ".#.#######.##.#.###.",
        ".#.......#.#....#...",
        ".#.#####.#.####.#.##",
        ".#.#...#.#....#.#...",
        ".###.#.#.####.###.##",
        ".....#.#......#.....",
        ".#####.########.###.",
        ".#..............#...",
        ".################.G."
    ]

    # แปลงแผนที่เป็น GridMap Object
    height = len(maze_layout)
    width = len(maze_layout[0])
    obstacles = set()
    start = (0, 0); goal = (0, 0)

    for y, row in enumerate(maze_layout):
        for x, char in enumerate(row):
            if char == '#': obstacles.add((x, y))
            elif char == 'S': start = (x, y)
            elif char == 'G': goal = (x, y)

    my_map = GridMap(width, height, obstacles, start, goal)

    print(f"Map Size: {width}x{height}")
    print(f"Start: {start}, Goal: {goal}")
    print("Testing Maze Algorithms...\n")

    # --- Run Algorithms ---
    
    # 1. BFS
    path, nodes = bfs(my_map)
    print(f"1. BFS -> Cost: {len(path)-1 if path else 'X'}, Nodes: {nodes}")
    visualize_path(my_map, path, "BFS_Maze")

    # 2. DFS
    path, nodes = dfs(my_map)
    print(f"2. DFS -> Cost: {len(path)-1 if path else 'X'}, Nodes: {nodes}")
    visualize_path(my_map, path, "DFS_Maze")

    # 3. A* Star
    path, nodes = a_star(my_map)
    print(f"3. A* -> Cost: {len(path)-1 if path else 'X'}, Nodes: {nodes}")
    visualize_path(my_map, path, "A_Star_Maze")