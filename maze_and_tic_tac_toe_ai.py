import random
import sys
import heapq
import json
import os
import time
from typing import List, Tuple, Optional, Dict, Set


# ==========================
# Utility: Console rendering
# ==========================
def print_grid(grid: List[List[str]]) -> None:
    for row in grid:
        print(" ".join(row))


# ======================================
# Lab 4: 8x8 Tic Tac Toe (Connect-4)
# ======================================
BoardSize = 8
EmptyCell = '.'
Human = 'X'
AI = 'O'
WinLength = 4

# ======================
# Learning configuration
# ======================
RL_FILE = "ttt_experience.json"
RL_ALPHA = 0.5
RL_GAMMA = 0.95
RL_EPSILON = 0.08
SEARCH_TIME_SEC = 0.6

# In-game AI move history: (state_key, (r, c))
TTT_AI_HISTORY: List[Tuple[str, Tuple[int, int]]] = []


def rl_load() -> Dict[str, Dict[str, float]]:
    try:
        if os.path.exists(RL_FILE):
            with open(RL_FILE, "r") as f:
                data = json.load(f)
                # Ensure proper types; ignore meta here
                store: Dict[str, Dict[str, float]] = {}
                for k, v in data.items():
                    if k == "__meta__":
                        continue
                    store[k] = {ak: float(av) for ak, av in v.items()}
                return store
    except Exception:
        pass
    return {}


def rl_load_meta() -> Dict[str, int]:
    try:
        if os.path.exists(RL_FILE):
            with open(RL_FILE, "r") as f:
                data = json.load(f)
                meta = data.get("__meta__", {})
                return {k: int(v) for k, v in meta.items()}
    except Exception:
        pass
    return {"games": 0}


def rl_save(store: Dict[str, Dict[str, float]], meta: Optional[Dict[str, int]] = None) -> None:
    try:
        out: Dict[str, Dict[str, float]] = {k: v for k, v in store.items()}
        if meta is None:
            meta = rl_load_meta()
        out_payload: Dict[str, object] = {k: v for k, v in out.items()}
        out_payload["__meta__"] = meta
        with open(RL_FILE, "w") as f:
            json.dump(out_payload, f)
    except Exception:
        pass


def ttt_board_key(board: List[List[str]], player_to_move: str) -> str:
    # Encode board, size, win len, and the side to move
    rows = ["".join(row) for row in board]
    return f"{BoardSize}x{BoardSize}:{WinLength}:{player_to_move}:" + "/".join(rows)


def rl_get_qvalues(store: Dict[str, Dict[str, float]], state_key: str) -> Dict[str, float]:
    return store.get(state_key, {})


def rl_set_qvalue(store: Dict[str, Dict[str, float]], state_key: str, action_key: str, value: float) -> None:
    if state_key not in store:
        store[state_key] = {}
    store[state_key][action_key] = value


def rl_learn_from_game(result: int) -> None:
    # result: 1 = AI win, 0 = draw, -1 = AI loss
    if not TTT_AI_HISTORY:
        return
    reward_final = 1.0 if result > 0 else (-1.0 if result < 0 else 0.1)
    store = rl_load()
    meta = rl_load_meta()
    meta["games"] = meta.get("games", 0) + 1
    G = reward_final
    for state_key, action in reversed(TTT_AI_HISTORY):
        action_key = f"{action[0]},{action[1]}"
        q_s = rl_get_qvalues(store, state_key)
        old = q_s.get(action_key, 0.0)
        new = old + RL_ALPHA * (G - old)
        rl_set_qvalue(store, state_key, action_key, new)
        G *= RL_GAMMA
    rl_save(store, meta)


def rl_current_epsilon() -> float:
    meta = rl_load_meta()
    games = meta.get("games", 0)
    # Anneal by 2% per game down to a floor
    eps = RL_EPSILON * (0.98 ** games)
    return 0.01 if eps < 0.01 else eps


def ttt_configure() -> None:
    global BoardSize, WinLength
    print("\nTic Tac Toe config (press Enter for defaults 8 8 4)")
    raw = input("rows cols winLength: ").strip()
    if not raw:
        BoardSize, WinLength = 8, 4
        return
    try:
        parts = [int(x) for x in raw.split()]
        if len(parts) != 3:
            print("Invalid input, using defaults (8 8 4).")
            BoardSize, WinLength = 8, 4
            return
        r, c, w = parts
        if r != c or r < 3 or r > 12 or w < 3 or w > r:
            print("Use square boards 3..12 and winLength between 3 and size. Using defaults.")
            BoardSize, WinLength = 8, 4
            return
        BoardSize, WinLength = r, w
    except ValueError:
        print("Invalid input, using defaults (8 8 4).")
        BoardSize, WinLength = 8, 4


def ttt_create_board() -> List[List[str]]:
    return [[EmptyCell for _ in range(BoardSize)] for _ in range(BoardSize)]


def ttt_in_bounds(r: int, c: int) -> bool:
    return 0 <= r < BoardSize and 0 <= c < BoardSize


def ttt_check_win(board: List[List[str]], player: str) -> bool:
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for r in range(BoardSize):
        for c in range(BoardSize):
            if board[r][c] != player:
                continue
            for dr, dc in directions:
                count = 0
                rr, cc = r, c
                while ttt_in_bounds(rr, cc) and board[rr][cc] == player and count < WinLength:
                    count += 1
                    rr += dr
                    cc += dc
                if count >= WinLength:
                    return True
    return False


def ttt_board_full(board: List[List[str]]) -> bool:
    return all(cell != EmptyCell for row in board for cell in row)


def ttt_get_empty_cells(board: List[List[str]]) -> List[Tuple[int, int]]:
    return [(r, c) for r in range(BoardSize) for c in range(BoardSize) if board[r][c] == EmptyCell]


def ttt_try_move(board: List[List[str]], r: int, c: int, player: str) -> bool:
    if board[r][c] != EmptyCell:
        return False
    board[r][c] = player
    won = ttt_check_win(board, player)
    board[r][c] = EmptyCell
    return won


def ttt_evaluate(board: List[List[str]], player: str) -> int:
    opponent = Human if player == AI else AI
    # Immediate terminal checks
    if ttt_check_win(board, AI):
        return 1_000_000
    if ttt_check_win(board, Human):
        return -1_000_000

    score = 0
    # Center preference
    center = BoardSize // 2
    for r in range(BoardSize):
        for c in range(BoardSize):
            if board[r][c] == AI:
                score += 3 - (abs(r - center) + abs(c - center))
            elif board[r][c] == Human:
                score -= 3 - (abs(r - center) + abs(c - center))

    # Line potential scoring
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    def line_score(cells: List[str]) -> int:
        s = 0
        for i in range(0, len(cells) - WinLength + 1):
            window = cells[i:i+WinLength]
            if opponent in window and AI in window:
                continue
            ai_count = window.count(AI)
            hu_count = window.count(Human)
            empty = window.count(EmptyCell)
            # Determine open ends (two-sided threats are stronger)
            left_open = (i - 1 >= 0 and cells[i - 1] == EmptyCell)
            right_open = (i + WinLength < len(cells) and cells[i + WinLength] == EmptyCell)
            open_ends = (1 if left_open else 0) + (1 if right_open else 0)

            if hu_count == 0 and ai_count > 0:
                # Attacking windows
                if ai_count == WinLength - 1 and empty == 1:
                    s += 9000 if open_ends == 2 else 6000
                elif ai_count == WinLength - 2 and empty == 2:
                    s += 1200 * (1 + open_ends)
                s += ai_count * ai_count * (8 if open_ends == 2 else 5)
            if ai_count == 0 and hu_count > 0:
                # Defensive windows
                if hu_count == WinLength - 1 and empty == 1:
                    s += 7000 if open_ends == 2 else 4500
                elif hu_count == WinLength - 2 and empty == 2:
                    s += 900 * (1 + open_ends)
                s += hu_count * hu_count * (6 if open_ends == 2 else 4)
        return s

    # Rows
    for r in range(BoardSize):
        score += line_score(board[r])
    # Cols
    for c in range(BoardSize):
        col = [board[r][c] for r in range(BoardSize)]
        score += line_score(col)
    # Diagonals TL-BR
    for start in range(BoardSize):
        diag1 = []
        r, c = 0, start
        while ttt_in_bounds(r, c):
            diag1.append(board[r][c])
            r += 1
            c += 1
        score += line_score(diag1)
    for start in range(1, BoardSize):
        diag1 = []
        r, c = start, 0
        while ttt_in_bounds(r, c):
            diag1.append(board[r][c])
            r += 1
            c += 1
        score += line_score(diag1)
    # Diagonals TR-BL
    for start in range(BoardSize):
        diag2 = []
        r, c = 0, start
        while ttt_in_bounds(r, c):
            diag2.append(board[r][c])
            r += 1
            c -= 1
        score += line_score(diag2)
    for start in range(1, BoardSize):
        diag2 = []
        r, c = start, BoardSize - 1
        while ttt_in_bounds(r, c):
            diag2.append(board[r][c])
            r += 1
            c -= 1
        score += line_score(diag2)

    return score


def ttt_count_immediate_wins(board: List[List[str]], player: str) -> Tuple[int, List[Tuple[int, int]]]:
    wins: List[Tuple[int, int]] = []
    for r in range(BoardSize):
        for c in range(BoardSize):
            if board[r][c] == EmptyCell and ttt_try_move(board, r, c, player):
                wins.append((r, c))
    return len(wins), wins


def ttt_find_forks(board: List[List[str]], player: str) -> List[Tuple[int, int]]:
    fork_moves: List[Tuple[int, int]] = []
    for r in range(BoardSize):
        for c in range(BoardSize):
            if board[r][c] != EmptyCell:
                continue
            board[r][c] = player
            count, _ = ttt_count_immediate_wins(board, player)
            board[r][c] = EmptyCell
            if count >= 2:
                fork_moves.append((r, c))
    return fork_moves


def ttt_generate_candidates(board: List[List[str]]) -> List[Tuple[int, int]]:
    # Consider empty cells adjacent within distance 2 to any occupied cell; fallback to all if board empty
    occupied: Set[Tuple[int, int]] = set()
    for r in range(BoardSize):
        for c in range(BoardSize):
            if board[r][c] != EmptyCell:
                occupied.add((r, c))
    empties = ttt_get_empty_cells(board)
    if not occupied:
        # prefer center first then random few
        center = (BoardSize // 2, BoardSize // 2)
        if board[center[0]][center[1]] == EmptyCell:
            return [center]
        return empties[: min(8, len(empties))]
    nbrs: List[Tuple[int, int]] = []
    seen: Set[Tuple[int, int]] = set()
    for r, c in occupied:
        for dr in (-2, -1, 0, 1, 2):
            for dc in (-2, -1, 0, 1, 2):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r + dr, c + dc
                if ttt_in_bounds(rr, cc) and board[rr][cc] == EmptyCell and (rr, cc) not in seen:
                    seen.add((rr, cc))
                    nbrs.append((rr, cc))
    if not nbrs:
        return empties
    random.shuffle(nbrs)
    return nbrs[: min(28, len(nbrs))]


TTT_TRANSPOSITION: Dict[str, Tuple[int, int, int]] = {}


def ttt_minimax(board: List[List[str]], depth: int, alpha: int, beta: int, maximizing: bool, deadline: Optional[float] = None) -> Tuple[int, Optional[Tuple[int, int]]]:
    if deadline is not None and time.time() > deadline:
        return ttt_evaluate(board, AI), None
    if depth == 0 or ttt_check_win(board, AI) or ttt_check_win(board, Human) or ttt_board_full(board):
        return ttt_evaluate(board, AI), None
    best_move: Optional[Tuple[int, int]] = None
    candidates = ttt_generate_candidates(board)
    # Transposition table lookup
    state_key = ttt_board_key(board, AI if maximizing else Human)
    tt_key = f"{state_key}|d={depth}|max={1 if maximizing else 0}"
    if tt_key in TTT_TRANSPOSITION:
        cached_depth, cached_r, cached_c = TTT_TRANSPOSITION[tt_key]
        if cached_r >= 0 and cached_c >= 0 and board[cached_r][cached_c] == EmptyCell:
            best_move = (cached_r, cached_c)
    # Move ordering by static evaluation after playing the move
    scored: List[Tuple[int, Tuple[int, int]]] = []
    for r, c in candidates:
        board[r][c] = AI if maximizing else Human
        scored.append((ttt_evaluate(board, AI), (r, c)))
        board[r][c] = EmptyCell
    candidates = [mv for _, mv in sorted(scored, key=lambda x: x[0], reverse=maximizing)]
    if maximizing:
        value = -10**9
        # Immediate wins
        for r, c in candidates:
            if ttt_try_move(board, r, c, AI):
                if deadline is None or time.time() <= deadline:
                    TTT_TRANSPOSITION[tt_key] = (depth, r, c)
                return 1_000_000, (r, c)
        # Prefer creating forks
        for r, c in candidates:
            board[r][c] = AI
            fork_after, _ = ttt_count_immediate_wins(board, AI)
            board[r][c] = EmptyCell
            if fork_after >= 2:
                if deadline is None or time.time() <= deadline:
                    TTT_TRANSPOSITION[tt_key] = (depth, r, c)
                return 600_000, (r, c)
        for r, c in candidates:
            if deadline is not None and time.time() > deadline:
                break
            board[r][c] = AI
            eval_score, _ = ttt_minimax(board, depth - 1, alpha, beta, False, deadline)
            board[r][c] = EmptyCell
            if eval_score > value:
                value = eval_score
                best_move = (r, c)
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        if deadline is None or time.time() <= deadline:
            rr, cc = (best_move if best_move else (-1, -1))
            TTT_TRANSPOSITION[tt_key] = (depth, rr, cc)
        return value, best_move
    else:
        value = 10**9
        # Opponent immediate wins
        for r, c in candidates:
            if ttt_try_move(board, r, c, Human):
                if deadline is None or time.time() <= deadline:
                    TTT_TRANSPOSITION[tt_key] = (depth, r, c)
                return -1_000_000, (r, c)
        # Avoid letting opponent create forks next
        for r, c in candidates:
            board[r][c] = Human
            fork_after, _ = ttt_count_immediate_wins(board, Human)
            board[r][c] = EmptyCell
            if fork_after >= 2:
                if deadline is None or time.time() <= deadline:
                    TTT_TRANSPOSITION[tt_key] = (depth, r, c)
                return -600_000, (r, c)
        for r, c in candidates:
            if deadline is not None and time.time() > deadline:
                break
            board[r][c] = Human
            eval_score, _ = ttt_minimax(board, depth - 1, alpha, beta, True, deadline)
            board[r][c] = EmptyCell
            if eval_score < value:
                value = eval_score
                best_move = (r, c)
            beta = min(beta, value)
            if alpha >= beta:
                break
        if deadline is None or time.time() <= deadline:
            rr, cc = (best_move if best_move else (-1, -1))
            TTT_TRANSPOSITION[tt_key] = (depth, rr, cc)
        return value, best_move


def ttt_ai_move(board: List[List[str]]) -> Tuple[int, int]:
    empties = ttt_get_empty_cells(board)
    # 1) Win if possible
    for r, c in empties:
        if ttt_try_move(board, r, c, AI):
            return r, c
    # 2) Block opponent
    for r, c in empties:
        if ttt_try_move(board, r, c, Human):
            return r, c
    # 3) Create a fork if possible
    ai_forks = ttt_find_forks(board, AI)
    if ai_forks:
        return ai_forks[0]
    # 4) Block opponent fork if any
    human_forks = ttt_find_forks(board, Human)
    for mv in human_forks:
        if board[mv[0]][mv[1]] == EmptyCell:
            return mv
    # 5) Learning-guided selection (epsilon-greedy) blended with alpha-beta
    # Build candidate moves
    candidates = ttt_generate_candidates(board)
    state_key = ttt_board_key(board, AI)
    store = rl_load()
    qvalues = rl_get_qvalues(store, state_key)
    # Epsilon exploration
    explore = random.random() < rl_current_epsilon()
    chosen: Optional[Tuple[int, int]] = None
    if not explore and qvalues:
        # Pick best known action among candidates
        best_val = -1e18
        for r, c in candidates:
            key = f"{r},{c}"
            val = qvalues.get(key, 0.0)
            if val > best_val and board[r][c] == EmptyCell:
                best_val = val
                chosen = (r, c)
    if chosen is None:
        # Use alpha-beta to pick move
        depth = 4 if len(empties) <= 12 else (3 if len(empties) <= 22 else 2)
        deadline = time.time() + SEARCH_TIME_SEC
        _, chosen = ttt_minimax(board, depth=depth, alpha=-10**9, beta=10**9, maximizing=True, deadline=deadline)
        if chosen is None and empties:
            chosen = random.choice(empties)
    if chosen is not None:
        # Record for learning
        TTT_AI_HISTORY.append((state_key, chosen))
        return chosen
    # 4) Fallback random
    return random.choice(empties) if empties else (-1, -1)


def ttt_play_game() -> None:
    ttt_configure()
    board = ttt_create_board()
    current = Human
    print(f"\n=== {BoardSize}x{BoardSize} Tic Tac Toe (Connect-{WinLength}) ===")
    print("Human = X, AI = O")
    print("Enter moves as: row col (0-indexed)")
    print_grid(board)

    while True:
        if current == Human:
            try:
                raw = input("Your move (row col): ").strip()
                if raw.lower() in {"q", "quit", "exit"}:
                    print("Exiting game.")
                    return
                parts = raw.split()
                if len(parts) != 2:
                    print("Please enter two integers: row col")
                    continue
                r, c = int(parts[0]), int(parts[1])
            except ValueError:
                print("Invalid input. Please enter integers like: 3 4")
                continue

            if not ttt_in_bounds(r, c) or board[r][c] != EmptyCell:
                print("Illegal move. Try again.")
                continue

            board[r][c] = Human
        else:
            r, c = ttt_ai_move(board)
            if r == -1:
                print("No moves available.")
                return
            board[r][c] = AI
            print(f"AI moves to: {r} {c}")

        print_grid(board)

        # Check end conditions
        if ttt_check_win(board, current):
            print(f"{current} wins!")
            # Learn from outcome
            rl_learn_from_game(1 if current == AI else -1)
            TTT_AI_HISTORY.clear()
            return
        if ttt_board_full(board):
            print("It's a draw.")
            rl_learn_from_game(0)
            TTT_AI_HISTORY.clear()
            return

        current = AI if current == Human else Human


# ======================================
# Lab 2 and 3: Maze BFS/DFS and A*
# ======================================
# We'll generate a random maze each run.
# 'S' start, 'G' goal, '#' walls, '.' free

def generate_maze(rows: int, cols: int) -> List[List[str]]:
    if rows < 5:
        rows = 5
    if cols < 5:
        cols = 5
    # Start with all walls
    maze = [['#' for _ in range(cols)] for _ in range(rows)]

    # Convert to grid of cells at odd coordinates then carve with DFS backtracker
    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols

    # Ensure odd dimensions for better structure
    if rows % 2 == 0:
        rows -= 1
        maze = [row[:cols] for row in maze[:rows]]
    if cols % 2 == 0:
        cols -= 1
        maze = [row[:cols] for row in maze]

    # Initialize cells
    for r in range(1, rows, 2):
        for c in range(1, cols, 2):
            maze[r][c] = '.'

    stack: List[Tuple[int, int]] = [(1, 1)]
    visited_cells: Set[Tuple[int, int]] = {(1, 1)}
    directions = [(2, 0), (-2, 0), (0, 2), (0, -2)]
    while stack:
        cr, cc = stack[-1]
        neighbors = []
        for dr, dc in directions:
            nr, nc = cr + dr, cc + dc
            if 1 <= nr < rows and 1 <= nc < cols and (nr, nc) not in visited_cells:
                neighbors.append((nr, nc, dr, dc))
        if neighbors:
            nr, nc, dr, dc = random.choice(neighbors)
            # Carve wall between
            maze[cr + dr // 2][cc + dc // 2] = '.'
            maze[nr][nc] = '.'
            visited_cells.add((nr, nc))
            stack.append((nr, nc))
        else:
            stack.pop()

    # Place S and G on far points
    maze[1][1] = 'S'
    maze[rows - 2][cols - 2] = 'G'

    # Optionally punch a few random holes to vary density
    holes = max(1, (rows * cols) // 60)
    for _ in range(holes):
        r = random.randrange(1, rows - 1)
        c = random.randrange(1, cols - 1)
        maze[r][c] = '.' if maze[r][c] == '#' else maze[r][c]
    return maze


Coord = Tuple[int, int]


def maze_find_char(maze: List[List[str]], ch: str) -> Optional[Coord]:
    for r, row in enumerate(maze):
        for c, cell in enumerate(row):
            if cell == ch:
                return r, c
    return None


def maze_in_bounds(maze: List[List[str]], r: int, c: int) -> bool:
    return 0 <= r < len(maze) and 0 <= c < len(maze[0])


def maze_neighbors(maze: List[List[str]], r: int, c: int) -> List[Coord]:
    deltas = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    nbrs: List[Coord] = []
    for dr, dc in deltas:
        rr, cc = r + dr, c + dc
        if maze_in_bounds(maze, rr, cc) and maze[rr][cc] != '#':
            nbrs.append((rr, cc))
    return nbrs


def reconstruct_path(came_from: Dict[Coord, Coord], end: Coord) -> List[Coord]:
    path = [end]
    while path[-1] in came_from:
        path.append(came_from[path[-1]])
    path.reverse()
    return path


def draw_path_on_maze(maze: List[List[str]], path: List[Coord]) -> List[List[str]]:
    drawn = [row[:] for row in maze]
    for r, c in path:
        if drawn[r][c] in {'S', 'G', '#'}:
            continue
        drawn[r][c] = '*'
    return drawn


def bfs_solve(maze: List[List[str]]) -> Optional[List[Coord]]:
    start = maze_find_char(maze, 'S')
    goal = maze_find_char(maze, 'G')
    if start is None or goal is None:
        return None
    from collections import deque
    q = deque([start])
    visited = {start}
    came_from: Dict[Coord, Coord] = {}
    while q:
        cur = q.popleft()
        if cur == goal:
            return reconstruct_path(came_from, cur)
        for nxt in maze_neighbors(maze, *cur):
            if nxt not in visited:
                visited.add(nxt)
                came_from[nxt] = cur
                q.append(nxt)
    return None


def dfs_solve(maze: List[List[str]]) -> Optional[List[Coord]]:
    start = maze_find_char(maze, 'S')
    goal = maze_find_char(maze, 'G')
    if start is None or goal is None:
        return None
    stack = [start]
    visited = {start}
    came_from: Dict[Coord, Coord] = {}
    while stack:
        cur = stack.pop()
        if cur == goal:
            return reconstruct_path(came_from, cur)
        for nxt in maze_neighbors(maze, *cur):
            if nxt not in visited:
                visited.add(nxt)
                came_from[nxt] = cur
                stack.append(nxt)
    return None


def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar_solve(maze: List[List[str]]) -> Optional[List[Coord]]:
    start = maze_find_char(maze, 'S')
    goal = maze_find_char(maze, 'G')
    if start is None or goal is None:
        return None
    open_heap: List[Tuple[int, Coord]] = []
    heapq.heappush(open_heap, (0, start))
    g_score: Dict[Coord, int] = {start: 0}
    came_from: Dict[Coord, Coord] = {}
    in_open = {start}
    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == goal:
            return reconstruct_path(came_from, current)
        in_open.discard(current)
        for neighbor in maze_neighbors(maze, *current):
            tentative = g_score[current] + 1
            if tentative < g_score.get(neighbor, sys.maxsize):
                came_from[neighbor] = current
                g_score[neighbor] = tentative
                f_score = tentative + manhattan(neighbor, goal)
                if neighbor not in in_open:
                    heapq.heappush(open_heap, (f_score, neighbor))
                    in_open.add(neighbor)
    return None


def run_maze_search(algorithm: str) -> None:
    print("\nMaze size config (press Enter for defaults 11 11)")
    raw = input("rows cols: ").strip()
    if not raw:
        rows, cols = 11, 11
    else:
        try:
            parts = [int(x) for x in raw.split()]
            if len(parts) != 2:
                print("Invalid input, using defaults 11 11.")
                rows, cols = 11, 11
            else:
                rows, cols = parts
        except ValueError:
            print("Invalid input, using defaults 11 11.")
            rows, cols = 11, 11
    maze = generate_maze(rows, cols)
    print("\nMaze:")
    print_grid(maze)
    if algorithm == 'BFS':
        path = bfs_solve(maze)
    elif algorithm == 'DFS':
        path = dfs_solve(maze)
    elif algorithm == 'A*':
        path = astar_solve(maze)
    else:
        print("Unknown algorithm.")
        return
    if path is None:
        print(f"No path found with {algorithm}.")
    else:
        print(f"\n{algorithm} path length: {len(path)}")
        drawn = draw_path_on_maze(maze, path)
        print_grid(drawn)


# ==================
# Main menu wrapper
# ==================
def main() -> None:
    while True:
        print("\nChoose an option:")
        print("1) Maze Game using BFS/DFS")
        print("2) Maze Game using A*")
        print("3) Tic-tac-toe game (8x8, 4-in-a-row)")
        print("4) Quit")
        choice = input("> ").strip()
        if choice == '1':
            algo = input("Choose BFS or DFS: ").strip().upper()
            if algo not in {"BFS", "DFS"}:
                print("Invalid choice.")
            else:
                run_maze_search(algo)
        elif choice == '2':
            run_maze_search('A*')
        elif choice == '3':
            ttt_play_game()
        elif choice == '4':
            print("Goodbye!")
            break
        else:
            print("Invalid option. Try again.")


if __name__ == '__main__':
    main()


