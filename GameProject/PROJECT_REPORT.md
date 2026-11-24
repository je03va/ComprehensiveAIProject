# Project Report: GameProject (Python Turtle AI/IT Education Games)

**Location:** `~/Desktop/AIProjects/GameProject/`
**Status as of 2026-11-15:** Both programs are complete and run; they pass a syntax check. The folder is not yet committed to git: it is untracked inside the parent `AIProjects` repo.
**Context:** This is a class assignment. Each file extends an example from class: the "Athletics turtle race" and the "Game-like Animation". A comment in the code mentions a "Part 1(b)", so the assignment probably has several parts.
**Tech:** Python 3 standard library only (`turtle`, `random`, `time`, `heapq`, `collections.deque`). Nothing to install.

---

## Files

| File | Lines | Purpose |
|---|---|---|
| `search_race.py` | 334 | **Packet Race:** BFS, DFS and A* race through the same maze |
| `data_structures_lesson.py` | 334 | **Data Structures Behind Your IT:** click-driven lessons on 5 data structures |
| `1115.mov` | ~100 MB | Screen recording (probably the demo video for submission) |

Run either program with `python3 <file>.py`. Each one opens a turtle window.

---

## 1. `search_race.py`: Packet Race (BFS vs DFS vs A*)

**Concept:** A network packet travels from a router (start, top-left) to a server (goal, bottom-right) through a "data center" grid maze. Three search algorithms race side by side, each on an identical copy of the maze.

**Structure:** one class, `PacketRaceGame`.
- `setup_screen()`: 1300×850 window, `tracer(0)` with manual `update()`. Sets up the 3 racers (BFS blue/turtle, DFS purple/arrow, A* green/classic) and the lesson text shown for each possible winner.
- **The search functions are generators.** `bfs`, `dfs` and `astar` each `yield (cell, path)` once per expanded node. `path` stays `None` until the goal is reached. This lets the main loop advance all three racers one step at a time, so the race is animated fairly.
  - BFS: FIFO `deque`.
  - DFS: LIFO list stack plus a visited set.
  - A*: `heapq` ordered by `(f = g + h, g, cell)` with a Manhattan-distance heuristic and `best_g` tracking.
  - Shared helpers: `neighbors()` (4 directions: right, down, left, up) and `build_path()` (follows parent links back from the goal).
- `make_maze(size, round_num)`: `random.seed(round_num)` makes every round reproducible. About 28% of cells become walls, 4 become "servers down" (they turn into walls after an alert) and 5 become "congested" (+3 wait steps). It uses BFS to regenerate the maze until a path exists.
- `run_round()`: each loop iteration advances every unfinished racer by one step. A racer in a congested cell waits. Explored cells are colored, and the final path is drawn when a racer finishes. **Round winner:** fewest steps, with ties broken by fewest nodes explored.
- `start_race()`: 4 rounds on mazes of 12, 14, 16 and 18 cells (the "400m"). **Champion:** most wins, with ties broken by fewest total steps. `show_final_banner()` draws a gold banner, the winner, a "what this teaches" lesson and a results table.

**Tunables:** `maze_pixels=360`, `congestion_delay=3`, `step_delay=0.03`, wall density `0.28`, and the `maze_sizes` list.

---

## 2. `data_structures_lesson.py`: Data Structures Behind Your IT

**Concept:** An interactive "education agent". The user clicks on-screen buttons to explore 5 data structures, each explained with an everyday IT analogy and an IT tip.

**Structure:** one class, `DataStructuresLesson`.
- **Button system:** `draw_button()` saves each button as `(x, y, w, h, action)`. `handle_click()` (registered with `onscreenclick`) runs the action whose rectangle contains the click. A `busy` flag ignores clicks during animations.
- **Typewriter title animation:** `typewriter()` reveals the title one letter at a time.
- **Redraw model:** every action updates its state, then calls `draw_lesson()`, which clears and redraws the whole screen.
- **Lessons:**

| Lesson | IT analogy | Demo buttons |
|---|---|---|
| Stack | Browser Back button / Undo | Visit Page (push), Back (pop), max 7 |
| Queue | Print queue / router buffer | Add Job (enqueue), Print (dequeue), max 6 |
| Hash Table | DNS cache | Lookup: hash = sum of character codes % 5 buckets, shows cache hit or miss |
| Tree | File system folders | Expand / Collapse (depth 1–3), drawn level by level |
| Graph | Internet routers (R1–R6) | Send Packet: BFS finds the fewest-hops route and highlights it in red |

- A "Menu" button returns to the main menu, which has one button per topic.

---

## Design choices common to both files
- Turtle graphics with `tracer(0)` and explicit `screen.update()` for smooth animation.
- One class per program, plus driver code at the bottom of the file.
- Beginner-friendly style: simple loops, comments in plain language, no external libraries.
- Both programs use an IT and networking theme: packets, routers, servers, DNS.

## Notes and possible next steps
- **Git:** `1115.mov` is about 100 MB, which is over GitHub's 100 MB file limit. Add it to `.gitignore`, or use Git LFS, before committing the folder.
- `search_race.py` uses the same seeds every time, so each round's winner is the same on every run. That is good for a demo. Remove `random.seed` to get varied races.
- The DFS racer's step count includes congestion waits. This is intended: steps measure time and nodes measure work.
- Possible extensions: a speed slider or keyboard control, letting the user choose the maze size, adding Dijkstra or greedy best-first search, and a quiz mode in the lesson app.
