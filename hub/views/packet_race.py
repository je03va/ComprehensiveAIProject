import ui
from registry import BY_KEY

ui.desktop_page(BY_KEY["packet_race"],
                "Click **Launch game window** and watch BFS, DFS and A* race a packet through four "
                "data-center mazes that grow each round (12 to 18 cells). Light colours show what each "
                "algorithm explored, the dark line is the route it found, and the final banner shows "
                "the champion by wins and total steps.")
