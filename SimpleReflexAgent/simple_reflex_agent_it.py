import random
import sqlite3
import matplotlib.pyplot as plt

THRESHOLD = 15
STEPS = 100

# PEAS:
# Performance: high uptime (no battery hits 0%) and little waste (don't replace a battery that still has lots of charge)
# Environment: 4 IT devices stored in a SQLite database; each battery drains by a random amount up to its own max rate
# Actuators: battery-swap robot
# Sensors: battery charge sensor (%) on every device

DEVICES = [  # (name, type, max drain % per step)
    ("UPS-Rack-A", "Server UPS", 3),
    ("Core-Switch", "Network", 5),
    ("Edge-Router", "Network", 7),
    ("Sensor-Hub", "IoT", 12),
]


class SimpleReflexAgent:
    def __init__(self):
        self.db = sqlite3.connect("batteries.db")
        self.db.executescript("""
            DROP TABLE IF EXISTS batteries; DROP TABLE IF EXISTS log;
            CREATE TABLE batteries (name TEXT PRIMARY KEY, type TEXT, charge INTEGER, max_drain INTEGER);
            CREATE TABLE log (step INTEGER, battery TEXT, charge INTEGER, action TEXT);""")
        self.db.executemany("INSERT INTO batteries VALUES (?, ?, 100, ?)", DEVICES)
        self.step = 0
        self.action = {}
        plt.ion()
        self.fig, (self.ax, self.bar) = plt.subplots(1, 2, figsize=(12, 4.5))

    def environment(self):
        """ every battery drains by its own random amount."""
        for name, max_drain in self.db.execute("SELECT name, max_drain FROM batteries").fetchall():
            self.db.execute("UPDATE batteries SET charge = MAX(0, charge - ?) WHERE name = ?",
                            (random.randint(1, max_drain), name))

    def perception_decision(self):
        """Sensors read every battery; the same single rule is applied to each."""
        self.step += 1
        for name, charge in self.db.execute("SELECT name, charge FROM batteries").fetchall():
            if charge <= THRESHOLD:  # one rule, current reading only
                self.action[name] = "replace"
                self.db.execute("UPDATE batteries SET charge = 100 WHERE name = ?", (name,))
            else:
                self.action[name] = "idle"
            # log the reading BEFORE the swap so the graph shows the dip
            self.db.execute("INSERT INTO log VALUES (?, ?, ?, ?)",
                            (self.step, name, charge, self.action[name]))
        self.db.commit()

    def performance_measure(self):
        outages = self.db.execute("SELECT COUNT(*) FROM log WHERE charge = 0").fetchone()[0]
        avg = self.db.execute("SELECT AVG(charge) FROM log WHERE action = 'replace'").fetchone()[0] or 0
        return max(0, 100 - 5 * outages - 0.3 * avg), outages, avg

    def draw(self):
        score, outages, _ = self.performance_measure()
        self.ax.clear()
        self.bar.clear()
        for name, _, _ in DEVICES:
            rows = self.db.execute("SELECT charge FROM log WHERE battery = ? ORDER BY step DESC LIMIT 50",
                                   (name,)).fetchall()
            data = [r[0] for r in reversed(rows)]
            self.ax.plot(range(self.step - len(data) + 1, self.step + 1), data, lw=2, label=name)
        self.ax.axhline(THRESHOLD, color="r", ls="--", lw=1)
        self.ax.set_ylim(0, 105)
        self.ax.set_xlabel("Time step")
        self.ax.set_ylabel("Battery %")
        self.ax.legend(loc="upper right", fontsize=8)
        self.ax.set_title(f"Step {self.step} | Outages={outages} | Score={score:.1f}")
        swaps = dict(self.db.execute("SELECT battery, COUNT(*) FROM log WHERE action='replace' GROUP BY battery"))
        self.bar.bar([d[0] for d in DEVICES], [swaps.get(d[0], 0) for d in DEVICES], color="tab:green")
        self.bar.set_title("Battery swaps per device")
        self.bar.tick_params(axis="x", labelrotation=20, labelsize=8)
        self.fig.tight_layout()
        self.fig.canvas.draw()
        plt.pause(0.01)


agent = SimpleReflexAgent()
for _ in range(STEPS):
    agent.environment()
    agent.perception_decision()
    agent.draw()

score, outages, avg = agent.performance_measure()
print(f"Performance summary (threshold {THRESHOLD}%)")
print(f"{'Device':<13}{'Swaps':>6}{'Outages':>9}{'Lowest %':>10}{'Avg % at swap':>15}")
for row in agent.db.execute("""SELECT battery, SUM(action='replace'), SUM(charge=0), MIN(charge),
                               AVG(CASE WHEN action='replace' THEN charge END)
                               FROM log GROUP BY battery"""):
    print(f"{row[0]:<13}{row[1]:>6}{row[2]:>9}{row[3]:>10}{(row[4] or 0):>15.1f}")
print(f"Total outages: {outages} | Average charge left at swap: {avg:.1f}%")
print(f"Performance score: {score:.1f}")
plt.ioff()  # keep the final graph open
plt.show()