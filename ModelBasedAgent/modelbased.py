import argparse
import logging
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Tunable parameters
HALF_LIFE_MIN = 15          # threat score halves every 15 minutes with no new evidence
BLOCK_THRESHOLD = 8.0       # IP threat score that triggers a block
ALERT_THRESHOLD = 5.0       # IP threat score that triggers an alert on a failed login
SPRAY_USERS = 5             # distinct usernames from one IP within SPRAY_WINDOW => spraying
SPRAY_WINDOW = timedelta(minutes=60)
USER_FAIL_LIMIT = 5         # failures on one account within USER_WINDOW (plus a raised IP score)
USER_WINDOW = timedelta(minutes=10)
SLOW_FAILS = 8              # failures from a never-seen IP within SLOW_WINDOW => slow-and-low attack
SLOW_WINDOW = timedelta(hours=6)

BLOCKING_ACTIONS = ("block_ip", "lock_account")


@dataclass(frozen=True)
class Event:
    """What the agent can see with percepts."""
    ts: datetime
    user: str
    ip: str
    country: str
    success: bool


class IntrusionAgent:
    def __init__(self):
        self.ip_score = defaultdict(float)            # accumulated, decaying threat evidence per IP
        self.ip_last_ts = {}                          # when each score was last decayed
        self.ip_fail_times = defaultdict(deque)       # failures per IP (6h window)
        self.ip_targets = defaultdict(deque)          # (ts, username) targeted per IP (60min window)
        self.user_fail_times = defaultdict(deque)     # failures per account (10min window)
        self.profiles = {}                            # username -> usual countries / hours / IPs
        self.known_ips = set()                        # every IP ever seen for a legit user
        self.blocked_ips = set()
        self.locked_users = set()

    # ---------- model helpers ----------
    def learn(self, ev):
        p = self.profiles.setdefault(ev.user, {"countries": set(), "hours": set(), "ips": set()})
        p["countries"].add(ev.country)
        p["hours"].add(ev.ts.hour)
        p["ips"].add(ev.ip)
        self.known_ips.add(ev.ip)

    def _decay(self, ip, now):
        """Decay the threat score for this IP based on how long since the last event from it."""
        last = self.ip_last_ts.get(ip)
        if last is not None:
            minutes = (now - last).total_seconds() / 60
            self.ip_score[ip] *= 0.5 ** (minutes / HALF_LIFE_MIN)
        self.ip_last_ts[ip] = now

    @staticmethod
    def _prune(dq, now, window, key=lambda x: x):
        """Remove old events from a deque that are outside the given time window."""
        while dq and now - key(dq[0]) > window:
            dq.popleft()

    def anomaly(self, ev):
        """How unusual is this event for THIS user? (0 = normal, 4 = very unusual)"""
        p = self.profiles.get(ev.user)
        if p is None:
            return 4
        a = 0
        if ev.country not in p["countries"]:
            a += 2
        if not any(((ev.ts.hour + d) % 24) in p["hours"] for d in (-1, 0, 1)):
            a += 1
        if ev.ip not in p["ips"]:
            a += 1
        return a

    def snapshot(self, ev):
        """Small view of the model, used for the walkthrough output."""
        return {
            "ip_score": round(self.ip_score[ev.ip], 2),
            "ip_fails_6h": len(self.ip_fail_times[ev.ip]),
            "distinct_targets_60m": len({u for _, u in self.ip_targets[ev.ip]}),
            "user_fails_10m": len(self.user_fail_times[ev.user]),
            "ip_blocked": ev.ip in self.blocked_ips,
            "user_locked": ev.user in self.locked_users,
        }

    # ---------- the perceive, update, decide, and act loop ----------
    def update_model(self, ev):
        self._decay(ev.ip, ev.ts)
        prof = self.profiles.get(ev.user)
        ip_familiar = prof is not None and ev.ip in prof["ips"]

        if not ev.success:
            weight = 0.75 if ip_familiar else 1.5          # failures from a familiar IP are less suspicious
            if prof is None:
                weight += 1.0                              # username does not exist
            self.ip_score[ev.ip] += weight
            self.ip_fail_times[ev.ip].append(ev.ts)
            self.ip_targets[ev.ip].append((ev.ts, ev.user))
            self.user_fail_times[ev.user].append(ev.ts)
        elif ip_familiar:
            self.ip_score[ev.ip] *= 0.5                    # a normal success reduces suspicion

        self._prune(self.ip_fail_times[ev.ip], ev.ts, SLOW_WINDOW)
        self._prune(self.ip_targets[ev.ip], ev.ts, SPRAY_WINDOW, key=lambda x: x[0])
        self._prune(self.user_fail_times[ev.user], ev.ts, USER_WINDOW)

    def decide(self, ev):
        """Decide what to do with this event, based on the current model state."""
        score = self.ip_score[ev.ip]
        prof = self.profiles.get(ev.user)

        if ev.ip in self.blocked_ips:
            return "block_ip", "IP already blocked"
        if score >= BLOCK_THRESHOLD:
            return "block_ip", f"threat score {score:.1f} >= {BLOCK_THRESHOLD}"
        distinct = len({u for _, u in self.ip_targets[ev.ip]})
        if distinct >= SPRAY_USERS:
            return "block_ip", f"password spraying: {distinct} accounts targeted in 60 min"
        if (ev.ip not in self.known_ips
                and len(self.ip_fail_times[ev.ip]) >= SLOW_FAILS):
            return "block_ip", f"slow-and-low: {len(self.ip_fail_times[ev.ip])} failures from unknown IP in 6h"

        if ev.user in self.locked_users and (prof is None or ev.ip not in prof["ips"]):
            return "lock_account", "account is locked and IP is unfamiliar"

        if not ev.success:
            if len(self.user_fail_times[ev.user]) >= USER_FAIL_LIMIT and score >= 4:
                return "lock_account", f"{len(self.user_fail_times[ev.user])} failures on account in 10 min"
            if score >= ALERT_THRESHOLD:
                return "alert", f"elevated IP threat score {score:.1f}"
            return "log_only", "isolated failure"

        a = self.anomaly(ev)
        if a >= 4:
            return "lock_account", f"successful login highly unusual for user (anomaly {a}/4)"
        if a >= 3:
            return "alert", f"successful login unusual for user (anomaly {a}/4)"
        return "allow", f"normal for user (anomaly {a}/4)"

    def act(self, ev, action):
        if action == "block_ip":
            self.blocked_ips.add(ev.ip)
        elif action == "lock_account":
            self.locked_users.add(ev.user)
        elif action == "allow" and ev.success and self.anomaly(ev) < 2:
            self.learn(ev)                                 # only learn from clearly normal logins

    def step(self, ev):
        """Process one event: update model, decide action, and act on it."""
        self.update_model(ev)
        action, reason = self.decide(ev)
        self.act(ev, action)
        return action, reason


# Performance measure
def reward(is_attack, success, action):
    caught = action in BLOCKING_ACTIONS
    if is_attack:
        if caught:
            return 2                       # malicious event stopped
        if action == "alert":
            return 1                       # flagged but not stopped
        return -15 if success else -1      # breach vs. missed attempt
    if caught:
        return -5                          # legitimate user blocked/locked (false positive)
    if action == "alert":
        return -1                          # unnecessary alert
    return 0


# Environment simulation
USERS = {
    # user: (home country, active hours, [familiar IPs])
    "John":   ("US", (8, 17), ["10.0.1.11", "10.0.9.1"]),
    "Bob":    ("US", (9, 18), ["10.0.1.12", "10.0.9.1"]),
    "Jess":   ("US", (8, 16), ["10.0.1.13"]),
    "Dave":   ("US", (10, 19), ["10.0.1.14", "10.0.9.1"]),
    "Vince":  ("US", (9, 17), ["10.0.1.15"]),
    "Steph":  ("US", (7, 15), ["10.0.1.16", "10.0.9.1"]),
    "Grace":  ("US", (11, 19), ["10.0.1.17"]),
    "Linda":  ("US", (8, 17), ["10.0.1.18", "10.0.9.1"]),
    "Isaiah": ("US", (9, 18), ["10.0.1.19"]),
    "Jude":   ("US", (8, 16), ["10.0.1.20", "10.0.9.1"]),
}


def _legit_login(rng, user, day, with_typo=True):
    """Generate one normal login event for a user, optionally with a typo first."""
    country, (h0, h1), ips = USERS[user]
    ts = day + timedelta(hours=rng.randint(h0, h1), minutes=rng.randint(0, 59), seconds=rng.randint(0, 59))
    ip = rng.choice(ips)
    out = []
    if with_typo and rng.random() < 0.10:
        out.append((Event(ts - timedelta(seconds=rng.randint(20, 60)), user, ip, country, False), False, "typo"))
    out.append((Event(ts, user, ip, country, True), False, "normal"))
    return out


def generate_history(rng, day):
    """Clean past activity used to build each user's baseline profile."""
    events = []
    for user in USERS:
        for _ in range(15):
            events += [e for e, _, _ in _legit_login(rng, user, day - timedelta(days=rng.randint(1, 5)), False)]
    return sorted(events, key=lambda e: e.ts)


def generate_day(rng, day):
    """One simulated day: normal traffic + scripted attacks. Returns (Event, is_attack, scenario)."""
    stream = []
    for user in USERS:
        for _ in range(rng.randint(4, 7)):
            stream += _legit_login(rng, user, day)

    # Legit-but-tricky 1: forgetful user (Dave), 4 failures then success from a familiar IP
    t = day + timedelta(hours=11, minutes=30)
    for i in range(4):
        stream.append((Event(t + timedelta(seconds=20 * i), "Dave", "10.0.1.14", "US", False), False, "forgetful_user"))
    stream.append((Event(t + timedelta(seconds=100), "Dave", "10.0.1.14", "US", True), False, "forgetful_user"))

    # Legit-but-tricky 2: business traveler (Vince) logs in from a hotel in the UK (daytime)
    stream.append((Event(day + timedelta(hours=14), "Vince", "192.0.2.150", "GB", True), False, "traveler"))

    # Attack 1: brute force on John from one IP, 15 tries 5s apart, guesses right on #12
    t = day + timedelta(hours=10)
    for i in range(15):
        stream.append((Event(t + timedelta(seconds=5 * i), "John", "203.0.113.50", "CN", i == 11), True, "brute_force"))

    # Attack 2: password spraying, one IP tries 10 different accounts, #9 succeeds
    t = day + timedelta(hours=15, minutes=20)
    for i, user in enumerate(USERS):
        stream.append((Event(t + timedelta(seconds=20 * i), user, "198.51.100.77", "RU", i == 8), True, "spraying"))

    # Attack 3: account takeover of Jess with a stolen valid password at 3 AM from Brazil
    t = day + timedelta(hours=3, minutes=12)
    stream.append((Event(t, "Jess", "192.0.2.99", "BR", True), True, "takeover"))
    stream.append((Event(t + timedelta(minutes=2), "Jess", "192.0.2.99", "BR", True), True, "takeover"))

    # Attack 4: slow-and-low, one failure every 12 minutes alternating two accounts (John, Bob)
    t = day + timedelta(hours=12)
    for i in range(12):
        stream.append((Event(t + timedelta(minutes=12 * i), ["John", "Bob"][i % 2],
                             "198.51.100.200", "RU", False), True, "slow_and_low"))

    return sorted(stream, key=lambda x: x[0].ts)



# Run + report
def setup_logging():
    log = logging.getLogger("agent")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fh = logging.FileHandler("security_events.log", mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(fh)
    return log


def run(seed=42):
    """Run the simulation with a given random seed and optionally print a walkthrough of certain scenarios."""
    rng = random.Random(seed)
    log = setup_logging()
    day = datetime(2026, 10, 10)

    agent = IntrusionAgent()
    for ev in generate_history(rng, day):
        agent.learn(ev)                                    # build baseline profiles

    stream = generate_day(rng, day)
    attack_ips = {"203.0.113.50": "brute force", "198.51.100.77": "spraying",
                  "192.0.2.99": "takeover", "198.51.100.200": "slow-and-low"}

    t_axis, score_agent = [], []
    total_a = 0
    ip_series = defaultdict(lambda: ([], []))
    block_times = {}
    m = {"stopped": 0, "breaches": 0, "missed": 0, "false_pos": 0, "bad_alerts": 0}

    def tally(is_attack, success, action):
        if is_attack:
            if action in BLOCKING_ACTIONS:
                m["stopped"] += 1
            elif success and action != "alert":
                m["breaches"] += 1
            elif action != "alert":
                m["missed"] += 1
        else:
            if action in BLOCKING_ACTIONS:
                m["false_pos"] += 1
            elif action == "alert":
                m["bad_alerts"] += 1

    #Event loop: perceive, update, decide, act
    for ev, is_attack, scenario in stream:
        action, reason = agent.step(ev)

        total_a += reward(is_attack, ev.success, action)
        tally(is_attack, ev.success, action)
        t_axis.append(ev.ts)
        score_agent.append(total_a)

        if ev.ip in attack_ips:
            xs, ys = ip_series[ev.ip]
            xs.append(len(xs) + 1)                         # attempt number from this IP
            ys.append(agent.ip_score[ev.ip])
            if action == "block_ip" and ev.ip not in block_times:
                block_times[ev.ip] = xs[-1]

        msg = (f"user={ev.user} ip={ev.ip} country={ev.country} "
               f"{'OK ' if ev.success else 'FAIL'} -> {action} ({reason})")
        if action in ("alert",) + BLOCKING_ACTIONS:
            log.warning("ALERT " + msg)
        else:
            log.info(msg)

    # ---------------- summary ----------------
    print("\n================ RESULTS ================")
    print(f"Events processed: {len(stream)}")
    print(f"{'Performance score':28}{total_a:>10}")
    for key, label in [("stopped", "Malicious events stopped"), ("breaches", "Breaches (attack succeeded)"),
                       ("missed", "Failed attempts missed"), ("false_pos", "Legit users blocked"),
                       ("bad_alerts", "Unnecessary alerts")]:
        print(f"{label:28}{m[key]:>10}")
    print("\nIPs blocked:", sorted(agent.blocked_ips))
    print("Accounts locked:", sorted(agent.locked_users))
    print("Log written to security_events.log")

    # ---------------- graphs ----------------
    fig, axes = plt.subplots(3, 1, figsize=(11, 13))

    ax = axes[0]
    for ip, label in attack_ips.items():
        if ip in ip_series and ip != "192.0.2.99":
            xs, ys = ip_series[ip]
            ax.plot(xs, ys, marker="o", markersize=3, label=f"{label} ({ip})")
            if ip in block_times:
                ax.scatter([block_times[ip]], [ys[block_times[ip] - 1]], marker="X", s=90, zorder=5)
    ax.axhline(BLOCK_THRESHOLD, color="red", linestyle="--", label=f"block threshold ({BLOCK_THRESHOLD})")
    ax.set_title("Internal model: IP threat score per attempt (X = moment the IP was blocked)")
    ax.set_xlabel("attempt number from that IP")
    ax.set_ylabel("threat score")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(t_axis, score_agent, label="Model-based agent", linewidth=2)
    ax.set_title("Performance measure: cumulative score")
    ax.set_ylabel("score")
    ax.legend()

    ax = axes[2]
    cats = ["stopped", "breaches", "missed", "false_pos", "bad_alerts"]
    names = ["Attacks\nstopped", "Breaches", "Missed\nattempts", "Legit users\nblocked", "Unnecessary\nalerts"]
    ax.bar(range(len(cats)), [m[c] for c in cats], 0.5, label="Model-based")
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(names)
    ax.set_title("Outcomes")
    ax.legend()

    fig.tight_layout()
    fig.savefig("agent_results.png", dpi=130)
    print("Graphs written to agent_results.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run(seed=args.seed)