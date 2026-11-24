"""
Module registry: the single place that describes every AI product in the suite.

Each entry tells the hub
  - where the product lives (folder, relative to AIProjects/)
  - how it is integrated ("native" page in the hub, "streamlit" app embedded
    from its own process and virtual environment, or "desktop" window)
  - the title/description pane shown at the top of its page
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent      # .../AIProjects
HUB_DIR = ROOT / "hub"

THEME = "Information Technology"
SUITE_NAME = "IT Ops AI Suite"

SECTIONS = ["Intelligent Agents", "Learning Agents", "Knowledge & Language", "Games & Simulation"]

MODULES = [
    # ---------------- Intelligent Agents ----------------
    dict(
        key="simple_reflex", section="Intelligent Agents", icon=":material/battery_charging_full:",
        title="Battery Swap Reflex Agent", technique="Simple Reflex Agent",
        kind="native", folder="SimpleReflexAgent", entry="simple_reflex_agent_it.py",
        description="Monitors the batteries of four IT devices (UPS, core switch, edge router, IoT hub) "
                    "stored in SQLite and swaps any battery at or below a charge threshold. One "
                    "condition-action rule, current percept only.",
        peas={"Performance": "No outages (0% charge), little waste at swap time",
              "Environment": "4 IT devices in a SQLite table, random drain per step",
              "Actuators": "Battery-swap robot",
              "Sensors": "Charge sensor (%) on each device"},
        enhancements=["Threshold, steps and seed are adjustable live",
                      "Animated run inside the GUI (no separate matplotlib window)",
                      "Threshold sweep finds the best-scoring threshold automatically"],
    ),
    dict(
        key="model_based", section="Intelligent Agents", icon=":material/shield_lock:",
        title="Login Intrusion Monitor", technique="Model-based Reflex Agent",
        kind="native", folder="ModelBasedAgent", entry="modelbased.py",
        description="Watches a day of login events and keeps an internal model (decaying IP threat "
                    "scores, per-user login profiles, failure windows) to block brute force, password "
                    "spraying, account takeover and slow-and-low attacks without locking out real users.",
        peas={"Performance": "+2 attack stopped, -15 breach, -5 legit user blocked, -1 needless alert",
              "Environment": "Simulated company login stream with scripted attacks",
              "Actuators": "Block IP, lock account, raise alert, allow",
              "Sensors": "Login event: time, user, IP, country, success"},
        enhancements=["Thresholds and threat-score half-life are tunable from the GUI",
                      "Searchable event log with the agent's reason for every decision",
                      "Live tester: send your own login to the trained agent and see its internal model"],
    ),
    # ---------------- Learning Agents ----------------
    dict(
        key="knn_classifier", section="Learning Agents", icon=":material/troubleshoot:",
        title="RAG Failure Diagnosis Agent", technique="Learning Agent: kNN classifier",
        kind="native", folder="LearningAgent", entry="learningAgent.py",
        description="Diagnoses why a retrieval-augmented generation (RAG) answer failed (8 failure "
                    "types). Compares engineered text-overlap features against TF-IDF, picks k by "
                    "cross-validation, and flags low-confidence predictions for human review.",
        peas={"Performance": "Accuracy, macro precision / recall / F1",
              "Environment": "1,200 labelled RAG failure cases",
              "Actuators": "Predicted failure type + confidence + review alert",
              "Sensors": "Question, retrieved context, model answer, ground truth"},
        enhancements=["Diagnose any case typed into the GUI, with class probabilities",
                      "Load a random unseen test case and compare to its true label",
                      "Learning Element, Performance Element, Critic and Problem Generator on one screen"],
    ),
    dict(
        key="knn_regressor", section="Learning Agents", icon=":material/timer:",
        title="Transcode Time Predictor (kNN)", technique="Learning Agent: kNN regressor",
        kind="streamlit", folder="kNNRegressorAgent", entry="app.py", port=8601,
        description="Predicts how long a cloud video-transcoding job will take (utime) from the input "
                    "video and output settings, with tuned k, feature-set selection, confidence alerts "
                    "and a problem generator that suggests job types to test next.",
        peas={"Performance": "MAE, RMSE, R² on held-out jobs",
              "Environment": "UCI Online Video Characteristics & Transcoding Time dataset",
              "Actuators": "Predicted utime + confidence",
              "Sensors": "Codec, resolution, bitrate, frame statistics, output settings"},
        enhancements=["Runs in its own virtual environment, embedded in the suite"],
    ),
    dict(
        key="mlr", section="Learning Agents", icon=":material/stacked_line_chart:",
        title="MLR vs kNN Learning Agent", technique="Learning Agent: Multilinear Regression",
        kind="streamlit", folder="KnnMLR", entry="app.py", port=8602,
        description="Same transcoding problem, solved with Multilinear Regression and compared head to "
                    "head with the tuned kNN. Human-on-the-loop: finished jobs are fed back and both "
                    "models retrain.",
        peas={"Performance": "MAE, RMSE, R², share of predictions >50% off",
              "Environment": "55k measured transcoding jobs + simulated new batches",
              "Actuators": "Predicted utime from both models",
              "Sensors": "Job features; measured utime when a job finishes (feedback)"},
        enhancements=["Runs in its own virtual environment, embedded in the suite"],
    ),
    # ---------------- Knowledge & Language ----------------
    dict(
        key="expert_system", section="Knowledge & Language", icon=":material/policy:",
        title="Network Security Advisor", technique="Expert System (forward chaining)",
        kind="native", folder="ExpertSystem", entry="expert_system.py",
        description="Interviews you about a network configuration, forward-chains over a JSON "
                    "knowledge base of security rules, and explains every vulnerability it finds with "
                    "severity, confidence, the facts that triggered it, and hardening steps.",
        peas={"Performance": "Correct, explained findings ranked by severity",
              "Environment": "A network configuration described by the user",
              "Actuators": "Ranked findings + hardening recommendations",
              "Sensors": "Yes/no answers to the knowledge base's questions"},
        enhancements=["Point-and-click interview instead of terminal prompts",
                      "Overall risk score and 'near-miss' rules that are one fact away from firing",
                      "Knowledge acquisition and rule deletion from the GUI"],
    ),
    dict(
        key="chatbot", section="Knowledge & Language", icon=":material/forum:",
        title="Article Q&A Chatbot", technique="Chatbot (Gemini LLM + TTS)",
        kind="native", folder="src/article_qa", entry="app.py",
        description="A source-grounded chatbot: give it an article (text or PDF) and ask questions. "
                    "Gemini answers only from that source, and answers can be read aloud with "
                    "text-to-speech.",
        peas={"Performance": "Answers grounded in the source, no invented facts",
              "Environment": "User-supplied article text or PDF",
              "Actuators": "Plain-English answer, optional spoken audio",
              "Sensors": "Typed questions + source documents"},
        enhancements=["Chat window with history instead of one CLI call per question",
                      "Optional follow-up memory so you can ask 'what about that?'",
                      "Spoken answers play right in the page"],
    ),
    dict(
        key="corpus_nlp", section="Knowledge & Language", icon=":material/monitoring:",
        title="AI Coding Assistant Discussion Monitor", technique="NLP: Corpus Analysis Agent",
        kind="streamlit", folder="CorpusNLP", entry="app.py", port=8603,
        description="Tracks how developers talk about AI coding assistants across Hacker News, Stack "
                    "Overflow, GitHub, arXiv and tech news: sentiment over time (VADER vs TextBlob), "
                    "risk indicators, assistant mentions, emerging topics and alerts.",
        peas={"Performance": "Agreement between sentiment models, alert precision",
              "Environment": "6 public text sources, Nov 2025 - Oct 2026",
              "Actuators": "Dashboards and alerts",
              "Sensors": "Documents with dates and sources"},
        enhancements=["Runs in its own virtual environment, embedded in the suite"],
    ),
    dict(
        key="sentiment", section="Knowledge & Language", icon=":material/sentiment_satisfied:",
        title="Developer Sentiment Analyzer", technique="NLP: Comparative Sentiment",
        kind="streamlit", folder="NLPSentiment", entry="dashboard.py", port=8604,
        description="Can off-the-shelf sentiment models understand developers? Compares TextBlob, "
                    "VADER (plus a tech-lexicon tuned VADER) and RoBERTa on labelled Stack Overflow and "
                    "GitHub posts.",
        peas={"Performance": "Accuracy, macro F1, neutral-called-negative rate",
              "Environment": "Labelled software-engineering text datasets",
              "Actuators": "Sentiment labels per model",
              "Sensors": "Developer posts and comments"},
        enhancements=["Runs in its own virtual environment, embedded in the suite"],
    ),
    # ---------------- Games & Simulation ----------------
    dict(
        key="ds_lesson", section="Games & Simulation", icon=":material/school:",
        title="Data Structures Behind Your IT", technique="Game-like Animated Agent",
        kind="desktop", folder="GameProject", entry="data_structures_lesson.py",
        description="A click-driven animated lesson: stacks, queues, hash tables, trees and graphs, "
                    "each explained with an everyday IT analogy (browser Back button, print queue, DNS "
                    "cache, file system, routers).",
        peas={"Performance": "Learner reaches every lesson and sees each operation animated",
              "Environment": "Turtle graphics window",
              "Actuators": "Animations, IT tips",
              "Sensors": "Mouse clicks on lesson buttons"},
        enhancements=["Launched from the suite in its own window"],
    ),
    dict(
        key="packet_race", section="Games & Simulation", icon=":material/route:",
        title="Packet Race: BFS vs DFS vs A*", technique="Simulation Race Game (search)",
        kind="desktop", folder="GameProject", entry="search_race.py",
        description="Three search algorithms race a packet from the router to the server through the "
                    "same data-center maze over four rounds with congestion. Shows how each one explores "
                    "and who finds the shortest route.",
        peas={"Performance": "Wins and total steps per algorithm",
              "Environment": "Growing grid mazes with congested cells",
              "Actuators": "Each racer's next move",
              "Sensors": "Neighbouring cells, walls, distance to goal (A*)"},
        enhancements=["Launched from the suite in its own window"],
    ),
]

BY_KEY = {m["key"]: m for m in MODULES}


def module_dir(m):
    return ROOT / m["folder"]
