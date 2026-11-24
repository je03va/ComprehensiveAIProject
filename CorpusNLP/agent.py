"""
Corpus Analysis NLP Agent
Research problem: how public discussion and sentiment about AI coding assistants
(Copilot, ChatGPT, Claude, Cursor, ...) changes over time, plus emerging topics
and risk indicators (security, jobs, bugs/hallucinations, licensing).

"""
import argparse
import html
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote

import matplotlib
matplotlib.use("Agg")  # save PNGs without opening windows
import matplotlib.pyplot as plt
import nltk
import pandas as pd
from urllib.request import Request, urlopen
from urllib.error import URLError
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

for pkg in ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4', 'vader_lexicon']:
    nltk.download(pkg, quiet=True)

# ---------------- Settings (plain constants) ----------------
DATA_DIR = "data"
OUTPUT_DIR = "outputs"
START_MONTH = "2025-11"            # analysis window (inclusive)
END_MONTH = "2026-10"
TIME_PERIOD = "M"                  # "M" = month, "W" = week

SENTIMENT_DROP_THRESHOLD = 0.05    # alert if mean VADER falls by more than this
RISK_SPIKE_PCT = 50                # alert if a risk rate rises by more than this %
MIN_RISK_DOCS = 5                  # ...and at least this many documents mention it
MIN_KEYWORD_DOCS = 8               # min documents for a word to count as "emerging"

# Sources that cover the whole window, so they are used for trends over time.
# tech_news (RSS only keeps recent weeks) and gutenberg (1920 baseline) are
# compared as snapshots instead.
TREND_SOURCES = ["hn", "stackoverflow", "github", "arxiv"]
ALL_SOURCES = ["hn", "stackoverflow", "github", "arxiv", "tech_news", "gutenberg"]

ASSISTANTS = {
    "copilot": ["copilot"],
    "chatgpt": ["chatgpt", "openai", "codex"],
    "claude": ["claude", "anthropic"],
    "cursor": ["cursor"],
}

RELEVANCE_WORDS = ["copilot", "chatgpt", "claude", "cursor", "codex", "gpt", "llm", "llms", "ai", "openai", "gemini"]

RISK_KEYWORDS = {
    "security": ["security", "vulnerability", "exploit", "leak", "secret", "injection", "malware", "cve"],
    "jobs": ["job", "layoff", "junior", "hiring", "career", "unemployment"],
    "bugs_hallucination": ["bug", "hallucination", "hallucinate", "incorrect", "broken", "wrong"],
    "licensing": ["license", "licensing", "copyright", "lawsuit", "gpl", "attribution"],
}

# Words that appear everywhere in this corpus and hide the real topics
EXTRA_STOPWORDS = {"use", "using", "like", "get", "would", "one", "also", "make", "code",
                   "work", "thing", "think", "know", "really", "even", "much", "way",
                   "could", "want", "need", "see", "go", "still", "lot", "time",
                   "quot", "nbsp", "amp", "gt", "lt", "http", "https", "www", "com"}

#---------------- Starter tokenizers ----------------
class Electronic_Books:
    # show_preview added so the agent can fetch many URLs without printing each one
    def accessData(self, ebook_url, show_preview=True):
        try:
            req = Request(ebook_url, headers={'User-Agent': 'Mozilla/5.0 (class demo)'})
            with urlopen(req, timeout=30) as response:
                xtersRead = response.read()
        except (URLError, TimeoutError) as e:
            print('Could not access the eBook:', e)
            self.rawData = ''
            return
        rawData = xtersRead.decode("utf-8-sig", "ignore")
        if show_preview:
            print('Character Length = ', len(rawData))
            print('\n DISPLAY SOME TEXT: \n', rawData[:500])
        self.rawData = rawData

    def Sentence_tokenizer(self):
        self.sentences = sent_tokenize(self.rawData)
        print('\n Number of Sentences = ', len(self.sentences))

    def Word_tokenizer(self):
        self.words = word_tokenize(self.rawData)
        print('\n Number of Words = ', len(self.words))


def month_ranges(start=START_MONTH, end=END_MONTH):
    """Yield (label, start_datetime, end_datetime) for every month in the window."""
    for p in pd.period_range(start, end, freq="M"):
        begin = p.start_time.to_pydatetime().replace(tzinfo=timezone.utc)
        finish = p.end_time.to_pydatetime().replace(tzinfo=timezone.utc)
        yield str(p), begin, finish


def strip_html(text):
    return html.unescape(re.sub(r"<[^>]+>", " ", text or ""))  # drop tags, decode &#x27; etc.


# ---------------- Data collection: 6 sources, each document has a date ----------------
class CorpusAgent(Electronic_Books):

    # SOURCE 1: Hacker News comments (Algolia API) - developer opinions
    def fetch_hn(self):
        docs, seen = [], set()
        queries = ["copilot", "cursor ai", "claude code", "chatgpt code", "ai coding"]
        for label, begin, finish in month_ranges():
            for q in queries:
                url = ("https://hn.algolia.com/api/v1/search?tags=comment&hitsPerPage=20"
                       f"&query={quote(q)}&numericFilters=created_at_i>{int(begin.timestamp())},"
                       f"created_at_i<{int(finish.timestamp())}")
                self.accessData(url, show_preview=False)
                if not self.rawData:
                    continue
                for hit in json.loads(self.rawData).get("hits", []):
                    if hit["objectID"] in seen:
                        continue
                    seen.add(hit["objectID"])
                    docs.append({"source": "hn", "date": hit["created_at"][:10],
                                 "title": hit.get("story_title") or "",
                                 "text": strip_html(hit.get("comment_text")),
                                 "url": "https://news.ycombinator.com/item?id=" + hit["objectID"]})
            print(f"  hn {label}: {len(docs)} docs so far")
        return docs

    # SOURCE 2: Stack Overflow questions (Stack Exchange API) - practical problems
    def fetch_stackoverflow(self):
        docs, seen = [], set()
        for label, begin, finish in month_ranges():
            for q in ["copilot", "chatgpt", "cursor ai"]:
                url = ("https://api.stackexchange.com/2.3/search/advanced?site=stackoverflow"
                       f"&q={quote(q)}&fromdate={int(begin.timestamp())}&todate={int(finish.timestamp())}"
                       "&pagesize=30&sort=creation&filter=withbody")
                self.accessData(url, show_preview=False)
                if not self.rawData:
                    continue
                for item in json.loads(self.rawData).get("items", []):
                    if item["question_id"] in seen:
                        continue
                    seen.add(item["question_id"])
                    date = datetime.fromtimestamp(item["creation_date"], timezone.utc)
                    docs.append({"source": "stackoverflow", "date": date.strftime("%Y-%m-%d"),
                                 "title": item["title"], "text": strip_html(item.get("body"))[:2000],
                                 "url": item["link"]})
                time.sleep(0.5)  # be polite to the free API
            print(f"  stackoverflow {label}: {len(docs)} docs so far")
        return docs

    # SOURCE 3: GitHub issues (GitHub search API) - bug reports and feature requests
    def fetch_github(self):
        docs = []
        for label, begin, finish in month_ranges():
            q = (f'copilot OR cursor OR "claude code" in:title type:issue '
                 f'created:{begin:%Y-%m-%d}..{finish:%Y-%m-%d}')
            url = f"https://api.github.com/search/issues?q={quote(q)}&per_page=30"
            self.accessData(url, show_preview=False)
            if self.rawData:
                for item in json.loads(self.rawData).get("items", []):
                    docs.append({"source": "github", "date": item["created_at"][:10],
                                 "title": item["title"], "text": (item.get("body") or "")[:2000],
                                 "url": item["html_url"]})
            print(f"  github {label}: {len(docs)} docs so far")
            time.sleep(7)  # unauthenticated limit is 10 searches per minute
        return docs

    # SOURCE 4: arXiv abstracts - research view of LLM code generation
    def fetch_arxiv(self):
        docs = []
        ns = {"a": "http://www.w3.org/2005/Atom"}
        for label, begin, finish in month_ranges():
            query = (f'abs:"code generation" AND abs:"large language model" AND '
                     f'submittedDate:[{begin:%Y%m%d}0000 TO {finish:%Y%m%d}2359]')
            url = ("https://export.arxiv.org/api/query?max_results=25&search_query="
                   + quote(query))
            self.accessData(url, show_preview=False)
            if self.rawData:
                for entry in ET.fromstring(self.rawData).findall("a:entry", ns):
                    docs.append({"source": "arxiv",
                                 "date": entry.find("a:published", ns).text[:10],
                                 "title": " ".join(entry.find("a:title", ns).text.split()),
                                 "text": " ".join(entry.find("a:summary", ns).text.split()),
                                 "url": entry.find("a:id", ns).text})
            print(f"  arxiv {label}: {len(docs)} docs so far")
            time.sleep(3)  # arXiv asks for 3 seconds between requests
        return docs

    # SOURCE 5: Tech news RSS feeds (The Verge, TechCrunch, Ars Technica) - media coverage
    def fetch_tech_news(self):
        feeds = ["https://www.theverge.com/rss/ai-artificial-intelligence/index.xml",
                 "https://techcrunch.com/category/artificial-intelligence/feed/",
                 "https://arstechnica.com/ai/feed/"]
        docs = []
        for feed in feeds:
            self.accessData(feed, show_preview=False)
            if not self.rawData:
                continue
            root = ET.fromstring(self.rawData)
            for item in root.iter():
                tag = item.tag.split("}")[-1]
                if tag not in ("item", "entry"):
                    continue
                fields = {child.tag.split("}")[-1]: (child.text or "") for child in item}
                if "pubDate" in fields:          # RSS format
                    date = parsedate_to_datetime(fields["pubDate"])
                else:                            # Atom format
                    date = datetime.fromisoformat(fields.get("published", fields.get("updated"))[:19])
                link = fields.get("link") or (item.find("{http://www.w3.org/2005/Atom}link").get("href"))
                docs.append({"source": "tech_news", "date": date.strftime("%Y-%m-%d"),
                             "title": fields.get("title", "").strip(),
                             "text": strip_html(fields.get("description") or fields.get("summary")
                                                or fields.get("content", ""))[:2000],
                             "url": link})
            print(f"  tech_news {feed}: {len(docs)} docs so far")
        return docs

    # SOURCE 6: Project Gutenberg "R.U.R." (1920, robots replacing workers) - historical baseline
    def fetch_gutenberg(self):
        self.accessData("https://www.gutenberg.org/cache/epub/59112/pg59112.txt")
        if not self.rawData:
            return []
        # keep only the play: text between the Gutenberg START and END markers
        body = re.split(r"\*\*\* START OF.*?\*\*\*", self.rawData)[-1].split("*** END")[0]
        self.rawData = body
        self.Sentence_tokenizer()
        docs = []
        for i in range(0, len(self.sentences), 10):   # 10-sentence chunks = 1 document
            docs.append({"source": "gutenberg", "date": "1920-01-01", "title": "R.U.R. (Capek)",
                         "text": " ".join(self.sentences[i:i + 10]),
                         "url": "https://www.gutenberg.org/ebooks/59112"})
        return docs

    # Cache each source in data/<source>.json so runs are offline and repeatable
    def load_source(self, name, refresh=False):
        path = os.path.join(DATA_DIR, name + ".json")
        if os.path.exists(path) and not refresh:
            with open(path) as f:
                docs = json.load(f)
            print(f"Loaded {len(docs)} {name} documents from cache ({path})")
            return docs
        print(f"Fetching {name} from the web...")
        try:
            docs = getattr(self, "fetch_" + name)()
        except Exception as e:  # bad JSON/XML from a site should not stop the agent
            print(f"  error while fetching {name}: {e}")
            docs = []
        if docs:
            os.makedirs(DATA_DIR, exist_ok=True)
            with open(path, "w") as f:
                json.dump(docs, f, indent=1)
            print(f"Saved {len(docs)} {name} documents to {path}")
            return docs
        if os.path.exists(path):
            print(f"!! Fetch failed for {name}. Falling back to cached file {path}")
            with open(path) as f:
                return json.load(f)
        print(f"!! Fetch failed for {name} and no cache exists. Skipping this source.")
        return []

    def load_corpus(self, sources=ALL_SOURCES, refresh=False):
        """Load all sources into a single DataFrame, filter for relevance and English text."""
        rows = []
        for name in sources:
            rows += self.load_source(name, refresh)
        df = pd.DataFrame(rows).drop_duplicates(subset=["source", "url", "text"])
        df["date"] = pd.to_datetime(df["date"])
        df["text"] = (df["title"] + ". " + df["text"]).str.strip()
        df = df[df["text"].str.len() > 30]
        # Relevance filter: discussion sources must mention an AI assistant
        pattern = r"\b(?:" + "|".join(RELEVANCE_WORDS) + r")\b"
        relevant = df["text"].str.lower().str.contains(pattern) | ~df["source"].isin(["hn", "stackoverflow", "github"])
        # English filter: VADER/TextBlob only understand English (mostly ASCII letters)
        english = df["text"].apply(lambda t: sum(c.isascii() for c in t) / len(t) > 0.9)
        print(f"Kept {(relevant & english).sum()} of {len(df)} documents after relevance/English filters")
        return df[relevant & english].reset_index(drop=True)

    # Use the starter tokenizers to report corpus size per source
    def corpus_statistics(self, df):
        rows = []
        for name, group in df.groupby("source"):
            print(f"\n--- {name} ---")
            self.rawData = " ".join(group["text"])
            self.Sentence_tokenizer()
            self.Word_tokenizer()
            rows.append({"source": name, "documents": len(group),
                         "sentences": len(self.sentences), "words": len(self.words),
                         "first_date": group["date"].min().date(),
                         "last_date": group["date"].max().date()})
        return pd.DataFrame(rows)


# ---------------- Text cleaning pipeline ----------------
STOP_WORDS = set(stopwords.words("english")) | EXTRA_STOPWORDS
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    """lowercase -> remove URLs/HTML -> tokenize (starter tokenizer) -> drop
    stopwords, punctuation and numbers -> lemmatize"""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = strip_html(text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha() and len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return [t for t in tokens if t not in STOP_WORDS]


# ---------------- INNOVATION 3: Sentiment over time (VADER) ----------------
vader = SentimentIntensityAnalyzer()


def add_sentiment(df):
    """Add VADER and TextBlob sentiment scores, plus a period label for each document."""
    df["vader"] = df["text"].apply(lambda t: vader.polarity_scores(t)["compound"])
    df["textblob"] = df["text"].apply(lambda t: TextBlob(t).sentiment.polarity)
    df["period"] = df["date"].dt.to_period(TIME_PERIOD).astype(str)
    return df


def sentiment_over_time(df):
    overall = df.groupby("period").agg(documents=("vader", "size"), vader=("vader", "mean"),
                                       textblob=("textblob", "mean")).round(3)
    by_source = df.pivot_table(index="period", columns="source", values="vader",
                               aggfunc="mean").round(3)
    return overall, by_source


# ---------------- 4: Emerging topics and risk keywords ----------------
def mentions(tokens, words):
    return any(w in tokens for w in words)


def add_flags(df):
    """Add boolean flags for each risk category and assistant, based on keywords."""
    df["tokens"] = df["text"].apply(clean_text)
    token_sets = df["tokens"].apply(set)
    for category, words in RISK_KEYWORDS.items():
        df["risk_" + category] = token_sets.apply(lambda s: mentions(s, words))
    for tool, words in ASSISTANTS.items():
        df["tool_" + tool] = token_sets.apply(lambda s: mentions(s, words))
    return df


def rate_by_period(df, prefix):
    """% of documents in each period with a flag (risk_* or tool_*)."""
    cols = [c for c in df.columns if c.startswith(prefix)]
    rates = df.groupby("period")[cols].mean() * 100
    rates.columns = [c[len(prefix):] for c in cols]
    return rates.round(1)


def top_keywords(df, n=15):
    """Count the most common words in the corpus after cleaning."""
    counts = df["tokens"].explode().value_counts().head(n)
    return counts.rename_axis("keyword").reset_index(name="count")


def emerging_topics(df, n=10):
    """Compare each word's document share in the latest period with its average
    share in all earlier periods. A big ratio means the topic is emerging."""
    periods = sorted(df["period"].unique())
    if len(periods) < 2:
        return pd.DataFrame(columns=["keyword", "latest_docs", "latest_share_%", "earlier_share_%", "growth_x"])
    latest, earlier = df[df["period"] == periods[-1]], df[df["period"] != periods[-1]]
    latest_docs = latest["tokens"].apply(set).explode().value_counts()
    earlier_docs = earlier["tokens"].apply(set).explode().value_counts()
    rows = []
    for word, count in latest_docs.items():
        if count < MIN_KEYWORD_DOCS:
            continue
        latest_share = count / len(latest) * 100
        earlier_share = earlier_docs.get(word, 0) / len(earlier) * 100
        growth = latest_share / max(earlier_share, 0.5)  # 0.5% floor avoids divide-by-zero
        rows.append({"keyword": word, "latest_docs": int(count),
                     "latest_share_%": round(latest_share, 1),
                     "earlier_share_%": round(earlier_share, 1), "growth_x": round(growth, 2)})
    if not rows:
        return pd.DataFrame(columns=["keyword", "latest_docs", "latest_share_%", "earlier_share_%", "growth_x"])
    return pd.DataFrame(rows).sort_values("growth_x", ascending=False).head(n).reset_index(drop=True)


def example_text(docs, most="negative"):
    """Pick one concrete document to back up a finding."""
    if docs.empty:
        return ""
    row = docs.sort_values("vader", ascending=(most == "negative")).iloc[0]
    return f"[{row['source']} {row['date'].date()}] {row['text'][:300]} ({row['url']})"


# ---------------- Automated alerts ----------------
def find_alerts(df, overall, risk_rates, emerging):
    """Detect sudden drops in sentiment, spikes in risk mentions, and emerging topics."""
    alerts = []
    periods = list(overall.index)
    for prev, cur in zip(periods, periods[1:]):
        docs = df[df["period"] == cur]
        change = overall.loc[cur, "vader"] - overall.loc[prev, "vader"]
        if change < -SENTIMENT_DROP_THRESHOLD:
            alerts.append({"period": cur, "type": "sentiment_drop", "item": "overall",
                           "detail": f"mean VADER {overall.loc[prev, 'vader']:+.3f} -> {overall.loc[cur, 'vader']:+.3f}",
                           "change": round(change, 3), "example": example_text(docs)})
        for category in risk_rates.columns:
            before, now = risk_rates.loc[prev, category], risk_rates.loc[cur, category]
            n_docs = int(docs["risk_" + category].sum())
            pct = (now - before) / before * 100 if before > 0 else 100.0
            if pct > RISK_SPIKE_PCT and n_docs >= MIN_RISK_DOCS:
                alerts.append({"period": cur, "type": "risk_spike", "item": category,
                               "detail": f"{before:.1f}% -> {now:.1f}% of documents ({n_docs} docs)",
                               "change": round(pct, 1),
                               "example": example_text(docs[docs["risk_" + category]])})
    if periods:
        latest = df[df["period"] == periods[-1]]
        for _, row in emerging.head(3).iterrows():
            if row["growth_x"] >= 2:
                matching = latest[latest["tokens"].apply(lambda t: row["keyword"] in t)]
                alerts.append({"period": periods[-1], "type": "emerging_topic", "item": row["keyword"],
                               "detail": f"in {row['latest_share_%']}% of docs vs {row['earlier_share_%']}% before",
                               "change": row["growth_x"], "example": example_text(matching, most="positive")})
    return pd.DataFrame(alerts, columns=["period", "type", "item", "detail", "change", "example"])


# ----------------  Repeatable evaluation (VADER vs TextBlob) ----------------
def label(score):
    return "positive" if score > 0.05 else "negative" if score < -0.05 else "neutral"


def compare_sentiment_methods(df):
    rows = []
    for name, g in list(df.groupby("source")) + [("ALL", df)]:
        rows.append({"source": name, "documents": len(g),
                     "vader_mean": round(g["vader"].mean(), 3),
                     "textblob_mean": round(g["textblob"].mean(), 3),
                     "correlation": round(g["vader"].corr(g["textblob"]), 3),
                     "label_agreement_%": round((g["vader"].apply(label) == g["textblob"].apply(label)).mean() * 100, 1),
                     "vader_negative_%": round((g["vader"] < -0.05).mean() * 100, 1)})
    return pd.DataFrame(rows)


def analyze(df):
    """Run the full analysis pipeline on a DataFrame of documents."""
    df = add_flags(add_sentiment(df.copy()))
    trend = df[df["source"].isin(TREND_SOURCES)]
    if trend.empty:          # e.g. only snapshot sources were selected
        trend = df
    overall, by_source = sentiment_over_time(trend)
    risk_rates = rate_by_period(trend, "risk_")
    emerging = emerging_topics(trend)
    return {"docs": df, "overall": overall, "by_source": by_source,
            "risk_rates": risk_rates, "tool_rates": rate_by_period(trend, "tool_"),
            "keywords": top_keywords(trend), "emerging": emerging,
            "alerts": find_alerts(trend, overall, risk_rates, emerging),
            "evaluation": compare_sentiment_methods(df)}


# ----------------  CSV logging ----------------
def append_csv(table, path):
    table.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def log_results(results, run_label, sources, start, end):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    docs = results["docs"].copy()
    docs["analyzed_at"] = stamp
    cols = ["analyzed_at", "source", "date", "period", "title", "vader", "textblob"] + \
           [c for c in docs.columns if c.startswith("risk_")] + ["url"]
    docs[cols].to_csv(os.path.join(OUTPUT_DIR, f"documents_{run_label}.csv"), index=False)

    alerts = results["alerts"].copy()
    alerts.insert(0, "run", run_label)
    alerts.insert(0, "logged_at", stamp)
    append_csv(alerts, os.path.join(OUTPUT_DIR, "alerts_log.csv"))

    summary = pd.DataFrame([{"logged_at": stamp, "run": run_label, "sources": "|".join(sources),
                             "window": f"{start}..{end}", "documents": len(docs),
                             "mean_vader": round(results["overall"]["vader"].mean(), 3),
                             "alerts": len(alerts),
                             "top_emerging": ", ".join(results["emerging"]["keyword"].head(5))}])
    append_csv(summary, os.path.join(OUTPUT_DIR, "run_log.csv"))
    results["evaluation"].to_csv(os.path.join(OUTPUT_DIR, f"evaluation_{run_label}.csv"), index=False)


# ---------------- Figures for the report ----------------
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]


def style(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=12)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#e6e5e0")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", rotation=45)


def save_figures(results, run_label):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = lambda name: os.path.join(OUTPUT_DIR, f"{name}_{run_label}.png")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(results["overall"].index, results["overall"]["vader"], marker="o", lw=2, color=COLORS[0], label="VADER")
    ax.plot(results["overall"].index, results["overall"]["textblob"], marker="o", lw=2, color=COLORS[1], label="TextBlob")
    ax.axhline(0, color="#999", lw=1)
    style(ax, "Mean sentiment per month (trend sources)", "mean score")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out("sentiment_over_time"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, col in enumerate(results["by_source"].columns):
        ax.plot(results["by_source"].index, results["by_source"][col], marker="o", lw=2, color=COLORS[i], label=col)
    ax.axhline(0, color="#999", lw=1)
    style(ax, "Mean VADER sentiment per month, by source", "mean VADER compound")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout(); fig.savefig(out("sentiment_by_source"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, col in enumerate(results["risk_rates"].columns):
        ax.plot(results["risk_rates"].index, results["risk_rates"][col], marker="o", lw=2, color=COLORS[i], label=col)
    style(ax, "Risk indicators: % of documents mentioning each category", "% of documents")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout(); fig.savefig(out("risk_trends"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, col in enumerate(results["tool_rates"].columns):
        ax.plot(results["tool_rates"].index, results["tool_rates"][col], marker="o", lw=2, color=COLORS[i], label=col)
    style(ax, "Share of documents mentioning each assistant", "% of documents")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout(); fig.savefig(out("assistant_mentions"), dpi=150); plt.close(fig)

    kw = results["keywords"].iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(kw["keyword"], kw["count"], color=COLORS[0])
    ax.set_title("Top keywords after cleaning", loc="left", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out("top_keywords"), dpi=150); plt.close(fig)

    docs = results["docs"]
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, (name, g) in enumerate(docs.groupby("source")):
        ax.scatter(g["vader"], g["textblob"], s=10, alpha=0.5, color=COLORS[i % len(COLORS)], label=name)
    ax.set_xlabel("VADER compound"); ax.set_ylabel("TextBlob polarity")
    ax.set_title("VADER vs TextBlob on the same documents", loc="left", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, markerscale=2)
    fig.tight_layout(); fig.savefig(out("vader_vs_textblob"), dpi=150); plt.close(fig)


def write_report(results, stats, run_label):
    path = os.path.join(OUTPUT_DIR, f"findings_{run_label}.txt")
    with open(path, "w") as f:
        f.write(f"CORPUS ANALYSIS NLP AGENT - findings ({run_label}, {datetime.now():%Y-%m-%d %H:%M})\n\n")
        f.write("Corpus statistics (starter tokenizers)\n" + stats.to_string(index=False) + "\n\n")
        f.write("Sentiment over time\n" + results["overall"].to_string() + "\n\n")
        f.write("VADER by source and period\n" + results["by_source"].to_string() + "\n\n")
        f.write("Risk indicators (% of documents)\n" + results["risk_rates"].to_string() + "\n\n")
        f.write("Assistant mentions (% of documents)\n" + results["tool_rates"].to_string() + "\n\n")
        f.write("Top keywords\n" + results["keywords"].to_string(index=False) + "\n\n")
        f.write("Emerging topics (latest period vs earlier)\n" + results["emerging"].to_string(index=False) + "\n\n")
        f.write("VADER vs TextBlob evaluation\n" + results["evaluation"].to_string(index=False) + "\n\n")
        f.write(f"ALERTS ({len(results['alerts'])})\n")
        for _, a in results["alerts"].iterrows():
            f.write(f"- {a['period']} {a['type']} [{a['item']}]: {a['detail']}\n    e.g. {a['example']}\n")
    return path


def run_agent(sources=ALL_SOURCES, start=START_MONTH, end=END_MONTH, refresh=False, run_label="full"):
    agent = CorpusAgent()
    df = agent.load_corpus(sources, refresh)
    in_window = (df["date"] >= start) & (df["date"] < pd.Period(end, "M").end_time)
    df = df[in_window | (df["source"] == "gutenberg")]   # baseline keeps its 1920 date
    stats = agent.corpus_statistics(df)
    results = analyze(df)
    log_results(results, run_label, sources, start, end)
    save_figures(results, run_label)
    report = write_report(results, stats, run_label)

    print("\n=========== PROACTIVE INTELLIGENCE ===========")
    print(stats.to_string(index=False))
    print("\nSentiment over time:\n", results["overall"].to_string())
    print("\nEmerging topics:\n", results["emerging"].head(5).to_string(index=False))
    print("\nVADER vs TextBlob:\n", results["evaluation"].to_string(index=False))
    print(f"\n{len(results['alerts'])} ALERTS:")
    for _, a in results["alerts"].iterrows():
        print(f"  ! {a['period']} {a['type']} [{a['item']}] {a['detail']}")
    print(f"\nReport: {report}   Figures/logs: {OUTPUT_DIR}/")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Corpus Analysis NLP Agent")
    parser.add_argument("--refresh", action="store_true", help="re-download all sources")
    parser.add_argument("--sources", default=",".join(ALL_SOURCES))
    parser.add_argument("--start", default=START_MONTH)
    parser.add_argument("--end", default=END_MONTH)
    parser.add_argument("--label", default="full", help="name used for output files")
    args = parser.parse_args()
    run_agent(args.sources.split(","), args.start, args.end, args.refresh, args.label)
