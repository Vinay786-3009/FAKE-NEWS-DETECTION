"""
Generate a SYNTHETIC demonstration dataset for the Fake News Detection &
News Credibility Analyzer.

IMPORTANT
---------
This dataset is fully synthetic. Every article is generated from templates
written for this project. No real-world news claims are fabricated here and
labelled as factual. The synthetic "REAL"-style articles are neutral,
source-attributed reporting templates; the synthetic "FAKE"-style articles
use the sensational linguistic patterns that fake-news corpora typically
contain. The dataset exists to demonstrate the complete ML + NLP pipeline
when a public dataset (e.g. the Kaggle "Fake and Real News Dataset") cannot
be downloaded automatically.
"""

import random
from datetime import datetime, timedelta

import pandas as pd

RNG_SEED = 42
random.seed(RNG_SEED)

REAL_SUBJECTS = ["politics", "world", "business", "technology", "health", "sports"]

REAL_TOPICS = [
    ("government", ["parliament", "budget", "ministry", "legislation"]),
    ("economy", ["inflation", "markets", "trade", "employment"]),
    ("health", ["hospital", "vaccine programme", "public health", "clinic"]),
    ("technology", ["semiconductors", "software", "research", "engineering"]),
    ("world", ["diplomacy", "summit", "treaty", "border talks"]),
]

REAL_OPENERS = [
    "Officials confirmed on {date} that {topic} figures were released following a routine review.",
    "A report published on {date} describes how {topic} developments were documented by the agency.",
    "According to a statement issued on {date}, the department reviewed {topic} data for the quarter.",
    "The ministry announced on {date} that {topic} statistics had been verified by an independent auditor.",
    "Analysts said on {date} that {topic} indicators remained within the expected range, citing official records.",
]

REAL_MIDDLES = [
    "The announcement followed a month-long review in which the numbers were cross-checked against records from {detail}.",
    "A spokesperson said the findings in {detail} were based on filings submitted under existing disclosure rules.",
    "Researchers noted that the results, published in {detail}, are consistent with data collected over the previous year.",
    "The agency stated that further updates concerning {detail} would be provided during the next scheduled briefing.",
    "Independent observers who reviewed the materials in {detail} described the process as routine and orderly.",
]

REAL_CLOSERS = [
    "No unusual activity was reported in connection with the matter.",
    "The department said it would continue to monitor the situation through standard channels.",
    "Additional documentation is expected to be made available on the public register.",
    "The office declined to comment beyond the prepared statement.",
]

FAKE_HEADLINES = [
    "SHOCKING!!! {topic} secret plan EXPOSED - what they don't want you to know!!!",
    "You won't BELIEVE what insiders revealed about {topic} - banned truth!!!",
    "BOMBSHELL: {topic} conspiracy finally uncovered, mainstream media silent!!!",
    "MIRACLE cure for {topic} discovered - doctors HATE this one trick!!!",
    "BREAKING: {topic} scandal destroys everything - share before it's deleted!!!",
]

FAKE_OPENERS = [
    "WAKE UP people!!! {topic} is a total LIE and we have PROOF!!!",
    "Shocking new evidence about {topic} has just leaked and they are trying to BAN this article!!!",
    "Insiders whispered the TRUTH about {topic} and it is worse than you could ever imagine!!!",
    "The mainstream media will NEVER show you this about {topic} - 100% guaranteed!!!",
    "This one weird trick about {topic} changed everything and experts are FURIOUS!!!",
]

FAKE_MIDDLES = [
    "Anonymous sources claim that {detail} proves the entire thing was staged from the beginning!!!",
    "Documents that nobody can verify (for reasons that are OBVIOUS) show that {detail} was hidden from YOU!!!",
    "A former employee who wants to stay completely anonymous said the truth about {detail} is UNBELIEVABLE!!!",
    "Everyone is talking about how {detail} was secretly changed overnight and NOBODY is asking questions!!!",
    "They deleted the evidence about {detail} but we saved it - share this before it disappears!!!",
]

FAKE_CLOSERS = [
    "SHARE THIS before it gets taken down!!! WAKE UP!!!",
    "If you believe the official story you are being DECEIVED!!! Wake up sheeple!!!",
    "Comment your reaction NOW and tag everyone you know!!!",
    "The fact that this is not on the news proves EVERYTHING!!!",
]


def _date_random() -> str:
    start = datetime(2019, 1, 1)
    return (start + timedelta(days=random.randint(0, 1000))).strftime("%d-%b-%y")


def _make_real() -> dict:
    topic_name, details = random.choice(REAL_TOPICS)
    topic = topic_name if topic_name != "world" else "diplomatic"
    detail = random.choice(details)
    text = " ".join([
        random.choice(REAL_OPENERS).format(date=_date_random(), topic=topic),
        random.choice(REAL_MIDDLES).format(detail=detail),
        random.choice(REAL_CLOSERS),
    ])
    title = (f"Officials review {topic} figures in {detail} records"
             if random.random() < 0.5
             else f"{topic.capitalize()} update issued after {detail} review")
    return {
        "title": title,
        "text": text,
        "subject": random.choice(REAL_SUBJECTS),
        "date": _date_random(),
        "label": "REAL",
    }


def _make_fake() -> dict:
    topic = random.choice(["the government", "vaccines", "the election", "the media", "5G networks"])
    detail = random.choice(["the leaked memo", "the secret footage", "the banned report", "the deleted files"])
    title = random.choice(FAKE_HEADLINES).format(topic=topic)
    text = " ".join([
        random.choice(FAKE_OPENERS).format(topic=topic),
        random.choice(FAKE_MIDDLES).format(detail=detail),
        random.choice(FAKE_CLOSERS),
    ])
    return {
        "title": title,
        "text": text,
        "subject": random.choice(["politics", "world", "entertainment", "health", "conspiracies"]),
        "date": _date_random(),
        "label": "FAKE",
    }


def build(n_per_class: int = 600) -> pd.DataFrame:
    rows = [_make_real() for _ in range(n_per_class)]
    rows += [_make_fake() for _ in range(n_per_class)]
    random.shuffle(rows)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build()
    df.to_csv("data/news_dataset.csv", index=False)
    n_real = (df["label"] == "REAL").sum()
    n_fake = (df["label"] == "FAKE").sum()
    print(f"Saved data/news_dataset.csv with {len(df)} rows ({n_real} REAL / {n_fake} FAKE)")
