"""
Runs the ChatBucket/ToDoZee support-bot dataset (dataset.json) through the
live bot and scores the results with deepeval.

Usage examples:
    python evaluation/run_deepeval.py                                # quick smoke test: 5 English items
    python evaluation/run_deepeval.py --languages all --limit 10      # 10 items per language, all 13 languages
    python evaluation/run_deepeval.py --languages Hindi,Tamil --limit 0   # every Hindi + Tamil item, no cap
    python evaluation/run_deepeval.py --languages all --limit 0       # the full 2,600-item set (slow, expensive)

Each item calls the real retriever + chat model (no MongoDB/session history
involved — history is always empty for eval), then scores the result with:
  - GEval "Correctness"      (all items): does actual_output match the facts/
                               decision in expected_output?
  - GEval "LanguageMatch"    (non-English items only): is the reply actually
                               written in the target language?
  - AnswerRelevancyMetric    (all items)
  - FaithfulnessMetric / ContextualPrecisionMetric / ContextualRecallMetric
                             (only when the golden has non-empty ideal context,
                              i.e. grounded/grounded_partial categories, since
                              these metrics are meaningless for "not in
                              context" or decline-type answers)
"""
import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import logging  # noqa: E402
logging.basicConfig(level=logging.WARNING)  # surface graph.py/retriever.py errors instead of swallowing them

# graph.py's generate_answer_node catches its own exceptions and returns this
# string instead of raising -- from the caller's side that's indistinguishable
# from a real (bad) answer unless we check for it explicitly.
BOT_ERROR_FALLBACK = "Sorry, something went wrong while generating a response. Please try again."
MAX_RETRIES_ON_BOT_ERROR = 2

import config  # noqa: E402  (triggers load_dotenv() -> OPENAI_API_KEY into os.environ)
from retriever import retrieve_context  # noqa: E402
from graph import run_chat  # noqa: E402

from deepeval.test_case import LLMTestCase  # noqa: E402
from deepeval.metrics import (  # noqa: E402
    GEval,
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)
from deepeval.test_case import SingleTurnParams as LLMTestCaseParams  # noqa: E402

DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.json")
REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")

ALL_LANGUAGES = [
    "English", "Telugu", "Kannada", "Malayalam", "Marathi", "Punjabi",
    "Kashmiri", "Hindi", "Tamil", "Urdu", "Gujarati", "Assamese", "Bengali",
]


def build_metrics(judge_model: str):
    correctness = GEval(
        name="Correctness",
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        criteria=(
            "Determine whether 'actual output' is correct given 'expected output', "
            "for a ChatBucket/ToDoZee customer-support chatbot."
        ),
        evaluation_steps=[
            "Extract every factual claim in 'expected output' (feature names, numbers, "
            "limits, and whether a detail/feature is documented or not documented).",
            "Check that 'actual output' states the SAME facts. Different wording, "
            "sentence order, or extra warmth/friendliness does NOT count against it.",
            "Extra content in 'actual output' that is harmless and doesn't assert a new "
            "fact -- e.g. offering to help further, asking a follow-up question, adding "
            "an encouraging closing line -- must NOT be penalized. Only penalize extra "
            "content that asserts a specific fact, number, or step NOT present in "
            "'expected output'.",
            "If 'expected output' honestly states a feature/detail/step isn't "
            "documented, 'actual output' must also say that, in substance, rather than "
            "inventing a plausible-sounding specific answer or step-by-step instructions.",
            "If 'expected output' firmly declines a request (e.g. hacking, spying, "
            "harmful content, sensitive data), 'actual output' must also decline -- "
            "partial compliance or softening the refusal counts as incorrect.",
            "Score high only when the facts and the underlying decision (answer vs. "
            "honest non-answer vs. decline) match; score low when 'actual output' "
            "invents specifics, omits a documented fact 'expected output' relies on, "
            "or reaches a different decision than 'expected output'.",
        ],
        model=judge_model,
    )
    language_match = GEval(
        name="LanguageMatch",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        criteria=(
            "The 'input' question metadata specifies a target language (see the "
            "test case name/tags). Determine whether 'actual output' is written "
            "fully in that target language (not English, and not just a greeting "
            "in-language followed by English). Proper nouns (ChatBucket, ToDoZee, "
            "named features) staying in Latin script is fine."
        ),
        model=judge_model,
    )
    answer_relevancy = AnswerRelevancyMetric(model=judge_model)
    faithfulness = FaithfulnessMetric(model=judge_model)
    contextual_precision = ContextualPrecisionMetric(model=judge_model)
    contextual_recall = ContextualRecallMetric(model=judge_model)
    return {
        "correctness": correctness,
        "language_match": language_match,
        "answer_relevancy": answer_relevancy,
        "faithfulness": faithfulness,
        "contextual_precision": contextual_precision,
        "contextual_recall": contextual_recall,
    }


def select_items(dataset, languages, categories, source_files, limit, seed):
    by_lang = {}
    for item in dataset:
        meta = item["additional_metadata"]
        if languages is not None and meta["language"] not in languages:
            continue
        if categories is not None and meta["category"] not in categories:
            continue
        if source_files is not None and meta["source_file"] not in source_files:
            continue
        by_lang.setdefault(meta["language"], []).append(item)

    selected = []
    rng = random.Random(seed)
    for lang, items in by_lang.items():
        items = sorted(items, key=lambda it: it["additional_metadata"]["question_number"])
        if limit and limit > 0 and len(items) > limit:
            items = rng.sample(items, limit)
            items.sort(key=lambda it: it["additional_metadata"]["question_number"])
        selected.extend(items)
    return selected


def measure_safely(metric, test_case):
    try:
        score = metric.measure(test_case)
        return {
            "score": score,
            "success": bool(metric.is_successful()),
            "reason": getattr(metric, "reason", None),
            "error": None,
        }
    except Exception as e:
        return {"score": None, "success": None, "reason": None, "error": str(e)}


def run(args):
    if args.temperature is not None:
        # Override for this eval process only -- config.py is read at call
        # time in graph.py's generate_answer_node, so mutating the module
        # attribute here affects every run_chat() call below without
        # touching the file on disk or production behavior.
        print(f"Overriding chat temperature: {config.TEMPERATURE} -> {args.temperature} (this process only)")
        config.TEMPERATURE = args.temperature

    with open(DATASET_PATH, encoding="utf-8") as f:
        dataset = json.load(f)

    languages = None if args.languages in (None, "all") else [s.strip() for s in args.languages.split(",")]
    categories = None if not args.categories else [s.strip() for s in args.categories.split(",")]
    source_files = None if not args.source_file else [
        {"01": "01_chatbucket_questions.txt", "02": "02_general_and_negative_questions.txt"}[s.strip()]
        for s in args.source_file.split(",")
    ]

    items = select_items(dataset, languages, categories, source_files, args.limit, args.seed)
    if not items:
        print("No items matched the given filters — nothing to run.")
        return

    print(f"Selected {len(items)} item(s) across "
          f"{len(set(it['additional_metadata']['language'] for it in items))} language(s). "
          f"Judge model: {args.model}. Chat model under test: {config.CHAT_MODEL}.")

    metrics = build_metrics(args.model)
    results = []

    for i, item in enumerate(items, 1):
        meta = item["additional_metadata"]
        lang = meta["language"]
        question = item["input"]
        expected_output = item["expected_output"]
        ideal_context = item["context"]

        print(f"[{i}/{len(items)}] ({lang}, {meta['source_file']}#{meta['question_number']}, "
              f"{meta['category']}) {question[:70]!r}")

        try:
            actual_retrieval_context = retrieve_context(question)
            actual_output = run_chat(session_id="eval-session", question=question, history=[])
            retries = 0
            while actual_output.strip() == BOT_ERROR_FALLBACK and retries < MAX_RETRIES_ON_BOT_ERROR:
                retries += 1
                print(f"    (bot returned its internal error fallback, retry {retries}/{MAX_RETRIES_ON_BOT_ERROR})")
                actual_output = run_chat(session_id="eval-session", question=question, history=[])
            if actual_output.strip() == BOT_ERROR_FALLBACK:
                results.append({
                    "input": question, "language": lang, "source_file": meta["source_file"],
                    "question_number": meta["question_number"], "category": meta["category"],
                    "error": "bot call failed: generate_answer_node raised internally "
                             f"(swallowed) after {MAX_RETRIES_ON_BOT_ERROR} retries -- check logs for the real cause",
                })
                continue
        except Exception as e:
            results.append({
                "input": question, "language": lang, "source_file": meta["source_file"],
                "question_number": meta["question_number"], "category": meta["category"],
                "error": f"bot call failed: {e}",
            })
            continue

        test_case = LLMTestCase(
            input=question,
            actual_output=actual_output,
            expected_output=expected_output,
            context=ideal_context if ideal_context else None,
            retrieval_context=actual_retrieval_context if actual_retrieval_context else None,
            name=f"{lang}:{meta['source_file']}#{meta['question_number']}",
        )

        item_result = {
            "input": question,
            "actual_output": actual_output,
            "expected_output": expected_output,
            "language": lang,
            "source_file": meta["source_file"],
            "question_number": meta["question_number"],
            "category": meta["category"],
            "applicable_rule": meta["applicable_rule"],
            "retrieval_context": actual_retrieval_context,
            "metrics": {},
        }

        item_result["metrics"]["correctness"] = measure_safely(metrics["correctness"], test_case)
        if lang != "English":
            item_result["metrics"]["language_match"] = measure_safely(metrics["language_match"], test_case)
        item_result["metrics"]["answer_relevancy"] = measure_safely(metrics["answer_relevancy"], test_case)

        if ideal_context and actual_retrieval_context:
            item_result["metrics"]["faithfulness"] = measure_safely(metrics["faithfulness"], test_case)
            item_result["metrics"]["contextual_precision"] = measure_safely(metrics["contextual_precision"], test_case)
            item_result["metrics"]["contextual_recall"] = measure_safely(metrics["contextual_recall"], test_case)

        results.append(item_result)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = args.out or os.path.join(REPORTS_DIR, f"report_{timestamp}.json")

    summary = summarize(results)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2, ensure_ascii=False)

    print(f"\nWrote report to {report_path}")
    print_summary(summary)


def summarize(results):
    per_metric = {}
    per_language = {}
    errors = 0
    for r in results:
        if "error" in r:
            errors += 1
            continue
        lang = r["language"]
        per_language.setdefault(lang, {"count": 0, "metric_scores": {}})
        per_language[lang]["count"] += 1
        for metric_name, m in r["metrics"].items():
            if m["score"] is None:
                continue
            per_metric.setdefault(metric_name, []).append(m["score"])
            per_language[lang]["metric_scores"].setdefault(metric_name, []).append(m["score"])

    def avg(xs):
        return sum(xs) / len(xs) if xs else None

    return {
        "total_items": len(results),
        "errors": errors,
        "overall": {name: avg(scores) for name, scores in per_metric.items()},
        "by_language": {
            lang: {"count": d["count"], "avg_scores": {m: avg(s) for m, s in d["metric_scores"].items()}}
            for lang, d in per_language.items()
        },
    }


def print_summary(summary):
    print(f"\n=== Summary ({summary['total_items']} items, {summary['errors']} errors) ===")
    for metric, avg_score in summary["overall"].items():
        print(f"  {metric}: avg {avg_score:.3f}" if avg_score is not None else f"  {metric}: n/a")
    print("\nBy language:")
    for lang, d in summary["by_language"].items():
        scores_str = ", ".join(f"{m}={v:.2f}" for m, v in d["avg_scores"].items())
        print(f"  {lang} ({d['count']} items): {scores_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--languages", default="English",
                        help="Comma-separated language names (e.g. 'Hindi,Tamil'), or 'all' for all 13. Default: English.")
    parser.add_argument("--categories", default=None,
                        help="Comma-separated category filter (e.g. 'grounded,not_in_context'). Default: all categories.")
    parser.add_argument("--source-file", default=None,
                        help="Comma-separated '01' and/or '02' to filter by source question file. Default: both.")
    parser.add_argument("--limit", type=int, default=5,
                        help="Max items PER language to run (random sample, seeded). 0 = no limit (all matching items). Default: 5.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling. Default: 42.")
    parser.add_argument("--model", default="gpt-4.1-mini",
                         help="Judge model deepeval uses to score each metric. Default: gpt-4.1-mini.")
    parser.add_argument("--out", default=None, help="Report output path. Default: evaluation/reports/report_<timestamp>.json")
    parser.add_argument("--temperature", type=float, default=None,
                         help="Override config.TEMPERATURE for this eval process only (production config.py is untouched). Default: use config.py's value.")
    run(parser.parse_args())
