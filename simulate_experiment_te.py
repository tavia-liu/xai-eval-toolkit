"""
Usage example:

export OPENAI_API_KEY=''
python simulate_experiment_te.py \
  --characteristics_file SURVEY_en.csv \
  --tasks_file PROBLEMS_en.csv \
  --ground_truth_file PROBLEMS_RESPONSES.csv \
  --slides_pdf ORIGINAL_VISUALIZATIONS_EN.pdf \
  --output_file responses.csv \
  --model gpt-4o \
  --max_slides 15
"""

import argparse
import base64
import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import openai
except ImportError:
    openai = None


def extract_group(pid: str) -> str:
    """Extract DE/IT/SSH from a participant_id."""
    if isinstance(pid, str):
        for part in pid.split("_"):
            if part in {"DE", "IT", "SSH"}:
                return part
    return ""


def render_slides(pdf_path: str, max_slides: int) -> List[str]:
    """Return up to ``max_slides`` pages from a PDF as base64-encoded PNGs."""
    images: List[str] = []
    if fitz is None:
        return images
    doc = fitz.open(pdf_path)
    for i in range(min(len(doc), max_slides)):
        pix = doc.load_page(i).get_pixmap(matrix=fitz.Matrix(2, 2))
        images.append(base64.b64encode(pix.tobytes("png")).decode("utf-8"))
    doc.close()
    return images


def describe_tasks(df: pd.DataFrame) -> pd.DataFrame:
    """Combine mushroom features and model prediction into one text field."""
    exclude = {"problem_id"}
    desc = []
    for _, row in df.iterrows():
        parts = [
            f"{c}: {v}"
            for c, v in row.items()
            if c not in exclude and str(v).strip().lower() not in {"nan", "n/a"}
        ]
        desc.append("; ".join(parts))
    df = df.copy()
    df["description"] = desc
    return df[["problem_id", "description"]]


def build_persona_from_survey(row: pd.Series) -> str:
    """
    Build a full persona string from all Q&A survey fields for one participant.
    Skips metadata like ID or timestamps.
    """
    exclude = {
        "candidate_id",
        "Start time",
        "End time",
        "Email address",
        "participant_id",
        "comment",
        "participant_group",
    }

    qa_pairs = []
    for col, val in row.items():
        if col not in exclude and str(val).strip() and str(val).strip().lower() not in {"nan", "n/a"}:
            qa_pairs.append(f"Q: {col}\nA: {val.strip()}")

    persona = (
        "You are a human participant who completed the following pre-survey:\n"
        + "\n".join(qa_pairs)
    )
    return persona


def call_llm(
    persona: str, group: str, images: List[str], task_desc: str, model: str, temp: float
) -> Tuple[str, str]:
    """Send a single prompt to the LLM and return (decision, certainty)."""

    def safe_text(x: str) -> str:
        # Remove characters that can't be encoded in latin-1
        if not isinstance(x, str):
            x = str(x)
        return x.encode("ascii", "ignore").decode("ascii")

    persona = safe_text(persona)
    task_desc = safe_text(task_desc)

    group_desc = {
        "DE": "a domain expert in mycology",
        "IT": "a data-visualisation student",
        "SSH": "a social sciences/humanities student",
    }.get(group, group)

    prompt = (
        f"Role-play as a human participant. You are {group_desc}.\n"
        f"Persona: {persona}\n\n"
        f"You have studied the explanation slides below. "
        f"Now a new mushroom is described: {task_desc}.\n"
        "Decide if it is edible or non-edible and select your certainty "
        "(definitely certain, moderately certain, I can't assess, "
        "moderately uncertain, definitely uncertain).\n"
        "Provide your answer in the form: decision | certainty"
    )

    if openai is not None and getattr(openai, "api_key", None):
        content = [{"type": "text", "text": prompt}] + [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
            for b64 in images
        ]
        try:
            resp = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": content},
                ],
                temperature=temp,
            )
            out = resp["choices"][0]["message"]["content"].strip()
            if "|" in out:
                decision, certainty = [s.strip() for s in out.split("|", 1)]
            else:
                decision, certainty = out, "I can't assess"
        except Exception:
            decision, certainty = "non-edible", "moderately certain"
    else:
        decision, certainty = "non-edible", "moderately certain"

    d = (
        decision.lower()
        .replace("eadible", "edible")
        .replace("non edible", "non-edible")
        .replace("non-eadible", "non-edible")
        .strip()
    )
    if d.startswith("non") and "edible" in d:
        d = "non-edible"
    elif "edible" in d:
        d = "edible"
    else:
        d = decision
    return d, certainty


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a direct full-survey TE simulation for XAI-FUNGI"
    )
    parser.add_argument("--characteristics_file", required=True)
    parser.add_argument("--tasks_file", required=True)
    parser.add_argument("--ground_truth_file", required=True)
    parser.add_argument("--slides_pdf", default="ORIGINAL_VISUALIZATIONS_EN.pdf")
    parser.add_argument("--output_file", default="responses.csv")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--max_slides", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    if not Path(args.slides_pdf).exists():
        raise FileNotFoundError(args.slides_pdf)
    slides = render_slides(args.slides_pdf, args.max_slides)

    gt = pd.read_csv(args.ground_truth_file, dtype=str, encoding="utf-8")
    participants = pd.read_csv(args.characteristics_file, dtype=str, encoding="utf-8")
    tasks = describe_tasks(pd.read_csv(args.tasks_file, dtype=str, encoding="utf-8"))

    valid_ids = set(gt["participant_id"].dropna())
    participants = participants[participants["participant_id"].isin(valid_ids)].copy()
    participants["group"] = participants["participant_id"].apply(extract_group)
    participants = participants[participants["group"].astype(bool)]
    participants = participants.fillna("")

    results = []
    for _, p in participants.iterrows():
        persona = build_persona_from_survey(p)
        for _, t in tasks.iterrows():
            decision, certainty = call_llm(
                persona, p["group"], slides, t["description"], args.model, args.temperature
            )
            results.append(
                (t["problem_id"], p["participant_id"], decision, certainty)
            )

    out_df = pd.DataFrame(
        results,
        columns=[
            "problem_id",
            "participant_id",
            "participant_decision_en",
            "participant_certainty_en",
        ],
    )
    out_df.to_csv(args.output_file, index=False)

    merged = pd.merge(
        gt, out_df, on=["problem_id", "participant_id"], suffixes=("_true", "_sim")
    )
    if len(merged):
        norm = (
            lambda s: s.lower()
            .strip()
            .replace("eadible", "edible")
            .replace("non-eadible", "non-edible")
            .replace("non edible", "non-edible")
        )
        dec_acc = (
            merged["participant_decision_en_true"].apply(norm)
            == merged["participant_decision_en_sim"].apply(norm)
        ).mean()
        cert_acc = (
            merged["participant_certainty_en_true"].str.lower().str.strip()
            == merged["participant_certainty_en_sim"].str.lower().str.strip()
        ).mean()
        print(f"Decision accuracy: {dec_acc:.2%}")
        print(f"Certainty accuracy: {cert_acc:.2%}")
    else:
        print("No overlapping participants for evaluation")


if __name__ == "__main__":
    if openai is not None and os.getenv("OPENAI_API_KEY"):
        openai.api_key = os.getenv("OPENAI_API_KEY")
    main()
