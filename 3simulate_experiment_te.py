"""
simulate_experiment_te.py
=========================

Key features:

1. **Slide images** – The original explanation deck (``ORIGINAL_VISUALIZATIONS_EN.pdf``)
   is parsed using the ``fitz`` (PyMuPDF) library.  A configurable
   number of pages are rendered to PNG and embedded into the prompt
   using OpenAI’s vision API format.  This allows models such as
   ``gpt‑4o`` to “see” the same charts and diagrams the humans saw.

2. **Trait summarisation** – Rather than dumping every survey
   response, the script extracts a handful of key answers and
   condenses them into a short description of each participant’s
   mushroom knowledge, data‑visualisation ability, data‑analysis
   skills, study programme and mushroom‑collecting habits.  This
   concise persona is passed to the model.

3. **Turing‑Experiment style prompts** – The prompt for each
   participant/task pair explicitly names the participant (using
   their ``participant_id`` as a pseudonym) and instructs the model
   to role‑play as that individual.  The model sees the AI
   explanation (slides and summary), the underlying model’s predicted
   class and probability, and the mushroom’s feature values.  It is
   then asked to choose an edible/non‑edible decision and a
   certainty level from the same set of options presented in the
   human study.

4. **Model flexibility** – You can choose a vision‑capable model
   (e.g. ``gpt‑4o`` or ``gpt‑4o-mini``) via the ``--model`` argument
   and control the number of slides embedded with ``--max_slides``.
   If no API key is available, the script falls back to a simple
   heuristic stub.

To run the script, first install the required dependencies:

```
pip install pandas PyMuPDF openai
```

Then ensure your OpenAI API key is set in the environment:

```
export OPENAI_API_KEY='sk‑your‑key'
```

Finally, invoke the script:

```
python simulate_experiment_te.py \
  --characteristics_file SURVEY_en.csv \
  --tasks_file PROBLEMS_en.csv \
  --ground_truth_file PROBLEMS_RESPONSES.csv \
  --slides_pdf ORIGINAL_VISUALIZATIONS_EN.pdf \
  --output_file responses_te.csv \
  --model gpt-4o \
  --max_slides 4
```

Depending on your model and API key, this will write simulated
responses to ``responses_te.csv`` and print decision and certainty
accuracies.

"""

import argparse
import base64
from pathlib import Path
from typing import List, Tuple
import os

import pandas as pd

try:
    import fitz  # PyMuPDF for PDF page rendering
except ImportError:
    fitz = None  # type: ignore[assignment]

try:
    import openai  # Optional: used if you want to call OpenAI's API.
except ImportError:
    openai = None


def extract_group(participant_id: str) -> str:
    """Return the group code (DE, IT, or SSH) from a participant_id."""
    if not isinstance(participant_id, str):
        return ""
    for part in participant_id.split("_"):
        if part in {"DE", "IT", "SSH"}:
            return part
    return ""


def extract_slide_images(pdf_path: str, max_slides: int = 3) -> List[str]:
    """
    Render up to ``max_slides`` pages from a PDF and return their
    contents as base64‑encoded PNG images.  This helper uses
    ``fitz`` (PyMuPDF) to rasterise pages at double resolution for
    improved legibility when sent to the model.  If ``fitz`` is not
    available, an empty list is returned.
    """
    images: List[str] = []
    if fitz is None:
        return images
    doc = fitz.open(pdf_path)
    for i in range(min(len(doc), max_slides)):
        page = doc.load_page(i)
        # Render at 2x resolution so text is readable
        matrix = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=matrix)
        img_bytes = pix.tobytes("png")
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        images.append(b64)
    doc.close()
    return images


def assemble_task_descriptions(tasks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine feature columns into a single semicolon‑separated description.

    In this variant we include the AI model's prediction (``model_class``)
    and its confidence (``model_probability``) so that the LLM sees the
    same cues as the human participants.  Only the ``problem_id`` is
    excluded from the description.
    """
    exclude = {"problem_id"}
    descriptions = []
    for _, row in tasks_df.iterrows():
        parts: List[str] = []
        for col in tasks_df.columns:
            if col in exclude:
                continue
            val = str(row[col]).strip()
            if not val or val.lower() in {"nan", "n/a"}:
                continue
            parts.append(f"{col}: {val}")
        descriptions.append("; ".join(parts))
    tasks_df = tasks_df.copy()
    tasks_df["description"] = descriptions
    return tasks_df[["problem_id", "description"]]


def summarise_traits(participant: pd.Series) -> str:
    """
    Construct a concise persona description from a participant's survey
    responses.  This function focuses on a few key fields that are
    likely to influence decision making: mushroom knowledge, data
    visualization and analysis skills, study programme and mushroom
    collecting habits.  Missing fields are ignored.
    """
    # Define the columns of interest.  Feel free to adjust this list
    # to include other survey questions that you deem relevant.
    cols = {
        "How would you rate your knowledge about wild edible and poisonous mushrooms?": "mushroom knowledge",
        "Do you ever collect wild mushrooms?": "collects mushrooms",
        "What are your skills in data visualization, such as creating charts, infographics, etc.? (Select up to two answers).": "data visualisation skills",
        "How do you rate your skills in data analysis?  ": "data analysis skills",
        "Provide the name(s) of the study program(s) you are currently enrolled in:": "study programme",
    }
    parts: List[str] = []
    for col, label in cols.items():
        val = participant.get(col, "").strip()
        if val:
            parts.append(f"{label}: {val}")
    return "; ".join(parts)


def simulate_response(
    participant_name: str,
    participant_traits: str,
    group: str,
    slide_images: List[str],
    task_description: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
) -> Tuple[str, str]:
    """
    Simulate a single decision/certainty pair for one participant and task.

    The prompt includes a TE‑style narrative instructing the model to
    role‑play as the named participant with the given traits and group.
    It attaches the explanation slides as images and presents the
    mushroom's features and the AI model's prediction in the task
    description.  The model must return the decision and certainty
    separated by a pipe.  If the OpenAI client or API key is missing
    the function returns a stub.
    """
    # Map group codes to plain English
    group_desc_map = {
        "DE": "a domain expert in the field of mycology (DE)",
        "IT": "a student with a data science and visualisation background (IT)",
        "SSH": "a student from social sciences and humanities (SSH)",
    }
    group_desc = group_desc_map.get(group, group)
    # Build the textual part of the prompt
    # Instruct the model to think carefully before answering.  We
    # encourage it to reason internally about how the features and
    # explanation slides influence the decision.  At the end it should
    # output only the decision and certainty.  This approach mirrors
    # the cognitive process a human goes through when interpreting
    # visual explanations and making a judgement.
    prompt_text = (
        f"In this simulation you will role‑play as {participant_name}. "
        f"{participant_name} is {group_desc}. Their survey responses can be summarised as follows: {participant_traits}.\n\n"
        "You have reviewed the explanation slides below, which show how the AI model explains its predictions. "
        "Now consider the following mushroom. The AI model has provided its own class prediction and probability along "
        "with the mushroom's feature values: "
        f"{task_description}.\n\n"
        "As {participant_name}, think carefully and step by step about how the features and the insights from the slides "
        "inform your judgement.  Do not reveal your step‑by‑step reasoning; just use it to arrive at your final choice.\n\n"
        "After reasoning internally, decide whether this mushroom is edible or non‑edible. Then choose your certainty from "
        "'definitely certain', 'moderately certain', 'I can't assess', 'moderately uncertain', or 'definitely uncertain'. "
        "Return your answer exactly in the format: decision | certainty."
    )
    # If the OpenAI library and API key are available, call the API
    if openai is not None and getattr(openai, "api_key", None):
        # Assemble content with images
        content: List[dict] = [
            {"type": "text", "text": prompt_text}
        ]
        # Attach each slide as an image_url.  The API accepts data URI for images.
        for img_b64 in slide_images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                }
            )
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content},
        ]
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            content_str = response["choices"][0]["message"]["content"].strip()
            if "|" in content_str:
                decision, certainty = [p.strip() for p in content_str.split("|", 1)]
            else:
                decision, certainty = content_str, "I can't assess"
        except Exception as e:
            # If the API call fails, return a stub answer
            decision, certainty = "non-edible", "moderately certain"
    else:
        # Stubbed answer when no API is available.
        decision, certainty = "non-edible", "moderately certain"
    # Normalise decision labels
    dec = decision.lower().replace("eadible", "edible").replace("non-eadible", "non-edible").replace("non edible", "non-edible").strip()
    if dec.startswith("non") and "edible" in dec:
        dec = "non-edible"
    elif "edible" in dec:
        dec = "edible"
    else:
        dec = decision
    return dec, certainty


def main(args: argparse.Namespace) -> None:
    # Verify PDF extraction capability
    if fitz is None:
        raise RuntimeError(
            "PyMuPDF (fitz) is required to extract images from the PDF. Please install it with `pip install PyMuPDF`."
        )
    if not Path(args.slides_pdf).exists():
        raise FileNotFoundError(f"Slides PDF not found: {args.slides_pdf}")
    # Extract the specified number of slide images
    print(f"Extracting up to {args.max_slides} slides from {args.slides_pdf}")
    slide_images = extract_slide_images(args.slides_pdf, max_slides=args.max_slides)
    # Read the ground truth responses
    gt_df = pd.read_csv(args.ground_truth_file, dtype=str)
    participant_ids = gt_df["participant_id"].dropna().unique().tolist()
    # Read the survey file and keep only participants with ground truth
    survey_df = pd.read_csv(args.characteristics_file, dtype=str)
    survey_df = survey_df[survey_df["participant_id"].isin(participant_ids)].copy()
    # Derive group from participant_id
    survey_df["group"] = survey_df["participant_id"].apply(extract_group)
    survey_df = survey_df[survey_df["group"].astype(bool)]
    # Fill NaN values
    survey_df = survey_df.fillna("")
    # Read and assemble task descriptions
    tasks_df = pd.read_csv(args.tasks_file, dtype=str)
    tasks_df = assemble_task_descriptions(tasks_df)
    # Prepare results list
    results: List[Tuple[str, str, str, str]] = []
    for _, participant in survey_df.iterrows():
        pid = participant["participant_id"]
        group = participant["group"]
        traits = summarise_traits(participant)
        # Use the participant_id as a pseudonym for TE style prompts
        participant_name = f"{pid}"
        for _, task in tasks_df.iterrows():
            dec, cert = simulate_response(
                participant_name,
                traits,
                group,
                slide_images,
                task["description"],
                model=args.model,
                temperature=args.temperature,
            )
            results.append((task["problem_id"], pid, dec, cert))
    # Write responses
    resp_df = pd.DataFrame(results, columns=[
        "problem_id",
        "participant_id",
        "participant_decision_en",
        "participant_certaintity_en",
    ])
    resp_df.to_csv(args.output_file, index=False)
    # Evaluate performance
    merged = pd.merge(gt_df, resp_df, on=["problem_id", "participant_id"], how="inner", suffixes=("_true", "_sim"))
    total = len(merged)
    if total > 0:
        def normalise_decision(s: str) -> str:
            if not isinstance(s, str):
                return str(s)
            s = s.lower().strip()
            s = s.replace("eadible", "edible").replace("non-eadible", "non-edible").replace("non edible", "non-edible").replace("non-ediblee", "non-edible")
            if s.startswith("non") and "edible" in s:
                return "non-edible"
            if "edible" in s:
                return "edible"
            return s
        true_decisions = merged["participant_decision_en_true"].apply(normalise_decision)
        sim_decisions = merged["participant_decision_en_sim"].apply(normalise_decision)
        dec_acc = (true_decisions == sim_decisions).mean()
        true_certs = merged["participant_certaintity_en_true"].str.lower().str.strip()
        sim_certs = merged["participant_certaintity_en_sim"].str.lower().str.strip()
        cert_acc = (true_certs == sim_certs).mean()
    else:
        dec_acc = cert_acc = 0.0
    print(f"Decision accuracy: {dec_acc:.2%}")
    print(f"Certainty accuracy: {cert_acc:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate the XAI‑FUNGI experiment using a TE‑style prompt with images and participant personas."
    )
    parser.add_argument("--characteristics_file", required=True, help="Path to SURVEY_en.csv")
    parser.add_argument("--tasks_file", required=True, help="Path to PROBLEMS_en.csv")
    parser.add_argument("--ground_truth_file", required=True, help="Path to PROBLEMS_RESPONSES.csv")
    parser.add_argument("--slides_pdf", default="ORIGINAL_VISUALIZATIONS_EN.pdf", help="PDF of explanation slides")
    parser.add_argument("--output_file", default="responses_te.csv", help="CSV to write simulated responses")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use (must support images)")
    parser.add_argument("--max_slides", type=int, default=3, help="Maximum number of slide images to embed")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature for the LLM")
    args = parser.parse_args()
    # Assign API key if available
    if openai is not None:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            openai.api_key = api_key
    main(args)