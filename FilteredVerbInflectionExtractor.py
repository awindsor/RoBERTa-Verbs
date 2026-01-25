from lemminflect import getAllInflections
import csv
import re


def get_inflections(lemma: str) -> set:
    """Return all verb inflections for a lemma using lemminflect."""
    infl = getAllInflections(lemma, upos="VERB")
    forms = set()
    for _, wordforms in infl.items():
        forms.update(wordforms)
    return forms


lemma_inflections = {}

with open(
    "/Users/awindsor/Library/CloudStorage/OneDrive-TheUniversityofMemphis/Documents/Research/NLP/NCTE Transcripts - Release/NCTE_verbs_tagged.csv",
    "r",
    encoding="utf-8-sig",
) as f:
    reader = csv.DictReader(f)
    lemmas = {row["lemma"] for row in reader if int(row["freq"]) >= 10}

lemmas.remove("be")

with (
    open(
        "/Users/awindsor/Library/CloudStorage/OneDrive-TheUniversityofMemphis/Documents/Research/NLP/NCTE Transcripts - Release/TAASSC Results/results_clause_database.txt",
        "r",
        encoding="utf-8-sig",
    ) as f,
    open(
        "/Users/awindsor/Library/CloudStorage/OneDrive-TheUniversityofMemphis/Documents/Research/NLP/NCTE Transcripts - Release/filtered_verb_inflections.csv",
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as out_f,
):
    reader = csv.DictReader(f, delimiter="\t")
    print(reader.fieldnames)
    fieldnames = list(reader.fieldnames) if reader.fieldnames is not None else []
    headers = fieldnames + ["inflected_form"]
    writer = csv.DictWriter(out_f, fieldnames=headers)
    for row in reader:
        lemma = row["lemma"]
        if lemma not in lemmas:
            continue
        if lemma not in lemma_inflections:
            inflections = get_inflections(lemma)
            lemma_inflections[lemma] = inflections
        else:
            inflections = lemma_inflections[lemma]
        sentence = row["sentence_text"].lower()
        inflected_form = [
            form
            for form in inflections
            if re.search(r"\b" + re.escape(form.lower()) + r"\b", sentence)
        ]

        if len(inflected_form) == 1:
            row["inflected_form"] = inflected_form[0]
            writer.writerow(row)
        else:
            print(
                f"Multiple or no inflected forms found for lemma {lemma} in sentence: {row['sentence_text']}. Found forms: {inflected_form} out of inflections {inflections}"
            )
