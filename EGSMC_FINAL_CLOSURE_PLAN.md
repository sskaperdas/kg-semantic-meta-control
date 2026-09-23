# EGSMC — Final Closure Plan

> **Στόχος:** να κλείσει άμεσα μία πλήρης, συνεπής υλοποίηση end-to-end.
>
> Από εδώ και πέρα **δεν προσθέτουμε νέα features**. Κλείνουμε μόνο:
>
> ```text
> Evidence V02
> → Validation analysis
> → Evidence gate
> → Final evaluation
> ```

---

# 1. Πού είμαστε ΤΩΡΑ

Στο τρέχον repository υπάρχουν ήδη:

```text
PASS  Top-K reader
PASS  semantic resources
PASS  VALID / INVALID / UNKNOWN checker
PASS  candidate annotation
PASS  simple VALID-first reranking
PASS  baseline / controlled evaluation

PASS  ontology evidence
PASS  1-hop / 2-hop graph evidence

DONE IN CODE:
      3-hop evidence
      3-hop witness count
      relation-role evidence
      relation-role count

DONE IN CODE:
      06_extract_evidence.py
      07_analyze_evidence_validation.py
```

Άρα **δεν ξανασχεδιάζουμε το evidence**.

Το Evidence V02 είναι ήδη γραμμένο στον κώδικα.

Αυτό που **δεν επιβεβαιώνεται από το ZIP** είναι ότι έχει ήδη ολοκληρωθεί full V02 run και έχουν αποθηκευτεί τα V02 outputs.

---

# 2. Μοναδικό blocker πριν τα runs

Το full:

```powershell
pytest
```

αποτυγχάνει επειδή το:

```text
tests/test_checker.py
```

χρησιμοποιεί παλιό API (`load_resources`, `check_triple`).

Τα νέα graph-evidence tests περνούν.

## Άμεση διόρθωση

Αντικατέστησε το `tests/test_checker.py` με:

```python
from pathlib import Path

from kgsmc.semantic_control.resources import SemanticResources
from kgsmc.semantic_control.checker import SemanticChecker


def test_checker_first_100():
    root = Path(__file__).resolve().parents[1]

    dataset_dir = root / "data" / "processed" / "drugbank_core"
    test_file = dataset_dir / "test.tsv"

    resources = SemanticResources(dataset_dir)
    checker = SemanticChecker(resources)

    checked = 0

    with test_file.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            head, relation, tail = line.split("\t")

            result = checker.check_triple(
                head,
                relation,
                tail,
            )

            assert result.status == "VALID"

            checked += 1

            if checked == 100:
                break

    assert checked == 100
```

Μετά:

```powershell
pytest -q
```

**Αν PASS → συνεχίζεις.**

Δεν κάνουμε άλλο test cleanup τώρα.

---

# 3. Evidence V02 — smoke run

Τρέξε:

```powershell
python scripts\06_extract_evidence.py `
  --datasets drugbank `
  --model ROTATE `
  --limit 100
```

Θέλουμε να δημιουργηθούν:

```text
runs/evidence/
  drugbank_rotate_valid_tail_top50_evidence_v02.jsonl.gz

runs/evidence/
  drugbank_rotate_valid_tail_evidence_summary_v02.json
```

Έλεγξε μόνο ότι:

```text
queries = 100
candidates ≈ 5000
```

και ότι υπάρχουν fields:

```text
has_three_hop_witness
three_hop_count
example_three_hop_path

relation_role_supported
relation_role_count
```

Αν αυτά υπάρχουν και οι τιμές φαίνονται λογικές:

**δεν κάνουμε άλλο manual debugging.**

---

# 4. Evidence V02 — FULL DrugBank validation

Αμέσως μετά:

```powershell
python scripts\06_extract_evidence.py `
  --datasets drugbank `
  --model ROTATE
```

Περιμένουμε:

```text
queries = 14,943
candidates = 747,150
```

Κρατάμε τα V02 outputs.

---

# 5. Validation evidence analysis

Τρέξε:

```powershell
python scripts\07_analyze_evidence_validation.py `
  --input_file "runs/evidence/drugbank_rotate_valid_tail_top50_evidence_v02.jsonl.gz" `
  --output_file "runs/analysis/drugbank_rotate_validation_analysis.json"
```

Το αρχείο πρέπει να μας δώσει:

```text
gold candidates
non-gold candidates

3-hop support rate:
gold
non-gold

3-hop witness count:
gold median / p90
non-gold median / p90

relation-role support:
gold
non-gold

relation-role count:
gold median
non-gold median
```

---

# 6. STOP εδώ για 5 λεπτά και πάρε μόνο μία απόφαση

Από το validation analysis θέλουμε να αποφασίσουμε:

```text
Ποιο structural signal έχει πραγματική διακριτική αξία;
```

## Κανόνας

Αν ένα signal είναι σχεδόν ίδιο για gold και non-gold:

```text
ΔΕΝ το χρησιμοποιούμε στο gate.
```

Αν υπάρχει καθαρή διαφορά:

```text
το κρατάμε.
```

Δεν προσθέτουμε νέο signal.

Δεν κάνουμε ML.

Δεν κάνουμε complicated optimization.

---

# 7. Τελικό Evidence Strength — κρατάμε το απλό

Δεν χρειάζεται continuous evidence score με πολλά weights.

Χρησιμοποιούμε κατηγορίες:

```text
CONFLICT
INSUFFICIENT
BASIC_SUPPORT
STRONG_SUPPORT
```

## Προτεινόμενη λογική

### `CONFLICT`

```text
semantic_status == INVALID
```

### `INSUFFICIENT`

```text
semantic_status == UNKNOWN
```

### `BASIC_SUPPORT`

```text
semantic_status == VALID
```

αλλά χωρίς το structural signal που αποδείχθηκε χρήσιμο στο validation.

### `STRONG_SUPPORT`

```text
semantic_status == VALID
+
το/τα structural signals που όντως ξεχώρισαν
gold από non-gold στο validation
```

**Μόνο αυτά.**

---

# 8. Τελικό Evidence Gate

Το gate πρέπει να είναι query-level και πολύ απλό.

Για κάθε query:

## Περίπτωση A — baseline rank-1 είναι `VALID`

```text
PRESERVE
```

Δεν πειράζουμε σωστή semantic κορυφή χωρίς λόγο.

---

## Περίπτωση B — baseline rank-1 είναι `INVALID` ή `UNKNOWN`

Αν υπάρχει candidate με:

```text
STRONG_SUPPORT
```

τότε:

```text
RERANK
```

και επιλέγουμε τον υψηλότερα αρχικά ranked `STRONG_SUPPORT` candidate.

---

## Περίπτωση C — baseline rank-1 είναι `INVALID` ή `UNKNOWN`
και δεν υπάρχει `STRONG_SUPPORT`

```text
ABSTAIN
```

Δεν κάνουμε αυθαίρετη παρέμβαση.

---

# 9. Γιατί αυτό αρκεί για να κλείσει το EGSMC

Η τελική λογική γίνεται:

```text
Frozen Top-K
    ↓
Semantic status
    +
Ontology evidence
    +
Validated structural evidence
    ↓
Evidence strength
    ↓
Gate
 ┌───────────────┬────────────────┐
 │               │                │
PRESERVE       RERANK          ABSTAIN
```

Αυτό είναι πλήρες:

```text
Evidence
+
Gate
+
Semantic Meta-Control
```

και είναι σαφώς διαφορετικό από το υπάρχον απλό:

```text
VALID first
UNKNOWN second
INVALID last
```

baseline.

---

# 10. Υλοποίηση gate — μόνο 2 νέα αρχεία

Μην ανοίξεις άλλο architecture.

Χρειαζόμαστε μόνο:

```text
src/kgsmc/semantic_control/gate.py
scripts/08_run_egsmc.py
```

Το `gate.py` πρέπει να έχει:

```python
class EvidenceGate:
    def classify_candidate(...):
        ...

    def decide_query(...):
        ...
```

και να επιστρέφει:

```text
PRESERVE
RERANK
ABSTAIN
```

Το `08_run_egsmc.py`:

```text
διαβάζει V02 annotated/evidence Top-K
→ εφαρμόζει gate
→ γράφει controlled ranking/decision
→ υπολογίζει metrics
```

Δεν χρειάζεται νέο framework.

---

# 11. Validation evaluation

Στο DrugBank validation συγκρίνουμε μόνο 3 συστήματα:

```text
1. BASE
2. SIMPLE SEMANTIC RERANKER
3. EGSMC
```

Metrics:

```text
MRR@50
Hits@1
Hits@10

Valid@1
Candidate Valid Rate@10

Coverage
Abstention Rate

Preserve Rate
Rerank Rate
```

Αυτό είναι αρκετό.

---

# 12. Freeze

Μετά το validation:

```text
FREEZE:
- evidence definition
- structural signal
- threshold αν υπάρχει
- gate policy
```

Δεν αλλάζει τίποτα από το test.

---

# 13. Τελικό test run

Με το παγωμένο EGSMC:

```text
DRUGBANK / ROTATE / test tail
```

Κάνουμε:

```text
Evidence V02 extraction
→ EGSMC gate
→ final evaluation
```

και αναφέρουμε τα ίδια metrics:

```text
BASE
SIMPLE SEMANTIC RERANKER
EGSMC
```

---

# 14. Αν μείνει χρόνος: μόνο ένα δεύτερο dataset

ΜΟΝΟ αν έχει μείνει πραγματικός χρόνος:

```text
EurostatKG / PairRE / validation tail
→ freeze/use same method
→ test tail
```

Αυτό είναι το καλύτερο πρόσθετο experiment.

## Δεν κάνουμε τώρα

```text
head prediction
όλα τα 10 models
40 Top-K files
νέα evidence signals
learned gate
LightGBM
νευρωνικό controller
hyperparameter sweep
```

---

# 15. Hard stop

Η εργασία κλείνει όταν έχουμε:

```text
[1] pytest PASS

[2] Full DrugBank validation Evidence V02

[3] Validation gold-vs-non-gold evidence analysis

[4] Ένα απλό frozen evidence gate

[5] DrugBank validation:
    BASE vs SIMPLE vs EGSMC

[6] DrugBank test:
    BASE vs SIMPLE vs EGSMC

[7] Προαιρετικά:
    EurostatKG / PairRE validation + test
```

Τίποτε άλλο δεν είναι απαραίτητο για μία ολοκληρωμένη end-to-end εικόνα.

---

# 16. Τι κάνουμε ΑΜΕΣΩΣ τώρα

Με αυστηρή σειρά:

```text
1. Fix tests/test_checker.py
2. pytest -q
3. Evidence V02 smoke 100
4. Evidence V02 full DrugBank validation
5. 07_analyze_evidence_validation.py
6. Δες μόνο ποιο structural signal κρατάμε
7. Implement minimal EvidenceGate
8. Evaluate BASE vs SIMPLE vs EGSMC on validation
9. FREEZE
10. Run final DrugBank test
11. STOP
```
