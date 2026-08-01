# Evidence-Gated Semantic Meta-Control (EGSMC)
## Σύντομος οδηγός αναφοράς για τα μοντέλα και τα Top-K αρχεία

> **Αυτό το αρχείο είναι προαιρετικό.**
>
> Δεν χρειάζεται να φορτώσεις ή να εκτελέσεις τα μοντέλα για να ξεκινήσεις την πτυχιακή.
> Τα Top-K predictions έχουν ήδη παραχθεί.
>
> Διάβασέ το μόνο για να καταλάβεις:
>
> - τι περιέχει ένα εκπαιδευμένο μοντέλο,
> - τι είναι τα mappings,
> - τι περιέχει ένα Top-K prediction file,
> - τι σημαίνουν `score`, `rank` και `is_gold`.

---

# 1. Τι υπάρχει στο πακέτο

```text
EGSMC_ALL_MODELS_TOPK_PACKAGE/
├─ 01_DATA/
├─ 02_TRAINED_MODELS/
├─ 03_TOPK_RESULTS/
├─ 04_SUMMARY_RESULTS/
└─ 05_HELPER_SCRIPTS/
```

Υπάρχουν:

```text
2 datasets × 5 models = 10 trained models
```

Τα μοντέλα είναι:

```text
TransE
DistMult
ComplEx
RotatE
PairRE
```

Για κάθε μοντέλο υπάρχουν Top-K predictions για:

```text
validation + test
head prediction + tail prediction
```

---

# 2. Τι είναι τα βασικά model files

Κάθε model folder περιέχει κυρίως:

```text
trained_model.pkl
training_triples/
model_results.json
```

## `trained_model.pkl`

Το εκπαιδευμένο PyTorch/PyKEEN μοντέλο.

## `training_triples/`

Τα ακριβή mappings που χρησιμοποιήθηκαν στην εκπαίδευση:

```text
entity label -> entity ID
relation label -> relation ID
```

## `model_results.json`

Τα βασικά αποτελέσματα του μοντέλου:

```text
MRR
Hits@1
Hits@3
Hits@10
```

---

# 3. Πώς φορτώνεται ένα μοντέλο, μόνο αν χρειαστεί

Το package έχει ήδη helper script:

```text
05_HELPER_SCRIPTS/load_trained_model.py
```

Παράδειγμα:

```powershell
python 05_HELPER_SCRIPTS\load_trained_model.py `
  --model-dir "02_TRAINED_MODELS\DRUGBANK\ROTATE" `
  --device cpu
```

Αναμενόμενη έξοδος:

```text
Model class: RotatE
Model entities: 28816
Factory entities: 28816
Model relations: 14
Factory relations: 14
Model and mappings loaded successfully.
```

## Γιατί εμφανίζονται περισσότερες relations;

Το DrugBank έχει 7 πραγματικές relations, αλλά χρησιμοποιήθηκαν και inverse relations:

```text
7 original + 7 inverse = 14
```

Το EurostatKG έχει:

```text
13 original + 13 inverse = 26
```

Αυτό είναι αναμενόμενο.

---

# 4. Τι είναι τα Top-K prediction files

Παράδειγμα:

```text
03_TOPK_RESULTS/DRUGBANK/ROTATE/test_tail_top50.jsonl.gz
```

Το αρχείο περιέχει ήδη τις 50 καλύτερες προβλέψεις του μοντέλου για κάθε query.

Δεν χρειάζεται να φορτώσεις το μοντέλο για να χρησιμοποιήσεις αυτό το αρχείο.

---

# 5. Δομή ενός Top-K record

Παράδειγμα tail prediction:

```json
{
  "query_type": "tail_prediction",
  "query": {
    "head": "drug:DB00001",
    "relation": "has_target",
    "gold_tail": "target:BE0000048"
  },
  "gold_rank_in_export": 3,
  "candidates": [
    {
      "entity": "target:BE0000999",
      "score": 12.7,
      "rank": 1,
      "is_gold": false
    },
    {
      "entity": "target:BE0000048",
      "score": 11.9,
      "rank": 3,
      "is_gold": true
    }
  ]
}
```

## Τι σημαίνουν τα πεδία

### `query_type`

```text
tail_prediction -> (head, relation, ?)
head_prediction -> (?, relation, tail)
```

### `score`

Το ranking score του μοντέλου.

Δεν είναι probability.

### `rank`

Η αρχική θέση του candidate.

### `gold_rank_in_export`

Η θέση της πραγματικής απάντησης στο Top-50.

Αν είναι `null`, η πραγματική απάντηση δεν βρίσκεται στο Top-50.

### `is_gold`

Χρησιμοποιείται μόνο για evaluation.

**Δεν επιτρέπεται να χρησιμοποιηθεί από τον semantic checker ή το reranking.**

---

# 6. Πώς βλέπεις τα πρώτα records

Το package έχει helper script:

```text
05_HELPER_SCRIPTS/read_candidates.py
```

Παράδειγμα:

```powershell
python 05_HELPER_SCRIPTS\read_candidates.py `
  --file "03_TOPK_RESULTS\DRUGBANK\ROTATE\test_tail_top50.jsonl.gz" `
  --limit 2
```

Για ένα record πρέπει να μπορείς να απαντήσεις:

```text
Ποιο είναι το query;
Ποιο είναι το gold entity;
Ποιος είναι ο rank-1 candidate;
Σε ποια θέση βρίσκεται το gold;
Τι σημαίνει το score;
```

---

# 7. Τι χρειάζεται να θυμάσαι

```text
Το μοντέλο παράγει scores.
Το Top-K file αποθηκεύει τις καλύτερες προβλέψεις.
Το semantic layer ελέγχει αν οι προβλέψεις είναι αποδεκτές.
```

Για την άμεση εργασία σου θα χρησιμοποιήσεις κυρίως:

```text
01_DATA/
03_TOPK_RESULTS/
```

Η φόρτωση των μοντέλων είναι προαιρετική και δεν αποτελεί προϋπόθεση για να ξεκινήσεις.
