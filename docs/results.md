# Data and results in detail

## Training data

**PAN 2022** (written discourse only): 12,264 problems, 1,046 texts, 56 authors.
Texts per discourse type: essays 2,986 · emails 10,116 · text messages 9,446 · business memos 1,980.

**PAN 2023** (written and spoken): 8,836 problems, 886 texts, 56 authors.
Texts per discourse type: essays 2,594 · emails 7,054 · interviews 6,090 · speech transcriptions 1,934.

In both editions the authors are native English speakers aged 18–22, writing on unrestricted topics with
varying formality.

**Discourse-type combinations of the problems**

| Pair | PAN 2022 | PAN 2023 | Combined |
|---|---:|---:|---:|
| email – text message | 7,484 | – | 7,484 |
| essay – email | 1,618 | 1,454 | 3,072 |
| business memo – text message | 780 | – | 780 |
| essay – text message | 1,182 | – | 1,182 |
| email – business memo | 1,014 | – | 1,014 |
| essay – business memo | 186 | – | 186 |
| speech transcription – email | – | 1,036 | 1,036 |
| email – interview | – | 4,564 | 4,564 |
| essay – interview | – | 884 | 884 |
| speech transcription – interview | – | 642 | 642 |
| essay – speech transcription | – | 256 | 256 |
| **total** | **12,264** | **8,836** | **21,100** |

Combined, the two editions give 21,100 problems, 1,932 texts and 112 authors.

## Data splitting

The combined data is split into training and validation sets with an author-disjoint method similar to
Martinez-Galicia et al. (2022): no text in one set has the same author as any text in the other.
With 21,100 problems but only 112 authors, this split alone is unbalanced, with more positive problems
than negative ones. To fix the balance:
- with A and B the texts of each set grouped by author, positives are `P = A × A` and negatives are
  `N = A × B`
- pairs whose two texts share a discourse type are removed
- positives and negatives are sampled at random from `P` and `N` in equal numbers

The result is 18,968 / 2,132 balanced problems with 101 / 11 authors. `scripts/prepare_data.py`
reproduces this split and checks that the two sets share no authors or texts.

## Model selection (validation)

| Model | Max tokens | AUC | c@1 | F0.5u | F1 | Brier | Overall |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|
| all-MiniLM-L12-v2 | 256 | **0.684** | **0.632** | **0.615** | **0.711** | **0.761** | **0.681** |
| nli-distilroberta-base-v2 | 512 | 0.572 | 0.509 | 0.553 | 0.667 | 0.692 | 0.598 |
| Stylometric model | – | 0.505 | 0.512 | 0.433 | 0.343 | 0.521 | 0.463 |

all-MiniLM-L12-v2 was selected for all submissions.

**Hyper-parameters, chosen on validation**

| Parameter | Value |
|---|---|
| base model | all-MiniLM-L12-v2 (max sequence length 256) |
| chunk length | 256 tokens |
| batch size | 32 |
| distance | cosine |
| loss | ContrastiveLoss |
| epochs | 189 |
| scheduler | WarmupLinear, 0 warm-up steps |
| learning rate / eps | 2e-5 / 1e-6 |
| non-answer band | p1 = 0.45, p2 = 0.54 |

## Official test results (TIRA)

| Run | AUC | c@1 | F0.5u | F1 | Brier | Overall |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| resolving-globe | 0.616 | 0.572 | 0.562 | 0.617 | 0.746 | **0.623** |
| reduced-graph | 0.616 | 0.572 | 0.562 | 0.617 | 0.746 | **0.623** |
| golden-ottoman | 0.598 | 0.546 | 0.550 | 0.622 | 0.744 | 0.612 |

All three runs use the same trained model:
- **resolving-globe** uses the built-in p1/p2.
- **reduced-graph** gives the same answers but takes p1 and p2 as command-line arguments (`--p1/--p2`
  in `av-predict`).
- **golden-ottoman** switches to sentence-aware chunking (`--soft`). It slightly raises F1 but lowers AUC
  and c@1, likely because its chunks no longer match the fixed windows the model was trained on.

With an overall score of **0.623**, the system ranked **first** in the PAN 2023 authorship verification
task.
