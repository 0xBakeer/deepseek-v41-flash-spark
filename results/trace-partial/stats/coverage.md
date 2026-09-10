# Expert coverage -- 10760 tokens, 50 sequences, layers 0-3

## Per layer

| layer | experts used | top-10% covers | top-25% covers | top-50% covers | entropy (bits, max 8.58) | unique experts / 6-token block (max 36) |
|---|---|---|---|---|---|---|
| 0 | 381 | 0.354 | 0.594 | 0.842 | 7.85 | 28.0 |
| 1 | 380 | 0.386 | 0.614 | 0.846 | 7.85 | 25.1 |
| 2 | 370 | 0.433 | 0.694 | 0.902 | 7.60 | 23.6 |
| 3 | 374 | 0.553 | 0.761 | 0.930 | 7.10 | 22.1 |

## Global coverage vs resident-expert budget (layers traced: 4 of 40, 1536 (layer,expert) keys)

Static resident set = the most frequent (layer, expert) pairs overall. 'covers' = share of routed slots that hit the resident set. Memory = FP4 experts only (18.8 MB each).

| budget (experts) | share of all keys | resident GB (FP4) | static coverage | LRU hit/token | LRU hit/6-token block |
|---|---|---|---|---|---|
| 200 | 0.13 | 3.8 | 0.494 | 0.606 | 0.437 |
| 400 | 0.26 | 7.5 | 0.680 | 0.778 | 0.676 |
| 460 | 0.30 | 8.6 | 0.724 | 0.811 | 0.724 |
| 600 | 0.39 | 11.3 | 0.809 | 0.869 | 0.808 |
| 800 | 0.52 | 15.0 | 0.896 | 0.923 | 0.888 |
| 1000 | 0.65 | 18.8 | 0.952 | 0.960 | 0.942 |
| 1200 | 0.78 | 22.6 | 0.985 | 0.986 | 0.979 |

## Coding vs general: overlap of the per-layer top-25% sets

| layer | Jaccard(top25 coding, top25 general) | coding slots covered by general's top25 |
|---|---|---|
| 0 | 0.25 | 0.390 |
| 1 | 0.18 | 0.329 |
| 2 | 0.27 | 0.385 |
| 3 | 0.31 | 0.619 |
