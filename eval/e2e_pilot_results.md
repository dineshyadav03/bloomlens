# End-to-end pilot -- NOT a benchmark (eval/PROTOCOL.md section 9)

119 of 119 pilot images attempted so far. 119 succeeded (100.0% schema-valid).

**This is a small, capped sample. Confidence intervals would be wide and are not computed; nothing here is a headline claim.**

| family | n | n ok | abstained rate | agrees with retrieval top-1 |
|---|---:|---:|---|---|
| id | 89 | 89 | 10.1% | 92.1% |
| near_ood | 15 | 15 | 93.3% | 66.7% |
| far_ood | 15 | 15 | 100.0% | 86.7% |

Latency (successful calls, n=119): p50 11466 ms, p95 30871 ms
Mean attempts per call: 1.00
Tokens (n=119 reported): mean 6973 in / 226 out
Estimated cost: $0.2477 over 119 priced calls -- estimated list-price equivalent; actual billed cost unknown

Failures by category: none
