# Threshold selection (dev only) -- eval/PROTOCOL.md section 4

Frozen: `max_cosine`, tau = 0.607094.
**Adopted: True** (near-OOD-dev abstention gain +71.0% vs the 10% bar; ID-dev abstention 4.9% vs the 8% ceiling).


| candidate | dev AUROC (ID vs near-OOD) |
|---|---:|
| max_cosine | 0.9356 <- frozen |
| max_softmax@0.1 | 0.8575 |
| max_softmax@0.05 | 0.8497 |
| max_softmax@0.02 | 0.8337 |
| max_softmax@0.01 | 0.8172 |
| margin | 0.8032 |
