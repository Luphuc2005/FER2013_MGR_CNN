# RAF-DB V5 + LGSA + R-Drop

Full candidate first; ablate components after the full run. No accuracy guarantee.

## Run on the server

From /home/ptbao/projects/FER2013_MGR_CNN:

```bash
sbatch run_rafdb_v5_lgsa_rdrop_v100.slurm.sh
```

The launcher runs the config/component checks, then a real batch-16 head/full SAM
smoke in a separate process. Only when those pass does training start. It refuses
a nonempty output directory and MGR_* overrides. The smoke model is discarded;
training initializes again from MS1M, without a FER checkpoint or resume.

Files:
- config_rafdb_siglip2_semantic_stable_v5_lgsa_rdrop.yaml
- run_rafdb_v5_lgsa_rdrop_v100.slurm.sh
- scripts/check_rafdb_v5_lgsa_rdrop.py
- losses/rdrop.py
- utils/rdrop_metrics.py
- train.py: optional training objective and diagnostics

Output: outputs/papers/rafdb_siglip2_semantic_stable_v5_lgsa_rdrop

## Exact objective

For the SAME already-augmented input batch, make two separate stochastic forwards
at the SAME weights. Each forward computes the full existing objective:

L_j = CE(fused_logits_j) + lambda_sem(epoch)*CE(semantic_logits_j)
      + 0.10*L_hard_j + 0.02*mean(CE(upper_j), CE(lower_j), CE(AU_j)).

p_j = softmax(fused_logits_j), each [B,7].
KL_sym = mean_batch(0.5*(KL(p_1||p_2) + KL(p_2||p_1))).
Sum KL over classes; temperature=1; float32 log-softmax and reduction.
Both predictions receive gradients.

L_full = (L_1 + L_2)/2 + 0.5*KL_sym.

lambda_rdrop=0.5 is an initial experimental setting, not a validated optimum.
No new schedule; original lambda_sem schedule stays unchanged.
The existing stochastic DropPath also varies between training forwards.

SAM computes this complete objective at theta, perturbs weights, computes it
again at theta+epsilon, restores weights, and applies the second gradient.
There are FOUR forwards per SAM step, not a KL between the two SAM weight states.
Batch size remains 16; peak memory and throughput must be checked on the V100.

## Preserved behavior

Backbone, DPA, classifier, semantic projectors, granularity gate, adaptive fusion,
SAM rho=0.02, AdamW, LRs, augmentation, label smoothing, freeze schedule and TTA
are copied from V5+LGSA. No S2/S3/S4 fusion, new teacher, or prototype attention.
No trainable parameters are added by R-Drop.

Inference uses exactly the existing fused V5 logits and ordinary evaluation;
there is no second stochastic forward or R-Drop loss during validation/test.
Original and HFlip TTA remain reported separately.

Setting training.lambda_rdrop=0.0 skips the additional forward entirely and uses
the original V5+LGSA objective in both SAM passes. For ablations use a fresh
config/output path; do not reuse this run directory. LGSA itself was fixed in
this working tree and remains enabled at 0.02 in both full and zero-R-Drop cases.

Training accuracy and granularity statistics use the first unperturbed prediction,
not the average of two predictions. Existing CE/semantic/hard loss logs are means
of the two unperturbed objectives. R-Drop diagnostics are sample-weighted and
counted once per batch:
- train_local_semantic_loss, train_weighted_local_semantic_loss
- train_rdrop_loss, train_weighted_rdrop_loss
- lambda_rdrop, rdrop_train_samples

These are added to training_history.csv and printed as [RDROP].
R-Drop metrics live outside the model/checkpoint and do not change model weights.

## Verification

```bash
# Config parity and Python syntax only; TensorFlow not required.
python scripts/check_rafdb_v5_lgsa_rdrop.py

# Synthetic component regression: KL, gradients, zero equivalence, actual SAM
# trainer with recorded input/weight states, forward count and metrics count.
./fer2013_env/bin/python scripts/check_rafdb_v5_lgsa_rdrop.py --smoke

# Real model, real first training batch, configured mixed precision and batch 16.
./fer2013_env/bin/python scripts/check_rafdb_v5_lgsa_rdrop.py --model-smoke
```

The model smoke requires existing RAF CSVs normalized to [0..6], prints class
names and split counts, and never uses validation/test samples in gradients.
It disables gradient sanitization on the throwaway model to expose nonfinite
gradients. Production settings are unchanged.

Local TensorFlow runtime, V100 smoke, full training and accuracy are unverified
until their respective checks are executed. Config/parse success is not runtime
success.
