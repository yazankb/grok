# Equal-compute multi-model grokking experiment.
#
# Design:
#   Each specialist runs for final_steps / n_models = 20000 / 4 = 5000 steps.
#   Total specialist gradient steps = 4 x 5000 = 20000 = one baseline run.
#   This is a fair compute comparison: you could have trained the baseline for
#   20000 steps OR trained 4 specialists for 5000 steps each.
#   The merged model then gets 20000 additional steps — does it grok faster
#   than the baseline which also starts from random init at step 0?
#
# No early stopping: specialists run their full budget and can potentially
#   start grokking their own shard, not just memorise it.
#
# Expected timing (check_val_every_n_epoch=25, ~40 steps/sec on GPU):
#   4 specialists x 5000 steps = 20000 steps  ~=  8 min
#   merged  model x 20000 steps               ~=  8 min
#   baseline model x 20000 steps              ~=  8 min
#   Total ~= 24 min  (comfortably under 30 min)
#
# 50% training data chosen because grokking is documented at ~5000-20000
# steps for this fraction in the original paper.

. "$PSScriptRoot\..\init_conda.ps1"

python "$PSScriptRoot\train_multi.py" `
    --math_operator        "+" `
    --train_data_pct       50 `
    --n_models             4 `
    --equal_compute `
    --final_steps          20000 `
    --overfit_threshold    99.0 `
    --experiment_name      "multi_fixed" `
    --gpu                  0 `
    --max_lr               1e-3 `
    --n_layers             2 `
    --n_heads              4 `
    --d_model              128 `
    --weight_decay         1 `
    --weight_decay_kind    to_zero `
    --warmup_steps         10
