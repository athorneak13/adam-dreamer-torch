# Debugging log: what broke, and how I caught it

This is a from-scratch PyTorch reproduction of DreamerV3 (arXiv 2301.04104, revision 2) on DeepMind Control. Most bugs never reached a training run: each function was checked against a hand-computed number, an invariant, or a unit test before it was committed, and I used Claude Code as a code reviewer throughout (I wrote the core: RSSM, losses, returns, actor-critic, replay, training loop). So this document is less a list of disasters than a record of **how I knew each piece was correct**, plus the handful of problems that only showed up in training.

Final result: walker-walk (proprio), 3 seeds, all three reached ~930–960 by 500k agent steps (1M env frames).

---

## 1. Where the paper and the official code disagree

The paper leaves several details open or gets them wrong. I resolved each against the official implementation (`danijar/dreamerv3`, commit e3f0224), looking only at configs and the specific detail in question.

| Question | Paper | Official code | What I used |
|---|---|---|---|
| Number of latents K | not stated | 32 at every size | 32 |
| Model size for DMC proprio | smallest size in the table is 12M | `dmc_proprio` uses a `size1m` preset: recurrent size 512, MLP hidden size 64, 4 classes per latent | `size1m`, as official |
| Recurrent size of the 12M model | 1024 | 2048 | n/a (I run `size1m`); the paper's value is a typo |
| Reward head loss | symlog MSE (world-model paragraph) | symexp twohot, 255 bins, zero-init output | twohot |
| GRU input embedding | one linear on the concatenation | separate Linear + RMSNorm + SiLU for h, z, a; gates computed from [h block, embedding] | official |
| Loss reduction | sum over time (Eq. 2) | sum over feature dimensions within each time step, then mean over batch and time | official, so the published loss weights apply |
| DMC replay ratio | 512, action repeat 2 | `dmc_proprio`: train ratio 1024 | paper (512, repeat 2) |
| Action range inside the dynamics | not mentioned | actions divided by max(1, \|a\|) before the RSSM | official (see 5.1, "Imagined actions") |
| Optimizer | LaProp + AGC | AGC per parameter tensor, both bias corrections, LR ramp from 0 over 1000 steps | official |
| LaProp β₂ | 0.99 | 0.999 | official (0.999) |
| Continue weights on the actor and critic losses | not in Eqs. 5–6 | γ^t × continue prediction on the start state × product of predicted continues | product of predicted continues only; γ^t and the start-state term are omitted (small for walker: no terminals, 0.997^14 ≈ 0.96) |
| RSSM state at the start of each replay sequence | not mentioned | stored from acting time and carried into training (`replay_context`) | zeros: each sequence starts fresh, and the posterior catches up within a few steps |

## 2. Traps I resolved in the design, before they became bugs

- **Reward/continue offset.** I store records TD3-style: record t = (x_t, a_t, r_t, c_t), with r_t caused by a_t. But the reward head at step t reads (h_t, z_t), and h_t has only seen actions up to a_{t−1}. So the head at step t+1 is trained against r_t: predictions `[:, 1:]` against targets `[:, :-1]`.
- **Episode boundaries inside a sequence.** A 64-step replay chunk can cross into a new episode. The pair (head at t+1, target at t) is masked out wherever `is_first[:, t+1]`; the RSSM state is zeroed there; and the replay λ-return restarts from the imagination return at the boundary, so episode A's returns never see episode B.
- **Truncation is not termination.** Every task here ends only at a time limit, so the continue flag stays 1 (Gymnasium: `1 − terminated`; DMC: `TimeStep.discount`). Storing c = 0 at time limits would teach the critic that the last steps of every episode are worthless.
- **Ring-buffer seam.** Once replay wraps, the newest and oldest records sit next to each other. Sampling counts offsets from the oldest record, so a sequence never jumps from the newest data to the oldest.
- **Reinforce after a no-grad rollout.** Imagination runs under `torch.no_grad()`; the actor loss recomputes log π(a_t | s_t) with gradients on the stored dreamed states. The actor's weights don't change between the two passes, so the loss stays on-policy.
- **Exact zeros where the design needs them.** The critic and reward head are zero-initialized, so uniform probabilities must give a value of exactly 0. That requires the bin expectation to be summed outward from an exactly-zero middle bin (see `make_bins` below).

## 3. How I checked correctness

Every component has tests with numbers worked out by hand before the code existed (106 tests in total). A few examples:

| Check | Expected | Test |
|---|---|---|
| twohot of y = 100 | 0.71 on 95.25, 0.29 on 111.66 | `test_twohot_worked_example` |
| uniform probabilities → value | exactly 0 | `test_uniform_expectation_is_exactly_zero` |
| straight-through gradient | ∂/∂p = upstream gradient | `test_straight_through_gradient` |
| KL below 1 nat | exactly zero gradient | `test_free_bits_zero_gradient_below_one_nat` |
| λ-return, 2 steps (γ 0.9, λ 0.5) | [4.87, 5.6] | `test_lambda_return_two_step_hand_case` |
| return scale, spread 20, previous S 19 | 19.01 | `test_return_scale_ema_example` |
| actor gradient, μ = 0, σ = 0.5, a = 0.3, Â = 0.16 | dL/dμ = −0.192, dL/dσ = +0.205 | `test_actor_loss_continuous_example` |
| LaProp over 8 steps | matches a plain re-statement of the official chain to 1e-6 | `test_matches_reference_over_many_steps` |
| replay sampling after wraparound | consecutive records, nothing overwritten | `test_sequences_are_consecutive_after_wrap` |
| resume from checkpoint | bit-identical to training straight through | `test_resumed_training_matches_uninterrupted` |

**Shape tests are not enough.** Two of the most serious bugs (below) passed every shape test. What caught them were **behavioral checks**: questions about what the model should depend on.

- Does h_3 change when only e_0 changes? (`test_observe_memory`)
- Do the prior and posterior differ? (`test_posts_differ_from_priors`)
- Does a_0 affect h_1 but not h_0, and is the last action never used? (`test_action_shift`)
- After `is_first` at t = 4 in one row, does that row forget everything before t = 4 while the other row doesn't? (`test_is_first_resets_state`)
- Does the world model overfit one fixed batch? (`test_world_model_overfits_one_batch`)

## 4. What I watch during training

| Metric | Healthy on walker-walk |
|---|---|
| `rec`, `rew` | fall fast in the first ~20k steps (`rew` starts at ln 255 = 5.54 from the zero-initialized head) |
| `dyn`, `rep` | a flat ~8–9 nats on walker; an overfit test confirmed the prior can learn (§5.1) |
| `ret` (imagined λ-return) | rises with the real returns; ret × (1 − γ) × 500 should roughly match the real episode return |
| `entropy` | falls as returns rise; the floor for 6 joints at std 0.1 is −5.3 |
| `scale` (S) | grows with the spread of returns |
| `act_oob` | fraction of imagined actions outside [−1, 1], clipped in the dynamics |
| episode returns | 16 envs finish together; I read the batch mean, min and max, not single episodes |

## 5. Bugs I actually hit

### 5.1 Training-level problems (diagnosed from metrics)

**Replay ratio too low: no learning on Pendulum.**
- *Symptom:* after 27k env steps, episode returns still at random level. The world model was fine (`rec` 3.1 → 0.03), but `ret` kept falling and `entropy` crept up.
- *Diagnosis:* the critic was far from the true values (Pendulum values are in the hundreds, `ret` was at −50), so advantages were noise and only the entropy bonus moved the actor. The replay ratio was 64; the paper uses 512 for control.
- *Fix:* more gradient steps per env step. The task was solved by ~25k steps; three seeds agreed.

**The replay ratio I configured wasn't the one I ran.**
- *Symptom:* none in the results. Found while writing the 16-env walker config.
- *Diagnosis:* the loop trained at most once per iteration, and each iteration steps all E envs. With 4 envs and `train_every: 2`, the real ratio was 256, not 512; with walker's 16 envs it would have been 64, the setting that had just failed on Pendulum.
- *Fix:* an `updates_due` counter that runs E / `train_every` train steps per iteration (commit 34faede). Verified by counting `train_step` calls in short runs: exactly 512 for both configs.
- *Lesson:* check that a config value means what the loop does with it.

**Imagined actions outside the range the world model was trained on.**
- *Symptom:* on walker around step 80k, imagined returns kept rising (~335 per episode implied) while real returns fell to ~110.
- *Diagnosis:* the actor samples a ~ Normal(tanh(·), σ) without squashing, and imagination fed those raw samples into the dynamics. The environment clips actions to [−1, 1], so the world model had only ever seen actions inside that range, and had to guess what 1.4 does. The actor can exploit those guesses. A diagnostic metric showed **30% of imagined actions** fell outside [−1, 1] at initialization.
- *Fix:* clip the action inside the RSSM step, as the official code does (commit 701d669). The actor's log-probability still uses the raw sample, so the Reinforce gradient stays unbiased.
- *Outcome:* the mismatch was real and the fix matches the official code, but it was not the cause of the dip. The three original walker runs (commit 2bb3faf) ran without it, recovered on their own and finished at ~930–960; see the next entry. The H = 15 runs with the fix (horizon sweep, commit 8d4db58) learned at about the same rate as those without it.
- *Lesson:* this was in my notes from the start ("the official code divides the action by max(1, |a|)"), parked as a detail, and only revisited when the metrics forced it.

**A false alarm: two seeds dipped at the same step.**
- *Symptom:* seeds 0 and 1 both fell from ~200 to ~110 between steps 72k and 80k, and seed 0 later sat at ~630 with entropy at the floor.
- *Investigation:* I suspected a broken prior, because `dyn` sat flat at ~8.5 nats. An overfit test on a predictable batch showed the prior does learn (KL 0.5 nats once there is context, ~10 nats at t = 0, where the state is all zeros). The dip was the transition from standing (~200) to walking: the walker falls while it learns to step.
- *Outcome:* both seeds recovered within two batches; all three finished at ~930–960.
- *Lesson:* a shared dip across seeds can be a shared learning phase, not a bug. I keep a stop rule with a step count instead of reacting to one batch.

**GPU no faster than CPU.**
- *Symptom:* ~0.5 s per train step on a cloud GPU, about the same as my laptop CPU.
- *Diagnosis:* profiling showed no device bug. One train step is ~133k small tensor operations (64 RSSM steps, 15 imagination steps, backward through all of it), so kernel-launch overhead dominates and the GPU sits mostly idle.
- *Response:* run three seeds in parallel on one GPU, at essentially no slowdown per run.

**Smaller operational issues.**
- An empty log file during a run: Python buffers stdout when it's redirected to a file. Fixed with `PYTHONUNBUFFERED=1`.
- `5.0e6` in YAML loads as a string (PyYAML wants a signed exponent), so `int(cfg.replay.capacity)` would have crashed. Written as `5000000`.
- W&B keeps one value per key per step, and all 16 envs finish on the same step, so the chart showed one env's return. The full data is in the logs and CSVs.
- Stopping a run with a plain `kill` (SIGTERM) skips Python's `finally` block, where the returns CSV is written. Runs are stopped with SIGINT.

### 5.2 Silent bugs: ran without errors, wrong results

These are the ones that matter most, because nothing crashes.

| Bug | How it hid | How I caught it | Fix |
|---|---|---|---|
| `observe` never updated `prev_state`: every step started from the initial state, so the RSSM had no memory across time | all shape and reset tests passed; a_0 → h_1 still worked | behavioral check: h_3 did not change when e_0 changed | `prev_state = post` each step |
| `priors` stacked from `posts` (copy-paste) | shapes identical | posterior and prior probs were equal; the KL would always have been 0 and the prior never trained | stack from `priors` |
| `make_bins` via float32 `linspace`: middle bin 7.45e-8, halves not exactly symmetric | values looked right to 4 digits | `bins[127] == 0` and an exact symmetry check both failed | build the non-negative half and mirror it |
| Bin expectation as a plain `(probs * bins).sum(-1)`, even on symmetric bins | mathematically correct | uniform probs gave −0.19 instead of 0: float32 rounding when ±4.85e8 terms meet 0.17 in one sum | sum the two halves separately, outward from the middle |
| RMSNorm gain divided instead of multiplied, and stored as a plain tensor (not trained, not moved to GPU) | g = 1 at init, so the output was correct | `named_parameters()` was empty; setting g = [2, 1] gave [0.42, 1.13] instead of [1.70, 1.13] | `nn.Parameter`, and multiply after normalizing |
| Return-scale EMA used (1 − S) instead of (1 − decay) | ran fine | hand example: S = 19 with spread 20 gave −341 instead of 19.01 | `(1 - decay) * spread` |
| Unimix parenthesized as ((1 − u)p + u)/C | the categorical sampler renormalizes, hiding it | probabilities summed to 0.26 | (1 − u)p + u/C |
| twohot for targets beyond the bin range: weights of 135.6 and −134.6 | label still summed to 1 | extreme-value test | clamp y to the bin range |
| Continue head returned a probability instead of a logit | worked for moderate values | BCE needs logits to stay finite for confident predictions (−200 → loss 200, not ∞) | return the logit; `predict()` gives the probability |
| LaProp skipped each parameter's first update (initialization in an `if`, the update in an `else`) | step 1 of the warmup has lr 0 anyway | comparison against a reference implementation over 8 steps | always run the update after initializing |

### 5.3 Errors that crashed, by pattern

Most crashes fell into five patterns, which is useful to know when reading unfamiliar PyTorch code:

- **Result computed, then discarded.** `x.expand(...)`, `x.reshape(...)`, `torch.where(...)`, `grad * factor` and a `predict` without `return`. PyTorch methods return new tensors; only methods ending in `_` modify in place.
- **Operator precedence.** `~mask.float()` applies `.float()` before `~`; `y - b.unsqueeze(-1)` unsqueezes only `b`; `w * a + s * b` applies the weight to the first term only.
- **Off-by-one ranges.** An MLP with `layers + 1` blocks; `range(H, 0, 1)` is empty; 127 instead of 128 points for half the bins, giving 253 bins.
- **PyTorch API confusion.** Calling module classes on tensors (`nn.Softmax(x)`, `nn.SiLU(x)`, `RMSNorm(x)`); `torch.max` and `torch.maximum` with a Python float; `F.one_hot(values, indices)` with the arguments swapped; float tensors used as indices.
- **Python gotchas.** `{a, b, c}` is a set, not a dict; unpacking a dict gives its keys; a stray comma turns an expression into a tuple.