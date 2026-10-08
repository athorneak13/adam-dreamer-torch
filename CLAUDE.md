# CLAUDE.md: adam-dreamer-torch

A from-scratch PyTorch reproduction of DreamerV3 (Hafner et al., arXiv 2301.04104, **revision 2**, Apr 2024) on DeepMind Control, continuous actions only. This is a portfolio project. Adam must be able to explain every line in an interview, so understanding matters more than speed.

Background notes: `docs/study-notes.md` covers every component in depth: equations, worked examples and derivations. Read the relevant section before implementing a component. This file is the spec. Where the two disagree, this file wins.

## Hard rules

1. **No copied implementation code.** Do not reproduce, paraphrase or port model, loss, replay or training code from `danijar/dreamerv3`, `NM512/dreamerv3-torch` or any other Dreamer implementation. Do not look up their source to answer "how should this be written". Work from the paper, this file and the study notes.
2. **Allowed from other repos:** explicit hyperparameter values (listed below), and the environment wrappers in `envs/` taken from the NM512 scaffold.
3. **Never change the index convention** (next section) partway through. If code or a paper equation uses a different convention, translate it into this one.
4. Do not add features outside scope: no discrete action spaces, Atari, Crafter or Minecraft.

## Who writes the code

**Adam writes the core implementation himself.** Claude Code must not write, edit or generate code for these parts, even when asked to "just fix it":

1. **The RSSM** (`src/models/rssm.py`, plus the encoder/posterior pieces in `networks.py`): the block GRU for h_t, the categorical z_t with unimix and straight-through, the prior p(z_t | h_t), the posterior q(z_t | h_t, e_t), observe() and imagine().
2. **The world-model losses** (`src/losses.py`, `src/models/distributions.py`, `src/models/heads.py`):
   - the prediction loss: reconstruction, twohot reward, binary continue
   - the dynamics and representation KL losses with free bits
   - symlog/symexp, twohot, and the bounded-normal actor distribution
3. **The actor-critic in imagination** (`src/returns.py`, and the actor and critic parts of `losses.py` and `agent.py`):
   - the λ-return and the replay target
   - the twohot critic with its EMA regularizer
   - continue weighting, the return scale S, and the Reinforce actor with entropy
4. **The training loop** (`src/agent.py`, `src/train.py`, `src/replay.py`): collecting N steps across E envs, the replay buffer with is_first, stored latents and the online queue, one gradient step (world model → imagine → critic → actor), and the slow-critic update.

What Claude Code should do for these parts:
- Answer questions: what a line or a PyTorch call does, why a shape is wrong, what the paper means.
- Review code Adam wrote. Point to the line and explain the problem, but let Adam make the fix.
- When Adam says "I want to do X", give a hint or a small baseline snippet that shows how to do it. Keep it short (roughly under 15 lines), on toy tensors, focused on the PyTorch technique (e.g. a straight-through sample, a reverse-time scan, gathering along one dimension). Do not hand over a finished DreamerV3 component. Put the snippet in chat; never write it into his files. He adapts it himself.
- When he's stuck but hasn't asked for code, start with the smallest useful hint in words, math or shapes, and escalate only if he asks.
- Help him debug: suggest what to print or test, and read tracebacks and outputs.
- If Adam explicitly asks Claude to write a specific piece of core code, confirm once before doing it, and keep it to the smallest piece asked for.

**Claude Code may write the rest after confirming with Adam:** the env setup and wrappers from the NM512 scaffold, config plumbing, logging/W&B, plotting, and the scaffolding for the test files. It may also propose test cases. Adam decides who writes each test.

## Working style

- Adam has ~7 years of deep RL, mostly TD3. Use TD3 comparisons when they help.
- One component at a time. Before Adam starts a component, offer a short plan if he wants one: the inputs, outputs, shapes and the equations involved. Then wait while he writes it.
- Short explanations, one concept at a time. Define every symbol before using it, and state all inputs and numbers up front.
- Prefer concrete numeric examples over abstract explanation.
- Name the assumption whenever there is one: which loss, which units, real vs. imagined, Option A vs. what DreamerV3 actually does.
- Ask before making a design decision listed under "Open decisions".

## Index convention (TD3 style, used everywhere)

- r_t and c_t are the reward and continue flag that result from taking a_t in s_t. Store them in the same replay record as a_t.
- Bootstraps use v_{t+1}. The paper's Eq. 5 writes v_t because it stores rewards with the arriving state. The math is the same; do not mix the two.
- Continue weights: w_0 = 1, w_{t+1} = w_t * c_t (stop-gradient).

## Algorithm summary

One cycle: collect N = B*T / replay_ratio agent steps across E envs, then run one gradient step:

1. **World model** on B sequences of length T. The GRU runs over T with batch B; everything else runs on B*T.
2. **Imagine** H = 15 steps from all B*T posterior states, using the prior and the actor. Dreamed steps are never stored in replay.
3. **Critic:**
   - Imagination loss: weight 1.0, continue-weighted.
   - Replay loss: weight 0.3, real sequences, skips is_last.
4. **Actor:** Reinforce, using the same continue weights.

### World model
- h_t = GRU(h_{t-1}, Linear([z_{t-1}, a_{t-1}])). The GRU is block-diagonal with 8 blocks, and deter = 8d.
- Encoder: e_t = enc(x_t). Posterior: q(z_t | h_t, e_t). Prior: p(z_t | h_t).
- z_t has K = 32 categorical latents with C = d/16 classes each. Mix 1% uniform into the probabilities. Straight-through gradients: z = onehot + p - sg(p).
- The reward, continue and decoder heads each read [h_t, z_t].
- Prediction loss:
  - vector observations: MSE in symlog space
  - images: MSE against pixels/255, sigmoid output, summed over pixels
  - continue: binary cross-entropy
  - reward: twohot cross-entropy
- Dynamics loss: max(1, KL[sg(q) || p]), weight 1.0. Representation loss: max(1, KL[q || sg(p)]), weight 0.1. Apply free bits per step, before reducing over the batch.

### Critic
- λ-return: R_t = r_t + γ c_t ((1-λ) v_{t+1} + λ R_{t+1}), with R_H = v_H.
- Replay target: R^rep_t = r_t + γ c_t ((1-λ) R^img_{t+1} + λ R^rep_{t+1}). R^img is the imagination return of the rollout that started at that real state. Compute imagination first, then replay. Replay critic gradients flow into the world model.
- Output: 255 logits. Bins are b = symexp(linspace(-20, 20, 255)), computed once at setup. Value v = sum(softmax * b); it is already in real units, so never symexp it again. The target y goes in raw: no symlog anywhere in the critic or reward loss.
- Twohot label: split weight linearly between the two neighboring bins (real-unit distances). Loss: cross-entropy. Example: y = 100 puts 0.71 on 95.25 and 0.29 on 111.66.
- Sum the bin expectation outward from the middle bin, so uniform probabilities give exactly 0.
- EMA regularizer: a second cross-entropy toward twohot(v_slow), weight 1.0. v_slow is an EMA copy of the critic with decay 0.98.
- Zero-initialize the output layer of the critic and the reward head.

### Actor
- Loss per step: -sg(w_t) * (sg((R_t - v(s_t)) / max(1, S)) * log π(a_t | s_t) + η H[π(·|s_t)]), with η = 3e-4.
- S = EMA(Per(R, 95) - Per(R, 5)), with EMA rate 0.01 (decay 0.99). This normalizes returns, not advantages.
- Continuous actions (bounded normal):
  - mean = tanh(network output)
  - std = (1.0 - 0.1) * sigmoid(raw + 2) + 0.1, so it stays in [0.1, 1.0]
  - the sampled action is not squashed, so there is no tanh Jacobian
  - use Normal(mean, std).log_prob(a).sum(-1)
- No gradient flows through the world model into the actor.

## Hyperparameters

Values come from the paper and the official config at commit e3f0224.

| Name | Value |
|---|---|
| batch B, sequence length T | 16, 64 |
| imagination horizon H | 15 |
| discount γ | 1 - 1/333 ≈ 0.997 |
| λ | 0.95 |
| loss scales: rec, rew, con, dyn, rep | 1, 1, 1, 1, 0.1 |
| loss scales: policy, value, repval | 1, 1, 0.3 |
| free nats | 1.0 |
| unimix, for latents and the actor | 0.01 |
| latents K | 32 |
| actor entropy η | 3e-4 |
| slow critic | rate 0.02 (decay 0.98), updated every step |
| return norm | 5th / 95th percentile, rate 0.01, limit 1.0 |
| optimizer | LaProp, lr 4e-5, β1 0.9, β2 0.999, eps 1e-20 |
| AGC clip | 0.3 |
| lr warmup | 1000 steps |
| activation, norm | SiLU, RMSNorm |
| replay capacity | 5e6, online queue on |
| actor MLP | 3 layers |
| critic MLP | 3 layers |
| reward head, continue head | 1 layer each |
| bins | 255 |
| encoder/decoder MLP | 3 layers |
| CNN | kernel 5, channel multipliers [2, 3, 4, 4] × depth |

**Model sizes** (official config). Every size uses K = 32 and 8 GRU blocks.

| Size | deter | hidden / units | classes C | CNN depth |
|---|---|---|---|---|
| 1M | 512 | 64 | 4 | 4 |
| 12M | 2048 | 256 | 16 | 16 |
| 200M | 8192 | 1024 | 64 | 64 |

**DMC settings: the paper and the official config disagree.** Ask Adam which to use.
- Paper: replay ratio 512, action repeat 2.
- Official config `dmc_proprio`: 1M model, train_ratio 1024, 1.1e6 steps.
- Official config `dmc_vision`: default 200M model, train_ratio 256.
- The official DMC env default is repeat 1.

## Repository layout

```
configs/      tiny_cpu.yaml, dmc_walker_walk.yaml, ablation_horizon.yaml
src/models/   networks.py (MLP/CNN, encoder, decoder), rssm.py, heads.py, distributions.py, world_model.py
src/          returns.py, losses.py, optim.py (LaProp + AGC), replay.py, agent.py, train.py
src/envs/     DMC wrapper (from the NM512 scaffold), E parallel envs
tests/        test_distributions.py, test_returns.py, test_losses.py, test_rssm.py
docs/         study-notes.md, equation-map.md, debugging.md, essay.md
```

- Write device-agnostic code: pass `device` around, and never call `.cuda()` directly.
- Build the MLP encoder/decoder path (vectors) first; add the CNN path (images) before the vision task.

## Build order and tests

Develop on CPU with `configs/tiny_cpu.yaml` (1M size, or smaller: d=64, K=4, C=8, B=2, T=8, H=5) and random tensors.

1. **distributions.py.**
   - symexp(symlog(x)) ≈ x.
   - Twohot of y = 100 gives 0.71 / 0.29 on bins 95.25 / 111.66, and label · bins = y.
   - Uniform probabilities give v = 0.
   - Straight-through: the forward pass is one-hot, and the gradient equals the gradient of the probabilities.
2. **rssm.py.**
   - Shapes are correct for observe() over T and imagine() over H.
   - is_first resets the state.
3. **losses.py.**
   - A KL below 1 nat gives exactly zero gradient.
   - The world model overfits one fixed batch to near-zero loss.
4. **returns.py.**
   - λ-return matches hand-derived 2-step and 3-step cases.
   - The replay target matches its hand-computed value.
   - S example: spread 20 with previous S = 19 gives 19.01; a spread of 0.3 divides by 1.
5. **Actor.**
   - Discrete check: π = (0.5, 0.3, 0.2), middle action sampled, Â = 0.16 gives logit gradient (0.08, -0.112, 0.032).
   - Continuous check: μ = 0, σ = 0.5, a = 0.3 gives dlogπ/dμ = 1.2 and dlogπ/dσ = -1.28.

Environment progression:
1. Gymnasium Pendulum-v1
2. DMC cartpole-swingup, proprio
3. DMC walker-walk, proprio
4. DMC walker-walk, 64×64 vision

## Open decisions (ask Adam)

- GRU input embedding: one Linear on the concatenation, or separate Linears per input; and normalization.
- Loss reduction over time: sum (paper Eq. 2) or mean.
- Episode starts: placeholder r_0 / c_0 and how is_first resets the state.
- Warm-up: how much random data to collect before training starts.
