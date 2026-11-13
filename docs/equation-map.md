# Equation map: DreamerV3 paper → code

Maps each equation in DreamerV3 (Hafner et al., [arXiv 2301.04104](https://arxiv.org/abs/2301.04104), **revision 2**, Apr 2024) to the function that implements it in this repo. The Notes columns record where the code follows the official implementation (`danijar/dreamerv3`, commit e3f0224) rather than the paper, and any convention of this repo.

## Conventions

- **Index convention (TD3 style).** Replay record $t$ holds $(x_t, a_t, r_t, c_t)$, where $r_t$ and $c_t$ result from taking $a_t$ in state $s_t$. The paper stores each reward with the state it arrives in; the math is the same.
- **Recurrence.** $h_t = f_\phi(h_{t-1}, z_{t-1}, a_{t-1})$, so the action feeding step $t$ is the previous record's action.
- **Features.** $s_t = [h_t, z_t]$, built by `RSSM.get_feat`.

| Symbol | Meaning | tiny_cpu (examples below) | walker-walk |
|---|---|---|---|
| $B$, $T$ | batch size, sequence length | 2, 8 | 16, 64 |
| $H$ | imagination horizon | 5 | 15 |
| $N$ | dreams per step, $B \cdot T$ | 16 | 1024 |
| $D$ | size of $h_t$ (`recurrent_dim`) | 512 | 512 |
| $G$ | GRU blocks (`gru_blocks`) | 8 | 8 |
| $K$, $C$ | categorical latents, classes per latent | 4, 8 | 32, 4 |
| $F$ | feature width $D + K C$ | 544 | 640 |
| $O$, $A$ | observation and action size | — | 24, 6 |

---

## Eq. 1: RSSM and world-model heads

$$
\begin{aligned}
\text{Sequence model:}\quad & h_t = f_\phi(h_{t-1}, z_{t-1}, a_{t-1}) \\
\text{Encoder:}\quad & z_t \sim q_\phi(z_t \mid h_t, x_t) \\
\text{Dynamics predictor:}\quad & \hat z_t \sim p_\phi(\hat z_t \mid h_t) \\
\text{Reward predictor:}\quad & \hat r_t \sim p_\phi(\hat r_t \mid h_t, z_t) \\
\text{Continue predictor:}\quad & \hat c_t \sim p_\phi(\hat c_t \mid h_t, z_t) \\
\text{Decoder:}\quad & \hat x_t \sim p_\phi(\hat x_t \mid h_t, z_t)
\end{aligned}
$$

| Term | Code | Notes |
|---|---|---|
| $f_\phi$ | `RSSM._gru_step` (called by `img_step`) | Block GRU, see below |
| $q_\phi$ | `VectorEncoder` → $e_t$, then `RSSM.posterior` on $[h_t, e_t]$ + `sample_categorical`, in `RSSM.obs_step` | The paper's "encoder" is two networks here |
| $p_\phi(\hat z_t \mid h_t)$ | `RSSM.prior` + `sample_categorical`, in `RSSM.img_step` | Prior and posterior share the same $h_t$. Layer counts (`prior_layers`, `post_layers`) follow the official config (`imglayers: 2`, `obslayers: 1`); the paper doesn't give them |
| reward | `heads.RewardHead` on `get_feat` | Twohot, 255 bins, zero-initialized output (Eqs. 10–12). The head on $s_{t+1}$ predicts $r_t$, because $s_{t+1}$ is the first state built from $a_t$ |
| continue | `heads.ContinueHead` on `get_feat` | One logit, binary cross-entropy. Same $s_{t+1} \to c_t$ offset |
| decoder | `networks.VectorDecoder` on `get_feat` | Predicts $\mathrm{symlog}(x_t)$, trained with MSE (Eq. 8). Vector observations only |

### Sequence model $f_\phi$: the block GRU

**Paper:** one sentence in Methods: a GRU with block-diagonal recurrent weights of 8 blocks, so the model can have many memory units without a quadratic increase in parameters and FLOPs. Everything else below follows the official implementation (`rssm.py::_core`).

**Goal:** produce $h_t$ from $h_{t-1}$, $z_{t-1}$ and $a_{t-1}$.

1. **Embed the three inputs.** Scale the action to $a / \max(1, \lvert a \rvert)$. Then three one-layer MLPs (Linear → RMSNorm → SiLU) map $h_{t-1}$, $z_{t-1}$ and $a_{t-1}$ to 64 numbers each, concatenated into a shared 192-wide embedding: `h_embedding`, `z_embedding`, `a_embedding`.
2. **Build each block's input.** Split $h_{t-1}$ into $G = 8$ blocks of $D/G = 64$. Each block gets its own 64-slice plus a copy of the shared 192, for 256 inputs per block, shape $(B, 8, 256)$. The raw slice gives each block exact access to its own memory; the shared embedding is the only way a block sees $z$, $a$ and the other blocks.
3. **Block hidden layer.** `gru_hidden = BlockLinear(G, 256, 64)`: each block multiplies by its own $256 \times 64$ matrix. Flatten to $(B, 512)$, apply `gru_norm = RMSNorm(512)` and SiLU, then unflatten to $(B, 8, 64)$.
4. **Block gate layer.** `gru_gates = BlockLinear(G, 64, 192)` gives 3 raw values per memory unit, split by `torch.chunk` into $\text{raw}_r$, $\text{raw}_c$, $\text{raw}_u$, each $(B, 8, 64)$.
5. **Gate equations** (element-wise, no weights):

$$
r = \sigma(\text{raw}_r), \qquad
\tilde h = \tanh(r \odot \text{raw}_c), \qquad
u = \sigma(\text{raw}_u - 1)
$$

$$
h_t = u \odot \tilde h + (1 - u) \odot h_{t-1}
$$

6. **Flatten** $(B, 8, 64) \to (B, 512)$ to get $h_t$.

**Where $h_{t-1}$ enters:** three times. Its own slice per block (step 2), the shared embedding of all of it (step 1), and raw in the final blend (step 5). The raw blend path lets memory, and gradients, pass through unchanged when $u \approx 0$.

**Differences from the textbook GRU:** the reset gate multiplies the raw candidate rather than $h_{t-1}$, and $-1$ on the update gate makes "keep the old value" the default at initialization. Both follow the official code.

**Why blocks:** a dense layer would connect every input to every output. Counting weights only:

| | Dense | 8 blocks |
|---|---|---|
| tiny_cpu ($D = 512$) | $704 \cdot 512 + 512 \cdot 1536 \approx 1.15$M | $8(256 \cdot 64) + 8(64 \cdot 192) \approx 0.23$M |
| 200M size ($D = 8192$) | $\approx 294$M | $\approx 59$M |

That's about 5× fewer at both sizes. `networks.BlockLinear` stores the $G$ small matrices as one parameter of shape $(G, \text{in}, \text{out})$ and applies them all at once with `einsum('...gi,gio->...go')`. It never builds the large matrix of zeros.

**RMSNorm** (`networks.RMSNorm`):

$$
y = \frac{x}{\sqrt{\tfrac{1}{n}\sum_i x_i^2 + \epsilon}} \odot g
$$

It normalizes each sample's vector to RMS ≈ 1 (per sample, not per batch), then a learned gain $g$, initialized to ones, sets the scale. $\epsilon$ prevents division by zero. It follows every hidden layer in the model.

### Latent $z_t$: categoricals, unimix and straight-through

`sample_categorical` turns logits $(\dots, K C)$ into:

$$
\tilde p = 0.99 \cdot \mathrm{softmax}(\ell) + 0.01 \cdot \tfrac{1}{C}, \qquad
z = \mathrm{onehot}(\text{sample} \sim \tilde p) + \tilde p - \mathrm{sg}(\tilde p)
$$

$K$ independent categoricals with $C$ classes each, flattened to $K C$. The forward pass is the one-hot sample; the backward pass uses the gradient of $\tilde p$ (straight-through). The 1% uniform mix (paper: "unimix") keeps every probability above zero, so the KL losses stay finite. The unimixed $\tilde p$ is what the KL losses in Eq. 3 use.

**In my own words** ($K = 4$ dice, $C = 8$ faces):

- **Softmax per die (`dim=-1`).** Each die's 8 probabilities sum to 1, and each die picks one face, so $z$ has 4 ones among 32 numbers. That gives $8^4 = 4096$ possible codes; a single softmax over all 32 would give only 32. Separate dice let each one encode a different aspect of the state.
- **Unimix.** $\tilde p = 0.99\,p + 0.01/8$, so no face drops below $0.00125$. Without it, a prior with $p = 0$ where $q > 0$ gives $\ln 0 = -\infty$, an infinite KL, and `NaN` gradients (the paper's "spikes in KL losses").
- **Why straight-through.** `Categorical.sample()` returns integers, so nothing can backpropagate through the sampled face. Without a fix, no gradient from the heads would reach the posterior, the encoder or the GRU through $z$.
- **What `onehot + probs - probs.detach()` does.** Forward: the two `probs` terms cancel, so $z$ is exactly the one-hot. Backward: $z = \text{probs} + c$ with $c$ a constant, so $\partial z / \partial\,\text{probs} = 1$, the gradient reaches the logits through `probs` once, and the sampling path is dead.
- **`probs` used twice.** `probs` goes into $z$ (for the heads and the GRU) and is also returned for the KL. Autograd adds the two gradients; that's the normal chain rule, not double counting.

### Rolling the RSSM through time

**`RSSM.observe`: real sequences (posterior).** Loops `obs_step` over $t = 0 \dots T-1$ with batch $B$:

- The previous action is zeros at $t = 0$, then `action[:, t-1]` (the recurrence uses $a_{t-1}$).
- Where `is_first` is set, $h$, $z$ and the previous action are zeroed before the step (episode reset).
- `img_step` computes $h_t$ and the prior; the posterior then reads $[h_t, e_t]$.
- The **posterior** state is fed forward to the next step, never the prior.
- Returns posteriors and priors, each stacked to $(B, T, \dots)$.

**`RSSM.imagine`: dreams (prior).** Starts from all $N = B \cdot T$ posterior states, flattened and detached. Each step:

1. The actor picks $a = \pi(\text{feat})$.
2. `img_step` gives the next state: the GRU, then a **prior** sample of $z$. There's no observation, so no posterior.

It runs $H$ steps under `torch.no_grad()` and returns states $(N, H+1, \dots)$ and actions $(N, H+1, A)$, where index 0 is the real start state. The action at index $H$ is sampled but never scored, because its reward would need a state $H+1$.

---

## Eq. 2: total world-model loss

$$
\mathcal{L}(\phi) \doteq \mathbb{E}_{q_\phi}\Big[\textstyle\sum_{t=1}^{T}\big(\beta_{\text{pred}}\,\mathcal{L}_{\text{pred}}(\phi) + \beta_{\text{dyn}}\,\mathcal{L}_{\text{dyn}}(\phi) + \beta_{\text{rep}}\,\mathcal{L}_{\text{rep}}(\phi)\big)\Big],
\qquad \beta_{\text{pred}} = 1,\ \beta_{\text{dyn}} = 1,\ \beta_{\text{rep}} = 0.1
$$

$\phi$ is every world-model weight: encoder, RSSM (GRU, prior, posterior), decoder, reward head, continue head.

| Term | Code | Notes |
|---|---|---|
| $\mathcal{L}(\phi)$ | `losses.world_model_loss` | Called from `Agent.train_step` on the $(B, T)$ replay batch |
| $\beta$ weights | `cfg.loss_scales`: `rec`, `rew`, `con` = 1; `dyn` = 1; `rep` = 0.1 | $\beta_{\text{pred}}$ is split into one scale per prediction head, all 1 |
| $\mathbb{E}_{q_\phi}[\sum_t \cdot]$ | `.mean()` over all $B \cdot T$ steps | Mean instead of sum over $t$, as in the official code. This only rescales the whole loss by $1/T$ |
| $z_t \sim q_\phi$ | posterior samples from `RSSM.observe` | "$\mathbb{E}_{q_\phi}$": the states the loss is computed on come from the posterior |

The world-model loss runs only on **real** replay steps ($T$), never on imagined ones ($H$).

## Eq. 3: prediction, dynamics and representation losses

$$
\begin{aligned}
\mathcal{L}_{\text{pred}}(\phi) &\doteq -\ln p_\phi(x_t \mid z_t, h_t) - \ln p_\phi(r_t \mid z_t, h_t) - \ln p_\phi(c_t \mid z_t, h_t) \\
\mathcal{L}_{\text{dyn}}(\phi) &\doteq \max\big(1,\ \text{KL}\big[\mathrm{sg}(q_\phi(z_t \mid h_t, x_t))\ \big\|\ p_\phi(z_t \mid h_t)\big]\big) \\
\mathcal{L}_{\text{rep}}(\phi) &\doteq \max\big(1,\ \text{KL}\big[q_\phi(z_t \mid h_t, x_t)\ \big\|\ \mathrm{sg}(p_\phi(z_t \mid h_t))\big]\big)
\end{aligned}
$$

| Term | Code | Notes |
|---|---|---|
| $-\ln p_\phi(x_t \mid z_t, h_t)$ | `recon_loss` → `rec` | Symlog squared error, summed over observation dims (Eq. 8) |
| $-\ln p_\phi(r_t \mid z_t, h_t)$ | `reward_loss` → `rew` | Twohot cross-entropy (Eqs. 11–12). Head on $s_{t+1}$ vs. $r_t$: `reward_logits[:, 1:]` vs. `reward[:, :-1]` |
| $-\ln p_\phi(c_t \mid z_t, h_t)$ | `continue_loss` → `con` | `binary_cross_entropy_with_logits` (paper: "logistic regression"). Same $s_{t+1} \to c_t$ offset |
| episode-boundary mask | `valid = ~is_first[:, 1:]` | Drops pairs where $s_{t+1}$ starts a new episode, so it can't be asked to predict the old episode's $r_t$, $c_t$ |
| $\text{KL}[q \,\|\, p]$ | `categorical_kl` | $\sum_{k}\sum_{c} q \,(\ln q - \ln p)$ over $K$ latents × $C$ classes, on the unimixed probabilities |
| $\mathrm{sg}(\cdot)$ | `.detach()` in `dyn_rep_losses` | The only difference between dyn and rep is which side is detached (and the weight) |
| $\max(1, \cdot)$ | `torch.clamp(..., min=free_nats)`, `free_nats = 1.0` | Applied per step, on the $(B, T)$ KL, **before** the mean |

**Paper wording vs. code:** the paper says the prediction loss trains "the decoder and reward predictor via the symlog squared loss." In this repo (and the official code), the reward head uses twohot cross-entropy instead (Eqs. 10–12); only the decoder uses symlog squared error.

### Which network each term trains

| Network | Gets gradient from | Weights |
|---|---|---|
| Decoder, reward head, continue head | $\mathcal{L}_{\text{pred}}$ | 1 |
| Prior MLP | $\mathcal{L}_{\text{dyn}}$ only | 1 |
| Posterior MLP + encoder | $\mathcal{L}_{\text{pred}}$ and $\mathcal{L}_{\text{rep}}$ | 1 vs. 0.1 |
| GRU (shared $h_t$) | all three | |

### In my own words

- **Prior vs. posterior.** The posterior sees the observation $x_t$ (through the embedding $e_t$); the prior sees only $h_t$. In a dream there's no $x_t$, so the posterior can't run. The dream starts from a real posterior state, and every step after that uses the prior. So the prior has to learn to guess what the posterior would have said.
- **Dynamics loss.** The posterior is detached, so the prior is trained to resemble the posterior.
- **Representation loss.** The prior is detached, so the posterior is trained to resemble the prior. We want the posterior to focus on the main features of $x_t$ that the prior could pick up from $h_t$. Bad case: the posterior encodes sensor noise from $x_t$, which the prior can never predict. So the posterior is gently taught not to stray too far from the prior.
- **Why 0.1.** Inside the posterior, $\mathcal{L}_{\text{pred}}$ (1) says "keep the information in $x_t$" and $\mathcal{L}_{\text{rep}}$ (0.1) says "stay close to the prior." If rep won, the posterior would learn to ignore $x_t$ and copy the prior (posterior collapse).
- **Free bits.** $\max(1, \text{KL})$: once a step's KL is below 1 nat, the loss is a constant, so its gradient is 0. This doesn't push $q$ and $p$ apart; it just stops pulling them together once they're close enough, and leaves that step to $\mathcal{L}_{\text{pred}}$. Paper: free bits "disables them while they are already minimized well to focus learning on the prediction loss."
- **Why per step.** Example: 15 steps with KL 0.2 and one with KL 10. Clamping per step, the bad step still gets a gradient. Averaging first gives $13/16 = 0.81$, which clamps to 1, so **no** step gets a gradient and the good steps hide the bad one.

---

## Eq. 4: actor and critic

$$
\text{Actor:}\quad a_t \sim \pi_\theta(a_t \mid s_t) \qquad\qquad \text{Critic:}\quad v_\psi(R_t \mid s_t)
$$

$\theta$ = actor weights, $\psi$ = critic weights (both separate from the world model's $\phi$). Paper: the critic learns "to approximate the distribution of returns for each state," so $v_\psi(R_t \mid s_t)$ is a distribution, and the single value is its expectation, $v_t \doteq \mathbb{E}[v_\psi(\cdot \mid s_t)]$.

| Term | Code | Notes |
|---|---|---|
| $\pi_\theta$ | `agent.Actor` → `bounded_normal` | Covered with Eq. 6 |
| $v_\psi(R_t \mid s_t)$ | `agent.Critic` (a `RewardHead`); `critic(feat)` → 255 bin logits | Same bins and twohot as the reward head |
| $v_t = \mathbb{E}[v_\psi(\cdot \mid s_t)]$ | `critic.predict(feat)` = `bins_expectation(softmax(logits), bins)` | $\sum_i \mathrm{softmax}(\text{logits})_i\, b_i$ (Eq. 10) |
| $s_t$ | `get_feat` = $[h_t, z_t]$, 544 wide | Both networks read only the state. The critic is a $V(s)$, not TD3's $Q(s, a)$ |

## Eq. 5: critic loss and λ-return

$$
\mathcal{L}(\psi) \doteq -\sum_{t=1}^{T} \ln p_\psi\big(R^\lambda_t \mid s_t\big),
\qquad
R^\lambda_t \doteq r_t + \gamma\, c_t\Big((1-\lambda)\,v_{t+1} + \lambda\, R^\lambda_{t+1}\Big),
\qquad
R^\lambda_H \doteq v_H
$$

$\gamma = 0.997$, $\lambda = 0.95$. The paper writes $v_t$ inside the bracket and $R^\lambda_T \doteq v_T$, because it stores each reward with the state it arrives in; in this repo's TD3 convention it's $v_{t+1}$, and the dream length is $H$.

| Term | Code | Notes |
|---|---|---|
| $R^\lambda_t$ | `returns.lambda_return(reward, cont, value, gamma, lam)` | Backward loop from $t = H-1$ to 0; returns $(N, H)$ |
| $r_t$, $c_t$ | `reward.predict(feat[:, 1:])`, `cont.predict(feat[:, 1:])` | Heads on $s_{t+1}$ predict $r_t$, $c_t$ (the first state that contains $a_t$) |
| $v_{t+1}$, $v_H$ | `critic.predict(feat)`, $(N, H+1)$ | `value[:, -1]` is the start $R_H = v_H$ |
| $-\ln p_\psi(R^\lambda_t \mid s_t)$ | `critic_loss` → `reward_loss(logits, target, bins)` | Twohot cross-entropy, reused from the reward head |
| states with a loss | `critic(feat[:, :-1])` | $s_0 \dots s_{H-1}$: $s_H$ has no return of its own |
| $\sum_t$ | `.mean()` over $N \cdot H$ | |

**Not in Eq. 5, in the code (and the official code):**

| Extra | Code | Notes |
|---|---|---|
| continue weights $w_t$ | `continue_weights(cont)`: $w_0 = 1$, $w_t = \prod_{k<t} c_k$ | Each step's loss × $w_t$. The official code also multiplies by $\gamma^t$ and includes the continue prediction on the start state (`agent.py`: `cumprod(disc * con) / disc`). For walker-walk (no terminals, $c \approx 1$) the difference is small |
| slow-critic regularizer | `+ slow_reg * reward_loss(logits, slow_value, bins)`, `slow_reg = 1` | Paper p. 6: "regularizing the critic towards predicting the outputs of an exponentially moving average of its own parameters" |
| slow critic update | `update_slow_critic`: $\psi_{\text{slow}} \leftarrow 0.98\,\psi_{\text{slow}} + 0.02\,\psi$ | Once per train step, no gradient, never optimized |

The replay critic loss (Methods, weight 0.3) is a separate section below.

### In my own words

- **One value, two calls.** `critic(feat)` gives the 255 logits, used in the critic loss. `critic.predict(feat)` gives the single $v_t$, used in the λ-return and the advantage.
- **λ = 0** gives $R_t = r_t + \gamma c_t v_{t+1}$, the one-step TD target (TD3's $y = r + \gamma(1-d)\,Q_{\text{target}}$, with $c$ in place of $1 - d$ and $V$ in place of $Q$).
- **λ = 1** gives $R_t = r_t + \gamma c_t R_{t+1}$: every dreamed reward, with the critic used only once, at the end ($v_H$).
- **Worked example** ($H = 2$, $\gamma = 1$, $c = 1$; $r_0 = 1$, $r_1 = 2$; $v_1 = 10$, $v_2 = 20$):

| $\lambda$ | $R_1$ | $R_0$ |
|---|---|---|
| 0 | $2 + 20 = 22$ | $1 + 10 = 11$ |
| 0.95 | $2 + 0.05(20) + 0.95(20) = 22$ | $1 + 0.05(10) + 0.95(22) = 22.4$ |
| 1 | $2 + 20 = 22$ | $1 + 22 = 23$ |

- **Why $R_H = v_H$.** The dream stops at $s_H$. A reward from $s_H$ would need the reward head on $s_{H+1}$, which was never dreamed, so the critic's $v_H$ stands in for everything after the dream.
- **Why backward.** It's recursive: each $R_t$ needs $R_{t+1}$, so start from $R_H = v_H$ and walk back. `returns[::-1]` flips the list back to forward order.
- **Where each value goes.** $v_0 \dots v_{H-1}$ go into the advantage $R_t - v_t$ (Eq. 6); $v_1 \dots v_H$ go into the λ-return.
- **Twohot is the label.** $-\ln p_\psi(R_t \mid s_t) = -\sum_i \mathrm{twohot}(R_t)_i \ln \mathrm{softmax}(\text{logits})_i$. With bins $[0, 10, 20]$: $R = 10$ gives label $[0, 1, 0]$ and loss $-\ln p_2$ (the paper's form exactly); $R = 15$ gives $[0, 0.5, 0.5]$ and loss $-0.5 \ln p_2 - 0.5 \ln p_3$.
- **Continue weights.** $w_t$ is the probability the walker is still alive at $s_t$. If $c_1 = 0.1$, then $r_1$ still fully counts, but $s_2$ is 90% likely to be a state after the walker fell, one that never happens in the real environment, so its critic loss is weighted by $w_2 = 0.1$.
- **Slow critic.** Its weights trail the critic's (2% per step), and it is never trained. Its prediction $v^{\text{slow}}_t$ is twohot-encoded as a second label at the same state. Example: return 40, slow critic 20, so the critic is pulled to about 30 instead of jumping to 40. It acts as an anchor against chasing noisy targets. Unlike TD3, the slow copy doesn't compute the target; the return uses the current critic.

---

## Replay critic loss (unnumbered, Methods p. 6)

Paper: "we apply the critic loss both to imagined trajectories with loss scale $\beta_{\text{val}} = 1$ and to trajectories sampled from the replay buffer with loss scale $\beta_{\text{repval}} = 0.3$. The critic replay loss uses the imagination returns $R^\lambda_t$ at the start states of the imagination rollouts as on-policy value annotations for the replay trajectory to then compute λ-returns over the replay rewards."

$$
R^{\text{rep}}_t = r_t + \gamma\, c_t\Big((1-\lambda)\,R^{\text{img}}_{t+1} + \lambda\, R^{\text{rep}}_{t+1}\Big),
\qquad R^{\text{rep}}_{T-1} = R^{\text{img}}_{T-1}
$$

| Term | Code | Notes |
|---|---|---|
| $R^{\text{img}}_t$ | `returns[:, 0].reshape(B, T)` | λ-return at dream index 0 of the dream that started at real state $s_t$. One per real state ($N = B \cdot T$ dreams) |
| $R^{\text{rep}}_t$ | `returns.replay_return(...)` → $(B, T-1)$ | Same loop as `lambda_return`, with `img_return` in place of `value` |
| $r_t$, $c_t$ | `batch["reward"]`, `batch["cont"]` | **Real** rewards and continues |
| episode boundary | `torch.where(is_first[:, i+1], img_return[:, i], R_t)` | If $s_{t+1}$ starts a new episode, restart with $R^{\text{rep}}_t = R^{\text{img}}_t$ |
| mask | `replay_mask = ~is_first[:, 1:]` | Drops the restarted steps from the loss |
| loss | `critic_loss(critic(world_model_dict["feat"][:, :-1]), ...)` × 0.3 | Same twohot loss and slow-critic regularizer as Eq. 5 |
| gradient into the world model | posterior features **not** detached | Official: `repval_grad: True` |

**Final observation:** `dmc_env.py` drops each episode's final observation, so there is no separate "last" record and boundaries are found from `is_first`. Cost: the last reward of each episode (1 in 1000 for walker-walk) gets no reward-head or replay-critic loss.

### In my own words

- **What it is.** The same critic and the same loss function, applied to **real** posterior states from the buffer, with targets built from **real** rewards. The imagination critic loss only ever sees dreamed rewards from the reward head.
- **$R^{\text{img}}$ vs. $R^{\text{rep}}$.** $R^{\text{img}}_t$ is the dream's λ-return at real state $s_t$ (dreamed rewards + critic): an ingredient, standing in for $V_{t+1}$. $R^{\text{rep}}_t$ is the target: real rewards + $R^{\text{img}}$ for the rest. Both estimate the same thing, the discounted sum of all future rewards from that state, in the same units, so one can stand in for the other's "rest of the future."
- **Why `returns[:, 0]`.** Column 0 is the dream step that sits on the real start state; every other column is a dreamed state.
- **Why the loop starts at $T-2$.** A chunk has $T$ states, indexed $0 \dots T-1$. The last one is set directly ($R^{\text{rep}}_{T-1} = R^{\text{img}}_{T-1}$), like $R_H = v_H$ in the dream, so the loop starts one before it. A dream has $H+1$ states, indexed $0 \dots H$, so `lambda_return` starts at $H-1$.
- **Worked example** ($T = 3$, $\gamma = 1$, $c = 1$, $\lambda = 0.95$; $r = [1, 2]$; $R^{\text{img}} = [20, 22.4, 25]$):

| | `is_first = [F, F, F]` | `is_first = [F, F, T]` (episode B starts at $s_2$) |
|---|---|---|
| $R^{\text{rep}}_2$ | 25 | 25 |
| $R^{\text{rep}}_1$ | $2 + 0.05(25) + 0.95(25) = 27$ | computed 27, replaced by $R^{\text{img}}_1 = 22.4$ |
| $R^{\text{rep}}_0$ | $1 + 0.05(22.4) + 0.95(27) = 27.77$ | $1 + 0.05(22.4) + 0.95(22.4) = 23.4$ |
| mask | $[1, 1]$ | $[1, 0]$ |

- **Boundary.** Episode A's return must not include episode B's values. The restarted $R^{\text{rep}}_1 = 22.4$ is just a copy of the dream return with no real reward in it, so the mask removes it.
- **A gap between targets** (e.g. $R^{\text{rep}}_0 = 27.77$ vs. $R^{\text{img}}_0 = 20$) means the dream is underestimating the real rewards. The replay loss pulls the critic toward the real-reward targets, and because the critic bootstraps every dream, the correction flows back into imagination.
- **Gradients.** The dream states were made under `no_grad`, so the imagination critic loss updates only the critic. The real states weren't detached, so the replay loss also nudges the world model to encode information that makes future returns easy to predict.

---

## Eq. 6: actor loss

$$
\mathcal{L}(\theta) \doteq -\sum_{t=1}^{T} \mathrm{sg}\!\left(\frac{R^\lambda_t - v_\psi(s_t)}{\max(1, S)}\right) \log \pi_\theta(a_t \mid s_t) \;+\; \eta\, \mathrm{H}\big[\pi_\theta(a_t \mid s_t)\big],
\qquad \eta = 3 \times 10^{-4}
$$

| Term | Code | Notes |
|---|---|---|
| $R^\lambda_t$ | `returns` from `lambda_return`, $(N, H)$ | The imagination λ-returns (Eq. 5), **every** column, not just column 0 |
| $v_\psi(s_t)$ | `values[:, :-1]` | Critic at dream states $s_0 \dots s_{H-1}$; $v_H$ is only the bootstrap |
| advantage $\hat A_t$ | `(returns - values[:, :-1]) / return_scale.scale()` | Computed under `no_grad` |
| $\mathrm{sg}(\cdot)$ | `torch.no_grad()` + `advantage.detach()` in `actor_loss` | |
| $\max(1, S)$ | `ReturnScale.scale()` | Eq. 7 |
| $\pi_\theta(\cdot \mid s_t)$ | `actor(feat[:, :-1])` → `bounded_normal` | `Normal`, batch shape $(N, H, A)$; walker-walk $A = 6$ |
| $\log \pi_\theta(a_t \mid s_t)$ | `dist.log_prob(actions[:, :-1]).sum(-1)` | Independent motors: log-probs add. Recomputed **with** gradient (`imagine` ran under `no_grad`) |
| $\mathrm{H}[\pi_\theta]$ | `dist.entropy().sum(-1)` | For a Gaussian, $\tfrac{1}{2}\ln(2\pi e \sigma^2)$ per motor: depends only on $\sigma$ |
| $\eta$ | `cfg.actor.entropy_coef = 3e-4` | |
| continue weights | `weight=cont_weights`, the same $w_t$ as the critic loss | Not in Eq. 6; in the official code |
| $\sum_t$ | `.mean()` over $N \cdot H$ | |

**Policy distribution (official code; the paper doesn't give it):** `bounded_normal`

$$
\mu = \tanh(\mu_{\text{raw}}), \qquad \sigma = (1 - 0.1)\,\mathrm{sigmoid}(\sigma_{\text{raw}} + 2) + 0.1
$$

- $\mu \in (-1, 1)$, matching DMC's action range. Samples are **not** squashed, so there's no tanh Jacobian in `log_prob`.
- $\sigma \in [0.1, 1]$. At initialization (output weights × 0.01, so $\sigma_{\text{raw}} \approx 0$), $\sigma = 0.9 \cdot \mathrm{sigmoid}(2) + 0.1 \approx 0.89$.
- Out-of-range samples: the environment clips them, and the RSSM rescales the copy fed to the GRU, $a / \max(1, \lvert a \rvert)$ (`rssm.py` line 106), which element-wise is a clip to $\pm 1$. `log_prob` uses the unclipped sample. Metric `act_oob` tracks how often this happens.

## Eq. 7: return scale

$$
S \doteq \mathrm{EMA}\big(\mathrm{Per}(R^\lambda_t, 95) - \mathrm{Per}(R^\lambda_t, 5),\ 0.99\big)
$$

| Term | Code | Notes |
|---|---|---|
| $\mathrm{Per}(R^\lambda, 95)$, $\mathrm{Per}(R^\lambda, 5)$ | `torch.quantile(returns.flatten(), 0.95 / 0.05)` | Over all $N \cdot H$ imagination returns of this step |
| EMA, decay 0.99 | `ReturnScale.update`: `spread = 0.99 * spread + 0.01 * new` | First call initializes $S$ to the first spread |
| $\max(1, S)$ | `ReturnScale.scale()` | Returns 1 before the first update |

Paper: "Previous work typically normalizes advantages rather than returns, which puts a fixed amount of emphasis on maximizing returns over entropy regardless of whether rewards are within reach. Scaling up advantages when rewards are sparse can amplify noise that outweighs the entropy regularizer and stagnates exploration."

### In my own words

- **Advantage.** $R^\lambda_t - v_t$: how much better the dream went from $s_t$ than the critic expected. Example: $22.4 - 20 = 2.4$. Positive makes $a_t$ more likely; negative makes it less likely.
- **Why $\mathrm{sg}$.** The advantage is the grade. The actor should improve by changing which actions it picks ($\log \pi$), not by changing the grade (moving $v_t$ or the returns).
- **The sign.** Loss $= -(2.4 \log \pi + \eta \mathrm{H})$. To shrink it, the optimizer pushes $\log \pi$ up, so the action becomes more likely.
- **Entropy.** It pushes only on $\sigma$. With $\eta = 0$, the policy would collapse onto whatever it already does most and stop exploring.
- **$S$, percentiles.** One step gives $16 \times 5 = 80$ returns. If $P_5 = 10$ and $P_{95} = 50$, the spread is 40; EMA example $0.99(38) + 0.01(40) = 38.02$. One outlier at 500 would make max − min $= 490$ (shrinking every advantage ~12×), but leaves $P_{95} - P_5 \approx 40$.
- **Why $\max(1, S)$.** If $S = 0.02$, dividing by $S$ would blow up advantages 50×, and with near-zero returns those differences are mostly noise. The max only scales **down**, never up: with tiny or absent rewards, the advantages stay small and entropy wins, so the actor keeps exploring.
- **Why divide at all.** One fixed set of hyperparameters (including $\eta$) across domains with very different reward scales: Atari spread ~2000, walker ~40, sparse ~0. Dividing puts advantages at about $\pm 1$ everywhere, so the reward-vs-entropy balance is the same.

---

## Eq. 8–9: symlog squared loss

$$
\mathcal{L}(\theta) \doteq \tfrac{1}{2}\big(f(x, \theta) - \mathrm{symlog}(y)\big)^2,
\qquad
\mathrm{symlog}(x) \doteq \mathrm{sign}(x)\ln(\lvert x \rvert + 1),
\qquad
\mathrm{symexp}(x) \doteq \mathrm{sign}(x)\big(\exp(\lvert x \rvert) - 1\big)
$$

| Term | Code | Notes |
|---|---|---|
| symlog, symexp | `distributions.symlog`, `distributions.symexp` | `log1p` / `expm1` for precision near 0 |
| Eq. 8 | `losses.recon_loss`: `sum((recon - symlog(obs).detach())**2, -1)` | Decoder only. No $\tfrac{1}{2}$, summed over observation dims: a constant rescale, same minimum |
| prediction in real units | $\mathrm{symexp}(f)$ | Not needed for the loss |

### In my own words

- **Why not $\ln$.** $\ln$ is undefined for negatives and $\ln(0) = -\infty$. symlog keeps the sign, is finite at 0 (the $+1$), is close to $x$ near 0, and compresses large magnitudes: $\mathrm{symlog}(1) = 0.69$, $\mathrm{symlog}(100) = 4.6$.
- **Why it helps.** Walker joint velocities reach about ±30 while angles stay within about ±3. In raw units the velocity errors would dominate the squared error; after symlog it's $3.43$ vs. $1.39$, so neither drowns out the other.

## Eq. 10–12: bins, expectation, twohot

$$
B \doteq \mathrm{symexp}\big([-20, \ldots, +20]\big),
\qquad
\hat y \doteq \mathrm{softmax}(f(x))^{\mathsf T} B,
\qquad
\mathcal{L}(\theta) \doteq -\mathrm{twohot}(y)^{\mathsf T} \log \mathrm{softmax}(f(x, \theta))
$$

$$
\mathrm{twohot}(y)_i \doteq
\begin{cases}
\lvert b_{k+1} - y \rvert \,/\, \lvert b_{k+1} - b_k \rvert & \text{if } i = k \\
\lvert b_k - y \rvert \,/\, \lvert b_{k+1} - b_k \rvert & \text{if } i = k + 1 \\
0 & \text{else}
\end{cases}
\qquad k \doteq \textstyle\sum_{j} \delta(b_j < y)
$$

| Term | Code | Notes |
|---|---|---|
| $B$ (Eq. 10) | `make_bins(255, 20.0)`, registered as a buffer | Mirrors the non-negative half so the middle bin is exactly 0 and the bins are exactly symmetric |
| $\hat y = \mathrm{softmax}^{\mathsf T} B$ (Eq. 10) | `bins_expectation`, called by `RewardHead.predict` / `Critic.predict` | Sums outward from the middle bin so uniform probabilities give exactly 0 |
| $\mathrm{twohot}(y)$ (Eq. 12) | `twohot_encode`: `searchsorted` for the bin above, weights = opposite distance / gap | Targets outside the range are clamped to the end bins |
| Eq. 11 | `losses.reward_loss`: `-(label * log_softmax(logits)).sum(-1)` | Used by the reward head, the critic, the replay critic and the slow-critic regularizer |
| zero-init output | `RewardHead` / `Critic` output layer weights = 0 | Paper p. 6: random init "can result in large predicted rewards that can delay the onset of learning" |

### In my own words

- **Bins.** 255 points evenly spaced in symlog space (step $40/254 = 0.157$), mapped to real units with symexp:

| index from middle | symlog value | bin in real units |
|---|---|---|
| 0 | 0 | 0 |
| 1 | 0.157 | 0.17 |
| 2 | 0.315 | 0.37 |
| 29 | 4.567 | 95.25 |
| 30 | 4.724 | 111.66 |

  Close together near 0, far apart at large values, but each gap is about the same percentage (~16%) of where it sits. That's why one set of bins works for rewards of 1 and returns of 10,000.
- **Twohot example, $y = 100$.** It falls between bins 29 and 30: $w_{95.25} = 11.66 / 16.41 = 0.71$ and $w_{111.66} = 4.75 / 16.41 = 0.29$. Check: $0.71(95.25) + 0.29(111.66) = 100$.
- **The loss is just $-\ln p$.** With bins $[0, 10, 20]$: $y = 10$ gives label $[0, 1, 0]$ and loss $-\ln p_2$; $y = 15$ gives $[0, 0.5, 0.5]$ and $-0.5\ln p_2 - 0.5\ln p_3$.
- **No symlog in this loss.** The bins already carry the compression (they're symexp-spaced), and the twohot split is linear in real units.

### Why twohot and not symlog MSE (the paper vs. the official code)

The paper's world-model paragraph says the reward head uses symlog MSE; the official code (and this repo) uses twohot. Both predict one number, the reward $r$.

**Method A, symlog MSE.** One output $o$:

$$\text{loss} = \big(o - \mathrm{symlog}(r)\big)^2, \qquad \text{prediction} = \mathrm{symexp}(o)$$

Example: $r = 100$, target $\mathrm{symlog}(100) = 4.62$. Output $o = 4.0$ gives loss $0.38$ and prediction $\mathrm{symexp}(4.0) = 53.6$.

**Method B, twohot.** 255 logits $\to$ softmax $\to$ probabilities $p_i$ over fixed bins $b_i$:

$$\text{prediction} = \sum_{i=1}^{255} p_i\, b_i, \qquad \text{loss} = -\sum_{i=1}^{255} y_i \ln p_i, \quad y = \mathrm{twohot}(r)$$

Example: $r = 100$ gives the label 0.71 on bin 95.25 and 0.29 on bin 111.66, so the loss is $-(0.71 \ln p_{95.25} + 0.29 \ln p_{111.66})$, minimized when the prediction is $0.71(95.25) + 0.29(111.66) = 100$.

**Zero-init output.** The last layer (64 → 255) starts with zero weights and bias, so every logit is 0, every $p_i = 1/255$, and the prediction is exactly 0 (the bins are symmetric, so each $+b$ cancels its $-b$). A random init could put probability on a far bin like $4.85 \times 10^8$; the paper says such random initial predictions "can delay the onset of learning."

**Reason 1: unbiased under noisy targets.** At one state, $r = 0$ half the time and $100$ half the time, so the true expected reward is 50.

| Method | Learns | Prediction |
|---|---|---|
| symlog MSE | $o$ = average of $\mathrm{symlog}(0) = 0$ and $\mathrm{symlog}(100) = 4.62$ = 2.31 | $\mathrm{symexp}(2.31) = e^{2.31} - 1 \approx 9.1$ |
| twohot | 50% on bin 0; 50% split 0.71 / 0.29 on 95.25 / 111.66 | $0.5 \times 0 + 0.5 \times 100 = 50$ |

Squashing shrinks big values more than small ones, so averaging in symlog space drags the prediction toward the small outcome. Twohot averages in real units.

**Reason 2: the gradient on the logits is $p - y$, always in $[-1, 1]$.** Derivation with 3 bins, logits $\ell_1, \ell_2, \ell_3$, label $y = [y_1, y_2, y_3]$:

1. Softmax with shared denominator $S = e^{\ell_1} + e^{\ell_2} + e^{\ell_3}$: $\;p_1 = e^{\ell_1}/S,\ p_2 = e^{\ell_2}/S,\ p_3 = e^{\ell_3}/S$.
2. Log of one: $\ln p_2 = \ln e^{\ell_2} - \ln S = \ell_2 - \ln S$ (using $\ln(a/b) = \ln a - \ln b$ and $\ln e^x = x$). This is `log_softmax`. Check with $\ell = [0, 1, 2]$: $S = 11.107$, and $\ln p_3 = \ln(7.389/11.107) = -0.408 = 2 - \ln 11.107$.
3. Substitute into the loss:
$$\text{loss} = -\big(y_1(\ell_1 - \ln S) + y_2(\ell_2 - \ln S) + y_3(\ell_3 - \ln S)\big)$$
4. Multiply out; the label sums to 1:
$$\text{loss} = -y_1 \ell_1 - y_2 \ell_2 - y_3 \ell_3 + (y_1 + y_2 + y_3)\ln S = -y_1 \ell_1 - y_2 \ell_2 - y_3 \ell_3 + \ln S$$
5. Differentiate with respect to $\ell_2$. Only $-y_2 \ell_2$ in the first three terms contains $\ell_2$, giving $-y_2$. For $\ln S$, the chain rule gives $\frac{1}{S} e^{\ell_2} = p_2$. So:
$$\frac{\partial\, \text{loss}}{\partial \ell_2} = p_2 - y_2 \qquad\text{(and likewise } p_1 - y_1,\ p_3 - y_3\text{)}$$

Numbers at init ($p = [\tfrac13, \tfrac13, \tfrac13]$, $y = [0, 0.71, 0.29]$): gradient $= [+0.33, -0.38, +0.04]$. Descent raises $\ell_2$, lowers $\ell_1$, barely moves $\ell_3$. Each entry is a difference of two probabilities, so it stays in $[-1, 1]$ whether the reward was 100 or 100,000.

In general notation, $\text{loss} = -\sum_i y_i \ln p_i$ with $p_i = e^{\ell_i} / \sum_j e^{\ell_j}$, and $\partial\,\text{loss} / \partial \ell_k = p_k - y_k$. Here $i$ and $j$ both mean "each bin" (two letters only so the two sums don't get mixed up), and $k$ is the one logit being differentiated.

---

## Unnumbered pieces

| Piece | Paper | Code | Notes |
|---|---|---|---|
| Unimix (1%) | Methods, "Distributions" | `sample_categorical` | See Eq. 1 |
| Block GRU | Methods, one sentence | `RSSM._gru_step`, `BlockLinear` | See Eq. 1 |
| Free bits | p. 5 | `clamp(min=free_nats)` | See Eq. 3 |
| Zero-init reward/critic output | p. 6 | `RewardHead` output weights and bias = 0 | See Eq. 10–12 |
| Continue weights $w_t$ | official code | `continue_weights` | See Eq. 5 |
| Slow critic | p. 6 | `update_slow_critic`, `slow_reg` | See Eq. 5 |
| Replay critic loss (0.3) | p. 6 | `replay_return`, `repval_loss` | Own section above |
| AGC | Methods, "Optimizer" | `optim.agc_clip_` | Below |
| LaProp | Methods, "Optimizer" | `optim.LaProp` | Below |
| Replay ratio | Methods | `run.train_every` | Below |

### Optimizer: AGC + LaProp

Per weight tensor, in order:

$$
\begin{aligned}
\text{AGC:}\quad & g \leftarrow g \,\big/\, \max\!\Big(1,\ \tfrac{\lVert g \rVert}{0.3 \cdot \max(10^{-3},\ \lVert p \rVert)}\Big) \\
\text{RMS:}\quad & v \leftarrow \beta_2 v + (1-\beta_2)\, g^2, \qquad u = g \,\big/\, \big(\sqrt{v / (1 - \beta_2^t)} + \epsilon\big) \\
\text{Momentum:}\quad & m \leftarrow \beta_1 m + (1-\beta_1)\, u, \qquad p \leftarrow p - \eta_t\, m / (1 - \beta_1^t)
\end{aligned}
$$

| Setting | Value | Notes |
|---|---|---|
| lr | $4 \times 10^{-5}$, linear warm-up over 1000 updates | Official config |
| $\beta_1$, $\beta_2$ | 0.9, 0.999 | **Paper says $\beta_2 = 0.99$**; this repo follows the official config (0.999) |
| $\epsilon$ | $10^{-20}$ | Paper |
| AGC | 0.3, $p_{\min} = 10^{-3}$ | Paper |
| Optimizers | one per module: world model, actor, critic | One `backward()` on the summed loss; each optimizer steps only its own parameters |

**In my own words: AGC**

- **$\lVert p \rVert$** is a tensor's L2 norm, $\sqrt{\text{sum of squares of all its entries}}$. Example: bias $[0.06, -0.08]$ has $\lVert p \rVert = 0.1$; a matrix of 100 entries of $\pm 1$ has $\lVert p \rVert = 10$. The gradient has the same shape, so $\lVert g \rVert$ is measured the same way.
- **Problem with a fixed cap** (`clip_grad_norm_(100)`, as in my TD3 work). A gradient of norm 50 passes for both tensors. That's modest for the matrix, but 500× the bias itself, which would wipe out what the bias had learned.
- **AGC's fix.** Each tensor's cap is 30% of its own norm: 3 for the matrix, 0.03 for the bias. Gradients under the cap are untouched; larger ones are scaled down to the cap.
- **`pmin`.** Zero-initialized layers (reward head, critic output) have $\lVert p \rVert = 0$, so the cap would be 0 and they could never learn. $\max(10^{-3}, \lVert p \rVert)$ gives them a cap of $0.0003$ until their weights grow.
- **Loss scales.** Multiply a loss by 10 and a fixed cap of 100 suddenly clips far more often, so it has to be re-tuned. AGC's threshold means the same thing everywhere (no step may change a tensor by more than ~30% of its size), so one value, 0.3, works across all domains. Paper: AGC "decouples the clipping threshold from the loss scales."

**In my own words: Adam vs. LaProp**

Adam, per weight (starting from $m = 0$, $v = 0$):

$$
m \leftarrow \beta_1 m + (1-\beta_1)\, g, \qquad
v \leftarrow \beta_2 v + (1-\beta_2)\, g^2, \qquad
\text{step} = \frac{\hat m}{\sqrt{\hat v} + \epsilon}, \qquad
p \leftarrow p - \text{lr}_t \cdot \text{step}
$$

- **$\beta_1$, $\beta_2$** are fixed hyperparameters (0.9, 0.999), not learned. $m$ and $v$ are optimizer state, one value per weight, initialized to zeros.
- **$m$** averages $g$, so signs cancel: it tracks the consistent **direction**. **$v$** averages $g^2$, so signs never cancel: it tracks the typical **size**. Example: gradients $+2, -2, +2, -2$ give $m \approx 0$ but $v = 4$.
- **Dividing by $\sqrt{v}$** normalizes: a weight with gradients around 500 and one with gradients around 0.002 both take steps of about $\pm 1$ (times lr).
- **"Step"** is the number multiplied by lr and subtracted from the weight. With lr $= 4 \times 10^{-5}$ and step 1, a weight moves by 0.00004.
- **The zero start.** $m$ and $v$ start at 0, as if there were fake 0 gradients before training. First update with $g = 2$: $m = 0.2$, $v = 0.004$, step $= 0.2/\sqrt{0.004} \approx 3.2$ instead of 1.
- **Bias correction $\hat m = m / (1 - \beta_1^t)$, $\hat v = v / (1 - \beta_2^t)$**, where $t$ is the update number and $\beta^t$ is $\beta$ to the power $t$. At $t = 1$: $\hat m = 0.2 / 0.1 = 2$, $\hat v = 0.004 / 0.001 = 4$, step $= 1$. At $t = 2$ with $g = 2$ then $4$: $\hat m = 0.58 / 0.19 = 3.05$, close to the sensible average of 3. The correction fades as $t$ grows ($1 - 0.9^{50} \approx 1$), but for $v$ it lasts much longer ($1 - 0.999^{1000} = 0.63$).
- **LR warm-up** (a second guard). $\text{lr}_t = 4 \times 10^{-5} \cdot \min(1, (t-1)/1000)$: 0 at update 1, $2 \times 10^{-5}$ at update 501, full from update 1001. Any inflated early steps get multiplied by a tiny lr.

LaProp reorders the same pieces:

| | Adam | LaProp |
|---|---|---|
| 1 | $m \leftarrow \beta_1 m + (1-\beta_1)\, g$ | $v \leftarrow \beta_2 v + (1-\beta_2)\, g^2$ |
| 2 | $v \leftarrow \beta_2 v + (1-\beta_2)\, g^2$ | $u = g / (\sqrt{\hat v} + \epsilon)$ |
| 3 | $\text{step} = \hat m / (\sqrt{\hat v} + \epsilon)$ | $m \leftarrow \beta_1 m + (1-\beta_1)\, u$ |
| 4 | $p \leftarrow p - \text{lr}_t \cdot \text{step}$ | $p \leftarrow p - \text{lr}_t \cdot \hat m$ |

- **$u$** is the gradient divided by its own typical size, so it's about $\pm 1$: $g = 2$ with $\sqrt{v} = 2$, and $g = 0.002$ with $\sqrt{v} = 0.002$, both give $u = 1$.
- **The difference is order.** Adam averages raw gradients, then divides by today's $\sqrt{\hat v}$. LaProp divides each gradient by the $\sqrt{\hat v}$ at the time it arrived, then averages. Everything entering LaProp's momentum is already about $\pm 1$.
- **How much it matters.** In ordinary training the two give nearly the same steps. The paper's reason is empirical: LaProp "allows for a smaller epsilon and avoids occasional instabilities that we observed under Adam." Treat it as the paper's stability choice, not a large effect.

### Training loop and replay ratio

Paper: the replay ratio is "the fraction of time steps trained on per time step collected from the environment, without action repeat."

$$
\text{replay ratio} = \frac{B \cdot T}{\texttt{train\_every}} = \frac{16 \times 64}{2} = 512
$$

| Setting (walker-walk) | Value |
|---|---|
| envs | 16 |
| action repeat | 2 |
| `total_steps` | 500,000 agent steps (summed over envs) = 1M env frames |
| `warmup_steps` | 5,120 random-action transitions (320 per env), no training |
| `train_every` | 2 transitions per gradient step |
| $B \times T$ | $16 \times 64$ |
| replay ratio | 512 (the config comment cites the paper; the official repo's `dmc_proprio` preset uses `train_ratio: 1024`) |

**What the loop actually does, plainly**

```
repeat:
    step all 16 envs once        → 16 new transitions into the buffer
    do 8 train steps             → 8 gradient updates
```

- After every step of the 16 parallel envs, do **8 train steps**. Always 8, once warm-up is over (the first 5,120 transitions, i.e. 320 parallel steps, are random actions with 0 train steps).
- Each train step samples 16 sequences of 64 transitions ($= 1024$ transitions) and does one gradient update of the world model, critic and actor.
- Why 8: `train_every: 2` means "one train step per 2 new transitions," and $16 / 2 = 8$.

**How the code counts to 8** (`train.py`). `updates_due` is a budget: each new transition adds 1, and each train step spends `train_every` = 2. `train_every` itself never changes.

| | `train_every` | `updates_due` |
|---|---|---|
| after the 16 envs step | 2 | 16 |
| after train step 1 | 2 | 14 |
| after train step 2 | 2 | 12 |
| … | 2 | … |
| after train step 8 | 2 | 0, loop stops |

**The replay ratio, from that loop**

```
repeat:
    step all 16 envs once        → 16 new transitions
    do 8 train steps             → each uses 1024 transitions → 8 × 1024 = 8192 transitions used
```

$$\text{replay ratio} = \frac{\text{transitions used for training}}{\text{new transitions}} = \frac{8192}{16} = 512$$

For every new transition collected, 512 transitions get fed through training, so over the run each transition is used about 512 times on average. It changes if either the number of train steps per loop or the batch size changes. Derivation of the config formula:

1. Each train step samples one batch of $B \cdot T = 1024$ transitions, so transitions used $=$ train steps $\times 1024$.
2. Replay ratio $= \dfrac{\text{train steps} \times 1024}{\text{new transitions}}$.
3. Regroup: $= \dfrac{\text{train steps}}{\text{new transitions}} \times 1024 = \dfrac{8}{16} \times 1024 = 512$. The first factor, $1/2$, is exactly what `train_every: 2` sets.

**In my own words**

- **What the replay ratio means.** How many transitions go through the networks per transition collected. Each pass collects 16 transitions and runs 8 gradient steps on batches of $16 \times 64 = 1024$ transitions, so $8 \times 1024 / 16 = 512$. TD3 with batch 256 and one gradient step per env step would be $256 / 1 = 256$.
- **What actually drives the code** is gradient steps per transition collected, $1 / \texttt{train\_every} = 1/2$. The replay ratio is that times the batch size, $\tfrac{1}{2} \times 1024 = 512$. It adds no mechanism; the paper uses it to compare settings with different batch shapes (Fig. 6: higher replay ratios learn faster per env step).
- **`train_every` counts transitions, not loop passes.** "For every 2 transitions collected, do 1 gradient step." With 1 env that's one gradient step every 2 passes; with 16 envs, 8 gradient steps every pass.
- **Replay ratio = average times each transition is sampled.** Exactly: total transitions sampled ÷ total collected. Buffer size cancels out of the average: with a FIFO buffer of size $M$, each gradient step samples a given transition with probability $\approx B T / M$, and it survives $M / \texttt{train\_every}$ gradient steps, so $B T / \texttt{train\_every}$ samples in expectation. Buffer size changes the spread instead. Here capacity (5M) exceeds the run (500k), so nothing is evicted and early transitions are sampled far more than late ones, but the average is still 512.
- **Halving the batch** keeps the same number of gradient steps but halves the samples per step, so the replay ratio halves. That means each transition is reused half as often, which mostly means noisier gradients, not necessarily half the learning.
- **Warm-up.** No training until the buffer holds 5,120 transitions (320 per env), so every env stream has enough for 64-long sequences.