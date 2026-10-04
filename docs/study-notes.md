# DreamerV3 Study Notes

Oct 2, 2026 · @Adam

## How to use this doc

Every concept is written in three layers: the one-line idea, the equation, then a worked example with numbers. For interview prep, the first layer of each section is enough; go deeper when implementing.

- **Paper version:** arXiv 2301.04104 **revision 2** (17 Apr 2024). Many third-party DreamerV3 implementations follow arXiv revision 1 (Jan 2023) and will not match these notes.
- **Verify in code:** marks a place where the paper is vague or inconsistent. The last section collects all of them.
- **Running example:** a Minecraft agent at step t = 5, looking at a tree. Specific numbers such as "die 7" and its probabilities are invented for illustration; individual latent dice have no assigned meaning.
- **Scope so far:** the world model only. Critic, actor and the full training loop come next.

## Interview cheat sheet

If you can explain each row, and what breaks without it, you understand the world model at interview depth. The derivations later in this doc are for confidence and debugging, not memorization.

| Component | One-line idea | What breaks without it |
| --- | --- | --- |
| GRU sequence model | Deterministic memory h\_t that summarizes the past | No long-range memory; a plain RNN's gradients vanish or explode |
| Posterior ("encoder") | Infers z\_t from h\_t plus the real observation | The model is never grounded in what actually happened |
| Prior (dynamics predictor) | Guesses z\_t from h\_t alone | No way to dream: imagination needs z\_t without an image |
| Categorical latents | z\_t is K one-hot dice, sampled with straight-through gradients | No gradient through the sampling step |
| 1% unimix | Every categorical is 99% network output + 1% uniform | Zero probabilities, infinite log-probs, KL spikes |
| Prediction loss | Reconstruct x\_t, predict r\_t and c\_t | z\_t carries no information; the paper's ablation shows reconstruction is the main learning signal |
| Dynamics loss (weight 1) | Pulls the prior toward the posterior | The prior's guesses are useless for dreaming |
| Representation loss (weight 0.1) | Pulls the posterior toward the prior, gently | Representations hold detail the prior can never predict |
| Free bits (1 nat) | Both KL losses switch off once below 1 nat | The KL squeezes information out of z\_t (trivial-to-predict but empty latents) |
| symlog | sign(x) · ln(\|x\| + 1), applied to vector inputs and targets | Inputs at wildly different scales blow up the squared error |
| λ-return (λ = 0.95, H = 15) | Weighted average of 1- to 15-step returns, bootstrapped from v\_{t+1} | 1-step targets are biased by the critic; full rollouts are noisy and bounded by H |
| Replay critic loss (weight 0.3) | λ-return on real sequences, bootstrapped from imagination returns | The critic only ever sees dreamed rewards |
| Twohot critic and reward (255 bins) | Cross-entropy over symexp-spaced bins; value = Σ p\_i b\_i | Gradients scale with rewards (MSE) or the mean is biased (symlog MSE) |
| Critic EMA regularizer (decay 0.98) | Extra cross-entropy toward a slow copy of the critic | The critic chases its own moving targets |
| Continue weighting | Each imagined step weighted by w\_t = Π c\_k, the chance it is still alive | Steps after a predicted episode end count fully |
| Return scale S | Divide advantages by max(1, 5th–95th percentile spread of returns) | One entropy coefficient can't fit every reward scale; normalizing advantages amplifies noise |
| Reinforce actor + entropy (η = 3e-4) | Raise log-prob of actions that beat v(s\_t); no gradient through the model | The actor exploits wrong slopes of the learned model, or stops exploring |

## Big picture

**Idea.** DreamerV3 is three learners trained at the same time from replayed experience: a world model that learns to predict, a critic that judges imagined states, and an actor that picks actions.

**Parameter symbols are grouped by loss, not by network.** Each symbol covers every network trained by one objective.

| Symbol | What it covers | Trained by |
| --- | --- | --- |
| φ (phi) | All world-model networks: GRU, encoder, posterior, prior, reward, continue, decoder | One loss, Eq. 2 |
| ψ (psi) | The critic | Its own loss, Eq. 5 |
| θ (theta) | The actor | Its own loss, Eq. 6 |

**Example.** The decoder's loss backpropagates into z\_t, through the straight-through estimator into the posterior, and into h\_t, which reaches the GRU. One symbol, φ, signals that these networks are coupled by a single loss. It does not mean shared weights.

## Why a GRU

**Idea.** A plain RNN overwrites its memory every step, so gradients shrink or blow up over long sequences. A GRU adds gates that let each unit copy its old value forward, which gives gradients a near-identity path through time.

### The RNN problem

```latex
h_t = \tanh(W h_{t-1} + U x_t + b)
```

Backpropagation multiplies one Jacobian per step, with a\_t = W h\_{t−1} + U x\_t + b:

```latex
\frac{\partial h_t}{\partial h_{t-1}} = \operatorname{diag}\big(\tanh'(a_t)\big)\, W
```

Over k steps, the product of k such matrices is bounded by:

```latex
\left\| \frac{\partial h_t}{\partial h_{t-k}} \right\| \le \|W\|^k
```

If ‖W‖ < 1, the gradient shrinks exponentially (vanishes). If ‖W‖ > 1, it can grow exponentially (explodes). W is n × n, which is why the Jacobians can be chained.

**Where the diag comes from (n = 2).** tanh acts element-wise, so output i depends only on input i. Every off-diagonal entry of its Jacobian is zero:

```latex
\frac{\partial \tanh(a)}{\partial a} = \begin{bmatrix} \tanh'(a_1) & 0 \\ 0 & \tanh'(a_2) \end{bmatrix}
```

Multiplying by W scales row i of W by tanh′(a\_i), which is at most 1.

### The GRU

Two inputs per step: h\_{t−1} (size n) and x\_t (size m). Every W is (n, m), every U is (n, n), every b is (n,), so each term below is (n,).

```latex
\begin{aligned} u_t &= \sigma(W_u x_t + U_u h_{t-1} + b_u) \\ r_t &= \sigma(W_r x_t + U_r h_{t-1} + b_r) \\ \tilde{h}_t &= \tanh\big(W x_t + U (r_t \odot h_{t-1}) + b\big) \\ h_t &= (1 - u_t) \odot h_{t-1} + u_t \odot \tilde{h}_t \end{aligned}
```

1. **Update gate u\_t**, values in (0, 1): how much of each unit gets replaced.
2. **Reset gate r\_t**, values in (0, 1): how much of the old state feeds the candidate.
3. **Candidate h̃\_t**: built from x\_t and a masked old state. ⊙ is element-wise: r ⊙ h = \[r₁h₁, r₂h₂\], then U multiplies that vector.
4. **New state**: a per-unit blend. u ≈ 0 keeps the old value; u ≈ 1 takes the candidate.

**Why it fixes the problem.** Differentiating the last line gives diag(1 − u\_t) plus terms that go through W. When u\_t ≈ 0 for a unit, that unit's gradient passes through almost unchanged, with no W and no tanh′.

### DreamerV3 specifics

- **The GRU input is not the observation.** It is the concatenation \[z\_{t−1}, a\_{t−1}, h\_{t−1}\] passed through one learned linear layer. The observation only enters later, through the posterior. Verify in code: the exact embedding structure.
- **Block-diagonal recurrence:** 8 blocks, 8d recurrent units (8192 at the 200M size). Feeding h\_{t−1} into the input mixes information across blocks.
- **Truncated backprop:** training unrolls T = 64 steps, so at most 63 Jacobians are chained.
- **Why not an LSTM:** the paper does not say. An LSTM uses the same additive-path idea with 3 gates and a separate cell state; a GRU is about 25% smaller per layer, and the two are comparable empirically. Dreamer has used a GRU since PlaNet.

## The RSSM (Eq. 1)

**Idea.** Each step, the GRU updates its memory h\_t. The model then forms z\_t in one of two ways: from the real observation (posterior) or from h\_t alone (prior). Heads read \[h\_t, z\_t\] to predict reward, continue and the observation.

(Diagram omitted in this export: one RSSM step, posterior vs. prior.)

Posterior and prior both produce a z\_t from the same h\_t. The KL losses pull them together so that, when there is no observation, the prior's guess can stand in for the posterior; h\_t and z\_t then feed the next GRU step.

```latex
\begin{aligned} \text{Sequence model:} \quad & h_t = f_\phi(h_{t-1}, z_{t-1}, a_{t-1}) \\ \text{Encoder (posterior):} \quad & z_t \sim q_\phi(z_t \mid h_t, x_t) \\ \text{Dynamics predictor (prior):} \quad & \hat{z}_t \sim p_\phi(\hat{z}_t \mid h_t) \\ \text{Reward predictor:} \quad & \hat{r}_t \sim p_\phi(\hat{r}_t \mid h_t, z_t) \\ \text{Continue predictor:} \quad & \hat{c}_t \sim p_\phi(\hat{c}_t \mid h_t, z_t) \\ \text{Decoder:} \quad & \hat{x}_t \sim p_\phi(\hat{x}_t \mid h_t, z_t) \end{aligned}
```

Strictly, "RSSM" is only the first three lines. The reward, continue and decoder heads are world-model heads outside the brace; all six share φ because one loss trains them.

| Network | Reads | Outputs | Architecture |
| --- | --- | --- | --- |
| enc (first half of "encoder") | x\_t = (i\_t, q\_t), raw | Embedding e\_t | CNN for the image, MLP for the vector |
| Posterior net (second half) | \[h\_t, e\_t\] | Logits for z\_t | MLP |
| GRU | Linear(\[z\_{t−1}, a\_{t−1}, h\_{t−1}\]) and h\_{t−1} | h\_t | Block-diagonal GRU |
| Prior | h\_t | Logits for ẑ\_t | MLP |
| Reward head | \[h\_t, z\_t\] | Twohot distribution over r\_t | 1-layer MLP |
| Continue head | \[h\_t, z\_t\] | One logit for c\_t | 1-layer MLP |
| Decoder | \[h\_t, z\_t\] | x̂\_t: image and vector | CNN, 2× upsample + conv per layer (sigmoid output) + MLP |

### Walkthrough at t = 5

1. **GRU:** h\_5 = f(h\_4, z\_4, a\_4). Say a\_4 = "move forward", the action that brought the tree into view.
2. **enc:** the raw observation x\_5 = (64×64×3 tree image, \[2 logs, health 0.8, …\]) becomes e\_5. No network sits before enc; the only preprocessing is a fixed symlog on the vector part.
3. **Posterior:** \[h\_5, e\_5\] → logits → sample z\_5.
4. **Prior, in parallel:** h\_5 → ẑ\_5, its guess without seeing the tree.
5. **Heads:** \[h\_5, z\_5\] → r̂\_5, ĉ\_5, x̂\_5.
6. **Next step:** h\_6 = f(h\_5, z\_5, a\_5).

### Two modes for z\_t

- **Real data** (world-model training, acting in the environment): z\_t comes from the posterior.
- **Dreaming** (actor-critic training): there is no image, so z\_t comes from the prior. The decoder is not run at all.

### Notation traps

- The GRU's generic input x\_t is not the observation x\_t.
- The GRU literature names its update gate z; that is unrelated to the latent z\_t.
- The paper's "encoder" is enc plus the posterior net. Figure 3 draws only enc as the trapezoid; h\_t enters at the z box.
- Hats mean predictions (x̂\_t, r̂\_t, ĉ\_t); no hat means the real values (x\_t, r\_t, c\_t).
- The actor, critic and reward head read \[h\_t, z\_t\], not h\_t alone.

## The latent z\_t: categorical dice

**Idea.** z\_t is K independent dice with C faces each. The network decides how loaded each die is, every die is rolled once, and the result is K one-hot rows.

**Sizes.** C = d/16, so 64 codes at the 200M size. K = 32 at every size: not stated in the DreamerV3 paper, confirmed in the official code. So z\_t is 32 × 64 at 200M. (The older DreamerV2 algorithm used 32 × 32.)

### From logits to z\_t

```latex
\ell = \text{MLP}([h_t, e_t]) \in \mathbb{R}^{K \cdot C} \;\longrightarrow\; \text{reshape to } \mathbb{R}^{K \times C}
```

```latex
p^{(k)} = \operatorname{softmax}\big(\ell^{(k)}\big), \qquad \tilde{p}^{(k)} = 0.99\, p^{(k)} + 0.01 \cdot \tfrac{1}{C}
```

Then sample one face per row from p̃, write it as a one-hot row, and flatten to length K · C before feeding the GRU or heads. The 1% uniform mix is a fixed safety step with nothing learned: no face ever has probability exactly 0, so log-probabilities and KLs stay finite.

**Example (K = 3, C = 4).** The roll (2, 4, 1) becomes:

```latex
z_t = \begin{bmatrix} 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 1 & 0 & 0 & 0 \end{bmatrix} \;\longrightarrow\; [0,1,0,0,\;0,0,0,1,\;1,0,0,0]
```

### Straight-through gradients

**The problem.** One die has p̃ = \[0.1, 0.7, 0.1, 0.1\] and the roll lands on face 2, giving \[0, 1, 0, 0\]. Nudge 0.7 up to 0.71 and the roll still gives \[0, 1, 0, 0\]. The sample's derivative with respect to the probabilities is zero, so no learning signal reaches the logits.

**The trick.** The forward pass uses the one-hot; the backward pass pretends the output was p̃.

```latex
z^{(k)} = \text{onehot}^{(k)} + \tilde{p}^{(k)} - \operatorname{sg}\big(\tilde{p}^{(k)}\big)
```

- **Forward (values):** \[0, 1, 0, 0\] + \[0.1, 0.7, 0.1, 0.1\] − \[0.1, 0.7, 0.1, 0.1\] = \[0, 1, 0, 0\].
- **Backward (derivatives):** the one-hot is a sample, treated as a constant, so its derivative is 0. The p̃ term gives I. The sg term is declared constant, so it gives 0. Total: 0 + I − 0 = I.

I is the identity matrix: 1 for each entry with respect to itself, 0 elsewhere. It is the vector version of "the derivative is 1".

**Why that helps.** Two different derivatives are involved, and only one was broken:

- ∂L/∂z is nonzero. z is an ordinary input to the decoder and heads, so "if entry 3 were slightly bigger, would reconstruction improve?" has a real answer.
- ∂z/∂p̃ was the flat link. The trick replaces only that link with I, so the useful signal passes through:

```latex
\frac{\partial L}{\partial \tilde{p}} = \frac{\partial L}{\partial z} \cdot I = \frac{\partial L}{\partial z}
```

**In code** (DreamerV2's Algorithm 1, adapted):

```python
sample = one_hot(draw(probs))              # discrete pick: no gradient
sample = sample + probs - stop_grad(probs) # forward: one-hot; backward: gradient of probs
```

### What z\_t carries

Nothing hand-designed. It holds whatever in x\_t helps reconstruct the observation and predict reward and continue, beyond what h\_t already carries. Capacity is at most K · log₂ C bits per step, e.g. 32 · log₂ 64 = 192 bits for DreamerV3 at 200M (160 bits for the older DreamerV2's 32 × 32).

## Batching: B versus B × T

**Idea.** Only the GRU is recurrent. Anything that needs h\_t must run inside the time loop on B items per step; everything else is flattened to B × T and run in one parallel pass.

With B = 16 sequences of T = 64 steps:

| Network | When it runs | Batch per call |
| --- | --- | --- |
| enc (CNN + MLP) | Once, before the loop | B × T = 1024 |
| GRU | Inside the loop, 64 iterations | B = 16 |
| Posterior net | Inside the loop: needs h\_t, and z\_t feeds h\_{t+1} | B = 16 |
| Prior | After the loop; nothing in the loop depends on it during training | 1024 |
| Reward, continue, decoder heads | After the loop | 1024 |

The posterior net is not recurrent itself; it is a plain MLP that must run in step with the GRU: h\_t → z\_t → h\_{t+1} → z\_{t+1}.

**Each head sees one step.** r̂\_5 comes from \[h\_5, z\_5\] only, never the history z\_{1:5}. It does not need the history because h\_5 already summarizes it.

**Caveat.** The 1024 items are not independent samples: the 64 steps of one sequence come from the same trajectory and are strongly correlated.

## Prediction loss

**Idea.** Each head outputs a distribution, and the loss scores how much probability it gave the real value. Written uniformly as a negative log-likelihood, it becomes a familiar loss per head: MSE, binary cross-entropy, twohot cross-entropy.

```latex
\mathcal{L}_{\text{pred}} = -\ln p_\phi(x_t \mid z_t, h_t) - \ln p_\phi(r_t \mid z_t, h_t) - \ln p_\phi(c_t \mid z_t, h_t)
```

It is a sum of three NLLs. The ground truth is the first argument: x\_t, r\_t, c\_t carry no hats.

### How "evaluate at x\_t" works

The network never sees x\_t. It outputs the parameters of a distribution, and a fixed formula, outside the network, plugs in the real value.

```latex
(h_t, z_t) \xrightarrow{\text{NN}} \mu \;\longrightarrow\; p(x_t \mid \mu) \;\longrightarrow\; -\ln p(x_t \mid \mu)
```

For a unit-variance Gaussian, the NLL is squared error plus a constant, so in code you just write MSE:

```latex
-\ln \mathcal{N}(x_t \mid \mu, I) = \tfrac{1}{2}\lVert x_t - \mu \rVert^2 + \text{const}
```

**Why σ is fixed.** The paper does not say. A known general reason: with a learned σ², the model can lower its loss by inflating σ² on hard parts instead of fitting them.

```latex
-\ln \mathcal{N}(x \mid \mu, \sigma^2) = \frac{(x - \mu)^2}{2\sigma^2} + \tfrac{1}{2}\ln \sigma^2 + \text{const}
```

### Decoder

The target is the raw x\_t = (i\_t, q\_t), never an embedding. There is one output head per modality, and their losses are summed:

```latex
-\ln p_\phi(x_t \mid z_t, h_t) = \text{loss}_{\text{img}}(i_t, \hat{i}_t) + \lVert \operatorname{symlog}(q_t) - \hat{q}_t \rVert^2
```

- **Image head:** sigmoid output in \[0, 1\], trained with MSE against pixels/255, summed over pixels (no symlog). Default code upsamples 2× then convolves at each layer, rather than a transposed conv. The image and vector terms share one loss scale of 1.0.
- **Mean only:** the decoder outputs the mean and is never sampled. It is not run during dreaming at all; the Figure 4 video frames are its means.

**symlog.** A fixed squashing function, applied element-wise:

```latex
\operatorname{symlog}(x) = \operatorname{sign}(x)\,\ln(|x| + 1), \qquad \operatorname{symexp}(x) = \operatorname{sign}(x)\,\big(e^{|x|} - 1\big)
```

symlog(1000) ≈ 6.9 and symlog(−1000) ≈ −6.9, while values near 0 pass almost unchanged. An inventory count of 500 and a health value of 0.3 can then share one MSE without per-environment normalization.

### Continue head

One logit ℓ\_t becomes p̂\_t = σ(ℓ\_t), the probability the episode continues. The label c\_t is 1 for continue, 0 for ended.

```latex
-\ln p_\phi(c_t \mid z_t, h_t) = -\big[\,c_t \ln \hat{p}_t + (1 - c_t)\ln(1 - \hat{p}_t)\,\big]
```

Only one term is ever active. c\_t = 1 leaves −ln p̂\_t, pushing p̂\_t toward 1; c\_t = 0 leaves −ln(1 − p̂\_t), pushing it toward 0.

### Reward head

The paper's world-model paragraph says symlog squared loss, but the code settles it: the reward head is symexp twohot, 255 bins, zero-initialized output, same as the critic. See Robust predictions below.

## KL divergence basics

**Idea.** KL measures how different two distributions are: zero when they are identical, growing as they disagree. In code it is a sum over an array, a closed-form formula, or a sample average; never an actual integral.

### The formulas

General discrete form, summing over every outcome x:

```latex
\text{KL}[\,q \,\|\, p\,] = \sum_{x} q(x) \ln \frac{q(x)}{p(x)}
```

Continuous form:

```latex
\text{KL}[\,q \,\|\, p\,] = \int q(x) \ln \frac{q(x)}{p(x)}\, dx
```

Dreamer's form, for K independent dice with C faces:

```latex
\text{KL}[\,q \,\|\, p\,] = \sum_{k=1}^{K} \sum_{c=1}^{C} q^{(k)}_c \ln \frac{q^{(k)}_c}{p^{(k)}_c}
```

Without independence you would sum over all C^K joint outcomes. Because the dice are independent, a full roll's probability is a product over dice, the log turns that product into a sum, and the KL splits into one KL per die.

### Worked example: K = 2, C = 3

|  | Face 1 | Face 2 | Face 3 |
| --- | --- | --- | --- |
| Die 1, posterior q | 0.7 | 0.2 | 0.1 |
| Die 1, prior p | 0.5 | 0.3 | 0.2 |
| Die 2, posterior q | 0.1 | 0.1 | 0.8 |
| Die 2, prior p | 0.1 | 0.1 | 0.8 |

```latex
\text{KL}^{(1)} = 0.7\ln\tfrac{0.7}{0.5} + 0.2\ln\tfrac{0.2}{0.3} + 0.1\ln\tfrac{0.1}{0.2} = 0.236 - 0.081 - 0.069 = 0.085
```

Die 2's rows are identical, so every term is ln 1 = 0. Total KL = 0.085 + 0 = 0.085 nats.

### In code

```python
import numpy as np
from scipy.stats import norm

# Discrete: one categorical
q = np.array([0.7, 0.2, 0.1]); p = np.array([0.5, 0.3, 0.2])
kl = np.sum(q * np.log(q / p))                       # 0.085 nats

# Continuous, closed form: two 1D Gaussians
mu_q, sig_q, mu_p, sig_p = 0.0, 1.0, 1.0, 2.0
kl = np.log(sig_p / sig_q) + (sig_q**2 + (mu_q - mu_p)**2) / (2 * sig_p**2) - 0.5   # 0.443

# Continuous, Monte Carlo: sample from q, average the log ratio
x = np.random.default_rng(0).normal(mu_q, sig_q, size=100_000)
kl_mc = np.mean(norm.logpdf(x, mu_q, sig_q) - norm.logpdf(x, mu_p, sig_p))   # ≈ 0.443
```

In PyTorch, `torch.distributions.kl_divergence(q, p)` uses a closed form when one is registered. DreamerV3 only needs the discrete case.

### Reducing over the batch

1. Compute one KL per (b, t) step: an array of shape (B, T), each entry summed over all K dice.
2. Apply free bits, max(1, KL), to each entry **before** reducing. Averaging first would let steps under 1 cancel steps over 1, and the clip would rarely act.
3. Reduce. Reshaping to B · T and taking one mean is the same as mean over t then mean over b, because every sequence has exactly T steps.

Eq. 2 sums over t while implementations usually average; the two differ by the constant factor T. Match the official code so the paper's loss weights carry over.

## Dynamics and representation losses

**Idea.** Both are the same KL between posterior and prior, split into two directions by stop-gradients. The dynamics loss (weight 1) moves the prior toward the posterior; the representation loss (weight 0.1) moves the posterior gently toward the prior.

```latex
\mathcal{L}_{\text{dyn}} = \max\!\big(1,\; \text{KL}\big[\,\operatorname{sg}(q_\phi(z_t \mid h_t, x_t)) \,\|\, p_\phi(z_t \mid h_t)\,\big]\big)
```

```latex
\mathcal{L}_{\text{rep}} = \max\!\big(1,\; \text{KL}\big[\,q_\phi(z_t \mid h_t, x_t) \,\|\, \operatorname{sg}(p_\phi(z_t \mid h_t))\,\big]\big)
```

```latex
\mathcal{L}(\phi) = \mathbb{E}_{q_\phi}\Big[\textstyle\sum_{t=1}^{T} \big(\mathcal{L}_{\text{pred}} + \mathcal{L}_{\text{dyn}} + 0.1\,\mathcal{L}_{\text{rep}}\big)\Big]
```

| Loss | Stop-gradient on | Gradient reaches | Pull | Weight |
| --- | --- | --- | --- | --- |
| Dynamics | Posterior q | Prior net, and the GRU through h\_t | Prior toward posterior | 1 |
| Representation | Prior p | Posterior net, enc, and the GRU through h\_t | Posterior toward prior | 0.1 |

### Why the stop-gradient on the posterior

A plain KL between two learned distributions can be lowered from either side. Without the stop-gradient, the cheap way is to make z\_t carry less information from x\_t so it is easier to predict; in the extreme, z\_t ignores the image. The stop-gradient forces the prior to do the work.

**The posterior acts as soft labels.** With q stopped, its entropy is a constant, so from the prior's side the loss is exactly classification cross-entropy:

```latex
\text{KL}[\,\operatorname{sg}(q) \,\|\, p\,] = \underbrace{\textstyle\sum_c q_c \ln \tfrac{1}{p_c}}_{\text{cross-entropy}} - \underbrace{\textstyle\sum_c q_c \ln \tfrac{1}{q_c}}_{\text{entropy of } q,\ \text{constant}}
```

The labels move, though: the posterior is itself trained by the prediction and representation losses, so the prior chases a target that shifts every optimizer step.

### Why the representation loss exists

Left alone, the posterior encodes every detail of x\_t that helps reconstruction, including detail the prior can never anticipate from h\_t, such as noise or unpredictable textures. That leaves a gap the prior cannot close, and dreaming relies entirely on the prior. The paper's stated purpose: more predictable representations, which allow a factorized prior (independent dice) for fast sampling in imagination. The weight is small because this pull fights the prediction loss.

### Free bits

max(1, KL) outputs the constant 1 whenever KL < 1 nat (≈ 1.44 bits), so its derivative is 0 there; above 1 the gradient passes normally.

| Step's KL | max(1, KL) | Gradient |
| --- | --- | --- |
| 0.55 | 1 | 0 |
| 0.60 | 1 | 0 |
| 2.30 | 2.30 | Normal |

- **Purpose:** once prior and posterior agree within 1 nat, stop pushing them together and leave the gradient budget to the prediction loss. This blocks the degenerate solution of an easy-to-predict but empty z\_t.
- **Granularity:** the clip applies to each step's KL summed over all K dice, not to one die.
- **History:** DreamerV2 did this with "KL balancing", one mixing coefficient α = 0.8. V3 uses separate loss weights plus free bits.

### There is no update order

All three terms sum into one scalar loss. One forward pass, one backward pass and one optimizer step update every world-model parameter at once; the stop-gradients only decide which parameters each term reaches. A stop-gradient never removes a term from the loss value you log, only from the gradient.

## Gradient derivations (optional depth)

**Idea.** The dynamics loss gives the prior's logits the gradient p − q: "prior minus posterior". The representation loss gives the posterior's logits q\_k(ln(q\_k / p\_k) − KL). Autodiff computes both; this section is for understanding and debugging.

**Notation.** ℓ^p are the prior's logits, ℓ^q the posterior's. Worked numbers use one invented die with 3 faces: posterior q = \[0.05, 0.9, 0.05\], prior p = \[0.3, 0.4, 0.3\].

```latex
\text{KL} = 0.05\ln\tfrac{0.05}{0.3} + 0.9\ln\tfrac{0.9}{0.4} + 0.05\ln\tfrac{0.05}{0.3} = -0.090 + 0.730 - 0.090 = 0.551 \text{ nats}
```

### Prior side: the dynamics loss

1. Start from the KL:

```latex
\text{KL}[\,q \,\|\, p\,] = \sum_c q_c \ln \frac{q_c}{p_c}
```

2. Split the log of the ratio, using ln(a/b) = ln a − ln b:

```latex
\text{KL} = \sum_c q_c \ln q_c - \sum_c q_c \ln p_c
```

3. q is stop-gradiented, so only the second sum produces gradients. (Its value still counts in the loss: 0.551 = −0.394 + 0.945.)

```latex
\mathcal{L}_{\text{eff}} = -\sum_c q_c \ln p_c
```

4. Write p\_c as a softmax of the prior's logits:

```latex
p_c = \frac{e^{\ell^p_c}}{\sum_j e^{\ell^p_j}}
```

5. Take its log, using ln(a/b) = ln a − ln b and ln e^x = x. The log of a sum does not split, so the second term stays whole:

```latex
\ln p_c = \ell^p_c - \ln \sum_j e^{\ell^p_j}
```

6. Substitute into step 3 and distribute the sum:

```latex
\mathcal{L}_{\text{eff}} = -\sum_c q_c \ell^p_c + \Big(\sum_c q_c\Big) \ln \sum_j e^{\ell^p_j}
```

7. q is a probability distribution, so its entries sum to 1:

```latex
\mathcal{L}_{\text{eff}} = -\sum_c q_c \ell^p_c + \ln \sum_j e^{\ell^p_j}
```

8. Differentiate each term with respect to ℓ^p\_2. Only c = 2 in the first sum, and only j = 2 inside the log, contain ℓ^p\_2. The second uses the chain rule d/dx ln u = (1/u)(du/dx):

```latex
\frac{\partial}{\partial \ell^p_2}\Big(-\sum_c q_c \ell^p_c\Big) = -q_2 \qquad \frac{\partial}{\partial \ell^p_2}\ln \sum_j e^{\ell^p_j} = \frac{e^{\ell^p_2}}{\sum_j e^{\ell^p_j}} = p_2
```

9. Add the two pieces:

```latex
\frac{\partial \mathcal{L}}{\partial \ell^p_2} = p_2 - q_2 = 0.4 - 0.9 = -0.5
```

For all three faces the gradient is p − q = \[0.25, −0.5, 0.25\]. Gradient descent subtracts it, so the face-2 logit rises and the prior moves toward the posterior's \[0.05, 0.9, 0.05\].

### Posterior side: the representation loss

Now p is the constant and the posterior logits ℓ^q move. It is harder because q appears in both terms of the KL.

1. **Multivariable chain rule.** If f depends on u and v, and both depend on x, add one term per path: df/dx = (∂f/∂u)(du/dx) + (∂f/∂v)(dv/dx). Check with f = u · v, u = x², v = 3x: directly f = 3x³ and df/dx = 9x²; by the rule, 3x · 2x + x² · 3 = 9x².
2. **Apply it to the KL**, which has six inputs, q\_1, q\_2, q\_3 and p\_1, p\_2, p\_3. Nudging ℓ^q\_2 moves all three q's, because they share the softmax denominator and must still sum to 1:

```latex
\frac{\partial \text{KL}}{\partial \ell^q_2} = \sum_{c=1}^{3} \frac{\partial \text{KL}}{\partial q_c}\frac{\partial q_c}{\partial \ell^q_2} + \sum_{c=1}^{3} \frac{\partial \text{KL}}{\partial p_c}\frac{\partial p_c}{\partial \ell^q_2}
```

The p sum is zero for two separate reasons: p is computed from ℓ^p, not ℓ^q, and the stop-gradient blocks it anyway.

3. **First factor.** Only one KL term contains q\_c. By the product rule, d/dq (q ln q) = ln q + 1, and q ln p\_c differentiates to the constant ln p\_c:

```latex
\frac{\partial \text{KL}}{\partial q_c} = \ln q_c + 1 - \ln p_c = \ln \frac{q_c}{p_c} + 1
```

4. **Second factor.** Write q\_c = e^{ℓ\_c} · S^{−1} with S = Σ\_j e^{ℓ\_j}, so ∂S/∂ℓ\_2 = e^{ℓ\_2}, and use the product rule:

```latex
\frac{\partial q_2}{\partial \ell^q_2} = q_2 - q_2^2 = q_2(1 - q_2) \qquad \frac{\partial q_1}{\partial \ell^q_2} = -\frac{e^{\ell_1} e^{\ell_2}}{S^2} = -q_1 q_2
```

5. **The three paths with numbers:**

| Path | ∂KL/∂q\_c | ∂q\_c/∂ℓ^q\_2 | Product |
| --- | --- | --- | --- |
| via q\_1 | −0.792 | −0.045 | 0.036 |
| via q\_2 | 1.811 | 0.090 | 0.163 |
| via q\_3 | −0.792 | −0.045 | 0.036 |
| **Total** |  |  | **0.234** |

6. **Simplify symbolically.** Write A\_c = ln(q\_c / p\_c) + 1 and factor out q\_2:

```latex
q_2\big[(1 - q_2)A_2 - q_1 A_1 - q_3 A_3\big] = q_2\Big[A_2 - \sum_c q_c A_c\Big]
```

Expanding (1 − q\_2)A\_2 produces −q\_2 A\_2, the missing piece that completes the sum over all faces. That sum is the KL plus Σ q\_c = 1:

```latex
\sum_c q_c A_c = \text{KL} + 1 \quad\Longrightarrow\quad \frac{\partial \text{KL}}{\partial \ell^q_2} = q_2\Big(\ln \frac{q_2}{p_2} - \text{KL}\Big)
```

7. **Read it.** Face 2: 0.9 × (0.811 − 0.551) = 0.234. Faces 1 and 3: 0.05 × (−1.792 − 0.551) = −0.117 each. The three sum to 0, as softmax gradients always do. Descent lowers the face-2 logit, so the posterior becomes less confident and moves toward \[0.3, 0.4, 0.3\].

### What these gradients are for

Logits are network outputs, not parameters, so they are never updated directly. The gradient keeps flowing back to the weights: posterior MLP, then enc through e\_t, then the GRU through h\_t. Before anything moves, it is multiplied by the loss weight (0.234 × 0.1 = 0.0234 for the representation loss), zeroed if the step's summed KL is under 1 nat, added to the other losses' gradients, reduced over B × T, and applied by AGC clipping then LaProp.

## Data collection and replay conventions

**Idea.** The agent acts in the real environment from its posterior state, writes every step to replay, and the world model trains on sequences sampled from replay. Neither DreamerV2 nor DreamerV3 has a pseudocode training loop; the closest is DreamerV1's Algorithm 1 (arXiv 1912.01603), whose skeleton carries over but whose details do not.

### Step 1: collect data

```latex
x_t \to z_t \sim q_\phi(z_t \mid h_t, x_t) \;\to\; a_t \sim \pi_\theta(a_t \mid h_t, z_t) \;\to\; \text{env} \;\to\; (x_{t+1}, r_{t+1}, c_{t+1})
```

The actor samples the action; the critic never picks actions. By default 16 environments run in parallel.

### Reward indexing

| Algorithm family | What is stored | The reward from taking a\_t in x\_t is labeled |
| --- | --- | --- |
| TD3, SAC, PPO | Independent transitions (s, a, r, s′) | Together with (s\_t, a\_t) |
| Dreamer | Sequences of per-step records | r\_{t+1}, stored with x\_{t+1} |

**Why Dreamer must use the second convention.** The reward head reads (h\_t, z\_t). h\_t contains actions only up to a\_{t−1}, and z\_t comes from x\_t, so the head cannot know a\_t. r\_t must be the reward that arrived together with x\_t, caused by a\_{t−1}. Likewise, c\_t flags whether x\_t is terminal.

**In code:** this doc uses the TD3 convention from the critic section on: r\_t and c\_t are stored with the action a\_t that produced them, and the bootstrap is v\_{t+1}. The official code stores them with the next observation instead; either works if the λ-return indexing matches.

### Step 0 of an episode

No action led to the first observation, so there is no real r\_0 or c\_0. The usual fill is r\_0 = 0 and c\_0 = 1, plus an is\_first flag. The flag resets the RSSM: h returns to its initial value and the previous action is set to zeros. This matters because a 64-step replay sequence can span the end of one episode and the start of the next. The paper is silent here; verify in code.

### Replay buffer

- **Capacity** 5 × 10⁶ steps, sampled uniformly.
- **Online queue:** each minibatch is first filled with fresh, non-overlapping trajectories, then topped up with uniform samples.
- **Latents in replay:** latent states are stored during collection to initialize the world model on replayed sequences, and fresh latents from training are written back.
- **Replay ratio:** time steps trained on per environment step collected, not counting action repeat. Divide it by the steps per minibatch and by the action repeat to get gradient steps per env step.

**Example (Atari):** replay ratio 32, minibatch 16 × 64 = 1024 steps, action repeat 4. Then 32 / 1024 / 4 = 1/128, one gradient step every 128 env steps, about 1.5M gradient steps over 200M env steps.

Steps 2 onward of the training loop (sample a batch, world-model update, imagination, actor-critic update) are covered in the sections below.

## Training loop and replay ratio

**Idea.** One cycle = collect N agent steps, then one gradient step that runs world model → imagine → critic → actor. Only real steps go into replay; dreamed steps are never stored.

```latex
N = \frac{B \cdot T}{\text{replay ratio}}
```

Replay ratio = time steps trained on per agent step collected. Action repeat is bookkeeping only: it multiplies agent steps into env frames.

**Example (Atari).** B · T = 16 × 64 = 1024, replay ratio 32, so N = 32 agent steps per gradient step. Across E = 16 envs that is 2 steps per env per cycle. With action repeat 4, that is 128 env frames per gradient step.

| Benchmark | Replay ratio | Envs E | Action repeat |
| --- | --- | --- | --- |
| Atari, DMLab | 32 | 16 | 4 |
| Minecraft | 32 | 64 | 1 |
| ProcGen | 64 | 16 | 1 |
| Atari100k | 128 | 1 | 4 |
| Control | 512 | 16 | 2 |
| BSuite | 1024 | 1 | 1 |

**Batch shapes inside one gradient step.**

| Stage | Sequence axis | Batch |
| --- | --- | --- |
| World model GRU | T = 64 (real) | B = 16 |
| Heads, KL losses | none | B · T = 1024 |
| Imagination | H = 15 (dreamed) | B · T = 1024 starts |
| Replay critic target | T = 64 (real, backward) | B = 16 |
| Actor, imagination critic | none | 1024 × 15 = 15,360 states |

**Online queue.** Each env is its own continuous stream. When an env completes a fresh 64-step chunk, that chunk goes into the next batch first; uniform samples fill the rest. Your simpler design: a FIFO of fresh chunks that pops at most a fixed number per batch.

**Imagination is cheap at 15,360 states.** Each state is a vector \[h, z\] (8192 + 32 · 64 = 10,240 numbers at 200M), no images, and Reinforce means no backprop through the 15-step rollout.

## Critic learning

**Idea.** The critic predicts the λ-return of each state as a distribution over 255 bins. It trains on imagined rollouts (weight 1.0) and on real replay sequences (weight 0.3).

**Index convention (TD3).** r\_t and c\_t result from taking a\_t in s\_t, and the bootstrap is v\_{t+1}. The paper's Eq. 5 writes v\_t because Dreamer stores each reward with the state it arrives in; the math is the same.

### λ-return

```latex
R^\lambda_t = r_t + \gamma c_t \big( (1-\lambda)\, v_{t+1} + \lambda R^\lambda_{t+1} \big), \qquad R^\lambda_H = v_H
```

It is a weighted average of k-step returns: weight (1−λ)λ^{k−1} on the k-step return for k < H, and λ^{H−1} ≈ 0.49 on the full 15-step return. R^λ\_t uses v\_{t+1} … v\_H, never v\_t.

### Replay target

Same recursion along the real 64-step sequence, with real rewards. Where imagination would bootstrap from v\_{t+1}, replay bootstraps from R^img\_{t+1}, the imagination return of the rollout that started at that real state.

```latex
R^{\text{rep}}_t = r_t + \gamma c_t \big( (1-\lambda)\, R^{\text{img}}_{t+1} + \lambda R^{\text{rep}}_{t+1} \big)
```

Order: imagine first, compute R^img, then run the replay recursion. Replay critic gradients flow into the world model.

| Loss | Sequence | Batch | Target | Weight |
| --- | --- | --- | --- | --- |
| Imagination | H = 15 dreamed | B · T = 1024 | R^img (λ = 0.95) | 1.0, continue-weighted |
| Replay | T = 64 real | B = 16 | R^rep | 0.3, skips is\_last |

### Twohot pipeline (no symlog anywhere)

1. **Setup, once:** b\_i = symexp(linspace(−20, 20, 255)). Real-unit numbers, dense near 0, sparse far out: …, 0, 0.17, …, 95.25, 111.66, …
2. **Output:** 255 logits → softmax p\_i.
3. **Value:** v\_t = Σ p\_i b\_i, already in real units. Used in the advantage and bootstraps; never fed into the loss.
4. **Label:** target y = 100 sits between 95.25 and 111.66 → weight 0.71 on 95.25 and 0.29 on 111.66 (0.71 · 95.25 + 0.29 · 111.66 = 100).
5. **Loss:** cross-entropy between that label and p. Gradient w.r.t. logits = p − twohot(y), bounded in \[−1, 1\] at any reward scale.

Output layer is zero-initialized, so v starts at exactly 0.

### EMA regularizer and continue weighting

- **EMA regularizer:** a second cross-entropy toward twohot(v\_slow), weight 1. v\_slow is an EMA copy of the critic with decay 0.98.
- **Continue weight:** probability the imagined episode is still alive at s\_t (stop-gradient).

```latex
w_0 = 1, \qquad w_{t+1} = w_t \cdot c_t
```

Example: c = (1.0, 0.95, 0.2) → w = (1, 1.0, 0.95, 0.19).

```latex
\mathcal{L}_{\text{critic}} = 1.0 \sum_t \operatorname{sg}(w_t)\Big(\text{CE}\big(\text{twohot}(R^{\text{img}}_t), p_t\big) + \text{CE}\big(\text{twohot}(v^{\text{slow}}_t), p_t\big)\Big) + 0.3\, \mathcal{L}_{\text{rep}}
```

**In the official code** (e3f0224): γ is folded into the continue target as (1 − terminal) · (1 − 1/333), so γ = 1 inside the imagination return; replay uses γ = 0.997 explicitly. Effective discount is the same.

## Actor learning

**Idea.** Reinforce on imagined rollouts, for both discrete and continuous actions: raise the log-probability of sampled actions that beat the critic's expectation, lower it for those that didn't. No gradient flows through the world model.

```latex
\mathcal{L}(\theta) = -\sum_t \operatorname{sg}(w_t)\Big( \operatorname{sg}\!\Big(\frac{R^\lambda_t - v(s_t)}{\max(1,S)}\Big) \ln \pi_\theta(a_t \mid s_t) + \eta\, \mathrm{H}\big[\pi_\theta(\cdot \mid s_t)\big] \Big)
```

The paper's Eq. 6 omits w\_t; the official code multiplies the whole per-step term by the same continue weight as the critic. η = 3 × 10⁻⁴.

### Pieces, one step at a time

1. **Advantage weight.** Â\_t = sg((R^λ\_t − v(s\_t)) / max(1, S)) is a plain number. v(s\_t) is the critic's guess before seeing a\_t; R^λ\_t is built from the reward a\_t produced plus v\_{t+1} … v\_H. Example: v = 10, R^λ = 13, S = 19.01 → Â ≈ 0.16.
2. **Log-probability** of the sampled action. Discrete example: π = (0.5, 0.3, 0.2), sampled the 0.3 action → ln 0.3 ≈ −1.20.
3. **Per-step loss** = −Â · ln π = −0.16 · (−1.20) = 0.192. Minimizing it raises π of that action. The other actions move only through the softmax; across the batch each gets its own samples and advantages.
4. **Entropy bonus.** Uniform over 3 actions: H = ln 3 ≈ 1.10; peaked (0.98, 0.01, 0.01): H ≈ 0.11. ηH ≈ 0.0003, about 500× smaller than a typical Â, so it matters mainly when advantages are near zero (no reward found yet). DreamerV3 adds no action noise; this is its only exploration pressure.

### Return scale S (Eq. 7)

```latex
S = \operatorname{EMA}\big(\operatorname{Per}(R^\lambda, 95) - \operatorname{Per}(R^\lambda, 5),\ 0.99\big)
```

Example: 5th percentile −2, 95th percentile 18 → spread 20; with previous S = 19, S ← 0.99 · 19 + 0.01 · 20 = 19.01. A spread of 0.3 divides by max(1, 0.3) = 1. Scale large returns down, never small ones up.

**Why returns, not advantages (PPO style).** Sparse rewards, advantages ≈ ±0.001 of critic noise. PPO divides by std ≈ 0.001 → noise becomes ±1, about 3000× the entropy term. DreamerV3: S ≈ 0.01 → divide by 1 → advantages stay ±0.001, comparable to ηH, so exploration survives. Percentiles ignore a single outlier return.

### Discrete gradient

```latex
\frac{\partial \ln \pi(a_t)}{\partial l} = \text{onehot}(a_t) - \pi
```

Example with Â = 0.16: ∂L/∂l = −0.16 · ((0, 1, 0) − (0.5, 0.3, 0.2)) = (0.08, −0.112, 0.032). Descent raises the sampled logit.

### Continuous gradient

Official `bounded_normal`: mean = tanh(network output), σ squashed into \[0.1, 1.0\]; the sampled action is not squashed, so log\_prob needs no tanh Jacobian (unlike SAC). In PyTorch: Normal(μ, σ).log\_prob(a), summed over action dims.

Example: μ = 0, σ = 0.5, a = 0.3, Â = 0.16.

```latex
\ln \pi(a) = -\frac{(a-\mu)^2}{2\sigma^2} - \ln\sigma - \tfrac12 \ln 2\pi \approx -0.41
```

```latex
\frac{\partial \ln \pi}{\partial \mu} = \frac{a-\mu}{\sigma^2} = 1.2, \qquad \frac{\partial \ln \pi}{\partial \sigma} = \frac{(a-\mu)^2}{\sigma^3} - \frac{1}{\sigma} = -1.28
```

∂L/∂μ = −0.192 → μ moves toward 0.3. ∂L/∂σ = +0.205 → σ shrinks, because the good action fell within one σ of the mean.

**Why Reinforce for continuous too.** DreamerV1 backpropagated action gradients through the learned dynamics (like TD3's ∇\_a Q): low variance, but it trusts the model's slopes. V3's Reinforce needs only outcome values; its higher variance is held down by the critic baseline, return scaling, and 15,360 samples per update.

**One-line interview answer:** V3 scales returns down when large but never up when small, so one fixed entropy coefficient works across dense and sparse rewards without amplifying noise.

## Robust predictions

**Idea.** One set of hyperparameters must work whether rewards are ±1 or ±100,000. Two different tools handle this, and only one of them applies symlog to data.

| Quantity | Applied to the data | Where scale robustness comes from |
| --- | --- | --- |
| Vector observations | symlog(x) as encoder input and decoder target | the squashed values themselves (100,000 → 11.5) |
| Images | x/255 − 0.5 in, sigmoid out, MSE vs x/255 | pixels are already bounded |
| Reward, critic | nothing: raw target y | cross-entropy gradient p − twohot(y), plus symexp bin spacing for range |

### The bins

Evenly spaced u, then symexp. The bins themselves get sparser as they grow, so 255 bins reach about ±4.85 × 10⁸ while staying fine near 0.

| u\_i (gap 0.157) | b\_i = symexp(u\_i) |
| --- | --- |
| 0 | 0 |
| 0.157 | 0.17 |
| 4.567 | 95.25 |
| 4.724 | 111.66 |

### Why not symlog MSE for the critic

Glider example: from s\_t the return is 0 or 100, 50/50, so the true value is 50.

- **Plain MSE critic (TD3 style):** learns 50, but its gradient 2(output − y) scales with the rewards: −200 at R = 100 vs −200,000 at R = 100,000.
- **Symlog MSE (the rejected option):** network output o\_t trained toward symlog(y), which is 0 or 4.62. It settles at 2.31, so v\_t = symexp(2.31) ≈ 9. Averaging in log space biases the value low.
- **Twohot (DreamerV3):** learns p ≈ 0.5 on the bin at 0, 0.355 on 95.25, 0.145 on 111.66 → v\_t ≈ 50. Correct, with gradients bounded in \[−1, 1\].

The raw y is used only once, to build the label. After that the loss compares two probability vectors and never sees the reward's magnitude.

**One-line interview answer:** twohot gives an unbiased expected value under noisy targets, with gradients whose size doesn't depend on reward magnitude.

## Verify in the official code

The paper leaves these open or contradicts itself. Resolve each against `danijar/dreamerv3`, pinned to commit e3f0224, before relying on it.

- [x] Number of latents K = 32 (code `stoch: 32`), C = 64 classes at 200M
- [ ] GRU input embedding: one linear on the concatenation, or separate linears per input, and any normalization
- [x] Imagination horizon H = 15 (code `imag_length: 15`)
- [x] Reward head loss: symexp twohot, 255 bins, zero-init output
- [x] Image term: MSE on pixels/255 with sigmoid output, summed over pixels; loss scales rec = rew = con = 1.0
- [ ] Loss reduction over time: sum (Eq. 2) or mean
- [ ] Episode starts: r\_0, c\_0 placeholders and is\_first state reset
- [x] Twohot expectation: summed outward from the middle bin so uniform probabilities give exactly 0
- [x] Repository commit pinned: e3f0224
- [x] 12M model recurrent units: code uses 2048 = 8d, so the paper's 1024 is a typo
