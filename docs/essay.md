# From flying model-free policies to learning world models

## 1. Model-free RL in the loop 


- Professionally, I have focused on model-free reinforcement learning for autonomous aerospace applications, working from simulation to real flight. This has involved working extensively with adding enhancements to baseline algorithms, along with reward design, and simulation tweaking for sim2real.
- A few thoughts on sim2real in DeepRL. A very tight loop from in person, real-world testing and data collection to DeepRL sim and agent tweaks is essential. Even if an agent performs well in your sim, there is no guarantee that it will perform well in the real world. Only constant hopping from one to the other leads to successful outcomes.
- Over the past two months I reproduced DreamerV3 (Hafner et al., 2023) from scratch in PyTorch: a world-model agent that learns its policy entirely inside its own imagined rollouts. It reached [930–960] on DeepMind Control walker-walk across three seeds. Building it made one question concrete for me: what is the difference between a representation that predicts and a model you can act inside? 
  

## 2. What an auxiliary prediction loss actually learns 


- A standard way to give a model-free agent some foresight is to add auxiliary losses. UNREAL (Jaderberg et al., 2016) is the classic example: an A3C agent whose shared CNN and LSTM are also trained on pixel control, reward prediction and value replay. Each task gets its own small head on top of the shared network, and none of those heads is ever used to act. They exist only to shape the recurrent state $h_t$, the vector that summarises everything the agent has seen.
- Pixel control is a Q-learning task, so it needs a maximum over actions, which works for discrete actions and breaks for continuous ones. In continuous control the natural version is prediction: from $h_t$ and the action taken, predict what comes next, whether that is the next observation, a latent of it, or the reward.
- A policy without auxiliary tasks and losses only needs $h_t$ to carry what is sufficient for choosing the current action. A prediction loss asks for more: $h_t$ must also carry what is predictive of the future. Information about how the dynamics will unfold, which the policy might never have discovered from the reward signal alone, is pushed into the representation the policy reads.
-The prediction only matters during training. It is computed, scored, and thrown away. The policy never acts on it and never imagines with it. An auxiliary prediction loss is a representation-learning trick, not a model.


## 3. Where that falls short of a world model 


- An auxiliary head and a world model are closer than they look. Take a recurrent agent that predicts its next observation from $h_t$ and the action it took. Feed that prediction back into the encoder in place of the real observation, predict again, and repeat: you are rolling the agent's own predictions forward in time. Add a head that predicts reward and one that predicts whether the episode ends, and you have built a world model. The gap between the two is not architecture. It is two missing pieces, and a decision to use them.
- The first missing piece is the loop. An auxiliary head's prediction is never fed back in; the next hidden state is always built from the real next observation. So the head is only ever trained, and only ever accurate, one step ahead of reality. It has never seen its own output as input, and nothing tells it whether a chain of its predictions stays on track. DreamerV3 closes the loop by design. Its recurrent core, $h_t = f(h_{t-1}, z_{t-1}, a_{t-1})$, computes the same kind of one-step prediction an auxiliary head would, but in a compact latent space, and its prior guesses the next latent $\hat z_t$ from $h_t$ alone. That guess goes straight back into the next step, so the model can roll itself forward 15 steps without touching the environment, cheaply, because the whole rollout lives in a vector of a few hundred numbers rather than in pixels.
- The second missing piece is a way to score the rollout. A trajectory you cannot score is useless for learning a policy. DreamerV3 attaches a reward head and a continue head to every latent state, so each imagined step comes with a predicted reward and a probability that the episode is still going. Together with the rolled-forward dynamics, that is enough to build returns for a critic and advantages for an actor, without the real environment.

- An auxiliary loss asks the agent to understand what comes next. A world model asks it to be able to produce what comes next, step after step, and say how good it is.


## 4. What changes when you learn in latent space 


- Once the policy learns only in imagination, everything depends on the prior staying faithful to reality. DreamerV3 keeps it honest with one KL divergence between the posterior, which has seen the real observation, and the prior, which has not, split into two losses by stop-gradients. The dynamics loss, at full weight, pulls the prior toward the posterior. The representation loss, at a tenth of the weight, nudges the posterior toward what the prior can predict, so the latent does not fill up with detail no dynamics model could foresee. Both switch off below one nat, which leaves the rest of the gradient to reconstruction. Remember that UNREAL found reconstruction hurt as an auxiliary task; here it is the main learning signal, because the latent has to be a model of the world rather than a feature for a policy. My from-scratch PyTorch reproduction reached 930–960 on DeepMind Control walker-walk across three seeds.

- The cost is that the policy is now only as good as the model. Around 80k steps on walker, my agent's imagined returns implied about 335 per episode while real returns sat near 110. My first suspect was a real mismatch: the actor samples unsquashed Gaussian actions, the environment clips them to [−1, 1], but imagination did not, so about 30% of imagined actions were outside the range the world model had ever seen. Clipping inside the dynamics, as the official implementation does, was a one-line fix. But the runs I had already started, without it, closed the gap on their own and finished at 930–960. The dip was the walker learning to step: it falls while it learns, and the model stays optimistic about a gait it has barely seen. The gap was a signal of where the data was thin, and it closed once the agent gathered that data.

- That failure mode matters in flight. A learned model does not say "I don't know." It produces a confident rollout, and the policy optimizes against it. The model is least accurate where the data is thinnest, near the edge of the envelope, which is exactly where a confidently wrong policy is most expensive.


## 5. What I would test next 


- The first thing I tested was the imagination horizon H: 5, 15 and 30 steps, three seeds each, 300k steps on walker-walk. With action repeat 2, that is 0.25, 0.75 and 1.5 seconds of imagined time. Longer was better and more reliable: final returns of 636 ± 27, 795 ± 101 and 930 ± 16 (the 930–960 above is H = 15 trained to 500k). All three H = 30 seeds passed 800 by about 210k steps; no H = 5 seed ever did. The critic bootstraps beyond the horizon, but a quarter of a second of imagination was not enough for the actor to see where a stride leads.
- For an aerial system, the lesson is to set the horizon in seconds, not steps, long enough to cover the slowest dynamics the task depends on. An attitude correction plays out in a fraction of a second; an energy or trajectory error can take many seconds to show. At a 50 Hz control rate, H = 30 is only 0.6 seconds. The catch is section 4: every extra imagined step is another chance for model error to compound, and the policy will find it. Walker never showed that cost up to H = 30. The next test is to push H to 60 and 120 and track the gap between imagined and real returns as the horizon grows, the same signal that exposed the clipping bug. Where that gap opens is how far ahead the model can be trusted.
- The question I want to take next is one scale up. Vision-language-action models learn mostly by behavior cloning: they imitate demonstrations and rarely practice. Demonstrations are expensive, and a cloned policy has no way to improve beyond them or recover from states the demonstrator never visited. A learned world model is one way to give such a policy practice without a robot in the loop: fine-tune it with reinforcement learning inside imagined rollouts, the way Dreamer's actor learns. Everything in this essay says that would be powerful, and everything in it says where it would break. A large policy is a strong optimizer, and it will find every place the model is wrong, now in a visual, high-dimensional world where those errors are much harder to see. How far can a policy that size be allowed to learn from a model's imagination before it has to go back and touch the real world?


---

<!--
References (check each before publishing):
- Hafner et al., DreamerV3, arXiv:2301.04104 (rev. 2, 2024)
- Jaderberg et al., UNREAL, arXiv:1611.05397 (2016)
- Schwarzer et al., Self-Predictive Representations (SPR), 2021 — verify arXiv ID
- Hafner et al., PlaNet, arXiv:1811.04551 (2018) — where the RSSM comes from
- Hansen et al., TD-MPC2, arXiv:2310.16828 (2024) — the "plan at decision time" contrast
- Repo: adam-dreamer-torch — docs/equation-map.md, docs/debugging.md
-->
