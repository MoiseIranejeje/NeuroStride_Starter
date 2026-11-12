# NeuroStride — TKS Tech Iterate

**Main question:** Can an SNN robot controller learn locomotion and adapt to new terrain, and how does it compare with an ANN controller?

This starter kit covers **Milestone 1 only**: run a genuine quadruped physics simulation, inspect sensors and motor actions, and save a baseline. It intentionally does **not** train a robot or claim any learned performance.

## Scientific foundations

- Jiang et al., *Fully Spiking Neural Network for Legged Robots* (arXiv v2): https://arxiv.org/html/2310.05022v2
- Published reference code: https://github.com/thisisnotahuman/FullySNNforLeggedRobots
- Ant-v5 documentation: https://gymnasium.farama.org/environments/mujoco/ant/

We will use a simpler MuJoCo Ant-v5 benchmark to learn the fundamentals first. It is **not an exact reproduction** of the paper's A1/Cassie/Humanoid/Isaac Gym setup.

## Windows installation

Install 64-bit Python 3.11 or 3.12 if not already installed. In PowerShell, navigate to this extracted folder and run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_baseline.py --controller random --episodes 3
.\.venv\Scripts\python.exe run_baseline.py --controller zero --episodes 3
```

If you use Python 3.12, replace `py -3.11` with `py -3.12`. Confirm your installed Python version first using `py -0p`. If your local Python isn't configured yet, we can switch to a cloud notebook.

**Watch the actual robot (desktop with graphics support):**

```powershell
.\.venv\Scripts\python.exe run_baseline.py --controller random --episodes 1 --render
```

If the viewer cannot open, omit `--render`. The headless experiment and CSV can still run.

## Linux/macOS installation

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run_baseline.py --controller random --episodes 3
```

Depending on the graphics stack, `--render` may need working OpenGL libraries.

## How the robot works

```
Observation (105 numbers)
    -> Controller (currently random values or all zeros)
    -> 8 motor commands, within [-1, 1]
    -> MuJoCo physics updates the robot
    -> New observation + reward + done signal
    -> Repeat
```

**105 observations**: body height/orientation, joint angles, velocities and contact forces (see Ant-v5 docs).

**8 actions**: eight values for eight actuated joints. The raw action values specify motor control, not where to walk.

**Reward**: the environment combines healthy/survival reward, forward speed reward, motor-use cost, and contact-force cost. A positive reward does not by itself demonstrate learning.

## What the results mean

Running the script creates `results/random_baseline.csv` and `results/zero_baseline.csv`.

- `steps`: number of simulated decision steps completed.
- `episode_reward`: sum of actual simulator rewards; not a percentage or accuracy score.
- `forward_displacement_m`: final minus initial x-position; can be negative.
- `min_torso_height_m`: lowest observed torso height over the episode.
- `end_reason`: ended because the simulator reported an unhealthy robot, time limit, or the experiment reached `--max-steps`.

For scientific comparison, training and evaluation must be repeated across multiple random seeds under controlled conditions. These first 3-episode runs are **only a functionality check**, not publishable evidence.

## Milestone 1 success checklist

- [ ] The environment loads without an error.
- [ ] The program prints `Observation shape: (105,)` and `Action shape: (8,)`.
- [ ] Random and zero baseline CSVs are created with real values.
- [ ] I can explain what one observation value means, what one motor action means, and what reward means.

### Understanding checkpoint

1. Does Ant-v5 currently have an SNN? Why or why not?
2. In the 105 observations, what is `observation[0]`?
3. Why can't we claim that the robot has learned just because it travelled forward in one run?

## Roadmap beyond Milestone 1

- M2: implement and train an ANN PPO locomotion baseline with reproducible logs.
- M3: understand and implement temporal encoding, LIF neurons, and the spike-to-action decoder.
- M4: integrate an SNN actor with a suitable RL training loop.
- M5: add terrain variations and test generalization/noise robustness.
- M6: run controlled ANN/SNN comparisons; write technical report and Loom walkthrough; seek expert validation.

**Project standard:** No preprogrammed success animations, fabricated results, or claims that the SNN saves battery power without a suitable measurement/analysis setup.
