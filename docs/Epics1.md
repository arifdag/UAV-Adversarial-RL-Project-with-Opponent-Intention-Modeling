Epic E1 Simulation Sandbox Ready (Week 1–2)
#	Story	Acceptance / DoD	Implementation Checklist
E1-1	As a researcher I want to build and launch the stock pybullet drones sim so that I can confirm my tool-chain works.	conda env list shows drones; python -m gym_pybullet_drones.examples.pid runs without error.	bash\nconda create -n drones python=3.10\nconda activate drones\npip install -e gym-pybullet-drones/ # local clone\npip install stable-baselines3[extra] gymnasium torch\n Installation recipe follows official docs (github.com)

E1-2	I can spawn two Crazyflie-2X quads in a head-to-head arena and step the Gym env from Python.	A 200-step loop prints non-NaN obs and rew.	python\nenv = MultiHoverAviary(num_drones=2, obs='kin', act='vel', gui=True) # API uses enums internally\nobs, _ = env.reset()\nfor _ in range(200):\n obs, rew, term, trunc, _ = env.step(np.zeros(env.action_space.shape))\n Defaults from learn.py example (raw.githubusercontent.com)

E1-3	I have a new custom env DogfightAviary that inherits MultiRLAviary.	pytest tests/test_dogfight_env.py passes space-shape assertions.	Create uav-intent-rl/envs/DogfightAviary.py:python\nclass DogfightAviary(MultiRLAviary):\n EPISODE_LEN_SEC = 30\n DEF_DMG_RADIUS = 0.3 # m, hit if closer and within FOV\n def _computeReward(self):\n return self._calc_hits() - 0.01 # shaping\n def _computeTerminated(self):\n return self._blue_down() or self._red_down()\n Use _getDroneStateVector() helpers exactly like HoverAviary (raw.githubusercontent.com)

E1-4	Episode video & CSV logs are saved for post-mortem.	After env = Monitor(...) a runs/2025-07-xx/ folder contains .mp4 + progress.csv.	Wrap env with gymnasium.wrappers.RecordVideo and RecordEpisodeStatistics; pass render_mode="rgb_array" if headless.
________________________________________
Epic E2 Scripted Baseline Opponent (Week 2)
Story	DoD	Tasks
E2-1 Red drone follows a scripted “pursue & fire” policy so that Blue has a stationary adversary.	100 episodes, Red hits Blue ≥ 65 % of runs.	1. Add uav-intent-rl/policies/scripted_red.py.2. Implement simple proportional controller: target Blue’s XY, maintain z=1 m, shoot when dist<0.3 m.3. Unit-test with deterministic seed.
E2-2 Arena resets with random spawn poses (±π yaw, 2–4 m separation) to avoid over-fitting.	Histograms of spawn distance show uniform distribution.	Modify DogfightAviary.reset() to randomise initial_xyzs before calling super().reset().
________________________________________
Epic E3 PPO Baseline (No Opponent Modeling) (Week 3–4)
Story	DoD	Implementation Notes
E3-1 Blue learns PPO policy against fixed Red.	Averaged over 10 eval runs, win-rate ≥ 60 % by 3 M steps.	Config: <project>/configs/ppo_nomodel.yaml -> n_envs: 8, γ: 0.99, lr: 3e-4, clip: 0.2.Use SB3 vectorised env wrapper make_vec_env(DogfightAviary).Log with TensorBoard.
E3-2 Best checkpoint exported.	models/baseline_no_model.zip committed and loads without error.	Call model.save(...) after EvalCallback threshold reached (pattern copied from learn.py) (raw.githubusercontent.com)

________________________________________
Epic E4 Self-Play League (Week 5–6)
Story	DoD	Tasks
E4-1 Convert env to Ray RLlib MultiAgentEnv so both drones have policies.	rllib rollout script works; printed sample shows two policy ids.	Implement policy_mapping_fn that assigns "blue" / "red".Use PPO with shared weights (config["share_observations"] = True) to speed learning.
E4-2 League Elo table auto-generated weekly.	CSV artifacts/elo_matrix.csv produced by cron job; heat-map appears in README.	Store past checkpoints every 0.5 M steps; run round-robin evaluation script that fills matrix.
________________________________________
Epic E5 Opponent-Intention Module (Week 7–8)
Story	DoD	Key Code Touchpoints
E5-1 Add auxiliary head that predicts Red’s next discrete action bucket.	Prediction accuracy ≥ 60 % on validation buffer.	1. Subclass torch.nn.Module → IntentPPOPolicy.2. Extra head self.opp_head = nn.Linear(latent_dim, act_dim).3. Loss L_total = L_PPO + λ·CrossEntropy(pred, a_red).4. Tune λ via sweep (0.1→1).
E5-2 Training script logs both RL reward and intent-head accuracy.	TensorBoard shows two curves; accuracy rises while episode return does not degrade.	Use SB3’s BaseAlgorithm hooks to add custom metrics.
________________________________________
Epic E6 Modelled Agent Beats Baseline (Week 9–10)
Story	DoD	Experiments
E6-1 Blue-model vs Blue-baseline benchmark.	Modelled Blue wins ≥ 70 % of 200 evaluation games (95 % CI).	Freeze baseline weights; run evaluation harness evaluate.py --blue_a model_intent --blue_b baseline_no_model.
E6-2 Ablation λ = 0 shows ≥ 10 % drop in win-rate.	Notebook notebooks/ablation.ipynb plots bar chart (CI bars).	Retrain with same seed but λ = 0; re-evaluate.
________________________________________
Epic E7 Behaviour Visualisation & Analysis (Week 11)
Story	DoD	Assets
E7-1 Trajectory plots highlight anticipatory manoeuvres.	figs/intent_vs_nomodel.svg shows Blue-model flanking earlier than baseline.	Dump JSON trajectory dicts; use Matplotlib to overlay xy paths and scatter shot-events.
E7-2 MP4 demo clip recorded.	media/dogfight_intent_demo.mp4 plays in README.	Use env record=True; trim with ffmpeg.
________________________________________
Epic E8 Paper & Repo Packaging (Week 12)
Story	DoD	Checklist
E8-1 6-page draft with reproducibility checklist ready for arXiv.	paper/draft.pdf builds via make and cites Panerati et al. 2021.	Include install snippet from project README; cite gym-pybullet-drones IROS-21 paper (utiasdsl.github.io)

E8-2 Public GitHub with one-line training command.	README.md first code block runs python train.py --config configs/ppo_intent.yaml.	Push models ≥ 50 MB to Git-LFS.
________________________________________
Additional Engineering Tips
•	Coding pattern: keep env-specific constants (hit-radius, ammo) in envs/config.py, import into env and reward-calc code to avoid magic numbers.
•	Data management: each training run writes under runs/YYYY-MM-DD_HH-MM-SS/ (TensorBoard + checkpoints + videos) to keep artefacts tidy.
•	CI: add a GitHub Action that runs pytest && python smoke_train.py --steps 200 on every push. The smoke script uses local=True flag in the example (raw.githubusercontent.com) to keep the job under 5 minutes.
•	Hyper-param tuning: once pipeline stabilises, integrate Optuna via SB3’s HyperOptCallback to search λ, lr, clip-range.
•	Sample efficiency: if training speed drags, bump n_envs to 32 with make_vec_env("shared_memory") (PyBullet is CPU-bound but scales well across cores).
