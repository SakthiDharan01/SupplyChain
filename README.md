# Governed Autonomous Mitigation in Supply Chain Digital Twins — code and results

Code, random seeds and result files for the paper **"Governed Autonomous Mitigation in Supply Chain Digital Twins Using Evolving Supplier Trust Beliefs"** (ICon INDIA 2026).

Authors: Sakthi Dharan M A, Srinivasan D, Tarun R R, S Sindhu — Department of Computer Science and Engineering, Easwari Engineering College, Chennai, India.

## What is here

| Folder | Contents |
|---|---|
| `code/` | Simulator (`sim.py`), main experiments, diagnostics, and the three revision experiments (`rev_a`, `rev_b`, `rev_c`) |
| `results/` | JSON and log outputs of every experiment, as used in the paper |
| `figures/` | The result figures (PDF) |

## Requirements

Python 3.13, NumPy 2.5, SciPy 1.18, scikit-learn 1.9, Matplotlib 3.11 (`pip install -r requirements.txt`). Older versions will probably work but were not tested.

## Reproducing the results

Run from inside `code/`. Each script writes its output JSON to the current folder.

```bash
cd code
python3 sim.py tune.pkl                  # grid search, tuning seeds 1000-1019
python3 explore_surprise.py              # tuning of the surprise-propagation filter
python3 run_experiments.py results.json  # main table, ablation, latency, correlation sweep (test seeds 0-49)
python3 extra_experiments.py             # cold-start comparison, autonomous-spend accounting
python3 diag_gate.py                     # gate precision and credible-interval coverage
python3 frontier.py                      # cost vs unreviewed spend
python3 make_fig.py                      # figures
# Revision experiments (reviewer-driven)
python3 rev_a_calibration_controls.py    # calibration (temperature, isotonic, conformal) and execution controls  -> rev_a.json
python3 rev_b_robustness.py              # graph errors, missing/delayed data, other disruption generators       -> rev_b.json
python3 rev_c_runtime.py                 # computational cost of one daily update                                -> rev_c.json
```

**Seeds:** tuning 1000–1019, supervised training 2000–2029, test 0–49.
**Reference outputs:** `results/results.json` and `results/results_repro.json` are identical, which confirms that the original experiments reproduce exactly. `results/rev_a.json`, `rev_b.json` and `rev_c.json` hold the revision results. Runtime numbers in `rev_c.json` depend on the machine.

## Notes

- The calibration maps in `rev_a_calibration_controls.py` are fitted with the hidden failure probability, which exists only in the simulator.
- No licence has been chosen yet. Until one is added, all rights are reserved by the authors.
