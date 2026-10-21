# Interest-Management Radius Tuning: GA vs LightGBM Cost Model

Motivation: Distributed virtual environments must limit how much state each avatar receives, and the
interest-management radius controls that trade-off between bandwidth and missed interactions. The paper
"Evolutionary optimization of parameters for distributed virtual environments" tunes such parameters with an
evolutionary algorithm, producing one setting for the whole workload. This experiment asks whether a learned
cost model that adapts the radius to the workload does better.

Question: Can a LightGBM cost model that conditions the interest-management radius on user count and movement
speed beat a single evolutionary-tuned radius on unseen, shifted avatar workloads in a simulated virtual
environment?

Dataset: Avatar workloads are generated in code with numpy (seeded random-waypoint simulation in a 2D arena,
no download). This is not the paper's own dataset (same_as_paper = false). Training workloads use 10-80
avatars with speeds 1-5; test workloads use 60-120 avatars with speeds 3-8 to test distribution shift.

Baseline: a genetic algorithm (population 8, 10 generations, mutation over the radius grid) that finds one
radius minimising mean training cost and applies it to every test workload.

Improved model: a LightGBM regressor trained on (workload features, radius, cost) rows. For each test
workload it picks the radius with the lowest predicted cost. Hyper-parameters are chosen by grouped
cross-validation on the training workloads only.

How to run:
```
pip install -r requirements.txt
python analysis.py
```

Results: results are written to results.json (and figures/regret_comparison.png).

## Reference

This project is motivated by the following paper:

> SE Parkin, P Andras, G Morgan (2007). *Evolutionary optimization of parameters for distributed virtual environments*. 2007 IEEE Congress on. https://ieeexplore.ieee.org/abstract/document/4424633/

Paper author: Prof. Graham Morgan, Newcastle University (graham.morgan@ncl.ac.uk)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
