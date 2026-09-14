# Classical controllers and planners

`make_experiment` supports `pd`, `lqr`, `mpc`, `mpcc`, and `gp_mpc` with the
classical `astrobee`, `cubesat`, and `sprint` environments. Use `asset` for a custom
spacecraft. MPC variants require the native MPC installation described in the
repository README; their imports are deferred until selected.

For example, hold CubeSat at a point above the Gateway:

```python
from smallsat_sim.api.experiments import make_experiment

experiment = make_experiment(
    vehicle="cubesat",
    controller="lqr",           # also pd or mpc
    planner="oracle",
    planner_radius=0.0,
    planner_options={"z_offset": 10.0},
    log=False,
)
try:
    while experiment.env.data.time < 20.0:
        command = experiment.controller.get_control_input(experiment.env)
        experiment.env.step(input=command)
finally:
    experiment.env.close()
```

These arguments are also fields in an experiment YAML. `planner_options` forwards
constructor options; `planner_radius`, `planner_spacing`, and
`planner_clearance_dist` retain their existing meanings.

| Planner name | Supported classical controllers | Behavior |
|---|---|---|
| `oracle` | PD, LQR, MPC | Circular references, or one point with radius zero; supports offsets and plane selection |
| `mission` | All five | Prescribed piecewise-line mission, required by contouring controllers |
| `mission_spline` | PD, LQR, MPC | Prescribed spline; its spline parameter is not exact physical arc length |
| `docking` | PD, LQR, MPC | Staged, geometry-checked approach; supply pre-dock and dock positions in `planner_options` |
| `ad_star` | PD, LQR, MPC | Grid search with spacecraft swept-pose clearance; use `overrides.planner` for configuration |

MPCC and GP-MPC reject incompatible planners before allocating a solver. Mission
and Oracle are reference generators; they do not compute obstacle avoidance or
actuator-feasible timing. A visual clearance overlay is not a collision guarantee.
AD* checks a snapshot of MuJoCo collision geometry and rechecks the next commanded
segment. It raises `NoPathError` when no safe reference is available. Stop the
vehicle and explicitly `replan(start=..., goal=...)` after an obstruction. Endpoints
must lie on its resolution grid. Replanning rebuilds the graph; it does not predict
moving obstacles. Optional background planning is cancelled when its environment
closes.

PD defaults use mass and maximum diagonal inertia with position bandwidth
0.35 rad/s and attitude bandwidth 0.8 rad/s. The experiment factory applies this
rule to custom PD assets, preserving each explicit gain override. NumPy PD and
vectorized PD use the same body-attitude feedback and bounded allocation. These
are regulation defaults; tune separately for faster tracking or fuel efficiency.

LQR uses a quaternion tangent model, exact discretization, cached gains and bounded
wrench allocation. It remains a local controller; a successful small-error test
does not establish large-angle capture or fault tolerance.

All MPC variants convert observations to the symbolic model's body-velocity
convention. Set `Ts = sim.dt * control_decimation`; inconsistent values are rejected.
`R` must match the asset's actuator count. Default solver build directories are
isolated per controller; set `control.<controller>.code_export_directory` to retain
one at a chosen location. Failed or nonfinite solutions raise `MPCSolverError`
instead of returning an invalid command. Callers must stop or invoke their own
fallback on this exception.

For independent GP-MPC trials, call `controller.reset(env)` after resetting the
environment. This clears the GP data and history as well as the solver warm start.
Online updates only use transitions following a valid command and a matching
control interval. Backward simulation-clock jumps also trigger a reset. GP learning
has not yet been shown to outperform nominal control across the spacecraft.

Docking stage changes require attitude alignment and low linear/angular rates.
The optional `approach_speed` limits reference motion. It is not a bound on actual
vehicle speed. Default `is_docked` requires pose/rate tolerances and contact with
`contact_body` (`gateway_full` by default), below `max_contact_force`. Set
`require_contact=False` only for a pose-only rendezvous task. Constructor checks
reject an obstructed exterior approach; the first reference query also checks the
initial route. The final `dock_distance` region allows intentional compliant
contact. No latch or flight docking hardware is modeled.

The paper docking harness explicitly disables the planner's early geometry
exception so that its existing `initial_penetration` outcome can be recorded. Its
own frozen success definition remains separate from `is_docked`. Existing paper
YAML snapshots were not retuned by these corrections; new results must retain
source/configuration fingerprints.

## Sustained MPCC and GP-MPC tracking

Use [astrobee_CL.py](../experiments/astrobee_CL.py) for an editable MPCC
mission-tracking example. Controller regression coverage lives in
[test_classical_corrections.py](../src/tests/test_classical_corrections.py).

`control.GPMPC.learning_enabled=False` freezes the untrained prior for an ablation;
it retains the GP uncertainty calculation. Optional GP settings are
`prior_variance`, `observation_noise`, `lengthscale`, `max_points`, and
`nlp_iterations` (default **1**, matching RTI). Training uses RK4 to match the
nominal prediction scheme. `reset(env)` also clears native solver memory.
An out-of-bounds GP command raises `ValueError` even if the solver reports success.

Both contouring controllers accept `progress_rate_limit` (default 0.2 m/s).
GP-MPC's optional `progress_weight` replaces its legacy distance-dependent progress
reward; MPCC uses `cost.q_theta`. These settings limit the optimizer's path progress; they do not guarantee a
physical spacecraft speed.
