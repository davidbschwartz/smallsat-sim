"""Optional supervised initialization; independent of the benchmark matrix."""

from flax import nnx
import jax
import jax.numpy as jnp
import optax

from ..storage.rollout_batch import make_training_batch
from .runner_metrics import report
from .runner_utils import save_training_data
from smallsat_sim.controllers.pd.common import compute_control
from smallsat_sim.controllers.pd.vectorized_controller import VectorizedPDController


def pd_demonstrator(env, planner, reference):
    controller = VectorizedPDController(env, planner)
    reference = jnp.broadcast_to(reference, (env.num_envs, 7))

    def actions(states):
        # q_error = q_reference * conjugate(q); invert the feature log-map.
        vector = states[:, 3:6]
        angle = jnp.linalg.norm(vector, axis=-1, keepdims=True)
        xyz = -0.5 * jnp.sinc(angle / (2 * jnp.pi)) * vector
        w = jnp.cos(angle / 2)
        ref_w, ref_xyz = reference[:, 3:4], reference[:, 4:7]
        q = jnp.concatenate(
            (
                w * ref_w - jnp.sum(xyz * ref_xyz, axis=-1, keepdims=True),
                w * ref_xyz + ref_w * xyz + jnp.cross(xyz, ref_xyz),
            ),
            axis=-1,
        )
        obs = jnp.concatenate(
            (states[:, :3] + reference[:, :3], q, states[:, 6:]), axis=-1
        )
        return compute_control(
            obs,
            reference,
            jnp.asarray(controller.v_ref),
            (controller.Kp_x, controller.Kd_x, controller.Kp_q, controller.Kd_q),
            controller._inverse_mixer,
            controller._groups,
            velocity_frame="body",
            xp=jnp,
            mixer=controller.B_matrix,
            bounds=controller._bounds,
        )

    return actions


@nnx.jit
def supervised_step(
    actor,
    actor_optimizer,
    critic,
    critic_optimizer,
    observations,
    actions,
    returns,
):
    actor_loss, actor_grad = nnx.value_and_grad(
        lambda model: jnp.square(
            model.deterministic_action(observations) - actions
        ).mean()
    )(actor)
    critic_loss, critic_grad = nnx.value_and_grad(
        lambda model: jnp.square(model(observations) - returns).mean()
    )(critic)
    actor_optimizer.update(actor_grad)
    critic_optimizer.update(critic_grad)
    return actor_loss, critic_loss


def pretrain(runner, *, mode="fresh"):
    if mode != "fresh":
        raise ValueError(
            "Pretraining is a fresh supervised initialization; use policy resume for RL"
        )
    runner.begin("pretraining", mode)
    demonstration = pd_demonstrator(runner.env, runner.planner, runner.reference_point)
    collector = runner.collector(
        "privileged", stochastic=False, demonstration=demonstration
    )
    result = runner.collect(collector, randomize=runner.env.train_with_failures)
    batch = make_training_batch(
        result, runner.context_scale, runner.agent.gamma, runner.agent.lam
    )
    actions = jnp.clip(
        result.actions.reshape(-1, runner.env.act_dim),
        runner.agent.actor.act_low,
        runner.agent.actor.act_high,
    )
    save_training_data(
        runner.ckpt_dir,
        runner.pretraining_data_file_name,
        {
            "observations": batch.observations,
            "actions": actions,
            "returns": batch.returns,
        },
    )
    actor_opt = nnx.Optimizer(
        runner.agent.actor, optax.adam(runner.training_cfg.pretraining_lr)
    )
    critic_opt = nnx.Optimizer(
        runner.agent.critic, optax.adam(runner.training_cfg.pretraining_lr)
    )
    for epoch in range(runner.training_cfg.pretraining_epochs):
        indices = jax.random.randint(
            runner._take_keys(),
            (runner.training_cfg.pretraining_batch_size,),
            0,
            actions.shape[0],
        )
        actor_loss, critic_loss = supervised_step(
            runner.agent.actor,
            actor_opt,
            runner.agent.critic,
            critic_opt,
            batch.observations[indices],
            actions[indices],
            batch.returns[indices],
        )
        report(
            runner,
            "pretraining",
            epoch + 1,
            {"actor_loss_mean": actor_loss, "critic_loss_mean": critic_loss},
        )
    # Supervised updates must not carry Adam moments into the on-policy optimizer.
    runner.agent.reset_optimizers()
    runner.save("pretraining")
