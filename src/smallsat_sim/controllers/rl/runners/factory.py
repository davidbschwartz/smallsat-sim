"""Choose the learning lifecycle once at the application boundary."""
def make_runner(env, planner, *, config):
    if config.algorithm == 'sac':
        from .off_policy_runner import OffPolicyRunner
        return OffPolicyRunner(env, planner, config=config)
    from .on_policy_runner import OnPolicyRunner
    return OnPolicyRunner(env, planner, config=config)
