"""Group custom actuator effects by their perturbation status."""

from .custom import CustomEffect


def actuator_keys(status_type):
    bindings = {}
    for key, name in ((" ", "STUCK_OFF"), ("=", "STUCK_ON")):
        if name not in status_type.__members__:
            continue
        description = name.lower()
        bindings[key] = {
            "description": description,
            "type": status_type[name].value,
            "warning": f"No {description} effect is configured.",
        }
    return bindings



class EffectCollection:
    """Dispatch keyboard requests, skipping custom effects without native bindings."""

    def key_callback(self, keycode=None):
        try:
            character = chr(keycode)
        except (TypeError, ValueError):
            return
        binding = self.keycode_dict.get(character)
        if binding is None:
            return
        for effect in self:
            status = getattr(effect, "failure_type", None)
            if status is not None and status.value == binding["type"]:
                effect.key_callback(keycode)
                return
        print(binding["warning"])


class ActuatorEffects(EffectCollection):
    """Apply actuator transformations in list order."""

    def __init__(self, perturbations):
        self.perturbations = perturbations
        self.keycode_dict = actuator_keys(self.status_type)

    def __iter__(self):
        return iter(self.perturbations)

    def apply(self, commanded_thrust, timestamp=0.0):
        actual_thrust = commanded_thrust
        for effect in self:
            if isinstance(effect, CustomEffect):
                actual_thrust = effect.apply_actuator(actual_thrust, timestamp)
            else:
                actual_thrust = effect.apply(actual_thrust, timestamp)
        return actual_thrust
