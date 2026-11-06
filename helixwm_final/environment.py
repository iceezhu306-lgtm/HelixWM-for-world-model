"""Use the exact old Walker factory under a separately validated Final protocol."""
from pathlib import Path
import hashlib
import json

from helixwm_matched.environment import MatchedWalkerEnv
from helixwm_matched.protocol import validate_protocol as validate_original_protocol
from .protocol import validate_protocol

BASE_PROTOCOL_SHA256 = '91747b104a16ea5cd1fb7df094dcd0cc7fc51f0bd64cfbdcd8cdf865f13c28c0'


def original_environment_contract():
    path = Path(__file__).resolve().parents[1]/'configs/dreamerv3_helixwm_matched_20261026.json'
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != BASE_PROTOCOL_SHA256:
        raise ValueError('Original shared environment contract bytes changed')
    return validate_original_protocol(json.loads(raw))


class FinalWalkerEnv(MatchedWalkerEnv):
    def __init__(self, config, *, diagnostics=False, env_factory=None):
        self.final_protocol = validate_protocol(config)
        base = original_environment_contract()
        # Other environment metadata can describe this new experiment; every
        # actual simulator field remains inherited from the original factory.
        for key in ('domain', 'task', 'observation', 'image_size', 'camera_id',
                    'action_dimensions', 'action_repeat', 'episode_action_decisions',
                    'episode_dmc_calls', 'privileged_current_state_to_policy'):
            if self.final_protocol['environment'].get(key) != base['environment'][key]:
                raise ValueError('Final environment differs from original contract: '+key)
        super().__init__(base, diagnostics=diagnostics, env_factory=env_factory)
