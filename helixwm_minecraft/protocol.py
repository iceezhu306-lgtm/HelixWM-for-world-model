"""Small, versioned IPC frames. No pickle and no simulator dependencies."""
import base64
import json
import math
import struct

SCHEMA_VERSION = 1
MAX_FRAME_BYTES = 1024 * 1024
RGB_SHAPE = (64, 64, 3)
RGB_BYTES = 64 * 64 * 3
TASK_ID = "HelixWMVisibleNavigation-v0"


class ProtocolError(RuntimeError):
    pass


def _exact(stream, count):
    chunks = []
    remaining = count
    while remaining:
        part = stream.read(remaining)
        if not part:
            raise EOFError("Worker stream ended inside a frame")
        chunks.append(part)
        remaining -= len(part)
    return b"".join(chunks)


def read_frame(stream):
    size = struct.unpack(">I", _exact(stream, 4))[0]
    if not 1 <= size <= MAX_FRAME_BYTES:
        raise ProtocolError("Invalid IPC frame size; possible stdout contamination")
    try:
        value = json.loads(_exact(stream, size).decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise ProtocolError("Invalid JSON IPC frame") from exc
    if not isinstance(value, dict):
        raise ProtocolError("IPC frame must contain an object")
    return value


def write_frame(stream, value):
    payload = json.dumps(value, allow_nan=False, separators=(",", ":")).encode("utf-8")
    if not 1 <= len(payload) <= MAX_FRAME_BYTES:
        raise ProtocolError("Outgoing IPC frame exceeds limit")
    stream.write(struct.pack(">I", len(payload)) + payload)
    stream.flush()


def integer(value, name, lower=0, upper=2147483647):
    if type(value) is not int or not lower <= value <= upper:
        raise ProtocolError("Invalid " + name)
    return value


def encode_rgb(image):
    if tuple(getattr(image, "shape", ())) != RGB_SHAPE or str(getattr(image, "dtype", "")) != "uint8":
        raise ProtocolError("Expected uint8 RGB image with shape (64,64,3)")
    data = image.tobytes(order="C")
    if len(data) != RGB_BYTES:
        raise ProtocolError("Unexpected RGB byte count")
    return {"shape": list(RGB_SHAPE), "dtype": "uint8", "encoding": "base64", "data": base64.b64encode(data).decode("ascii")}


def decode_rgb(value):
    if not isinstance(value, dict) or set(value) != {"shape", "dtype", "encoding", "data"}:
        raise ProtocolError("Observation contains unexpected fields")
    if value["shape"] != list(RGB_SHAPE) or value["dtype"] != "uint8" or value["encoding"] != "base64":
        raise ProtocolError("Unexpected observation contract")
    try:
        raw = base64.b64decode(value["data"], validate=True)
    except (TypeError, ValueError) as exc:
        raise ProtocolError("Invalid observation encoding") from exc
    if len(raw) != RGB_BYTES:
        raise ProtocolError("Unexpected observation byte count")
    return raw


def validate_action(action):
    if not isinstance(action, (list, tuple)) or len(action) != 6:
        raise ProtocolError("Action must contain six normalized numbers")
    values = []
    for value in action:
        if type(value) not in (int, float) or not math.isfinite(value) or not -1 <= value <= 1:
            raise ProtocolError("Action must be finite and within [-1,1]")
        values.append(float(value))
    return values


def native_action(action, action_repeat):
    """Camera order is MineRL pitch,yaw; total rotation is per decision."""
    x = validate_action(action)
    integer(action_repeat, "action_repeat", 1, 16)
    return {
        "forward": int(x[0] > 0.2), "back": int(x[0] < -0.2),
        "left": int(x[1] < -0.2), "right": int(x[1] > 0.2),
        "camera": [10.0 * x[3] / action_repeat, 15.0 * x[2] / action_repeat],
        "jump": int(x[4] > 0.0), "sprint": int(x[5] > 0.0),
        "attack": 0, "sneak": 0,
    }


def validate_result(value, expected_seed=None):
    expected = {"observation", "reward", "terminated", "truncated", "env_steps", "episode_ticks", "episode_seed", "task_id"}
    if not isinstance(value, dict) or set(value) != expected:
        raise ProtocolError("Unexpected result fields; privileged data must not cross IPC")
    decode_rgb(value["observation"])
    if type(value["reward"]) not in (int, float) or not math.isfinite(value["reward"]):
        raise ProtocolError("Nonfinite reward")
    if type(value["terminated"]) is not bool or type(value["truncated"]) is not bool or (value["terminated"] and value["truncated"]):
        raise ProtocolError("Invalid terminal flags")
    integer(value["env_steps"], "env_steps", 0, 16)
    integer(value["episode_ticks"], "episode_ticks")
    integer(value["episode_seed"], "episode_seed")
    if expected_seed is not None and value["episode_seed"] != expected_seed:
        raise ProtocolError("Worker returned the wrong episode seed")
    if value["task_id"] != TASK_ID:
        raise ProtocolError("Worker returned an unexpected task identity")
    return value
