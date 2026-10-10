"""Per-finger haptic feedback from Newton contacts, in Isaac Lab's glove format.

Each channel is the normal contact force [N] between one finger's links (proximal, intermediate, distal) and the
grasped object(s), the same quantity Isaac Lab's `per_finger_object_grip` signal produces, in its default channel
order: thumb, index, middle, ring, pinky. `send()` forwards a frame to any device implementing Isaac Lab's
`HapticFeedbackReceiver` protocol (`send_haptic(endpoint, values)`, endpoints "left"/"right"); Isaac Teleop then
applies the gain/deadband curve and writes the Manus glove vibration motors.
"""
import numpy as np
import warp as wp
from newton.sensors import SensorContact

CHANNELS = ("thumb", "index", "middle", "ring", "pinky")
_RIG_FINGER = {"thumb": "Thumb", "index": "Index", "middle": "Middle", "ring": "Ring", "pinky": "Little"}


class FingerHaptics:
    def __init__(self, model, body_labels, object_bodies, sides=("Left", "Right")):
        """body_labels: model body labels; object_bodies: body indices of the grasped object(s).
        Construct before allocating the Contacts buffer (SensorContact requests contact forces)."""
        names = [b.rsplit("/", 1)[-1] for b in body_labels]
        self.rows = []  # (endpoint, channel) per sensing body
        sensing = []
        for side in sides:
            for c, ch in enumerate(CHANNELS):
                for k in ("Metacarpal", "Proximal", "Intermediate", "Distal"):
                    n = f"{side}Hand{_RIG_FINGER[ch]}{k}"
                    if n in names and not (ch != "thumb" and k == "Metacarpal"):
                        sensing.append(names.index(n))
                        self.rows.append((side.lower(), c))
        self.sensor = SensorContact(model, sensing_bodies=sensing, counterpart_bodies=list(object_bodies))
        self.endpoints = [s.lower() for s in sides]
        row_channel = [self.endpoints.index(e) * len(CHANNELS) + c for e, c in self.rows]
        self._row_channel = wp.array(np.array(row_channel, np.int32), device=model.device)
        self.channels = wp.zeros(len(self.endpoints) * len(CHANNELS), dtype=float, device=model.device)

    def update(self, state, contacts):
        """All on the device (graph-capturable): sensor forces -> per-finger normal force in self.channels."""
        from .hand_control import haptic_channels_kernel
        self.sensor.update(state, contacts)
        self.channels.zero_()
        wp.launch(haptic_channels_kernel, dim=len(self.rows), inputs=[self.sensor.force_matrix,
                  self.sensor.force_matrix_friction, self._row_channel, self.sensor.force_matrix.shape[1]],
                  outputs=[self.channels], device=self.channels.device)

    @property
    def values(self):
        """{endpoint: per-finger normal force [N]} (copies 10 floats to the host)."""
        v = self.channels.numpy().reshape(len(self.endpoints), len(CHANNELS))
        return {e: v[i] for i, e in enumerate(self.endpoints)}

    def normalized(self, full_scale_n=10.0):
        return {e: np.clip(v / full_scale_n, 0.0, 1.0) for e, v in self.values.items()}

    def send(self, receiver):
        """Forward the latest frame (raw newtons; the device side maps them) to a HapticFeedbackReceiver."""
        for endpoint, v in self.values.items():
            receiver.send_haptic(endpoint, v.tolist())


__all__ = ["CHANNELS", "FingerHaptics"]
