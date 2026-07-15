"""Gradual speed control for MountainCarContinuous (car-game style throttle).

Hold Right to ramp forward force toward +1, hold Left toward -1.
Release keys and speed coasts back toward 0.
Use this as a drop-in action source (action in [-1, 1]).

Test standalone:
    python controll.py
    python controll.py --accel 1.5 --decel 2.0 --max-speed 1.0
"""

from __future__ import annotations

import argparse
import sys
import threading
import time

import numpy as np


class SpeedController:
    """Car-game style throttle: gradual accel / reverse / coast to zero.

    Parameters
    ----------
    max_speed:
        Clamp for the signed speed / force in ``[-max_speed, max_speed]``.
        For MountainCarContinuous this should usually be ``1.0``.
    acceleration:
        Units per second while Left or Right is held.
    deceleration:
        Units per second of coast-down toward 0 when neither key is held.
    """

    def __init__(
        self,
        max_speed: float = 1.0,
        acceleration: float = 1.5,
        deceleration: float = 2.0,
    ) -> None:
        if max_speed <= 0:
            raise ValueError("max_speed must be positive")
        if acceleration < 0 or deceleration < 0:
            raise ValueError("acceleration and deceleration must be >= 0")

        self.max_speed = float(max_speed)
        self.acceleration = float(acceleration)
        self.deceleration = float(deceleration)

        self._lock = threading.Lock()
        self._speed = 0.0
        self._forward = False  # Right
        self._backward = False  # Left
        self.reset_requested = False
        self.save_requested = False
        self.quit_requested = False

        from pynput import keyboard

        self._keyboard = keyboard
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )

    @property
    def speed(self) -> float:
        with self._lock:
            return self._speed

    def start(self) -> None:
        self._listener.start()

    def stop(self) -> None:
        self._listener.stop()

    def set_forward(self, pressed: bool) -> None:
        with self._lock:
            self._forward = pressed

    def set_backward(self, pressed: bool) -> None:
        with self._lock:
            self._backward = pressed

    def reset(self) -> None:
        with self._lock:
            self._speed = 0.0

    def _on_press(self, key) -> None:
        keyboard = self._keyboard
        with self._lock:
            if key == keyboard.Key.right:
                self._forward = True
            elif key == keyboard.Key.left:
                self._backward = True
            elif key == keyboard.Key.esc:
                self.quit_requested = True
            else:
                try:
                    char = key.char.lower() if key.char else ""
                except AttributeError:
                    char = ""
                if char == "q":
                    self.quit_requested = True
                elif char == "r":
                    self.reset_requested = True
                    self._speed = 0.0
                elif char == "s":
                    self.save_requested = True

    def _on_release(self, key) -> None:
        keyboard = self._keyboard
        with self._lock:
            if key == keyboard.Key.right:
                self._forward = False
            elif key == keyboard.Key.left:
                self._backward = False

    def update(self, dt: float) -> float:
        """Advance throttle by ``dt`` seconds; return current speed in [-max, max]."""
        if dt < 0:
            raise ValueError("dt must be >= 0")

        with self._lock:
            forward = self._forward
            backward = self._backward

            if forward and not backward:
                self._speed += self.acceleration * dt
            elif backward and not forward:
                self._speed -= self.acceleration * dt
            elif not forward and not backward:
                # Coast toward zero (car-game feel).
                if self._speed > 0:
                    self._speed = max(0.0, self._speed - self.deceleration * dt)
                elif self._speed < 0:
                    self._speed = min(0.0, self._speed + self.deceleration * dt)
            # both held → hold current speed (cancel inputs)

            self._speed = max(-self.max_speed, min(self.max_speed, self._speed))
            return self._speed

    def get_action(self) -> np.ndarray:
        """MountainCarContinuous action: Box(-1, 1, (1,)) as float32 array."""
        with self._lock:
            return np.asarray([self._speed], dtype=np.float32)


def speed_bar(speed: float, max_speed: float, width: int = 40) -> str:
    """ASCII bar centered at 0: left = reverse, right = forward."""
    half = width // 2
    if max_speed <= 0:
        filled = 0
    else:
        filled = int(round(abs(speed) / max_speed * half))
    filled = max(0, min(half, filled))

    left = " " * (half - filled) + "#" * filled if speed < 0 else " " * half
    right = "#" * filled + " " * (half - filled) if speed > 0 else " " * half
    center = "|"
    return f"[{left}{center}{right}]"


def main(args: argparse.Namespace) -> None:
    controller = SpeedController(
        max_speed=args.max_speed,
        acceleration=args.accel,
        deceleration=args.decel,
    )
    controller.start()

    dt = args.dt
    print(
        "SpeedController test (MountainCarContinuous throttle)\n"
        "  → : accelerate forward  (toward +max)\n"
        "  ← : accelerate backward (toward -max)\n"
        "  (release) : coast toward 0\n"
        "  R : reset speed to 0\n"
        "  S : (save flag; used by play.py)\n"
        "  Q / Esc : quit\n"
        f"  accel={args.accel}/s  decel={args.decel}/s  "
        f"max={args.max_speed}  dt={dt}s\n"
    )

    try:
        while not controller.quit_requested:
            if controller.reset_requested:
                controller.reset_requested = False
                controller.reset()
            if controller.save_requested:
                controller.save_requested = False
            speed = controller.update(dt)
            bar = speed_bar(speed, args.max_speed)
            line = f"\r  speed={speed:+7.3f}  {bar}  "
            sys.stdout.write(line)
            sys.stdout.flush()
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        controller.stop()
        print("\nDone.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test gradual Left/Right speed control for MountainCarContinuous."
    )
    parser.add_argument(
        "--max-speed",
        type=float,
        default=1.0,
        help="Max |speed| / action magnitude (default: 1.0).",
    )
    parser.add_argument(
        "--accel",
        type=float,
        default=1.5,
        help="Acceleration while holding Left/Right, units per second.",
    )
    parser.add_argument(
        "--decel",
        type=float,
        default=2.0,
        help="Coast-down rate toward 0 when no key is held, units per second.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.05,
        help="Simulation step size in seconds.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
