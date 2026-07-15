# Mountain Car data collection

Collect keyboard demonstrations for [MountainCarContinuous-v0](https://gymnasium.farama.org/environments/classic_control/mountain_car_continuous/) and save them as a D4RL-style HDF5 dataset for OfflineRL-Kit (e.g. `run_example/run_combo.py`).

## Requirements

- `gymnasium`
- `pygame` (human rendering)
- `pynput` (keyboard)
- `h5py`, `numpy`

These are installed in the OfflineRL-Kit Docker image under the Data Collection section.

## Collect demos

```bash
python play.py
# recommended throttle feel (faster ramp / coast)
python play.py --accel 2.0 --decel 3.0
# optional
python play.py --output ../demostrations/mountain_car_human.hdf5 --seed 0 --step-delay 0.05
python play.py --accel 2.0 --decel 3.0 --max-speed 1.0
```

Default output: `data_colllection/demostrations/mountain_car_human.hdf5`  
If that file already exists, new episodes are **appended**.

Throttle uses `SpeedController` (`controll.py`): force ramps gradually while ←/→ are held and coasts toward 0 when released. A live force bar is shown in the terminal.

### Controls

| Key | Action |
| --- | --- |
| ← | Ramp force toward `-1.0` |
| → | Ramp force toward `+1.0` |
| (neither) | Coast force toward `0.0` |
| `R` | Abort current episode and reset |
| `S` | Save mid-session |
| `Q` / `Esc` | Save and quit |

## Inspect trajectories

```bash
python play.py --view
python play.py --view --output ../demostrations/mountain_car_human.hdf5
```

Lists each trajectory with status:

- `success` — reached the goal (`terminated`)
- `timeout` — hit the 200-step limit (`truncated`)
- `incomplete` — no terminal/timeout (e.g. aborted with `R`, or quit mid-episode)

## Delete trajectories by status

```bash
python play.py --delete incomplete
python play.py --delete timeout
python play.py --delete incomplete timeout
```

Allowed statuses: `success`, `timeout`, `incomplete`. Matching trajectories are removed and the HDF5 is rewritten.

## Dataset format

Default file: `../demostrations/mountain_car_human.hdf5`

| Key | Shape | Notes |
| --- | --- | --- |
| `observations` | `(N, 2)` | position, velocity |
| `actions` | `(N, 1)` | continuous force in `[-1, 1]` as float32 |
| `rewards` | `(N,)` | typically `-1` per step |
| `terminals` | `(N,)` | goal reached |
| `timeouts` | `(N,)` | episode truncated |
| `next_observations` | `(N, 2)` | for dynamics / MBRL |

### Load into OfflineRL-Kit

```python
import h5py
from offlinerlkit.utils.load_dataset import qlearning_dataset

with h5py.File("data_colllection/demostrations/mountain_car_human.hdf5", "r") as f:
    raw = {k: f[k][:] for k in f.keys()}
dataset = qlearning_dataset(env, dataset=raw)
real_buffer.load_dataset(dataset)
```
