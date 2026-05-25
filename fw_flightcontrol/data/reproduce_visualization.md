# Reproduce Visualization Setup

Changes required to enable live telemetry plots (`plot_anim`) and 3D FlightGear
visualization (`fgear`) in `full_mppi_analysis.py`.

All edits are in the editable venv package at:
`{venv}/src/fw-jsbgym/fw_jsbgym/`

---

## 1. System installs

```bash
sudo apt install python3-tk     # required for plot_anim live telemetry window
sudo apt install flightgear     # required for fgear 3D visualization
```

---

## 2. `fw_jsbgym/visualizers/visualizer.py`

### a) Add `import sys` at the top

```python
import subprocess
import sys          # add this
import time
import matplotlib.pyplot as plt
from fw_jsbgym.simulation.jsb_simulation import Simulation
from pkg_resources import resource_filename
```

### b) Fix `PlotVisualizer` subprocess to use the venv Python

The subprocess was launching system `python3` (no matplotlib). Use `sys.executable` instead.

```python
# before
cmd = f"python {viz_plot_path} --tele-file {telemetry_file} --animate"
# after
cmd = f"{sys.executable} {viz_plot_path} --tele-file {telemetry_file} --animate"
```

### c) Replace `launch_flightgear()` with the following

Adds `_find_fgfs()` to locate FlightGear (apt binary first, AppImage fallback),
points FlightGear at the bundled x8 aircraft via `--fg-aircraft`, and uses the
actual aircraft id instead of hardcoded c172p.

```python
def _find_fgfs(self) -> str:
    """Return the FlightGear executable, preferring the system binary over the AppImage."""
    import shutil, os
    if shutil.which('fgfs'):
        return 'fgfs'
    appimage = os.path.expanduser(
        '$HOME/Apps/FlightGear-2020.3.17/FlightGear-2020.3.17-x86_64.AppImage'
    )
    if os.path.isfile(appimage):
        return appimage
    raise FileNotFoundError(
        "FlightGear not found. Install it with:  sudo apt install flightgear"
    )

def launch_flightgear(self, aircraft_fgear_id: str = 'x8') -> subprocess.Popen:
    fgfs = self._find_fgfs()
    # Point FlightGear at the bundled aircraft directory so the x8 model is found
    fg_aircraft_dir = resource_filename('fw_jsbgym', 'fgdata/Aircraft')
    cmd: str = (
        f'{fgfs} --fdm=null'
        f' --fg-aircraft={fg_aircraft_dir}'
        f' --native-fdm={self.TYPE},{self.DIRECTION},{self.RATE},{self.SERVER},{self.PORT},{self.PROTOCOL}'
        f' --aircraft={aircraft_fgear_id}'
        f' --timeofday={self.TIME}'
        f' --lat={self.START_LAT} --lon={self.START_LON} --altitude={self.START_ALT}'
        f' --disable-ai-traffic --disable-real-weather-fetch'
    )

    flightgear_process = subprocess.Popen(cmd,
                                          shell=True,
                                          stdout=subprocess.PIPE,
                                          stderr=subprocess.STDOUT)
    print("Started FlightGear process with PID: ", flightgear_process.pid)
    while True:
        out: str = flightgear_process.stdout.readline().decode()
        if self.LOADED_MESSAGE in out:
            print("FlightGear loaded successfully.")
            break
        else:
            print(out.strip())
    return flightgear_process
```

---

## 3. `fw_jsbgym/envs/jsbsim_env.py`

`options` is `None` by default and the original code crashed trying to iterate it.
Find the line (around line 295) inside `reset()`:

```python
# before
if "telemetry_file" in options:
# after
if options and "telemetry_file" in options:
```

---

## 4. `fw_jsbgym/fgdata/Aircraft/x8/x8-set.xml`

The default view 0 was `internal=true` (camera inside the drone body).
Replace the entire `<view n="0">` block with an external chase camera:

```xml
<view n="0">
    <name>Chase View</name>
    <type>lookfrom</type>
    <internal type="bool">false</internal>
    <config>
        <from-model type="bool">true</from-model>
        <from-model-idx type="int">0</from-model-idx>
        <x-offset-m type="double">0.0</x-offset-m>
        <y-offset-m type="double">2.0</y-offset-m>
        <z-offset-m type="double">6.0</z-offset-m>
        <pitch-offset-deg type="double">-10.0</pitch-offset-deg>
        <default-field-of-view-deg type="double">60.0</default-field-of-view-deg>
        <limits>
            <enabled type="bool">false</enabled>
        </limits>
    </config>
</view>
```

Offsets: 6 m behind (positive z = aft in FG), 2 m above, camera angled -10° down.
Press **V** in FlightGear to cycle views; scroll wheel adjusts FOV.

---

## 5. `full_mppi_analysis.py` (already in the repo)

This file was already updated — no changes needed on the new device.
To verify, `make_env()` should accept `render_mode` and `telemetry_file` parameters,
and `--render-mode` should be a recognised CLI argument.

---

## Usage

```bash
# live telemetry plot (matplotlib window per run)
python full_mppi_analysis.py --render-mode plot_anim --steps 500 --skip-env-model

# 3D FlightGear visualization (~15-30 s startup)
python full_mppi_analysis.py --render-mode fgear --steps 500 --skip-env-model
```
