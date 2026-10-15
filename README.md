# ENVE 3120 Fluid Mechanics — Computer Labs 4 and 5

Two guided Python notebooks for Fall 2026. Run the experiments in a browser with Binder, or install the same environment on your laptop. Each notebook includes complete input tables, worked equations, redrawn schematics, interactive controls, progress logs, conservation checks, and downloadable results.

## Open a lab in Binder

| Experiment | Launch | Preview |
|---|---|---|
| **Lab 4: Three reservoirs** — branch flows, local losses, hydraulic and energy grade lines, pressure screening | [![Launch Lab 4 in Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/UConn-EFC/Fall_2026_ENVE_3120_Fluid_Mechanics_Lab4_Lab5/main?urlpath=%2Fdoc%2Ftree%2FLab_4_Three_Reservoirs.ipynb) | [Read Lab 4](Lab_4_Three_Reservoirs.ipynb) |
| **Lab 5: Hardy–Cross network balancing** — continuity, loop corrections, reservoir exchange, pipe sizing | [![Launch Lab 5 in Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/UConn-EFC/Fall_2026_ENVE_3120_Fluid_Mechanics_Lab4_Lab5/main?urlpath=%2Fdoc%2Ftree%2FLab_5_Hardy_Cross.ipynb) | [Read Lab 5](Lab_5_Hardy_Cross.ipynb) |

These notebook links were generated with the form at [mybinder.org](https://mybinder.org/). They use the repository’s published `main` branch. You can also [launch the full JupyterLab workspace](https://mybinder.org/v2/gh/UConn-EFC/Fall_2026_ENVE_3120_Fluid_Mechanics_Lab4_Lab5/main?urlpath=lab).

## Student quick start

1. Select the badge for your lab. Allow the environment to build and the Python kernel to start. The first launch can take several minutes.
2. Use **Kernel → Restart Kernel and Run All Cells** for the complete example. Then work through the explanations and experiments in order. **Shift + Enter** runs one cell and advances.
3. Make predictions before inspecting results. Edit cells marked **YOUR INPUTS**, enter written responses in the indicated Markdown cells, and change one model input at a time.
4. Read the logs. `INFO` explains progress; `WARNING` identifies an assumption or a result requiring interpretation. A successful computation must also satisfy the displayed conservation checks.
5. Try the interactive panels. After an input changes, compute again before using its results. The panel marks earlier results stale and disables saving until a new run succeeds.
6. Save and download your edited notebook and result ZIPs **before closing Binder**. Download links appear in the export cells. You can download the notebook from JupyterLab’s file browser using **right-click → Download**. Binder sessions are temporary; saving inside Binder does not save to your laptop or update GitHub.

The GitHub previews retain baseline tables and figures. Interactive controls activate when their cells run in Binder or JupyterLab. If controls are unavailable, the ordinary input and calculation cells still perform the experiments.

## What each experiment teaches

**Lab 4** follows a Properties → Compute → Results → Save workflow, with a table mapping all 13 steps of the assigned WaterCAD/WaterGEMS exercise to notebook actions. Students inspect the loss ledger, distinguish elevation from hydraulic and total head, reconstruct pressure profiles, calibrate roughness, vary fitting losses, and raise the middle reservoir until its flow reverses.

**Lab 5** builds a continuity-balanced starting flow, selects preliminary diameters, works one loop correction by hand, and runs sequential Hardy–Cross corrections. Two geometric loops and a third reservoir-path condition determine the flow distribution. A simultaneous flow-and-head solution checks the result independently. Students compare demand cases and diameter choices, then inspect individual pipes with the profile explorer.

### Model assumptions to discuss in reports

- **Lab 4 local losses:** bend angles do not uniquely specify loss coefficients. Unspecified fitting coefficients are labelled, editable teaching assumptions. The model uses an idealized common total head at the central junction and reports pipe-side pressures. It does not resolve pressure minima inside fittings or reproduce every commercial-software convention.
- **Lab 5 demand:** the assigned node table totals **78.10 L/s**; the population-based daily average is **52.0833 L/s**. Both cases are supplied. Their ratio is not silently treated as an assigned peak factor.
- **Lab 5 head model:** the friction-dominated nodal equations omit local losses and junction velocity-head differences. Pipe-side EGL adds velocity head to HGL; it does not resolve reservoir entrances or junction transitions.
- **Design criteria:** the diameter catalog and pressure/speed screens are disclosed teaching choices. Conservation and convergence verify the stated equations; they do not establish an optimized or fully validated municipal design.

## Run on your laptop

Use **Python 3.12**. From the repository root, create a virtual environment, activate it, and install the pinned direct dependencies:

```bash
python -m venv .venv
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyterlab
```

Binder uses `environment.yml` to select Python 3.12 and install the same requirements. The computational cells use packaged inputs and do not download external datasets.

## Save and reproduce a case

Exports include exact JSON inputs, CSV results, conservation checks, PNG/SVG figures, and a manifest. The corresponding lab’s session log may contain several scenarios; the JSON inputs identify the exported case. Repeated exports do not pick up stale figures from earlier runs.

After extracting an archive, use `load_saved_inputs(...)` and the appropriate solver as illustrated in each notebook. Keep the edited notebook as well: written responses and your experiment code are part of the work. Generated results belong in `outputs/` and are excluded from version control.

## Validation

Three review passes checked physical equations and inputs; plot clarity, mathematical rendering, and interaction; then fresh-kernel execution and export reproducibility. The numerical suite passed **110 checks**, including mass and energy conservation, independent-solver agreement, flow reversal, invalid inputs, and reproduction from saved configurations. The notebooks were also tested on Windows and Linux, and the interactive controls were exercised in JupyterLab.

These checks apply to the stated teaching models. Read each experiment’s assumptions when interpreting pressures, fitting losses, and design suitability.

## References

Colebrook, C. F. (1939). Turbulent flow in pipes, with particular reference to the transition region between the smooth and rough pipe laws. *Journal of the Institution of Civil Engineers, 11*(4), 133–156. https://doi.org/10.1680/ijoti.1939.13150

Cross, H. (1936). *Analysis of flow in networks of conduits or conductors* (Engineering Experiment Station Bulletin No. 286). University of Illinois. https://hdl.handle.net/2142/4433

Swamee, P. K., & Jain, A. K. (1976). Explicit equations for pipe-flow problems. *Journal of the Hydraulics Division, 102*(5), 657–664. https://doi.org/10.1061/JYCEAJ.0004542

These papers support the friction and network methods; assigned dimensions and explicitly labelled teaching assumptions are not attributed to them.
