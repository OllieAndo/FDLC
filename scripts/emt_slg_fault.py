"""EMT single-line-to-ground fault on a two-bus feeder (VeraGrid).

Source (Thevenin equivalent generator) -- line -- Bus1 (resistive load + fault).
Phase A at Bus1 is shorted to ground at t=0.04 s and cleared at t=0.10 s.
The fault is a single-phase shunt resistor whose resistance is stepped by EMT events.

Next: swap the source template for a grid-following / grid-forming inverter
(VeraGridEngine.Templates.Emt.vsc_gfl_emt / vsc_gfm_emt) and compare waveforms.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import VeraGridEngine.api as vg
from VeraGridEngine.Devices.Branches.line import Line
from VeraGridEngine.Devices.Events.emt_event import EmtEvent
from VeraGridEngine.Devices.Injections.generator import Generator
from VeraGridEngine.Devices.Injections.load import Load
from VeraGridEngine.Devices.Substation.bus import Bus
from VeraGridEngine.enumerations import DynamicIntegrationMethod, EmtInitializationMethod, EmtSolverTypes
from VeraGridEngine.Simulations.EMT.emt_driver import EmtSimulationDriver
from VeraGridEngine.Simulations.EMT.emt_options import EmtOptions
from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowOptions
from VeraGridEngine.Templates.Emt.load_RLC_emt_template import get_shunt_r_emt_template
from VeraGridEngine.Templates.Emt.pi_line_emt_template import get_pi_line_emt_template
from VeraGridEngine.Templates.Emt.thevenin_equivalent_emt_generator_template import (
    get_generator_thevenin_rl_emt_template_with_ref,
)
from VeraGridEngine.Utils.Symbolic.templates_common_functions import set_emt_model

from fdlc.features import phasor, sequence_components

F_BASE = 50.0
T_FAULT, T_CLEAR, T_END = 0.04, 0.10, 0.14
R_OPEN, R_FAULT = 1e6, 1e-3  # p.u.
DT = 2e-5


def build_grid() -> tuple[vg.MultiCircuit, dict]:
    grid = vg.MultiCircuit(Sbase=100.0, fbase=F_BASE)
    bus0 = Bus(name="Bus0", Vnom=10.0, is_slack=True)
    bus1 = Bus(name="Bus1", Vnom=10.0)
    grid.add_bus(bus0)
    grid.add_bus(bus1)

    line = Line(name="Line 0-1", bus_from=bus0, bus_to=bus1, r=0.02, x=0.08, b=0.0, rate=100.0)
    grid.add_line(line)
    gen = Generator(name="Gen1", P=20.0, vset=1.0, Snom=100.0, r1=0.001, x1=1.7)
    grid.add_generator(bus=bus0, api_obj=gen)
    load = Load(name="Load1", P=20.0, Q=0.0)
    grid.add_load(bus=bus1, api_obj=load)
    fault = Load(name="Fault", P=0.0, Q=0.0)
    grid.add_load(bus=bus1, api_obj=fault)

    vf = grid.var_factory
    set_emt_model(device=gen, model=get_generator_thevenin_rl_emt_template_with_ref(vf).block, var_factory=vf)
    set_emt_model(
        device=line,
        model=get_pi_line_emt_template(vf=vf, phN=False, phA=True, phB=True, phC=True).block,
        var_factory=vf,
    )
    set_emt_model(
        device=load, model=get_shunt_r_emt_template(vf=vf, phA=True, phB=True, phC=True).block, var_factory=vf
    )
    set_emt_model(
        device=fault,
        model=get_shunt_r_emt_template(vf=vf, phA=True, phB=False, phC=False, name="FaultR").block,
        var_factory=vf,
    )
    return grid, {"gen": gen, "bus1": bus1, "fault": fault}


def add_slg_fault(grid: vg.MultiCircuit, fault: Load) -> None:
    r_name = "R_A_FaultR_1ph"
    fault.emt_model.set_parameter_in_model(var_name=r_name, new_value=R_OPEN)
    r_var = next(v for b in fault.emt_model.get_all_blocks() for v in b.event_dict if v.name == r_name)

    group = vg.EmtEventsGroup(name="SLG phase A at Bus1")
    grid.add_emt_events_group(group)
    for t, r in ((T_FAULT, R_FAULT), (T_CLEAR, R_OPEN)):
        grid.add_emt_event(
            EmtEvent(device=fault, parameter=r_var, time=t, value=r, group=group, force_step_alignment=True)
        )


def run(grid: vg.MultiCircuit):
    pf = vg.power_flow(grid, options=PowerFlowOptions(solver_type=vg.SolverType.NR, verbose=0))
    options = EmtOptions(
        time_step=DT,
        simulation_time=T_END,
        tolerance=1e-4,
        solver_type=EmtSolverTypes.StructuralAD,
        integration_method=DynamicIntegrationMethod.DaeTrapezoidal,
        initialization_method=EmtInitializationMethod.Auto,
        verbose=0,
    )
    driver = EmtSimulationDriver(grid=grid, options=options, pf_results=pf)
    driver.run()
    res = driver.results
    if res is None or not (res.has_event_group_results[0] and res.converged[0] and res.well_initialized[0]):
        raise RuntimeError("EMT simulation did not converge / initialise")
    return res


def signals(res, device, names: list[str]) -> np.ndarray:
    by_name = {v.name: v for v in res.devices_vars_info[device]}
    cols = [res.uid2idx_vars[by_name[n].uid] for n in names]
    return res.values[:, cols, 0].T


def main() -> None:
    grid, dev = build_grid()
    add_slg_fault(grid, dev["fault"])
    res = run(grid)

    time_array = np.asarray(res.time_array)
    t = (time_array - time_array[0]) / np.timedelta64(1, "s")
    v_bus1 = signals(res, dev["bus1"], ["v_A", "v_B", "v_C"])
    i_gen = signals(res, dev["gen"], ["i_A", "i_B", "i_C"])
    i_fault = signals(res, dev["fault"], ["i_A"])[0]

    fs = 1.0 / DT
    for label, t_end in (("pre-fault", T_FAULT), ("during fault", T_CLEAR)):
        k = np.searchsorted(t, t_end)
        i0, i1, i2 = np.abs(sequence_components(phasor(i_gen[:, :k], fs, F_BASE)))
        print(f"{label:>13}: |I1|={i1:.3f} |I2|/|I1|={i2 / i1:.3f} |I0|/|I1|={i0 / i1:.3f}")

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 7))
    for ax, data, title in (
        (axes[0], v_bus1, "Bus1 phase voltages (p.u.)"),
        (axes[1], i_gen, "Source phase currents (p.u.)"),
    ):
        for k, ph in enumerate("ABC"):
            ax.plot(t, data[k], label=ph)
        ax.set_title(title)
        ax.legend(loc="upper right")
    axes[2].plot(t, i_fault, color="k")
    axes[2].set_title("Fault current, phase A (p.u.)")
    for ax in axes:
        ax.axvspan(T_FAULT, T_CLEAR, color="red", alpha=0.08)
    axes[2].set_xlabel("time (s)")
    fig.tight_layout()

    out = Path("results")
    out.mkdir(exist_ok=True)
    fig.savefig(out / "emt_slg_fault.png", dpi=150)
    np.savez(out / "emt_slg_fault.npz", t=t, v_bus1=v_bus1, i_gen=i_gen, i_fault=i_fault)
    print(f"saved {out / 'emt_slg_fault.png'}")


if __name__ == "__main__":
    main()
