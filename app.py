# -*- coding: utf-8 -*-
"""Simple desktop application (PySide6) for the localizable bottom-up
household load model.

A thin layer over ``model_runner`` -- controls on the left, plot on the
right. The simulation runs in a separate thread so the window stays responsive.

Run:        python app.py
Smoke test: python app.py --smoke     (no window, verifies build + rendering)
"""
from __future__ import annotations

import os
import sys

# Make sure matplotlib uses the PySide6 binding.
os.environ.setdefault("QT_API", "pyside6")

import numpy as np
import pandas as pd

from PySide6.QtCore import Qt, QObject, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QComboBox, QSpinBox,
    QDoubleSpinBox, QCheckBox, QPushButton, QGroupBox, QFormLayout,
    QVBoxLayout, QHBoxLayout, QProgressBar, QFileDialog, QMessageBox,
    QScrollArea,
)

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import household_simulation as sd
from model_runner import (
    RunConfig, run, INTERVALS, DAY_TYPES, CLASS_MIX, HORIZONS,
    SimulationCancelled,
)

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]

GROUP_LABELS = {
    "Kitchen": "Kitchen", "Cleaning": "Laundry & washing",
    "Entertainment_Work": "Entertainment & work", "Personal_Care": "Personal care",
    "Always_On": "Always on", "Climate_Control": "Climate control",
    "Lighting": "Lighting",
}
GROUP_COLORS = {
    "Kitchen": "#e69f00", "Cleaning": "#0072b2",
    "Entertainment_Work": "#9467bd", "Personal_Care": "#e377c2",
    "Always_On": "#009e73", "Climate_Control": "#d55e00",
    "Lighting": "#f0e442",
}
APPLIANCE_LABELS = {
    "kettle": "Kettle", "toaster": "Toaster", "oven": "Oven",
    "hob": "Hob", "microwave": "Microwave",
    "vacuum": "Vacuum cleaner", "washing_machine": "Washing machine",
    "dryer": "Dryer (heat pump)",
    "dishwasher": "Dishwasher", "iron": "Iron",
    "tv": "Television", "pc": "PC / printer",
    "hair_dryer": "Hair dryer",
    "refrigerator": "Refrigerator", "freezer": "Freezer",
    "router": "Router", "small_appliances": "Small appliances",
    "boiler": "Boiler (DHW)", "heating_v2": "Heating", "cooling_v2": "Cooling",
}


# ---------------------------------------------------------------------------
# Compute thread
# ---------------------------------------------------------------------------
class Worker(QObject):
    progress = Signal(int, str)
    finished = Signal(dict)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, cfg: RunConfig):
        super().__init__()
        self.cfg = cfg
        self._stop = False

    def request_stop(self):
        """Called from the GUI thread — flips the flag the worker polls."""
        self._stop = True

    @Slot()
    def run(self):
        try:
            res = run(self.cfg,
                      progress=lambda p, m: self.progress.emit(p, m),
                      stop_check=lambda: self._stop)
            self.finished.emit(res)
        except SimulationCancelled:
            self.cancelled.emit()
        except Exception as e:  # noqa: BLE001
            import traceback
            self.failed.emit(f"{e}\n\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Household load model")
        self.resize(1120, 680)
        self._result = None
        self._thread = None
        self._worker = None

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.addWidget(self._build_controls(), 0)
        root.addWidget(self._build_plot(), 1)

    # ---- left panel ----------------------------------------------------
    def _build_controls(self) -> QWidget:
        panel = QWidget()
        panel.setMaximumWidth(340)
        lay = QVBoxLayout(panel)

        # Basic parameters
        g_base = QGroupBox("Basic parameters")
        f = QFormLayout(g_base)
        self.cmb_month = QComboBox(); self.cmb_month.addItems(MONTHS)
        self.cmb_month.setCurrentIndex(1)  # February (as in the validation)
        self.spn_hh = QSpinBox(); self.spn_hh.setRange(1, 400); self.spn_hh.setValue(50)
        self.cmb_interval = QComboBox(); self.cmb_interval.addItems(INTERVALS.keys())
        self.cmb_interval.setCurrentText("10 min")
        self.cmb_day = QComboBox(); self.cmb_day.addItems(DAY_TYPES.keys())
        self.cmb_horizon = QComboBox(); self.cmb_horizon.addItems(HORIZONS.keys())
        self.cmb_horizon.setCurrentText("Day")
        self.spn_iter = QSpinBox(); self.spn_iter.setRange(1, 50); self.spn_iter.setValue(5)
        self.cmb_class = QComboBox(); self.cmb_class.addItems(CLASS_MIX.keys())
        f.addRow("Month:", self.cmb_month)
        f.addRow("Number of households:", self.spn_hh)
        f.addRow("Resolution:", self.cmb_interval)
        f.addRow("Day type:", self.cmb_day)
        f.addRow("Horizon:", self.cmb_horizon)
        f.addRow("Iterations:", self.spn_iter)
        f.addRow("Equipment:", self.cmb_class)
        lay.addWidget(g_base)

        # Paper features
        g_feat = QGroupBox("Features")
        vf = QVBoxLayout(g_feat)
        self.chk_reactive = QCheckBox("Reactive power Q (per-appliance cos φ)")
        self.chk_reactive.setChecked(True)
        self.chk_local = QCheckBox("Localization by the national profile (TDD4)")
        self.chk_local.setChecked(True)
        self.chk_eff = QCheckBox("Efficiency scenario (most efficient appliances)")
        self.chk_dishwasher_hw = QCheckBox("Dishwasher connected to hot water (DHW)")
        self.chk_pv = QCheckBox("Photovoltaics (pvlib / PVGIS)")
        self.chk_pv.toggled.connect(self._toggle_pv)
        for w in (self.chk_reactive, self.chk_local, self.chk_eff,
                  self.chk_dishwasher_hw, self.chk_pv):
            vf.addWidget(w)
        lay.addWidget(g_feat)

        lay.addWidget(self._build_appliances())

        # PV parameters
        self.g_pv = QGroupBox("PV parameters")
        pf = QFormLayout(self.g_pv)
        self.spn_lat = QDoubleSpinBox(); self.spn_lat.setRange(-90, 90); self.spn_lat.setDecimals(3); self.spn_lat.setValue(49.193)
        self.spn_lon = QDoubleSpinBox(); self.spn_lon.setRange(-180, 180); self.spn_lon.setDecimals(3); self.spn_lon.setValue(16.612)
        self.spn_kwp = QDoubleSpinBox(); self.spn_kwp.setRange(0.1, 1000); self.spn_kwp.setValue(5.0)
        self.spn_tilt = QDoubleSpinBox(); self.spn_tilt.setRange(0, 90); self.spn_tilt.setValue(35.0)
        self.spn_az = QDoubleSpinBox(); self.spn_az.setRange(0, 360); self.spn_az.setValue(180.0)
        pf.addRow("Latitude (°):", self.spn_lat)
        pf.addRow("Longitude (°):", self.spn_lon)
        pf.addRow("Power (kWp):", self.spn_kwp)
        pf.addRow("Tilt (°):", self.spn_tilt)
        pf.addRow("Azimuth (°, 180 = south):", self.spn_az)
        self.g_pv.setEnabled(False)
        lay.addWidget(self.g_pv)

        # Display
        g_view = QGroupBox("Display")
        vv = QFormLayout(g_view)
        self.cmb_break = QComboBox(); self.cmb_break.addItems(["Total", "Groups", "Appliances"])
        self.cmb_break.setCurrentText("Groups")
        vv.addRow("Breakdown:", self.cmb_break)
        lay.addWidget(g_view)

        # Actions
        self.btn_run = QPushButton("▶ Run")
        self.btn_run.clicked.connect(self.on_run)
        self.btn_stop = QPushButton("■ Stop")
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_stop.setEnabled(False)
        self.btn_save = QPushButton("💾 Save CSV")
        self.btn_save.clicked.connect(self.on_save)
        self.btn_save.setEnabled(False)
        row_actions = QHBoxLayout()
        row_actions.addWidget(self.btn_run, 1)
        row_actions.addWidget(self.btn_stop, 1)
        lay.addLayout(row_actions)
        lay.addWidget(self.btn_save)

        self.progress = QProgressBar(); self.progress.setRange(0, 100)
        self.status = QLabel("Ready.")
        self.status.setWordWrap(True)
        lay.addWidget(self.progress)
        lay.addWidget(self.status)
        lay.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        scroll.setMaximumWidth(360)
        return scroll

    # ---- appliance selection -------------------------------------------
    def _build_appliances(self) -> QGroupBox:
        g = QGroupBox("Appliances")
        v = QVBoxLayout(g)
        self.appl_checks = {}
        self.group_checks = {}
        self._group_children = {}
        for grp in sd.GROUP_KEYS:
            keys = [k for k in sd.APPLIANCE_KEYS
                    if k in sd.APPLIANCE_GROUPS[grp] and k in APPLIANCE_LABELS]
            if not keys:
                continue
            gc = QCheckBox(GROUP_LABELS.get(grp, grp))
            gc.setTristate(True)
            fnt = gc.font(); fnt.setBold(True); gc.setFont(fnt)
            gc.clicked.connect(lambda _=False, gg=grp: self._on_group_clicked(gg))
            v.addWidget(gc)
            self.group_checks[grp] = gc
            self._group_children[grp] = keys
            on = grp != "Climate_Control"
            for k in keys:
                cb = QCheckBox("   " + APPLIANCE_LABELS[k])
                cb.setChecked(on)
                cb.toggled.connect(lambda _=False, gg=grp: self._sync_group(gg))
                v.addWidget(cb)
                self.appl_checks[k] = cb
            self._sync_group(grp)
        return g

    def _on_group_clicked(self, grp):
        keys = self._group_children[grp]
        target = not all(self.appl_checks[k].isChecked() for k in keys)
        for k in keys:
            self.appl_checks[k].setChecked(target)
        self._sync_group(grp)

    def _sync_group(self, grp):
        states = [self.appl_checks[k].isChecked() for k in self._group_children[grp]]
        gc = self.group_checks[grp]
        gc.blockSignals(True)
        gc.setCheckState(Qt.Checked if all(states)
                         else Qt.PartiallyChecked if any(states) else Qt.Unchecked)
        gc.blockSignals(False)

    # ---- right panel ---------------------------------------------------
    def _build_plot(self) -> QWidget:
        wrap = QWidget()
        v = QVBoxLayout(wrap)
        self.fig = Figure(figsize=(7, 6.5), tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        # Two stacked axes: top = per-CP breakdown, bottom = feeder aggregate.
        self.ax = self.fig.add_subplot(2, 1, 1)
        self.ax_feeder = self.fig.add_subplot(2, 1, 2, sharex=self.ax)
        self.ax_q = None
        self.ax_feeder_q = None
        v.addWidget(NavigationToolbar(self.canvas, wrap))
        v.addWidget(self.canvas, 1)
        self._blank_plot()
        return wrap

    def _blank_plot(self):
        self.ax.clear()
        self.ax.set_xlabel("Hour")
        self.ax.set_ylabel("Power (W/CP)")
        self.ax.set_xlim(0, 24)
        self.ax.text(0.5, 0.5, "Set the parameters and run the simulation.",
                     ha="center", va="center", transform=self.ax.transAxes,
                     color="gray")
        self.ax_feeder.clear()
        self.ax_feeder.set_xlabel("Hour")
        self.ax_feeder.set_ylabel("Feeder power (kW)")
        self.ax_feeder.set_xlim(0, 24)
        self.canvas.draw_idle()

    def _toggle_pv(self, on: bool):
        self.g_pv.setEnabled(on)

    # ---- run -----------------------------------------------------------
    def _build_config(self) -> RunConfig:
        ea = {k for k, cb in self.appl_checks.items() if cb.isChecked()}
        if ea & set(sd.APPLIANCE_GROUPS["Kitchen"]):
            ea.add("food_prep")  # scheduler for kitchen appliances
        return RunConfig(
            month=self.cmb_month.currentIndex() + 1,
            n_households=self.spn_hh.value(),
            interval_seconds=INTERVALS[self.cmb_interval.currentText()],
            period_type=DAY_TYPES[self.cmb_day.currentText()],
            horizon=HORIZONS[self.cmb_horizon.currentText()],
            iterations=self.spn_iter.value(),
            equip_class=self.cmb_class.currentText(),
            enabled_appliances=ea,
            reactive=self.chk_reactive.isChecked(),
            localize=self.chk_local.isChecked(),
            efficiency=self.chk_eff.isChecked(),
            dishwasher_hot_water=self.chk_dishwasher_hw.isChecked(),
            pv=self.chk_pv.isChecked(),
            pv_params={
                "latitude": self.spn_lat.value(), "longitude": self.spn_lon.value(),
                "kwp": self.spn_kwp.value(), "tilt": self.spn_tilt.value(),
                "azimuth": self.spn_az.value(),
            },
        )

    def on_run(self):
        cfg = self._build_config()
        if not cfg.enabled_appliances:
            QMessageBox.warning(self, "No appliances",
                                "Select at least one appliance.")
            return
        self.btn_run.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress.setValue(0)
        self._thread = QThread()
        self._worker = Worker(cfg)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.on_progress)
        self._worker.finished.connect(self.on_finished)
        self._worker.failed.connect(self.on_failed)
        self._worker.cancelled.connect(self.on_cancelled)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._worker.cancelled.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def on_stop(self):
        if self._worker is not None:
            self._worker.request_stop()
            self.btn_stop.setEnabled(False)
            self.status.setText("Stop requested…")

    @Slot(int, str)
    def on_progress(self, pct: int, msg: str):
        self.progress.setValue(pct)
        self.status.setText(msg)

    @Slot(dict)
    def on_finished(self, res: dict):
        self._result = res
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_save.setEnabled(True)
        self.plot(res)
        N = res["meta"]["n_households"]
        self.status.setText(
            f"Done — mean {res['total'].mean() / N:.0f} W/CP, "
            f"peak {res['total'].max() / N:.0f} W/CP"
            + (f", TDD factor {res['meta']['localize_factor']:.3f}"
               if res['meta']['localize'] else "")
        )

    @Slot(str)
    def on_failed(self, err: str):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.status.setText("Simulation error.")
        QMessageBox.critical(self, "Error", err)

    @Slot()
    def on_cancelled(self):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.status.setText("Simulation cancelled.")

    # ---- plotting ------------------------------------------------------
    def plot(self, res: dict):
        N = res["meta"]["n_households"]
        t = res["t"]
        avg_per_cp = res["total"] / N
        single_cp = res.get("single_cp")
        mode = self.cmb_break.currentText()

        # ---- top panel: single-CP raw profile -----------------------------
        # Shows one representative household from the last iteration -- keeps
        # the raw appliance switching visible. Averaged per-CP would be
        # identical in shape to the feeder aggregate below (just rescaled),
        # so we plot the raw single-CP instead.
        self.ax.clear()
        if self.ax_q is not None:
            self.ax_q.remove()
            self.ax_q = None

        single_appl = res.get("single_cp_appliances") or {}

        # Groups from single-CP appliance breakdown (same GROUP_KEYS mapping
        # as the feeder aggregate). Only populated when we actually have the
        # per-appliance decomposition of the tracked household.
        single_groups: dict = {}
        for g, keys in sd.APPLIANCE_GROUPS.items():
            arrs = [single_appl[k] for k in keys if k in single_appl]
            if arrs:
                single_groups[g] = np.sum(arrs, axis=0)

        if mode == "Groups" and single_groups:
            bottom = np.zeros_like(t)
            for g in sd.GROUP_KEYS:
                if g not in single_groups:
                    continue
                y = single_groups[g]
                if y.max() <= 0:
                    continue
                self.ax.fill_between(t, bottom, bottom + y, step="mid",
                                     label=GROUP_LABELS.get(g, g),
                                     color=GROUP_COLORS.get(g), alpha=0.85)
                bottom = bottom + y
            self.ax.plot(t, single_cp, "k-", lw=1.0, label="Single CP total")
        elif mode == "Appliances" and single_appl:
            cmap = matplotlib.colormaps["tab20"]
            bottom = np.zeros_like(t)
            items = [(k, v) for k, v in single_appl.items() if np.asarray(v).max() > 0]
            for i, (k, v) in enumerate(items):
                y = np.asarray(v)
                self.ax.fill_between(t, bottom, bottom + y, step="mid",
                                     label=k, color=cmap(i % 20), alpha=0.85)
                bottom = bottom + y
            self.ax.plot(t, single_cp, "k-", lw=1.0, label="Single CP total")
        elif single_cp is not None:
            self.ax.plot(t, single_cp, "-", color="#009e73", lw=1.0,
                         label="Single CP active P (W)")
        else:
            self.ax.plot(t, avg_per_cp, "-", color="#009e73", lw=1.6,
                         label="Feeder average / CP")

        # Reactive power on the secondary axis: the single-CP curve
        # computed from the same household's own appliance breakdown
        # (bottom-up sum P_i * tan phi_i). If unavailable, fall back to
        # the per-CP average.
        q_single = res.get("reactive_single_cp")
        if q_single is not None:
            self.ax_q = self.ax.twinx()
            self.ax_q.plot(t, q_single, ":", color="#555555", lw=1.6,
                           label="Single CP reactive Q (var)")
            self.ax_q.set_ylabel("Reactive power (var)")
            self.ax_q.legend(loc="upper right", fontsize=7)
        elif res["reactive"] is not None:
            self.ax_q = self.ax.twinx()
            self.ax_q.plot(t, res["reactive"] / N, ":", color="#555555", lw=1.6,
                           label="Reactive Q / CP (feeder average)")
            self.ax_q.set_ylabel("Reactive power (var/CP)")
            self.ax_q.legend(loc="upper right", fontsize=7)

        self.ax.set_xlabel("Hour")
        self.ax.set_ylabel("Power (W/CP)")
        n_days = res["meta"].get("n_days", 1)
        horizon_label = f", horizon {n_days} day{'s' if n_days > 1 else ''}" if n_days > 1 else ""
        self.ax.set_title(f"Single connection point (representative household){horizon_label}",
                          fontsize=9)
        self.ax.set_xlim(0, t[-1] if len(t) else 24)
        self.ax.set_ylim(bottom=0)
        self.ax.legend(loc="upper left", fontsize=7)

        # ---- bottom panel: feeder aggregate (kW, kvar) -------------------
        self.ax_feeder.clear()
        if self.ax_feeder_q is not None:
            self.ax_feeder_q.remove()
            self.ax_feeder_q = None

        p_kw = res["total"] / 1000.0
        if mode == "Groups" and res["groups"]:
            bottom = np.zeros_like(p_kw)
            for g in sd.GROUP_KEYS:
                if g not in res["groups"]:
                    continue
                y = res["groups"][g] / 1000.0
                if y.max() <= 0:
                    continue
                self.ax_feeder.fill_between(t, bottom, bottom + y, step="mid",
                                            label=GROUP_LABELS.get(g, g),
                                            color=GROUP_COLORS.get(g), alpha=0.85)
                bottom = bottom + y
            self.ax_feeder.plot(t, p_kw, "k-", lw=1.1, label="Feeder total")
        elif mode == "Appliances" and res["appliances"]:
            cmap = matplotlib.colormaps["tab20"]
            bottom = np.zeros_like(p_kw)
            items = [(k, v) for k, v in res["appliances"].items() if np.asarray(v).max() > 0]
            for i, (k, v) in enumerate(items):
                y = np.asarray(v) / 1000.0
                self.ax_feeder.fill_between(t, bottom, bottom + y, step="mid",
                                            label=k, color=cmap(i % 20), alpha=0.85)
                bottom = bottom + y
            self.ax_feeder.plot(t, p_kw, "k-", lw=1.1, label="Feeder total")
        else:
            self.ax_feeder.plot(t, p_kw, "-", color="#009e73", lw=1.6,
                                label="Feeder active P (kW)")

        if res["pv"] is not None:
            self.ax_feeder.axhline(0, color="black", lw=0.6)
            self.ax_feeder.fill_between(t, 0.0, -res["pv"] / 1000.0, step="mid",
                                        color="#f0c000", alpha=0.40,
                                        label="PV generation (−)")
            self.ax_feeder.plot(t, res["net"] / 1000.0, "-", color="#0072b2",
                                lw=1.8, label="Net grid draw (kW)")
        if res["reactive"] is not None:
            self.ax_feeder_q = self.ax_feeder.twinx()
            self.ax_feeder_q.plot(t, res["reactive"] / 1000.0, ":",
                                  color="#555555", lw=1.6,
                                  label="Feeder reactive Q (kvar)")
            self.ax_feeder_q.set_ylabel("Reactive power (kvar)")
            self.ax_feeder_q.legend(loc="upper right", fontsize=7)

        self.ax_feeder.set_xlabel("Hour")
        self.ax_feeder.set_ylabel("Feeder power (kW)")
        self.ax_feeder.set_title(f"Feeder aggregate (sum of {N} CPs){horizon_label}",
                                 fontsize=9)
        self.ax_feeder.set_xlim(0, t[-1] if len(t) else 24)
        if res["pv"] is None:
            self.ax_feeder.set_ylim(bottom=0)
        self.ax_feeder.legend(loc="upper left", fontsize=7, ncol=2)

        self.canvas.draw_idle()

    # ---- export --------------------------------------------------------
    def on_save(self):
        if self._result is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "output.csv",
                                              "CSV (*.csv)")
        if not path:
            return
        res = self._result
        df = pd.DataFrame({"hour": res["t"], "P_total_W": res["total"]})
        for g, arr in res["groups"].items():
            df[f"P_{g}_W"] = arr
        for k, arr in res["appliances"].items():
            df[f"A_{k}_W"] = arr
        if res["reactive"] is not None:
            df["Q_total_var"] = res["reactive"]
            # per-appliance reactive-power breakdown (Hannagan et al. 2023
            # cos φ; appliances with cos φ = 1 have Q ≡ 0 and are omitted).
            for k, arr in res.get("reactive_appliances", {}).items():
                df[f"Q_{k}_var"] = arr
        if res["pv"] is not None:
            df["PV_W"] = res["pv"]
            df["net_load_W"] = res["net"]
        df.to_csv(path, index=False)
        self.status.setText(f"Saved: {path}")


def _smoke():
    """Build the window and perform one run + render without the event loop."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow()
    cfg = win._build_config()
    cfg.iterations = 1
    cfg.n_households = 5
    cfg.pv = False
    res = run(cfg)
    win.plot(res)
    win.on_save  # reference, we do not save
    print("SMOKE OK: total mean W/CP =",
          round(float(res["total"].mean()) / cfg.n_households, 1))
    return 0


def main():
    if "--smoke" in sys.argv:
        sys.exit(_smoke())
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
