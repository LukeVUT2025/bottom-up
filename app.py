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
from model_runner import RunConfig, run, INTERVALS, DAY_TYPES, CLASS_MIX

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

    def __init__(self, cfg: RunConfig):
        super().__init__()
        self.cfg = cfg

    @Slot()
    def run(self):
        try:
            res = run(self.cfg, progress=lambda p, m: self.progress.emit(p, m))
            self.finished.emit(res)
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
        self.spn_iter = QSpinBox(); self.spn_iter.setRange(1, 50); self.spn_iter.setValue(5)
        self.cmb_class = QComboBox(); self.cmb_class.addItems(CLASS_MIX.keys())
        f.addRow("Month:", self.cmb_month)
        f.addRow("Number of households:", self.spn_hh)
        f.addRow("Resolution:", self.cmb_interval)
        f.addRow("Day type:", self.cmb_day)
        f.addRow("Iterations:", self.spn_iter)
        f.addRow("Equipment:", self.cmb_class)
        lay.addWidget(g_base)

        # Paper features
        g_feat = QGroupBox("Features")
        vf = QVBoxLayout(g_feat)
        self.chk_reactive = QCheckBox("Reactive power Q₀ (capacitive baseline + motor term)")
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
        self.btn_save = QPushButton("💾 Save CSV")
        self.btn_save.clicked.connect(self.on_save)
        self.btn_save.setEnabled(False)
        lay.addWidget(self.btn_run)
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
        self.fig = Figure(figsize=(7, 5), tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax_q = None
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
        self.progress.setValue(0)
        self._thread = QThread()
        self._worker = Worker(cfg)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.on_progress)
        self._worker.finished.connect(self.on_finished)
        self._worker.failed.connect(self.on_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    @Slot(int, str)
    def on_progress(self, pct: int, msg: str):
        self.progress.setValue(pct)
        self.status.setText(msg)

    @Slot(dict)
    def on_finished(self, res: dict):
        self._result = res
        self.btn_run.setEnabled(True)
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
        self.status.setText("Simulation error.")
        QMessageBox.critical(self, "Error", err)

    # ---- plotting ------------------------------------------------------
    def plot(self, res: dict):
        N = res["meta"]["n_households"]
        t = res["t"]
        total = res["total"] / N
        mode = self.cmb_break.currentText()

        self.ax.clear()
        if self.ax_q is not None:
            self.ax_q.remove()
            self.ax_q = None

        if mode == "Groups" and res["groups"]:
            bottom = np.zeros_like(total)
            for g in sd.GROUP_KEYS:
                if g not in res["groups"]:
                    continue
                y = res["groups"][g] / N
                if y.max() <= 0:
                    continue
                self.ax.fill_between(t, bottom, bottom + y, step="mid",
                                     label=GROUP_LABELS.get(g, g),
                                     color=GROUP_COLORS.get(g), alpha=0.85)
                bottom = bottom + y
            self.ax.plot(t, total, "k-", lw=1.1, label="Total load")
        elif mode == "Appliances" and res["appliances"]:
            cmap = matplotlib.colormaps["tab20"]
            bottom = np.zeros_like(total)
            items = [(k, v) for k, v in res["appliances"].items() if np.asarray(v).max() > 0]
            for i, (k, v) in enumerate(items):
                y = np.asarray(v) / N
                self.ax.fill_between(t, bottom, bottom + y, step="mid",
                                     label=k, color=cmap(i % 20), alpha=0.85)
                bottom = bottom + y
            self.ax.plot(t, total, "k-", lw=1.1, label="Total load")
        else:
            self.ax.plot(t, total, "-", color="#009e73", lw=1.6, label="Load")

        # Photovoltaics: generation is drawn NEGATIVE (as a reduction in load).
        # Grid draw = load - generation (negative = export to the grid).
        if res["pv"] is not None:
            self.ax.axhline(0, color="black", lw=0.6)
            self.ax.fill_between(t, 0.0, -res["pv"] / N, step="mid",
                                 color="#f0c000", alpha=0.40, label="PV generation (−)")
            self.ax.plot(t, res["net"] / N, "-", color="#0072b2", lw=1.8,
                         label="Grid draw (load − generation)")

        # Reactive power on the secondary axis.
        if res["reactive"] is not None:
            self.ax_q = self.ax.twinx()
            self.ax_q.plot(t, res["reactive"] / N, ":", color="#555555", lw=1.6,
                           label="Reactive power Q₀")
            self.ax_q.set_ylabel("Reactive power (var/CP)")
            self.ax_q.legend(loc="upper right", fontsize=7)

        self.ax.set_xlabel("Hour")
        self.ax.set_ylabel("Power (W/CP)")
        self.ax.set_xlim(0, 24)
        if res["pv"] is None:
            self.ax.set_ylim(bottom=0)
        self.ax.legend(loc="upper left", fontsize=7, ncol=2)
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
            df["Q_var"] = res["reactive"]
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
