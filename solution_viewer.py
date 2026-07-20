"""
solution_viewer.py  --  interactive tabbed viewer for Semiconductor-Solver
                        .npz results.

Layout (three sections):
    * top bar   : Load .npz | file label
    * left bar  : Display Mode (1D/2D/3D), Multiplot, Field, 1D cut controls,
                  Set cut points, log scale, 3D style, Plot
    * plot area : a Notebook of tabs, one plot per (field, mode); clicking Plot
                  opens a new tab, or re-selects/updates the existing tab.

The .npz is expected to hold (as written by DriftDiffusionSolver.save_bias_npz):
    nodes (N,2), elements (Ne,3), the per-node field arrays, and
    terminal_names / terminal_voltages describing the operating point.

3D rendering styles (inspired by a standalone TCAD-style visualizer):
    * "tcad"   : orthographic, flat box aspect, axes off, horizontal colorbar.
    * "normal" : perspective, z-axis labelled, side colorbar.

Run:  python3 solution_viewer.py       (needs Python 3.6+ with tkinter + Tk)
"""

import os
import numpy as np

# Only the plotting core is imported at module load, so the data/draw helpers
# stay importable headless.  Tkinter + the TkAgg backend are imported lazily
# inside run_gui() (they need a display).
from matplotlib.figure import Figure
from matplotlib.tri import Triangulation, LinearTriInterpolator
from matplotlib.ticker import AutoMinorLocator

UM = 1e6            # metres <-> micrometres
_TINY = 1e-30       # floor for log10

# .npz keys the band diagram / derived band energies need
_BAND_KEYS = ("potential", "electron_fermi_potential", "hole_fermi_potential",
              "ni", "Nc", "Nv", "VT")

# derived (computed) Field entries: band edges + intrinsic level [eV]
EC_LABEL = "conduction band (Ec)"
EV_LABEL = "valence band (Ev)"
EI_LABEL = "intrinsic level (Ei)"
EFN_LABEL = "electron quasi-Fermi level (EFn)"
EFP_LABEL = "hole quasi-Fermi level (EFp)"
DERIVED = (EC_LABEL, EV_LABEL, EI_LABEL, EFN_LABEL, EFP_LABEL)

# per-field metadata: key -> (title, latex symbol, latex unit, default log)
FIELD_INFO = {
    "potential":
        ("Electrostatic Potential", r"\psi", r"\mathrm{V}", False),
    "electron_fermi_potential":
        ("Electron Quasi-Fermi Potential", r"\phi_n", r"\mathrm{V}", False),
    "hole_fermi_potential":
        ("Hole Quasi-Fermi Potential", r"\phi_p", r"\mathrm{V}", False),
    "electron_concentration":
        ("Electron Concentration", "n", r"\mathrm{m^{-3}}", True),
    "hole_concentration":
        ("Hole Concentration", "p", r"\mathrm{m^{-3}}", True),
    "space_charge_density":
        ("Space Charge Density", r"\rho", r"\mathrm{C/m^{3}}", False),
    "electric_field_x":
        ("Electric Field $E_x$", "E_x", r"\mathrm{V/m}", False),
    "electric_field_y":
        ("Electric Field $E_y$", "E_y", r"\mathrm{V/m}", False),
    "electric_field_magnitude":
        ("Electric Field $|E|$", "|E|", r"\mathrm{V/m}", False),
    EC_LABEL:
        ("Conduction Band $E_c$", "E_c", r"\mathrm{eV}", False),
    EV_LABEL:
        ("Valence Band $E_v$", "E_v", r"\mathrm{eV}", False),
    EI_LABEL:
        ("Intrinsic Fermi Level $E_i$", "E_i", r"\mathrm{eV}", False),
    EFN_LABEL:
        ("Electron Quasi-Fermi Level $E_{Fn}$", "E_{Fn}", r"\mathrm{eV}", False),
    EFP_LABEL:
        ("Hole Quasi-Fermi Level $E_{Fp}$", "E_{Fp}", r"\mathrm{eV}", False),
    "current_density_magnitude": ("Current Density $|J|$", "|J|", r"\mathrm{A/m^{2}}", True),
    "current_density_x":  ("Current Density $J_x$", "J_x", r"\mathrm{A/m^{2}}", False),
    "current_density_y":  ("Current Density $J_y$", "J_y", r"\mathrm{A/m^{2}}", False),
    "electron_current_density_x": ("Electron $J_{n,x}$", "J_{n,x}", r"\mathrm{A/m^{2}}", False),
    "electron_current_density_y": ("Electron $J_{n,y}$", "J_{n,y}", r"\mathrm{A/m^{2}}", False),
    "hole_current_density_x": ("Hole $J_{p,x}$", "J_{p,x}", r"\mathrm{A/m^{2}}", False),
    "hole_current_density_y": ("Hole $J_{p,y}$", "J_{p,y}", r"\mathrm{A/m^{2}}", False),
}

def field_array(data, key):
    """Node array for a field: a stored one, or a derived band energy [eV].

    Derived (require potential + material scalars):
        E_i = -psi,  E_c = E_i + VT*ln(Nc/ni),  E_v = E_i - VT*ln(Nv/ni).
    """
    if key == EI_LABEL:
        return -np.asarray(data["potential"], float)
    if key == EFN_LABEL:
        return -np.asarray(data["electron_fermi_potential"], float)
    if key == EFP_LABEL:
        return -np.asarray(data["hole_fermi_potential"], float)
    if key in (EC_LABEL, EV_LABEL):
        ni = float(data["ni"]); Nc = float(data["Nc"])
        Nv = float(data["Nv"]); VT = float(data["VT"])
        Ei = -np.asarray(data["potential"], float)
        off = VT * np.log(Nc / ni) if key == EC_LABEL else -VT * np.log(Nv / ni)
        return Ei + off
    return np.asarray(data[key])


def field_title(key):
    return FIELD_INFO.get(key, (key, "", "", False))[0]


def field_default_log(key):
    return FIELD_INFO.get(key, (key, "", "", False))[3]


def field_zlabel(key, logscale):
    """LaTeX axis/colorbar label for a field, honouring the log flag."""
    _, sym, unit, _ = FIELD_INFO.get(key, (key, "", "", False))
    if not sym:
        return ("log10 " if logscale else "") + key
    body = r"\log_{10}(%s)" % sym if logscale else sym
    return r"$%s\;(%s)$" % (body, unit) if unit else r"$%s$" % body


# ===========================================================================
#  data helpers  (pure, GUI-independent, headless-testable)
# ===========================================================================
def plottable_fields(data):
    """Per-node (N,) arrays that can be plotted as a field."""
    n = data["nodes"].shape[0]
    return [k for k in data.files
            if data[k].ndim == 1 and data[k].shape[0] == n]


def build_triangulation(data):
    """Exact triangulation from stored node coords + connectivity."""
    x, y = data["nodes"].T
    if "elements" in data.files:
        return Triangulation(x, y, triangles=data["elements"])
    return Triangulation(x, y)                       # Delaunay fallback


def operating_point(data):
    """'name=+V, ...' string of the terminal voltages, or ''."""
    if "terminal_names" in data.files and "terminal_voltages" in data.files:
        parts = []
        for n, v in zip(data["terminal_names"], data["terminal_voltages"]):
            parts.append("%s=%+.3g V" % (n, float(v)))
        return ", ".join(parts)
    if "bias_voltage" in data.files:
        return "V = %+.3g V" % float(data["bias_voltage"])
    return ""


def default_cut(nodes, along):
    """(fixed, lo, hi) in METRES for a mid-line full-extent cut.

    along='x' -> fixed = y0 (mid height), range = [x_min, x_max]
    along='y' -> fixed = x0 (mid width),  range = [y_min, y_max]
    """
    xmin, ymin = nodes.min(axis=0)
    xmax, ymax = nodes.max(axis=0)
    if along == "x":
        return 0.5 * (ymin + ymax), xmin, xmax
    return 0.5 * (xmin + xmax), ymin, ymax


def cut_samples(along, fixed, lo, hi, num=400):
    """Sample points (xs, ys) [m] and arc length s [m] for a straight cut."""
    pad = (hi - lo) * 1e-6 + 1e-12
    var = np.linspace(lo + pad, hi - pad, num)
    if along == "x":
        xs, ys = var, np.full_like(var, fixed)
    else:
        xs, ys = np.full_like(var, fixed), var
    return xs, ys, np.abs(var - var[0])


def _values(z, logscale):
    z = np.asarray(z, float)
    return np.log10(np.clip(z, _TINY, None)) if logscale else z


def _tri_um(data):
    """Triangulation with coordinates in micrometres (for 2D/3D axes)."""
    x, y = data["nodes"].T
    if "elements" in data.files:
        return Triangulation(x * UM, y * UM, triangles=data["elements"])
    return Triangulation(x * UM, y * UM)


# ===========================================================================
#  draw routines  (take a Figure, clear it, draw)
# ===========================================================================
def draw_geometry(fig, data):
    """Plot the mesh (triangle edges), Gmsh-style."""
    fig.clear()
    ax = fig.add_subplot(111)
    x, y = data["nodes"].T
    ax.triplot(x * UM, y * UM, data["elements"], color="k", lw=0.5)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x\;(\mu\mathrm{m})$")
    ax.set_ylabel(r"$y\;(\mu\mathrm{m})$")
    ax.set_title("Mesh: %d nodes, %d elements"
                 % (data["nodes"].shape[0], data["elements"].shape[0]))


def draw_1d(fig, data, tri, keys, along, fixed, lo, hi, logscale):
    """1-D line cut of one or more fields (multiplot = len(keys) > 1)."""
    fig.clear()
    ax = fig.add_subplot(111)
    xs, ys, s = cut_samples(along, fixed, lo, hi)
    for k in keys:
        prof = LinearTriInterpolator(tri, field_array(data, k))(xs, ys)
        ax.plot(s * UM, _values(prof, logscale), lw=2, label=field_title(k))
    ax.set_xlabel(r"distance along %s-cut  ($\mu\mathrm{m}$)" % along)
    ax.set_ylabel(field_zlabel(keys[0], logscale) if len(keys) == 1 else "value")
    if len(keys) > 1:
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.margins(x=0)

def draw_vector_2d(fig, data, xkey, ykey, title, style="quiver", logmag=False):
    """Quiver or streamline of a vector field, over a |·| magnitude background."""
    fig.clear()
    ax = fig.add_subplot(111)
    x, y = data["nodes"].T
    Vx = np.asarray(data[xkey], float)
    Vy = np.asarray(data[ykey], float)
    mag = np.hypot(Vx, Vy)

    tri = _tri_um(data)
    tcf = ax.tricontourf(tri, _values(mag, logmag), levels=60, cmap="viridis")
    fig.colorbar(tcf, ax=ax,
                 label=("log10 " if logmag else "") + r"$|J|\;(\mathrm{A/m^2})$")

    if style == "quiver":
        step = max(1, len(x) // 400)          # thin out arrows for readability
        i = slice(None, None, step)
        ax.quiver(x[i] * UM, y[i] * UM, Vx[i], Vy[i],
                  color="k", scale=None, width=0.003)
    else:  # streamline: interpolate onto a regular grid
        xi = np.linspace(x.min(), x.max(), 60)
        yi = np.linspace(y.min(), y.max(), 60)
        Xg, Yg = np.meshgrid(xi, yi)
        raw_tri = Triangulation(x, y, triangles=data["elements"])
        Ux = LinearTriInterpolator(raw_tri, Vx)(Xg, Yg)
        Uy = LinearTriInterpolator(raw_tri, Vy)(Xg, Yg)
        ax.streamplot(Xg * UM, Yg * UM,
                      np.ma.filled(Ux, 0), np.ma.filled(Uy, 0),
                      color="k", density=1.2, linewidth=0.7)

    ax.set_aspect("equal")
    ax.set_xlabel(r"$x\;(\mu\mathrm{m})$")
    ax.set_ylabel(r"$y\;(\mu\mathrm{m})$")
    ax.set_title(title)

def draw_2d(fig, data, tri, key, logscale):
    fig.clear()
    ax = fig.add_subplot(111)
    tcf = ax.tricontourf(_tri_um(data), _values(field_array(data, key), logscale),
                         levels=60, cmap="coolwarm")
    fig.colorbar(tcf, ax=ax, label=field_zlabel(key, logscale))
    ax.set_xlabel(r"$x\;(\mu\mathrm{m})$")
    ax.set_ylabel(r"$y\;(\mu\mathrm{m})$")
    ax.set_title(field_title(key))
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.set_aspect("equal")


def draw_3d(fig, data, tri, key, logscale, style="tcad"):
    """3-D surface plot.  style='tcad' (flat/ortho) or 'normal' (perspective)."""
    fig.clear()
    ax = fig.add_subplot(111, projection="3d")
    z = _values(field_array(data, key), logscale)
    surf = ax.plot_trisurf(_tri_um(data), z, cmap="coolwarm",
                           edgecolor="k", linewidth=0.3, antialiased=True)
    ax.set_xlabel(r"$x\;(\mu\mathrm{m})$")
    ax.set_ylabel(r"$y\;(\mu\mathrm{m})$")

    title = field_title(key)
    zlabel = field_zlabel(key, logscale)

    if style == "tcad":
        # flat, orthographic "device" view with a horizontal colorbar
        nx, ny = data["nodes"][:, 0], data["nodes"][:, 1]
        xr = float(nx.max() - nx.min())
        yr = float(ny.max() - ny.min())
        ax.view_init(elev=25, azim=-90)
        ax.set_box_aspect((xr, yr, 0.1 * xr) if xr > 0 else (1, 1, 1))
        ax.set_axis_off()
        ax.set_proj_type("ortho")

        cax = fig.add_axes([0.25, 0.12, 0.50, 0.03])
        cbar = fig.colorbar(surf, cax=cax, orientation="horizontal")
        zmin, zmax = float(np.min(z)), float(np.max(z))
        if zmax > zmin:
            cbar.set_ticks(np.linspace(zmin, zmax, 5))
        cbar.ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        cbar.ax.tick_params(which="major", length=10)
        cbar.ax.tick_params(which="minor", length=5)
        cbar.set_label(zlabel)
        cbar.ax.set_title(title)
    else:
        # conventional perspective view
        fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label=zlabel)
        ax.set_zlabel(zlabel)
        ax.set_title(title)
        for axis in (ax.xaxis, ax.yaxis):
            axis.get_major_formatter().set_useOffset(False)
            try:
                axis.get_major_formatter().set_scientific(False)
            except AttributeError:
                pass


def band_diagram_available(data):
    """True if the file has everything the band diagram needs."""
    return all(k in data.files for k in _BAND_KEYS)


def draw_band_diagram(fig, data, tri, along, fixed, lo, hi):
    """1-D energy band diagram: E_c, E_v, E_i, E_Fn, E_Fp along a cut [eV].

    Derived from the three potentials and the material scalars in the file:
        E_i = -psi,  E_Fn = -phi_n,  E_Fp = -phi_p,
        E_c = E_i + VT*ln(Nc/ni),   E_v = E_i - VT*ln(Nv/ni).
    """
    missing = [k for k in _BAND_KEYS if k not in data.files]
    if missing:
        raise KeyError("band diagram needs %s in the .npz" % ", ".join(missing))

    fig.clear()
    ax = fig.add_subplot(111)
    xs, ys, s = cut_samples(along, fixed, lo, hi)
    s_um = s * UM

    def interp(key):
        return LinearTriInterpolator(tri, data[key])(xs, ys)

    ni = float(data["ni"]); Nc = float(data["Nc"])
    Nv = float(data["Nv"]); VT = float(data["VT"])
    dEc = VT * np.log(Nc / ni)          # E_c - E_i [eV]
    dEv = VT * np.log(Nv / ni)          # E_i - E_v [eV]

    Ei = -interp("potential")
    EFn = -interp("electron_fermi_potential")
    EFp = -interp("hole_fermi_potential")
    Ec = Ei + dEc
    Ev = Ei - dEv

    ax.plot(s_um, Ec, color="tab:blue", lw=2.2, label=r"$E_c$  conduction band")
    ax.plot(s_um, Ev, color="tab:red", lw=2.2, label=r"$E_v$  valence band")
    ax.plot(s_um, Ei, color="0.4", lw=1.2, ls="--", label=r"$E_i$  intrinsic")
    ax.plot(s_um, EFn, color="tab:green", lw=1.8, ls="-.",
            label=r"$E_{Fn}$  electron quasi-Fermi")
    ax.plot(s_um, EFp, color="tab:orange", lw=1.8, ls="-.",
            label=r"$E_{Fp}$  hole quasi-Fermi")
    ax.fill_between(s_um, Ev, Ec, color="0.85", alpha=0.5, zorder=0)

    # junction marker(s): where n and p cross over, if carriers are present
    try:
        if ("electron_concentration" in data.files
                and "hole_concentration" in data.files):
            n = np.ma.filled(interp("electron_concentration"), np.nan)
            p = np.ma.filled(interp("hole_concentration"), np.nan)
            d = np.sign(n - p)
            for j in np.where(np.diff(d) != 0)[0]:
                if np.isfinite(d[j]) and np.isfinite(d[j + 1]):
                    ax.axvline(s_um[j], color="0.0", lw=1.0, ls=":")
    except Exception:                                  # noqa: BLE001
        pass

    ax.set_xlabel(r"distance along %s-cut  ($\mu\mathrm{m}$)" % along)
    ax.set_ylabel(r"Energy $(\mathrm{eV})$")
    ax.legend(loc="best", framealpha=0.9, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.margins(x=0)


# ===========================================================================
#  GUI  (tkinter imported lazily so the helpers above stay headless-safe)
# ===========================================================================
def run_gui():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import (
        FigureCanvasTkAgg, NavigationToolbar2Tk)

    class ClosableNotebook(ttk.Notebook):
        """ttk.Notebook whose tabs each carry a clickable close (X) button.

        Clicking the X calls ``close_callback(tab_widget_id)``.  The X image is
        drawn at runtime (no external files needed).
        """
        _style_ready = False

        def __init__(self, master, **kw):
            if not ClosableNotebook._style_ready:
                self._init_style()
                ClosableNotebook._style_ready = True
            kw["style"] = "Closable.TNotebook"
            super().__init__(master, **kw)
            self._active = None
            self.close_callback = None
            self.bind("<ButtonPress-1>", self._press, True)
            self.bind("<ButtonRelease-1>", self._release)

        def _init_style(self):
            style = ttk.Style()
            img = tk.PhotoImage("img_tabclose", width=16, height=16)
            for t in range(4, 12):                 # draw an X (2 px thick)
                for d in (0, 1):
                    img.put("#666666", (t + d, t))
                    img.put("#666666", (t + d, 15 - t))
            self._imgs = (img,)                    # keep a reference alive
            try:
                style.element_create("close", "image", "img_tabclose",
                                     border=6, sticky='')
                style.layout("Closable.TNotebook",
                             [("Closable.TNotebook.client", {"sticky": "nswe"})])
                style.layout("Closable.TNotebook.Tab", [
                    ("Closable.TNotebook.tab", {"sticky": "nswe", "children": [
                        ("Closable.TNotebook.padding", {"side": "top",
                         "sticky": "nswe", "children": [
                            ("Closable.TNotebook.focus", {"side": "top",
                             "sticky": "nswe", "children": [
                                ("Closable.TNotebook.label", {"side": "left",
                                 "sticky": ''}),
                                ("Closable.TNotebook.close", {"side": "left",
                                 "sticky": ''}),
                            ]})
                        ]})
                    ]})
                ])
            except tk.TclError:                    # already defined -> reuse
                pass

        def _press(self, event):
            try:
                element = self.identify(event.x, event.y)
            except tk.TclError:
                return
            if "close" in element:
                self._active = self.index("@%d,%d" % (event.x, event.y))
                self.state(["pressed"])
                return "break"

        def _release(self, event):
            if not self.instate(["pressed"]):
                return
            self.state(["!pressed"])
            try:
                element = self.identify(event.x, event.y)
                index = self.index("@%d,%d" % (event.x, event.y))
            except tk.TclError:
                element, index = "", None
            if ("close" in element and index is not None
                    and index == self._active and callable(self.close_callback)):
                self.close_callback(self.tabs()[index])
            self._active = None

    class SolutionViewer(tk.Tk):
        def __init__(self):
            super().__init__()
            super().__init__()
            self.title("Semiconductor-Solver  .npz  viewer")
            try:
                self.state("zoomed")              # Windows / macOS
            except tk.TclError:
                self.attributes("-zoomed", True)  # most Linux window managers
            self.minsize(900, 640)

            self.data = None
            self.tri = None
            self.fields = []
            self.cut = None            # dict(along, fixed, lo, hi) [m] or None
            self.folder_path = None    # folder chosen via "Add Folder"
            self.loaded_path = None
            self._anim = None          # active animation state, or None
            self._anim_after = None    # pending .after() id, or None
            # Each "Add Multiplot" click creates an independent multiplot
            # window (tab).  multi_windows maps tab-title -> list of fields;
            # _multi_counter gives each new window a unique number/title.
            self.multi_windows = {}    # title -> [field, ...]
            self._multi_counter = 0
            self.tabs = {}             # title -> (frame, fig, canvas)

            self._build_top_bar(tk, ttk, filedialog, messagebox)
            self._build_left_bar(tk, ttk, messagebox)
            self._build_plot_area(tk, ttk, FigureCanvasTkAgg,
                                  NavigationToolbar2Tk)
            self._update_mode_states()

        # ---- section 1: top bar -------------------------------------------
        def _build_top_bar(self, tk, ttk, filedialog, messagebox):
            self._fd = filedialog
            self._mb = messagebox
            top = ttk.Frame(self, padding=6)
            top.pack(side=tk.TOP, fill=tk.X)

            ttk.Button(top, text="Add Folder", command=self.on_add_folder
                       ).pack(side=tk.LEFT, padx=4)
            ttk.Button(top, text="Load .npz ...", command=self.on_load
                       ).pack(side=tk.LEFT, padx=4)
            ttk.Button(top, text="Clean Plot Area",
                    command=self.on_clean_plot_area).pack(side=tk.RIGHT, padx=4)

            ttk.Button(top, text="Plot IV Curve",
                       command=self.on_plot_iv).pack(side=tk.RIGHT, padx=4)
            # Device type (to the left of "Plot IV Curve"): controls whether
            # that button draws a single I-V curve (2 Terminal) or a family
            # of output characteristics grouped by base bias (3 Terminal).
            self.device_var = tk.StringVar(value="2 Terminal")
            ttk.Combobox(top, textvariable=self.device_var, width=11,
                         state="readonly",
                         values=["2 Terminal", "3 Terminal"]
                         ).pack(side=tk.RIGHT, padx=(0, 4))
            ttk.Label(top, text="Device type:").pack(side=tk.RIGHT)

            # folder directory (top) + file name (bottom), small font
            info = ttk.Frame(top)
            info.pack(side=tk.LEFT, padx=8)
            self.dir_lbl = ttk.Label(info, text="(no folder)",
                                     font=("TkDefaultFont", 8))
            self.dir_lbl.pack(side=tk.TOP, anchor="w")
            self.file_lbl = ttk.Label(info, text="(no file loaded)",
                                      font=("TkDefaultFont", 8))
            self.file_lbl.pack(side=tk.TOP, anchor="w")

            ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X)

        # ---- section 2: left bar ------------------------------------------
        def _build_left_bar(self, tk, ttk, messagebox):
            left = ttk.Frame(self, padding=8)
            left.pack(side=tk.LEFT, fill=tk.Y)

            # Mesh geometry toggle (above Display Mode): auto-opens/closes a
            # dedicated mesh tab in the plot area.
            self.geom_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(left, text="Show mesh geometry",
                            variable=self.geom_var,
                            command=self._on_geom_toggle
                            ).grid(row=0, column=0, columnspan=3, sticky="w",
                                   pady=(0, 8))

            # Display Mode
            ttk.Label(left, text="Display Mode:").grid(row=1, column=0,
                                                       sticky="w", pady=(0, 2))
            self.mode_var = tk.StringVar(value="1D")
            mode_cb = ttk.Combobox(left, textvariable=self.mode_var, width=6,
                                   state="readonly", values=["1D", "2D", "3D"])
            mode_cb.grid(row=1, column=1, sticky="w")
            mode_cb.bind("<<ComboboxSelected>>",
                         lambda e: self._update_mode_states())

            # Multiplot  (directly below Display Mode) + "Add Multiplot"
            # button.  Multiplot=Yes turns on the multiplot feature (and
            # deactivates the single-Field selector); each "Add Multiplot"
            # click then spawns a new multiplot window.
            ttk.Label(left, text="Multiplot:").grid(row=2, column=0,
                                                    sticky="w", pady=(10, 0))
            self.multi_var = tk.StringVar(value="No")
            self.multi_cb = ttk.Combobox(left, textvariable=self.multi_var,
                                         width=6, state="readonly",
                                         values=["No", "Yes"])
            self.multi_cb.grid(row=2, column=1, sticky="w", pady=(10, 0))
            self.multi_cb.bind("<<ComboboxSelected>>",
                               lambda e: self._on_multi_change())
            self.add_multi_btn = ttk.Button(left, text="Add Multiplot",
                                            width=13,
                                            command=self.on_add_multiplot)
            self.add_multi_btn.grid(row=2, column=2, sticky="w",
                                    pady=(10, 0), padx=(4, 0))

            # Multiplot windows details: a single button that opens a popup,
            # so the left bar does not grow as more windows are added.
            self.details_btn = ttk.Button(left, text="Multiplot Windows Details",
                                          command=self.on_multi_details)
            self.details_btn.grid(row=3, column=0, columnspan=3,
                                  sticky="we", pady=(6, 4))
            self.details_btn.grid_remove()        # shown once a window exists

            # Field  (active only when Multiplot=No)
            ttk.Label(left, text="Field:").grid(row=4, column=0, sticky="w",
                                                pady=(6, 0))
            self.field_var = tk.StringVar()
            self.field_cb = ttk.Combobox(left, textvariable=self.field_var,
                                         state="readonly", width=24)
            self.field_cb.grid(row=4, column=1, columnspan=2, sticky="w",
                               pady=(6, 0))
            self.field_cb.bind("<<ComboboxSelected>>", self._on_field_change)

            # Energy band diagram toggle (below Field): auto-opens/closes a tab.
            # "Refresh" (to its right) re-draws the band tab for the current
            # "1D cut along" axis / cut points.
            self.band_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(left, text="Energy band diagram",
                            variable=self.band_var,
                            command=self._on_band_toggle
                            ).grid(row=11, column=0, columnspan=2, sticky="w",
                                   pady=(6, 0))
            self.band_refresh_btn = ttk.Button(left, text="Refresh", width=8,
                                               command=self.on_band_refresh)
            self.band_refresh_btn.grid(row=11, column=2, sticky="w", pady=(6, 0))

            # 1D cut direction
            ttk.Label(left, text="1D cut along").grid(row=6, column=0,
                                                      sticky="w", pady=(10, 0))
            self.dir_var = tk.StringVar(value="x")
            dirf = ttk.Frame(left)
            dirf.grid(row=6, column=1, sticky="w", pady=(10, 0))
            self.rb_x = ttk.Radiobutton(dirf, text="x", variable=self.dir_var,
                                        value="x", command=self._on_dir_change)
            self.rb_y = ttk.Radiobutton(dirf, text="y", variable=self.dir_var,
                                        value="y", command=self._on_dir_change)
            self.rb_x.pack(side=tk.LEFT)
            self.rb_y.pack(side=tk.LEFT)

            # cut point entries (labels change with direction)
            self.l_fixed = ttk.Label(left, text="x0")
            self.l_fixed.grid(row=7, column=0, sticky="e", pady=3)
            self.e_fixed = ttk.Entry(left, width=10)
            self.e_fixed.grid(row=7, column=1, sticky="w")
            ttk.Label(left, text="µm").grid(row=7, column=2, sticky="w")

            self.l_lo = ttk.Label(left, text="y_min")
            self.l_lo.grid(row=8, column=0, sticky="e", pady=3)
            self.e_lo = ttk.Entry(left, width=10)
            self.e_lo.grid(row=8, column=1, sticky="w")
            ttk.Label(left, text="µm").grid(row=8, column=2, sticky="w")

            self.l_hi = ttk.Label(left, text="y_max")
            self.l_hi.grid(row=9, column=0, sticky="e", pady=3)
            self.e_hi = ttk.Entry(left, width=10)
            self.e_hi.grid(row=9, column=1, sticky="w")
            ttk.Label(left, text="µm").grid(row=9, column=2, sticky="w")

            self.set_btn = ttk.Button(left, text="Set cut points",
                                      command=self.on_set_cut)
            self.set_btn.grid(row=10, column=0, columnspan=2, sticky="w",
                              pady=(4, 12))

            # log scale
            self.log_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(left, text="log scale", variable=self.log_var
                            ).grid(row=12, column=0, columnspan=2, sticky="w",
                                   pady=(6, 6))

            # 3D style (active in 3D mode)
            ttk.Label(left, text="3D style:").grid(row=13, column=0, sticky="w",
                                                   pady=(0, 6))
            self.style_var = tk.StringVar(value="tcad")
            self.style_cb = ttk.Combobox(left, textvariable=self.style_var,
                                         width=8, state="readonly",
                                         values=["tcad", "normal"])
            self.style_cb.grid(row=13, column=1, sticky="w", pady=(0, 6))

            # Plot + close-tab
            ttk.Button(left, text="Plot", command=self.on_plot
                       ).grid(row=14, column=0, columnspan=2, sticky="we",
                              ipady=4)
            self.animate_btn = ttk.Button(left, text="Animate",
                                          command=self.on_animate)
            self.animate_btn.grid(row=14, column=2, sticky="we",
                                  padx=(4, 0), ipady=4)
            style = ttk.Style(self)
            style.configure("Big.TLabelframe.Label",
                            font=("TkDefaultFont", 11, "bold"))
            vec = ttk.LabelFrame(left, text="Current density (vector)",
                                 style="Big.TLabelframe")
            vec.grid(row=15, column=0, columnspan=3, sticky="we", pady=(10, 0))

            ttk.Label(vec, text="Field:").grid(row=0, column=0, sticky="w", padx=4, pady=2)
            self.vec_var = tk.StringVar(value="Total (J)")
            ttk.Combobox(vec, textvariable=self.vec_var, width=16, state="readonly",
                         values=["Total (J)", "Electron (Jn)", "Hole (Jp)"]
                         ).grid(row=0, column=1, columnspan=6, sticky="w",
                                padx=4, pady=2)

            ttk.Label(vec, text="Style:").grid(row=1, column=0, sticky="w", padx=4, pady=2)
            self.vec_style = tk.StringVar(value="quiver")
            ttk.Combobox(vec, textvariable=self.vec_style, width=16, state="readonly",
                         values=["quiver", "streamline"]
                         ).grid(row=1, column=1, columnspan=6, sticky="w",
                                padx=4, pady=2)

            # Visualizer: On / Off -- mutually-exclusive tick boxes.  "Off" plots
            # the loaded file only.  "On" animates the selected field/style across
            # every .npz in the loaded file's folder that shares the reference
            # operating point but sweeps the chosen Ramp Terminal -- with the
            # Hold/Ramp terminals chosen explicitly (no value-guessing).
            ttk.Label(vec, text="Visualizer:").grid(row=2, column=0, sticky="w",
                                                    padx=4, pady=(8, 2))
            self.vis_on = tk.BooleanVar(value=False)
            self.vis_off = tk.BooleanVar(value=True)
            ttk.Checkbutton(vec, text="On", variable=self.vis_on,
                            command=lambda: self._on_vis_toggle("on")
                            ).grid(row=2, column=1, sticky="w", pady=(8, 2))
            ttk.Checkbutton(vec, text="Off", variable=self.vis_off,
                            command=lambda: self._on_vis_toggle("off")
                            ).grid(row=2, column=2, columnspan=5, sticky="w",
                                   pady=(8, 2))

            # Terminals: read-only, auto-discovered from the folder's .npz files
            # (small font).  Rendered dynamically -- one box per terminal, so 2-,
            # 3- or 4-terminal devices all display correctly.
            ttk.Label(vec, text="Terminals:").grid(row=3, column=0, sticky="w",
                                                   padx=4, pady=2)
            self.term_frame = ttk.Frame(vec)
            self.term_frame.grid(row=3, column=1, columnspan=6, sticky="w",
                                 padx=4, pady=2)
            self.term_boxes = []            # list of read-only Entry widgets
            self.terminals = []             # discovered terminal names

            # Hold Terminal / Ramp Terminal selectors.
            ttk.Label(vec, text="Hold Terminal:").grid(row=4, column=0, sticky="w",
                                                       padx=4, pady=2)
            self.hold_term = tk.StringVar(value="")
            self.hold_term_cb = ttk.Combobox(vec, textvariable=self.hold_term,
                                             width=8, state="disabled", values=[])
            self.hold_term_cb.grid(row=4, column=1, columnspan=2, sticky="w",
                                   padx=4, pady=2)
            ttk.Label(vec, text="Ramp Terminal:").grid(row=4, column=3,
                                                       columnspan=2, sticky="e",
                                                       padx=(4, 2), pady=2)
            self.ramp_term = tk.StringVar(value="")
            self.ramp_term_cb = ttk.Combobox(vec, textvariable=self.ramp_term,
                                             width=8, state="disabled", values=[])
            self.ramp_term_cb.grid(row=4, column=5, columnspan=2, sticky="w",
                                   padx=(0, 4), pady=2)

            # Hold Bias: the held terminal's voltage [V].
            ttk.Label(vec, text="Hold Bias:").grid(row=5, column=0, sticky="w",
                                                   padx=4, pady=2)
            self.hold_var = tk.StringVar(value="")
            self.hold_entry = ttk.Entry(vec, textvariable=self.hold_var, width=8)
            self.hold_entry.grid(row=5, column=1, columnspan=2, sticky="w",
                                 padx=4, pady=2)
            ttk.Label(vec, text="V").grid(row=5, column=3, sticky="w", pady=2)

            # Ramp Bias: start : step : max  [V].
            ttk.Label(vec, text="Ramp Bias:").grid(row=6, column=0, sticky="w",
                                                   padx=4, pady=2)
            self.ramp_start = tk.StringVar(value="")
            self.ramp_step = tk.StringVar(value="")
            self.ramp_max = tk.StringVar(value="")
            self.ramp_start_e = ttk.Entry(vec, textvariable=self.ramp_start, width=6)
            self.ramp_start_e.grid(row=6, column=1, sticky="w", padx=(4, 0), pady=2)
            ttk.Label(vec, text=":").grid(row=6, column=2, sticky="w")
            self.ramp_step_e = ttk.Entry(vec, textvariable=self.ramp_step, width=6)
            self.ramp_step_e.grid(row=6, column=3, sticky="w", padx=2, pady=2)
            ttk.Label(vec, text=":").grid(row=6, column=4, sticky="w")
            self.ramp_max_e = ttk.Entry(vec, textvariable=self.ramp_max, width=6)
            self.ramp_max_e.grid(row=6, column=5, sticky="w", padx=2, pady=2)
            ttk.Label(vec, text="V").grid(row=6, column=6, sticky="w")

            ttk.Button(vec, text="Plot Current Density",
                       command=self.on_plot_vector
                       ).grid(row=7, column=0, columnspan=7, sticky="we",
                              padx=4, pady=(6, 4))

            self._update_vec_vis_state()

            # "Close current tab" sits at the very end of the left panel.
            ttk.Button(left, text="Close current tab", command=self.on_close_tab
                       ).grid(row=16, column=0, columnspan=3, sticky="we",
                              pady=(12, 0))

            self._left = left

        def _on_geom_toggle(self):
            """Checkbox handler: open the mesh-geometry tab or close it."""
            from tkinter import ttk
            if self.geom_var.get():
                if self.data is None:
                    self._mb.showwarning("No file", "Load a .npz first.")
                    self.geom_var.set(False)
                    return
                fig, canvas = self._get_or_create_tab(ttk, "Mesh geometry")
                draw_geometry(fig, self.data)
                try:
                    fig.tight_layout()
                except Exception:                          # noqa: BLE001
                    pass
                canvas.draw()
            elif "Mesh geometry" in self.tabs:
                self._close_title("Mesh geometry")

        def _on_band_toggle(self):
            """Checkbox handler: open the energy-band-diagram tab or close it."""
            from tkinter import ttk
            if self.band_var.get():
                if self.data is None:
                    self._mb.showwarning("No file", "Load a .npz first.")
                    self.band_var.set(False)
                    return
                if not band_diagram_available(self.data):
                    self._mb.showwarning(
                        "Band diagram",
                        "This file lacks the potentials / material scalars "
                        "needed for the energy band diagram.")
                    self.band_var.set(False)
                    return
                along = self.dir_var.get()
                if self.cut and self.cut["along"] == along:
                    fixed = self.cut["fixed"]
                    lo, hi = self.cut["lo"], self.cut["hi"]
                else:
                    fixed, lo, hi = default_cut(self.data["nodes"], along)
                fig, canvas = self._get_or_create_tab(ttk, "Energy band diagram")
                draw_band_diagram(fig, self.data, self.tri, along, fixed, lo, hi)
                try:
                    fig.tight_layout()
                except Exception:                          # noqa: BLE001
                    pass
                canvas.draw()
            elif "Energy band diagram" in self.tabs:
                self._close_title("Energy band diagram")

        def on_band_refresh(self):
            """Redraw the band-diagram tab for the CURRENT 1D-cut axis / points."""
            from tkinter import ttk
            if self.data is None:
                self._mb.showwarning("No file", "Load a .npz first.")
                return
            if not band_diagram_available(self.data):
                self._mb.showwarning(
                    "Band diagram",
                    "This file lacks the potentials / material scalars "
                    "needed for the energy band diagram.")
                return
            self.band_var.set(True)                 # keep the checkbox in sync
            along = self.dir_var.get()
            if self.cut and self.cut["along"] == along:
                fixed = self.cut["fixed"]
                lo, hi = self.cut["lo"], self.cut["hi"]
            else:
                fixed, lo, hi = default_cut(self.data["nodes"], along)
            fig, canvas = self._get_or_create_tab(ttk, "Energy band diagram")
            try:
                draw_band_diagram(fig, self.data, self.tri, along, fixed, lo, hi)
                try:
                    fig.tight_layout()
                except Exception:                      # noqa: BLE001
                    pass
                canvas.draw()
            except Exception as exc:                   # noqa: BLE001
                self._mb.showerror("Plot failed", str(exc))

        def _update_field_state(self):
            """Field selector is active only when Multiplot is No."""
            self.field_cb.configure(
                state="readonly" if self.multi_var.get() == "No"
                else "disabled")

        def _update_add_multi_state(self):
            """'Add Multiplot' is active iff Multiplot=Yes (in 1D mode)."""
            on = (self.mode_var.get() == "1D"
                  and self.multi_var.get() == "Yes")
            self.add_multi_btn.configure(state="normal" if on else "disabled")

        def _update_animate_state(self):
            """'Animate' is active iff Display Mode=1D and Multiplot=Yes."""
            on = (self.mode_var.get() == "1D"
                  and self.multi_var.get() == "Yes")
            self.animate_btn.configure(state="normal" if on else "disabled")

        def _on_field_change(self, _e=None):
            """Default log-scale to the field's natural setting (user can override)."""
            self.log_var.set(field_default_log(self.field_var.get()))

        def _update_multi_details_btn(self):
            """Show the details button only when multiplot windows exist."""
            if self.multi_var.get() == "Yes" and self.multi_windows:
                self.details_btn.grid()
            else:
                self.details_btn.grid_remove()

        def on_multi_details(self):
            """Popup listing every multiplot window and its curves."""
            if not self.multi_windows:
                self._mb.showinfo(
                    "Multiplot windows",
                    "No multiplot windows yet. Use \"Add Multiplot\".")
                return
            tk = __import__("tkinter")
            from tkinter import ttk
            win = tk.Toplevel(self)
            win.title("Multiplot windows details")
            win.transient(self)

            ttk.Label(win, text="Multiplot windows: %d" % len(self.multi_windows),
                      font=("TkDefaultFont", 10, "bold")
                      ).grid(row=0, column=0, sticky="w", padx=10, pady=(10, 4))
            row = 1
            for title, flds in self.multi_windows.items():
                ttk.Label(win, text="%s  —  %d curve%s"
                          % (title, len(flds), "" if len(flds) == 1 else "s"),
                          font=("TkDefaultFont", 9, "bold")
                          ).grid(row=row, column=0, sticky="w",
                                 padx=10, pady=(6, 0))
                row += 1
                for i, f in enumerate(flds, 1):
                    ttk.Label(win, text="    %d.  %s  (%s)"
                              % (i, field_title(f), f)
                              ).grid(row=row, column=0, sticky="w",
                                     padx=16, pady=1)
                    row += 1
            ttk.Button(win, text="Close", command=win.destroy
                       ).grid(row=row, column=0, pady=10)
            win.grab_set()

        # ---- section 3: plot area (tabbed) --------------------------------
        def _build_plot_area(self, tk, ttk, FigureCanvasTkAgg, NavToolbar):
            self._FigureCanvasTkAgg = FigureCanvasTkAgg
            self._NavToolbar = NavToolbar
            right = ttk.Frame(self, padding=4)
            right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
            self.nb = ClosableNotebook(right)
            self.nb.close_callback = self._close_tab_by_frameid
            self.nb.pack(fill=tk.BOTH, expand=True)
            # double-click a tab label also closes it
            self.nb.bind("<Double-Button-1>", self._on_tab_dblclick)

        # ---- tab management ----------------------------------------------
        def _get_or_create_tab(self, ttk, title):
            if title in self.tabs:
                frame, fig, canvas = self.tabs[title]
                self.nb.select(frame)
                return fig, canvas
            frame = ttk.Frame(self.nb)
            self.nb.add(frame, text=title)
            fig = Figure(figsize=(7, 5))
            canvas = self._FigureCanvasTkAgg(fig, master=frame)
            canvas.get_tk_widget().pack(fill="both", expand=True)
            self._NavToolbar(canvas, frame)
            self.nb.select(frame)
            self.tabs[title] = (frame, fig, canvas)
            return fig, canvas

        def _close_title(self, title):
            if self._anim and self._anim.get("title") == title:
                self._stop_animation()          # don't animate a closed tab
            frame, _fig, _canvas = self.tabs.pop(title)
            self.nb.forget(frame)
            frame.destroy()
            if title == "Mesh geometry" and hasattr(self, "geom_var"):
                self.geom_var.set(False)   # keep the checkbox in sync
            if title == "Energy band diagram" and hasattr(self, "band_var"):
                self.band_var.set(False)
            if title in self.multi_windows:
                del self.multi_windows[title]
                self._update_multi_details_btn()

        def on_plot_iv(self):
            """Plot I-V from all solution_*.npz in the folder.

            "2 Terminal": single I-V curve (current vs voltage of the
            widest-range / collector terminal).
            "3 Terminal": a FAMILY of output curves -- collector current vs
            collector voltage, one line per base bias.
            """
            import glob
            folder = self.folder_path or (
                os.path.dirname(self.loaded_path) if self.loaded_path else None)
            if not folder:
                self._mb.showwarning("No folder",
                                     "Add a folder or load a .npz first.")
                return
            from tkinter import ttk
            rows = []   # (names, volts, currs)
            for f in sorted(glob.glob(os.path.join(folder, "solution_*.npz"))):
                try:
                    d = np.load(f, allow_pickle=False)
                    if all(k in d.files for k in ("terminal_names",
                            "terminal_voltages", "terminal_currents")):
                        rows.append(([str(x) for x in d["terminal_names"]],
                                     np.asarray(d["terminal_voltages"], float),
                                     np.asarray(d["terminal_currents"], float)))
                    d.close()
                except Exception:                      # noqa: BLE001
                    continue
            if len(rows) < 2:
                self._mb.showwarning("IV curve",
                    "Need >=2 solution_*.npz (with terminal_currents) in:\n%s"
                    % folder)
                return

            names = rows[0][0]
            V = np.array([r[1] for r in rows])
            I = np.array([r[2] for r in rows])
            ranges = V.max(axis=0) - V.min(axis=0)

            def _idx(nm):
                return names.index(nm) if nm in names else None

            # collector = 'c_contact', else the widest-range terminal
            col = _idx("c_contact")
            if col is None:
                col = int(np.argmax(ranges))

            fig, canvas = self._get_or_create_tab(ttk, "IV curve")
            fig.clear()
            ax = fig.add_subplot(111)

            if self.device_var.get() == "3 Terminal":
                # base = 'b_contact', else the 2nd-widest-range terminal
                base = _idx("b_contact")
                if base is None:
                    r2 = ranges.copy()
                    r2[col] = -1.0
                    base = int(np.argmax(r2))
                groups = {}
                for k in range(len(rows)):
                    key = round(float(V[k, base]), 3)
                    groups.setdefault(key, []).append((V[k, col], I[k, col]))
                for vb in sorted(groups):
                    pts = sorted(groups[vb])
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    ax.plot(xs, ys, "o-", lw=2,
                            label=r"$V_{%s}=%.3g$ V" % (names[base], vb))
                ax.legend(fontsize=8, title="base bias")
                ax.set_title("Output characteristics (%d families)" % len(groups))
            else:
                order = np.argsort(V[:, col])
                ax.plot(V[order, col], I[order, col], "o-", lw=2)
                ax.set_title("I-V curve (%d points)" % len(rows))

            ax.set_xlabel(r"$V_{%s}$ (V)" % names[col])
            ax.set_ylabel(r"$I_{%s}$ (A/m)" % names[col])
            ax.grid(True, alpha=0.3)
            try:
                fig.tight_layout()
            except Exception:                          # noqa: BLE001
                pass
            canvas.draw()

        def on_close_tab(self):
            cur = self.nb.select()
            for title, (frame, _f, _c) in list(self.tabs.items()):
                if str(frame) == cur:
                    self._close_title(title)
                    break

        def _close_tab_by_frameid(self, frame_id):
            """Close the tab whose frame widget id matches (used by the X button)."""
            for title, (frame, _f, _c) in list(self.tabs.items()):
                if str(frame) == str(frame_id):
                    self._close_title(title)
                    break

        def _on_tab_dblclick(self, event):
            try:
                idx = self.nb.index("@%d,%d" % (event.x, event.y))
            except tk.TclError:
                return
            frame_id = self.nb.tabs()[idx]
            for title, (frame, _f, _c) in list(self.tabs.items()):
                if str(frame) == frame_id:
                    self._close_title(title)
                    break

        # ---- widget state logic ------------------------------------------
        def _update_mode_states(self):
            mode = self.mode_var.get()
            is1d = mode == "1D"
            cut_state = "normal" if is1d else "disabled"
            for w in (self.rb_x, self.rb_y, self.e_fixed, self.e_lo, self.e_hi,
                      self.set_btn):
                w.configure(state=cut_state)
            self.multi_cb.configure(state="readonly" if is1d else "disabled")
            self.style_cb.configure(
                state="readonly" if mode == "3D" else "disabled")
            if not is1d:                       # multiplot is 1D-only
                self.multi_var.set("No")
            self._update_multi_details_btn()
            self._update_field_state()
            self._update_add_multi_state()
            self._update_animate_state()

        def _on_dir_change(self):
            along = self.dir_var.get()
            if along == "x":
                self.l_fixed.config(text="y0")
                self.l_lo.config(text="x_min")
                self.l_hi.config(text="x_max")
            else:
                self.l_fixed.config(text="x0")
                self.l_lo.config(text="y_min")
                self.l_hi.config(text="y_max")
            self.cut = None                    # invalidate previous cut
            self._prefill_cut()

        def _prefill_cut(self):
            if self.data is None:
                return
            fixed, lo, hi = default_cut(self.data["nodes"], self.dir_var.get())
            for entry, val in ((self.e_fixed, fixed),
                               (self.e_lo, lo), (self.e_hi, hi)):
                entry.delete(0, "end")
                entry.insert(0, "%.4g" % (val * UM))

        def _on_multi_change(self):
            """Multiplot Yes/No just toggles the feature; windows are added
            one at a time via the 'Add Multiplot' button."""
            if self.multi_var.get() == "Yes" and self.mode_var.get() != "1D":
                self._mb.showinfo("Multiplot", "Multiplot is for 1D mode.")
                self.multi_var.set("No")
            self._update_multi_details_btn()
            self._update_field_state()
            self._update_add_multi_state()
            self._update_animate_state()

        def on_add_multiplot(self):
            """Open the setup dialog and, on OK, create a NEW multiplot window."""
            if self.multi_var.get() != "Yes":
                return                          # button is disabled anyway
            if self.data is None:
                self._mb.showwarning("No file", "Load a .npz first.")
                return
            if self.mode_var.get() != "1D":
                self._mb.showinfo("Multiplot", "Multiplot is for 1D mode.")
                return
            self._open_multiplot_dialog(on_ok=self._create_multiplot_window)

        def _create_multiplot_window(self, fields):
            """Register a fresh multiplot tab from the chosen fields.

            The curves are NOT drawn here -- the tab shows a placeholder and
            is only rendered when the user presses the Plot button.
            """
            fields = [f for f in fields if f]
            if not fields:
                return
            from tkinter import ttk
            self._multi_counter += 1
            title = "Multiplot %d (1D)" % self._multi_counter
            self.multi_windows[title] = fields
            fig, canvas = self._get_or_create_tab(ttk, title)   # creates + selects
            fig.clear()
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5,
                    "Press \"Plot\" to draw:\n" +
                    "\n".join(field_title(f) for f in fields),
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=10, color="0.4")
            ax.set_axis_off()
            canvas.draw()
            self._update_multi_details_btn()

        def _current_multi_title(self):
            """Title of the currently selected tab if it is a multiplot window."""
            cur = self.nb.select()
            for title, (frame, _f, _c) in self.tabs.items():
                if str(frame) == cur and title in self.multi_windows:
                    return title
            return None

        def _draw_multiplot(self, title):
            """Render an existing multiplot window with the current settings."""
            from tkinter import ttk
            fields = self.multi_windows[title]
            along = self.dir_var.get()
            if self.cut and self.cut["along"] == along:
                fixed = self.cut["fixed"]
                lo, hi = self.cut["lo"], self.cut["hi"]
            else:
                fixed, lo, hi = default_cut(self.data["nodes"], along)
            fig, canvas = self._get_or_create_tab(ttk, title)
            try:
                draw_1d(fig, self.data, self.tri, fields,
                        along, fixed, lo, hi, self.log_var.get())
                fig.suptitle("[%s]" % operating_point(self.data))
                try:
                    fig.tight_layout()
                except Exception:                      # noqa: BLE001
                    pass
                canvas.draw()
            except Exception as exc:                   # noqa: BLE001
                self._mb.showerror("Plot failed", str(exc))

        def _plot_current_multiplot(self):
            """Plot button in multiplot mode: (re)draw the active window."""
            if not self.multi_windows:
                self._mb.showinfo(
                    "Multiplot",
                    "Click \"Add Multiplot\" to create a multiplot window "
                    "first, then press Plot.")
                return
            title = self._current_multi_title()
            if title is None:                    # current tab isn't a multiplot
                title = next(reversed(self.multi_windows))   # latest window
            self._draw_multiplot(title)

        # ---- animation over a set of files -------------------------------
        def _stop_animation(self):
            """Cancel any in-flight animation and free its state."""
            if self._anim_after is not None:
                try:
                    self.after_cancel(self._anim_after)
                except Exception:                      # noqa: BLE001
                    pass
                self._anim_after = None
            self._anim = None

        def on_animate(self):
            """Animate the active multiplot's curves across many files.

            Pops up a multi-file picker (from the "Add Folder" directory),
            then redraws the current multiplot window once per file -- same
            curves, each file's data -- clearing the previous frame.
            """
            if not (self.mode_var.get() == "1D"
                    and self.multi_var.get() == "Yes"):
                return                              # button is disabled anyway
            if not self.multi_windows:
                self._mb.showinfo(
                    "Animate", "Create a multiplot window first "
                    "(Add Multiplot), then press Animate.")
                return
            title = self._current_multi_title()
            if title is None:
                title = next(reversed(self.multi_windows))
            fields = self.multi_windows[title]

            init = self.folder_path or (
                os.path.dirname(self.loaded_path) if self.loaded_path else None)
            files = self._fd.askopenfilenames(
                title="Select files to animate",
                initialdir=init or None,
                filetypes=[("NumPy archive", "*.npz"), ("All files", "*.*")])
            files = list(files)
            if len(files) < 1:
                return
            # order by peak terminal voltage (falls back to filename order)
            try:
                def _vkey(f):
                    d = np.load(f, allow_pickle=False)
                    v = float(np.asarray(d["terminal_voltages"], float).max())
                    d.close()
                    return v
                files.sort(key=_vkey)
            except Exception:                          # noqa: BLE001
                files.sort()

            self._stop_animation()
            self._anim = dict(title=title, fields=fields, files=files, idx=0)
            self._animate_step()

        def _animate_step(self):
            a = self._anim
            if not a:
                return
            if a.get("kind") == "vector":
                self._vec_animate_step()
                return
            if a["idx"] >= len(a["files"]):
                self._stop_animation()
                return
            from tkinter import ttk
            f = a["files"][a["idx"]]
            try:
                d = np.load(f, allow_pickle=False)
                tri = build_triangulation(d)
                along = self.dir_var.get()
                if self.cut and self.cut["along"] == along:
                    fixed = self.cut["fixed"]
                    lo, hi = self.cut["lo"], self.cut["hi"]
                else:
                    fixed, lo, hi = default_cut(d["nodes"], along)
                fig, canvas = self._get_or_create_tab(ttk, a["title"])
                draw_1d(fig, d, tri, a["fields"], along, fixed, lo, hi,
                        self.log_var.get())
                fig.suptitle("[%s]  (%d/%d)"
                             % (operating_point(d), a["idx"] + 1, len(a["files"])))
                try:
                    fig.tight_layout()
                except Exception:                      # noqa: BLE001
                    pass
                canvas.draw()
                d.close()                              # free memory each frame
            except Exception as exc:                   # noqa: BLE001
                self._mb.showerror("Animate failed",
                                   "%s\n\n%s" % (os.path.basename(f), exc))
                self._stop_animation()
                return
            a["idx"] += 1
            self._anim_after = self.after(700, self._animate_step)

        def _open_multiplot_dialog(self, initial=None, on_ok=None):
            """Modal 'number of curves + per-curve field' picker.

            On OK the chosen (non-empty) field list is passed to *on_ok*.
            """
            if not self.fields:
                self._mb.showwarning("No file", "Load a .npz first.")
                return
            initial = list(initial or [])
            tk = __import__("tkinter")
            from tkinter import ttk
            win = tk.Toplevel(self)
            win.title("Multiplot setup")
            win.transient(self)
            win.grab_set()

            ttk.Label(win, text="Number of curves:").grid(
                row=0, column=0, padx=8, pady=8, sticky="w")
            count_var = tk.IntVar(value=max(2, len(initial)))
            count_cb = ttk.Combobox(win, textvariable=count_var, width=5,
                                    state="readonly",
                                    values=list(range(1, 11)))
            count_cb.grid(row=0, column=1, padx=8, pady=8, sticky="w")

            fld_frame = ttk.Frame(win)
            fld_frame.grid(row=1, column=0, columnspan=2, padx=8, sticky="w")
            combos = []

            def rebuild(*_):
                for w in fld_frame.winfo_children():
                    w.destroy()
                combos.clear()
                for i in range(int(count_var.get())):
                    ttk.Label(fld_frame, text="Curve %d:" % (i + 1)).grid(
                        row=i, column=0, sticky="e", pady=2)
                    cb = ttk.Combobox(fld_frame, values=self.fields, width=26,
                                      state="readonly")
                    if i < len(initial):
                        cb.set(initial[i])
                    cb.grid(row=i, column=1, sticky="w", padx=4, pady=2)
                    combos.append(cb)

            count_cb.bind("<<ComboboxSelected>>", rebuild)
            rebuild()

            def ok():
                sel = [c.get() for c in combos if c.get()]
                win.destroy()
                if on_ok is not None:
                    on_ok(sel)

            def cancel():
                win.destroy()

            ttk.Button(win, text="OK", command=ok).grid(
                row=2, column=0, pady=10)
            ttk.Button(win, text="Cancel", command=cancel).grid(
                row=2, column=1, pady=10)

        # ---- callbacks ----------------------------------------------------
        def on_add_folder(self):
            """Pick a folder; 'Load .npz' and 'Animate' then work from it."""
            folder = self._fd.askdirectory(title="Select a solutions folder")
            if not folder:
                return
            self.folder_path = folder
            self.dir_lbl.config(text=folder)

        def on_load(self):
            path = self._fd.askopenfilename(
                title="Select a solution .npz",
                initialdir=self.folder_path or None,
                filetypes=[("NumPy archive", "*.npz"), ("All files", "*.*")])
            if not path:
                return
            try:
                data = np.load(path, allow_pickle=False)
                fields = plottable_fields(data)
                if not fields:
                    raise ValueError("No per-node field arrays in this file.")
                if self.data is not None:          # free the previous handle
                    try:
                        self.data.close()
                    except Exception:              # noqa: BLE001
                        pass
                self.data = data
                self.loaded_path = path
                self.tri = build_triangulation(data)
                # derived band-energy fields (Ec/Ev/Ei) when the file supports them
                derived = list(DERIVED) if band_diagram_available(data) else []
                self.fields = fields + derived
            except Exception as exc:                       # noqa: BLE001
                self._mb.showerror("Load failed", str(exc))
                return

            self.field_cb["values"] = self.fields
            self.field_var.set("potential" if "potential" in fields
                               else fields[0])
            self._on_field_change()
            op = operating_point(data)
            self.dir_lbl.config(text=os.path.dirname(path) or ".")
            self.file_lbl.config(
                text="%s   [%s]" % (os.path.basename(path), op) if op
                else os.path.basename(path))
            self._on_dir_change()              # set labels + prefill cut
            if self.geom_var.get():            # refresh open tabs for the new file
                self._on_geom_toggle()
            if self.band_var.get():
                self._on_band_toggle()
            if self.vis_on.get():              # re-scan terminals for the new file
                self._scan_terminals()

        def on_set_cut(self):
            try:
                fixed = float(self.e_fixed.get()) * 1e-6
                lo = float(self.e_lo.get()) * 1e-6
                hi = float(self.e_hi.get()) * 1e-6
            except ValueError:
                self._mb.showerror("Bad input",
                                   "Enter numeric cut values (in um).")
                return
            if hi <= lo:
                self._mb.showerror("Bad range", "max must exceed min.")
                return
            self.cut = dict(along=self.dir_var.get(),
                            fixed=fixed, lo=lo, hi=hi)
            self._mb.showinfo("Cut set",
                              "Cut points updated for %s-cut." % self.cut["along"])

        def on_clean_plot_area(self):
            self._stop_animation()
            # close every open plot tab
            for title in list(self.tabs.keys()):
                self._close_title(title)
            # reset multiplot state
            self.multi_windows = {}
            self._multi_counter = 0
            # reset all left-panel controls to defaults
            self.mode_var.set("1D")
            self.multi_var.set("No")
            self.geom_var.set(False)
            self.band_var.set(False)
            self.log_var.set(False)
            self.style_var.set("tcad")
            self.dir_var.set("x")
            # reset the Current-density Visualizer back to Off + clear its inputs
            self.vis_on.set(False)
            self.vis_off.set(True)
            for w in self.term_frame.winfo_children():
                w.destroy()
            self.term_boxes = []
            self.terminals = []
            self.hold_term_cb.configure(values=[])
            self.ramp_term_cb.configure(values=[])
            for v in (self.hold_term, self.ramp_term, self.hold_var,
                      self.ramp_start, self.ramp_step, self.ramp_max):
                v.set("")
            self._update_vec_vis_state()
            if self.fields:
                self.field_var.set("potential" if "potential" in self.fields
                                   else self.fields[0])
            self.cut = None
            # refresh widget states + cut entries
            self._update_mode_states()          # also refreshes field/add-multi/details
            self._on_dir_change()               # relabels + prefills cut entries

        _VEC_MAP = {
            "Total (J)":     ("current_density_x", "current_density_y",
                              "Total current density"),
            "Electron (Jn)": ("electron_current_density_x",
                              "electron_current_density_y",
                              "Electron current density"),
            "Hole (Jp)":     ("hole_current_density_x", "hole_current_density_y",
                              "Hole current density"),
        }

        def _on_vis_toggle(self, which):
            """Keep the Visualizer On/Off tick boxes mutually exclusive."""
            if which == "on":
                self.vis_on.set(True)
                self.vis_off.set(False)
                self._scan_terminals()          # discover + populate on turn-On
            else:
                self.vis_off.set(True)
                self.vis_on.set(False)
                self._stop_animation()          # leaving On stops any sweep
            self._update_vec_vis_state()

        def _update_vec_vis_state(self):
            """Terminal selectors + bias inputs are active only when On."""
            on = self.vis_on.get()
            for w in (self.hold_entry, self.ramp_start_e,
                      self.ramp_step_e, self.ramp_max_e):
                w.configure(state="normal" if on else "disabled")
            for cb in (self.hold_term_cb, self.ramp_term_cb):
                cb.configure(state="readonly" if on else "disabled")

        def _vec_folder(self):
            """Folder to scan for the sweep, or None."""
            return (os.path.dirname(self.loaded_path) if self.loaded_path
                    else None) or self.folder_path

        def _terminal_names(self):
            """Canonical terminal-name list for the loaded device, or []."""
            if self.data is not None and "terminal_names" in self.data.files:
                return [str(x) for x in self.data["terminal_names"]]
            folder = self._vec_folder()
            if folder:
                import glob
                for f in sorted(glob.glob(os.path.join(folder, "*.npz"))):
                    try:
                        d = np.load(f, allow_pickle=False)
                        names = ([str(x) for x in d["terminal_names"]]
                                 if "terminal_names" in d.files else [])
                        d.close()
                    except Exception:                  # noqa: BLE001
                        continue
                    if names:
                        return names
            return []

        def _scan_terminals(self):
            """Discover terminals from the folder and fill the Terminals display
            + Hold/Ramp dropdowns.  Names come from the stored terminal_names,
            never from file names."""
            from tkinter import ttk
            names = self._terminal_names()
            self.terminals = names

            # (re)build the read-only Terminals boxes -- one per terminal, so
            # 2-, 3- or 4-terminal devices all show correctly (small font).
            for w in self.term_frame.winfo_children():
                w.destroy()
            self.term_boxes = []
            for i, nm in enumerate(names):
                e = ttk.Entry(self.term_frame, width=9, justify="center",
                              font=("TkDefaultFont", 10))
                e.insert(0, nm)
                e.configure(state="readonly")
                e.grid(row=0, column=i, padx=1)
                self.term_boxes.append(e)

            self.hold_term_cb.configure(values=names)
            self.ramp_term_cb.configure(values=names)
            if self.hold_term.get() not in names:
                self.hold_term.set("")
            if self.ramp_term.get() not in names:
                self.ramp_term.set("")
            self._prefill_bias_defaults(names)

        def _prefill_bias_defaults(self, names):
            """Best-effort defaults from the folder's operating points: ramp =
            the widest-varying terminal, hold = the largest-|bias| held one,
            plus the detected sweep range.  Only fills fields left blank."""
            if not names or self.data is None \
                    or "terminal_voltages" not in self.data.files:
                return
            import glob
            ref = np.asarray(self.data["terminal_voltages"], float)
            rows = [ref]
            folder = self._vec_folder()
            if folder:
                for f in sorted(glob.glob(os.path.join(folder, "*.npz"))):
                    try:
                        d = np.load(f, allow_pickle=False)
                        same = ("terminal_names" in d.files
                                and [str(x) for x in d["terminal_names"]] == names)
                        if same:
                            rows.append(np.asarray(d["terminal_voltages"], float))
                        d.close()
                    except Exception:                  # noqa: BLE001
                        continue
            V = np.array(rows)
            spread = V.max(axis=0) - V.min(axis=0)
            ramp_i = int(np.argmax(spread))
            held = [i for i in range(len(names)) if i != ramp_i]
            hold_i = max(held, key=lambda i: abs(ref[i])) if held else ramp_i
            if not self.ramp_term.get():
                self.ramp_term.set(names[ramp_i])
            if not self.hold_term.get():
                self.hold_term.set(names[hold_i])
            if not self.hold_var.get().strip():
                self.hold_var.set("%g" % ref[hold_i])
            uniq = np.unique(np.round(V[:, ramp_i], 9))
            if len(uniq) >= 2:
                if not self.ramp_start.get().strip():
                    self.ramp_start.set("%g" % uniq[0])
                if not self.ramp_max.get().strip():
                    self.ramp_max.set("%g" % uniq[-1])
                if not self.ramp_step.get().strip():
                    self.ramp_step.set("%g" % float(np.min(np.diff(uniq))))

        def _parse_bias_inputs(self):
            """Parse Hold/Ramp bias entries into (hold, [ramp values]).

            Returns None (after a warning) on any invalid input.  Values may be
            positive, negative, integer or decimal.
            """
            def _f(s, what):
                s = s.strip()
                if s == "":
                    raise ValueError("%s is empty." % what)
                return float(s)
            try:
                hold = _f(self.hold_var.get(), "Hold Bias")
                start = _f(self.ramp_start.get(), "Ramp start")
                step = _f(self.ramp_step.get(), "Ramp step")
                vmax = _f(self.ramp_max.get(), "Ramp max")
            except ValueError as exc:
                self._mb.showwarning("Bias input", str(exc))
                return None
            if step == 0:
                self._mb.showwarning("Bias input", "Ramp step cannot be 0.")
                return None
            if (vmax - start) * step < 0:
                self._mb.showwarning(
                    "Bias input",
                    "Ramp step sign (%g) does not lead from start (%g) to "
                    "max (%g)." % (step, start, vmax))
                return None
            n = int(round((vmax - start) / step))
            ramp = [start + k * step for k in range(n + 1)]
            return hold, ramp

        def on_plot_vector(self):
            """Plot current density.

            Visualizer Off: quiver/streamline for the LOADED file only.
            Visualizer On : animate the selected field/style across every .npz
            in the loaded file's folder whose stored terminal voltages match the
            Hold Bias and the Ramp Bias sweep -- matched by voltage value, never
            by file name, so any device / terminal naming works.
            """
            if self.data is None:
                self._mb.showwarning("No file", "Load a .npz first.")
                return
            xkey, ykey, title = self._VEC_MAP[self.vec_var.get()]
            style = self.vec_style.get()
            if xkey not in self.data.files:
                self._mb.showwarning("Missing field",
                    "This .npz has no %s (re-save with current density)." % xkey)
                return
            if self.vis_on.get():
                self._plot_vector_visualizer(xkey, ykey, title, style)
                return
            from tkinter import ttk
            fig, canvas = self._get_or_create_tab(ttk, "%s (%s)" % (title, style))
            try:
                draw_vector_2d(fig, self.data, xkey, ykey, title, style,
                               logmag=self.log_var.get())
                fig.suptitle("[%s]" % operating_point(self.data))
                try:
                    fig.tight_layout()
                except Exception:                      # noqa: BLE001
                    pass
                canvas.draw()
            except Exception as exc:                   # noqa: BLE001
                self._mb.showerror("Plot failed", str(exc))

        def _plot_vector_visualizer(self, xkey, ykey, title, style):
            """Build and start the bias-sweep animation for Visualizer=On.

            Hold and Ramp terminals are chosen explicitly, so selection is
            unambiguous: a folder .npz is included iff its Hold Terminal sits at
            the Hold Bias, its Ramp Terminal sits at one of the ramp voltages,
            and every OTHER terminal matches the loaded reference operating
            point.  Terminals are addressed by name -- never by file name -- and
            the value-guessing that broke on a ground terminal at 0 V is gone.
            """
            import glob
            names = self.terminals or self._terminal_names()
            hterm, rterm = self.hold_term.get(), self.ramp_term.get()
            if not names or hterm not in names or rterm not in names:
                self._mb.showwarning(
                    "Visualizer",
                    "Select a Hold Terminal and a Ramp Terminal first.")
                return
            if hterm == rterm:
                self._mb.showwarning(
                    "Visualizer",
                    "Hold Terminal and Ramp Terminal must be different.")
                return
            parsed = self._parse_bias_inputs()
            if parsed is None:
                return
            hold, ramp = parsed
            folder = self._vec_folder()
            if not folder:
                self._mb.showwarning(
                    "No folder", "Load a .npz from a folder of solutions first.")
                return

            hi, ri = names.index(hterm), names.index(rterm)
            # the loaded file fixes the "other" terminals (e.g. a grounded
            # contact); requiring them to match keeps a different bias family
            # from leaking into the sweep.
            ref = (np.asarray(self.data["terminal_voltages"], float)
                   if self.data is not None
                   and "terminal_voltages" in self.data.files else None)
            tol = 1e-6

            match = {}                              # ramp value -> file path
            for f in sorted(glob.glob(os.path.join(folder, "*.npz"))):
                try:
                    d = np.load(f, allow_pickle=False)
                    ok = ("terminal_names" in d.files
                          and "terminal_voltages" in d.files
                          and xkey in d.files
                          and [str(x) for x in d["terminal_names"]] == names)
                    volts = (np.asarray(d["terminal_voltages"], float)
                             if ok else None)
                    d.close()
                except Exception:                  # noqa: BLE001
                    continue
                if not ok:
                    continue
                if abs(volts[hi] - hold) > tol:                    # hold matches?
                    continue
                r = next((rv for rv in ramp
                          if abs(volts[ri] - rv) <= tol), None)     # ramp matches?
                if r is None:
                    continue
                if ref is not None and not all(                    # others anchored?
                        abs(volts[k] - ref[k]) <= tol
                        for k in range(len(names)) if k not in (hi, ri)):
                    continue
                match.setdefault(round(r, 9), f)

            # keep sweep order (works for ascending or descending ramps)
            ordered = [match[round(r, 9)] for r in ramp if round(r, 9) in match]
            if not ordered:
                step = ramp[1] - ramp[0] if len(ramp) > 1 else 0
                self._mb.showwarning(
                    "Visualizer",
                    "No .npz in\n%s\nmatches %s = %g V (hold) with %s swept "
                    "over %g:%g:%g V.\n(Terminals matched by name against each "
                    "file's stored voltages.)"
                    % (folder, hterm, hold, rterm, ramp[0], step, ramp[-1]))
                return

            from tkinter import ttk
            self._stop_animation()
            atitle = "%s (%s)" % (title, style)
            self._anim = dict(kind="vector", title=atitle, files=ordered,
                              xkey=xkey, ykey=ykey, vtitle=title, style=style,
                              idx=0, loop=True)
            self._get_or_create_tab(ttk, atitle)   # create/select before frame 1
            self._animate_step()

        def _vec_animate_step(self):
            """One frame of the Visualizer sweep, looping continuously."""
            a = self._anim
            if not a:
                return
            if a["idx"] >= len(a["files"]):
                if a.get("loop") and a["files"]:
                    a["idx"] = 0
                else:
                    self._stop_animation()
                    return
            from tkinter import ttk
            f = a["files"][a["idx"]]
            try:
                d = np.load(f, allow_pickle=False)
                fig, canvas = self._get_or_create_tab(ttk, a["title"])
                draw_vector_2d(fig, d, a["xkey"], a["ykey"], a["vtitle"],
                               a["style"], logmag=self.log_var.get())
                fig.suptitle("[%s]   (%d/%d)"
                             % (operating_point(d), a["idx"] + 1,
                                len(a["files"])))
                try:
                    fig.tight_layout()
                except Exception:                  # noqa: BLE001
                    pass
                canvas.draw()
                d.close()
            except Exception as exc:               # noqa: BLE001
                self._mb.showerror("Visualizer failed",
                                   "%s\n\n%s" % (os.path.basename(f), exc))
                self._stop_animation()
                return
            a["idx"] += 1
            self._anim_after = self.after(700, self._animate_step)

        def on_plot(self):
            if self.data is None:
                self._mb.showwarning("No file", "Load a .npz first.")
                return
            from tkinter import ttk
            mode = self.mode_var.get()
            # In 1D multiplot mode, the Plot button (re)draws the currently
            # active multiplot window with the current log-scale / cut
            # settings.  New windows are created only via "Add Multiplot".
            if mode == "1D" and self.multi_var.get() == "Yes":
                self._plot_current_multiplot()
                return
            log = self.log_var.get()
            field = self.field_var.get()
            try:
                if mode == "1D":
                    if not field:
                        self._mb.showwarning("No field", "Select a field.")
                        return
                    along = self.dir_var.get()
                    if self.cut and self.cut["along"] == along:
                        fixed = self.cut["fixed"]
                        lo, hi = self.cut["lo"], self.cut["hi"]
                    else:
                        fixed, lo, hi = default_cut(self.data["nodes"], along)
                    title = "%s (1D)" % field
                    fig, canvas = self._get_or_create_tab(ttk, title)
                    draw_1d(fig, self.data, self.tri, [field],
                            along, fixed, lo, hi, log)
                else:
                    if not field:
                        self._mb.showwarning("No field", "Select a field.")
                        return
                    title = "%s (%s)" % (field, mode)
                    fig, canvas = self._get_or_create_tab(ttk, title)
                    if mode == "2D":
                        draw_2d(fig, self.data, self.tri, field, log)
                    else:
                        draw_3d(fig, self.data, self.tri, field, log,
                                style=self.style_var.get())

                fig.suptitle("[%s]" % operating_point(self.data))
                if mode != "3D":
                    try:
                        fig.tight_layout()
                    except Exception:                      # noqa: BLE001
                        pass
                canvas.draw()
            except Exception as exc:                       # noqa: BLE001
                self._mb.showerror("Plot failed", str(exc))

    SolutionViewer().mainloop()


if __name__ == "__main__":
    run_gui()
