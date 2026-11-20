"""Abbildungen (statisch, Poster-tauglich). Farben nach validierter Palette:
Kategorien in fester Reihenfolge, Sequenzen einfarbig hell->dunkel, Baselines/unzulässige
Protokolle in Grau, Referenzlinien (Zufallsniveau) dünn und beschriftet."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import textwrap  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GRAY_BAR = "#c3c2b7"
BLUES = LinearSegmentedColormap.from_list("blues", ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                                                    "#256abf", "#184f95", "#0d366b"])
SENSOR_COLORS = {"S01": SERIES[0], "S02": SERIES[1], "S05": SERIES[2], "S06": SERIES[3], "mehrere": MUTED}

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "savefig.dpi": 200, "savefig.bbox": "tight", "legend.frameon": False,
})


def _save(fig, path):
    fig.savefig(path)
    plt.close(fig)


def _chance(ax, chance, horizontal=True, label=True):
    if chance is None:
        return
    if horizontal:
        ax.axvline(chance, color=MUTED, lw=1)
        if label:
            ax.text(chance, 1.005, f" Zufall {chance:.0%}", transform=ax.get_xaxis_transform(), color=MUTED,
                    fontsize=8, va="bottom", ha="left")
    else:
        ax.axhline(chance, color=MUTED, lw=1)
        if label:
            ax.text(1.005, chance, f"Zufall {chance:.0%}", transform=ax.get_yaxis_transform(), color=MUTED,
                    fontsize=8, va="center", ha="left")


def bars(df: pd.DataFrame, value: str, label: str, path, err_lo=None, err_hi=None, err_sd=None, gray=None,
         title="", xlabel="", chance=None, xlim=(0, 1.0), note=None):
    """Horizontale Balken (ein Wert pro Kategorie), sortiert wie übergeben (oben = erste Zeile)."""
    gray = set(gray or [])
    n = len(df)
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * n + 1.3))
    y = np.arange(n)[::-1]
    colors = [GRAY_BAR if lab in gray else SERIES[0] for lab in df[label]]
    ax.barh(y, df[value], color=colors, height=0.62, edgecolor=SURFACE, linewidth=2)
    right = df[value].to_numpy(dtype=float).copy()
    if err_sd is not None:
        sd = df[err_sd].fillna(0).to_numpy(dtype=float)
        ax.errorbar(df[value], y, xerr=sd, fmt="none", ecolor=INK2, elinewidth=1, capsize=2)
        right = right + sd
    elif err_lo is not None:
        lo = (df[value] - df[err_lo]).clip(lower=0)
        hi = (df[err_hi] - df[value]).clip(lower=0)
        ax.errorbar(df[value], y, xerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=1, capsize=2)
        right = right + hi.to_numpy(dtype=float)
    for yi, v, rx in zip(y, df[value], right):
        ax.text(min(rx, xlim[1]) + 0.012 * (xlim[1] - xlim[0]), yi, f"{v:.0%}" if xlim[1] <= 1.0 else f"{v:.2f}",
                va="center", ha="left", fontsize=9, color=INK)
    ax.set_yticks(y, df[label])
    ax.set_xlim(*xlim)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    _chance(ax, chance)
    ax.set_title(title, loc="left", pad=16)
    ax.set_xlabel(xlabel)
    if note:
        fig.text(0.01, -0.02, note, fontsize=8, color=MUTED, ha="left", va="top", wrap=True)
    _save(fig, path)


def heatmap(mat: pd.DataFrame, path, title="", cbar_label="", vmin=0.0, vmax=1.0, annotate=True, fmt="{:.0%}",
            xlabel="", ylabel="", figsize=None, mark_missing=True):
    r, c = mat.shape
    fig, ax = plt.subplots(figsize=figsize or (0.85 * c + 2.6, 0.42 * r + 1.6))
    vals = mat.to_numpy(dtype=float)
    im = ax.imshow(np.ma.masked_invalid(vals), cmap=BLUES, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(c), mat.columns, rotation=30, ha="right")
    ax.set_yticks(range(r), mat.index)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(c + 1) - .5, minor=True)
    ax.set_yticks(np.arange(r + 1) - .5, minor=True)
    ax.grid(which="minor", color=SURFACE, lw=2)
    ax.tick_params(which="minor", length=0)
    if annotate:
        for i in range(r):
            for j in range(c):
                v = vals[i, j]
                if np.isnan(v):
                    if mark_missing:
                        ax.text(j, i, "–", ha="center", va="center", color=MUTED, fontsize=8)
                    continue
                dark = (v - vmin) / (vmax - vmin + 1e-12) > 0.55
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8, color="#ffffff" if dark else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.outline.set_visible(False)
    cb.set_label(cbar_label, color=INK2)
    if vmax <= 1.0:
        cb.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _save(fig, path)


def confusion(cm, labels, path, title=""):
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    n = len(labels)
    fig, ax = plt.subplots(figsize=(0.28 * n + 2.4, 0.28 * n + 1.6))
    im = ax.imshow(cmn, cmap=BLUES, vmin=0, vmax=1)
    ax.set_xticks(range(n), labels, rotation=90, fontsize=7)
    ax.set_yticks(range(n), labels, fontsize=7)
    ax.set_xlabel("vorhergesagte Person")
    ax.set_ylabel("wahre Person")
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    cb.set_label("Anteil je wahrer Person", color=INK2)
    ax.set_title(title, loc="left")
    _save(fig, path)


def lines(series: dict, path, x="seconds", y="acc", title="", xlabel="", ylabel="", chance=None, ylim=(0, 1.02)):
    """Bis zu 4 Linien (feste Farbreihenfolge), Legende + Direktbeschriftung am Linienende."""
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ends = []
    for k, (name, d) in enumerate(series.items()):
        col = SERIES[k % len(SERIES)]
        ax.plot(d[x], d[y], color=col, lw=2, marker="o", ms=5, label=name, markeredgecolor=SURFACE, mew=1.5)
        ends.append([float(d[x].iloc[-1]), float(d[y].iloc[-1]), float(d[y].iloc[-1])])
    # Endbeschriftungen ohne Überlappung (Mindestabstand in Datenkoordinaten)
    span, last = (ylim[1] - ylim[0]) * 0.05, -np.inf
    for i in sorted(range(len(ends)), key=lambda i: ends[i][1]):
        ends[i][2] = max(ends[i][1], last + span)
        last = ends[i][2]
    for xe, ye, yl in ends:
        ax.text(xe, yl, f"  {ye:.0%}", color=INK2, fontsize=8, va="center")
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    _chance(ax, chance, horizontal=False)
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if len(series) > 1:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=min(3, len(series)))
    _save(fig, path)


def grouped_bars(df: pd.DataFrame, cat: str, group: str, value: str, path, title="", ylabel="", chance=None,
                 err_lo=None, err_hi=None, cat_order=None, group_order=None, note=None):
    cats = cat_order or list(dict.fromkeys(df[cat]))
    groups = group_order or list(dict.fromkeys(df[group]))
    w = 0.8 / len(groups)
    fig, ax = plt.subplots(figsize=(max(6.5, 1.25 * len(cats) + 2), 4.0))
    for k, g in enumerate(groups):
        d = df[df[group] == g].set_index(cat).reindex(cats)
        x = np.arange(len(cats)) - 0.4 + w * (k + 0.5)
        ax.bar(x, d[value], width=w, color=SERIES[k], label=g, edgecolor=SURFACE, linewidth=2)
        if err_lo is not None:
            ax.errorbar(x, d[value], yerr=[(d[value] - d[err_lo]).clip(lower=0), (d[err_hi] - d[value]).clip(lower=0)],
                        fmt="none", ecolor=INK2, elinewidth=1, capsize=2)
        for xi, v in zip(x, d[value]):
            if not np.isnan(v):
                ax.text(xi, v + 0.015, f"{v:.0%}", ha="center", va="bottom", fontsize=7.5, color=INK2)
    wrapped = [textwrap.fill(str(c), 18, break_long_words=False) for c in cats]
    ax.set_xticks(np.arange(len(cats)), wrapped, rotation=0)
    n_lines = max(w.count("\n") + 1 for w in wrapped)
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    _chance(ax, chance, horizontal=False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title(title, loc="left")
    ax.set_ylabel(ylabel)
    if note:
        fig.text(0.01, -0.01 - 0.045 * n_lines, note, fontsize=8, color=MUTED, ha="left", va="top")
    _save(fig, path)


def importance(df: pd.DataFrame, path, title="", top=20):
    d = df.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.6, 0.3 * len(d) + 1.4))
    order = [s for s in dict.fromkeys(df["sensor"]) if s != "mehrere"]
    cmap = {s: SENSOR_COLORS.get(s, SERIES[k % len(SERIES)]) for k, s in enumerate(order)}
    cmap["mehrere"] = MUTED
    cols = [cmap.get(s, MUTED) for s in d["sensor"]]
    ax.barh(range(len(d)), d["importance"], xerr=d.get("importance_sd"), color=cols, height=0.65,
            edgecolor=SURFACE, linewidth=2, error_kw={"ecolor": INK2, "elinewidth": 0.8})
    ax.set_yticks(range(len(d)), d["feature"], fontsize=8)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.set_xlabel("Abfall der Clip-Accuracy bei Permutation")
    shown = [s for s in cmap if s in set(d["sensor"])]
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap[s]) for s in shown]
    labels = shown
    ax.legend(handles, labels, loc="lower right", title="Sensor")
    ax.set_title(title, loc="left")
    _save(fig, path)


def permutation_strip(rows: list[dict], path, title="", chance=None):
    """Beobachteter Wert vs. Nullverteilung (Label-Permutation auf Trial-Ebene) je Protokoll."""
    fig, ax = plt.subplots(figsize=(7.2, 1.0 * len(rows) + 1.4))
    rng = np.random.default_rng(0)
    for k, r in enumerate(rows[::-1]):
        null = np.asarray(r["null"])
        ax.scatter(null, k + rng.uniform(-.15, .15, len(null)), s=14, color=GRAY_BAR, label="permutierte Labels" if k == 0 else None)
        ax.scatter([r["observed"]], [k], s=90, marker="D", color=SERIES[0], edgecolor=SURFACE, lw=1.5,
                   label="echte Labels" if k == 0 else None, zorder=3)
        ax.text(r["observed"], k + 0.28, f"{r['observed']:.0%}  (p={r['p_value']:.3f})", ha="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(rows)), [r["label"] for r in rows[::-1]])
    ax.set_xlim(0, 1.05)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    _chance(ax, chance)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2)
    ax.set_xlabel("Clip-Accuracy")
    ax.set_title(title, loc="left", pad=28)
    _save(fig, path)


def embedding(emb, persons, path, highlight=None, title=""):
    """2D-Einbettung: alle Personen grau, max. 3 hervorgehoben (Farbe + Direktlabel)."""
    persons = np.asarray(persons)
    uniq = list(dict.fromkeys(persons))
    highlight = highlight or uniq[:3]
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    m = ~np.isin(persons, highlight)
    ax.scatter(emb[m, 0], emb[m, 1], s=8, color=GRAY_BAR, alpha=0.7, lw=0, label="andere Personen")
    for k, p in enumerate(highlight):
        mm = persons == p
        ax.scatter(emb[mm, 0], emb[mm, 1], s=14, color=SERIES[k], lw=0, label=p)
        cx, cy = np.median(emb[mm], axis=0)
        ax.text(cx, cy, f" {p}", fontsize=9, fontweight="bold", color=INK)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title(title, loc="left")
    _save(fig, path)


def split_diagram(path):
    """Schema: Random-Split vs. Block-Split mit Gap vs. Cross-Activity (für Poster/Doku)."""
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 3.6), sharex=True)
    n = 20
    rng = np.random.default_rng(3)
    rows = [("Random-Fenster/-Clips (unzulässig)", rng.choice(["tr", "te"], n, p=[.8, .2])),
            ("Block-Split mit Gap (Hauptprotokoll)", np.array(["tr"] * 7 + ["gap"] + ["te"] * 4 + ["gap"] + ["tr"] * 7)),
            ("Cross-Activity (neue Aufnahme)", None)]
    colors = {"tr": SERIES[0], "te": SERIES[1], "gap": "#e1e0d9"}
    for ax, (name, seq) in zip(axes, rows):
        if seq is None:
            ax.barh(0, 9.6, left=0, color=SERIES[0], height=.6, edgecolor=SURFACE, lw=2)
            ax.barh(0, 9.6, left=10.4, color=SERIES[1], height=.6, edgecolor=SURFACE, lw=2)
            ax.text(4.8, 0, "Training: andere Aktivitäten", ha="center", va="center", color="#ffffff", fontsize=8)
            ax.text(15.2, 0, "Test: Gehen (komplett ungesehen)", ha="center", va="center", color="#ffffff", fontsize=8)
        else:
            for k, s in enumerate(seq):
                ax.barh(0, 1, left=k, color=colors[s], height=.6, edgecolor=SURFACE, lw=2)
        ax.set_yticks([])
        ax.set_title(name, loc="left", fontsize=9)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[-1].set_xticks([])
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[k]) for k in ("tr", "te", "gap")]
    fig.legend(handles, ["Training", "Test", "Gap (verworfen)"], loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Train und Test müssen zeitlich getrennt sein (Zeitachse eines Gehen-Trials, Kästchen = Clip)", x=0.01, ha="left", fontsize=10, color=INK2)
    plt.tight_layout()
    _save(fig, path)
