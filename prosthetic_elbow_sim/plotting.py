"""Paper-style figures for elbow-control trials (matplotlib, no ROS imports)."""

import numpy as np

from prosthetic_elbow_sim.features import CLASSES, label_to_class, mode_to_class

CLASS_COLOURS = {'rest': '#eeeeee', 'flex': '#fbe3d6', 'extend': '#d6e6f5', 'hold': '#e3e3c8'}


def _style(plt):
    plt.rcParams.update({
        'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.alpha': 0.25, 'figure.dpi': 110, 'savefig.dpi': 300,
    })


def _shade_phases(ax, t, intended):
    classes = [label_to_class(x) for x in intended]
    start = 0
    for i in range(1, len(classes) + 1):
        if i == len(classes) or classes[i] != classes[start]:
            ax.axvspan(t[start], t[i - 1], color=CLASS_COLOURS[classes[start]], lw=0, zorder=0)
            start = i


def trial_figure(path, title, t, intended, modes, activation=None, command_deg=None,
                 actual_deg=None, emg_t=None, emg=None):
    """4-panel figure: raw EMG, activation, decoded intent, elbow angle (cmd vs actual)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _style(plt)
    panels = [p for p in ('emg', 'act', 'mode', 'angle')
              if (p != 'emg' or emg is not None) and (p != 'act' or activation is not None)]
    heights = {'emg': 1.2, 'act': 1.0, 'mode': 0.6, 'angle': 1.4}
    fig, axes = plt.subplots(len(panels), 1, figsize=(7.2, 1.35 * sum(heights[p] for p in panels) + 0.6),
                             sharex=True, gridspec_kw={'height_ratios': [heights[p] for p in panels]})
    axes = np.atleast_1d(axes)
    for ax, p in zip(axes, panels):
        _shade_phases(ax, t, intended)
        if p == 'emg':
            ax.plot(emg_t, emg[:, 0], lw=0.3, color='#c0392b', label='biceps')
            ax.plot(emg_t, emg[:, 1] - 2.5 * np.std(emg[:, 0]) * 3, lw=0.3, color='#2471a3', label='triceps (offset)')
            ax.set_ylabel('sEMG')
            ax.set_yticks([])
            ax.legend(loc='upper right', fontsize=7, frameon=False, ncol=2)
        elif p == 'act':
            ax.plot(t, activation[:, 0], color='#c0392b', lw=1, label='biceps')
            ax.plot(t, activation[:, 1], color='#2471a3', lw=1, label='triceps')
            ax.set_ylabel('activation')
            ax.set_ylim(-0.02, 1.02)
            ax.legend(loc='upper right', fontsize=7, frameon=False, ncol=2)
        elif p == 'mode':
            dec = [CLASSES.index(mode_to_class(m)) for m in modes]
            ax.step(t, dec, where='post', color='black', lw=1)
            ax.set_yticks(range(len(CLASSES)))
            ax.set_yticklabels(CLASSES, fontsize=7)
            ax.set_ylabel('decoded')
            ax.grid(False)
        else:
            if command_deg is not None:
                ax.plot(t, command_deg, color='black', lw=1.2, label='commanded')
            if actual_deg is not None:
                ax.plot(t, actual_deg, color='#e67e22', lw=1, ls='--', label='simulated joint')
            ax.set_ylabel('elbow angle (deg)')
            ax.legend(loc='upper right', fontsize=7, frameon=False, ncol=2)
    axes[-1].set_xlabel('time (s)')
    handles = [plt.Rectangle((0, 0), 1, 1, color=CLASS_COLOURS[c]) for c in CLASSES]
    fig.legend(handles, [f'intended: {c}' for c in CLASSES], loc='upper center', ncol=4,
               fontsize=7, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle(title, fontsize=10, y=1.03)
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def comparison_figure(path, rows):
    """Bar charts comparing trials/decoders on the main metrics."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _style(plt)
    metrics = [('intent_accuracy_%', 'Intent accuracy (%)', True),
               ('onset_delay_ms', 'Onset delay (ms)', False),
               ('false_motion_deg', 'Unwanted motion (deg)', False),
               ('tracking_rmse_deg', 'Tracking RMSE (deg)', False)]
    metrics = [m for m in metrics
               if all(m[0] in r for r in rows) and any(r[m[0]] == r[m[0]] for r in rows)]
    names = [r['trial'] for r in rows]
    fig, axes = plt.subplots(1, len(metrics), figsize=(2.2 * len(metrics) + 0.6, 2.8))
    colours = ['#34495e', '#c0392b', '#2471a3', '#27ae60', '#8e44ad', '#d35400']
    axes = np.atleast_1d(axes)
    for ax, (key, label, higher_better) in zip(axes, metrics):
        vals = [r[key] for r in rows]
        bars = ax.bar(range(len(rows)), vals, color=colours[:len(rows)] * (1 + len(rows) // 6))
        ax.set_title(label + (' ↑' if higher_better else ' ↓'), fontsize=8)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(names, rotation=30, ha='right', fontsize=7)
        for b, v in zip(bars, vals):
            if v == v:  # not NaN
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f'{v:g}',
                        ha='center', va='bottom', fontsize=7)
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def confusion_figure(path, names_and_matrices):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _style(plt)
    n = len(names_and_matrices)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 2.6))
    axes = np.atleast_1d(axes)
    for ax, (name, m) in zip(axes, names_and_matrices):
        rowsum = m.sum(axis=1, keepdims=True)
        pct = np.divide(m, rowsum, out=np.zeros_like(m, dtype=float), where=rowsum > 0) * 100
        ax.imshow(pct, cmap='Blues', vmin=0, vmax=100)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                ax.text(j, i, f'{pct[i, j]:.0f}', ha='center', va='center', fontsize=7,
                        color='white' if pct[i, j] > 60 else 'black')
        ax.set_xticks(range(len(CLASSES)))
        ax.set_xticklabels(CLASSES, fontsize=7, rotation=30)
        ax.set_yticks(range(len(CLASSES)))
        ax.set_yticklabels(CLASSES, fontsize=7)
        ax.set_xlabel('decoded', fontsize=8)
        ax.set_ylabel('intended', fontsize=8)
        ax.set_title(name, fontsize=8)
        ax.grid(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
