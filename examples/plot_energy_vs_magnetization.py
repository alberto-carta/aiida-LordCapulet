"""Plot total energy vs total magnetization for each example JSON in examples/0[1-9]_*.

One HTML per example folder at examples/plots/<example>_energy_vs_magnetization.html.
Scatter + right-side horizontal histogram (# states vs ΔE). ΔE = E − E_min per
example (each plot zeroed independently). Hover shows source label, pk, |M|, E_abs.
"""
#%%
import glob
import json
import os

import plotly.graph_objects as go
from plotly.subplots import make_subplots

EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(EXAMPLES_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

SOURCE_LABELS = {
    'afm_workchain': 'standard magnetic search',
    'constrained_scan': 'constrained scan',
}

EXAMPLE_DIRS = sorted(
    d for d in glob.glob(os.path.join(EXAMPLES_DIR, '0[1-9]_*')) if os.path.isdir(d)
)


def _infer_constrained_label(example_name: str) -> str:
    name = example_name.lower()
    if 'random' in name:
        return 'random proposal'
    if 'gp' in name or 'bayes' in name or 'gaussian' in name:
        return 'GP proposal'
    return 'constrained scan'


def _iter_records(example_dir):
    for json_path in glob.glob(os.path.join(example_dir, '*.json')):
        with open(json_path, 'r') as f:
            data = json.load(f)
        calcs = data.get('calculations', {})
        for pk, calc in calcs.items():
            out = calc.get('output_parameters') or {}
            energy = out.get('energy')
            tot_mag = out.get('total_magnetization')
            abs_mag = out.get('absolute_magnetization')
            if energy is None or tot_mag is None:
                continue
            src = calc.get('calculation_source', 'unknown')
            label = SOURCE_LABELS.get(src, src)
            if src == 'constrained_scan':
                label = _infer_constrained_label(os.path.basename(example_dir))
            yield {
                'pk': pk,
                'energy_abs': energy,
                'tot_mag': tot_mag,
                'abs_mag': abs_mag,
                'source': src,
                'label': label,
                'example': os.path.basename(example_dir),
                'json': os.path.basename(json_path),
            }


def _build_figure(records, example_name):
    e_min = min(r['energy_abs'] for r in records)
    for r in records:
        r['energy'] = r['energy_abs'] - e_min

    title = f"{example_name} — total energy vs total magnetization  (E_min={e_min:.4f} eV)"

    by_label = {}
    for r in records:
        by_label.setdefault(r['label'], []).append(r)

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.78, 0.22],
        shared_yaxes=True,
        horizontal_spacing=0.02,
        subplot_titles=('E vs M_tot', '# states'),
    )
    for label, items in by_label.items():
        fig.add_trace(
            go.Scatter(
                x=[r['tot_mag'] for r in items],
                y=[r['energy'] for r in items],
                mode='markers',
                name=label,
                legendgroup=label,
                marker=dict(size=8, opacity=0.8),
                customdata=[[r['pk'], r['example'], r['abs_mag'], r['json'], r['energy_abs']] for r in items],
                hovertemplate=(
                    '<b>source:</b> ' + label +
                    '<br><b>example:</b> %{customdata[1]}'
                    '<br><b>pk:</b> %{customdata[0]}'
                    '<br><b>M_tot:</b> %{x:.4f} μB'
                    '<br><b>|M|:</b> %{customdata[2]:.4f} μB'
                    '<br><b>ΔE:</b> %{y:.6f} eV'
                    '<br><b>E_abs:</b> %{customdata[4]:.6f} eV'
                    '<br><b>file:</b> %{customdata[3]}<extra></extra>'
                ),
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Histogram(
                y=[r['energy'] for r in items],
                name=label,
                legendgroup=label,
                showlegend=False,
                opacity=0.75,
                nbinsy=30,
            ),
            row=1, col=2,
        )

    fig.update_layout(
        title=title,
        legend_title='calculation source',
        template='plotly_white',
        hovermode='closest',
        barmode='stack',
        width=1200,
        height=750,
    )
    fig.update_xaxes(title_text='total magnetization  (μ<sub>B</sub>)', row=1, col=1)
    fig.update_yaxes(title_text='ΔE = E − E<sub>min</sub>  (eV)', row=1, col=1)
    fig.update_xaxes(title_text='# states', row=1, col=2)
    return fig, e_min


#%%
written = []
for d in EXAMPLE_DIRS:
    example_name = os.path.basename(d)
    records = list(_iter_records(d))
    if not records:
        print(f"[skip] {example_name}: no JSON with energy+total_magnetization")
        continue
    fig, e_min = _build_figure(records, example_name)
    out_html = os.path.join(PLOTS_DIR, f"{example_name}_energy_vs_magnetization.html")
    fig.write_html(out_html, include_plotlyjs='cdn')
    written.append((example_name, len(records), out_html))
    print(f"[ok]   {example_name}: {len(records)} calcs → {out_html}")

print(f"\nWrote {len(written)} plot(s) to {PLOTS_DIR}")
# %%
