"""Assignment 1 dashboard: saved-model inference, never training."""

from pathlib import Path
import json

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


# 1. Load the artifacts produced by `uv run python main.py train`.
ARTIFACT_DIR = Path("artifacts")
REQUIRED = [
    "metadata.json", "preprocessor.joblib", "sklearn_model.joblib",
    "manual_pytorch_model.pt", "pytorch_module_model.pt",
    "X_test.npy", "y_test.npy", "dashboard_test_data.csv",
    "test_predictions.csv",
    "policy_replay_summary.csv",
    "december_policy_comparison.csv",
]
missing = [name for name in REQUIRED if not (ARTIFACT_DIR / name).is_file()]
if missing:
    raise FileNotFoundError(
        "Missing training artifacts: " + ", ".join(missing)
        + ". Run `uv run python main.py train` first."
    )

with (ARTIFACT_DIR / "metadata.json").open(encoding="utf-8") as file:
    metadata = json.load(file)

preprocessor = joblib.load(ARTIFACT_DIR / "preprocessor.joblib")
sklearn_model = joblib.load(ARTIFACT_DIR / "sklearn_model.joblib")
test_rows = pd.read_csv(ARTIFACT_DIR / "dashboard_test_data.csv")
saved_predictions = pd.read_csv(ARTIFACT_DIR / "test_predictions.csv")
policy_replay = pd.read_csv(
    ARTIFACT_DIR / "policy_replay_summary.csv"
)

december_policies = pd.read_csv(
    ARTIFACT_DIR / "december_policy_comparison.csv"
)
y_true = np.load(ARTIFACT_DIR / "y_test.npy", allow_pickle=False).astype(int).ravel()


feature_columns = metadata["numeric_features"] + metadata["categorical_features"]
X_test = np.asarray(preprocessor.transform(test_rows[feature_columns]), dtype=np.float64)
saved_X_test = np.load(ARTIFACT_DIR / "X_test.npy", allow_pickle=False)

if X_test.shape != saved_X_test.shape or not np.allclose(X_test, saved_X_test, atol=1e-8):
    raise ValueError("Saved test data and preprocessor do not match X_test.npy. Rerun training.")
if not (len(test_rows) == len(y_true) == len(saved_predictions)):
    raise ValueError("Artifact row counts disagree. Rerun training.")
if not np.array_equal(test_rows[metadata["target"]].to_numpy().astype(int), y_true):
    raise ValueError("Saved target arrays disagree. Rerun training.")
if X_test.shape[1] != metadata["n_model_features"]:
    raise ValueError("Model feature count differs from metadata. Rerun training.")

# Re-create the same architecture; this only constructs layers and loads weights.
class TorchLogisticRegression(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = torch.nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x)


manual_checkpoint = torch.load(
    ARTIFACT_DIR / "manual_pytorch_model.pt", map_location="cpu", weights_only=True
)
module_checkpoint = torch.load(
    ARTIFACT_DIR / "pytorch_module_model.pt", map_location="cpu", weights_only=True
)
if (int(manual_checkpoint["n_features"]) != X_test.shape[1]
        or int(module_checkpoint["n_features"]) != X_test.shape[1]):
    raise ValueError("Model and test-feature dimensions disagree. Rerun training.")

module_model = TorchLogisticRegression(X_test.shape[1]).double()
module_model.load_state_dict(module_checkpoint["state_dict"])
module_model.eval()

with torch.inference_mode():
    X_tensor = torch.from_numpy(X_test)
    manual_logits = X_tensor @ manual_checkpoint["weights"].double() + manual_checkpoint["bias"].double()
    module_logits = module_model(X_tensor)
    manual_prob = torch.sigmoid(manual_logits).numpy().ravel()
    module_prob = torch.sigmoid(module_logits).numpy().ravel()

probabilities = {
    "Scikit-learn": sklearn_model.predict_proba(X_test)[:, 1],
    "Manual PyTorch": manual_prob,
    "PyTorch nn.Module": module_prob,
}

# Check that the live inference above reproduces the training notebook's exports.
assert np.array_equal(saved_predictions["actual"].to_numpy().astype(int), y_true)
for name, column in [
    ("Scikit-learn", "sklearn_probability"),
    ("Manual PyTorch", "manual_probability"),
    ("PyTorch nn.Module", "torch_probability"),
]:
    if not np.allclose(probabilities[name], saved_predictions[column].to_numpy(), atol=1e-6, rtol=0):
        raise ValueError(f"{name} predictions don't match test_predictions.csv. Rerun training.")

model_names = list(probabilities)
default_model = model_names[0]

# Final policy selected before December evaluation.
selected_policy = metadata["selected_policy"]
policy_thresholds = metadata["policy_thresholds"]

default_threshold = float(metadata["best_threshold"])

# Hypothetical staffing capacity.
assistance_capacity = float(
    metadata["assistance_capacity_fraction"]
)

# Cost parameters saved by the training notebook.
tp_cost = float(metadata["true_positive_cost"])
fp_cost = float(metadata["false_positive_cost"])
fn_cost = float(metadata["false_negative_cost"])

# Check that app and training agree on the selected policy.
if not np.isclose(
    default_threshold,
    policy_thresholds[selected_policy]
):
    raise ValueError(
        "Saved threshold and selected policy disagree. "
        "Rerun training."
    )

# Used only to explain why the policy was selected.
all_replay_policies_exceeded = not bool(
    policy_replay["within_capacity_in_replay"].any()
)




# 2. Descriptive metrics and visual styling. All probabilities were loaded above.
from html import escape
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score

INK = '#14263D'
MUTED = '#66788A'
TEAL = '#167C78'
CORAL = '#D66A57'
BLUE = '#4C70BC'
GRID = '#E9EEF3'
PAPER = '#FFFFFF'
PALE = '#F5F8FB'
MODEL_NAMES = list(probabilities)
FEATURE_NAMES = metadata['numeric_features'] + metadata['categorical_features']


def chart_style(fig, height=350):
    """Consistent, legible style shared by all Plotly figures."""
    fig.update_layout(
        template='plotly_white', height=height, paper_bgcolor=PAPER,
        plot_bgcolor=PAPER, font={'family': 'Arial, sans-serif', 'size': 12, 'color': INK},
        margin={'l': 55, 'r': 20, 't': 40, 'b': 50},
        legend={'orientation': 'h', 'y': 1.1, 'x': 0, 'font': {'size': 11}},
        hoverlabel={'bgcolor': INK, 'font_color': 'white'},
    )
    fig.update_xaxes(showline=False, zeroline=False, gridcolor=GRID)
    fig.update_yaxes(showline=False, zeroline=False, gridcolor=GRID)
    return fig

def service_cost(tp, fp, fn):
    """
    Hypothetical service-level cost.

    Every prioritized visitor requires assistance time.
    Each eventual purchaser who is not prioritized
    carries an additional assumed service penalty.

    This is not observed lost revenue.
    """
    return float(
        tp_cost * tp
        + fp_cost * fp
        + fn_cost * fn
    )

def classification_at(model_name, threshold):

    if model_name not in probabilities:
        raise ValueError("Unknown model")

    labels = (
        probabilities[model_name] >= float(threshold)
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        labels,
        labels=[0, 1],
    ).ravel()

    n = len(y_true)
    contacted = int(tp + fp)
    contact_rate = contacted / n

    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),

        "cost": service_cost(tp, fp, fn),

        "flagged": contacted,
        "flagged_pct": 100 * contact_rate,

        "over_capacity": (
            contact_rate > assistance_capacity + 1e-12
        ),

        "precision": (
            100 * tp / contacted
            if contacted else 0.0
        ),

        "recall": (
            100 * tp / (tp + fn)
            if tp + fn else 0.0
        ),

        "n": n,
    }


def summary_table():
    rows = []
    for name, scores in [
        ('Always negative', np.zeros(len(y_true))),
        ('Always positive', np.ones(len(y_true))),
        *probabilities.items(),
    ]:
        pred = (scores >= default_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        rows.append({
            'Model': name,
            'ROC-AUC': round(float(roc_auc_score(y_true, scores)), 4),
            'Avg precision': round(float(average_precision_score(y_true, scores)), 4),
            'Precision (%)': round(100 * tp / (tp + fp), 1) if tp + fp else 0.,
            'Recall (%)': round(100 * tp / (tp + fn), 1) if tp + fn else 0.,
            'Flagged (%)': round(100 * (tp + fp) / len(y_true), 1),
            'Service cost': f'{service_cost(tp, fp, fn):,.0f}',
        })
    return pd.DataFrame(rows)


def stat_tile(label, value, note, accent=False):
    colour = TEAL if accent else INK
    return (f'<div class="metric-card"><div class="metric-label">{escape(label)}</div>'
            f'<div class="metric-number" style="color:{colour}">{escape(str(value))}</div>'
            f'<div class="metric-note">{escape(note)}</div></div>')


def header_stats():
    selected = classification_at('Scikit-learn', default_threshold)
    auc = roc_auc_score(y_true, probabilities['Scikit-learn'])
    baseline = fn_cost * int(y_true.sum())
    reduction = (baseline - selected['cost']) / baseline * 100 if baseline else 0.
    return '<div class="metric-grid">' + ''.join([
        stat_tile('Held-out sessions', f'{len(y_true):,}', 'December test only'),
        stat_tile('Observed purchase rate', f'{100*y_true.mean():.1f}%', f'{int(y_true.sum())} purchases'),
        stat_tile('Model discrimination', f'{auc:.3f}', 'ROC-AUC · same model 3 ways'),
        stat_tile(
            'Service cost',
            f'{selected["cost"]:,.0f}',
            f'{reduction:+.1f}% reduction vs no contact',
            accent=True
        ),
    ]) + '</div>'


def model_agreement_summary():
    sk = probabilities['Scikit-learn']
    ma = probabilities['Manual PyTorch']
    nn = probabilities['PyTorch nn.Module']
    return (
        '<div class="insight"><strong>Agreement is the result.</strong> '
        'All three methods fit the <em>same</em> regularized logistic regression. '
        f'Maximum difference from scikit-learn: <strong>{np.max(np.abs(sk-ma)):.8f}</strong> '
        '(manual) and '
        f'<strong>{np.max(np.abs(sk-nn)):.8f}</strong> (nn.Module). '
        f'Manual vs nn.Module: <strong>{np.max(np.abs(ma-nn)):.8f}</strong>. '
        'Small optimizer differences need not change thresholded decisions.</div>'
    )


def policy_replay_table():
    """
    Format the saved historical policy comparison.
    These are September-November validation results.
    """

    data = policy_replay[
        [
            "policy",
            "mean_monthly_recall",
            "mean_contact_rate",
            "max_contact_rate",
            "within_capacity_in_replay",
        ]
    ].copy()

    for name in [
        "mean_monthly_recall",
        "mean_contact_rate",
        "max_contact_rate",
    ]:
        data[name] = (100 * data[name]).round(1)

    return data.rename(columns={
        "policy": "Policy",
        "mean_monthly_recall": "Mean recall %",
        "mean_contact_rate": "Mean contacts %",
        "max_contact_rate": "Worst month contacts %",
        "within_capacity_in_replay": "All months <=20%?",
    })


def policy_test_table():
    """
    Format the frozen December policy results.
    December outcomes did not select the policy.
    """

    data = december_policies[
        [
            "policy",
            "threshold",
            "sessions_contacted",
            "contact_rate",
            "purchasers_contacted",
            "precision",
            "recall",
            "illustrative_cost",
            "capacity_exceeded",
        ]
    ].copy()

    for name in [
        "contact_rate",
        "precision",
        "recall",
    ]:
        data[name] = (100 * data[name]).round(1)

    data["threshold"] = data["threshold"].round(4)

    return data.rename(columns={
        "policy": "Policy",
        "threshold": "Cutoff",
        "sessions_contacted": "Contacts",
        "contact_rate": "Contact %",
        "purchasers_contacted": "Purchasers",
        "precision": "Precision %",
        "recall": "Recall %",
        "illustrative_cost": "Service cost",
        "capacity_exceeded": "Above 20%?",
    })


# 3. Model comparison charts: predicted vs observed (not two overplotted columns).
def calibration_chart(model_name):
    """Bin test probabilities into eight equally sized groups; compare means with observed rates."""
    scores = probabilities[model_name]
    # Rank-based qcut makes group sizes equal even when scores are tied.
    groups = pd.qcut(pd.Series(scores).rank(method='first'), q=8, labels=False)
    data = pd.DataFrame({'pred': scores, 'actual': y_true, 'bin': groups})
    summarized = data.groupby('bin', observed=True).agg(
        predicted=('pred', 'mean'), observed=('actual', 'mean'),
        count=('actual', 'size'), positives=('actual', 'sum'),
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode='lines', name='Perfect calibration',
        line={'color': '#B4C0CD', 'dash': 'dash', 'width': 2}, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=summarized['predicted'], y=summarized['observed'],
        mode='lines+markers', name='December outcome rate',
        marker={'size': 10, 'color': TEAL, 'line': {'color': PAPER, 'width': 2}},
        line={'color': TEAL, 'width': 2.5},
        customdata=summarized[['count', 'positives']].to_numpy(),
        hovertemplate='Mean predicted: %{x:.1%}<br>Observed: %{y:.1%}'
                      '<br>Sessions: %{customdata[0]}<br>Purchases: %{customdata[1]}<extra></extra>',
    ))
    fig.update_layout(
        xaxis_title='Mean predicted purchase probability',
        yaxis_title='Observed purchase rate',
        title={'text': 'Predicted versus actual · 8 equal-sized score groups', 'font': {'size': 15}},
    )
    chart_style(fig, 360)
    # Zoom to the useful range while keeping the identity line honest.
    max_p = max(float(summarized['predicted'].max()), float(summarized['observed'].max()))
    lim = min(1.0, max(0.35, max_p * 1.18))
    fig.update_xaxes(range=[0, lim], tickformat='.0%')
    fig.update_yaxes(range=[0, lim], tickformat='.0%')
    return fig


def probability_distribution(model_name):
    scores = probabilities[model_name]
    fig = go.Figure()
    for outcome, label, color in [(0, 'No purchase', BLUE), (1, 'Purchase', CORAL)]:
        selected = scores[y_true == outcome]
        fig.add_trace(go.Histogram(
            x=selected, name=label, nbinsx=35, histnorm='probability density',
            opacity=0.55, marker_color=color,
            hovertemplate='Score bin: %{x:.2f}<br>Within-class density: %{y:.2f}<extra></extra>'
        ))
    fig.add_vline(x=default_threshold, line={'color': TEAL, 'width': 2, 'dash': 'dot'},
                  annotation_text=f'Policy {default_threshold:.2f}',
                  annotation_position='top right')
    fig.update_layout(
        barmode='overlay',
        title={'text': 'Prediction scores by actual outcome', 'font': {'size': 15}},
        xaxis_title='Predicted purchase probability',
        yaxis_title='Within-class probability density',
    )
    chart_style(fig, 360)
    fig.update_xaxes(range=[0, 1], tickformat='.0%')
    return fig


def probability_differences():
    """Magnify tiny numerical differences that an y=x agreement plot hides."""
    base = probabilities['Scikit-learn']
    fig = go.Figure()
    for name, color, symbol in [('Manual PyTorch', TEAL, 'circle'),
                                ('PyTorch nn.Module', CORAL, 'x')]:
        delta = (probabilities[name] - base) * 1e5
        fig.add_trace(go.Scattergl(
            x=base, y=delta, mode='markers', name=name,
            marker={'size': 5, 'color': color, 'symbol': symbol},
            opacity=0.45,
            hovertemplate='Scikit-learn score: %{x:.4f}<br>Difference (×100,000): %{y:.3f}<extra></extra>'
        ))
    fig.add_hline(y=0, line={'color': '#A5B4C5', 'dash': 'dash'})
    fig.update_layout(
        title={'text': 'Numerical prediction differences · magnified ×100,000', 'font': {'size': 15}},
        xaxis_title='Scikit-learn predicted probability',
        yaxis_title='(PyTorch − scikit-learn) × 100,000',
    )
    chart_style(fig, 295)
    return fig


def model_charts(model_name):
    return calibration_chart(model_name), probability_distribution(model_name)

# 4. Exploratory data summaries. These charts describe held-out December only.
exploration = test_rows.copy()
exploration['TotalPages'] = (exploration['Administrative'] + exploration['Informational']
                             + exploration['ProductRelated'])
duration_sum = (exploration['Administrative_Duration'] + exploration['Informational_Duration']
                + exploration['ProductRelated_Duration'])
exploration['ProductPageShare'] = np.divide(
    exploration['ProductRelated'], exploration['TotalPages'],
    out=np.zeros(len(exploration)), where=exploration['TotalPages'] != 0)
exploration['AvgDurationPerPage'] = np.divide(
    duration_sum, exploration['TotalPages'], out=np.zeros(len(exploration)),
    where=exploration['TotalPages'] != 0)
exploration['Outcome'] = np.where(y_true == 1, 'Purchase', 'No purchase')


DERIVED = [
    "TotalPages",
    "ProductPageShare",
    "AvgDurationPerPage",
]

# Retrospective descriptive variables.
# These are NOT available to the entry-time model.
POST_SESSION_FEATURES = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
]

DESCRIPTIVE_ONLY = set(
    POST_SESSION_FEATURES + DERIVED
)

# Numerical variables supported by the explorer.
NUMERIC = (
    metadata["numeric_features"]
    + POST_SESSION_FEATURES
    + DERIVED
)

# The explorer can show more than the actual model inputs.
FEATURE_OPTIONS = list(dict.fromkeys(
    FEATURE_NAMES
    + ["Month"]
    + POST_SESSION_FEATURES
    + DERIVED
))

FEATURE_OPTIONS = [
    f for f in FEATURE_OPTIONS
    if f in exploration.columns
]


def feature_context(feature):
    """
    Explain how this feature relates to prediction time.
    """

    if feature in DESCRIPTIVE_ONLY:
        return (
            "Retrospective completed-session description only. "
            "Not available at session entry and never "
            "used for prediction."
        )

    if feature == "Month":
        return (
            "Month is known at entry; the model represents "
            "it using engineered MonthSin and MonthCos features."
        )

    return (
        "Entry-time predictor used by the saved "
        "logistic-regression models."
    )



def target_composition():
    positive = int(y_true.sum())
    negative = len(y_true) - positive
    fig = go.Figure(go.Pie(
        labels=['No purchase', 'Purchase'], values=[negative, positive], hole=0.72,
        sort=False, marker={'colors': [BLUE, CORAL]},
        textinfo='none', hovertemplate='%{label}: %{value:,} sessions (%{percent})<extra></extra>',
    ))
    fig.update_layout(
        template='plotly_white', height=255, paper_bgcolor=PAPER,
        margin={'t': 10, 'b': 5, 'l': 5, 'r': 5},
        showlegend=True,
        legend={'orientation': 'h', 'x': 0.1, 'y': -0.03},
        annotations=[{'text': f'<b>{100 * positive/len(y_true):.1f}%</b><br><span style="font-size:12px">purchase rate</span>',
                      'showarrow': False, 'x': 0.5, 'y': 0.5, 'font': {'size': 23, 'color': INK}}],
    )
    return fig


def numeric_feature(feature):
    return feature in NUMERIC and exploration[feature].nunique(dropna=True) > 10


def feature_charts(feature, view):
    """Return distribution, conditional purchase-rate chart, and transparent coverage note."""
    if feature not in FEATURE_OPTIONS:
        raise ValueError('Unknown feature')
    if numeric_feature(feature):
        return numerical_feature_charts(feature, view)
    return categorical_feature_charts(feature)


def numerical_feature_charts(feature, view):
    values = pd.to_numeric(exploration[feature], errors='coerce')
    finite = values.notna() & np.isfinite(values)
    low = float(values[finite].min())
    high_full = float(values[finite].max())
    if view == 'Typical range (≤ 99th percentile)':
        high = float(np.quantile(values[finite], .99))
        high = max(high, low)
    else:
        high = high_full
    shown = finite & (values <= high)
    n_excluded = int(finite.sum() - shown.sum())

    fig = go.Figure()
    for outcome, label, color in [(0, 'No purchase', BLUE), (1, 'Purchase', CORAL)]:
        samples = values[shown & (y_true == outcome)]
        fig.add_trace(go.Histogram(
            x=samples, name=label, nbinsx=32, histnorm='probability',
            opacity=0.56, marker_color=color,
            hovertemplate='Value: %{x:.2f}<br>Share within shown class: %{y:.1%}<extra></extra>',
        ))
    fig.update_layout(
        barmode='overlay', title={'text': f'{feature} · distribution by actual outcome', 'font': {'size': 15}},
        xaxis_title=feature, yaxis_title='Share within shown class',
    )
    chart_style(fig, 345)
    fig.update_yaxes(tickformat='.0%')
    if high > low:
        fig.update_xaxes(range=[low, high])
        if feature in {"BounceRates", "ExitRates", "ProductPageShare"}:
            fig.update_xaxes(tickformat=".0%")


        # Separate zero values when they represent a large share of sessions.
        valid = values[finite]
        outcomes = pd.Series(y_true, index=exploration.index)

        zero_share = (valid == 0).mean()
        separate_zero = zero_share >= 0.10 and (valid > 0).any()

        candidates = valid[valid > 0] if separate_zero else valid

        percent_features = {'BounceRates', 'ExitRates', 'ProductPageShare'}
        is_integer = np.allclose(
            valid, np.round(valid), rtol=0, atol=1e-10
        )

        def nice_number(x):
            if feature in percent_features:
                return f'{100*x:.2f}'.rstrip('0').rstrip('.') + '%'

            if is_integer:
                return f'{x:,.0f}'

            if abs(x) >= 1000:
                text = f'{x:,.0f}'
            elif abs(x) >= 10:
                text = f'{x:,.1f}'
            elif abs(x) >= 1:
                text = f'{x:,.2f}'
            elif abs(x) >= .01:
                text = f'{x:.3f}'
            else:
                text = f'{x:.5f}'

            return text.rstrip('0').rstrip('.') if '.' in text else text

        # Keep the zero group separate, if appropriate.
        groups = []

        if separate_zero:
            groups.append(('0 (none)', valid[valid == 0].index))

        # Create up to five quantile-based groups from the remaining values.
        boundaries = np.unique(
            np.quantile(candidates, np.linspace(0, 1, 6))
        )

        if len(boundaries) < 2:
            candidate_groups = [candidates]
        else:
            codes = pd.cut(
                candidates,
                bins=boundaries,
                include_lowest=True,
                labels=False
            )

            candidate_groups = [
                candidates[codes == i]
                for i in range(len(boundaries) - 1)
            ]

        # Display actual observed min/max, not artificial bin edges.
        for group in candidate_groups:
            if group.empty:
                continue

            low_value = float(group.min())
            high_value = float(group.max())

            if np.isclose(low_value, high_value, atol=1e-12):
                label = nice_number(low_value)
            else:
                label = (
                    f'{nice_number(low_value)}–{nice_number(high_value)}'
                )

            groups.append((label, group.index))

        labels = [label for label, indices in groups]

        counts = np.asarray([
            len(indices) for _, indices in groups
        ])

        rates = np.asarray([
            outcomes.loc[indices].mean()
            for _, indices in groups
        ])

        rate_fig = rate_plot(
            labels,
            rates,
            counts,
            f'{feature} · observed purchase rate by value range',
            height=345
        )

        # Show how many sessions contribute to each bar.
        rate_fig.update_traces(
            text=[f'n={count:,}' for count in counts],
            textposition='outside',
            textfont={'size': 10, 'color': MUTED},
            cliponaxis=False
        )

        rate_fig.update_layout(
            margin={'l': 55, 'r': 20, 't': 48, 'b': 90}
        )

        rate_fig.update_xaxes(
            tickangle=-25,
            title_text='Observed values in each group'
        )

        note = (
            f'The left plot shows {shown.sum():,} of '
            f'{len(valid):,} valid sessions. '
            f'{n_excluded:,} high-end values are hidden from that plot. '
            if n_excluded else
            f'The left plot shows all {len(valid):,} valid sessions. '
        )

        note += (
            'The right plot uses all valid sessions. '
            'Ranges are the actual minimum and maximum values observed '
            'within each approximately equal-frequency group; '
            'ties are kept together. Hover for each group’s sample size. '
        )

        if separate_zero:
            note += (
                f'Zero is shown separately because it accounts for '
                f'{zero_share:.1%} of sessions. '
            )

        if feature in DERIVED:
            note += feature_context(feature)

        return fig, rate_fig, note



def rate_plot(labels, rates, counts, title, height=345):
    fig = go.Figure(go.Bar(
        x=labels, y=100 * np.asarray(rates, dtype=float),
        marker={'color': TEAL, 'line': {'width': 0}},
        customdata=np.asarray(counts),
        hovertemplate='Group: %{x}<br>Purchase rate: %{y:.1f}%<br>Sessions: %{customdata}<extra></extra>'
    ))
    fig.add_hline(y=100 * y_true.mean(), line={'color': '#9EACBD', 'dash': 'dash'},
                  annotation_text=f'December overall: {100*y_true.mean():.1f}%',
                  annotation_position='top left')
    fig.update_layout(title={'text': title, 'font': {'size': 15}},
                      xaxis_title='', yaxis_title='Purchases / sessions (%)')
    chart_style(fig, height)
    maximum = float(np.max(rates)) * 100 if len(rates) else 15
    fig.update_yaxes(range=[0, min(100, max(27, maximum*1.18))], ticksuffix='%')
    fig.update_xaxes(tickangle=-30)
    return fig


def grouped_rate_chart(values, feature):
    data = pd.DataFrame({'category': values, 'purchase': y_true})
    results = data.groupby('category', observed=True)['purchase'].agg(['mean', 'count'])
    order = results['count'].sort_values(ascending=False).index[:10]
    results = results.reindex(order)
    return rate_plot(list(results.index), results['mean'], results['count'],
                     f'{feature} · observed purchase rate by group')


def categorical_feature_charts(feature):
    data = pd.DataFrame({'category': exploration[feature].astype(str), 'purchase': y_true})
    top = data['category'].value_counts().head(10).index.to_list()
    data['display'] = data['category'].where(data['category'].isin(top), 'Other categories')
    order = data['display'].value_counts().index.tolist()
    counts = pd.crosstab(data['display'], data['purchase']).reindex(index=order, columns=[0,1], fill_value=0)
    fig = go.Figure()
    for code, label, color in [(0, 'No purchase', BLUE), (1, 'Purchase', CORAL)]:
        fig.add_trace(go.Bar(
            x=counts.index, y=counts[code], name=label, marker_color=color,
            hovertemplate='%{x}<br>Sessions: %{y:,}<extra></extra>'
        ))
    fig.update_layout(barmode='stack', title={'text': f'{feature} · session counts by category', 'font': {'size': 15}},
                      xaxis_title='', yaxis_title='Sessions')
    chart_style(fig, 345)
    fig.update_xaxes(tickangle=-30)
    grouped = data.groupby('display', observed=True)['purchase'].agg(['mean','count']).reindex(order)
    rate_fig = rate_plot(order, grouped['mean'].to_numpy(), grouped['count'].to_numpy(),
                         f'{feature} · observed purchase rate', height=345)
    note = ('Groups ordered by December sample size; at most 10 categories are shown separately. '
            'Remaining categories are grouped as “Other categories”. '
            'Small categories have less reliable observed rates.')
    if feature in DERIVED:
        note += " " + feature_context(feature)
    return fig, rate_fig, note


# 5. Decision simulation: only thresholding already saved predictions.

def decision_summary(model_name, threshold):

    m = classification_at(model_name, threshold)

    capacity_status = (
        f"Exceeds illustrative {100*assistance_capacity:.0f}% capacity"
        if m["over_capacity"]
        else f"Within illustrative {100*assistance_capacity:.0f}% capacity"
    )

    cards = (
        '<div class="metric-grid decision-grid">'
        + "".join([
            stat_tile(
                "Service cost",
                f'{m["cost"]:,.0f}',
                "Hypothetical service units",
                accent=True,
            ),

            stat_tile(
                "Visitors prioritized",
                f'{m["flagged_pct"]:.1f}%',
                f'{m["flagged"]:,} of {m["n"]:,} · {capacity_status}',
            ),

            stat_tile(
                "Purchaser recall",
                f'{m["recall"]:.1f}%',
                f'{m["tp"]:,} of {int(y_true.sum()):,} eventual purchasers',
            ),

            stat_tile(
                "Precision among contacts",
                f'{m["precision"]:.1f}%',
                f'{m["tp"]:,} purchasers among {m["flagged"]:,} contacts',
            ),
        ])
        + "</div>"
    )

    footnote = (
        f'<div class="small-note">'
        f'Assistance cost: <b>{tp_cost:g}</b> per visitor contacted '
        f'(whether they buy or not); '
        f'assumed unassisted-purchaser service penalty: '
        f'<b>{fn_cost:g}</b>. '
        f'Total = contacts × {tp_cost:g} + '
        f'unassisted purchasers × {fn_cost:g}. '
        f'The frozen selection is <b>{escape(selected_policy)}</b> '
        f'at <b>{default_threshold:.4f}</b>. '
        f'Slider changes are December sensitivity analysis, '
        f'not policy selection. '
        f'These costs are not measured lost revenue.'
        f'</div>'
    )

    return cards + footnote




def confusion_chart(model_name, threshold):
    m = classification_at(model_name, threshold)

    fig = go.Figure()

    squares = [
        (
            0, 1, m["tn"],
            "#E9F0F7",
            "True negative",
            "No contact; no purchase",
        ),
        (
            1, 1, m["fp"],
            "#FDE7E2",
            "False positive",
            "Contact; no purchase",
        ),
        (
            0, 0, m["fn"],
            "#FFF1D9",
            "False negative",
            "No contact; purchased",
        ),
        (
            1, 0, m["tp"],
            "#E1F2ED",
            "True positive",
            "Contact; purchased",
        ),
    ]

    for x, y, count, color, title, note in squares:

        fig.add_shape(
            type='rect',
            x0=x + 0.025,
            x1=x + 0.975,
            y0=y + 0.025,
            y1=y + 0.975,
            line={'color': 'white', 'width': 2},
            fillcolor=color,
            layer='below'
        )

        # Position each text element separately.
        # This prevents titles, counts, and notes from overlapping.
        for y_offset, label, size, weight in [
            (0.77, title, 12, False),
            (0.48, f'{count:,}', 30, True),
            (0.22, note, 11, False),
        ]:

            fig.add_annotation(
                x=x + 0.5,
                y=y + y_offset,
                text=f'<b>{label}</b>' if weight else label,
                showarrow=False,
                xanchor='center',
                yanchor='middle',
                font={
                    'family': 'Arial',
                    'size': size,
                    'color': INK if weight else MUTED
                }
            )

    fig.update_layout(
        title={
            'text': 'December confusion matrix',
            'font': {'size': 15}
        },

        xaxis={
            'range': [0, 2],
            'tickmode': 'array',
            'tickvals': [0.5, 1.5],
            'ticktext': ['No purchase', 'Purchase'],
            'title': 'Predicted',
            'showgrid': False,
            'zeroline': False,
            'fixedrange': True
        },

        yaxis={
            'range': [0, 2],
            'tickmode': 'array',
            'tickvals': [0.5, 1.5],
            'ticktext': ['Purchase', 'No purchase'],
            'title': 'Actual',
            'showgrid': False,
            'zeroline': False,
            'fixedrange': True
        },

        height=370,

        margin={
            'l': 94,
            'r': 14,
            't': 44,
            'b': 65
        },

        paper_bgcolor=PAPER,
        plot_bgcolor=PAPER,

        font={
            'family': 'Arial, sans-serif',
            'size': 12,
            'color': INK
        }
    )

    return fig



def cost_curve(model_name, threshold):
    scores = probabilities[model_name]
    x_values = np.linspace(0, 1, 201)
    costs = []
    for t in x_values:

        predicted = scores >= t

        true_positive = int(
            np.sum(predicted & (y_true == 1))
        )

        false_positive = int(
            np.sum(predicted & (y_true == 0))
        )

        false_negative = int(
            np.sum(~predicted & (y_true == 1))
        )

        costs.append(
            service_cost(
                true_positive,
                false_positive,
                false_negative,
            )
        )
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_values, y=costs, name='December illustrative service cost',
        mode='lines', line={'color': TEAL, 'width': 3},
        hovertemplate='Threshold: %{x:.3f}<br>Cost: %{y:,.0f}<extra></extra>',
    ))
    baseline = fn_cost * int(np.sum(y_true))
    fig.add_hline(y=baseline, line={'color': '#AAB5C4', 'dash': 'dash'},
                  annotation_text=f'Always negative: {baseline:,.0f}',
                  annotation_position='top right')
    fig.add_vline(x=default_threshold, line={'color': '#9AADC0', 'dash': 'dot'})
    selected_cost = classification_at(model_name, threshold)['cost']
    fig.add_trace(go.Scatter(
        x=[threshold], y=[selected_cost], mode='markers', name='Current slider',
        marker={'size': 12, 'color': CORAL, 'line': {'color': PAPER, 'width': 2}},
        hovertemplate='Current threshold: %{x:.2f}<br>Cost: %{y:,.0f}<extra></extra>',
    ))
    fig.update_layout(
        title={'text': 'Illustrative service cost across thresholds', 'font': {'size': 15}},
        xaxis_title='Probability threshold', yaxis_title='Hypothetical cost units',
        showlegend=False,
    )
    chart_style(fig, 355)
    fig.update_xaxes(range=[0,1])
    return fig


def decision_charts(model_name, threshold):
    return decision_summary(model_name, threshold), confusion_chart(model_name, threshold), cost_curve(model_name, threshold)

# 6. UI only: Gradio callbacks call the chart functions above, never any fit routine.
CUSTOM_CSS = """
.gradio-container { max-width: 1380px !important; margin: auto !important;
  background: #F6F8FB !important; color: #14263D !important; }
#hero { padding: 25px 30px 19px; background: linear-gradient(110deg,#17304B,#235A60);
  border-radius: 18px; color: white !important; margin-bottom: 13px; }
#hero h1 { color: #FFFFFF !important; font-size: clamp(27px,3vw,39px) !important;
  letter-spacing: -0.035em; margin: 3px 0 8px !important; }
#hero p { color: #D7E5EC !important; max-width: 850px; font-size: 14px; line-height: 1.6; }
#hero .eyebrow { color: #B7F3E6 !important; font-weight: 700; letter-spacing: .13em;
  font-size: 11px; text-transform: uppercase; }
.metric-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px;
 margin: 7px 0 16px; }
.metric-card { background:#FFFFFF; border:1px solid #E4EAF0; border-radius:14px;
  padding: 15px 18px; min-height:111px; box-shadow: 0 2px 9px rgba(20,38,61,.035); }
.metric-label { color:#687B8D; font-size:12px; font-weight:650; }
.metric-number { font-size:29px; line-height:1.3; font-weight:780; letter-spacing:-.045em;
  margin: 5px 0 3px; font-variant-numeric:tabular-nums; }
.metric-note { color:#64778B; font-size:11px; line-height:1.4; }
.section-lede { color:#64778B !important; font-size:13px !important; }
.insight { border-left:3px solid #167C78; background:#E9F5F3; border-radius:8px;
 padding:13px 16px; color:#244D4C; font-size:13px; line-height:1.7; }
.small-note { background:#FFFFFF; border:1px solid #E5EAF1; border-radius:10px;
  padding:12px 15px; color:#5D7184; font-size:12px; line-height:1.7; }
.gradio-container .tab-nav { border-bottom:1px solid #DEE7EE !important; }
.gradio-container .tab-nav button {font-size:13px !important; font-weight:650 !important;}
.gradio-container .tabitem { padding-top:16px !important; }
.gradio-container .plot-container { border-radius:12px !important; }
#dashboard-footer { color:#8191A0; font-size:11px; text-align:center; padding:15px; }
@media (max-width: 900px) { .metric-grid { grid-template-columns:repeat(2,minmax(0,1fr)); } }
@media (max-width: 540px) { .metric-card {padding:12px;} #hero{padding:18px;}
 .metric-number{font-size:24px;} }
"""

k = int(
    np.floor(assistance_capacity * len(y_true))
)
top_indices = np.argsort(-probabilities['Scikit-learn'], kind='stable')[:k]
top_positives = int(y_true[top_indices].sum())

# Gradio 6 moved styling arguments from Blocks() into launch().
APP_THEME = gr.themes.Soft(primary_hue='teal', neutral_hue='slate')
GRADIO_V6 = int(gr.__version__.split('.')[0]) >= 6
BLOCKS_KW = {'title': 'Session intelligence · Assignment 1'}
if not GRADIO_V6:
    BLOCKS_KW.update(theme=APP_THEME, css=CUSTOM_CSS)

with gr.Blocks(**BLOCKS_KW) as demo:
    gr.HTML('''<div id="hero">
    <div class="eyebrow">Assignment 01 / Model lab</div>
    <h1>Shopping session intelligence</h1>
    <p>
        One logistic regression, implemented three ways.
        Explore held-out December predictions, inspect descriptive
        browsing behavior, and evaluate a hypothetical entry-time
        customer-assistance policy.
        Models are loaded from disk; this dashboard never retrains them.
    </p>
    </div>''')
    gr.HTML(header_stats())

    with gr.Tabs():
        with gr.Tab('01   Model comparison'):
            gr.Markdown('### Model performance')
            gr.Markdown(
                f'<span class="section-lede">All figures use the held-out December sessions. '
                f'Threshold-based columns use the historically chosen **{default_threshold:.2f}**; '
                'ROC-AUC and average precision use raw scores.</span>'
            )
            gr.Dataframe(value=summary_table(), interactive=False, show_label=False,
                         row_count=5, column_widths=['26%', '11%', '13%', '11%', '11%', '11%', '11%'])
            gr.HTML(model_agreement_summary())
            with gr.Row():
                model_select = gr.Dropdown(choices=MODEL_NAMES, value=default_model,
                                           label='Inspect saved model', scale=1)
                gr.Markdown('**Why this view?** A binary target has only two actual values. '
                            'Grouping predicted scores into bins makes predicted-vs-actual '
                            'behavior easier to read than thousands of overlapping points.')
            with gr.Row(equal_height=True):
                calibration = gr.Plot(value=calibration_chart(default_model), label='Predicted vs actual')
                distributions = gr.Plot(value=probability_distribution(default_model), label='Scores by actual outcome')
            model_select.change(model_charts, inputs=model_select,
                                outputs=[calibration, distributions])
            with gr.Accordion('How closely do the three implementations agree?', open=False):
                gr.Markdown('The next chart **magnifies** the extremely small difference '
                            'between PyTorch and scikit-learn by **100,000×**. '
                            'The two PyTorch traces coincide because the saved predictions are identical.')
                gr.Plot(value=probability_differences(), show_label=False)
            gr.HTML(
                f'<div class="insight"><strong>Capacity diagnostic (not model tuning):</strong> '
                f'The top 20% of December scores contain <strong>{top_positives}</strong> '
                f'of <strong>{int(y_true.sum())}</strong> purchase sessions '
                f'({100*top_positives/y_true.sum():.1f}% capture). '
                'This is a <em>retrospective</em> ranking diagnostic, '
                'not a deployable live policy or measured incremental revenue.</div>'
            )

        with gr.Tab('02   Data explorer'):
            gr.Markdown('### What does December look like?')
            gr.Markdown('This tab uses **December only**, not the full original dataset. '
                        'The target is imbalanced; raw counts and purchase rates communicate '
                        'different things. All charts describe observed associations, not causality.')
            with gr.Row(equal_height=True):
                with gr.Column(scale=1, min_width=300):
                    gr.Markdown('**Purchase outcomes**')
                    gr.Plot(value=target_composition(), show_label=False)
                    gr.Markdown(f'**{len(y_true)-int(y_true.sum()):,}** no-purchase sessions · '
                                f'**{int(y_true.sum()):,}** purchases')
                with gr.Column(scale=3, min_width=350):
                    gr.Markdown('**Inspect a feature**')
                    feature_select = gr.Dropdown(
                        choices=FEATURE_OPTIONS,
                        value="VisitorType",
                        label="Entry-time or retrospective feature",
                    )
                    view_select = gr.Radio(
                        choices=['Typical range (≤ 99th percentile)', 'Full range'],
                        value='Typical range (≤ 99th percentile)',
                        label='Numerical feature display', interactive=True)
                    gr.Markdown('For numerical features, the left chart compares the '
                                '**within-class distributions**, and the right chart shows '
                                '**observed purchase rates across value ranges**. '
                                'For categorical features, they show counts and purchase rates.')
            initial_feature = feature_charts(
                "VisitorType",
                "Typical range (≤ 99th percentile)",
            )
            with gr.Row(equal_height=True):
                feature_left = gr.Plot(value=initial_feature[0], show_label=False)
                feature_right = gr.Plot(value=initial_feature[1], show_label=False)
            feature_note = gr.Markdown(initial_feature[2])
            for control in [feature_select, view_select]:
                control.change(feature_charts, inputs=[feature_select, view_select],
                               outputs=[feature_left, feature_right, feature_note])
            gr.HTML(
                '<div class="insight">'
                '<strong>Prediction-time integrity:</strong> '
                'The model uses calendar, visitor, traffic and technical '
                'information available at session entry, plus engineered '
                'calendar and visitor interactions. '
                '<code>ProductRelated</code>, durations and other '
                'full-session variables are available here '
                '<em>only for retrospective exploration</em>; '
                'they do not enter any saved model.'
                '</div>'
            )

        with gr.Tab('03   Decision simulator'):
            gr.Markdown('### What changes when the threshold moves?')
            gr.Markdown(
                'Explore the **tradeoff between contacting visitors '
                'and not prioritizing eventual purchasers**. '
                'The slider is a December test-set *sensitivity visualization*, '
                '**not** a method for selecting a new threshold.'
            )
            with gr.Row():
                simulator_model = gr.Dropdown(choices=MODEL_NAMES, value=default_model,
                                               label='Saved model', scale=1)
                simulator_threshold = gr.Slider(minimum=0, maximum=1, step=.01,
                                                 value=default_threshold,
                                                 label='Purchase-probability threshold', scale=2)
            gr.HTML(
                '<div class="small-note">'
                '<b>Interpretation:</b> lowering the threshold generally '
                'prioritizes more visitors, increasing staffing demand. '
                f'The frozen historical policy is <b>{escape(selected_policy)}</b> '
                f'at <b>{default_threshold:.4f}</b>. '
                f'The {100*assistance_capacity:.0f}% contact budget is '
                'illustrative, not a guaranteed staffing limit.'
                '</div>'
            )
            dashboard_kpis = gr.HTML(value=decision_summary(default_model, default_threshold))
            with gr.Row(equal_height=True):
                cm_plot = gr.Plot(value=confusion_chart(default_model, default_threshold),
                                  show_label=False)
                cost_plot = gr.Plot(value=cost_curve(default_model, default_threshold),
                                    show_label=False)
            for control in [simulator_model, simulator_threshold]:
                control.change(decision_charts,
                               inputs=[simulator_model, simulator_threshold],
                               outputs=[dashboard_kpis, cm_plot, cost_plot])
            gr.Markdown(
                '**Important:** The plotted December cost curve is descriptive. '
                'Our policy was selected using earlier months, before evaluating '
                'December. Contacting any visitor incurs illustrative staff cost; '
                'not prioritizing an eventual purchaser carries an additional '
                'hypothetical service penalty. Neither that penalty nor the '
                'effect of assistance on sales was measured by this dataset.'
            )

            
            with gr.Accordion(
                "Policy audit · historical replay and December comparison",
                open=False,
            ):

                gr.Markdown(
                    f"**Frozen decision: {selected_policy} "
                    f"(threshold {default_threshold:.4f}).** "
                    "The policy was selected before December outcomes "
                    "were evaluated. The historical replay tested "
                    "cutoffs fitted using previous months."
                )

                gr.Markdown(
                    "#### September–November historical replay"
                )

                gr.Dataframe(
                    value=policy_replay_table(),
                    interactive=False,
                    show_label=False,
                )

                gr.Markdown(
                    "#### December holdout: previously frozen policies"
                )

                gr.Dataframe(
                    value=policy_test_table(),
                    interactive=False,
                    show_label=False,
                )

                if all_replay_policies_exceeded:
                    gr.Markdown(
                        "**Why Policy A was retained:** "
                        "All three candidates exceeded the hypothetical "
                        "20% contact limit in at least one historical "
                        "replay month. Therefore, the previously defined "
                        "fallback selected conservative Policy A. "
                        "This does not guarantee future capacity compliance."
                    )

                gr.Markdown(
                    "**Interpretation:** Policies B and C use a "
                    "historically calibrated capacity cutoff; C also "
                    "applies an illustrative cost-based floor. "
                    "The comparison is an offline policy analysis, "
                    "not a simulation of actual staffing queues. "
                    "December outcomes were not used to choose "
                    "the final policy."
                )


    gr.HTML(
        '<div id="dashboard-footer">'
        'December holdout · No retraining at app startup · '
        'Session-entry propensity prediction · Assignment 1'
        '</div>'
    )

if __name__ == '__main__':
    launch_options = {'inbrowser': False, 'show_error': True}
    if GRADIO_V6:
        launch_options.update(theme=APP_THEME, css=CUSTOM_CSS)
    demo.launch(**launch_options)
