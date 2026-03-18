# Bar chart of per-cancer Pearson correlation

import pandas as pd
import plotly.graph_objects as go

df = pd.read_csv("./results/global_baseline_results.csv")
df = df.sort_values("mean_pearson_r", ascending=True)

fig = go.Figure()
fig.add_trace(go.Bar(
    x=df["mean_pearson_r"],
    y=df["cancer_type"],
    error_x=dict(type='data', array=df["std"].tolist(), visible=True),
    orientation='h',
    marker_color='steelblue',
    text=[f"{v:.3f}" for v in df["mean_pearson_r"]],
    textposition='outside',
))

fig.update_layout(
    title="Global Baseline: Per-Cancer Gene Expression Prediction (Pearson R)",
    xaxis_title="Mean Pearson Correlation (Top-50 HVGs)",
    yaxis_title="Cancer Type",
    xaxis=dict(range=[0, 0.1]),
    height=400,
    width=750,
    plot_bgcolor="white",
    paper_bgcolor="white",
)
fig.update_xaxes(showgrid=True, gridcolor='lightgrey')

fig.write_image("./results/global_baseline_bar.png", scale=2)