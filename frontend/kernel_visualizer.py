import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import csv
import os

debug = False if os.environ["DASH_DEBUG_MODE"] == "False" else True

color_map_attn = {"iree_attention.csv": "blue", "torch_attention.csv": "orange", "triton_attention.csv": "green"}
color_map_conv = {"iree_conv.csv": "blue", "torch_conv.csv": "orange"}
color_map_gemm = {"iree_gemm.csv": "blue", "iree_gemm_tk.csv": "green", "hipblaslt-gemm.csv": "orange", "rocblas-gemm.csv": "yellow"}
golden_csv_path = os.path.dirname(os.path.abspath(__file__)) + "/golden_csv"

attn_data = []
attn_files = [f"{golden_csv_path}/iree_attention.csv", f"{golden_csv_path}/torch_attention.csv", f"{golden_csv_path}/triton_attention.csv"]
for file in attn_files:
    with open(file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row.update({"csv": file.split("/")[-1]})
            row["tflops"] = float(row["tflops"])
            row["arithmetic_intensity"] = float(row["arithmetic_intensity"])
            attn_data.append(row)

conv_data = []
conv_files = [f"{golden_csv_path}/iree_conv.csv", f"{golden_csv_path}/torch_conv.csv"]
for file in conv_files:
    with open(file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row.update({"csv": file.split("/")[-1]})
            row["tflops"] = float(row["tflops"])
            row["arithmetic_intensity"] = float(row["arithmetic_intensity"])
            conv_data.append(row)

gemm_data = []
gemm_files = [f"{golden_csv_path}/iree_gemm.csv", f"{golden_csv_path}/iree_gemm_tk.csv", f"{golden_csv_path}/hipblaslt-gemm.csv", f"{golden_csv_path}/rocblas-gemm.csv"]
for file in gemm_files:
    with open(file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row.update({"csv": file.split("/")[-1]})
            row["tflops"] = float(row["tflops"])
            row["arithmetic_intensity"] = float(row["arithmetic_intensity"])
            if "fp" in row["dtype"]:
                row["dtype"] = row["dtype"].replace("fp", "f")
            if "A" in row and "B" in row:
                row["tA"] = row.pop("A")
                row["tB"] = row.pop("B")
            gemm_data.append(row)

df_attn = pd.DataFrame(attn_data)
df_conv = pd.DataFrame(conv_data)
df_gemm = pd.DataFrame(gemm_data)

app = dash.Dash(__name__, suppress_callback_exceptions=True)

def create_attention_roofline_page():
    return html.Div([
        html.H1("Attention Performance Visualizer"),
        
        html.Div([
            dcc.Dropdown(
                id='csv-dropdown-attn',
                options=[{'label': 'All CSVs', 'value': 'all'}] + 
                        [{'label': m, 'value': m} for m in df_attn['csv'].unique()],
                value='all',
                multi=True,
                style={'width': '100%'}
            ),
            dcc.Dropdown(
                id='dtype-dropdown-attn',
                options=[{'label': 'All Dtypes', 'value': 'all'}] + 
                        [{'label': d, 'value': d} for d in df_attn['dtype'].unique()],
                value='all',
                multi=True,
                style={'width': '100%'}
            ),
            dcc.Dropdown(
                id='model-dropdown-attn',
                options=[{'label': 'All Models', 'value': 'all'}] + 
                        [{'label': m, 'value': m} for m in df_attn['tag'].unique()],
                value='all',
                multi=True,
                style={'width': '100%'}
            ),
            dcc.Dropdown(
                id='batch-dropdown-attn',
                options=[{'label': 'All Batches', 'value': 'all'}] + 
                        [{'label': m, 'value': m} for m in df_attn['B'].unique()],
                value='all',
                multi=True,
                style={'width': '100%'}
            ),
        ], style={'width': '100%', 'display': 'flex', 'justifyContent': 'space-between'}),
        
        dcc.Tabs([
            dcc.Tab(label='Attention Roofline Plot', children=[
                dcc.Graph(id='roofline-plot-attn')
            ]),
        ])
    ])

def create_convolution_roofline_page():
    return html.Div([
        html.H1("Convolution Performance Visualizer"),
        
        html.Div([
            dcc.Dropdown(
                id='csv-dropdown-conv',
                options=[{'label': 'All CSVs', 'value': 'all'}] + 
                        [{'label': m, 'value': m} for m in df_conv['csv'].unique()],
                value='all',
                multi=True,
                style={'width': '100%'}
            ),
            dcc.Dropdown(
                id='dtype-dropdown-conv',
                options=[{'label': 'All Dtypes', 'value': 'all'}] + 
                        [{'label': d, 'value': d} for d in df_conv['input_dtype'].unique()],
                value='all',
                multi=True,
                style={'width': '100%'}
            ),
            dcc.Dropdown(
                id='model-dropdown-conv',
                options=[{'label': 'All Models', 'value': 'all'}] + 
                        [{'label': m, 'value': m} for m in df_conv['tag'].unique()],
                value='all',
                multi=True,
                style={'width': '100%'}
            ),
            dcc.Dropdown(
                id='batch-dropdown-conv',
                options=[{'label': 'All Batches', 'value': 'all'}] + 
                        [{'label': m, 'value': m} for m in df_conv['B'].unique()],
                value='all',
                multi=True,
                style={'width': '100%'}
            ),
        ], style={'width': '100%', 'display': 'flex', 'justifyContent': 'space-between'}),
        
        dcc.Tabs([
            dcc.Tab(label='Roofline Plot', children=[
                dcc.Graph(id='roofline-plot-conv')
            ]),
        ])
    ])

def create_gemm_roofline_page():
    return html.Div([
        html.H1("GEMM Performance Visualizer"),
        
        html.Div([
            dcc.Dropdown(
                id='csv-dropdown-gemm',
                options=[{'label': 'All CSVs', 'value': 'all'}] + 
                        [{'label': d, 'value': d} for d in df_gemm['csv'].unique()],
                value='all',
                multi=True,
                style={'width': '100%'}
            ),
            dcc.Dropdown(
                id='dtype-dropdown-gemm',
                options=[{'label': 'All Dtypes', 'value': 'all'}] + 
                        [{'label': d, 'value': d} for d in df_gemm['dtype'].unique()],
                value='all',
                multi=True,
                style={'width': '100%'}
            ),
            dcc.Dropdown(
                id='model-dropdown-gemm',
                options=[{'label': 'All Models', 'value': 'all'}] + 
                        [{'label': m, 'value': m} for m in df_gemm['tag'].unique()],
                value='all',
                multi=True,
                style={'width': '100%'}
            ),
            dcc.Dropdown(
                id='tA-dropdown-gemm',
                options=[{'label': 'All Transpose A', 'value': 'all'}] + 
                        [{'label': m, 'value': m} for m in df_gemm['tA'].unique()],
                value='all',
                multi=True,
                style={'width': '100%'}
            ),
            dcc.Dropdown(
                id='tB-dropdown-gemm',
                options=[{'label': 'All Transpose B', 'value': 'all'}] + 
                        [{'label': m, 'value': m} for m in df_gemm['tB'].unique()],
                value='all',
                multi=True,
                style={'width': '100%'}
            ),
        ], style={'width': '100%', 'display': 'flex', 'justifyContent': 'space-between'}),
        
        dcc.Tabs([
            dcc.Tab(label='Roofline Plot', children=[
                dcc.Graph(id='roofline-plot-gemm')
            ]),
        ])
    ])

# Main layout with page routing
app.layout = html.Div([
    dcc.Location(id='url', refresh=False, pathname='/gemm'),
    html.Nav([
        dcc.Link('GEMM Kernel Performance', href='/gemm'),
        html.Span(" | "),
        dcc.Link('Attention Kernel Performance', href='/attention'),
        html.Span(" | "),
        dcc.Link('Convolution Kernel Performance', href='/convolution'),
    ]),
    html.Div(id='page-content')
], style={'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#f8f9fa'})

# Update the page content based on the URL
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/gemm':
        return create_gemm_roofline_page()
    if pathname == '/attention':
        return create_attention_roofline_page()
    if pathname == '/convolution':
        return create_convolution_roofline_page()

@app.callback(
    Output('roofline-plot-attn', 'figure'),
    [Input('csv-dropdown-attn', 'value'),
     Input('dtype-dropdown-attn', 'value'),
     Input('model-dropdown-attn', 'value'),
     Input('batch-dropdown-attn', 'value')]
)
def update_attn_graphs(selected_csv, selected_dtype, selected_model, selected_batch):
    filtered_df = df_attn
    if len(selected_csv) != 0 and 'all' not in selected_csv:
        filtered_df = filtered_df[filtered_df['csv'].isin(selected_csv)]
    if len(selected_dtype) != 0 and 'all' not in selected_dtype:
        filtered_df = filtered_df[filtered_df['dtype'].isin(selected_dtype)]
    if len(selected_model) != 0 and 'all' not in selected_model:
        filtered_df = filtered_df[filtered_df['tag'].isin(selected_model)]
    if len(selected_batch) != 0 and 'all' not in selected_batch:
        filtered_df = filtered_df[filtered_df['B'].isin(selected_batch)]
    
    # Roofline Plot
    peak_memory_bandwidth = 5.3  # TB/s
    if len(selected_dtype) != 1 or 'all' in selected_dtype:
        peak_compute = 1300  # TFLOP/s
    else:
        tflops_map = {
            "f32": 653.7,
            "f16": 1307.4,
            "bf16": 1307.4,
            "f8E4M3FNUZ": 2614.9,
            "i8": 2614.9,
        }
        peak_compute = tflops_map[selected_dtype[0]]
    x_range = np.logspace(0, 4, 100)
    y_memory = peak_memory_bandwidth * x_range
    y_compute = np.full_like(x_range, peak_compute)
    y_roofline = np.minimum(y_memory, y_compute)

    roofline_fig = go.Figure()

    for csv_file, color in color_map_attn.items():
        csv_filtered_df = filtered_df[filtered_df['csv'] == csv_file]
        if not csv_filtered_df.empty:
            roofline_fig.add_trace(go.Scatter(
                x=csv_filtered_df['arithmetic_intensity'],
                y=csv_filtered_df['tflops'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=color,  # Use the color mapped from the 'csv' property
                    symbol='circle',
                ),
                text=[f"Kernel: {name}<br>Performance: {perf:.2f} TFLOP/s<br>Arithmetic Intensity: {intensity:.2f} flops/bytes"
                      for name, perf, intensity in zip(csv_filtered_df['name'], csv_filtered_df['tflops'], csv_filtered_df['arithmetic_intensity'])],
                hoverinfo='text',
                name=csv_file.split("/")[-1]  # Use the CSV file name for the legend
            ))
    roofline_fig.add_trace(go.Scatter(
        x=x_range,
        y=y_roofline,
        mode='lines',
        name='Roofline',
        line=dict(color='red', dash='dash')
    ))
    roofline_fig.update_layout(
        title='Roofline Plot',
        xaxis_title='Arithmetic Intensity (FLOP/byte)',
        yaxis_title='Performance (TFLOP/s)',
        xaxis_type='log',
        yaxis_type='log'
    )
    
    return roofline_fig

@app.callback(
    Output('roofline-plot-conv', 'figure'),
    [Input('csv-dropdown-conv', 'value'),
     Input('dtype-dropdown-conv', 'value'),
     Input('model-dropdown-conv', 'value'),
     Input('batch-dropdown-conv', 'value')]
)
def update_conv_graphs(selected_csv, selected_dtype, selected_model, selected_batch):
    filtered_df = df_conv
    if len(selected_csv) != 0 and 'all' not in selected_csv:
        filtered_df = filtered_df[filtered_df['csv'].isin(selected_csv)]
    if len(selected_dtype) != 0 and 'all' not in selected_dtype:
        filtered_df = filtered_df[filtered_df['input_dtype'].isin(selected_dtype)]
    if len(selected_model) != 0 and 'all' not in selected_model:
        filtered_df = filtered_df[filtered_df['tag'].isin(selected_model)]
    if len(selected_batch) != 0 and 'all' not in selected_batch:
        filtered_df = filtered_df[filtered_df['B'].isin(selected_batch)]
    
    # Roofline Plot
    peak_memory_bandwidth = 5.3  # TB/s
    if len(selected_dtype) != 1 or 'all' in selected_dtype:
        peak_compute = 1300  # TFLOP/s
    else:
        tflops_map = {
            "f32": 653.7,
            "f16": 1307.4,
            "bf16": 1307.4,
            "f8E4M3FNUZ": 2614.9,
            "i8": 2614.9,
        }
        peak_compute = tflops_map[selected_dtype[0]]
    x_range = np.logspace(0, 4, 100)
    y_memory = peak_memory_bandwidth * x_range
    y_compute = np.full_like(x_range, peak_compute)
    y_roofline = np.minimum(y_memory, y_compute)

    roofline_fig = go.Figure()

    for csv_file, color in color_map_conv.items():
        csv_filtered_df = filtered_df[filtered_df['csv'] == csv_file]
        if not csv_filtered_df.empty:
            roofline_fig.add_trace(go.Scatter(
                x=csv_filtered_df['arithmetic_intensity'],
                y=csv_filtered_df['tflops'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=color,  # Use the color mapped from the 'csv' property
                    symbol='circle',
                ),
                text=[f"Kernel: {name}<br>Performance: {perf:.2f} TFLOP/s<br>Arithmetic Intensity: {intensity:.2f} flops/bytes"
                      for name, perf, intensity in zip(csv_filtered_df['name'], csv_filtered_df['tflops'], csv_filtered_df['arithmetic_intensity'])],
                hoverinfo='text',
                name=csv_file.split("/")[-1]  # Use the CSV file name for the legend
            ))
    roofline_fig.add_trace(go.Scatter(
        x=x_range,
        y=y_roofline,
        mode='lines',
        name='Roofline',
        line=dict(color='red', dash='dash')
    ))
    roofline_fig.update_layout(
        title='Roofline Plot',
        xaxis_title='Arithmetic Intensity (FLOP/byte)',
        yaxis_title='Performance (TFLOP/s)',
        xaxis_type='log',
        yaxis_type='log'
    )
    
    return roofline_fig

@app.callback(
    Output('roofline-plot-gemm', 'figure'),
    [Input('csv-dropdown-gemm', 'value'),
     Input('dtype-dropdown-gemm', 'value'),
     Input('model-dropdown-gemm', 'value'),
     Input('tA-dropdown-gemm', 'value'),
     Input('tB-dropdown-gemm', 'value')]
)
def update_gemm_graphs(selected_csv, selected_dtype, selected_model, selected_tA, selected_tB):
    filtered_df = df_gemm
    if len(selected_csv) != 0 and 'all' not in selected_csv:
        filtered_df = filtered_df[filtered_df['csv'].isin(selected_csv)]
    if len(selected_dtype) != 0 and 'all' not in selected_dtype:
        filtered_df = filtered_df[filtered_df['dtype'].isin(selected_dtype)]
    if len(selected_model) != 0 and 'all' not in selected_model:
        filtered_df = filtered_df[filtered_df['tag'].isin(selected_model)]
    if len(selected_tA) != 0 and 'all' not in selected_tA:
        filtered_df = filtered_df[filtered_df['tA'].isin(selected_tA)]
    if len(selected_tB) != 0 and 'all' not in selected_tB:
        filtered_df = filtered_df[filtered_df['tB'].isin(selected_tB)]
    
    # Roofline Plot
    peak_memory_bandwidth = 5.3  # TB/s
    if len(selected_dtype) != 1 or 'all' in selected_dtype:
        peak_compute = 1300  # TFLOP/s
    else:
        tflops_map = {
            "f32": 653.7,
            "f16": 1307.4,
            "bf16": 1307.4,
            "f8E4M3FNUZ": 2614.9,
            "i8": 2614.9,
        }
        peak_compute = tflops_map[selected_dtype[0]]
    x_range = np.logspace(0, 4, 100)
    y_memory = peak_memory_bandwidth * x_range
    y_compute = np.full_like(x_range, peak_compute)
    y_roofline = np.minimum(y_memory, y_compute)

    roofline_fig = go.Figure()

    for csv_file, color in color_map_gemm.items():
        csv_filtered_df = filtered_df[filtered_df['csv'] == csv_file]
        if not csv_filtered_df.empty:
            roofline_fig.add_trace(go.Scatter(
                x=csv_filtered_df['arithmetic_intensity'],
                y=csv_filtered_df['tflops'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=color,  # Use the color mapped from the 'csv' property
                    symbol='circle',
                ),
                text=[f"Kernel: {name}<br>M={m}, N={n}, K={k}<br>Performance: {perf:.2f} TFLOP/s<br>Arithmetic Intensity: {intensity:.2f} flops/bytes"
                      for name, m, n, k, perf, intensity in zip(csv_filtered_df['name'], csv_filtered_df['M'], csv_filtered_df['N'], csv_filtered_df['K'], csv_filtered_df['tflops'], csv_filtered_df['arithmetic_intensity'])],
                hoverinfo='text',
                name=csv_file.split("/")[-1]  # Use the CSV file name for the legend
            ))
    roofline_fig.add_trace(go.Scatter(
        x=x_range,
        y=y_roofline,
        mode='lines',
        name='Roofline',
        line=dict(color='red', dash='dash')
    ))
    roofline_fig.update_layout(
        title='Roofline Plot',
        xaxis_title='Arithmetic Intensity (FLOP/byte)',
        yaxis_title='Performance (TFLOP/s)',
        xaxis_type='log',
        yaxis_type='log'
    )
    
    return roofline_fig

server = app.server

if __name__ == '__main__':
    app.run(debug=debug)
