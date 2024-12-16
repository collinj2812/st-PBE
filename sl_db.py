import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px

# with open('data_dict_final.pkl', 'rb') as f:
#     data = pickle.load(f)


# Modified LazyLoader to handle the new structure
class LazyLoader:
    def __init__(self):
        self.cache = {}

    def get_data(self, case, level2, method, *args):
        cache_key = f"{case}_{level2}_{method}_{'_'.join(str(arg) for arg in args)}"

        if cache_key in self.cache:
            return self.cache[cache_key]

        if method == 'OCFE' and len(args) >= 2:
            filename = f'data/case{case}/{level2}/OCFE/{args[0]}_{args[1]}.pkl'
        elif method == 'DPBE' and len(args) >= 1:
            filename = f'data/case{case}/{level2}/DPBE/{args[0]}.pkl'
        elif method == 'MC' and len(args) >= 1:
            filename = f'data/case{case}/{level2}/MC/{args[0]}.pkl'
        else:
            filename = f'data/case{case}/{level2}/{method}.pkl'

        try:
            with open(filename, 'rb') as f:
                result = pickle.load(f)

                # Navigate through remaining args if any
                if method == 'OCFE' and len(args) > 2:
                    for arg in args[2:]:
                        result = result[arg]
                elif method == 'DPBE' and len(args) > 1:
                    for arg in args[1:]:
                        result = result[arg]
                elif method == 'MC' and len(args) > 1:
                    for arg in args[1:]:
                        result = result[arg]

                self.cache[cache_key] = result
                return result
        except FileNotFoundError:
            print(f"File not found: {filename}")
            return None
# Initialize the loader at the start of your app
loader = LazyLoader()


# Slider for time
time = st.slider(
    'Select time',
    min_value=0, max_value=50, value=10, step=1,
    label_visibility="visible",
    help=None,
    on_change=None,
    args=None,
    kwargs=None,
    disabled=False,
    key=None,
)

# Sidebar for selecting case
case_select = st.sidebar.selectbox(
    'Case',
    options=['1', '2', '3']
)
case = f'case{case_select}'
if case == 'case3':
    case = 'case4'  # case 3 in the paper is case 4 in the dashboard data

# Case-specific parameters in an expander
with st.sidebar.expander("Case Parameters", expanded=True):
    if case == 'case1' or case == 'case3' or case == 'case4':
        level2_select = st.selectbox(
            'Growth rate G',
            options=['0.5', '1.0', '5.0']
        )
        level2 = f'G_{level2_select}'
    elif case == 'case2':
        level2_select = st.selectbox(
            r'Agglomeration rate beta $\beta$',
            options=['0.1', '0.5']
        )
        level2 = f'beta_{level2_select}'

# Initialize dataframe based on case
if case == 'case4':
    # Create dummy dataframe with enough points (100 to match histogram bins)
    x = np.linspace(0, 60, 100)
    df = pd.DataFrame({
        'x': x,
        'y': [None] * 100,  # Make array of same length as x
        'label': ['Domain'] * 100  # Make array of same length as x
    })
else:
    # Load exact solution data
    exact_data = loader.get_data(case, level2, 'exact')
    df = pd.DataFrame({
        'x': exact_data['xx'],
        'y': exact_data[str(time)],
        'label': 'Exact solution'
    })
# Method selection
method = st.multiselect(
    'Select Method',
    options=['OCFE', 'DPBE', 'MOM', 'MC'],
    default=['DPBE']
)

# OCFE parameters in their own expander
if 'OCFE' in method:
    with st.sidebar.expander("OCFE Parameters", expanded=True):
        level3_OCFE_select = st.selectbox(
            'Artificial diffusion $D_a$',
            options=['0.00001', '0.0001', '0.001', '0.01', '0.1', '1.0']
        )
        if level3_OCFE_select == '0.00001':
            level3_OCFE = f'D_a_0.00001'
        elif level3_OCFE_select == '0.0001':
            level3_OCFE = f'D_a_0.00010'
        elif level3_OCFE_select == '0.001':
            level3_OCFE = f'D_a_0.00100'
        elif level3_OCFE_select == '0.01':
            level3_OCFE = f'D_a_0.01000'
        elif level3_OCFE_select == '0.1':
            level3_OCFE = f'D_a_0.10000'
        elif level3_OCFE_select == '1.0':
            level3_OCFE = f'D_a_1.00000'

        level4_OCFE_select = st.slider(
            'Number of finite elements',
            min_value=5, max_value=10, value=6, step=1
        )
        level4_OCFE = f'n_elements_{level4_OCFE_select}'
        level5_OCFE_select = st.slider(
            'Number of collocation points',
            min_value=5, max_value=10, value=5, step=1
        )
        level5_OCFE = f'n_col_{level5_OCFE_select}'

    # Load OCFE data
    ocfe_data = loader.get_data(case, level2, 'OCFE', level3_OCFE, level4_OCFE, level5_OCFE)
    df_OCFE = pd.DataFrame({
        'x': ocfe_data['xx'],
        'y': ocfe_data[str(time)],
        'label': 'OCFE'
    })
    df = pd.concat([df, df_OCFE])

# DPBE parameters in their own expander
if 'DPBE' in method:
    with st.sidebar.expander("DPBE Parameters", expanded=True):
        level3_DPBE_select = st.selectbox(
            'Scheme',
            options=['First order', 'Flux limited upwind', 'WENO5'],
            index=2
        )
        if level3_DPBE_select == 'First order':
            level3_DPBE = 'scheme_first'
        elif level3_DPBE_select == 'Flux limited upwind':
            level3_DPBE = 'scheme_limited'
        elif level3_DPBE_select == 'WENO5':
            level3_DPBE = 'scheme_WENO5'

        selected_number_of_classes = st.slider(
            'Number of classes',
            min_value=10, max_value=200, value=50, step=1
        )
        level4_DPBE = f'no_class_{selected_number_of_classes}'
    # Load DPBE data
    dpbe_data = loader.get_data(case, level2, 'DPBE', level3_DPBE, level4_DPBE)
    df_DPBE = pd.DataFrame({
        'x': dpbe_data['L_i'],
        'y': dpbe_data[str(time)],
        'label': 'DPBE'
    })
    df = pd.concat([df, df_DPBE])

# MC parameters in their own expander
if 'MC' in method:
    with st.sidebar.expander("MC Parameters", expanded=True):
        level3_MC_select = st.selectbox(
            'Number of samples',
            options=['1000', '5000', '10000', '50000'],
            index=2
        )
    level3_MC = f'no_particles_{level3_MC_select}'
    # Load MC data
    mc_data = loader.get_data(case, level2, 'MC', level3_MC)
    data_MC = mc_data[str(time)]
    no_begin = mc_data['0'].shape[1]
    no_now = data_MC.shape[1]

# add dummy data for case 4 if only MC is selected
if case == 'case4' and len(method) == 1 and 'MC' in method or len(method) == 0 or case == 'case4' and len(method) == 1 and 'MOM' in method:
    df = pd.DataFrame({'x': [0, 60], 'y': [None, None], 'label': ''})

# Create the line chart using Plotly Express
fig = px.line(df, x='x', y='y', color='label')

# add a title
fig.update_layout(
    xaxis_title='V',
    yaxis_title='n(V)'
)

if 'MC' in method:
    # Calculate histogram data
    hist_values, bin_edges = np.histogram(data_MC, bins=100, density=True)

    # Add the histogram trace to the figure
    fig.add_trace(
        go.Line(
            x=bin_edges[:-1],  # Use left edges of bins as x values
            y=hist_values*no_now/no_begin,  # Normalize to the number of samples at t=0
            name='Monte Carlo',
            opacity=0.5,
        )
    )

if 'MOM' in method:
    # Calculate the maximum y value across all traces, excluding None values
    max_y = 0  # Default value
    for trace in fig.data:
        if trace.y is not None and any(y is not None for y in trace.y):
            trace_max = max(y for y in trace.y if y is not None)
            max_y = max(max_y, trace_max)

    # If max_y is still 0 (no valid y values), set a default height
    if max_y == 0:
        max_y = 1.0

    line_height = 0.05 * max_y

    # Load MOM data
    mom_data = loader.get_data(case, level2, 'MOM')
    x_position = mom_data[str(time)][0]
    std = mom_data[str(time)][1]
    fig.add_shape(
        type="line",
        x0=x_position,
        x1=x_position,
        y0=0,
        y1=line_height,
        name='MOM',
        line=dict(
            color='black',
            width=2,
        )
    )

    # Add a trace for the vertical marker legend entry
    fig.add_trace(
        go.Scatter(
            x=[x_position - std, x_position + std],  # Two points to create a visible line
            y=[line_height / 2, line_height / 2],  # Same height for a horizontal line
            mode='lines',
            name='Method of moments',  # Legend label for the vertical shape
            line=dict(color='black', width=2),
            showlegend=True
        )
    )

# Update the layout to accommodate both plots
fig.update_layout(
    yaxis2=dict(
        title='Count',
        overlaying='y',
        side='right'
    ),
    barmode='overlay',
    xaxis_title='V',  # Example with LaTeX
    yaxis_title='n(V)'  # Example with LaTeX and units
)

st.plotly_chart(fig)
