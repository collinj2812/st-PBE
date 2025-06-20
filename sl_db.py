import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import pickle
# import scienceplots
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import matplotlib.image as mpimg
import random

# plt.style.use('science')


def horizontal_ridgeline_plot(data, x_range=None, labels=None, figsize=(10, 6), colors=None,
                              alpha=0.7, overlap=0.7, fill=True, linewidth=2, linecolor='black',
                              title=None, xlabel=None, ylabel=None, grid=False, xlabels=True,
                              colormap='viridis'):
    """
    Create a horizontal ridgeline plot (joy plot) from precomputed density values.

    Parameters:
    -----------
    data : pandas.DataFrame, numpy.ndarray, or list of lists
        DataFrame where each column contains precomputed density values,
        or a 2D numpy array where each row is a distribution,
        or a list of lists where each inner list is a distribution.
    x_range : array-like, optional
        The x values that correspond to the precomputed densities.
        If None, will use numpy.arange(len(first_distribution)).
    labels : list, optional
        Labels for each distribution. If None, uses DataFrame column names or indices.
    figsize : tuple, default (10, 6)
        Size of the figure.
    colors : list or string, optional
        Colors for each distribution. If None, uses colormap.
    alpha : float, default 0.7
        Transparency of the fill.
    overlap : float, default 0.7
        Controls how much the distributions overlap.
        0 means no overlap, values > 1 create more spacing between distributions.
    fill : bool, default True
        Whether to fill the distributions.
    linewidth : float, default 2
        Width of the distribution outline.
    linecolor : string, default 'black'
        Color of the distribution outline. Use None to match the fill color.
    title : string, optional
        Title of the plot.
    xlabel : string, optional
        Label for the x-axis.
    ylabel : string, optional
        Label for the y-axis.
    grid : bool, default False
        Whether to show a grid.
    xlabels : bool, default True
        Whether to show the x labels.
    colormap : string, default 'viridis'
        Colormap to use if colors is None.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object.
    ax : matplotlib.axes.Axes
        The matplotlib Axes object.
    """
    # Convert data to DataFrame for consistent processing
    if isinstance(data, np.ndarray):
        # If it's a 2D array, each row is a distribution
        if data.ndim == 2:
            data_df = pd.DataFrame({f'Distribution {i + 1}': data[i, :] for i in range(data.shape[0])})
        # If it's a 1D array, treat it as a single distribution
        else:
            data_df = pd.DataFrame({'Distribution 1': data})
    elif isinstance(data, list):
        # Check if it's a list of lists or a single list
        if any(isinstance(item, (list, np.ndarray)) for item in data):
            # List of lists - each inner list is a distribution
            data_df = pd.DataFrame({f'Distribution {i + 1}': dist for i, dist in enumerate(data)})
        else:
            # Single list - treat as one distribution
            data_df = pd.DataFrame({'Distribution 1': data})
    elif isinstance(data, pd.DataFrame):
        # Already a DataFrame
        data_df = data
    else:
        raise TypeError("data must be a pandas DataFrame, numpy array, or list")

    n_distributions = len(data_df.columns)

    # Default x_range if not provided
    if x_range is None:
        x_range = np.arange(len(data_df.iloc[:, 0]))

    # Default labels if not provided
    if labels is None:
        labels = data_df.columns

    # Setup colors
    if colors is None:
        cmap = plt.get_cmap(colormap)
        colors = [cmap(i / max(1, n_distributions - 1)) for i in range(n_distributions)]
    elif isinstance(colors, str):
        colors = [colors] * n_distributions

    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)

    # Calculate spacing factor for distributions
    # For overlap > 1, we create more space between distributions
    spacing = 1.0
    if overlap > 1:
        spacing = overlap
        overlap = 0.7  # Reset to default for density scaling

    # Plot each distribution
    for i, col in enumerate(data_df.columns):
        # Get distribution values
        y_values = data_df[col].values

        # Calculate x offset for this distribution
        # This determines the horizontal position of the distribution
        x_pos = i * spacing

        # For filling, we need the left edge of the shape
        left = np.zeros_like(y_values) + x_pos

        # Get color for this distribution
        color = colors[i]

        # Set line color
        lc = linecolor if linecolor is not None else color

        # Plot the outline - for horizontal plot, swap x and y
        ax.plot(y_values + x_pos, x_range, color=lc, linewidth=linewidth, zorder=n_distributions - i + 1)

        # Fill if requested - for horizontal plot, use fill_betweenx
        if fill:
            ax.fill_betweenx(x_range, left, y_values + x_pos, alpha=alpha, color=color, zorder=n_distributions - i)

    # Set x-ticks and labels at the center of each distribution
    if xlabels:
        x_tick_pos = [i * spacing + 0.3 for i in range(n_distributions)]
        ax.set_xticks(x_tick_pos)
        ax.set_xticklabels(labels)
        # Rotate labels if there are many to prevent overlap
        if n_distributions > 5:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    else:
        ax.set_xticks([])

    # Set plot limits
    # Y-axis should cover the range of original values
    ax.set_ylim(min(x_range), max(x_range))

    # X-axis should include all distributions with padding
    x_limit_max = (n_distributions - 1) * spacing + 1.2
    ax.set_xlim(-0.05, x_limit_max)

    # Set grid
    if grid:
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Set labels and title
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Tight layout
    plt.tight_layout()

    return fig, ax


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

# Set page config
st.set_page_config(layout="wide")

# Custom CSS to set max-width
st.markdown("""
<style>
    .main > div {
        max-width: 1000px;
        margin: auto;
    }

    /* Optional: Also limit the width of the block container */
    .block-container {
        max-width: 1000px;
        padding-left: 5rem;
        padding-right: 5rem;
    }
</style>
""", unsafe_allow_html=True)

# Create a title
st.title("MPC for continuous crystallization")

st.markdown("""
This Streamlit app accompanies the paper A tutorial overview of model predictive control for continuous crystallization: current possibilities and future perspectives (2025, Collin R. Johnson, Kerstin Wohlgemuth and Sergio Lucia). It provides interactive visualizations of population balance equations and crystallizer modeling across three main areas:

- **Tab 1**: Population balance equation solution methods
- **Tab 2**: Continuous crystallizer reactor models
- **Tab 3**: Combined models
""")

# Create tabs for different functionality views
tab1, tab2, tab3 = st.tabs(["Population balance equation", "Crystallizer model", "Overall model"])

# Tab 1: Your original functionality (split into columns)
with tab1:
    st.header("Solution methods for the population balance equation")


    with st.expander("Details", expanded=False):
        st.markdown("""
        This tab compares numerical solution methods for the population balance equation (PBE), which describes how crystal size distributions evolve over time:
        
        ```
        ∂n(L,t)/∂t + ∂[G(L,t)n(L,t)]/∂L = B(L,t) + A(L,t)
        ```
        
        **Solution Methods:**
        - **OCFE**: Orthogonal collocation on finite elements
        - **DPBE**: Direct discretization with various schemes  
        - **MOM**: Method of moments (statistical approach)
        - **MC**: Monte Carlo simulation
        
        **Test Cases:**
        - Case 1: Pure growth (G > 0)
        - Case 2: Pure agglomeration (β > 0) 
        - Case 3: Growth + agglomeration + nucleation
        """)

    # Create two columns for side-by-side view
    left_col, right_col = st.columns([1,3])

    # Initialize session state variables
    if 'counter_active' not in st.session_state:
        st.session_state.counter_active = False
    if 'current_value' not in st.session_state:
        st.session_state.current_value = 10


    def toggle_counter():
        st.session_state.counter_active = not st.session_state.counter_active

    with right_col:

        with st.expander("📖 Explanations", expanded=False):
            st.markdown("""
            ### Understanding the Population Balance Equation

            The PBE describes how crystal size distribution evolves over time through:

            **Growth (G):** Crystals increase in size at a rate proportional to supersaturation
            - G = 0.5: Slow growth conditions
            - G = 1.0: Moderate growth rate  
            - G = 5.0: Fast growth conditions

            **Nucleation (N):** Formation of new crystals from solution
            - Primary nucleation occurs spontaneously
            - Secondary nucleation is induced by existing crystals

            **Agglomeration (β):** Crystals stick together forming larger particles
            - β = 0.1: Low agglomeration tendency
            - β = 0.5: High agglomeration rate
            """)


    # Add toggle button
    right_col.button(
        "Start/Stop",
        on_click=toggle_counter,
        type="primary" if not st.session_state.counter_active else "secondary"
    )

    # Add increment amount input
    increment = 1

    # Ensure current_value is an integer
    current_value = int(st.session_state.current_value)

    # container for image
    image_container = right_col.container()

    # Create a placeholder for the plots at the top
    plot_container = right_col.container()

    # Slider for time
    time_slider = right_col.slider(
        'Select time',
        min_value=0,
        max_value=50,
        value=current_value,
        step=1,
        label_visibility="visible",
        help=None,
        on_change=None,
        args=None,
        kwargs=None,
        disabled=False,
        key="time_slider"
    )

    # Sidebar for selecting case
    case_select = left_col.selectbox(
        'Case',
        options=['1', '2', '3']
    )
    case = f'case{case_select}'
    if case == 'case3':
        case = 'case4'  # case 3 in the paper is case 4 in the dashboard data

    # Case-specific parameters in an expander
    with left_col.expander("Case Parameters", expanded=False):
        # Growth rate G
        if case == 'case1' or case == 'case4':
            level2_select = st.selectbox(
                'Growth rate G',
                options=['0.5', '1.0', '5.0']
            )
            level2 = f'G_{level2_select}'
            G_value = level2_select
        else:
            st.selectbox(
                'Growth rate G',
                options=['0.0'],
                disabled=True,
                key='G_disabled'
            )
            G_value = '0.0'
            level2 = None

        # Agglomeration rate beta
        if case == 'case2':
            beta_select = st.selectbox(
                r'Agglomeration rate β',
                options=['0.1', '0.5']
            )
            level2 = f'beta_{beta_select}'
            beta_value = beta_select
        else:
            if case == 'case1':
                st.selectbox(
                    r'Agglomeration rate β',
                    options=['0.0'],
                    disabled=True,
                    key='beta_disabled'
                )
            elif case == 'case4':
                st.selectbox(
                    r'Agglomeration rate β',
                    options=['0.1'],
                    disabled=True,
                    key='beta_disabled'
                )

        if case == 'case1' or case == 'case2':
            st.selectbox(
                'Nucleation rate N',
                options=['0.0'],
                disabled=True,
                key='N_disabled'
            )
        elif case == 'case4':
            st.selectbox(
                'Nucleation rate N',
                options=['0.01'],
                disabled=True,
                key='N_disabled'
            )

    # Method selection
    method = left_col.multiselect(
        'Select Method',
        options=['OCFE', 'DPBE', 'MOM', 'MC'],
        default=['DPBE']
    )

    # Initialize dataframe based on case
    if case == 'case4':
        # Create dummy dataframe with enough points (100 to match histogram bins)
        x = np.linspace(0, 60, 100)
        df = pd.DataFrame({
            'x': x,
            'y': [None] * len(x),
            'label': [''] * len(x)
        })
    else:
        # Load exact solution data
        exact_data = loader.get_data(case, level2, 'exact')
        df = pd.DataFrame({
            'x': exact_data['xx'],
            'y': exact_data[str(time_slider)],
            'label': 'Exact solution'
        })

    # OCFE parameters
    if 'OCFE' in method:
        with left_col.expander("OCFE Parameters", expanded=False):
            level3_OCFE_select = st.selectbox(
                'Artificial diffusion D_a',
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
            'y': ocfe_data[str(time_slider)],
            'label': 'OCFE'
        })
        df = pd.concat([df, df_OCFE])

    # DPBE parameters
    if 'DPBE' in method:
        with left_col.expander("DPBE Parameters", expanded=False):
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
            'y': dpbe_data[str(time_slider)],
            'label': 'DPBE'
        })
        df = pd.concat([df, df_DPBE])

    # MC parameters
    if 'MC' in method:
        with left_col.expander("MC Parameters", expanded=False):
            level3_MC_select = st.selectbox(
                'Number of samples',
                options=['1000', '5000', '10000', '50000'],
                index=2
            )
        level3_MC = f'no_particles_{level3_MC_select}'
        # Load MC data
        mc_data = loader.get_data(case, level2, 'MC', level3_MC)
        data_MC = mc_data[str(time_slider)]
        no_begin = mc_data['0'].shape[1]
        no_now = data_MC.shape[1]

    # Add dummy data for case 4 if only MC is selected or no method is selected
    if case == 'case4' and (len(method) == 0 or (len(method) == 1 and ('MC' in method or 'MOM' in method))):
        df = pd.DataFrame({'x': [0, 60], 'y': [None, None], 'label': ''})

    # Update the current value in session state
    st.session_state.current_value = time_slider

    # Create the plots
    with plot_container.container():
        # Create the line chart using Plotly Express
        fig = px.line(df, x='x', y='y', color='label')

        # Add title and axis labels
        fig.update_layout(
            xaxis_title='V',
            yaxis_title='n(V)'
        )

        if 'MC' in method:
            # Calculate histogram data
            hist_values, bin_edges = np.histogram(data_MC, bins=100, density=True)

            # Add the histogram trace
            fig.add_trace(
                go.Line(
                    x=bin_edges[:-1],
                    y=hist_values * no_now / no_begin,
                    name='Monte Carlo',
                    opacity=0.5,
                )
            )

        if 'MOM' in method:
            # Calculate max y value for scaling
            max_y = 0
            for trace in fig.data:
                if trace.y is not None and any(y is not None for y in trace.y):
                    trace_max = max(y for y in trace.y if y is not None)
                    max_y = max(max_y, trace_max)

            if max_y == 0:
                max_y = 1.0

            line_height = 0.05 * max_y

            # Load and plot MOM data
            mom_data = loader.get_data(case, level2, 'MOM')
            x_position = mom_data[str(time_slider)][0]
            std = mom_data[str(time_slider)][1]

            # Add vertical line
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

            # Add legend entry
            fig.add_trace(
                go.Scatter(
                    x=[x_position - std, x_position + std],
                    y=[line_height / 2, line_height / 2],
                    mode='lines',
                    name='Method of moments',
                    line=dict(color='black', width=2),
                    showlegend=True
                )
            )

        # Update layout
        fig.update_layout(
            yaxis2=dict(
                title='Count',
                overlaying='y',
                side='right'
            ),
            barmode='overlay',
            xaxis_title='V',
            yaxis_title='n(V)'
        )

        st.plotly_chart(fig)

    with image_container.container():
        # split
        subcol1, subcol2 = st.columns([3, 2])
        with subcol1:
            # load monte carlo data
            if case_select == '3':
                case_MC_select = 4
            else:
                case_MC_select = case_select
            with open(f'data/casecase{case_MC_select}/{level2}/MC/no_particles_50000.pkl', 'rb') as f:
                data = pickle.load(f)

            circle_sizes = random.sample(list(data[str(time_slider)][0]*0.05), 500)


            # Load the image
            img = mpimg.imread("figures/empty_batch_crystallizer.jpg")
            img_height, img_width = img.shape[0], img.shape[1]

            # Create figure and axis
            fig, ax = plt.subplots(figsize=(10, 8))

            # Display the image
            ax.imshow(img)

            # Define the domain where circles can be placed (as a percentage of image size)
            # Adjust these values based on where in the image you want circles to appear
            domain_x_min = 0.28 * img_width  # 20% from left edge
            domain_x_max = 0.72 * img_width  # 80% from left edge (20% from right edge)
            domain_y_min = 0.3 * img_height  # 30% from top
            domain_y_max = 0.82 * img_height  # 90% from top (10% from bottom)

            # Scale factor for circle size
            scale_factor = 5  # Adjust based on your image size

            # Plot circles at random positions within the domain
            for diameter in circle_sizes:
                # Generate random position
                x = random.uniform(domain_x_min, domain_x_max)
                y = random.uniform(domain_y_min, domain_y_max)

                # Calculate radius (scaled)
                radius = diameter * scale_factor

                # Create a circle patch
                circle = Circle((x, y), radius,
                               edgecolor='None',
                               facecolor='black',
                               linewidth=2)

                # Add the circle to the plot
                ax.add_patch(circle)

            # Turn off axis labels since this is an image
            ax.axis('off')


            # Show the plot
            st.pyplot(fig)

    # Handle counter increment
    if st.session_state.counter_active:
        st.session_state.current_value = min(
            50,  # max_value
            int(st.session_state.current_value + increment)
        )
        time.sleep(0.5)
        st.rerun()

# Tab 2: Right side functionality (split into columns)
with tab2:
    st.header("Solution for continuous phase of crystallizers")

    # Create two columns
    left_col, right_col = st.columns([1, 3])

    # Initialize session state variables for tab2 counter
    if 'tab2_counter_active' not in st.session_state:
        st.session_state.tab2_counter_active = False
    if 'tab2_current_time' not in st.session_state:
        st.session_state.tab2_current_time = 20


    # Toggle function for tab2 counter
    def toggle_tab2_counter():
        st.session_state.tab2_counter_active = not st.session_state.tab2_counter_active


    # Left column - Controls
    with left_col:
        st.subheader("Analysis Parameters")

        # Case selection
        case_select = st.selectbox(
            'Crystallizer',
            options=['MSMPR', 'MSMPR Cascade', 'Tubular'],
            key="MSMPR"
        )

        # Analysis parameters
        with st.expander("Model parameters", expanded=False):
            if case_select == 'MSMPR Cascade':
                parameter = st.slider(
                    'Number of single crystallizers',
                    min_value=2, max_value=6, value=2, step=1,
                    key="msmpr_stages_no"
                )
            elif case_select == 'Tubular':
                parameter = st.slider(
                    'Number of finite volumes',
                    min_value=10, max_value=200, value=80, step=1,
                    key="tubular_stages_no"
                )
                diffusion = st.slider(
                    'Diffusion coefficient',
                    min_value=0.001, max_value=1.0, value=0.001, step=0.001,
                    key="tubular_diffusion"
                )

        # Time selection
        time_point = st.slider(
            'Time Point',
            min_value=0,
            max_value=400,
            value=st.session_state.tab2_current_time,
            step=1,
            key="tab2_left_time"
        )

        # Add Start/Stop button
        st.button(
            "Start/Stop",
            on_click=toggle_tab2_counter,
            type="primary" if not st.session_state.tab2_counter_active else "secondary",
            key="tab2_counter_button"
        )



    # Right column - Plot
    with right_col:
        if case_select == 'MSMPR':
            # load MSMPR pickle data
            with open(f'data/contdata/data_MSMPR_1_stages.pkl', 'rb') as f:
                data = pickle.load(f)
            # split in three
            subcol1, subcol2, subcol3 = st.columns([3, 1, 2])
            with subcol1:
                # Picture
                st.image("figures/MSMPR.jpg", caption="MSMPR sketch", width=300)
            with subcol2:
                st.write("Model inputs:")
                st.metric(label='T_{j,in', value=data['T_j_in'][time_point])
                st.metric(label='F_{j,in', value=data['F_j_in'][time_point])
                st.metric(label='F_feed', value=data['F_feed'][time_point])
                st.metric(label='T_feed', value=data['T_feed'][time_point])
            with subcol3:
                st.write("Model states:")

                fig, ax = plt.subplots(3, 1, figsize=(2, 3))

                ax[0].scatter([1], data['c'][time_point])

                # set y limits
                min_mult = 0.98
                max_mult = 1.02

                # set y limits
                ax[0].set_ylim([data['c'].min() * min_mult, data['c'].max() * max_mult])
                ax[0].set_ylabel('c [g/g]')

                ax[1].scatter([1], data['T_PM'][time_point])
                ax[1].set_ylim([data['T_PM'].min() * min_mult, data['T_PM'].max() * max_mult])
                ax[1].set_ylabel('T_PM [K]')

                ax[2].scatter([1], data['T_TM'][time_point])
                ax[2].set_ylim([data['T_TM'].min() * min_mult, data['T_TM'].max() * max_mult])
                ax[2].set_ylabel('T_TM [K]')

                ax[0].spines['top'].set_visible(False)
                ax[0].spines['right'].set_visible(False)
                ax[0].spines['bottom'].set_visible(False)
                ax[1].spines['top'].set_visible(False)
                ax[1].spines['right'].set_visible(False)
                ax[1].spines['bottom'].set_visible(False)
                ax[2].spines['top'].set_visible(False)
                ax[2].spines['right'].set_visible(False)
                ax[2].spines['bottom'].set_visible(False)

                ax[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                ax[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                ax[2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                fig.align_ylabels()
                # fig.tight_layout()
                st.pyplot(fig)

        if case_select == 'MSMPR Cascade':
            # split in two
            subcol1, subcol2 = st.columns([1, 4])
            with subcol2:
                # check for number of single crystallizers
                if parameter == 2:
                    # Picture
                    st.image("figures/MSMPR_2stage_dashboard.jpg", caption="MSMPR cascade sketch", width=600)
                elif parameter == 3:
                    # Picture
                    st.image("figures/MSMPR_3stage_dashboard.jpg", caption="MSMPR cascade sketch", width=600)
                elif parameter == 4:
                    # Picture
                    st.image("figures/MSMPR_4stage_dashboard.jpg", caption="MSMPR cascade sketch", width=600)
                elif parameter == 5:
                    # Picture
                    st.image("figures/MSMPR_5stage_dashboard.jpg", caption="MSMPR cascade sketch", width=600)
                elif parameter == 6:
                    # Picture
                    st.image("figures/MSMPR_6stage_dashboard.jpg", caption="MSMPR cascade sketch", width=600)

            # load correct data
            if parameter == 2:
                with open(f'data/contdata/data_MSMPR_2_stages.pkl', 'rb') as f:
                    data = pickle.load(f)
            elif parameter == 3:
                with open(f'data/contdata/data_MSMPR_3_stages.pkl', 'rb') as f:
                    data = pickle.load(f)
            elif parameter == 4:
                with open(f'data/contdata/data_MSMPR_4_stages.pkl', 'rb') as f:
                    data = pickle.load(f)
            elif parameter == 5:
                with open(f'data/contdata/data_MSMPR_5_stages.pkl', 'rb') as f:
                    data = pickle.load(f)
            elif parameter == 6:
                with open(f'data/contdata/data_MSMPR_6_stages.pkl', 'rb') as f:
                    data = pickle.load(f)

            # split again
            subcol1, subcol2 = st.columns([1, 4])
            with subcol1:
                st.write("Model inputs:")
                st.metric(label='T_{j,in', value=data['T_j_in'][time_point][0])
                st.metric(label='F_{j,in', value=data['F_j_in'][time_point][0])
                st.metric(label='F_feed', value=data['F_feed'][time_point])
                st.metric(label='T_feed', value=data['T_feed'][time_point])

            with subcol2:
                if parameter == 2:
                    st.write("Model states:")

                    fig, ax = plt.subplots(3, 1, figsize=(6, 4), sharex='col', sharey='row')

                    pos_1 = 0.5
                    pos_2 = 2

                    ax[0].scatter([pos_1], data['c'][time_point][0])
                    ax[0].scatter([pos_2], data['c'][time_point][1])

                    ax[0].set_xlim([0, 3])

                    # set y limits
                    min_mult = 0.98
                    max_mult = 1.02

                    # set y limits
                    ax[0].set_ylim([data['c'].min() * min_mult, data['c'].max() * max_mult])
                    ax[0].set_ylabel('c [g/g]')

                    ax[1].scatter([pos_1], data['T_PM'][time_point][0])
                    ax[1].scatter([pos_2], data['T_PM'][time_point][1])
                    ax[1].set_ylim([data['T_PM'].min() * min_mult, data['T_PM'].max() * max_mult])
                    ax[1].set_ylabel('T_PM [K]')

                    ax[2].scatter([pos_1], data['T_TM'][time_point][0])
                    ax[2].scatter([pos_2], data['T_TM'][time_point][1])
                    ax[2].set_ylim([data['T_TM'].min() * min_mult, data['T_TM'].max() * max_mult])
                    ax[2].set_ylabel('T_TM [K]')

                    # set all spines but far left to invisible
                    for i in range(3):
                        ax[i].spines['top'].set_visible(False)
                        ax[i].spines['right'].set_visible(False)
                        ax[i].spines['top'].set_visible(False)
                        ax[i].spines['right'].set_visible(False)
                        ax[i].spines['bottom'].set_visible(False)
                        ax[i].spines['bottom'].set_visible(False)
                        ax[i].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

                    fig.align_ylabels()
                    # fig.tight_layout()
                    st.pyplot(fig)

                elif parameter == 3:
                    st.write("Model states:")

                    fig, ax = plt.subplots(3, 1, figsize=(6, 4), sharex='col', sharey='row')
                    pos_1 = 0.2
                    pos_2 = 1.65
                    pos_3 = 3.0
                    ax[0].scatter([pos_1], data['c'][time_point][0])
                    ax[0].scatter([pos_2], data['c'][time_point][1])
                    ax[0].scatter([pos_3], data['c'][time_point][2])
                    ax[0].set_xlim([0, 4])

                    # set y limits
                    min_mult = 0.98
                    max_mult = 1.02
                    # set y limits
                    ax[0].set_ylim([data['c'].min() * min_mult, data['c'].max() * max_mult])
                    ax[0].set_ylabel('c [g/g]')
                    ax[1].scatter([pos_1], data['T_PM'][time_point][0])
                    ax[1].scatter([pos_2], data['T_PM'][time_point][1])
                    ax[1].scatter([pos_3], data['T_PM'][time_point][2])
                    ax[1].set_ylim([data['T_PM'].min() * min_mult, data['T_PM'].max() * max_mult])
                    ax[1].set_ylabel('T_PM [K]')
                    ax[2].scatter([pos_1], data['T_TM'][time_point][0])
                    ax[2].scatter([pos_2], data['T_TM'][time_point][1])
                    ax[2].scatter([pos_3], data['T_TM'][time_point][2])
                    ax[2].set_ylim([data['T_TM'].min() * min_mult, data['T_TM'].max() * max_mult])
                    ax[2].set_ylabel('T_TM [K]')
                    # set all spines but far left to invisible
                    for i in range(3):
                        ax[i].spines['top'].set_visible(False)
                        ax[i].spines['right'].set_visible(False)
                        ax[i].spines['top'].set_visible(False)
                        ax[i].spines['right'].set_visible(False)
                        ax[i].spines['bottom'].set_visible(False)
                        ax[i].spines['bottom'].set_visible(False)
                        ax[i].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                    fig.align_ylabels()
                    # fig.tight_layout()
                    st.pyplot(fig)
                elif parameter == 4:
                    st.write("Model states:")

                    fig, ax = plt.subplots(3, 1, figsize=(8, 4), sharex='col', sharey='row')
                    pos_1 = 0.2
                    pos_2 = 1.2
                    pos_3 = 2.2
                    pos_4 = 3.2
                    ax[0].scatter([pos_1], data['c'][time_point][0])
                    ax[0].scatter([pos_2], data['c'][time_point][1])
                    ax[0].scatter([pos_3], data['c'][time_point][2])
                    ax[0].scatter([pos_4], data['c'][time_point][3])
                    ax[0].set_xlim([0, 4])

                    # set y limits
                    min_mult = 0.98
                    max_mult = 1.02
                    # set y limits
                    ax[0].set_ylim([data['c'].min() * min_mult, data['c'].max() * max_mult])
                    ax[0].set_ylabel('c [g/g]')
                    ax[1].scatter([pos_1], data['T_PM'][time_point][0])
                    ax[1].scatter([pos_2], data['T_PM'][time_point][1])
                    ax[1].scatter([pos_3], data['T_PM'][time_point][2])
                    ax[1].scatter([pos_4], data['T_PM'][time_point][3])
                    ax[1].set_ylim([data['T_PM'].min() * min_mult, data['T_PM'].max() * max_mult])
                    ax[1].set_ylabel('T_PM [K]')
                    ax[2].scatter([pos_1], data['T_TM'][time_point][0])
                    ax[2].scatter([pos_2], data['T_TM'][time_point][1])
                    ax[2].scatter([pos_3], data['T_TM'][time_point][2])
                    ax[2].scatter([pos_4], data['T_TM'][time_point][3])
                    ax[2].set_ylim([data['T_TM'].min() * min_mult, data['T_TM'].max() * max_mult])
                    ax[2].set_ylabel('T_TM [K]')
                    # set all spines but far left to invisible
                    for i in range(3):
                        ax[i].spines['top'].set_visible(False)
                        ax[i].spines['right'].set_visible(False)
                        ax[i].spines['top'].set_visible(False)
                        ax[i].spines['right'].set_visible(False)
                        ax[i].spines['bottom'].set_visible(False)
                        ax[i].spines['bottom'].set_visible(False)
                        ax[i].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                    fig.align_ylabels()
                    # fig.tight_layout()
                    st.pyplot(fig)
                elif parameter == 5:
                    st.write("Model states:")

                    fig, ax = plt.subplots(3, 1, figsize=(8, 4), sharex='col', sharey='row')
                    pos_1 = 0.2
                    pos_2 = 1.2
                    pos_3 = 2.2
                    pos_4 = 3.2
                    pos_5 = 4.2
                    ax[0].scatter([pos_1], data['c'][time_point][0])
                    ax[0].scatter([pos_2], data['c'][time_point][1])
                    ax[0].scatter([pos_3], data['c'][time_point][2])
                    ax[0].scatter([pos_4], data['c'][time_point][3])
                    ax[0].scatter([pos_5], data['c'][time_point][4])
                    ax[0].set_xlim([0, 5])

                    # set y limits
                    min_mult = 0.98
                    max_mult = 1.02
                    # set y limits
                    ax[0].set_ylim([data['c'].min() * min_mult, data['c'].max() * max_mult])
                    ax[0].set_ylabel('c [g/g]')
                    ax[1].scatter([pos_1], data['T_PM'][time_point][0])
                    ax[1].scatter([pos_2], data['T_PM'][time_point][1])
                    ax[1].scatter([pos_3], data['T_PM'][time_point][2])
                    ax[1].scatter([pos_4], data['T_PM'][time_point][3])
                    ax[1].scatter([pos_5], data['T_PM'][time_point][4])
                    ax[1].set_ylim([data['T_PM'].min() * min_mult, data['T_PM'].max() * max_mult])
                    ax[1].set_ylabel('T_PM [K]')
                    ax[2].scatter([pos_1], data['T_TM'][time_point][0])
                    ax[2].scatter([pos_2], data['T_TM'][time_point][1])
                    ax[2].scatter([pos_3], data['T_TM'][time_point][2])
                    ax[2].scatter([pos_4], data['T_TM'][time_point][3])
                    ax[2].scatter([pos_5], data['T_TM'][time_point][4])
                    ax[2].set_ylim([data['T_TM'].min() * min_mult, data['T_TM'].max() * max_mult])
                    ax[2].set_ylabel('T_TM [K]')
                    # set all spines but far left to invisible
                    for i in range(3):
                        ax[i].spines['top'].set_visible(False)
                        ax[i].spines['right'].set_visible(False)
                        ax[i].spines['top'].set_visible(False)
                        ax[i].spines['right'].set_visible(False)
                        ax[i].spines['bottom'].set_visible(False)
                        ax[i].spines['bottom'].set_visible(False)
                        ax[i].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                    fig.align_ylabels()
                    # fig.tight_layout()
                    st.pyplot(fig)
                elif parameter == 6:
                    st.write("Model states:")

                    fig, ax = plt.subplots(3, 1, figsize=(8, 4), sharex='col', sharey='row')
                    pos_1 = 0.1
                    pos_2 = 1.05
                    pos_3 = 2.15
                    pos_4 = 3.25
                    pos_5 = 4.35
                    pos_6 = 5.45
                    ax[0].scatter([pos_1], data['c'][time_point][0])
                    ax[0].scatter([pos_2], data['c'][time_point][1])
                    ax[0].scatter([pos_3], data['c'][time_point][2])
                    ax[0].scatter([pos_4], data['c'][time_point][3])
                    ax[0].scatter([pos_5], data['c'][time_point][4])
                    ax[0].scatter([pos_6], data['c'][time_point][5])
                    ax[0].set_xlim([0, 6])

                    # set y limits
                    min_mult = 0.98
                    max_mult = 1.02
                    # set y limits
                    ax[0].set_ylim([data['c'].min() * min_mult, data['c'].max() * max_mult])
                    ax[0].set_ylabel('c [g/g]')
                    ax[1].scatter([pos_1], data['T_PM'][time_point][0])
                    ax[1].scatter([pos_2], data['T_PM'][time_point][1])
                    ax[1].scatter([pos_3], data['T_PM'][time_point][2])
                    ax[1].scatter([pos_4], data['T_PM'][time_point][3])
                    ax[1].scatter([pos_5], data['T_PM'][time_point][4])
                    ax[1].scatter([pos_6], data['T_PM'][time_point][5])
                    ax[1].set_ylim([data['T_PM'].min() * min_mult, data['T_PM'].max() * max_mult])
                    ax[1].set_ylabel('T_PM [K]')
                    ax[2].scatter([pos_1], data['T_TM'][time_point][0])
                    ax[2].scatter([pos_2], data['T_TM'][time_point][1])
                    ax[2].scatter([pos_3], data['T_TM'][time_point][2])
                    ax[2].scatter([pos_4], data['T_TM'][time_point][3])
                    ax[2].scatter([pos_5], data['T_TM'][time_point][4])
                    ax[2].scatter([pos_6], data['T_TM'][time_point][5])
                    ax[2].set_ylim([data['T_TM'].min() * min_mult, data['T_TM'].max() * max_mult])
                    ax[2].set_ylabel('T_TM [K]')
                    # set all spines but far left to invisible
                    for i in range(3):
                        ax[i].spines['top'].set_visible(False)
                        ax[i].spines['right'].set_visible(False)
                        ax[i].spines['top'].set_visible(False)
                        ax[i].spines['right'].set_visible(False)
                        ax[i].spines['bottom'].set_visible(False)
                        ax[i].spines['bottom'].set_visible(False)
                        ax[i].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                    fig.align_ylabels()
                    # fig.tight_layout()
                    st.pyplot(fig)
        elif case_select == 'Tubular':
            # load tubular data
            with open(f'data/contdata/data_Tubular_{parameter}_finite_volumes.pkl', 'rb') as f:
                data = pickle.load(f)

            # split in two
            subcol1, subcol2 = st.columns([1, 4])
            with subcol2:
                # Picture
                st.image("figures/tubular.jpg", caption="Tubular crystallizer sketch", width=600)


            # split in two
            subcol1, subcol2 = st.columns([1, 4])
            with subcol1:
                st.write("Model inputs:")
                st.metric(label='T_j,in', value=data['T_j_in'][time_point])
                st.metric(label='F_j,in', value=data['F_j'][time_point])
                st.metric(label='F_feed', value=data['F'][time_point])
                st.metric(label='T_feed', value=data['T_in'][time_point])
            with subcol2:
                fig, ax = plt.subplots(3, 1, figsize=[6, 6], sharex='col')

                # Plot the data
                ax[0].plot(data['L'], data['c'][time_point])
                ax[0].set_ylabel('c [g/g]')
                ax[0].set_ylim([data['c'].min() * 0.98, data['c'].max() * 1.02])
                ax[1].plot(data['L'], data['T_PM'][time_point])
                ax[1].set_ylabel('T_PM [K]')
                ax[1].set_ylim([data['T_PM'].min() * 0.98, data['T_PM'].max() * 1.02])
                ax[2].plot(data['L'], data['T_TM'][time_point])
                ax[2].set_ylabel('T_TM [K]')
                ax[2].set_ylim([data['T_TM'].min() * 0.98, data['T_TM'].max() * 1.02])
                ax[2].set_xlabel('Length [m]')

                fig.align_ylabels()
                # fig.tight_layout()


                st.pyplot(fig)




        if case_select == 'MSMPR' or case_select == 'MSMPR Cascade':
            # plt color cycle
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

            fig, ax = plt.subplots(3, 1, figsize=(6, 3), sharex='col')

            # Plot the data
            if case_select == 'MSMPR':
                ax[0].plot(data['c'])
                ax[0].set_ylabel('c [g/g]')
                ax[1].plot(data['T_PM'])
                ax[1].set_ylabel('T_PM [K]')
                ax[2].plot(data['T_TM'])
                ax[2].set_ylabel('T_TM [K]')
                # display current values as x
                ax[0].scatter(time_point, data['c'][time_point], color=colors[0], label='Time point', s=50)
                ax[1].scatter(time_point, data['T_PM'][time_point], color=colors[0], label='Time point', s=50)
                ax[2].scatter(time_point, data['T_TM'][time_point], color=colors[0], label='Time point', s=50)

            elif case_select == 'MSMPR Cascade':
                ax[0].plot(data['c'])
                ax[0].set_ylabel('c [g/g]')
                ax[1].plot(data['T_PM'])
                ax[1].set_ylabel('T_PM [K]')
                ax[2].plot(data['T_TM'])
                ax[2].set_ylabel('T_TM [K]')
                # display current values as x
                for i in range(parameter):
                    ax[0].scatter(time_point, data['c'][time_point][i], color=colors[i], label='Time point', s=20)
                    ax[1].scatter(time_point, data['T_PM'][time_point][i], color=colors[i], label='Time point', s=20)
                    ax[2].scatter(time_point, data['T_TM'][time_point][i], color=colors[i], label='Time point', s=20)

            ax[2].set_xlabel('Time [s]')

            # fig.tight_layout()
            fig.align_ylabels()

            st.pyplot(fig)


    # Handle counter increment logic
    if st.session_state.tab2_counter_active:
        # Update the counter value
        st.session_state.tab2_current_time = min(
            400,  # max_value
            int(st.session_state.tab2_current_time + 5)
        )
        # If we've reached the max value, loop back to the beginning
        if st.session_state.tab2_current_time >= 400:
            st.session_state.tab2_current_time = 0

        # Delay based on animation speed
        time.sleep(0.1)
        # Rerun the app to update the UI
        st.rerun()


# Tab 3: Bottom functionality (with columns)
with tab3:

    st.header("Solution for continuous phase of crystallizers")

    # Create two columns
    left_col, right_col = st.columns([1, 3])

    # Initialize session state variables for tab3 counter (UNIQUE FOR TAB 3)
    if 'tab3_counter_active' not in st.session_state:
        st.session_state.tab3_counter_active = False
    if 'tab3_current_time' not in st.session_state:
        st.session_state.tab3_current_time = 20


    # Toggle function for tab3 counter (UNIQUE FOR TAB 3)
    def toggle_tab3_counter():
        st.session_state.tab3_counter_active = not st.session_state.tab3_counter_active


    # Left column - Controls
    with left_col:
        st.subheader("Analysis Parameters")

        # Case selection
        case_select = st.selectbox(
            'Crystallizer',
            options=['MSMPR', 'MSMPR Cascade', 'Tubular'],
            key="MSMPR2"
        )

        # Analysis parameters
        with st.expander("Model parameters", expanded=False):
            if case_select == 'MSMPR Cascade':
                parameter = st.slider(
                    'Number of single crystallizers',
                    min_value=2, max_value=6, value=2, step=1,
                    key="msmpr_stages_no2"
                )
            elif case_select == 'Tubular':
                parameter = st.slider(
                    'Number of finite volumes',
                    min_value=10, max_value=200, value=80, step=10,
                    key="tubular_stages_no2"
                )

        # Time selection (USING TAB3 CURRENT TIME)
        time_point = st.slider(
            'Time Point',
            min_value=0,
            max_value=199,
            value=st.session_state.tab3_current_time,  # Changed to tab3_current_time
            step=1,
            key="tab3_left_time"  # Changed key
        )

        # Add Start/Stop button (USING TAB3 FUNCTION AND STATE)
        st.button(
            "Start/Stop",
            on_click=toggle_tab3_counter,  # Changed to tab3 function
            type="primary" if not st.session_state.tab3_counter_active else "secondary",  # Changed to tab3 state
            key="tab3_counter_button"  # Changed key
        )



    # Right column - Plot
    with right_col:
        if case_select == 'MSMPR':
            # load MSMPR pickle data
            with open(f'data/fulldata/data_MSMPR_1_stages_DPBE.pkl', 'rb') as f:
                data = pickle.load(f)
            # split in three
            subcol1, subcol2, subcol3 = st.columns([3, 1, 2])
            with subcol1:
                # Picture
                st.image("figures/MSMPR.jpg", caption="MSMPR sketch", width=300)
            with subcol2:
                st.write("Model inputs:")
                st.metric(label='T_j,in', value=data['T_j_in'][time_point])
                st.metric(label='F_j,in', value=data['F_j_in'][time_point])
                st.metric(label='F_feed', value=data['F_feed'][time_point])
                st.metric(label='T_feed', value=data['T_feed'][time_point])
            with subcol3:
                st.write("Model states:")

                fig, ax = plt.subplots(4, 1, figsize=(2, 3))

                ax[0].scatter([1], data['c'][time_point])

                # set y limits
                min_mult = 0.98
                max_mult = 1.02

                # set y limits
                ax[0].set_ylim([data['c'].min() * min_mult, data['c'].max() * max_mult])
                ax[0].set_ylabel('c [g/g]')

                ax[1].scatter([1], data['T_PM'][time_point])
                ax[1].set_ylim([data['T_PM'].min() * min_mult, data['T_PM'].max() * max_mult])
                ax[1].set_ylabel('T_PM [K]')

                ax[2].scatter([1], data['T_TM'][time_point])
                ax[2].set_ylim([data['T_TM'].min() * min_mult, data['T_TM'].max() * max_mult])
                ax[2].set_ylabel('T_TM [K]')

                # ax[3].plot(data['L_i'], data['n'][time_point,:])
                ax[3].plot(-1*data['n'][time_point,:] * data['L_i'], data['L_i'])
                ax[3].set_xlim([-1.1*(data['n'][time_point,:] * data['L_i']).max(), 1.1*(data['n'][time_point,:] * data['L_i']).max()])
                ax[3].set_ylim([0, 0.005])
                ax[3].set_ylabel('n_L [-]')

                ax[0].spines['top'].set_visible(False)
                ax[0].spines['right'].set_visible(False)
                ax[0].spines['bottom'].set_visible(False)
                ax[1].spines['top'].set_visible(False)
                ax[1].spines['right'].set_visible(False)
                ax[1].spines['bottom'].set_visible(False)
                ax[2].spines['top'].set_visible(False)
                ax[2].spines['right'].set_visible(False)
                ax[2].spines['bottom'].set_visible(False)
                ax[3].spines['top'].set_visible(False)
                ax[3].spines['right'].set_visible(False)
                ax[3].spines['bottom'].set_visible(False)

                ax[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                ax[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                ax[2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                ax[3].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                fig.align_ylabels()
                # fig.tight_layout()
                st.pyplot(fig)

        if case_select == 'MSMPR Cascade':
            # split in two
            subcol1, subcol2 = st.columns([1, 4])
            with subcol2:
                # check for number of single crystallizers
                if parameter == 2:
                    # Picture
                    st.image("figures/MSMPR_2stage_dashboard.jpg", caption="MSMPR cascade sketch", width=600)
                elif parameter == 3:
                    # Picture
                    st.image("figures/MSMPR_3stage_dashboard.jpg", caption="MSMPR cascade sketch", width=600)
                elif parameter == 4:
                    # Picture
                    st.image("figures/MSMPR_4stage_dashboard.jpg", caption="MSMPR cascade sketch", width=600)
                elif parameter == 5:
                    # Picture
                    st.image("figures/MSMPR_5stage_dashboard.jpg", caption="MSMPR cascade sketch", width=600)
                elif parameter == 6:
                    # Picture
                    st.image("figures/MSMPR_6stage_dashboard.jpg", caption="MSMPR cascade sketch", width=600)

            # load correct data
            if parameter == 2:
                with open(f'data/fulldata/data_MSMPR_2_stages_DPBE.pkl', 'rb') as f:
                    data = pickle.load(f)
            elif parameter == 3:
                with open(f'data/fulldata/data_MSMPR_3_stages_DPBE.pkl', 'rb') as f:
                    data = pickle.load(f)
            elif parameter == 4:
                with open(f'data/fulldata/data_MSMPR_4_stages_DPBE.pkl', 'rb') as f:
                    data = pickle.load(f)
            elif parameter == 5:
                with open(f'data/fulldata/data_MSMPR_5_stages_DPBE.pkl', 'rb') as f:
                    data = pickle.load(f)
            elif parameter == 6:
                with open(f'data/fulldata/data_MSMPR_6_stages_DPBE.pkl', 'rb') as f:
                    data = pickle.load(f)

            # split again
            subcol1, subcol2 = st.columns([1, 5])
            with subcol1:
                st.write("Model inputs:")
                st.metric(label='T_j,in', value=data['T_j_in'][time_point][0])
                st.metric(label='F_j,in', value=data['F_j_in'][time_point][0])
                st.metric(label='F_feed', value=data['F_feed'][time_point])
                st.metric(label='T_feed', value=data['T_feed'][time_point])

            with subcol2:
                if parameter == 2:
                    fig, ax = plt.subplots(1, 2, figsize=(6, 4), sharey='row')

                    n_classes = data['L_i'].shape[0]
                    ax[0].plot(data['L_i'], data['n'][time_point,:n_classes])
                    ax[1].plot(data['L_i'], data['n'][time_point,n_classes:])
                    ax[0].set_xlim([0, 0.01])
                    ax[0].set_ylabel('n_L [-]')
                    ax[0].set_xlabel('L [m]')
                    ax[1].set_xlabel('L [m]')

                    for i in range(2):
                        ax[i].spines['top'].set_visible(False)
                        ax[i].spines['right'].set_visible(False)
                        ax[i].spines['top'].set_visible(False)
                        ax[i].spines['right'].set_visible(False)

                    fig.align_ylabels()
                    # fig.tight_layout()
                    st.pyplot(fig)

                elif parameter == 3:
                    fig, ax = plt.subplots(1, 3, figsize=(6, 4), sharey='row')

                    n_classes = data['L_i'].shape[0]
                    ax[0].plot(data['L_i'], data['n'][time_point,:n_classes])
                    ax[1].plot(data['L_i'], data['n'][time_point,n_classes:n_classes*2])
                    ax[2].plot(data['L_i'], data['n'][time_point,n_classes*2:])
                    ax[0].set_xlim([0, 0.01])
                    ax[0].set_ylabel('n_L [-]')
                    ax[0].set_xlabel('L [m]')
                    ax[1].set_xlabel('L [m]')
                    ax[2].set_xlabel('L [m]')

                    for i in range(3):
                        ax[i].spines['top'].set_visible(False)
                        ax[i].spines['right'].set_visible(False)
                        ax[i].spines['top'].set_visible(False)
                        ax[i].spines['right'].set_visible(False)

                    fig.align_ylabels()
                    # fig.tight_layout()
                    st.pyplot(fig)
                elif parameter == 4:
                    fig, ax = plt.subplots(1, 4, figsize=(6, 4), sharey='row')

                    n_classes = data['L_i'].shape[0]
                    ax[0].plot(data['L_i'], data['n'][time_point,:n_classes])
                    ax[1].plot(data['L_i'], data['n'][time_point,n_classes:n_classes*2])
                    ax[2].plot(data['L_i'], data['n'][time_point,n_classes*2:n_classes*3])
                    ax[3].plot(data['L_i'], data['n'][time_point,n_classes*3:])
                    ax[0].set_xlim([0, 0.01])
                    ax[0].set_ylabel('n_L [-]')
                    ax[0].set_xlabel('L [m]')
                    ax[1].set_xlabel('L [m]')
                    ax[2].set_xlabel('L [m]')
                    ax[3].set_xlabel('L [m]')

                    for i in range(4):
                        ax[i].spines['top'].set_visible(False)
                        ax[i].spines['right'].set_visible(False)
                        ax[i].spines['top'].set_visible(False)
                        ax[i].spines['right'].set_visible(False)

                    fig.align_ylabels()
                    # fig.tight_layout()
                    st.pyplot(fig)
                elif parameter == 5:
                    fig, ax = plt.subplots(1, 5, figsize=(6, 4), sharey='row')

                    n_classes = data['L_i'].shape[0]
                    ax[0].plot(data['L_i'], data['n'][time_point,:n_classes])
                    ax[1].plot(data['L_i'], data['n'][time_point,n_classes:n_classes*2])
                    ax[2].plot(data['L_i'], data['n'][time_point,n_classes*2:n_classes*3])
                    ax[3].plot(data['L_i'], data['n'][time_point,n_classes*3:n_classes*4])
                    ax[4].plot(data['L_i'], data['n'][time_point,n_classes*4:])
                    ax[0].set_xlim([0, 0.01])
                    ax[0].set_ylabel('n_L [-]')
                    ax[0].set_xlabel('L [m]')
                    ax[1].set_xlabel('L [m]')
                    ax[2].set_xlabel('L [m]')
                    ax[3].set_xlabel('L [m]')
                    ax[4].set_xlabel('L [m]')

                    for i in range(5):
                        ax[i].spines['top'].set_visible(False)
                        ax[i].spines['right'].set_visible(False)
                        ax[i].spines['top'].set_visible(False)
                        ax[i].spines['right'].set_visible(False)

                    fig.align_ylabels()
                    # fig.tight_layout()
                    st.pyplot(fig)
                elif parameter == 6:
                    fig, ax = plt.subplots(1, 6, figsize=(6, 4), sharey='row')

                    n_classes = data['L_i'].shape[0]
                    ax[0].plot(data['L_i'], data['n'][time_point,:n_classes])
                    ax[1].plot(data['L_i'], data['n'][time_point,n_classes:n_classes*2])
                    ax[2].plot(data['L_i'], data['n'][time_point,n_classes*2:n_classes*3])
                    ax[3].plot(data['L_i'], data['n'][time_point,n_classes*3:n_classes*4])
                    ax[4].plot(data['L_i'], data['n'][time_point,n_classes*4:n_classes*5])
                    ax[5].plot(data['L_i'], data['n'][time_point,n_classes*5:])
                    ax[0].set_xlim([0, 0.01])
                    ax[0].set_ylabel('n_L [-]')
                    ax[0].set_xlabel('L [m]')
                    ax[1].set_xlabel('L [m]')
                    ax[2].set_xlabel('L [m]')
                    ax[3].set_xlabel('L [m]')
                    ax[4].set_xlabel('L [m]')
                    ax[5].set_xlabel('L [m]')

                    for i in range(6):
                        ax[i].spines['top'].set_visible(False)
                        ax[i].spines['right'].set_visible(False)
                        ax[i].spines['top'].set_visible(False)
                        ax[i].spines['right'].set_visible(False)

                    fig.align_ylabels()
                    # fig.tight_layout()
                    st.pyplot(fig)
        elif case_select == 'Tubular':
            # load tubular data
            with open(f'data/fulldata/data_Tubular_{parameter}_finite_volumes_DPBE.pkl', 'rb') as f:
                data = pickle.load(f)

            # split in two
            subcol1, subcol2 = st.columns([1, 4])
            with subcol2:
                # Picture
                st.image("figures/tubular.jpg", caption="Tubular crystallizer sketch", width=600)


            # split in two
            subcol1, subcol2 = st.columns([1, 4])
            with subcol1:
                st.write("Model inputs:")
                st.metric(label='T_jin', value=data['T_j_in'][time_point])
                st.metric(label='F_j,in', value=data['F_j'][time_point])
                st.metric(label='F_feed', value=data['F'][time_point])
                st.metric(label='T_feed', value=data['T_in'][time_point])
            with subcol2:
                # fig, ax = plt.subplots(3, 1, figsize=[6, 6], sharex='col')
                #
                # # Plot the data
                # ax[0].plot(data['L'], data['c'][time_point])
                # ax[0].set_ylabel('c [g/g]')
                # ax[0].set_ylim([data['c'].min() * 0.98, data['c'].max() * 1.02])
                # ax[1].plot(data['L'], data['T_PM'][time_point])
                # ax[1].set_ylabel('$T_{\\text{PM}} [K]$')
                # ax[1].set_ylim([data['T_PM'].min() * 0.98, data['T_PM'].max() * 1.02])
                # ax[2].plot(data['L'], data['T_TM'][time_point])
                # ax[2].set_ylabel('$T_{\\text{TM}} [K]$')
                # ax[2].set_ylim([data['T_TM'].min() * 0.98, data['T_TM'].max() * 1.02])
                # ax[2].set_xlabel('Length [m]')
                #
                # fig.align_ylabels()
                # # fig.tight_layout()
                until = 155

                # convert to data['n'][time_point].reshape((-1,200))[:,:until] to dict
                data_dict = {f'data{i}': data['n'][time_point].reshape((-1,200))[i,:until] for i in range(data['n'][time_point].reshape((-1,200)).shape[0])}

                df = pd.DataFrame(data_dict)
                data_array = data['n'][time_point].reshape((-1,200))[:,:until]

                # normalize for each entry in data_array the max value to 1
                data_array = data_array / np.max(data_array, axis=1)[:, np.newaxis]

                # convert to list
                data_list = data_array.tolist()

                custom_labels = np.round(data['L'], 1)

                fig, ax = horizontal_ridgeline_plot(
                    data=data_array,  # Can use data_array directly
                    x_range=data['L_i'][:until],
                    fill=True,
                    overlap=2,  # Adjust for better spacing
                    colormap='viridis',
                    ylabel="L_i [mu m]",  # This will be the y-axis label (domain of your distributions)
                    xlabel='Length [m]',
                    grid=True,
                    labels=custom_labels
                )


                # fig, ax = plt.subplots()
                # ax.plot(data['L_i'][:until], (data['n'][time_point].reshape((-1,200))[:,:until]).tolist()[0])
                st.pyplot(fig)




        if case_select == 'MSMPR' or case_select == 'MSMPR Cascade':
            # plt color cycle
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

            fig, ax = plt.subplots(3, 1, figsize=(6, 3), sharex='col')

            # Plot the data
            if case_select == 'MSMPR':
                ax[0].plot(data['c'])
                ax[0].set_ylabel('c [g/g]')
                ax[1].plot(data['T_PM'])
                ax[1].set_ylabel('T_PM [K]')
                ax[2].plot(data['T_TM'])
                ax[2].set_ylabel('T_TM [K]')
                # display current values as x
                ax[0].scatter(time_point, data['c'][time_point], color=colors[0], label='Time point', s=50)
                ax[1].scatter(time_point, data['T_PM'][time_point], color=colors[0], label='Time point', s=50)
                ax[2].scatter(time_point, data['T_TM'][time_point], color=colors[0], label='Time point', s=50)

            elif case_select == 'MSMPR Cascade':
                ax[0].plot(data['c'])
                ax[0].set_ylabel('c [g/g]')
                ax[1].plot(data['T_PM'])
                ax[1].set_ylabel('T_PM [K]')
                ax[2].plot(data['T_TM'])
                ax[2].set_ylabel('T_TM [K]')
                # display current values as x
                for i in range(parameter):
                    ax[0].scatter(time_point, data['c'][time_point][i], color=colors[i], label='Time point', s=20)
                    ax[1].scatter(time_point, data['T_PM'][time_point][i], color=colors[i], label='Time point', s=20)
                    ax[2].scatter(time_point, data['T_TM'][time_point][i], color=colors[i], label='Time point', s=20)

            ax[2].set_xlabel('Time [s]')

            # fig.tight_layout()
            fig.align_ylabels()

            st.pyplot(fig)


    # Handle counter increment logic
    if st.session_state.tab3_counter_active:  # Changed to tab3_counter_active
        # Update the counter value
        st.session_state.tab3_current_time = min(  # Changed to tab3_current_time
            400,  # max_value
            int(st.session_state.tab3_current_time + 5)  # Changed to tab3_current_time
        )
        # If we've reached the max value, loop back to the beginning
        if st.session_state.tab3_current_time >= 400:  # Changed to tab3_current_time
            st.session_state.tab3_current_time = 0  # Changed to tab3_current_time

        # Delay based on animation speed
        time.sleep(0.1)
        # Rerun the app to update the UI
        st.rerun()
