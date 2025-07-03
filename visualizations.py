import plotly.graph_objects as go
import streamlit as st
import pandas as pd

# Style configuration
NODE_STYLES = {
    'd': {"color": "blue", "symbol": "square", "label": "Depot", "size": 18},
    'c': {"color": "green", "symbol": "circle", "label": "Customer", "size": 12},
    'f': {"color": "orange", "symbol": "triangle-up", "label": "Station", "size": 14}
}

ROUTE_COLORS = [
    '#ff3333',  # Red
    '#ff8533',  # Orange
    '#ffe433',  # Yellow
    '#b4ff33',  # Lime
    '#33ffed',  # Teal
    '#3381ff',  # Blue
    '#8d33ff',  # Purple
    '#ff33da',  # Pink
    '#ffb0b0',  # Light red
    '#ffcfb0',  # Light orange
    '#faffb0',  # Light yellow
    '#b8ffb0',  # Light green
    '#b0fffb',  # Light teal
    '#b0c6ff',  # Light blue
    '#ffc7f0'   # Light pink
]

def normalize_node_types(nodes_df):
    """Map raw types to expected lowercase codes: D→d, S→f, C→c"""
    type_map = {'D': 'd', 'C': 'c', 'S': 'f'}
    nodes_df['type'] = nodes_df['type'].map(type_map).fillna(nodes_df['type'])
    return nodes_df


def plot_nodes_map(nodes_df):
    """Plot nodes with proper legend differentiation and legend below x-axis"""
    nodes_df = normalize_node_types(nodes_df)

    fig = go.Figure()

    for node_type, group in nodes_df.groupby('type'):
        style = NODE_STYLES.get(node_type.lower(), {})
        if not style:
            continue

        fig.add_trace(go.Scatter(
            x=group['x'],
            y=group['y'],
            mode='markers+text',
            marker=dict(
                size=style["size"],
                color=style["color"],
                symbol=style["symbol"],
                line=dict(width=2, color='black'),
                opacity=0.9
            ),
            text=group['id'],
            textposition="middle center",
            textfont=dict(size=10, color="white", family="Arial, sans-serif"),
            name=style["label"],
            legendgroup=style["label"],
            showlegend=True
        ))

    fig.update_layout(
        title="Node Map",
        title_x=0.5,
        xaxis_title="X Coordinate",
        yaxis_title="Y Coordinate",
        width=900,
        height=700,
        showlegend=True,
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
        margin=dict(l=50, r=50, t=50, b=80),
        hovermode='closest'
    )

    fig.update_traces(marker_showscale=False, selector=dict(type='scatter'))
    st.plotly_chart(fig, use_container_width=True)


def plot_solution_step_by_step(
    nodes_df,
    solution_routes,
    step,
    helper=None,
    time_dict=None,
    energy_dict=None,
    distance_dict=None,
    charge_dict=None,
    show_table=True
):
    """
    Enhanced visualization function with complete metrics display
    
    Parameters:
    - nodes_df: DataFrame with node coordinates and attributes
    - solution_routes: List of routes (each route is a list of nodes)
    - step: Current step to visualize (0 to max_steps)
    - helper: Helper class instance for calculating metrics
    - time_dict: Precomputed {(vehicle_idx, node): time}
    - energy_dict: Precomputed {(vehicle_idx, node): energy}
    - distance_dict: Precomputed {(vehicle_idx, node): distance}
    - charge_dict: Precomputed {(vehicle_idx, node): charge}
    - show_table: Whether to display the metrics table
    
    Returns:
    - DataFrame with step metadata
    """  
    nodes_df = normalize_node_types(nodes_df.copy())

    fig = go.Figure()
    metadata_records = []
    
    # 1. Plot all nodes with appropriate styling
    for node_type, group in nodes_df.groupby('type'):
        style = NODE_STYLES.get(node_type, {})
        fig.add_trace(go.Scatter(
            x=group['x'],
            y=group['y'],
            mode='markers+text',
            marker=dict(
                size=style.get("size", 12),
                color=style.get("color", "gray"),
                symbol=style.get("symbol", "circle"),
                line=dict(width=2, color='black')
            ),
            text=group['id'],
            textposition="middle center",
            textfont=dict(size=10, color="white"),
            name=style.get("label", "Node"),
            hoverinfo="text",
            hovertext=group.apply(
                lambda row: f"Node: {row['id']}<br>Type: {node_type}<br>Demand: {row.get('demand', 0)}",
                axis=1
            )
        ))
    
    # 2. Generate metrics dictionaries if helper is provided
    if helper and all(d is None for d in [time_dict, energy_dict, distance_dict, charge_dict]):
        time_dict, energy_dict, distance_dict, charge_dict = helper.prepare_visualization_data(solution_routes)
    
    # 3. Plot route segments up to the current step
    edge_count = 0
    for r_index, route in enumerate(solution_routes):
        color = ROUTE_COLORS[r_index % len(ROUTE_COLORS)]
        
        for i in range(1, len(route)):
            if edge_count >= step:
                break
                
            n1, n2 = route[i-1], route[i]
            
            try:
                node1 = nodes_df[nodes_df['id'] == n1].iloc[0]
                node2 = nodes_df[nodes_df['id'] == n2].iloc[0]
            except IndexError:
                continue
            
            # Add arrow annotation
            fig.add_annotation(
                x=node2['x'],
                y=node2['y'],
                ax=node1['x'],
                ay=node1['y'],
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.5,
                arrowwidth=2,
                arrowcolor=color,
                opacity=0.8
            )
            
            # Prepare metadata record with fallback values
            metrics = {
                "Vehicle": r_index+1,
                "From Node": n1,
                "To Node": n2,
                "Total Time (unit)": time_dict.get((r_index, n2), "N/A") if time_dict else "N/A",
                "Energy (unit)": energy_dict.get((r_index, n2), "N/A") if energy_dict else "N/A",
                "Cummulative Distance (unit)": distance_dict.get((r_index, n2), "N/A") if distance_dict else "N/A",
                "Energy Recharged (unit)": charge_dict.get((r_index, n2), 0) if charge_dict else 0
            }
            
            # Format numeric values
            for key in ["Total Time (unit)", "Energy (unit)", "Cummulative Distance (unit)", "Energy Recharged (unit)"]:
                if isinstance(metrics[key], (int, float)):
                    metrics[key] = f"{metrics[key]:.2f}"
            
            metadata_records.append(metrics)
            edge_count += 1
    
    # 4. Configure plot layout
    fig.update_layout(
        title=f"Solution Visualization (Step {step}/{sum(len(r)-1 for r in solution_routes)})",
        title_x=0.5,
        xaxis_title="X Coordinate",
        yaxis_title="Y Coordinate",
        width=900,
        height=700,
        showlegend=True,
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
        margin=dict(l=50, r=50, t=50, b=80),
        hovermode='closest'
    )

    # 5. Display the plot
    st.plotly_chart(fig, use_container_width=True)
    
    # 6. Prepare and return metadata
    metadata_df = pd.DataFrame(metadata_records)
    
    if show_table and not metadata_df.empty:
        st.subheader("Route Metrics")
        # Reorder columns for better presentation
        col_order = ["Vehicle", "From Node", "To Node", "Cummulative Distance (unit)", 
                    "Total Time (unit)", "Energy (unit)", "Energy Recharged (unit)"]
        metadata_df = metadata_df[col_order]
        st.dataframe(metadata_df)
    
    return metadata_df


def show_metrics(total_cost, total_dist, total_energy, total_time):
    """Display metrics with improved formatting"""
    metrics = {
        "Total Cost": f"{total_cost:,.2f}",
        "Total Distance": f"{total_dist:,.2f} km",
        "Total Energy": f"{total_energy:,.2f} kWh",
        "Total Time": f"{total_time:,.2f} min"
    }

    cols = st.columns(len(metrics))
    for col, (name, value) in zip(cols, metrics.items()):
        with col:
            st.metric(label=name, value=value)


def show_time_window_chart(node_ids, earliest_arrival, latest_arrival, actual_times):
    """Improved time window visualization"""
    df = pd.DataFrame({
        'Node ID': node_ids,
        'Earliest Time': earliest_arrival,
        'Latest Time': latest_arrival,
        'Arrival Time': actual_times
    })

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['Node ID'],
        y=df['Earliest Time'],
        mode='lines',
        line=dict(dash='dash', color='green'),
        name='Earliest Time'
    ))

    fig.add_trace(go.Scatter(
        x=df['Node ID'],
        y=df['Latest Time'],
        mode='lines',
        line=dict(dash='dash', color='red'),
        name='Latest Time'
    ))

    fig.add_trace(go.Scatter(
        x=df['Node ID'],
        y=df['Arrival Time'],
        mode='markers+lines',
        marker=dict(size=10, color='blue'),
        line=dict(color='blue', width=2),
        name='Actual Arrival'
    ))

    fig.update_layout(
        title='Time Window Compliance',
        title_x=0.5,
        xaxis_title='Node ID',
        yaxis_title='Time',
        width=900,
        height=500,
        legend=dict(x=0.8, y=1),
        margin=dict(l=50, r=50, t=50, b=50),
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)

