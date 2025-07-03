import streamlit as st
import pandas as pd
from pathlib import Path
from document_processor import get_parameters
from initialize import Heuristic
from visualizations import (
    plot_nodes_map, plot_solution_step_by_step
)
import hashlib
from simulated_annealing import SimulatedAnnealingOptimizer
import os
from document_processor_bot import DocumentProcessor
from tabu_search import TabuSearchOptimizer

# Enhanced cache management
@st.cache_data(ttl=1, show_spinner=False)
def load_params_from_path(path):
    return get_parameters(path)

@st.cache_data(ttl=1, show_spinner=False)
def plot_nodes_map_cached(nodes_df):
    return plot_nodes_map(nodes_df)

@st.cache_data(ttl=1, show_spinner=False)
def plot_solution_step_cached(nodes_df, routes, step, **kwargs):
    return plot_solution_step_by_step(nodes_df, routes, step, **kwargs)

# Nuclear reset function
def full_reset():
    st.cache_data.clear()
    st.session_state.clear()
    st.rerun()

# App title and layout
st.set_page_config(page_title="EVRP-TW-PR Optimizer", layout="wide")
st.title("EVRP-TW-PR Optimizer")

# Sidebar - file uploader and action buttons
st.sidebar.header("Upload Instance File")

# File uploader with cache busting
uploaded_file = st.sidebar.file_uploader(
    "Choose a .txt file", 
    type="txt",
    key=f"file_uploader_{st.session_state.get('upload_counter', 0)}"
)

if uploaded_file:
    # Create unique temp file name using file content hash
    file_content = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_content).hexdigest()
    temp_filename = f"temp_instance_{file_hash}.txt"
    
    with open(temp_filename, "wb") as f:
        f.write(file_content)
    instance_path = Path(temp_filename)

    if st.sidebar.button("Load Instance"):
        try:
            # Clear previous data and caches
            st.cache_data.clear()
            for key in ['solver', 'routes', 'time_dict', 'energy_dict', 'distance_dict', 'charge_dict']:
                st.session_state.pop(key, None)
                
            st.session_state['params'] = load_params_from_path(str(instance_path))
            st.session_state['loaded'] = True
            st.session_state['file_hash'] = file_hash
            st.success("Instance file loaded successfully!")
        except Exception as e:
            st.error(f"Error loading instance: {str(e)}")

    if st.sidebar.button("Initialize Solution"):
        if st.session_state.get('loaded'):
            try:
                solver_params = st.session_state['params'].copy()
                solver_params['h'] = solver_params['r']
                solver = Heuristic(solver_params)
                routes = solver.initial_solution()
                # routes = solver.random_feasible_initial_solution()

                if hasattr(solver, 'helper') and hasattr(solver.helper, 'prepare_visualization_data'):
                    time_dict, energy_dict, distance_dict, charge_dict = solver.helper.prepare_visualization_data(routes)
                    st.session_state.update({
                        'solver': solver,
                        'routes': routes,
                        'time_dict': time_dict,
                        'energy_dict': energy_dict,
                        'distance_dict': distance_dict,
                        'charge_dict': charge_dict
                    })
                else:
                    st.session_state.update({
                        'solver': solver,
                        'routes': routes
                    })
                    
                st.success("Feasible solution initialized!")
            except Exception as e:
                st.error(f"Error initializing solution: {str(e)}")

if st.sidebar.button("Reset System", type="primary"):
    full_reset()

# Main pages
tabs = st.tabs(["1. Instance Viewer", "2. Solution Viewer", "3. Optimization", "4. Q&A Chatbot"])

# Tab 1: Instance Viewer
with tabs[0]:
    if not st.session_state.get('loaded'):
        st.warning("Please load an instance file first")
        st.stop()
        
    file_display_name = uploaded_file.name if uploaded_file else "No file loaded"
    st.header(f"EVRP-TW-PR : Loaded File - {file_display_name}")

    # Create nodes DataFrame
    nodes_data = []
    params = st.session_state['params']
    
    for node in params['all_nodes']:
        nodes_data.append({
            'id': node,
            'type': node[0],
            'x': params['locations'][node][0],
            'y': params['locations'][node][1],
            'demand': params['demand'].get(node, 0),
            'ready_time': params['ready_time'].get(node, 0),
            'due_date': params['due_date'].get(node, 0),
            'service_time': params['service_time'].get(node, 0)
        })
    
    nodes_df = pd.DataFrame(nodes_data)
    
    st.subheader("Node Information")
    st.dataframe(nodes_df)

    vehicle_df = pd.DataFrame({
        'Battery Capacity (Q)': [params['Q']],
        'Load Capacity (C)': [params['C']],
        'Inverse Refueling Rate (g)': [params['g']],
        'Energy Consumption Rate (r)': [params['r']],
        'Average Velocity (v)': [params['v']]
    })
    st.subheader("Vehicle Constraints")
    st.dataframe(vehicle_df)

    st.subheader("Node Map")
    plot_nodes_map_cached(nodes_df)

# Tab 2: Solution Viewer
with tabs[1]:
    if not st.session_state.get('routes'):
        st.warning("Please initialize a solution first")
        st.stop()
        
    routes = st.session_state['routes']
    solver = st.session_state['solver']
    
    # Recreate nodes DataFrame
    nodes_data = []
    params = st.session_state['params']
    for node in params['all_nodes']:
        nodes_data.append({
            'id': node,
            'type': node[0],
            'x': params['locations'][node][0],
            'y': params['locations'][node][1],
            'demand': params['demand'].get(node, 0),
            'ready_time': params['ready_time'].get(node, 0),
            'due_date': params['due_date'].get(node, 0),
            'service_time': params['service_time'].get(node, 0)
        })
    nodes_df = pd.DataFrame(nodes_data)

    try:
        checker = solver.checker
        total_cost = sum(
            checker.total_cost(r) for r in routes
        )
        total_dist = sum(solver.helper.distance_one_route(r) for r in routes)
        total_energy = sum(solver.helper.energy_one_route(r) for r in routes)
        total_time = max(solver.helper.time_one_route(r) for r in routes)
        vehicles_used = len(routes)
        
        metrics_data = {
            'Metric': ['Vehicles Used', 'Total Distance', 'Total Energy', 'Makespan (Max End Time)', 'Total Cost (Weighted)'],
            'Value': [vehicles_used, total_dist, total_energy, total_time, total_cost]
        }
        
        st.subheader("Solution Metrics")
        st.dataframe(pd.DataFrame(metrics_data))

    except Exception as e:
        st.error(f"Error calculating metrics: {str(e)}")
        st.subheader("Routes")
        for i, route in enumerate(routes, 1):
            st.write(f"Vehicle {i}: {' → '.join(route)}")
        st.warning("Could not calculate all metrics due to missing methods")

    # Step-by-step route visualization
    st.subheader("Step-by-Step Route Visualization")
    step = st.slider("Select Step", 0, sum(len(r) - 1 for r in routes), 0, 1)
    
    try:
        time_dict = st.session_state.get('time_dict')
        energy_dict = st.session_state.get('energy_dict')
        distance_dict = st.session_state.get('distance_dict')
        charge_dict = st.session_state.get('charge_dict')
        
        if None in [time_dict, energy_dict, distance_dict, charge_dict]:
            if hasattr(solver, 'helper') and hasattr(solver.helper, 'prepare_visualization_data'):
                time_dict, energy_dict, distance_dict, charge_dict = solver.helper.prepare_visualization_data(routes)
                st.session_state.update({
                    'time_dict': time_dict,
                    'energy_dict': energy_dict,
                    'distance_dict': distance_dict,
                    'charge_dict': charge_dict
                })

        plot_solution_step_cached(
            nodes_df, routes, step,
            time_dict=time_dict,
            energy_dict=energy_dict,
            distance_dict=distance_dict,
            charge_dict=charge_dict
        )

    except Exception as e:
        st.error(f"Error in route visualization: {str(e)}")
        plot_solution_step_cached(nodes_df, routes, step)

     # Route details
    st.subheader("Route Details")
    for i, route in enumerate(routes, 1):
        st.write(f"**Vehicle {i} Route:** {' → '.join(route)}")

# Tab 3: Optimization (Final Debugged Version)
with tabs[2]:
    # Initialization check (keep this the same)
    if 'routes' not in st.session_state:
        st.warning("Please initialize a solution in Tab 2 first.")
        st.stop()

    st.header("Optimization")  # Changed to generic title
    
    # Initialize session state (keep this similar but adjust for both methods)
    if 'optimized_routes' not in st.session_state:
        st.session_state.optimized_routes = [list(route) for route in st.session_state.routes]
    
    if 'optimized_cost' not in st.session_state:
        try:
            st.session_state.optimized_cost = sum(
                st.session_state.solver.checker.total_cost(r) 
                for r in st.session_state.routes
            )
        except Exception as e:
            st.error(f"Error calculating initial cost: {str(e)}")
            st.session_state.optimized_cost = float('inf')

    # Algorithm selection
    algorithm = st.selectbox("Select Optimization Algorithm", 
                           ["Simulated Annealing", "Tabu Search", "Particle Swarm Optimization (PSO)", "Large Neighborhood Search (LNS)"],
                           index=0)

    # Algorithm-specific parameters
    with st.expander(f"{algorithm} Parameters"):
        if algorithm == "Simulated Annealing":
            col1, col2, col3 = st.columns(3)
            with col1:
                initial_temp = st.number_input("Initial Temperature", value=1000.0, min_value=1.0)
            with col2:
                cooling_rate = st.number_input("Cooling Rate", value=0.995, min_value=0.9, max_value=0.999, step=0.001)
            with col3:
                max_iterations = st.number_input("Max Iterations", value=500, min_value=10)
        elif algorithm == "Tabu Search":
            col1, col2 = st.columns(2)
            with col1:
                max_iterations = st.number_input("Max Iterations", value=500, min_value=10)
            with col2:
                tabu_tenure = st.number_input("Tabu Tenure", value=20, min_value=5, max_value=100)
        elif algorithm == "Particle Swarm Optimization (PSO)":
            col1, col2 = st.columns(2)
            with col1:
                num_particles = st.number_input("Number of Particles", value=20, min_value=2)
            with col2:
                max_iterations = st.number_input("Max Iterations", value=200, min_value=10)
        else:  # LNS
            col1, col2, col3 = st.columns(3)
            with col1:
                max_iterations = st.number_input("Max Iterations", value=200, min_value=10)
            with col2:
                ruin_fraction = st.number_input("Ruin Fraction (0-1)", value=0.3, min_value=0.05, max_value=0.9, step=0.05, format="%.2f")
            with col3:
                accept_worse_prob = st.number_input("Accept Worse Probability", value=0.05, min_value=0.0, max_value=1.0, step=0.01, format="%.2f")

    # Optimization button - modified for all methods
    if st.button(f"Run {algorithm}"):
        with st.spinner(f"Optimizing with {algorithm}..."):
            try:
                def clean_node(node):
                    while isinstance(node, list) and len(node) == 1:
                        node = node[0]
                    return str(node).strip()
                def flatten_node(node):
                    while isinstance(node, list) and len(node) == 1:
                        node = node[0]
                    return str(node).strip() 
                initial_routes = []
                depot_start = flatten_node(st.session_state.params['depot_start'])
                depot_end = flatten_node(st.session_state.params['depot_end'])
                for i, route in enumerate(st.session_state.routes):
                    try:
                        cleaned = [flatten_node(n) for n in route if flatten_node(n)]
                        if cleaned[0] != depot_start:
                            cleaned.insert(0, depot_start)
                        if cleaned[-1] != depot_end:
                            cleaned.append(depot_end)
                        initial_routes.append(cleaned)
                    except Exception as e:
                        st.error(f"Error cleaning route {i}: {str(e)}")
                        initial_routes.append([depot_start, depot_end])
                if 'h' not in st.session_state.params:
                    st.session_state.params['h'] = 1.0
                st.subheader("Debug: Initial Routes Passed to Optimizer")
                for idx, route in enumerate(initial_routes):
                    st.write(f"Route {idx+1}: {route}")
                st.session_state.params['initial_routes'] = initial_routes
                if algorithm == "Simulated Annealing":
                    optimizer = SimulatedAnnealingOptimizer(
                        parameters=st.session_state.params,
                        initial_routes=initial_routes,
                        initial_temperature=initial_temp,   
                        cooling_rate=cooling_rate,
                        max_iterations=max_iterations
                    )
                    optimized_routes, optimized_cost = optimizer.optimize_with_diagnosis()
                elif algorithm == "Tabu Search":
                    optimizer = TabuSearchOptimizer(
                        parameters=st.session_state.params,
                        initial_routes=initial_routes,
                        max_iterations=max_iterations,
                        tabu_tenure=tabu_tenure
                    )
                    optimized_routes, optimized_cost = optimizer.optimize()
                    optimizer.print_stats()
                elif algorithm == "Particle Swarm Optimization (PSO)":
                    from pso import PSOOptimizer
                    optimizer = PSOOptimizer(
                        st.session_state.params,
                        num_particles=int(num_particles),
                        max_iterations=int(max_iterations)
                    )
                    optimized_routes, optimized_cost = optimizer.optimize()
                else:  # LNS
                    from lns import LNSOptimizer
                    optimizer = LNSOptimizer(
                        st.session_state.params,
                        max_iterations=int(max_iterations),
                        ruin_fraction=float(ruin_fraction),
                        accept_worse_prob=float(accept_worse_prob)
                    )
                    optimized_routes, optimized_cost = optimizer.optimize(initial_routes=initial_routes)
                def clean_route_output(route):
                    return [clean_node(n) for n in route if clean_node(n)]
                final_routes = [clean_route_output(route) for route in optimized_routes]
                final_routes = []
                for route in optimized_routes:
                    cleaned = [flatten_node(n) for n in route if flatten_node(n)]
                    if cleaned[0] != depot_start:
                        cleaned.insert(0, depot_start)
                    if cleaned[-1] != depot_end:
                        cleaned.append(depot_end)
                    final_routes.append(cleaned)
                st.session_state.optimized_routes = final_routes
                st.session_state.optimized_cost = optimized_cost
                try:
                    initial_cost = sum(
                        st.session_state.solver.checker.total_cost(r) 
                        for r in st.session_state.routes
                    )
                    st.success(f"""
                    **Optimization Complete**  
                    - Initial Cost: {initial_cost:.2f}  
                    - Optimized Cost: {optimized_cost:.2f}  
                    - Improvement: {initial_cost - optimized_cost:.2f}
                    """)
                    st.subheader("Route Changes")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Before Optimization**")
                        for i, route in enumerate(st.session_state.routes, 1):
                            st.write(f"Vehicle {i}: {' → '.join(str(n) for n in route)}")
                    with col2:
                        st.write("**After Optimization**")
                        for i, route in enumerate(final_routes, 1):
                            st.write(f"Vehicle {i}: {' → '.join(str(n) for n in route)}")
                except Exception as e:
                    st.error(f"Error displaying results: {str(e)}")
                print("Raw optimized routes:", optimized_routes)
                print("Cleaned routes:", final_routes)
                print("=== COST FUNCTION DETERMINISM TEST ===")
                for i, r in enumerate(optimized_routes):
                    c1 = solver.checker.total_cost(r)
                    c2 = solver.checker.total_cost(r)
                    print(f"Route {i+1}: {r}")
                    print(f"  Cost 1: {c1}")
                    print(f"  Cost 2: {c2}")
                    if c1 != c2:
                        print("  WARNING: Cost function is non-deterministic for this route!")
            except Exception as e:
                st.error(f"Optimization failed: {str(e)}")
                st.session_state.optimized_routes = [list(route) for route in st.session_state.routes]
                try:
                    st.session_state.optimized_cost = sum(
                        st.session_state.solver.checker.total_cost(r) 
                        for r in st.session_state.routes
                    )
                except:
                    st.session_state.optimized_cost = float('inf')

    # Display optimized solution (rest of your existing code remains the same)
    routes = st.session_state.optimized_routes
    solver = st.session_state.solver
    
    # Create nodes DataFrame with error handling
    try:
        nodes_data = []
        params = st.session_state.params
        for node in params['all_nodes']:
            nodes_data.append({
                'id': str(node),
                'type': str(node[0]),
                'x': float(params['locations'][node][0]),
                'y': float(params['locations'][node][1]),
                'demand': float(params['demand'].get(node, 0)),
                'ready_time': float(params['ready_time'].get(node, 0)),
                'due_date': float(params['due_date'].get(node, 0)),
                'service_time': float(params['service_time'].get(node, 0))
            })
        nodes_df = pd.DataFrame(nodes_data)
    except Exception as e:
        st.error(f"Error creating nodes dataframe: {str(e)}")
        st.stop()

    # Metrics display with enhanced error handling
    try:
        try:
            total_cost = st.session_state.optimized_cost
            total_dist = sum(solver.helper.distance_one_route(r) for r in routes)
            total_energy = sum(solver.helper.energy_one_route(r) for r in routes)
            total_time = max(solver.helper.time_one_route(r) for r in routes)
            
            metrics_data = {
                'Metric': ['Vehicles Used', 'Total Distance', 'Total Energy', 'Makespan', 'Total Cost'],
                'Value': [len(routes), total_dist, total_energy, total_time, total_cost]
            }
            st.subheader("Optimized Metrics")
            st.dataframe(pd.DataFrame(metrics_data))

        except Exception as e:
            st.error(f"Error calculating metrics: {str(e)}")
            st.subheader("Basic Route Info")
            st.write(f"Number of vehicles: {len(routes)}")
            st.write(f"Total cost: {st.session_state.optimized_cost:.2f}")

    except Exception as e:
        st.error(f"Critical error in metrics display: {str(e)}")

    # Visualization with fallbacks
    try:
        st.subheader("Optimized Route Visualization")
        max_step = max(1, sum(len(r)-1 for r in routes))  # Ensure at least 1 step
        step = st.slider("View Step", 0, max_step, 0, key="optimized_viz_slider")
        
        try:
            # Try getting visualization data
            time_dict = st.session_state.get('time_dict', {})
            energy_dict = st.session_state.get('energy_dict', {})
            distance_dict = st.session_state.get('distance_dict', {})
            charge_dict = st.session_state.get('charge_dict', {})
            
            plot_solution_step_cached(
                nodes_df, 
                routes, 
                step,
                time_dict=time_dict,
                energy_dict=energy_dict,
                distance_dict=distance_dict,
                charge_dict=charge_dict
            )
        except Exception as e:
            st.error(f"Detailed visualization failed: {str(e)}")
            # Fallback to simple plot
            try:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots()
                for route in routes:
                    x = [nodes_df.loc[nodes_df['id'] == str(node), 'x'].values[0] for node in route]
                    y = [nodes_df.loc[nodes_df['id'] == str(node), 'y'].values[0] for node in route]
                    ax.plot(x, y, marker='o')
                st.pyplot(fig)
            except:
                st.warning("Could not generate any visualization")
                st.subheader("Route Details")
                for i, route in enumerate(routes, 1):
                    st.write(f"Vehicle {i}: {' → '.join(str(node) for node in route)}")
                    
    except Exception as e:
        st.error(f"Visualization setup failed: {str(e)}")
        st.subheader("Route Details")
        for i, route in enumerate(routes, 1):
            st.write(f"Vehicle {i}: {' → '.join(str(node) for node in route)}")

# Tab 4: Document Q&A Chatbot
with tabs[3]:
    from model_setup import Chatbot
    from document_processor_bot import DocumentProcessor
    from langchain.schema import HumanMessage

    st.header("💬 Document Q&A Chatbot")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "chatbot" not in st.session_state:
        chatbot = Chatbot()
        st.session_state["chatbot"] = chatbot
    else:
        chatbot = st.session_state["chatbot"]

    if "document_processor" not in st.session_state:
        document_processor = DocumentProcessor()
        st.session_state["document_processor"] = document_processor
    else:
        document_processor = st.session_state["document_processor"]

    with st.expander("Chatbot Settings"):
        st.header("Document Management")

        st.subheader("Process New Documents")
        doc_dir = st.text_input("Document Directory Path", value="./documents")
        store_name = st.text_input("Vector Store Name", value="my_vectorstore")

        if st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                try:
                    if not os.path.exists(doc_dir):
                        st.error(f"Directory '{doc_dir}' does not exist!")
                    else:
                        vectorstore_path = document_processor.process_directory(doc_dir, store_name)
                        st.success(f"Documents processed and stored as '{store_name}'")
                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")

        st.subheader("Select Vector Store")
        selected_store = None
        try:
            available_stores = document_processor.list_available_vectorstores()
            if available_stores:
                selected_store = st.selectbox("Available Vector Stores", options=available_stores)
                if st.button("Load Selected Vector Store"):
                    with st.spinner("Loading vector store..."):
                        store_path = os.path.join(document_processor.vector_db_path, selected_store)
                        chatbot.initialize_for_qa(store_path, search_type="similarity", k=5)
                        st.success(f"Vector store '{selected_store}' loaded successfully!")
                        st.session_state["messages"] = []
            else:
                st.info("No vector stores available. Process documents first.")
        except Exception as e:
            st.error(f"Error listing vector stores: {str(e)}")

        st.subheader("Self-Refinement Settings")
        chatbot.refinement_iterations = st.slider("Max Refinement Iterations", 1, 5, 3)
        chatbot.refinement_threshold = st.slider("Refinement Threshold", 0.1, 1.0, 0.8, step=0.1)

        st.subheader("Visualize Query Embedding")
        query_input = st.text_input("Query to Visualize", value="Write query here")
        if st.button("Visualize Embedding"):
            if selected_store and query_input.strip():
                with st.spinner("Generating UMAP projection..."):
                    try:
                        document_processor.visualize_query_projection(store_name=selected_store, query=query_input)
                        st.success("Visualization complete! Check the embedding_visualization folder.")
                    except Exception as e:
                        st.error(f"Visualization failed: {e}")
            else:
                st.warning("Please select a vector store and enter a query.")

        st.subheader("Debug Document Chunks")
        if st.button("Preview Split Chunks"):
            with st.spinner("Loading and splitting documents..."):
                try:
                    docs = document_processor.load_documents(doc_dir)
                    chunks = document_processor.split_documents(docs)
                    st.write(f"📄 Total Chunks: {len(chunks)}")
                    for i, chunk in enumerate(chunks[:5]):
                        st.markdown(f"**Chunk {i+1}**")
                        st.code(chunk.page_content[:500])
                        st.text(f"Metadata: {chunk.metadata}")
                except Exception as e:
                    st.error(f"Failed to preview chunks: {e}")

        st.subheader("Evaluation Settings")
        enable_evaluation = st.toggle("Show Evaluation Metrics", value=False)
        reference_answer = st.text_area("Reference Answer") if enable_evaluation else ""

    with st.container():
        for message in st.session_state.messages[:]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("role") == "assistant" and message.get("refinement_steps"):
                    with st.expander("View Refinement Process"):
                        for i, step in enumerate(message["refinement_steps"]):
                            st.markdown(f"**Step {i+1}**")
                            st.markdown(f"*Previous Answer:* {step.get('previous_answer', '')}")
                            st.markdown(f"*Feedback:* {step.get('feedback', '')}")
                            st.markdown(f"*Refined Answer:* {step.get('refined_answer', '')}")
                            st.divider()
                if message.get("formatted_sources") or message.get("sources"):
                    with st.expander("Sources"):
                        st.markdown(message.get("formatted_sources", ""))
                        st.write(message.get("sources", ""))

    if prompt := st.chat_input("Ask a question about your documents..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            try:
                with st.spinner("Generating response..."):
                    response = chatbot(prompt)
                    if hasattr(chatbot, 'last_refinement_steps') and chatbot.last_refinement_steps:
                        final = chatbot.last_refinement_steps[-1].get("refined_answer")
                        if final:
                            response = final
                    st.markdown(response)
                    msg_data = {
                        "role": "assistant",
                        "content": response,
                        "sources": chatbot.last_sources,
                        "formatted_sources": chatbot.formatted_sources,
                        "refinement_steps": getattr(chatbot, "last_refinement_steps", [])
                    }
                    st.session_state.messages.append(msg_data)

                    if enable_evaluation:
                        with st.expander("Evaluation Metrics"):
                            eval_args = {"query": prompt, "response": response}
                            if reference_answer.strip():
                                eval_args["reference_answer"] = reference_answer
                            evaluation = chatbot.evaluate_answer(**eval_args)
                            cols = st.columns(4)
                            cols[0].metric("Relevance", f"{evaluation['relevance']:.2f}")
                            cols[1].metric("Faithfulness", f"{evaluation['faithfulness']:.2f}")
                            cols[2].metric("Coherence", f"{evaluation['coherence']:.2f}")
                            cols[3].metric("Hallucination", f"{evaluation['hallucination_score']:.2f}")
                            if "similarity_to_reference" in evaluation:
                                st.metric("Similarity", f"{evaluation['similarity_to_reference']:.2f}")

            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
                st.session_state.messages.append({"role": "assistant", "content": str(e)})
