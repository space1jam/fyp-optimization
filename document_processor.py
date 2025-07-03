import math
import pandas as pd
import statistics
from typing import Dict, Any
import numpy as np
import logging
from pathlib import Path

# Configure logging
def setup_logger(log_file: str = "parameter_processor.log") -> logging.Logger:
    """Set up a logger with both file and console output."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # Capture all levels
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Only show info+ in console
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logger()

def get_parameters(file: str, num: int = 0) -> Dict[str, Any]:
    """
    Process a transportation problem data file and extract parameters.
    
    Args:
        file: Path to input text file with problem data
        num: Number of dummy stations to add (default 0)
        
    Returns:
        Dictionary containing all problem parameters.
    """
    logger.info(f"Starting processing for file: {file}")
    
    try:
        # ===== File Reading =====
        if not Path(file).exists():
            raise FileNotFoundError(f"File {file} does not exist")
        
        logger.debug("Reading data file with pandas")
        df = pd.read_csv(file, sep=r'\s+')
        df_filtered = df.iloc[:-5]  # Assuming last 5 rows are metadata
        original_data = df_filtered.to_numpy()
        logger.debug(f"Loaded {len(original_data)} rows of data")
        
        # ===== Parameter Extraction =====
        params = {}
        required_params = {
            'Q': 'Vehicle fuel tank capacity',
            'C': 'Vehicle load capacity',
            'g': 'inverse refueling rate',
            'r': 'fuel consumption rate',
            'v': 'average Velocity'
        }
        
        logger.debug("Scanning file for parameters")
        with open(file, 'r') as f:
            for line in f:
                for param, keyword in required_params.items():
                    if keyword in line:
                        params[param] = float(line.split('/')[1])
                        logger.debug(f"Found {param} = {params[param]}")
                        break
        
        # Validate parameters
        missing = [p for p in required_params if p not in params]
        if missing:
            logger.error(f"Missing parameters: {missing}")
            raise ValueError(f"Missing required parameters: {missing}")
        
        # ===== Data Processing =====
        logger.debug("Processing nodes and creating depot copy")
        depot_copy = original_data[0].copy()
        depot_copy[0] = "D0_end"
        
        # Station handling
        is_station = df_filtered["Type"] == 'f'
        station_count = np.sum(is_station)
        stations_copy = original_data[1:station_count+1].copy()
        original_stations = [str(row[0]) for row in stations_copy]
        logger.info(f"Found {station_count} original stations")
        
        # Dummy stations
        if num > 0:
            logger.debug(f"Generating {num} dummy stations")
            replicate_stations = np.tile(stations_copy, (num, 1))
            for count, row in enumerate(replicate_stations):
                row[0] = f"S_dummy{count}"
            final_data = np.vstack((original_data, depot_copy, replicate_stations))
        else:
            final_data = np.vstack((original_data, depot_copy))
        
        # ===== Node Categorization =====
        logger.debug("Categorizing nodes")
        clients = [str(row[0]) for row in final_data if str(row[1]).lower() == 'c']
        stations = [str(row[0]) for row in final_data if str(row[1]).lower() == 'f']
        all_nodes = [str(row[0]) for row in final_data]
        
        logger.info(f"Identified: {len(clients)} clients, {len(stations)} stations, "
                    f"{len(all_nodes)} total nodes")
        
        # ===== Attribute Processing =====
        attributes = {}
        attribute_fields = ['locations', 'demand', 'ready_time', 'due_date', 'service_time']
        
        for attr in attribute_fields:
            attributes[attr] = {}
        
        for row in final_data:
            node = str(row[0])
            try:
                attributes['locations'][node] = (float(row[2]), float(row[3]))
                attributes['demand'][node] = float(row[4])
                attributes['ready_time'][node] = float(row[5])
                attributes['due_date'][node] = float(row[6])
                attributes['service_time'][node] = float(row[7])
            except (ValueError, IndexError) as e:
                logger.warning(f"Error processing node {node}: {str(e)}")
                continue
        
        # ===== Distance Calculations =====
        logger.debug("Calculating distance and time matrices")
        arcs, times = {}, {}
        v = params['v']
        
        nodes = list(attributes['locations'].keys())
        total_pairs = len(nodes) ** 2
        logger.debug(f"Computing {total_pairs} node pairs")
        
        for i in nodes:
            for j in nodes:
                dx = attributes['locations'][i][0] - attributes['locations'][j][0]
                dy = attributes['locations'][i][1] - attributes['locations'][j][1]
                distance = math.sqrt(dx**2 + dy**2)
                arcs[(i, j)] = distance
                times[(i, j)] = distance / v
        
        # ===== Statistics =====
        logger.debug("Calculating time statistics")
        travel_time_series = [attributes['due_date'][c] - attributes['ready_time'][c] 
                            for c in clients]
        
        if len(travel_time_series) > 1:
            std = statistics.stdev(travel_time_series)
        else:
            std = 0
            logger.warning("Only one client found - standard deviation set to 0")
        
        mean = statistics.mean(travel_time_series)
        logger.info(f"Time statistics - Mean: {mean:.2f}, Std: {std:.2f}")
        
        # ===== Final Compilation =====
        parameters = {
            **params,
            "clients": clients,
            "stations": stations,
            "all_nodes": all_nodes,
            "depot_start": ["D0"],
            "depot_end": ["D0_end"],
            **attributes,
            "arcs": arcs,
            "times": times,
            "final_data": final_data,
            "original_stations": original_stations,
            "std": std,
            "mean": mean,
            "time_series": travel_time_series,
            "normal_times": times
        }
        
        logger.info("Successfully processed all parameters")
        return parameters
        
    except Exception as e:
        logger.critical(f"Critical error processing {file}: {str(e)}", exc_info=True)
        raise ValueError(f"Error processing file {file}: {str(e)}") from e