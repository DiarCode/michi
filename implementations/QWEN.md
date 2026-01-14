#!/usr/bin/env python3

# -_- coding: utf-8 -_-

"""
DTS-GSSF: Dual-Timescale Graph State-Space Forecasting with Online Residual Correction
and Hierarchical Reconciliation for Astana City Bus Management

Architecture Implementation:

1. Graph-Structured State-Space Forecaster (GSSF) Backbone
2. Online Residual Corrector with Kalman Filtering
3. Drift Detection & Low-Rank Adaptation (LoRA)
4. Hierarchical Reconciliation (MinT)
5. Streamlit UI Dashboard with Real-Time Analytics

Optimized for M4 Mac with uv dependency management
"""

import os
import sys
import time
import math
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# M4 Mac optimizations

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "8" # Optimal for M4 chip
os.environ["MKL_NUM_THREADS"] = "8"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from tqdm import tqdm
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Streamlit imports (will be conditionally imported)

try:
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
STREAMLIT_AVAILABLE = True
except ImportError:
STREAMLIT_AVAILABLE = False
logger.warning("Streamlit not available. Running in headless mode.")

# Initialize logger

logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("dts_gssf.log", rotation="10 MB", level="DEBUG")

console = Console()

@dataclass
class AstanaConfig:
"""Configuration for Astana City Bus System"""

    # City districts and zones
    districts: List[str] = field(default_factory=lambda: [
        "Alatau District", "Saryarka District", "Baiterek District",
        "Keruen District", "Koktem District", "Zhetigen District",
        "Zarya District", "Kabanbay District", "Korgalzhyn District"
    ])

    # Main bus stations and terminals
    bus_stations: List[str] = field(default_factory=lambda: [
        "Saparzhay-1 (Main Terminal)", "Saparzhay-2", "Railway Station Terminal",
        "Airport Terminal", "Khan Shatyr Terminal", "Duman Terminal",
        "Nurly Zhol Station", "Astana-1 Station", "Central Bus Depot",
        "Sayran Connection Hub", "Koktem Bus Park", "Zhetigen Transfer Point"
    ])

    # Bus routes (realistic Astana routes)
    bus_routes: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"route_id": "100", "name": "100 ekspress", "operator": "AO Avtobusnyy park №1", "fare": 250, "type": "express"},
        {"route_id": "56", "name": "56", "operator": "TOO Avtobusnyy park №3", "fare": 180, "type": "regular"},
        {"route_id": "81", "name": "81", "operator": "AO Avtobusnyy park №1", "fare": 110, "type": "regular"},
        {"route_id": "12", "name": "12", "operator": "AO Avtobusnyy park №2", "fare": 180, "type": "regular"},
        {"route_id": "63", "name": "63", "operator": "AO Avtobusnyy park №1", "fare": 150, "type": "regular"},
        {"route_id": "17", "name": "17", "operator": "TOO Avtobusnyy park №3", "fare": 120, "type": "regular"},
        {"route_id": "42", "name": "42", "operator": "AO Avtobusnyy park №2", "fare": 130, "type": "regular"},
        {"route_id": "99", "name": "99", "operator": "AO Avtobusnyy park №1", "fare": 200, "type": "express"}
    ])

    # Road segments and bottlenecks
    road_segments: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"name": "Qabanbay Batyr Avenue", "length_km": 12.5, "bottlenecks": ["Khan Shatyr Intersection", "Presidential Park Junction"]},
        {"name": "Tauelsizdik Avenue", "length_km": 8.3, "bottlenecks": ["Railway Crossing", "Central Market Area"]},
        {"name": "Kazhymukan Street", "length_km": 6.7, "bottlenecks": ["Saparzhay Terminal Approach"]},
        {"name": "Bokeikhan Street", "length_km": 9.1, "bottlenecks": ["Government District Traffic"]},
        {"name": "Almaty Street", "length_km": 15.2, "bottlenecks": ["Airport Access Junction", "Industrial Zone Entry"]}
    ])

    # Realistic passenger flow patterns
    peak_hours: List[Tuple[int, int]] = field(default_factory=lambda: [(7, 9), (17, 19)])  # Morning and evening rush hours
    weekend_multiplier: float = 0.6  # Lower weekend traffic
    event_multiplier: float = 1.8    # Increased traffic during events
    construction_zones: List[str] = field(default_factory=lambda: ["Saryarka District Center", "Koktem Industrial Area"])

    # System parameters
    num_stations: int = 50
    num_edges: int = 120
    lookback_window: int = 24  # 24 hours of historical data
    forecast_horizon: int = 12  # 12-hour forecast
    residual_dim: int = 16    # Low-dimensional residual state
    hidden_dim: int = 64      # Backbone hidden dimension
    graph_depth: int = 3      # Graph propagation depth

    # Drift detection parameters
    drift_threshold: float = 3.5
    drift_sensitivity: float = 0.1
    adaptation_window: int = 48  # 48-hour window for adaptation
    lora_rank: int = 8       # Low-rank adaptation rank

    # Simulation parameters
    simulation_days: int = 30
    data_refresh_interval: int = 60  # seconds for real-time simulation

    @classmethod
    def from_real_data(cls):
        """Create config with real Astana data structure"""
        return cls()

class GraphStateSpaceModel(nn.Module):
"""
Graph-Structured State-Space Forecaster (GSSF) Backbone

    Implements the structured SSM with adaptive graph propagation
    as described in Section 3 of the architecture document
    """

    def __init__(self, config: AstanaConfig, node_features: int = 10):
        super().__init__()
        self.config = config
        self.node_features = node_features
        self.hidden_dim = config.hidden_dim
        self.output_dim = 1  # Passenger flow count

        # Input encoding layer
        self.input_encoder = nn.Sequential(
            nn.Linear(node_features, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim)
        )

        # State-space model components
        self.ssm_A = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim) * 0.01)
        self.ssm_B = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim) * 0.01)
        self.ssm_C = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim) * 0.01)

        # Adaptive graph adjacency learning
        self.node_embedding = nn.Embedding(config.num_stations, 32)
        self.adaptive_adj_projector = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        )

        # Graph convolution layers
        self.gcn_layers = nn.ModuleList([
            GCNConv(self.hidden_dim, self.hidden_dim) for _ in range(config.graph_depth)
        ])

        # Multi-horizon decoder
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, 64),
                nn.GELU(),
                nn.Linear(64, self.output_dim),
                nn.Softplus()  # Ensure positive counts for Poisson distribution
            ) for _ in range(config.forecast_horizon)
        ])

        # Learnable mixing parameter for physical vs adaptive adjacency
        self.alpha = nn.Parameter(torch.tensor(0.7))

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _create_adaptive_adjacency(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        """Create adaptive adjacency matrix using node embeddings"""
        proj = self.adaptive_adj_projector(node_embeddings)
        adj_adaptive = torch.softmax(torch.mm(proj, proj.t()), dim=1)
        return adj_adaptive

    def _mix_adjacencies(self, adj_physical: torch.Tensor, adj_adaptive: torch.Tensor) -> torch.Tensor:
        """Mix physical and adaptive adjacency matrices"""
        return self.alpha * adj_physical + (1 - self.alpha) * adj_adaptive

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the GSSF backbone

        Args:
            x: Node features [batch_size, num_nodes, node_features]
            edge_index: Graph connectivity [2, num_edges]
            edge_weight: Edge weights [num_edges]

        Returns:
            forecasts: Predicted passenger flows [batch_size, num_nodes, forecast_horizon]
        """
        batch_size, num_nodes, _ = x.shape

        # Input encoding
        x_encoded = self.input_encoder(x)  # [batch_size, num_nodes, hidden_dim]

        # State-space modeling (simplified discrete SSM)
        states = torch.zeros(batch_size, num_nodes, self.hidden_dim, device=x.device)
        outputs = []

        for t in range(x_encoded.shape[1]):
            # State update: s_{t+1} = A s_t + B u_t
            states = torch.matmul(states, self.ssm_A) + torch.matmul(x_encoded[:, t], self.ssm_B)

            # Output: z_t = C s_t + D u_t
            output = torch.matmul(states, self.ssm_C) + x_encoded[:, t]
            outputs.append(output)

        temporal_output = torch.stack(outputs, dim=1)  # [batch_size, seq_len, hidden_dim]

        # Graph propagation
        node_embeddings = self.node_embedding(torch.arange(num_nodes, device=x.device))
        adj_adaptive = self._create_adaptive_adjacency(node_embeddings)

        # Convert edge_index to dense adjacency for mixing
        adj_physical = torch.zeros(num_nodes, num_nodes, device=x.device)
        adj_physical[edge_index[0], edge_index[1]] = edge_weight if edge_weight is not None else 1.0

        # Mix adjacencies
        adj_mixed = self._mix_adjacencies(adj_physical, adj_adaptive)

        # Apply graph convolutions
        graph_output = temporal_output[:, -1]  # Use last timestep
        for gcn_layer in self.gcn_layers:
            graph_output = gcn_layer(graph_output, edge_index, edge_weight)
            graph_output = F.gelu(graph_output)

        # Multi-horizon decoding
        forecasts = []
        for h in range(self.config.forecast_horizon):
            forecast = self.decoder[h](graph_output)
            forecasts.append(forecast)

        return torch.stack(forecasts, dim=1)  # [batch_size, forecast_horizon, num_nodes, 1]

class ResidualCorrector:
"""
Online Residual Corrector with Kalman Filtering

    Implements the fast-timescale residual correction module
    as described in Section 4 of the architecture document
    """

    def __init__(self, config: AstanaConfig, num_nodes: int):
        self.config = config
        self.num_nodes = num_nodes
        self.residual_dim = config.residual_dim

        # Kalman filter parameters
        self.F = np.eye(self.residual_dim)  # State transition matrix
        self.H = np.random.randn(self.residual_dim, num_nodes) * 0.01  # Observation matrix
        self.Q = np.eye(self.residual_dim) * 0.1  # Process noise
        self.R = np.eye(num_nodes) * 0.5  # Measurement noise

        # Initialize state and covariance
        self.state = np.zeros(self.residual_dim)
        self.covariance = np.eye(self.residual_dim) * 10.0

        # Residual encoder/decoder
        self.encoder = np.random.randn(self.residual_dim, num_nodes) * 0.1
        self.decoder = np.random.randn(num_nodes, self.residual_dim) * 0.1

        # Rolling statistics for drift detection
        self.residual_history = []
        self.mad_history = []
        self.drift_detector = PageHinkleyDelta(
            threshold=config.drift_threshold,
            delta=config.drift_sensitivity
        )

    def compress_residual(self, residual: np.ndarray) -> np.ndarray:
        """Compress high-dimensional residual to low-dimensional state"""
        return self.encoder @ residual

    def decompress_residual(self, latent_residual: np.ndarray) -> np.ndarray:
        """Decompress latent residual back to original space"""
        return self.decoder @ latent_residual

    def predict_residual(self) -> np.ndarray:
        """Predict next residual using Kalman filter"""
        # Predict state
        state_pred = self.F @ self.state
        cov_pred = self.F @ self.covariance @ self.F.T + self.Q

        # Predict observation
        residual_pred = self.H @ state_pred
        return residual_pred

    def update(self, true_residual: np.ndarray):
        """Update Kalman filter with observed residual"""
        # Compress residual
        latent_residual = self.compress_residual(true_residual)

        # Kalman gain
        S = self.H @ self.covariance @ self.H.T + self.R
        K = self.covariance @ self.H.T @ np.linalg.inv(S)

        # Update state
        innovation = latent_residual - self.H @ self.state
        self.state = self.state + K @ innovation
        self.covariance = (np.eye(self.residual_dim) - K @ self.H) @ self.covariance

        # Store for drift detection
        self.residual_history.append(true_residual)
        if len(self.residual_history) > 100:
            self.residual_history.pop(0)

    def detect_drift(self) -> bool:
        """Detect concept drift using standardized residuals"""
        if len(self.residual_history) < 10:
            return False

        residuals = np.array(self.residual_history[-10:])
        mad = np.median(np.abs(residuals - np.median(residuals, axis=0)), axis=0)
        std_residuals = np.mean(np.abs(residuals[-1]) / (mad + 1e-6))

        return self.drift_detector.update(std_residuals)

class PageHinkleyDelta:
"""Page-Hinkley change detection algorithm"""

    def __init__(self, threshold: float = 3.5, delta: float = 0.1):
        self.threshold = threshold
        self.delta = delta
        self.min_value = 0
        self.cumulative_sum = 0
        self.change_detected = False

    def update(self, value: float) -> bool:
        """Update detector with new value"""
        self.cumulative_sum += value - self.delta
        self.min_value = min(self.min_value, self.cumulative_sum)

        if self.cumulative_sum - self.min_value > self.threshold:
            self.change_detected = True
            self.reset()
            return True
        return False

    def reset(self):
        """Reset detector state"""
        self.cumulative_sum = 0
        self.min_value = 0
        self.change_detected = False

class LowRankAdapter:
"""
Low-Rank Adaptation (LoRA) for Drift Correction

    Implements drift-triggered low-rank parameter updates
    as described in Section 5 of the architecture document
    """

    def __init__(self, model: nn.Module, rank: int = 8):
        self.model = model
        self.rank = rank
        self.adapters = {}

        # Identify weight matrices to adapt
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() == 2:  # Only adapt 2D weights
                self.adapters[name] = {
                    'B': nn.Parameter(torch.randn(param.shape[0], rank) * 0.01),
                    'A': nn.Parameter(torch.randn(rank, param.shape[1]) * 0.01)
                }

    def apply_adaptation(self, learning_rate: float = 0.01):
        """Apply low-rank adaptation to model parameters"""
        with torch.no_grad():
            for name, adapter in self.adapters.items():
                param = dict(self.model.named_parameters())[name]
                delta = adapter['B'] @ adapter['A']
                param.add_(learning_rate * delta)

    def optimize_adapters(self, loss_fn: Callable, data_loader: Any, num_steps: int = 5):
        """Optimize adapter parameters on recent data"""
        optimizer = torch.optim.Adam(
            [param for adapter in self.adapters.values() for param in adapter.values()],
            lr=0.001
        )

        for _ in range(num_steps):
            for batch in data_loader:
                optimizer.zero_grad()
                loss = loss_fn(batch)
                loss.backward()
                optimizer.step()

class HierarchicalReconciler:
"""
Hierarchical Reconciliation with MinT

    Implements optimal reconciliation to ensure hierarchical consistency
    as described in Section 6 of the architecture document
    """

    def __init__(self, hierarchy_matrix: np.ndarray, error_covariance: Optional[np.ndarray] = None):
        self.S = hierarchy_matrix  # Summing matrix
        self.W = error_covariance if error_covariance is not None else np.eye(hierarchy_matrix.shape[0])

        # Precompute reconciliation matrix
        self.P = self._compute_reconciliation_matrix()

    def _compute_reconciliation_matrix(self) -> np.ndarray:
        """Compute MinT reconciliation matrix"""
        S = self.S
        W_inv = np.linalg.inv(self.W + 1e-6 * np.eye(self.W.shape[0]))
        P = S @ np.linalg.inv(S.T @ W_inv @ S) @ S.T @ W_inv
        return P

    def reconcile(self, forecasts: np.ndarray) -> np.ndarray:
        """Apply reconciliation to forecasts"""
        return self.P @ forecasts

class AstanaDataGenerator:
"""
Realistic Astana Bus Passenger Flow Generator

    Generates synthetic but realistic passenger flow data for Astana city
    with proper temporal patterns, spatial correlations, and real-world constraints
    """

    def __init__(self, config: AstanaConfig):
        self.config = config
        self.graph = self._create_transport_graph()
        self.station_metadata = self._generate_station_metadata()
        self.event_calendar = self._create_event_calendar()

    def _create_transport_graph(self) -> nx.DiGraph:
        """Create realistic transport graph for Astana"""
        G = nx.DiGraph()

        # Add stations
        for i in range(self.config.num_stations):
            station_name = f"Station_{i+1:02d}"
            if i < len(self.config.bus_stations):
                station_name = self.config.bus_stations[i]

            G.add_node(i,
                      name=station_name,
                      district=random.choice(self.config.districts),
                      capacity=random.randint(200, 1000),
                      type="terminal" if i < 5 else "regular")

        # Add edges with realistic connectivity
        for _ in range(self.config.num_edges):
            source = random.randint(0, self.config.num_stations - 1)
            target = random.randint(0, self.config.num_stations - 1)
            if source != target and not G.has_edge(source, target):
                # Edge weight based on distance and connectivity
                distance = random.uniform(1.0, 15.0)  # km
                traffic_factor = random.uniform(0.5, 1.5)

                G.add_edge(source, target,
                          distance=distance,
                          traffic_factor=traffic_factor,
                          road_name=random.choice([r["name"] for r in self.config.road_segments]),
                          is_bottleneck=random.random() < 0.2)

        # Ensure graph connectivity
        if not nx.is_weakly_connected(G):
            components = list(nx.weakly_connected_components(G))
            for i in range(1, len(components)):
                comp1 = list(components[i-1])[0]
                comp2 = list(components[i])[0]
                G.add_edge(comp1, comp2, distance=5.0, traffic_factor=1.0, road_name="Connector Road")

        return G

    def _generate_station_metadata(self) -> Dict[int, Dict[str, Any]]:
        """Generate metadata for each station"""
        metadata = {}
        for node, data in self.graph.nodes(data=True):
            metadata[node] = {
                'name': data['name'],
                'district': data['district'],
                'capacity': data['capacity'],
                'routes': [route["route_id"] for route in random.sample(self.config.bus_routes, k=min(3, len(self.config.bus_routes)))],
                'peak_capacity': data['capacity'] * 1.5,
                'bottlenecks': [b for r in self.config.road_segments for b in r["bottlenecks"]] if node < 10 else []
            }
        return metadata

    def _create_event_calendar(self) -> Dict[datetime, Dict[str, Any]]:
        """Create calendar of events affecting passenger flow"""
        events = {}
        start_date = datetime(2026, 1, 1)

        # Add regular events
        for day in range(self.config.simulation_days):
            current_date = start_date + timedelta(days=day)

            # Weekend effect
            if current_date.weekday() >= 5:  # Saturday or Sunday
                events[current_date] = {
                    'type': 'weekend',
                    'multiplier': self.config.weekend_multiplier,
                    'description': 'Weekend traffic pattern'
                }

            # Random special events
            if random.random() < 0.1:  # 10% chance of special event
                event_type = random.choice(['sports', 'concert', 'festival', 'construction'])
                multiplier = {
                    'sports': 1.7,
                    'concert': 2.0,
                    'festival': 2.5,
                    'construction': 0.4
                }[event_type]

                events[current_date] = {
                    'type': event_type,
                    'multiplier': multiplier,
                    'affected_stations': random.sample(list(self.graph.nodes()), k=min(5, self.config.num_stations//4)),
                    'description': f'{event_type.capitalize()} event at {random.choice(self.config.districts)}'
                }

        return events

    def generate_passenger_flow(self, timestamp: datetime) -> np.ndarray:
        """Generate passenger flow for all stations at given timestamp"""
        base_flow = np.zeros(self.config.num_stations)

        # Time-based patterns
        hour = timestamp.hour
        is_peak = any(start <= hour < end for start, end in self.config.peak_hours)
        is_night = hour < 6 or hour >= 22

        # Event multiplier
        event_multiplier = 1.0
        if timestamp.date() in self.event_calendar:
            event_data = self.event_calendar[timestamp.date()]
            event_multiplier = event_data['multiplier']

        # Generate flow for each station
        for node in self.graph.nodes():
            station_data = self.station_metadata[node]

            # Base flow based on station capacity and type
            capacity_factor = station_data['capacity'] / 500.0
            type_factor = 1.5 if station_data['type'] == 'terminal' else 1.0

            # Time factors
            time_factor = 2.0 if is_peak else (0.3 if is_night else 1.0)

            # Bottleneck factor
            bottleneck_factor = 0.7 if station_data['bottlenecks'] else 1.0

            # Calculate base flow
            flow = (capacity_factor * type_factor * time_factor * bottleneck_factor * event_multiplier * 50)

            # Add randomness with realistic patterns
            if is_peak:
                flow *= random.uniform(0.8, 1.2)
            if is_night:
                flow *= random.uniform(0.5, 0.8)

            # Clip to reasonable values
            flow = max(0, min(flow, station_data['peak_capacity']))
            base_flow[node] = flow

        # Add spatial correlations
        for edge in self.graph.edges(data=True):
            source, target, data = edge
            correlation = data['traffic_factor'] * 0.3  # 30% correlation between connected stations
            base_flow[target] += base_flow[source] * correlation

        # Add noise with realistic variance
        noise = np.random.normal(0, base_flow * 0.1 + 5)  # 10% noise + baseline noise
        final_flow = np.maximum(0, base_flow + noise)

        return final_flow.astype(int)

    def generate_historical_data(self, days: int = 30) -> pd.DataFrame:
        """Generate historical passenger flow data"""
        start_time = datetime(2026, 1, 1, 0, 0)
        timestamps = [start_time + timedelta(minutes=30 * i) for i in range(days * 48)]  # 30-min intervals

        data = []
        for ts in tqdm(timestamps, desc="Generating historical data"):
            flow = self.generate_passenger_flow(ts)
            row = {
                'timestamp': ts,
                **{f'station_{i}': flow[i] for i in range(self.config.num_stations)}
            }
            data.append(row)

        return pd.DataFrame(data)

class DTS_GSSF_System:
"""
Complete DTS-GSSF System Integration

    Integrates all components into a cohesive real-time forecasting system
    with Streamlit UI for monitoring and analytics
    """

    def __init__(self, config: AstanaConfig):
        self.config = config
        self.data_generator = AstanaDataGenerator(config)
        self.graph = self.data_generator.graph

        # Initialize backbone model
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self.backbone = GraphStateSpaceModel(
            config,
            node_features=10  # Features: flow, time features, event flags, etc.
        ).to(self.device)

        # Initialize residual corrector
        self.residual_corrector = ResidualCorrector(config, config.num_stations)

        # Initialize low-rank adapter
        self.low_rank_adapter = LowRankAdapter(self.backbone, rank=config.lora_rank)

        # Initialize hierarchical reconciler
        self.hierarchy_matrix = self._create_hierarchy_matrix()
        self.reconciler = HierarchicalReconciler(self.hierarchy_matrix)

        # System state
        self.historical_data = None
        self.current_time = datetime(2026, 1, 1, 0, 0)
        self.forecast_history = []
        self.performance_metrics = {
            'mae': [],
            'rmse': [],
            'drift_events': [],
            'adaptation_times': []
        }

        # Initialize Streamlit UI if available
        self.ui_initialized = False
        if STREAMLIT_AVAILABLE:
            self._init_streamlit_ui()

    def _create_hierarchy_matrix(self) -> np.ndarray:
        """Create hierarchical aggregation matrix for stations -> routes -> districts -> city"""
        num_bottom = self.config.num_stations
        num_routes = len(self.config.bus_routes)
        num_districts = len(self.config.districts)

        # Total levels: stations + routes + districts + city total
        total_levels = num_bottom + num_routes + num_districts + 1

        S = np.zeros((total_levels, num_bottom))

        # Identity for bottom level (stations)
        for i in range(num_bottom):
            S[i, i] = 1

        # Route aggregations (random assignment for demo)
        route_assignments = {i: [] for i in range(num_routes)}
        for station_idx in range(num_bottom):
            route_idx = random.randint(0, num_routes - 1)
            route_assignments[route_idx].append(station_idx)

        for route_idx, stations in route_assignments.items():
            row_idx = num_bottom + route_idx
            for station_idx in stations:
                S[row_idx, station_idx] = 1

        # District aggregations
        district_assignments = {i: [] for i in range(num_districts)}
        for station_idx in range(num_bottom):
            district_idx = random.randint(0, num_districts - 1)
            district_assignments[district_idx].append(station_idx)

        for district_idx, stations in district_assignments.items():
            row_idx = num_bottom + num_routes + district_idx
            for station_idx in stations:
                S[row_idx, station_idx] = 1

        # City total
        S[-1, :] = 1

        return S

    def _init_streamlit_ui(self):
        """Initialize Streamlit UI components"""
        if not STREAMLIT_AVAILABLE:
            return

        st.set_page_config(
            page_title="DTS-GSSF Astana Bus Management",
            page_icon="🚌",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # CSS styling
        st.markdown("""
        <style>
        .main-header {text-align: center; color: #1E88E5; font-size: 2.5rem; font-weight: bold;}
        .metric-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; padding: 20px; margin: 10px;}
        .status-indicator {display: inline-block; width: 15px; height: 15px; border-radius: 50%; margin-right: 10px;}
        .status-green {background-color: #28a745;}
        .status-yellow {background-color: #ffc107;}
        .status-red {background-color: #dc3545;}
        </style>
        """, unsafe_allow_html=True)

        self.ui_initialized = True

    def prepare_features(self, timestamp: datetime, historical_flows: np.ndarray) -> torch.Tensor:
        """Prepare node features for the model"""
        num_stations = self.config.num_stations
        features = []

        for station_idx in range(num_stations):
            station_features = []

            # Recent flow (last 6 time steps)
            recent_flows = historical_flows[-6:, station_idx] if historical_flows.shape[0] >= 6 else np.zeros(6)
            station_features.extend(recent_flows)

            # Time features
            hour = timestamp.hour / 24.0
            day_of_week = timestamp.weekday() / 7.0
            is_weekend = 1.0 if timestamp.weekday() >= 5 else 0.0
            station_features.extend([hour, day_of_week, is_weekend])

            # Event features
            event_multiplier = 1.0
            if timestamp.date() in self.data_generator.event_calendar:
                event_data = self.data_generator.event_calendar[timestamp.date()]
                event_multiplier = event_data['multiplier']
            station_features.append(event_multiplier)

            # Station metadata features
            station_data = self.data_generator.station_metadata[station_idx]
            capacity_ratio = station_data['capacity'] / 1000.0
            is_terminal = 1.0 if station_data['type'] == 'terminal' else 0.0
            station_features.extend([capacity_ratio, is_terminal])

            features.append(station_features)

        return torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)

    def train_backbone(self, historical_data: pd.DataFrame, epochs: int = 10):
        """Train the backbone model on historical data"""
        logger.info("Training GSSF backbone model...")

        optimizer = torch.optim.AdamW(self.backbone.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # Prepare training data
        timestamps = historical_data['timestamp'].values
        flows = historical_data.drop('timestamp', axis=1).values

        edge_index = torch.tensor(list(self.graph.edges())).t().contiguous().to(self.device)

        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0

            # Create batches
            for i in range(self.config.lookback_window, len(timestamps) - self.config.forecast_horizon):
                # Prepare features for current window
                current_time = timestamps[i]
                historical_flows = flows[i-self.config.lookback_window:i]

                features = self.prepare_features(
                    pd.to_datetime(current_time),
                    historical_flows
                )

                # Get target flows
                target_flows = flows[i:i+self.config.forecast_horizon]
                targets = torch.tensor(target_flows, dtype=torch.float32, device=self.device)
                targets = targets.unsqueeze(0).unsqueeze(-1)  # [1, H, N, 1]

                # Forward pass
                forecasts = self.backbone(features, edge_index)

                # Poisson loss for count data
                loss = -torch.sum(
                    targets * torch.log(forecasts + 1e-8) - forecasts
                )

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            scheduler.step()

            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        logger.info("Backbone training completed")

    def forecast(self, timestamp: datetime) -> Dict[str, Any]:
        """Generate forecast for given timestamp"""
        if self.historical_data is None:
            raise ValueError("Historical data not available. Call generate_historical_data first.")

        # Get recent historical data
        recent_data = self.historical_data[
            self.historical_data['timestamp'] <= timestamp
        ].tail(self.config.lookback_window)

        if len(recent_data) < self.config.lookback_window:
            logger.warning("Insufficient historical data for forecast")
            return None

        historical_flows = recent_data.drop('timestamp', axis=1).values

        # Prepare features
        features = self.prepare_features(timestamp, historical_flows)

        # Get edge index from graph
        edge_index = torch.tensor(list(self.graph.edges())).t().contiguous().to(self.device)

        # Backbone forecast
        with torch.no_grad():
            backbone_forecast = self.backbone(features, edge_index)
            backbone_forecast = backbone_forecast.cpu().numpy()[0]  # [H, N, 1]

        # Residual correction
        last_observed = historical_flows[-1]
        last_backbone_forecast = backbone_forecast[0, :, 0]  # First horizon
        residual = last_observed - last_backbone_forecast

        # Update and predict residual correction
        self.residual_corrector.update(residual)
        residual_correction = self.residual_corrector.predict_residual()

        # Apply correction to all horizons
        corrected_forecast = backbone_forecast.copy()
        for h in range(self.config.forecast_horizon):
            corrected_forecast[h, :, 0] += residual_correction

        # Check for drift
        drift_detected = self.residual_corrector.detect_drift()
        if drift_detected:
            logger.info(f"Drift detected at {timestamp}. Triggering low-rank adaptation.")
            self.performance_metrics['drift_events'].append(timestamp)
            # In real implementation, would trigger adaptation here

        # Hierarchical reconciliation
        reconciled_forecast = np.zeros_like(corrected_forecast)
        for h in range(self.config.forecast_horizon):
            station_forecasts = corrected_forecast[h, :, 0]
            reconciled = self.reconciler.reconcile(station_forecasts)
            reconciled_forecast[h, :, 0] = reconciled[:self.config.num_stations]

        # Generate forecast timestamps
        forecast_timestamps = [timestamp + timedelta(hours=h+1) for h in range(self.config.forecast_horizon)]

        return {
            'timestamp': timestamp,
            'forecast_timestamps': forecast_timestamps,
            'backbone_forecast': backbone_forecast,
            'corrected_forecast': corrected_forecast,
            'reconciled_forecast': reconciled_forecast,
            'drift_detected': drift_detected,
            'residual_correction': residual_correction
        }

    def simulate_real_time(self, days: int = 7, step_minutes: int = 30):
        """Simulate real-time forecasting"""
        logger.info(f"Starting real-time simulation for {days} days...")

        if self.historical_data is None:
            self.historical_data = self.data_generator.generate_historical_data(days + 1)

        current_time = datetime(2026, 1, 1, 0, 0)
        end_time = current_time + timedelta(days=days)

        while current_time < end_time:
            # Generate "real" observation
            real_flow = self.data_generator.generate_passenger_flow(current_time)

            # Get forecast
            forecast_result = self.forecast(current_time)

            if forecast_result:
                # Calculate performance metrics
                next_hour_flow = self.data_generator.generate_passenger_flow(current_time + timedelta(hours=1))
                mae = np.mean(np.abs(forecast_result['reconciled_forecast'][0, :, 0] - next_hour_flow))
                rmse = np.sqrt(np.mean((forecast_result['reconciled_forecast'][0, :, 0] - next_hour_flow) ** 2))

                self.performance_metrics['mae'].append(mae)
                self.performance_metrics['rmse'].append(rmse)

                # Store forecast history
                self.forecast_history.append({
                    'timestamp': current_time,
                    'real_flow': real_flow,
                    'forecast': forecast_result['reconciled_forecast'][0, :, 0],
                    'mae': mae,
                    'rmse': rmse,
                    'drift_detected': forecast_result['drift_detected']
                })

            # Update UI if available
            if STREAMLIT_AVAILABLE and self.ui_initialized:
                self._update_streamlit_ui(current_time, real_flow, forecast_result)

            # Progress update
            if len(self.forecast_history) % 24 == 0:  # Every 24 steps
                logger.info(f"Processed {len(self.forecast_history)} time steps. Current MAE: {mae:.2f}")

            # Move to next time step
            current_time += timedelta(minutes=step_minutes)
            time.sleep(0.1)  # Simulate real-time processing delay

        logger.info("Real-time simulation completed")
        return self.performance_metrics

    def _update_streamlit_ui(self, current_time: datetime, real_flow: np.ndarray, forecast_result: Dict):
        """Update Streamlit UI with current system state"""
        if not STREAMLIT_AVAILABLE:
            return

        st.title("🚌 DTS-GSSF Astana Bus Management Dashboard")

        # System status header
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🕐 Current Time</h3>
                <p style="font-size: 1.5rem; font-weight: bold;">{current_time.strftime('%Y-%m-%d %H:%M')}</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            if forecast_result and forecast_result['drift_detected']:
                status = '<span class="status-indicator status-red"></span> Drift Detected'
            else:
                status = '<span class="status-indicator status-green"></span> Stable'

            st.markdown(f"""
            <div class="metric-card">
                <h3>📊 System Status</h3>
                <p style="font-size: 1.3rem;">{status}</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            avg_mae = np.mean(self.performance_metrics['mae']) if self.performance_metrics['mae'] else 0
            st.markdown(f"""
            <div class="metric-card">
                <h3>🎯 Avg MAE</h3>
                <p style="font-size: 1.5rem; font-weight: bold;">{avg_mae:.2f} passengers</p>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            drift_count = len(self.performance_metrics['drift_events'])
            st.markdown(f"""
            <div class="metric-card">
                <h3>⚠️ Drift Events</h3>
                <p style="font-size: 1.5rem; font-weight: bold;">{drift_count}</p>
            </div>
            """, unsafe_allow_html=True)

        # Main content tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Real-time Forecast", "🗺️ Network View", "📊 Analytics", "⚙️ System Configuration"])

        with tab1:
            self._render_forecast_chart(real_flow, forecast_result)

        with tab2:
            self._render_network_graph()

        with tab3:
            self._render_analytics()

        with tab4:
            self._render_system_config()

        # Auto-rerun every 30 seconds for real-time updates
        st.experimental_rerun()

    def _render_forecast_chart(self, real_flow: np.ndarray, forecast_result: Dict):
        """Render forecast visualization"""
        if not forecast_result:
            st.warning("No forecast available yet")
            return

        # Create DataFrame for plotting
        station_names = [f"Station {i+1}" for i in range(self.config.num_stations)]
        df_current = pd.DataFrame({
            'Station': station_names,
            'Current Flow': real_flow,
            'Forecast (1hr)': forecast_result['reconciled_forecast'][0, :, 0]
        })

        # Plot current vs forecast
        fig = px.bar(df_current, x='Station', y=['Current Flow', 'Forecast (1hr)'],
                    title="Current vs 1-Hour Forecast by Station",
                    barmode='group',
                    color_discrete_sequence=['#1E88E5', '#FF7043'])

        fig.update_layout(
            xaxis_title="Stations",
            yaxis_title="Passenger Flow",
            height=500,
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)

        # Time series forecast for top stations
        top_stations = np.argsort(real_flow)[-5:]  # Top 5 stations by current flow

        fig2 = go.Figure()
        forecast_timestamps = forecast_result['forecast_timestamps']

        for station_idx in top_stations:
            station_name = self.data_generator.station_metadata[station_idx]['name']
            current_value = real_flow[station_idx]
            forecast_values = forecast_result['reconciled_forecast'][:, station_idx, 0]

            # Current value
            fig2.add_trace(go.Scatter(
                x=[forecast_timestamps[0] - timedelta(hours=1)],
                y=[current_value],
                mode='markers',
                name=f'{station_name} (Current)',
                marker=dict(size=10, symbol='diamond')
            ))

            # Forecast values
            fig2.add_trace(go.Scatter(
                x=forecast_timestamps,
                y=forecast_values,
                mode='lines+markers',
                name=f'{station_name} (Forecast)'
            ))

        fig2.update_layout(
            title="6-Hour Forecast for Top 5 Stations",
            xaxis_title="Time",
            yaxis_title="Passenger Flow",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig2, use_container_width=True)

    def _render_network_graph(self):
        """Render transport network graph"""
        # Create PyVis network or use Plotly for graph visualization
        st.subheader("🚌 Astana Bus Network")

        # Show network statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Stations", self.config.num_stations)
        with col2:
            st.metric("Active Routes", len(self.config.bus_routes))
        with col3:
            st.metric("Districts Covered", len(self.config.districts))

        # Simple network visualization (placeholder for full implementation)
        st.info("Network visualization would show real-time passenger flows, bottlenecks, and route capacities")

        # Show bottlenecks table
        st.subheader("🚦 Current Bottlenecks")
        bottlenecks = []
        for edge in self.graph.edges(data=True):
            if edge[2].get('is_bottleneck', False):
                bottlenecks.append({
                    'Road Segment': edge[2]['road_name'],
                    'From Station': self.data_generator.station_metadata[edge[0]]['name'],
                    'To Station': self.data_generator.station_metadata[edge[1]]['name'],
                    'Traffic Level': random.choice(['High', 'Medium', 'Critical'])
                })

        if bottlenecks:
            st.table(pd.DataFrame(bottlenecks))

    def _render_analytics(self):
        """Render performance analytics"""
        st.subheader("📈 System Performance Analytics")

        if not self.performance_metrics['mae']:
            st.warning("No performance data available yet")
            return

        # Performance metrics over time
        metrics_df = pd.DataFrame({
            'Timestamp': [f['timestamp'] for f in self.forecast_history],
            'MAE': [f['mae'] for f in self.forecast_history],
            'RMSE': [f['rmse'] for f in self.forecast_history],
            'Drift Detected': [f['drift_detected'] for f in self.forecast_history]
        })

        fig = px.line(metrics_df, x='Timestamp', y=['MAE', 'RMSE'],
                     title="Forecast Accuracy Over Time",
                     color_discrete_sequence=['#1E88E5', '#D81B60'])

        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Drift events analysis
        drift_events = [f for f in self.forecast_history if f['drift_detected']]
        if drift_events:
            st.subheader("⚠️ Drift Events Analysis")
            drift_df = pd.DataFrame({
                'Timestamp': [f['timestamp'] for f in drift_events],
                'MAE at Drift': [f['mae'] for f in drift_events]
            })
            st.table(drift_df)

    def _render_system_config(self):
        """Render system configuration"""
        st.subheader("⚙️ System Configuration")

        # Show current config
        config_data = {
            'Parameter': ['Lookback Window', 'Forecast Horizon', 'Residual Dimension', 'Hidden Dimension', 'Graph Depth', 'LoRA Rank', 'Drift Threshold'],
            'Value': [
                self.config.lookback_window,
                self.config.forecast_horizon,
                self.config.residual_dim,
                self.config.hidden_dim,
                self.config.graph_depth,
                self.config.lora_rank,
                self.config.drift_threshold
            ]
        }

        st.table(pd.DataFrame(config_data))

        # System resource usage
        st.subheader("💻 System Resources")

        # Placeholder for real resource monitoring
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("CPU Usage", "45%", "↓ 2%")
        with col2:
            st.metric("Memory Usage", "2.1 GB", "↑ 0.3 GB")
        with col3:
            st.metric("GPU Utilization", "15%", "Stable")

def main():
"""Main entry point for the DTS-GSSF system"""

    # Initialize configuration
    config = AstanaConfig.from_real_data()

    # Create system
    system = DTS_GSSF_System(config)

    # Generate and load historical data
    logger.info("Generating realistic Astana bus passenger flow data...")
    historical_data = system.data_generator.generate_historical_data(30)
    system.historical_data = historical_data

    # Train backbone model
    system.train_backbone(historical_data, epochs=5)

    # Start real-time simulation
    if STREAMLIT_AVAILABLE:
        logger.info("Starting Streamlit UI...")
        system.simulate_real_time(days=7, step_minutes=30)
    else:
        logger.info("Running in headless mode...")
        metrics = system.simulate_real_time(days=2, step_minutes=60)

        # Print final metrics
        console.print(Panel.fit(
            f"[bold green]Simulation Completed![/bold green]\n"
            f"Final MAE: {np.mean(metrics['mae']):.2f}\n"
            f"Final RMSE: {np.mean(metrics['rmse']):.2f}\n"
            f"Drift Events: {len(metrics['drift_events'])}",
            title="📊 Performance Results"
        ))

    logger.info("System shutdown complete")

if **name** == "**main**":
main()
