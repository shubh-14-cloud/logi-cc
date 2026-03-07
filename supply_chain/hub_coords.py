"""
supply_chain/hub_coords.py
===========================
Geographic coordinates and metadata for each logistics hub.
Used by the Plotly Scattergeo map in the Streamlit frontend.
"""

from __future__ import annotations
from typing import Dict, Any

# lat/lon + display metadata for each hub
HUB_GEO: Dict[str, Dict[str, Any]] = {
    "HUB-NYC": {"lat": 40.7128, "lon": -74.0060, "city": "New York",    "state": "NY", "region": "Northeast"},
    "HUB-BOS": {"lat": 42.3601, "lon": -71.0589, "city": "Boston",     "state": "MA", "region": "Northeast"},
    "HUB-MIA": {"lat": 25.7617, "lon": -80.1918, "city": "Miami",      "state": "FL", "region": "Southeast"},
    "HUB-ATL": {"lat": 33.7490, "lon": -84.3880, "city": "Atlanta",    "state": "GA", "region": "Southeast"},
    "HUB-CHI": {"lat": 41.8781, "lon": -87.6298, "city": "Chicago",    "state": "IL", "region": "Midwest"},
    "HUB-DAL": {"lat": 32.7767, "lon": -96.7970, "city": "Dallas",     "state": "TX", "region": "South"},
    "HUB-DEN": {"lat": 39.7392, "lon": -104.9903,"city": "Denver",     "state": "CO", "region": "Mountain"},
    "HUB-PHX": {"lat": 33.4484, "lon": -112.0740,"city": "Phoenix",    "state": "AZ", "region": "Southwest"},
    "HUB-LAX": {"lat": 34.0522, "lon": -118.2437,"city": "Los Angeles","state": "CA", "region": "West"},
    "HUB-SEA": {"lat": 47.6062, "lon": -122.3321,"city": "Seattle",    "state": "WA", "region": "Northwest"},
}

# Route connections for drawing edge lines on the map
# Mirrors the route definitions in environment.py
HUB_ROUTES = [
    ("HUB-NYC", "HUB-CHI"),
    ("HUB-NYC", "HUB-ATL"),
    ("HUB-NYC", "HUB-BOS"),
    ("HUB-NYC", "HUB-MIA"),
    ("HUB-CHI", "HUB-DAL"),
    ("HUB-CHI", "HUB-DEN"),
    ("HUB-CHI", "HUB-LAX"),
    ("HUB-LAX", "HUB-SEA"),
    ("HUB-LAX", "HUB-DEN"),
    ("HUB-LAX", "HUB-PHX"),
    ("HUB-DAL", "HUB-ATL"),
    ("HUB-DAL", "HUB-MIA"),
    ("HUB-ATL", "HUB-MIA"),
    ("HUB-SEA", "HUB-DEN"),
    ("HUB-DEN", "HUB-PHX"),
    ("HUB-BOS", "HUB-NYC"),
]


def risk_to_hex(score: float) -> str:
    """Map a 0-1 risk score to a hex colour (green -> yellow -> orange -> red)."""
    if score >= 0.82:
        return "#ff2020"   # CRITICAL — bright red
    if score >= 0.65:
        return "#ff6600"   # HIGH — orange
    if score >= 0.45:
        return "#ffaa00"   # MEDIUM — amber
    if score >= 0.25:
        return "#ffdd00"   # LOW — yellow
    return "#00cc55"       # NOMINAL — green


def risk_label(score: float) -> str:
    if score >= 0.82: return "CRITICAL"
    if score >= 0.65: return "HIGH"
    if score >= 0.45: return "MEDIUM"
    if score >= 0.25: return "LOW"
    return "NOMINAL"
