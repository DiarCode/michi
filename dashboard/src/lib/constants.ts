// Astana coordinate bounds for the map projection
export const MAP_LON_MIN = 71.25;
export const MAP_LON_SPAN = 0.4;
export const MAP_LAT_MAX = 51.25;
export const MAP_LAT_SPAN = 0.3;

// Load / occupancy thresholds (percent)
export const LOAD_HIGH = 80;
export const LOAD_MID = 50;

// Station capacity (aligns with backend STATION_CAPACITY)
export const STATION_CAPACITY = 3000;

// Ridership tiers for fallback load estimation
export const RIDERSHIP_HIGH = 3000;
export const RIDERSHIP_MID = 2000;

// Heatmap sizing
export const HEATMAP_MIN_PX = 8;
export const HEATMAP_RANGE_PX = 20;

// Peak hours for load calculation
export const MORNING_PEAK = [7, 9] as const;
export const EVENING_PEAK = [17, 19] as const;