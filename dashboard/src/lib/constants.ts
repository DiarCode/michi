// Load / occupancy thresholds (percent)
export const LOAD_HIGH = 80
export const LOAD_MID = 50

// Station capacity (aligns with backend STATION_CAPACITY)
export const STATION_CAPACITY = 3000

// Ridership tiers for fallback load estimation
export const RIDERSHIP_HIGH = 3000
export const RIDERSHIP_MID = 2000

// Peak hours for load calculation
export const MORNING_PEAK = [7, 9] as const
export const EVENING_PEAK = [17, 19] as const

// Canonical route IDs used by seed data
export const SEED_ROUTE_IDS = ["R12", "R18", "R25", "R31", "R40"] as const
