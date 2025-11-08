export type HazardType = "pothole" | "speed_bump" | "debris";

export type HazardSeverity = "low" | "med" | "high";

export interface Hazard {
  id: string;
  type: HazardType;
  coord: [number, number]; // lat, lng snapped to route
  routeId: string;
  kmFromStart: number;
  severity: HazardSeverity;
}

export interface HazardToggleState {
  pothole: boolean;
  speed_bump: boolean;
  debris: boolean;
}
