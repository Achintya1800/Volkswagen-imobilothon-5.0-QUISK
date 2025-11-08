import type { Hazard } from "@/types/hazards";

export interface Route {
  id: string;
  name: string;
  etaMins: number;
  distanceKm: number;
  traffic: "Light" | "Moderate" | "Heavy";
  callouts: string[];
  via: string;
  polyline: [number, number][];
  color: string;
  hazards?: Hazard[];
}

// Mock routes data with realistic coordinates for a city route
export const mockRoutes: Route[] = [
  {
    id: "r1",
    name: "Route A",
    etaMins: 24,
    distanceKm: 12.3,
    traffic: "Moderate",
    callouts: ["Fewer hazards", "Fewer potholes", "Speed bumps ahead"],
    via: "via major roads",
    color: "#3b82f6", // blue
    polyline: [
      [40.7580, -73.9855],
      [40.7590, -73.9845],
      [40.7600, -73.9835],
      [40.7610, -73.9825],
      [40.7620, -73.9815],
      [40.7630, -73.9805],
      [40.7640, -73.9795],
      [40.7650, -73.9785],
      [40.7660, -73.9775],
      [40.7670, -73.9765],
    ],
  },
  {
    id: "r2",
    name: "Route B",
    etaMins: 28,
    distanceKm: 13.8,
    traffic: "Light",
    callouts: ["Watch for stalled vehicle", "Low flooding risk"],
    via: "via local streets",
    color: "#10b981", // green
    polyline: [
      [40.7580, -73.9855],
      [40.7585, -73.9840],
      [40.7595, -73.9830],
      [40.7605, -73.9820],
      [40.7615, -73.9810],
      [40.7625, -73.9800],
      [40.7635, -73.9790],
      [40.7645, -73.9780],
      [40.7655, -73.9770],
      [40.7670, -73.9765],
    ],
  },
  {
    id: "r3",
    name: "Route C",
    etaMins: 22,
    distanceKm: 14.5,
    traffic: "Heavy",
    callouts: ["Fewer hazards", "Speed bumps ahead"],
    via: "via major roads",
    color: "#8b5cf6", // purple
    polyline: [
      [40.7580, -73.9855],
      [40.7592, -73.9850],
      [40.7602, -73.9840],
      [40.7612, -73.9830],
      [40.7622, -73.9820],
      [40.7632, -73.9810],
      [40.7642, -73.9800],
      [40.7652, -73.9790],
      [40.7662, -73.9780],
      [40.7670, -73.9765],
    ],
  },
];

export const origin: [number, number] = [40.7580, -73.9855];
export const destination: [number, number] = [40.7670, -73.9765];
