import L from "leaflet";
import type { HazardType } from "@/types/hazards";

/**
 * Create a DivIcon for a hazard marker
 */
export function createHazardIcon(type: HazardType, isPulsing: boolean = false): L.DivIcon {
  const colors = {
    pothole: "#3b82f6", // blue
    speed_bump: "#f97316", // orange
    debris: "#a855f7", // purple
  };

  const color = colors[type];
  const pulseClass = isPulsing ? "animate-ping" : "";

  let iconHtml = "";

  switch (type) {
    case "pothole":
      iconHtml = `
        <div class="relative ${pulseClass}">
          <div class="w-5 h-5 rounded-full border-2 bg-white/90" style="border-color: ${color}; box-shadow: 0 2px 4px rgba(0,0,0,0.3);"></div>
        </div>
      `;
      break;
    case "speed_bump":
      iconHtml = `
        <div class="relative ${pulseClass}">
          <div class="w-5 h-3 rounded-sm border border-white bg-white/90" style="background-color: ${color}; box-shadow: 0 2px 4px rgba(0,0,0,0.3);"></div>
        </div>
      `;
      break;
    case "debris":
      iconHtml = `
        <div class="relative ${pulseClass}">
          <div style="
            width: 0; 
            height: 0; 
            border-left: 8px solid transparent;
            border-right: 8px solid transparent;
            border-bottom: 14px solid ${color};
            filter: drop-shadow(0 2px 4px rgba(0,0,0,0.3));
          "></div>
        </div>
      `;
      break;
  }

  return L.divIcon({
    className: "hazard-marker",
    html: iconHtml,
    iconSize: [20, 20],
    iconAnchor: [10, 10],
  });
}

/**
 * Pretty print hazard type
 */
export function prettyHazardType(type: HazardType): string {
  const names = {
    pothole: "Pothole",
    speed_bump: "Speed bump",
    debris: "Debris",
  };
  return names[type];
}

/**
 * Get severity color class
 */
export function getSeverityColor(severity: string): string {
  const colors = {
    low: "bg-green-100 text-green-800",
    med: "bg-yellow-100 text-yellow-800",
    high: "bg-red-100 text-red-800",
  };
  return colors[severity as keyof typeof colors] || colors.med;
}
