import polyline from "@mapbox/polyline";
import type { Route } from "@/data/mockRoutes";
import { generateHazardsForRoute } from "@/utils/hazardGenerator";

interface OSRMRoute {
  distance: number;
  duration: number;
  geometry: string;
  legs: Array<{
    steps: Array<{
      name: string;
      distance: number;
      duration: number;
      maneuver: {
        type: string;
        modifier?: string;
      };
    }>;
  }>;
}

interface OSRMResponse {
  code: string;
  routes: OSRMRoute[];
}

// Use blue palette: muted for alternates, stronger for primary
const COLORS = ["#60A5FA", "#3B82F6", "#1E40AF", "#2563EB"];
const CALLOUTS_POOL = [
  "Fewer hazards",
  "Fewer potholes",
  "Speed bumps ahead",
  "Watch for stalled vehicle",
  "Low flooding risk",
  "Scenic route",
];

export const getRoutes = async (
  startLon: number,
  startLat: number,
  endLon: number,
  endLat: number,
  densityFactor: number = 1.5
): Promise<{ routes: Route[]; steps: string[][] }> => {
  const url = `https://router.project-osrm.org/route/v1/driving/${startLon},${startLat};${endLon},${endLat}?alternatives=3&overview=full&geometries=polyline6&steps=true`;

  const response = await fetch(url);

  if (!response.ok) {
    throw new Error("Routing failed");
  }

  const data: OSRMResponse = await response.json();

  if (data.code !== "Ok" || !data.routes || data.routes.length === 0) {
    throw new Error("No routes found");
  }

  // Find fastest route for traffic comparison
  const fastestDuration = Math.min(...data.routes.map((r) => r.duration));

  const routes: Route[] = data.routes.map((osrmRoute, idx) => {
    // Decode polyline6
    const coords = polyline.decode(osrmRoute.geometry, 6);
    const latLngs: [number, number][] = coords.map(([lat, lng]) => [lat, lng]);

    // Calculate traffic level
    const durationDiff = ((osrmRoute.duration - fastestDuration) / fastestDuration) * 100;
    let traffic: "Light" | "Moderate" | "Heavy";
    if (durationDiff <= 10) {
      traffic = "Light";
    } else if (durationDiff <= 25) {
      traffic = "Moderate";
    } else {
      traffic = "Heavy";
    }

    // Pick 2-3 random callouts
    const numCallouts = 2 + Math.floor(Math.random() * 2);
    const shuffled = [...CALLOUTS_POOL].sort(() => Math.random() - 0.5);
    const callouts = shuffled.slice(0, numCallouts);

    // Generate hazards for this route with density factor
    const hazards = generateHazardsForRoute(`r${idx + 1}`, latLngs, densityFactor);

    return {
      id: `r${idx + 1}`,
      name: `Route ${String.fromCharCode(65 + idx)}`,
      etaMins: Math.round(osrmRoute.duration / 60),
      distanceKm: parseFloat((osrmRoute.distance / 1000).toFixed(1)),
      traffic,
      callouts,
      polyline: latLngs,
      color: COLORS[idx % COLORS.length],
      via: idx === 0 ? "via major roads" : "via local streets",
      hazards,
    };
  });

  // Extract steps for each route
  const steps = data.routes.map((osrmRoute) =>
    osrmRoute.legs.flatMap((leg) =>
      leg.steps
        .filter((step) => step.name)
        .map((step) => {
          const action = step.maneuver.type === "turn" 
            ? `Turn ${step.maneuver.modifier || ""}`
            : step.maneuver.type === "depart"
            ? "Head"
            : "Continue";
          return `${action} on ${step.name || "unnamed road"}`;
        })
    )
  );

  // If fewer than 4 routes, synthesize Route D via Lonavala
  if (routes.length < 4) {
    const viaLon = 73.405;
    const viaLat = 18.750; // Lonavala coordinates

    try {
      // Fetch two legs: start -> via, via -> end
      const leg1Response = await fetch(
        `https://router.project-osrm.org/route/v1/driving/${startLon},${startLat};${viaLon},${viaLat}?overview=full&geometries=polyline6&steps=true`
      );
      const leg2Response = await fetch(
        `https://router.project-osrm.org/route/v1/driving/${viaLon},${viaLat};${endLon},${endLat}?overview=full&geometries=polyline6&steps=true`
      );

      if (leg1Response.ok && leg2Response.ok) {
        const leg1Data: OSRMResponse = await leg1Response.json();
        const leg2Data: OSRMResponse = await leg2Response.json();

        if (
          leg1Data.code === "Ok" &&
          leg1Data.routes.length > 0 &&
          leg2Data.code === "Ok" &&
          leg2Data.routes.length > 0
        ) {
          const leg1Route = leg1Data.routes[0];
          const leg2Route = leg2Data.routes[0];

          // Decode and concatenate polylines
          const coords1 = polyline.decode(leg1Route.geometry, 6);
          const coords2 = polyline.decode(leg2Route.geometry, 6);
          const stitchedCoords = [
            ...coords1.map(([lat, lng]) => [lat, lng] as [number, number]),
            ...coords2.slice(1).map(([lat, lng]) => [lat, lng] as [number, number]), // Skip first point to avoid duplicate
          ];

          // Sum distance and duration
          const totalDistance = leg1Route.distance + leg2Route.distance;
          const totalDuration = leg1Route.duration + leg2Route.duration;

          // Calculate traffic level
          const durationDiff = ((totalDuration - fastestDuration) / fastestDuration) * 100;
          let traffic: "Light" | "Moderate" | "Heavy";
          if (durationDiff <= 10) {
            traffic = "Light";
          } else if (durationDiff <= 25) {
            traffic = "Moderate";
          } else {
            traffic = "Heavy";
          }

          // Concatenate steps
          const leg1Steps = leg1Route.legs.flatMap((leg) =>
            leg.steps
              .filter((step) => step.name)
              .map((step) => {
                const action = step.maneuver.type === "turn"
                  ? `Turn ${step.maneuver.modifier || ""}`
                  : step.maneuver.type === "depart"
                  ? "Head"
                  : "Continue";
                return `${action} on ${step.name || "unnamed road"}`;
              })
          );
          const leg2Steps = leg2Route.legs.flatMap((leg) =>
            leg.steps
              .filter((step) => step.name)
              .map((step) => {
                const action = step.maneuver.type === "turn"
                  ? `Turn ${step.maneuver.modifier || ""}`
                  : step.maneuver.type === "depart"
                  ? "Head"
                  : "Continue";
                return `${action} on ${step.name || "unnamed road"}`;
              })
          );

          const stitchedSteps = [...leg1Steps, ...leg2Steps];

          // Generate hazards for stitched route
          const hazards = generateHazardsForRoute("r4", stitchedCoords, densityFactor);

          // Add Route D
          routes.push({
            id: "r4",
            name: "Route D",
            etaMins: Math.round(totalDuration / 60),
            distanceKm: parseFloat((totalDistance / 1000).toFixed(1)),
            traffic,
            callouts: ["Scenic route", "Fewer hazards"],
            polyline: stitchedCoords,
            color: COLORS[3 % COLORS.length],
            via: "via Lonavala",
            hazards,
          });

          steps.push(stitchedSteps);
        }
      }
    } catch (error) {
      console.warn("Failed to synthesize Route D via Lonavala:", error);
    }
  }

  // Sort routes by ETA
  const sortedIndices = routes
    .map((r, idx) => ({ route: r, step: steps[idx], idx }))
    .sort((a, b) => a.route.etaMins - b.route.etaMins);

  const sortedRoutes = sortedIndices.map((item) => item.route);
  const sortedSteps = sortedIndices.map((item) => item.step);

  return { routes: sortedRoutes, steps: sortedSteps };
}
