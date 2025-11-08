import type { Hazard, HazardType, HazardSeverity } from "@/types/hazards";

/**
 * Calculate cumulative distance along polyline in kilometers
 */
export function calculateCumulativeDistances(
  polyline: [number, number][]
): number[] {
  const distances = [0];
  for (let i = 1; i < polyline.length; i++) {
    const [lat1, lng1] = polyline[i - 1];
    const [lat2, lng2] = polyline[i];
    const dist = haversineDistance(lat1, lng1, lat2, lng2);
    distances.push(distances[i - 1] + dist);
  }
  return distances;
}

/**
 * Haversine distance in kilometers
 */
function haversineDistance(
  lat1: number,
  lon1: number,
  lat2: number,
  lon2: number
): number {
  const R = 6371; // Earth radius in km
  const dLat = ((lat2 - lat1) * Math.PI) / 180;
  const dLon = ((lon2 - lon1) * Math.PI) / 180;
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos((lat1 * Math.PI) / 180) *
      Math.cos((lat2 * Math.PI) / 180) *
      Math.sin(dLon / 2) *
      Math.sin(dLon / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
}

/**
 * Find nearest vertex index at a given fraction of the route
 */
function nearestVertex(
  polyline: [number, number][],
  cumulativeDistances: number[],
  fraction: number,
  jitter: number = 0
): number {
  const totalDistance = cumulativeDistances[cumulativeDistances.length - 1];
  const targetDistance = totalDistance * fraction;

  // Find the vertex closest to target distance
  let closestIdx = 0;
  let minDiff = Math.abs(cumulativeDistances[0] - targetDistance);

  for (let i = 1; i < cumulativeDistances.length; i++) {
    const diff = Math.abs(cumulativeDistances[i] - targetDistance);
    if (diff < minDiff) {
      minDiff = diff;
      closestIdx = i;
    }
  }

  // Apply jitter (±3-5 vertices)
  const jitterAmount = Math.floor(Math.random() * 3 + 3) * (Math.random() > 0.5 ? 1 : -1);
  const jitteredIdx = Math.max(
    0,
    Math.min(polyline.length - 1, closestIdx + jitterAmount + jitter)
  );

  return jitteredIdx;
}

/**
 * Generate synthetic hazards for a route with density-based placement
 * @param routeId Route identifier
 * @param polyline Route coordinates
 * @param densityFactor Multiplier for hazard count (default 1.5×)
 */
export function generateHazardsForRoute(
  routeId: string,
  polyline: [number, number][],
  densityFactor: number = 1.5
): Hazard[] {
  if (polyline.length < 10) return [];

  const cumulativeDistances = calculateCumulativeDistances(polyline);
  const totalDistanceKm = cumulativeDistances[cumulativeDistances.length - 1];

  const hazards: Hazard[] = [];
  const severities: HazardSeverity[] = ["low", "med", "high"];

  // Density per 100km
  const baseDensity = {
    pothole: 6,
    speed_bump: 5,
    debris: 4,
  };

  // Calculate counts based on route length and density factor
  const counts = {
    pothole: Math.max(3, Math.min(10, Math.round((totalDistanceKm / 100) * baseDensity.pothole * densityFactor))),
    speed_bump: Math.max(3, Math.min(10, Math.round((totalDistanceKm / 100) * baseDensity.speed_bump * densityFactor))),
    debris: Math.max(3, Math.min(10, Math.round((totalDistanceKm / 100) * baseDensity.debris * densityFactor))),
  };

  // Generate placement fractions
  const generatePlacements = (count: number): number[] => {
    const placements: number[] = [];
    for (let i = 0; i < count; i++) {
      // Spread evenly with some variation
      const fraction = (i + 1) / (count + 1) + (Math.random() - 0.5) * 0.05;
      placements.push(Math.max(0.05, Math.min(0.95, fraction)));
    }
    return placements;
  };

  let hazardIndex = 0;

  // Place potholes
  const potholePlacements = generatePlacements(counts.pothole);
  potholePlacements.forEach((fraction) => {
    const vertexIdx = nearestVertex(polyline, cumulativeDistances, fraction, hazardIndex);
    const coord = polyline[vertexIdx];
    const kmFromStart = cumulativeDistances[vertexIdx];
    const severity = severities[Math.floor(Math.random() * severities.length)];

    hazards.push({
      id: `${routeId}-pothole-${hazardIndex}`,
      type: "pothole",
      coord,
      routeId,
      kmFromStart,
      severity,
    });
    hazardIndex++;
  });

  // Place speed bumps
  const speedBumpPlacements = generatePlacements(counts.speed_bump);
  speedBumpPlacements.forEach((fraction) => {
    const vertexIdx = nearestVertex(polyline, cumulativeDistances, fraction, hazardIndex);
    const coord = polyline[vertexIdx];
    const kmFromStart = cumulativeDistances[vertexIdx];
    const severity = severities[Math.floor(Math.random() * severities.length)];

    hazards.push({
      id: `${routeId}-speedbump-${hazardIndex}`,
      type: "speed_bump",
      coord,
      routeId,
      kmFromStart,
      severity,
    });
    hazardIndex++;
  });

  // Place debris
  const debrisPlacements = generatePlacements(counts.debris);
  debrisPlacements.forEach((fraction) => {
    const vertexIdx = nearestVertex(polyline, cumulativeDistances, fraction, hazardIndex);
    const coord = polyline[vertexIdx];
    const kmFromStart = cumulativeDistances[vertexIdx];
    const severity = severities[Math.floor(Math.random() * severities.length)];

    hazards.push({
      id: `${routeId}-debris-${hazardIndex}`,
      type: "debris",
      coord,
      routeId,
      kmFromStart,
      severity,
    });
    hazardIndex++;
  });

  return hazards;
}

/**
 * Calculate distance between a point and a hazard
 */
export function distanceToHazard(
  point: [number, number],
  hazard: Hazard
): number {
  return haversineDistance(point[0], point[1], hazard.coord[0], hazard.coord[1]);
}

/**
 * Calculate remaining distance to destination from a point along the route
 */
export function remainingDistance(
  polyline: [number, number][],
  cumulativeDistances: number[],
  currentVertexIdx: number
): number {
  const totalDistance = cumulativeDistances[cumulativeDistances.length - 1];
  const currentDistance = cumulativeDistances[currentVertexIdx];
  return totalDistance - currentDistance;
}
