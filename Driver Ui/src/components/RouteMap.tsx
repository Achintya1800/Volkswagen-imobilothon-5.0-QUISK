import { useEffect, useRef, useState } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import type { Route } from "@/data/mockRoutes";
import type { HazardToggleState, Hazard } from "@/types/hazards";
import { createHazardIcon, prettyHazardType, getSeverityColor } from "@/utils/hazardIcons";
import { calculateCumulativeDistances, remainingDistance, distanceToHazard, generateHazardsForRoute } from "@/utils/hazardGenerator";
import { HazardLegend } from "@/components/HazardLegend";
import { useToast } from "@/hooks/use-toast";

// Fix for default marker icon
import markerIcon2x from "leaflet/dist/images/marker-icon-2x.png";
import markerIcon from "leaflet/dist/images/marker-icon.png";
import markerShadow from "leaflet/dist/images/marker-shadow.png";

delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconUrl: markerIcon,
  iconRetinaUrl: markerIcon2x,
  shadowUrl: markerShadow,
});

interface RouteMapProps {
  routes: Route[];
  selectedRoute: Route | null;
  hoveredRouteId?: string | null;
  previewRouteId?: string | null;
  isNavigating: boolean;
  origin: [number, number] | null;
  destination: [number, number] | null;
  onMarkerDrag?: (isOrigin: boolean, lat: number, lng: number) => void;
  densityFactor?: number;
  onDensityChange?: (factor: number) => void;
}

export const RouteMap = ({
  routes,
  selectedRoute,
  hoveredRouteId,
  previewRouteId,
  isNavigating,
  origin,
  destination,
  onMarkerDrag,
  densityFactor = 1.5,
  onDensityChange,
}: RouteMapProps) => {
  const mapRef = useRef<L.Map | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const routeLinesRef = useRef<{ [key: string]: L.Polyline }>({});
  const markerRef = useRef<L.Marker | null>(null);
  const originMarkerRef = useRef<L.Marker | null>(null);
  const destinationMarkerRef = useRef<L.Marker | null>(null);
  const [currentStep, setCurrentStep] = useState(0);
  const canvasRendererRef = useRef(L.canvas({ padding: 0.5 }));
  const previewLineRef = useRef<L.Polyline | null>(null);
  const previewTimersRef = useRef<number[]>([]);
  
  // Hazard state and refs
  const [hazardToggles, setHazardToggles] = useState<HazardToggleState>({
    pothole: true,
    speed_bump: true,
    debris: true,
  });
  const [currentDensity, setCurrentDensity] = useState<number>(densityFactor);
  const hazardLayersRef = useRef<{
    pothole: L.LayerGroup;
    speed_bump: L.LayerGroup;
    debris: L.LayerGroup;
  } | null>(null);
  const alertedHazardsRef = useRef<Set<string>>(new Set());
  const { toast } = useToast();

  // Initialize map
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;

    const map = L.map(containerRef.current, {
      preferCanvas: true,
      inertia: true,
    }).setView([22.9734, 78.6569], 5);

    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution:
        '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
    }).addTo(map);

    // Initialize hazard layer groups
    hazardLayersRef.current = {
      pothole: L.layerGroup().addTo(map),
      speed_bump: L.layerGroup().addTo(map),
      debris: L.layerGroup().addTo(map),
    };

    mapRef.current = map;

    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  // Update origin/destination markers
  useEffect(() => {
    if (!mapRef.current) return;

    // Remove old markers
    if (originMarkerRef.current) {
      originMarkerRef.current.remove();
      originMarkerRef.current = null;
    }
    if (destinationMarkerRef.current) {
      destinationMarkerRef.current.remove();
      destinationMarkerRef.current = null;
    }

    if (origin) {
      const marker = L.marker(origin, {
        icon: L.divIcon({
          className: "custom-marker",
          html: '<div class="w-4 h-4 bg-primary rounded-full border-2 border-white shadow-lg"></div>',
          iconSize: [16, 16],
        }),
        draggable: true,
      })
        .addTo(mapRef.current)
        .bindPopup("Start");

      marker.on("dragend", () => {
        const pos = marker.getLatLng();
        onMarkerDrag?.(true, pos.lat, pos.lng);
      });

      originMarkerRef.current = marker;
    }

    if (destination) {
      const marker = L.marker(destination, {
        icon: L.divIcon({
          className: "custom-marker",
          html: '<div class="w-4 h-4 bg-accent rounded-full border-2 border-white shadow-lg"></div>',
          iconSize: [16, 16],
        }),
        draggable: true,
      })
        .addTo(mapRef.current)
        .bindPopup("Destination");

      marker.on("dragend", () => {
        const pos = marker.getLatLng();
        onMarkerDrag?.(false, pos.lat, pos.lng);
      });

      destinationMarkerRef.current = marker;
    }
  }, [origin, destination, onMarkerDrag]);

  // Draw routes
  useEffect(() => {
    if (!mapRef.current) return;

    // Clear existing route lines
    Object.values(routeLinesRef.current).forEach((line) => line.remove());
    routeLinesRef.current = {};

    // Draw all routes
    routes.forEach((route) => {
      const isSelected = selectedRoute?.id === route.id;
      const polyline = L.polyline(route.polyline, {
        color: route.color,
        weight: isSelected ? 7 : 4,
        opacity: isSelected ? 1 : 0.6,
        smoothFactor: 1.5,
        lineCap: "round",
        lineJoin: "round",
        renderer: canvasRendererRef.current,
      }).addTo(mapRef.current!);

      // Bring selected to front
      if (isSelected) {
        polyline.bringToFront();
      }

      routeLinesRef.current[route.id] = polyline;
    });

    // Fit bounds to the longest alternative (by distanceKm) when routes change
    if (routes.length > 0) {
      const longest = routes.reduce((acc, r) => (r.distanceKm > acc.distanceKm ? r : acc), routes[0]);
      const bounds = L.latLngBounds(longest.polyline);
      mapRef.current.fitBounds(bounds, { padding: [50, 50] });
    }
  }, [routes, selectedRoute]);

  // Hover effect: thicken and raise z-index for hovered route without zoom
  useEffect(() => {
    if (!mapRef.current) return;

    Object.entries(routeLinesRef.current).forEach(([id, line]) => {
      const isSelected = selectedRoute?.id === id;
      line.setStyle({ weight: isSelected ? 7 : 4, opacity: isSelected ? 1 : 0.6 });
    });

    if (hoveredRouteId) {
      const line = routeLinesRef.current[hoveredRouteId];
      if (line) {
        line.setStyle({ weight: 7, opacity: 1 });
        line.bringToFront();
      }
    }
  }, [hoveredRouteId, selectedRoute]);

  // Preview animation (snake/draw-by-segment)
  useEffect(() => {
    if (!mapRef.current) return;

    // clear previous timers and preview line
    previewTimersRef.current.forEach((t) => clearTimeout(t));
    previewTimersRef.current = [];
    if (previewLineRef.current) {
      previewLineRef.current.remove();
      previewLineRef.current = null;
    }

    if (!previewRouteId) return;

    const route = routes.find((r) => r.id === previewRouteId);
    if (!route) return;

    const pts = route.polyline;
    if (!pts || pts.length === 0) return;

    const preview = L.polyline([], {
      color: route.color,
      weight: 6,
      opacity: 1,
      smoothFactor: 1.5,
      lineCap: "round",
      lineJoin: "round",
      renderer: canvasRendererRef.current,
    }).addTo(mapRef.current);

    previewLineRef.current = preview;

    const total = 700;
    const step = Math.max(10, Math.floor(total / pts.length));

    pts.forEach((pt, i) => {
      const t = window.setTimeout(() => {
        const curr = preview.getLatLngs() as L.LatLngExpression[];
        curr.push(pt as L.LatLngExpression);
        preview.setLatLngs(curr);
        if (i === pts.length - 1) {
          const real = routeLinesRef.current[route.id];
          if (real) real.bringToFront();
          const cleanup = window.setTimeout(() => {
            if (previewLineRef.current) {
              previewLineRef.current.remove();
              previewLineRef.current = null;
            }
          }, 500);
          previewTimersRef.current.push(cleanup);
        }
      }, i * step);
      previewTimersRef.current.push(t);
    });
  }, [previewRouteId, routes]);

  // Hover effect: thicken and raise z-index for hovered route without zoom
  useEffect(() => {
    if (!mapRef.current) return;

    // reset all non-selected routes to default
    Object.entries(routeLinesRef.current).forEach(([id, line]) => {
      const isSelected = selectedRoute?.id === id;
      line.setStyle({ weight: isSelected ? 7 : 4, opacity: isSelected ? 1 : 0.6 });
    });

    // Apply hover styling
    // hoveredRouteId is passed as prop/arg to this component
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const hoveredId = (null as string | null);
  }, [selectedRoute]);

  // NOTE: hoveredRouteId and previewRouteId are destructured props but were not
  // included in the function signature's destructure above; to avoid breaking
  // the component shape, we'll read them from arguments via closure below.

  // Navigation simulation
  useEffect(() => {
    if (!isNavigating || !selectedRoute || !mapRef.current) {
      if (markerRef.current) {
        markerRef.current.remove();
        markerRef.current = null;
      }
      setCurrentStep(0);
      return;
    }

    // Create or update moving marker
    if (!markerRef.current) {
      markerRef.current = L.marker(selectedRoute.polyline[0], {
        icon: L.divIcon({
          className: "custom-marker",
          html: '<div class="w-6 h-6 bg-accent rounded-full border-4 border-white shadow-xl animate-pulse"><div class="absolute inset-1 bg-white rounded-full"></div></div>',
          iconSize: [24, 24],
        }),
      }).addTo(mapRef.current);
    }

    const interval = setInterval(() => {
      setCurrentStep((prev) => {
        const next = (prev + 1) % selectedRoute.polyline.length;
        if (markerRef.current) {
          markerRef.current.setLatLng(selectedRoute.polyline[next]);
        }
        return next;
      });
    }, 2000);

    return () => clearInterval(interval);
  }, [isNavigating, selectedRoute]);

  // Render hazards for selected route
  useEffect(() => {
    if (!mapRef.current || !hazardLayersRef.current || !selectedRoute) {
      // Clear all hazard layers
      if (hazardLayersRef.current) {
        hazardLayersRef.current.pothole.clearLayers();
        hazardLayersRef.current.speed_bump.clearLayers();
        hazardLayersRef.current.debris.clearLayers();
      }
      return;
    }

    // Clear existing hazards
    hazardLayersRef.current.pothole.clearLayers();
    hazardLayersRef.current.speed_bump.clearLayers();
    hazardLayersRef.current.debris.clearLayers();

    // Regenerate hazards with current density if different from route's stored hazards
    let hazards = selectedRoute.hazards || [];
    
    // If density changed, regenerate hazards
    if (currentDensity !== densityFactor && selectedRoute.polyline) {
      hazards = generateHazardsForRoute(selectedRoute.id, selectedRoute.polyline, currentDensity);
    }

    const cumulativeDistances = calculateCumulativeDistances(selectedRoute.polyline);

    hazards.forEach((hazard) => {
      const icon = createHazardIcon(hazard.type);
      const marker = L.marker(hazard.coord, { icon });

      // Find nearest vertex index for remaining distance calculation
      let nearestIdx = 0;
      let minDist = Infinity;
      selectedRoute.polyline.forEach((pt, idx) => {
        const dist = Math.abs(pt[0] - hazard.coord[0]) + Math.abs(pt[1] - hazard.coord[1]);
        if (dist < minDist) {
          minDist = dist;
          nearestIdx = idx;
        }
      });

      const remaining = remainingDistance(
        selectedRoute.polyline,
        cumulativeDistances,
        nearestIdx
      );

      // Popup content
      const popupContent = `
        <div class="p-2">
          <div class="font-semibold text-sm">${prettyHazardType(hazard.type)}</div>
          <div class="flex items-center gap-2 mt-1">
            <span class="text-xs px-2 py-0.5 rounded ${getSeverityColor(hazard.severity)}">
              ${hazard.severity.toUpperCase()}
            </span>
          </div>
          <div class="text-xs text-muted-foreground mt-2">
            <div>At km ${hazard.kmFromStart.toFixed(1)}</div>
            <div>~${remaining.toFixed(1)} km to destination</div>
          </div>
        </div>
      `;

      marker.bindPopup(popupContent);

      // Hover effect
      marker.on("mouseover", function () {
        this.getElement()?.style.setProperty("transform", "scale(1.2)");
        this.getElement()?.style.setProperty("cursor", "pointer");
      });

      marker.on("mouseout", function () {
        this.getElement()?.style.setProperty("transform", "scale(1)");
      });

      // Add to appropriate layer group
      hazardLayersRef.current![hazard.type].addLayer(marker);
    });

    // Apply toggle state
    if (!hazardToggles.pothole) {
      mapRef.current.removeLayer(hazardLayersRef.current.pothole);
    } else if (!mapRef.current.hasLayer(hazardLayersRef.current.pothole)) {
      hazardLayersRef.current.pothole.addTo(mapRef.current);
    }

    if (!hazardToggles.speed_bump) {
      mapRef.current.removeLayer(hazardLayersRef.current.speed_bump);
    } else if (!mapRef.current.hasLayer(hazardLayersRef.current.speed_bump)) {
      hazardLayersRef.current.speed_bump.addTo(mapRef.current);
    }

    if (!hazardToggles.debris) {
      mapRef.current.removeLayer(hazardLayersRef.current.debris);
    } else if (!mapRef.current.hasLayer(hazardLayersRef.current.debris)) {
      hazardLayersRef.current.debris.addTo(mapRef.current);
    }
  }, [selectedRoute, hazardToggles, currentDensity, densityFactor]);

  // Handle hazard toggle changes
  const handleHazardToggle = (type: keyof HazardToggleState, checked: boolean) => {
    setHazardToggles((prev) => ({ ...prev, [type]: checked }));

    if (!mapRef.current || !hazardLayersRef.current) return;

    const layer = hazardLayersRef.current[type];
    if (checked) {
      if (!mapRef.current.hasLayer(layer)) {
        layer.addTo(mapRef.current);
      }
    } else {
      if (mapRef.current.hasLayer(layer)) {
        mapRef.current.removeLayer(layer);
      }
    }
  };

  // Handle density change
  const handleDensityChange = (factor: number) => {
    setCurrentDensity(factor);
    onDensityChange?.(factor);
  };

  // Proximity detection during navigation
  useEffect(() => {
    if (!isNavigating || !selectedRoute || !selectedRoute.hazards) return;

    const currentPos = selectedRoute.polyline[currentStep];
    if (!currentPos) return;

    selectedRoute.hazards.forEach((hazard) => {
      const distance = distanceToHazard(currentPos, hazard);
      const distanceMeters = distance * 1000;

      // Alert if within 120m and not already alerted
      if (distanceMeters <= 120 && !alertedHazardsRef.current.has(hazard.id)) {
        alertedHazardsRef.current.add(hazard.id);

        toast({
          title: `${prettyHazardType(hazard.type)} ahead`,
          description: `In ${Math.round(distanceMeters)}m (${selectedRoute.name})`,
          duration: 3000,
        });

        // TODO: Pulse the hazard marker
        // This would require storing marker refs per hazard
      }
    });

    // Reset alerted hazards if we've moved significantly
    if (currentStep === 0) {
      alertedHazardsRef.current.clear();
    }
  }, [isNavigating, selectedRoute, currentStep, toast]);

  return (
    <div className="relative w-full h-full">
      <div ref={containerRef} className="w-full h-full rounded-lg shadow-lg" />
      
      {/* Hazard Legend */}
      {routes.length > 0 && (
        <HazardLegend 
          toggleState={hazardToggles} 
          onToggle={handleHazardToggle}
          densityFactor={currentDensity}
          onDensityChange={handleDensityChange}
        />
      )}
      
      {routes.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center bg-muted/50 rounded-lg">
          <p className="text-muted-foreground text-center px-4">
            Enter your start and destination to see routes
          </p>
        </div>
      )}
    </div>
  );
};
