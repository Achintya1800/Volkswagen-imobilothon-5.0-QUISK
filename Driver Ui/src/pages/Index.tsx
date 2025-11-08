import { useState } from "react";
import { Navigation2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { RouteCard } from "@/components/RouteCard";
import { RouteMap } from "@/components/RouteMap";
import { NavigationPanel } from "@/components/NavigationPanel";
import { AutocompleteInput } from "@/components/AutocompleteInput";
import type { Route } from "@/data/mockRoutes";
import { useToast } from "@/hooks/use-toast";
import { getRoutes } from "@/services/routing";
import type { GeocodingResult } from "@/services/geocoding";

const Index = () => {
  const [originText, setOriginText] = useState("");
  const [destinationText, setDestinationText] = useState("");
  const [originCoords, setOriginCoords] = useState<[number, number] | null>(null);
  const [destinationCoords, setDestinationCoords] = useState<[number, number] | null>(null);
  const [routes, setRoutes] = useState<Route[]>([]);
  const [routeSteps, setRouteSteps] = useState<string[][]>([]);
  const [selectedRoute, setSelectedRoute] = useState<Route | null>(null);
  const [hoveredRouteId, setHoveredRouteId] = useState<string | null>(null);
  const [previewRouteId, setPreviewRouteId] = useState<string | null>(null);
  const [isNavigating, setIsNavigating] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [densityFactor, setDensityFactor] = useState<number>(1.5);
  const { toast } = useToast();

  const handleOriginSelect = (result: GeocodingResult) => {
    setOriginText(result.display_name);
    setOriginCoords([parseFloat(result.lat), parseFloat(result.lon)]);
  };

  const handleDestinationSelect = (result: GeocodingResult) => {
    setDestinationText(result.display_name);
    setDestinationCoords([parseFloat(result.lat), parseFloat(result.lon)]);
  };

  const handleMarkerDrag = async (isOrigin: boolean, lat: number, lng: number) => {
    if (isOrigin) {
      setOriginCoords([lat, lng]);
    } else {
      setDestinationCoords([lat, lng]);
    }

    // Auto re-route if both markers are set
    if (originCoords && destinationCoords) {
      const [startLat, startLng] = isOrigin ? [lat, lng] : originCoords;
      const [endLat, endLng] = isOrigin ? destinationCoords : [lat, lng];
      await fetchRoutes(startLng, startLat, endLng, endLat);
    }
  };

  const fetchRoutes = async (
    startLon: number,
    startLat: number,
    endLon: number,
    endLat: number,
    density: number = densityFactor
  ) => {
    setIsLoading(true);
    try {
      const { routes: fetchedRoutes, steps } = await getRoutes(
        startLon,
        startLat,
        endLon,
        endLat,
        density
      );

      setRoutes(fetchedRoutes);
      setRouteSteps(steps);
      setSelectedRoute(fetchedRoutes[0]);

      if (fetchedRoutes.length === 1) {
        toast({
          title: "Only one route found",
          description: "Only one good route found for this journey",
        });
      } else {
        toast({
          title: "Routes loaded",
          description: `Found ${fetchedRoutes.length} routes for your journey`,
        });
      }
    } catch (error) {
      console.error("Routing error:", error);
      toast({
        title: "Error",
        description: "Failed to load routes. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleGetRoutes = async () => {
    if (!originCoords || !destinationCoords) {
      toast({
        title: "Missing information",
        description: "Please select both starting location and destination from the suggestions",
        variant: "destructive",
      });
      return;
    }

    await fetchRoutes(
      originCoords[1],
      originCoords[0],
      destinationCoords[1],
      destinationCoords[0]
    );
  };

  const handleSelectRoute = (route: Route) => {
    setSelectedRoute(route);
  };

  const handlePreview = (route: Route) => {
    setSelectedRoute(route);
    setPreviewRouteId(route.id);
    toast({
      title: "Route preview",
      description: `Previewing ${route.name}`,
    });
  };

  const handleStart = (route: Route) => {
    setSelectedRoute(route);
    setIsNavigating(true);
    setPreviewRouteId(null);
    toast({
      title: "Navigation started",
      description: `Following ${route.name}`,
    });
  };

  const handleCardHover = (route: Route, hovering: boolean) => {
    setHoveredRouteId(hovering ? route.id : null);
  };

  const handleEndNavigation = () => {
    setIsNavigating(false);
    toast({
      title: "Navigation ended",
      description: "You've exited navigation mode",
    });
  };

  return (
    <div className="h-screen flex flex-col bg-background">
      {/* Top App Bar */}
      <header className="bg-card border-b border-border shadow-sm">
        <div className="px-6 py-4">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <Navigation2 className="w-6 h-6 text-primary" />
              <h1 className="text-2xl font-bold text-foreground">
                RouteSense
              </h1>
            </div>
            <div className="flex-1 flex items-center gap-3 max-w-3xl">
              <AutocompleteInput
                placeholder="Starting location"
                value={originText}
                onChange={setOriginText}
                onSelect={handleOriginSelect}
                disabled={isNavigating}
              />
              <AutocompleteInput
                placeholder="Destination"
                value={destinationText}
                onChange={setDestinationText}
                onSelect={handleDestinationSelect}
                disabled={isNavigating}
              />
              <Button
                onClick={handleGetRoutes}
                disabled={isLoading || isNavigating}
                className="px-6"
              >
                {isLoading ? "Loading..." : "Give routes"}
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Map Section */}
        <div className="flex-1 p-4">
          <RouteMap
            routes={routes}
            selectedRoute={selectedRoute}
            hoveredRouteId={hoveredRouteId}
            previewRouteId={previewRouteId}
            isNavigating={isNavigating}
            origin={originCoords}
            destination={destinationCoords}
            onMarkerDrag={handleMarkerDrag}
            densityFactor={densityFactor}
            onDensityChange={setDensityFactor}
          />
        </div>

        {/* Routes/Navigation Panel */}
        <div className="w-80 border-l border-border bg-card overflow-hidden">
          {isNavigating && selectedRoute ? (
            <NavigationPanel
              routeName={selectedRoute.name}
              steps={routeSteps[routes.indexOf(selectedRoute)] || []}
              onEnd={handleEndNavigation}
            />
          ) : (
            <div className="h-full overflow-y-auto p-4 space-y-3">
              {routes.length === 0 ? (
                <div className="flex items-center justify-center h-full text-center text-muted-foreground px-4">
                  <p>Enter your start and destination to see routes</p>
                </div>
              ) : (
                <>
                  <h2 className="font-semibold text-lg text-foreground mb-4">
                    Available Routes
                  </h2>
                  {routes.map((route) => (
                    <RouteCard
                      key={route.id}
                      route={route}
                      isSelected={selectedRoute?.id === route.id}
                      onSelect={() => handleSelectRoute(route)}
                      onPreview={() => handlePreview(route)}
                      onStart={() => handleStart(route)}
                      onHover={(hovering) => handleCardHover(route, hovering)}
                    />
                  ))}
                </>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Index;
