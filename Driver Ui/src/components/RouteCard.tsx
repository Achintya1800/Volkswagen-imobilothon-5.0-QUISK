import { Clock, TrendingUp, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { Route } from "@/data/mockRoutes";

interface RouteCardProps {
  route: Route;
  isSelected: boolean;
  onSelect: () => void;
  onPreview: () => void;
  onStart: () => void;
  onHover?: (hovering: boolean) => void;
}

export const RouteCard = ({
  route,
  isSelected,
  onSelect,
  onPreview,
  onStart,
  onHover,
}: RouteCardProps) => {
  const trafficColors = {
    Light: "bg-route-green/20 text-route-green",
    Moderate: "bg-accent/20 text-accent",
    Heavy: "bg-destructive/20 text-destructive",
  };

  return (
    <Card
      className={`p-4 cursor-pointer transition-all duration-300 hover:shadow-lg ${
        isSelected ? "ring-2 ring-accent shadow-lg" : ""
      }`}
      onClick={onSelect}
      onMouseEnter={() => onHover?.(true)}
      onMouseLeave={() => onHover?.(false)}
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "Enter") onSelect();
      }}
      role="button"
      aria-pressed={isSelected}
    >
      <div className="space-y-3">
        <div className="flex items-start justify-between">
          <div>
            <h3 className="font-semibold text-lg text-foreground">
              {route.name}
            </h3>
            <p className="text-sm text-muted-foreground">{route.via}</p>
          </div>
          <Badge className={trafficColors[route.traffic]}>
            {route.traffic}
          </Badge>
        </div>

        <div className="flex gap-4 text-sm">
          <div className="flex items-center gap-1 text-muted-foreground">
            <Clock className="w-4 h-4" />
            <span>{route.etaMins} min</span>
          </div>
          <div className="flex items-center gap-1 text-muted-foreground">
            <TrendingUp className="w-4 h-4" />
            <span>{route.distanceKm} km</span>
          </div>
        </div>

        <div className="space-y-1">
          {route.callouts.map((callout, idx) => (
            <div
              key={idx}
              className="flex items-start gap-2 text-xs text-muted-foreground group hover:text-foreground transition-colors"
            >
              <AlertCircle className="w-3 h-3 mt-0.5 flex-shrink-0" />
              <span>{callout}</span>
            </div>
          ))}
        </div>

        <div className="flex gap-2 pt-2">
          <Button
            size="sm"
            variant="default"
            className="flex-1"
            onClick={(e) => {
              e.stopPropagation();
              onStart();
            }}
          >
            Start
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={(e) => {
              e.stopPropagation();
              onPreview();
            }}
          >
            Preview
          </Button>
        </div>
      </div>
    </Card>
  );
};
