import { Card } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import type { HazardToggleState } from "@/types/hazards";

interface HazardLegendProps {
  toggleState: HazardToggleState;
  onToggle: (type: keyof HazardToggleState, checked: boolean) => void;
  densityFactor: number;
  onDensityChange: (factor: number) => void;
}

const DENSITY_OPTIONS = [
  { value: 0.5, label: "0.5×" },
  { value: 1.0, label: "1×" },
  { value: 1.5, label: "1.5×" },
  { value: 2.0, label: "2×" },
];

export const HazardLegend = ({ 
  toggleState, 
  onToggle, 
  densityFactor, 
  onDensityChange 
}: HazardLegendProps) => {
  const densityIndex = DENSITY_OPTIONS.findIndex((opt) => opt.value === densityFactor);
  
  return (
    <Card className="absolute top-24 left-4 z-[1000] p-3 shadow-lg bg-card/95 backdrop-blur-sm">
      <div className="space-y-3">
        <div className="border-b pb-2">
          <h3 className="font-semibold text-sm text-foreground">Hazard Index</h3>
          <p className="text-xs text-muted-foreground mt-0.5">
            Simulated for demo
          </p>
        </div>

        <div className="space-y-2">
          {/* Pothole */}
          <div className="flex items-center gap-2">
            <Checkbox
              id="pothole-toggle"
              checked={toggleState.pothole}
              onCheckedChange={(checked) =>
                onToggle("pothole", checked as boolean)
              }
            />
            <Label
              htmlFor="pothole-toggle"
              className="flex items-center gap-2 text-sm cursor-pointer"
            >
              <div
                className="w-4 h-4 rounded-full border-2 flex-shrink-0"
                style={{
                  borderColor: "#3b82f6",
                  backgroundColor: "transparent",
                }}
              />
              <span>Potholes</span>
            </Label>
          </div>

          {/* Speed Bump */}
          <div className="flex items-center gap-2">
            <Checkbox
              id="speed_bump-toggle"
              checked={toggleState.speed_bump}
              onCheckedChange={(checked) =>
                onToggle("speed_bump", checked as boolean)
              }
            />
            <Label
              htmlFor="speed_bump-toggle"
              className="flex items-center gap-2 text-sm cursor-pointer"
            >
              <div
                className="w-4 h-3 rounded-sm flex-shrink-0"
                style={{
                  backgroundColor: "#f97316",
                  border: "1px solid white",
                }}
              />
              <span>Speed bumps</span>
            </Label>
          </div>

          {/* Debris */}
          <div className="flex items-center gap-2">
            <Checkbox
              id="debris-toggle"
              checked={toggleState.debris}
              onCheckedChange={(checked) =>
                onToggle("debris", checked as boolean)
              }
            />
            <Label
              htmlFor="debris-toggle"
              className="flex items-center gap-2 text-sm cursor-pointer"
            >
              <div
                className="w-0 h-0 flex-shrink-0"
                style={{
                  borderLeft: "6px solid transparent",
                  borderRight: "6px solid transparent",
                  borderBottom: "10px solid #a855f7",
                }}
              />
              <span>Debris</span>
            </Label>
          </div>
        </div>

        {/* Density Slider */}
        <div className="border-t pt-3 space-y-2">
          <Label className="text-xs font-medium">Demo hazard density</Label>
          <div className="flex items-center gap-3">
            <Slider
              value={[densityIndex]}
              onValueChange={(values) => {
                const newIndex = values[0];
                onDensityChange(DENSITY_OPTIONS[newIndex].value);
              }}
              max={DENSITY_OPTIONS.length - 1}
              step={1}
              className="flex-1"
            />
            <span className="text-xs font-semibold w-10 text-right">
              {DENSITY_OPTIONS[densityIndex]?.label || "1×"}
            </span>
          </div>
        </div>
      </div>
    </Card>
  );
};
