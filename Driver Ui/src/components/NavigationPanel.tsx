import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ArrowRight, Navigation } from "lucide-react";

interface NavigationPanelProps {
  routeName: string;
  steps: string[];
  onEnd: () => void;
}

export const NavigationPanel = ({
  routeName,
  steps,
  onEnd,
}: NavigationPanelProps) => {
  return (
    <Card className="h-full overflow-hidden flex flex-col">
      <div className="p-4 bg-primary text-primary-foreground">
        <div className="flex items-center justify-between mb-2">
          <h3 className="font-semibold flex items-center gap-2">
            <Navigation className="w-5 h-5" />
            Navigating {routeName}
          </h3>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-2">
        {steps.map((turn, idx) => (
          <div
            key={idx}
            className={`flex items-start gap-3 p-3 rounded-lg transition-colors ${
              idx === 0
                ? "bg-accent/10 border border-accent"
                : "bg-secondary/50"
            }`}
          >
            <div
              className={`flex items-center justify-center w-6 h-6 rounded-full text-xs font-semibold ${
                idx === 0
                  ? "bg-accent text-accent-foreground"
                  : "bg-muted text-muted-foreground"
              }`}
            >
              {idx + 1}
            </div>
            <div className="flex-1">
              <p
                className={`text-sm ${
                  idx === 0 ? "font-semibold text-foreground" : "text-muted-foreground"
                }`}
              >
                {turn}
              </p>
              {idx === 0 && (
                <p className="text-xs text-muted-foreground mt-1">
                  In 500 meters
                </p>
              )}
            </div>
            {idx === 0 && <ArrowRight className="w-5 h-5 text-accent" />}
          </div>
        ))}
      </div>

      <div className="p-4 border-t">
        <Button
          variant="destructive"
          className="w-full"
          size="lg"
          onClick={onEnd}
        >
          End Navigation
        </Button>
      </div>
    </Card>
  );
};
