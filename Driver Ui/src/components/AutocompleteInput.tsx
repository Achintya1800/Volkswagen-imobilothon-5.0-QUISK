import { useState, useEffect, useRef } from "react";
import { Input } from "@/components/ui/input";
import { geocodeLocation, type GeocodingResult } from "@/services/geocoding";
import { Card } from "@/components/ui/card";

interface AutocompleteInputProps {
  placeholder: string;
  value: string;
  onChange: (value: string) => void;
  onSelect: (result: GeocodingResult) => void;
  disabled?: boolean;
}

export const AutocompleteInput = ({
  placeholder,
  value,
  onChange,
  onSelect,
  disabled,
}: AutocompleteInputProps) => {
  const [results, setResults] = useState<GeocodingResult[]>([]);
  const [showResults, setShowResults] = useState(false);
  const timeoutRef = useRef<NodeJS.Timeout>();
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }

    if (value.trim().length < 2) {
      setResults([]);
      return;
    }

    timeoutRef.current = setTimeout(async () => {
      try {
        const geocodeResults = await geocodeLocation(value);
        setResults(geocodeResults);
        setShowResults(true);
      } catch (error) {
        console.error("Geocoding error:", error);
        setResults([]);
      }
    }, 250);

    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, [value]);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setShowResults(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  return (
    <div ref={containerRef} className="relative flex-1">
      <Input
        placeholder={placeholder}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        onFocus={() => results.length > 0 && setShowResults(true)}
      />
      {showResults && results.length > 0 && (
        <Card className="absolute z-50 w-full mt-1 max-h-60 overflow-y-auto shadow-lg">
          {results.map((result, idx) => (
            <button
              key={idx}
              className="w-full text-left px-4 py-2 hover:bg-accent/10 transition-colors text-sm"
              onClick={() => {
                onSelect(result);
                setShowResults(false);
              }}
            >
              <div className="font-medium text-foreground">{result.display_name}</div>
              {result.address && (
                <div className="text-xs text-muted-foreground">
                  {[result.address.city, result.address.state]
                    .filter(Boolean)
                    .join(", ")}
                </div>
              )}
            </button>
          ))}
        </Card>
      )}
    </div>
  );
};
