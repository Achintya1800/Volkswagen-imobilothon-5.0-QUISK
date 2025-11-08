export interface GeocodingResult {
  display_name: string;
  lat: string;
  lon: string;
  address?: {
    city?: string;
    state?: string;
    country?: string;
  };
}

export const geocodeLocation = async (
  query: string
): Promise<GeocodingResult[]> => {
  if (!query || query.trim().length < 2) return [];

  const response = await fetch(
    `https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(
      query
    )}&format=json&addressdetails=1&countrycodes=IN&limit=5`,
    {
      headers: {
        "User-Agent": "RouteSense/1.0",
      },
    }
  );

  if (!response.ok) {
    throw new Error("Geocoding failed");
  }

  return response.json();
}
