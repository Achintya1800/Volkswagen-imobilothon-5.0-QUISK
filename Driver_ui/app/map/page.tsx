"use client";
export const dynamic = "force-dynamic";

import { useState } from "react";
import nextDynamic from "next/dynamic";   // ✅ renamed
import { DestinationModal } from "@/components/destination-modal";

const MapView = nextDynamic(() =>
  import("@/components/map-view").then(m => m.MapView),
  { ssr: false }
);

type LatLng = { lat: number; lng: number };

export default function MapPage() {
  const [showModal, setShowModal] = useState(true);
  const [start, setStart] = useState<LatLng | null>(null);
  const [end, setEnd] = useState<LatLng | null>(null);
  const [route, setRoute] = useState<[number, number][] | null>(null);
  const [hazards] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function geocode(addr: string) {
    const url = `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(addr)}&limit=1`;
    const r = await fetch(url);
    const j = await r.json();
    if (!j[0]) throw new Error("Address not found");
    return { lat: +j[0].lat, lng: +j[0].lon };
  }

  async function getRouteOSRM(s: LatLng, e: LatLng) {
    const url = `https://router.project-osrm.org/route/v1/driving/${s.lng},${s.lat};${e.lng},${e.lat}?overview=full&geometries=geojson`;
    const r = await fetch(url);
    const j = await r.json();
    return j.routes[0].geometry.coordinates.map(([lon, lat]: any) => [lat, lon]);
  }

  const handleConfirm = async (s: string, e: string) => {
    setLoading(true);
    try {
      const S = await geocode(s);
      const E = await geocode(e);
      setStart(S); 
      setEnd(E);
      setRoute(await getRouteOSRM(S, E));
      setShowModal(false);
    } catch (err: any) {
      setError(err.message);
    }
    setLoading(false);
  };

  return (
    <div className="w-full h-screen relative">
      <div className="absolute inset-0 z-0">
        <MapView start={start} end={end} routeGeo={route} hazards={hazards} />
      </div>

      <div className="relative z-10">
        {showModal && <DestinationModal onConfirm={handleConfirm} />}

        {loading && (
          <div className="absolute top-4 right-4 bg-white p-2 rounded shadow">
            Routing...
          </div>
        )}

        {error && (
          <div className="absolute top-4 left-4 bg-red-100 text-red-700 p-2 rounded">
            {error}
          </div>
        )}
      </div>
    </div>
  );
}
