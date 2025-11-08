// "use client"

// import { useEffect, useRef } from "react"

// // We'll lazy-load leaflet only inside useEffect (browser only)
// export function MapView({
//   start,
//   end,
//   routeGeo,
//   hazards,
//   zoom = 13,
// }: {
//   start: { lat: number; lng: number } | null
//   end: { lat: number; lng: number } | null
//   routeGeo: [number, number][] | null
//   hazards: { id?: string; lat: number; lng: number; type?: string; score?: number }[]
//   zoom?: number
// }) {
//   const mapContainer = useRef<HTMLDivElement>(null)

//   useEffect(() => {
//     // Ensure this only runs client-side
//     if (typeof window === "undefined") return
//     ;(async () => {
//       const L = await import("leaflet")
//       await import("leaflet/dist/leaflet.css")

//       // Fix default icons
//       delete (L.Icon.Default.prototype as any)._getIconUrl
//       L.Icon.Default.mergeOptions({
//         iconRetinaUrl: require("leaflet/dist/images/marker-icon-2x.png"),
//         iconUrl: require("leaflet/dist/images/marker-icon.png"),
//         shadowUrl: require("leaflet/dist/images/marker-shadow.png"),
//       })

//       if (!mapContainer.current) return

//       // Initialize or reset the map
//       mapContainer.current.innerHTML = ""
//       const map = L.map(mapContainer.current).setView(
//         [start?.lat ?? 20.5937, start?.lng ?? 78.9629],
//         zoom
//       )

//       L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
//         attribution: "© OpenStreetMap contributors",
//       }).addTo(map)

//       if (start) L.marker([start.lat, start.lng]).addTo(map).bindPopup("Start")
//       if (end) L.marker([end.lat, end.lng]).addTo(map).bindPopup("Destination")

//       if (routeGeo && routeGeo.length > 0) {
//         const polyline = L.polyline(routeGeo, { weight: 6, opacity: 0.85 })
//         polyline.addTo(map)
//         map.fitBounds(polyline.getBounds(), { padding: [40, 40] })
//       }

//       hazards.forEach((h) => {
//         const marker = L.marker([h.lat, h.lng]).addTo(map)
//         marker.bindPopup(`${h.type ?? "Hazard"}<br/>score: ${h.score ?? "—"}`)
//       })
//     })()
//   }, [start, end, routeGeo, hazards, zoom])

//   return <div ref={mapContainer} className="w-full h-full" />
// }



"use client";
export const dynamic = "force-dynamic";

import { useEffect, useRef } from "react";

type LatLng = { lat: number; lng: number };

export function MapView({
  start,
  end,
  routeGeo,
  hazards = [],
}: {
  start: LatLng | null;
  end: LatLng | null;
  routeGeo: [number, number][] | null;
  hazards?: { lat?: number; lng?: number; [key: string]: any }[];
}) {
  const mapContainer = useRef<HTMLDivElement>(null);
  const mapRef = useRef<any>(null);

  useEffect(() => {
    if (typeof window === "undefined") return;

    (async () => {
      const L = (await import("leaflet")).default;
      await import("leaflet/dist/leaflet.css");

      // Fix markers
      delete (L.Icon.Default.prototype as any)._getIconUrl;
      L.Icon.Default.mergeOptions({
        iconRetinaUrl: require("leaflet/dist/images/marker-icon-2x.png"),
        iconUrl: require("leaflet/dist/images/marker-icon.png"),
        shadowUrl: require("leaflet/dist/images/marker-shadow.png"),
      });

      // Init map once
      if (!mapRef.current) {
        mapRef.current = L.map(mapContainer.current!, {
          center: [19.076, 72.8777],
          zoom: 12,
        });

        L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
          maxZoom: 19,
        }).addTo(mapRef.current);
      }

      const map = mapRef.current;

      // Clear non-tile layers
      map.eachLayer((layer: any) => {
        if (!layer._url) map.removeLayer(layer);
      });

      if (start?.lat && start?.lng) {
        L.marker([start.lat, start.lng]).addTo(map);
        map.setView([start.lat, start.lng], 13);
      }

      if (end?.lat && end?.lng) L.marker([end.lat, end.lng]).addTo(map);

      if (Array.isArray(routeGeo) && routeGeo.length > 0) {
        L.polyline(routeGeo, { weight: 4 }).addTo(map);
      }

      hazards.forEach((h) => {
        if (h.lat && h.lng) {
          L.circleMarker([h.lat, h.lng], { radius: 6 }).addTo(map);
        }
      });
    })();
  }, [start, end, routeGeo, hazards]);

  return <div ref={mapContainer} id="map" className="w-full h-full"></div>;
}
