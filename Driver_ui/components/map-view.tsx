

// "use client"

// import { useEffect } from "react"
// import { MapContainer, TileLayer, Marker, Polyline, Popup, useMap } from "react-leaflet"
// import L from "leaflet"
// import "leaflet/dist/leaflet.css"

// interface LatLng {
//   lat: number
//   lng: number
// }

// interface MapViewProps {
//   start: LatLng | null
//   end: LatLng | null
//   routeGeo: [number, number][] | null
//   hazards: { id?: string; lat: number; lng: number; type?: string; score?: number }[]
//   zoom?: number
// }

// /* --- Fit map bounds dynamically --- */
// function FitBounds({
//   routeGeo,
//   start,
//   end,
// }: {
//   routeGeo: [number, number][] | null
//   start: LatLng | null
//   end: LatLng | null
// }) {
//   const map = useMap()
//   useEffect(() => {
//     const pts: L.LatLngExpression[] = []
//     if (start) pts.push([start.lat, start.lng])
//     if (end) pts.push([end.lat, end.lng])
//     if (routeGeo && routeGeo.length) pts.push(...routeGeo)
//     if (pts.length) {
//       const bounds = L.latLngBounds(pts)
//       map.fitBounds(bounds, { padding: [40, 40] })
//     }
//   }, [map, routeGeo, start, end])
//   return null
// }

// /* --- Fix default Leaflet icons --- */
// function fixLeafletMarkerIcon() {
//   if (typeof window === "undefined") return

//   delete (L.Icon.Default.prototype as any)._getIconUrl
//   L.Icon.Default.mergeOptions({
//     iconRetinaUrl: require("leaflet/dist/images/marker-icon-2x.png"),
//     iconUrl: require("leaflet/dist/images/marker-icon.png"),
//     shadowUrl: require("leaflet/dist/images/marker-shadow.png"),
//   })
// }

// /* --- Main Map Component --- */
// export function MapView({ start, end, routeGeo, hazards, zoom = 13 }: MapViewProps) {
//   if (typeof window === "undefined") return null
//   useEffect(() => {
//     fixLeafletMarkerIcon()
//   }, [])

//   const center: LatLng = start ?? end ?? { lat: 20.5937, lng: 78.9629 } // India fallback

//   return (
//     <div className="w-full h-full">
//       <MapContainer
//         center={[center.lat, center.lng]}
//         zoom={zoom}
//         style={{ height: "100%", width: "100%" }}
//       >
//         <TileLayer
//           attribution='© OpenStreetMap contributors'
//           url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
//         />

//         {start && (
//           <Marker position={[start.lat, start.lng]}>
//             <Popup>Start</Popup>
//           </Marker>
//         )}
//         {end && (
//           <Marker position={[end.lat, end.lng]}>
//             <Popup>Destination</Popup>
//           </Marker>
//         )}

//         {routeGeo && routeGeo.length > 0 && (
//           <Polyline positions={routeGeo} weight={6} opacity={0.85} />
//         )}

//         {hazards.map((h) => (
//           <Marker key={h.id ?? `${h.lat}-${h.lng}`} position={[h.lat, h.lng]}>
//             <Popup>
//               {h.type ?? "Hazard"} <br /> score: {h.score ?? "—"}
//             </Popup>
//           </Marker>
//         ))}

//         <FitBounds routeGeo={routeGeo} start={start} end={end} />
//       </MapContainer>
//     </div>
//   )
// }


"use client"

import { useEffect, useRef } from "react"

// We'll lazy-load leaflet only inside useEffect (browser only)
export function MapView({
  start,
  end,
  routeGeo,
  hazards,
  zoom = 13,
}: {
  start: { lat: number; lng: number } | null
  end: { lat: number; lng: number } | null
  routeGeo: [number, number][] | null
  hazards: { id?: string; lat: number; lng: number; type?: string; score?: number }[]
  zoom?: number
}) {
  const mapContainer = useRef<HTMLDivElement>(null)

  useEffect(() => {
    // Ensure this only runs client-side
    if (typeof window === "undefined") return
    ;(async () => {
      const L = await import("leaflet")
      await import("leaflet/dist/leaflet.css")

      // Fix default icons
      delete (L.Icon.Default.prototype as any)._getIconUrl
      L.Icon.Default.mergeOptions({
        iconRetinaUrl: require("leaflet/dist/images/marker-icon-2x.png"),
        iconUrl: require("leaflet/dist/images/marker-icon.png"),
        shadowUrl: require("leaflet/dist/images/marker-shadow.png"),
      })

      if (!mapContainer.current) return

      // Initialize or reset the map
      mapContainer.current.innerHTML = ""
      const map = L.map(mapContainer.current).setView(
        [start?.lat ?? 20.5937, start?.lng ?? 78.9629],
        zoom
      )

      L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        attribution: "© OpenStreetMap contributors",
      }).addTo(map)

      if (start) L.marker([start.lat, start.lng]).addTo(map).bindPopup("Start")
      if (end) L.marker([end.lat, end.lng]).addTo(map).bindPopup("Destination")

      if (routeGeo && routeGeo.length > 0) {
        const polyline = L.polyline(routeGeo, { weight: 6, opacity: 0.85 })
        polyline.addTo(map)
        map.fitBounds(polyline.getBounds(), { padding: [40, 40] })
      }

      hazards.forEach((h) => {
        const marker = L.marker([h.lat, h.lng]).addTo(map)
        marker.bindPopup(`${h.type ?? "Hazard"}<br/>score: ${h.score ?? "—"}`)
      })
    })()
  }, [start, end, routeGeo, hazards, zoom])

  return <div ref={mapContainer} className="w-full h-full" />
}
