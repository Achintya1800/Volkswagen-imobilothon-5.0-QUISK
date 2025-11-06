// // map-page.tsx (or e.g. app/page.tsx content)
// "use client"

// import { useState } from "react"
// import dynamic from "next/dynamic"
// import { DestinationModal } from "@/components/destination-modal"
// // import { MapView } from "@/components/map-view" // adjust import
// const MapView = dynamic(() => import("@/components/map-view").then(mod => mod.MapView), {
//   ssr: false,
// })
// type LatLng = { lat: number; lng: number }

// export default function MapPage() {
//   const [showModal, setShowModal] = useState(true)
//   const [start, setStart] = useState<LatLng | null>(null)
//   const [end, setEnd] = useState<LatLng | null>(null)
//   const [route, setRoute] = useState<[number, number][] | null>(null)
//   const [hazards, setHazards] = useState<any[]>([]) // plug your hazard feed here
//   const [loading, setLoading] = useState(false)
//   const [error, setError] = useState<string | null>(null)

//   async function geocode(address: string) {
//     // Nominatim simple geocode (no key). Respect usage policies in production.
//     const params = new URLSearchParams({ q: address, format: "json", limit: "1" })
//     const res = await fetch(`https://nominatim.openstreetmap.org/search?${params.toString()}`, {
//       headers: { "Accept-Language": "en" }
//     })
//     const json = await res.json()
//     if (!json || json.length === 0) throw new Error("Address not found")
//     const lat = parseFloat(json[0].lat)
//     const lon = parseFloat(json[0].lon)
//     return { lat, lng: lon }
//   }

//   // Replaced OSRM call with OpenRouteService using provided API key.
//   async function getRouteORS(s: LatLng, e: LatLng) {
//     // OpenRouteService Directions API (driving-car)
//     // Note: For production, store the key in environment variables (server-side)
//     const ORS_API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6Ijc4NGM1Mzk5N2NmMDQ0YTI4NWU3N2VkOWE2ZjE4OTVkIiwiaCI6Im11cm11cjY0In0="
//     const url = `https://api.openrouteservice.org/v2/directions/driving-car?start=${s.lng},${s.lat}&end=${e.lng},${e.lat}`
//     const res = await fetch(url, {
//       headers: {
//         Authorization: ORS_API_KEY,
//         "Accept": "application/json, application/geo+json, application/gpx+xml, img/png; charset=utf-8"
//       }
//     })
//     const json = await res.json()
//     if (!json || !json.features || json.features.length === 0) {
//       throw new Error("Routing failed")
//     }
//     // GeoJSON coordinates are [lon, lat]; convert to [lat, lon] for Leaflet
//     const coords: [number, number][] = (json.features[0].geometry.coordinates as [number, number][])
//       .map(([lon, lat]) => [lat, lon])
//     return coords
//   }

//   const handleConfirm = async (startAddr: string, endAddr: string) => {
//     setLoading(true)
//     setError(null)
//     try {
//       const s = await geocode(startAddr)
//       const e = await geocode(endAddr)
//       setStart(s)
//       setEnd(e)
//       const routeCoords = await getRouteORS(s, e) // <- use ORS function
//       setRoute(routeCoords)
//       // TODO: request hazard data for bounding box / along route, for now stub:
//       setHazards([]) // plug server hazards
//       setShowModal(false)
//     } catch (err: any) {
//       console.error(err)
//       setError(err?.message ?? "Unexpected error")
//     } finally {
//       setLoading(false)
//     }
//   }

//   return (
//     <div className="w-full h-screen relative">
//       {showModal && <DestinationModal onConfirm={handleConfirm} />}

//       {loading && (
//         <div className="absolute top-4 right-4 bg-white bg-opacity-90 p-2 rounded shadow">
//           Routing...
//         </div>
//       )}

//       {error && (
//         <div className="absolute top-4 left-4 bg-red-100 text-red-800 p-2 rounded">
//           Error: {error}
//         </div>
//       )}

//       <div className="w-full h-full">
//         <MapView start={start} end={end} routeGeo={route} hazards={hazards} />
//       </div>
//     </div>
//   )
// }


"use client"

import { useState } from "react"
import dynamic from "next/dynamic"
import { DestinationModal } from "@/components/destination-modal"

// Dynamically load MapView client-only
const MapView = dynamic(() => import("@/components/map-view").then((mod) => mod.MapView), {
  ssr: false,
})

type LatLng = { lat: number; lng: number }

export default function MapPage() {
  const [showModal, setShowModal] = useState(true)
  const [start, setStart] = useState<LatLng | null>(null)
  const [end, setEnd] = useState<LatLng | null>(null)
  const [route, setRoute] = useState<[number, number][] | null>(null)
  const [hazards, setHazards] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  /* --- Simple Nominatim Geocoder --- */
  async function geocode(address: string) {
    const params = new URLSearchParams({ q: address, format: "json", limit: "1" })
    const res = await fetch(
      `https://nominatim.openstreetmap.org/search?${params.toString()}`,
      { headers: { "Accept-Language": "en" } }
    )
    const json = await res.json()
    if (!json || json.length === 0) throw new Error("Address not found")
    return { lat: parseFloat(json[0].lat), lng: parseFloat(json[0].lon) }
  }

  /* --- Free OSRM Routing API --- */
  async function getRouteOSRM(s: LatLng, e: LatLng) {
    const url = `https://router.project-osrm.org/route/v1/driving/${s.lng},${s.lat};${e.lng},${e.lat}?overview=full&geometries=geojson`
    const res = await fetch(url)
    const json = await res.json()
    if (!json || json.code !== "Ok" || !json.routes || json.routes.length === 0) {
      throw new Error("Routing failed")
    }
    const coords: [number, number][] = json.routes[0].geometry.coordinates.map(
      ([lon, lat]: [number, number]) => [lat, lon]
    )
    return coords
  }

  /* --- When user submits start & end --- */
  const handleConfirm = async (startAddr: string, endAddr: string) => {
    setLoading(true)
    setError(null)
    try {
      const s = await geocode(startAddr)
      const e = await geocode(endAddr)
      setStart(s)
      setEnd(e)
      const routeCoords = await getRouteOSRM(s, e)
      setRoute(routeCoords)
      setHazards([]) // Later: fetch real hazards here
      setShowModal(false)
    } catch (err: any) {
      console.error(err)
      setError(err?.message ?? "Unexpected error")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="w-full h-screen relative">
      {showModal && <DestinationModal onConfirm={handleConfirm} />}

      {loading && (
        <div className="absolute top-4 right-4 bg-white bg-opacity-90 p-2 rounded shadow">
          Routing...
        </div>
      )}

      {error && (
        <div className="absolute top-4 left-4 bg-red-100 text-red-800 p-2 rounded">
          Error: {error}
        </div>
      )}

      <div className="w-full h-full">
        <MapView start={start} end={end} routeGeo={route} hazards={hazards} />
      </div>
    </div>
  )
}
