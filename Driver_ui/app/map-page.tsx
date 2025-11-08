
// "use client"

// import { useState } from "react"
// import dynamic from "next/dynamic"
// import { DestinationModal } from "@/components/destination-modal"

// // Dynamically load MapView client-only
// const MapView = dynamic(() => import("@/components/map-view").then((mod) => mod.MapView), {
//   ssr: false,
// })

// type LatLng = { lat: number; lng: number }

// export default function MapPage() {
//   const [showModal, setShowModal] = useState(true)
//   const [start, setStart] = useState<LatLng | null>(null)
//   const [end, setEnd] = useState<LatLng | null>(null)
//   const [route, setRoute] = useState<[number, number][] | null>(null)
//   const [hazards, setHazards] = useState<any[]>([])
//   const [loading, setLoading] = useState(false)
//   const [error, setError] = useState<string | null>(null)

//   /* --- Simple Nominatim Geocoder --- */
//   async function geocode(address: string) {
//     const params = new URLSearchParams({ q: address, format: "json", limit: "1" })
//     const res = await fetch(
//       `https://nominatim.openstreetmap.org/search?${params.toString()}`,
//       { headers: { "Accept-Language": "en" } }
//     )
//     const json = await res.json()
//     if (!json || json.length === 0) throw new Error("Address not found")
//     return { lat: parseFloat(json[0].lat), lng: parseFloat(json[0].lon) }
//   }

//   /* --- Free OSRM Routing API --- */
//   async function getRouteOSRM(s: LatLng, e: LatLng) {
//     const url = `https://router.project-osrm.org/route/v1/driving/${s.lng},${s.lat};${e.lng},${e.lat}?overview=full&geometries=geojson`
//     const res = await fetch(url)
//     const json = await res.json()
//     if (!json || json.code !== "Ok" || !json.routes || json.routes.length === 0) {
//       throw new Error("Routing failed")
//     }
//     const coords: [number, number][] = json.routes[0].geometry.coordinates.map(
//       ([lon, lat]: [number, number]) => [lat, lon]
//     )
//     return coords
//   }

//   /* --- When user submits start & end --- */
//   const handleConfirm = async (startAddr: string, endAddr: string) => {
//     setLoading(true)
//     setError(null)
//     try {
//       const s = await geocode(startAddr)
//       const e = await geocode(endAddr)
//       setStart(s)
//       setEnd(e)
//       const routeCoords = await getRouteOSRM(s, e)
//       setRoute(routeCoords)
//       setHazards([]) // Later: fetch real hazards here
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


