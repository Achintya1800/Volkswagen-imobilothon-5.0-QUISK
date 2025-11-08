

// "use client"

// import { useState, useEffect } from "react"
// import { MapView } from "@/components/map-view"
// import { AlertPanel } from "@/components/alert-panel"
// import { VerificationModal } from "@/components/verification-modal"
// import { DestinationModal } from "@/components/destination-modal"
// import { MapPin } from "lucide-react"
// import { DashcamPreview } from "@/components/dashcam-preview"
// import { DriverFeedbackPopup } from "@/components/DriverFeedbackPopup"

// export default function Home() {
//   const [location, setLocation] = useState<{ lat: number; lng: number } | null>(null)
//   const [isOnline, setIsOnline] = useState(true)
//   const [hazards, setHazards] = useState([
//     { id: 1, type: "Road Obstruction", location: "I-95 North", time: "2 min ago" },
//     { id: 2, type: "Accident Scene", location: "Exit 45", time: "5 min ago" },
//     { id: 3, type: "Weather Alert", location: "Route 22", time: "10 min ago" },
//   ])

//   const [selectedHazard, setSelectedHazard] = useState<null | {
//     id: number
//     type: string
//     location: string
//     time: string
//   }>(null)

//   const [hasSelectedRoute, setHasSelectedRoute] = useState(false)
//   const [route, setRoute] = useState<{ start: string; end: string } | null>(null)
//   const [showDriverPopup, setShowDriverPopup] = useState(false)

//   const [routeCoords, setRouteCoords] = useState<{
//     start: { lat: number; lng: number } | null
//     end: { lat: number; lng: number } | null
//   }>({ start: null, end: null })

//   // 🌍 Convert place names → coordinates via Nominatim
//   const handleRouteSelection = async (startLocation: string, endLocation: string) => {
//     async function geocode(place: string) {
//       try {
//         const res = await fetch(
//           https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(place)}
//         )
//         const data = await res.json()
//         if (data.length > 0) {
//           return { lat: parseFloat(data[0].lat), lng: parseFloat(data[0].lon) }
//         }
//       } catch (err) {
//         console.error("Geocode error:", err)
//       }
//       return null
//     }

//     const [startCoords, endCoords] = await Promise.all([
//       geocode(startLocation),
//       geocode(endLocation),
//     ])

//     setRoute({ start: startLocation, end: endLocation })
//     setRouteCoords({ start: startCoords, end: endCoords })
//     setHasSelectedRoute(true)

//     // 🎯 Show driver feedback popup 5s after map appears
//     setTimeout(() => setShowDriverPopup(true), 5000)
//   }

//   // 🌐 Detect online/offline
//   useEffect(() => {
//     const handleOnline = () => setIsOnline(true)
//     const handleOffline = () => setIsOnline(false)
//     window.addEventListener("online", handleOnline)
//     window.addEventListener("offline", handleOffline)
//     return () => {
//       window.removeEventListener("online", handleOnline)
//       window.removeEventListener("offline", handleOffline)
//     }
//   }, [])

//   return (
//     <main className="min-h-screen flex flex-col bg-background text-foreground">
//       {!hasSelectedRoute && <DestinationModal onConfirm={handleRouteSelection} />}

//       {/* Header */}
//       <header className="bg-white border-b border-gray-200 px-4 py-3 sm:px-6 sm:py-4 flex-shrink-0">
//         <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 sm:gap-4">
//           {route && (
//             <div className="flex items-center gap-2 sm:gap-6 flex-1 min-w-0 flex-wrap sm:flex-nowrap">
//               <div className="flex items-center gap-1 sm:gap-2 min-w-0">
//                 <MapPin className="w-4 h-4 sm:w-5 sm:h-5 text-blue-600 flex-shrink-0" />
//                 <span className="font-semibold text-sm sm:text-base text-gray-900 truncate">
//                   {route.start}
//                 </span>
//                 <span className="text-gray-400 hidden sm:inline">→</span>
//                 <span className="font-semibold text-sm sm:text-base text-gray-900 truncate">
//                   {route.end}
//                 </span>
//               </div>
//               <button
//                 className="text-orange-500 hover:text-orange-600 font-medium text-xs sm:text-sm whitespace-nowrap"
//                 onClick={() => {
//                   setHasSelectedRoute(false)
//                   setRoute(null)
//                   setRouteCoords({ start: null, end: null })
//                   setShowDriverPopup(false)
//                 }}
//               >
//                 Change
//               </button>
//             </div>
//           )}

//           {/* Connection status */}
//           <div className="flex items-center gap-2 sm:gap-3 w-full sm:w-auto sm:ml-auto">
//             <div className="text-right flex-1 sm:flex-none">
//               <p className="text-xs text-gray-500">
//                 ETA: 2h 17m • 128 km remaining
//               </p>
//             </div>
//             <span
//               className={`px-2 py-1 sm:px-3 sm:py-1 rounded-full text-xs sm:text-sm font-medium whitespace-nowrap flex-shrink-0 ${
//                 isOnline ? "bg-green-100 text-green-700" : "bg-red-100 text-red-700"
//               }`}
//             >
//               {isOnline ? "Online" : "Offline"}
//             </span>
//           </div>
//         </div>
//       </header>

//       {/* Main Content */}
//       <div className="flex-1 flex flex-col lg:flex-row gap-3 sm:gap-4 p-3 sm:p-4 overflow-hidden">
//         {hasSelectedRoute ? (
//           <div className="flex-1 rounded-lg overflow-hidden border border-gray-200 bg-gray-100 min-h-64 sm:min-h-96">
//             <MapView
//               start={routeCoords.start}
//               end={routeCoords.end}
//               routeGeo={null}
//               hazards={hazards}
//             />
//           </div>
//         ) : (
//           <div className="flex-1 flex items-center justify-center border border-dashed border-gray-300 rounded-lg text-gray-500 text-sm sm:text-base">
//             Select your start and destination to begin navigation
//           </div>
//         )}

//         {/* Sidebar */}
//         <div className="w-full lg:w-80 flex flex-col gap-3 sm:gap-4">
//           <AlertPanel
//             isOnline={isOnline}
//             hazards={hazards}
//             onVerify={(hazard) => {
//               if (!selectedHazard) setSelectedHazard(hazard)
//             }}
//           />
//         </div>
//       </div>

//       {/* Verification Modal */}
//       {selectedHazard && (
//         <VerificationModal
//           hazard={selectedHazard}
//           onClose={() => setSelectedHazard(null)}
//           onSubmit={(feedback) => {
//             console.log("Feedback for:", selectedHazard, feedback)
//             setSelectedHazard(null)
//           }}
//         />
//       )}

//       {/* 🎤 Driver Feedback Popup */}
//       {showDriverPopup && (
//         <DriverFeedbackPopup onClose={() => setShowDriverPopup(false)} />
//       )}

//       {/* Dashcam */}
//       <div className="hidden md:block">
//         <DashcamPreview isRecording={true} />
//       </div>

//       <div className="md:hidden fixed bottom-4 right-4 z-40">
//         <DashcamPreview isRecording={true} />
//       </div>
//     </main>
//   )
// }



"use client";
export const dynamic = "force-dynamic";

import { useState, useEffect } from "react";
import { MapPin } from "lucide-react";
import { MapView } from "@/components/map-view";
import { AlertPanel } from "@/components/alert-panel";
import { VerificationModal } from "@/components/verification-modal";
import { DestinationModal } from "@/components/destination-modal";
import { DashcamPreview } from "@/components/dashcam-preview";
import { DriverFeedbackPopup } from "@/components/DriverFeedbackPopup";

export default function Home() {
  const [isOnline, setIsOnline] = useState(true);
  const [hazards] = useState([]);
  const [route, setRoute] = useState<any>(null);
  const [routeCoords, setRouteCoords] = useState({ start: null, end: null });
  const [selectedHazard, setSelectedHazard] = useState(null);
  const [hasSelectedRoute, setHasSelectedRoute] = useState(false);
  const [showDriverPopup, setShowDriverPopup] = useState(false);

  async function geocode(place: string) {
    const r = await fetch(`https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(place)}`);
    const d = await r.json(); return { lat: +d[0].lat, lng: +d[0].lon };
  }

  const handleRouteSelection = async (s: string, e: string) => {
    const S = await geocode(s), E = await geocode(e);
    setRoute({ start: s, end: e });
    setRouteCoords({ start: S, end: E });
    setHasSelectedRoute(true);
    setTimeout(() => setShowDriverPopup(true), 5000);
  };

  useEffect(() => {
    const on = () => setIsOnline(true), off = () => setIsOnline(false);
    window.addEventListener("online", on); window.addEventListener("offline", off);
    return () => { window.removeEventListener("online", on); window.removeEventListener("offline", off); };
  }, []);

  return (
    <main className="min-h-screen relative">
      <div className="absolute inset-0 z-0">
        <MapView start={routeCoords.start} end={routeCoords.end} routeGeo={null} hazards={hazards} />
      </div>

      <div className="relative z-10">
        {!hasSelectedRoute && <DestinationModal onConfirm={handleRouteSelection} />}

        <header className="bg-white border-b p-4 flex justify-between">
          {route && (
            <div className="flex gap-2 items-center">
              <MapPin className="text-blue-600" />
              <span>{route.start} → {route.end}</span>
            </div>
          )}
        </header>

        <div className="p-4">
          <AlertPanel
            isOnline={isOnline}
            hazards={hazards}
            onVerify={(h) => setSelectedHazard(h)}
          />
        </div>

        {selectedHazard && (
          <VerificationModal hazard={selectedHazard} onClose={() => setSelectedHazard(null)} onSubmit={() => setSelectedHazard(null)} />
        )}

        {showDriverPopup && <DriverFeedbackPopup onClose={() => setShowDriverPopup(false)} />}

        <div className="fixed bottom-4 right-4 z-20">
          <DashcamPreview isRecording />
        </div>
      </div>
    </main>
  );
}
