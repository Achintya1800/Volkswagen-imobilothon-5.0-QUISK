"use client"

import { useState, useEffect } from "react"
import { MapView } from "@/components/map-view"
import { AlertPanel } from "@/components/alert-panel"
import { VerificationModal } from "@/components/verification-modal"
import { DestinationModal } from "@/components/destination-modal"
import { MapPin } from "lucide-react"
import { DashcamPreview } from "@/components/dashcam-preview" // Import DashcamPreview component

export default function Home() {
  const [currentScreen, setCurrentScreen] = useState<"navigation" | "offline" | "verification" | "upload">("navigation")
  const [location, setLocation] = useState<{ lat: number; lng: number } | null>(null)
  const [isOnline, setIsOnline] = useState(true)
  const [hazards, setHazards] = useState([])
  const [showVerification, setShowVerification] = useState(false)
  const [hasSelectedRoute, setHasSelectedRoute] = useState(false)
  const [route, setRoute] = useState<{ start: string; end: string } | null>(null)

  useEffect(() => {
    if ("geolocation" in navigator) {
      navigator.geolocation.watchPosition((position) => {
        setLocation({
          lat: position.coords.latitude,
          lng: position.coords.longitude,
        })
      })
    }
  }, [])

  useEffect(() => {
    const handleOnline = () => setIsOnline(true)
    const handleOffline = () => setIsOnline(false)

    window.addEventListener("online", handleOnline)
    window.addEventListener("offline", handleOffline)

    return () => {
      window.removeEventListener("online", handleOnline)
      window.removeEventListener("offline", handleOffline)
    }
  }, [])

  const handleRouteSelection = (startLocation: string, endLocation: string) => {
    setRoute({ start: startLocation, end: endLocation })
    setHasSelectedRoute(true)
    // Call your backend API here with the route data
    // await fetch('/api/route', { method: 'POST', body: JSON.stringify({ start: startLocation, end: endLocation }) })
  }

  return (
    <main className="min-h-screen flex flex-col bg-background text-foreground">
      {!hasSelectedRoute && <DestinationModal onConfirm={handleRouteSelection} />}

      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-4 py-3 sm:px-6 sm:py-4 flex-shrink-0">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 sm:gap-4">
          {/* Route Info */}
          {route && (
            <div className="flex items-center gap-2 sm:gap-6 flex-1 min-w-0 flex-wrap sm:flex-nowrap">
              <div className="flex items-center gap-1 sm:gap-2 min-w-0">
                <MapPin className="w-4 h-4 sm:w-5 sm:h-5 text-blue-600 flex-shrink-0" />
                <span className="font-semibold text-sm sm:text-base text-gray-900 truncate">{route.start}</span>
                <span className="text-gray-400 hidden sm:inline">→</span>
                <span className="font-semibold text-sm sm:text-base text-gray-900 truncate">{route.end}</span>
              </div>
              <button className="text-orange-500 hover:text-orange-600 font-medium text-xs sm:text-sm whitespace-nowrap">
                Change
              </button>
            </div>
          )}

          {/* Status Row */}
          <div className="flex items-center gap-2 sm:gap-3 w-full sm:w-auto sm:ml-auto">
            <div className="text-right flex-1 sm:flex-none">
              <p className="text-xs text-gray-500">ETA: 2h 17m • 128 km remaining</p>
            </div>
            <span
              className={`px-2 py-1 sm:px-3 sm:py-1 rounded-full text-xs sm:text-sm font-medium whitespace-nowrap flex-shrink-0 ${isOnline ? "bg-green-100 text-green-700" : "bg-red-100 text-red-700"}`}
            >
              {isOnline ? "Online" : "Offline"}
            </span>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 flex flex-col lg:flex-row gap-3 sm:gap-4 p-3 sm:p-4 overflow-hidden">
        {/* Map */}
        <div className="flex-1 rounded-lg overflow-hidden border border-gray-200 bg-gray-100 min-h-64 sm:min-h-96">
          <MapView location={location} hazards={hazards} />
        </div>

        {/* Sidebar */}
        <div className="w-full lg:w-80 flex flex-col gap-3 sm:gap-4">
          <AlertPanel isOnline={isOnline} hazards={hazards} onVerify={() => setShowVerification(true)} />
        </div>
      </div>

      {/* Modals */}
      {showVerification && (
        <VerificationModal
          onClose={() => setShowVerification(false)}
          onSubmit={() => {
            setShowVerification(false)
          }}
        />
      )}

      {/* Dashcam Preview */}
      <div className="hidden md:block">
        <DashcamPreview isRecording={true} />
      </div>

      {/* Mobile dashcam preview */}
      <div className="md:hidden fixed bottom-4 right-4 z-40">
        <DashcamPreview isRecording={true} />
      </div>
    </main>
  )
}
