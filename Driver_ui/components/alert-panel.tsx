"use client"

import { AlertCircle, MapPin, Zap } from "lucide-react"

interface AlertPanelProps {
  isOnline: boolean
  hazards: any[]
  onVerify: () => void
}

export function AlertPanel({ isOnline, hazards, onVerify }: AlertPanelProps) {
  // Mock hazard data
  const mockHazards = [
    {
      id: 1,
      type: "Road Obstruction",
      severity: "high",
      distance: "0.5 km",
      location: "I-95 North",
      timestamp: "2 min ago",
    },
    {
      id: 2,
      type: "Accident Scene",
      severity: "critical",
      distance: "1.2 km",
      location: "Exit 45",
      timestamp: "5 min ago",
    },
    {
      id: 3,
      type: "Weather Alert",
      severity: "medium",
      distance: "3.1 km",
      location: "Route 22",
      timestamp: "10 min ago",
    },
  ]

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-3 sm:p-4 flex flex-col gap-3 sm:gap-4 overflow-y-auto max-h-64 sm:max-h-none">
      <div className="flex items-center gap-2">
        <AlertCircle className="w-5 h-5 text-orange-500 flex-shrink-0" />
        <h2 className="text-base sm:text-lg font-semibold text-gray-900">Active Hazards</h2>
        <span className="ml-auto text-xs bg-orange-500/20 text-orange-600 px-2 py-1 rounded whitespace-nowrap">
          {mockHazards.length}
        </span>
      </div>

      <div className="space-y-2 sm:space-y-3">
        {mockHazards.map((hazard) => (
          <div
            key={hazard.id}
            className="bg-gray-50 border border-gray-300 rounded-lg p-2 sm:p-3 hover:border-orange-500/50 transition"
          >
            <div className="flex items-start gap-3">
              <div
                className={`w-2 h-2 mt-1.5 rounded-full flex-shrink-0 ${
                  hazard.severity === "critical"
                    ? "bg-red-500"
                    : hazard.severity === "high"
                      ? "bg-orange-500"
                      : "bg-yellow-500"
                }`}
              />
              <div className="flex-1 min-w-0">
                <h3 className="font-semibold text-gray-900 text-sm">{hazard.type}</h3>
                <div className="flex items-center gap-1 text-xs text-gray-600 mt-1">
                  <MapPin className="w-3 h-3 flex-shrink-0" />
                  <span className="truncate">{hazard.location}</span>
                </div>
                <div className="flex items-center justify-between gap-2 mt-2 flex-wrap">
                  <span className="text-xs text-gray-500">
                    {hazard.distance} • {hazard.timestamp}
                  </span>
                  <button
                    onClick={onVerify}
                    className="text-xs px-2 py-1 bg-orange-500/20 hover:bg-orange-500/30 text-orange-600 rounded transition whitespace-nowrap"
                  >
                    Verify
                  </button>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div
        className={`p-3 rounded-lg flex items-start gap-2 ${
          isOnline ? "bg-green-500/10 border border-green-500/30" : "bg-red-500/10 border border-red-500/30"
        }`}
      >
        <Zap className={`w-4 h-4 mt-0.5 flex-shrink-0 ${isOnline ? "text-green-600" : "text-red-600"}`} />
        <div>
          <p className={`text-sm font-medium ${isOnline ? "text-green-700" : "text-red-700"}`}>
            {isOnline ? "Connected" : "Offline Mode"}
          </p>
          <p className="text-xs text-gray-600 mt-1">
            {isOnline ? "Real-time updates active" : "Using cached hazard data"}
          </p>
        </div>
      </div>
    </div>
  )
}
