"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"

interface DestinationModalProps {
  onConfirm: (startLocation: string, endLocation: string) => void
}

export function DestinationModal({ onConfirm }: DestinationModalProps) {
  const [startLocation, setStartLocation] = useState("")
  const [endLocation, setEndLocation] = useState("")

  const handleConfirm = () => {
    if (startLocation.trim() && endLocation.trim()) {
      onConfirm(startLocation, endLocation)
    }
  }

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
      <div className="bg-white border border-gray-200 rounded-lg p-8 w-full max-w-md shadow-lg">
        <h2 className="text-2xl font-bold text-gray-900 mb-6">Enter Your Route</h2>

        <div className="space-y-4">
          <div>
            <Label htmlFor="start" className="text-gray-700 mb-2 block">
              Starting Location
            </Label>
            <Input
              id="start"
              placeholder="Enter start location"
              value={startLocation}
              onChange={(e) => setStartLocation(e.target.value)}
              className="bg-gray-50 border-gray-300 text-gray-900 placeholder:text-gray-400"
            />
          </div>

          <div>
            <Label htmlFor="end" className="text-gray-700 mb-2 block">
              Destination
            </Label>
            <Input
              id="end"
              placeholder="Enter destination"
              value={endLocation}
              onChange={(e) => setEndLocation(e.target.value)}
              className="bg-gray-50 border-gray-300 text-gray-900 placeholder:text-gray-400"
            />
          </div>
        </div>

        <Button
          onClick={handleConfirm}
          disabled={!startLocation.trim() || !endLocation.trim()}
          className="w-full mt-6 bg-orange-500 hover:bg-orange-600 text-white font-semibold"
        >
          Start Navigation
        </Button>
      </div>
    </div>
  )
}
