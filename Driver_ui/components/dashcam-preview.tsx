"use client"

import type React from "react"

import { useEffect, useRef, useState } from "react"
import { ChevronDown, ChevronUp, GripHorizontal } from "lucide-react"

interface DashcamPreviewProps {
  isRecording?: boolean
}

export function DashcamPreview({ isRecording = true }: DashcamPreviewProps) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const windowRef = useRef<HTMLDivElement>(null)
  const [resolution, setResolution] = useState("1080p")
  const [isMinimized, setIsMinimized] = useState(false)
  const [position, setPosition] = useState({ x: 20, y: 80 })
  const [isDragging, setIsDragging] = useState(false)
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 })

  useEffect(() => {
    const accessCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "environment" },
        })
        if (videoRef.current) {
          videoRef.current.srcObject = stream
        }
      } catch (error) {
        console.error("Camera access denied:", error)
      }
    }

    accessCamera()

    return () => {
      if (videoRef.current?.srcObject) {
        const tracks = (videoRef.current.srcObject as MediaStream).getTracks()
        tracks.forEach((track) => track.stop())
      }
    }
  }, [])

  useEffect(() => {
    if (!isDragging) return

    const handleMouseMove = (e: MouseEvent) => {
      const newX = e.clientX - dragOffset.x
      const newY = e.clientY - dragOffset.y
      setPosition({ x: newX, y: newY })
    }

    const handleMouseUp = () => {
      setIsDragging(false)
    }

    window.addEventListener("mousemove", handleMouseMove)
    window.addEventListener("mouseup", handleMouseUp)

    return () => {
      window.removeEventListener("mousemove", handleMouseMove)
      window.removeEventListener("mouseup", handleMouseUp)
    }
  }, [isDragging, dragOffset])

  const handleMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!windowRef.current) return
    setIsDragging(true)
    const rect = windowRef.current.getBoundingClientRect()
    setDragOffset({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    })
  }

  return (
    <div
      ref={windowRef}
      className={`fixed bg-slate-950 border-2 border-cyan-500/40 rounded-lg overflow-hidden shadow-2xl z-40 transition-all duration-300 ${
        isMinimized ? "w-40 h-auto" : "w-80 h-48"
      }`}
      style={{
        left: `${position.x}px`,
        top: `${position.y}px`,
      }}
    >
      {/* Header with Toggle Button and Drag Handle */}
      <div
        onMouseDown={handleMouseDown}
        className="bg-gradient-to-b from-black/60 to-black/30 h-10 md:h-12 flex items-center justify-between px-3 md:px-4 cursor-move hover:bg-black/40 transition-colors"
      >
        <div className="flex items-center gap-2">
          <GripHorizontal className="w-3 h-3 md:w-4 md:h-4 text-cyan-400/60 flex-shrink-0" />
          <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
          <span className="text-green-400 font-semibold text-xs md:text-sm">LIVE</span>
        </div>

        <button
          onClick={() => setIsMinimized(!isMinimized)}
          className="pointer-events-auto p-1 hover:bg-slate-800 rounded transition-colors"
          aria-label="Toggle camera preview"
        >
          {isMinimized ? (
            <ChevronUp className="w-4 h-4 text-cyan-400" />
          ) : (
            <ChevronDown className="w-4 h-4 text-cyan-400" />
          )}
        </button>
      </div>

      {/* Video Feed - Hidden when minimized */}
      {!isMinimized && (
        <div className="relative w-full h-40 md:h-48">
          <video ref={videoRef} autoPlay playsInline className="w-full h-full object-cover" />

          {/* Overlay Controls */}
          <div className="absolute inset-0 pointer-events-none">
            {/* REC Indicator */}
            <div className="absolute top-2 right-2 flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full animate-pulse ${isRecording ? "bg-red-500" : "bg-slate-500"}`} />
              <span className={`font-semibold text-xs md:text-sm ${isRecording ? "text-red-400" : "text-slate-400"}`}>
                {isRecording ? "REC" : "STANDBY"}
              </span>
            </div>

            {/* Bottom Right Resolution */}
            <div className="absolute bottom-1 md:bottom-2 right-2 md:right-3 text-cyan-400 text-xs font-mono font-bold tracking-wider">
              {resolution}
            </div>
          </div>

          {/* Subtle Border Glow */}
          <div className="absolute inset-0 rounded-lg pointer-events-none border-2 border-cyan-500/20" />
        </div>
      )}
    </div>
  )
}
