"use client"

import { useEffect, useRef, useState } from "react"

export function DriverFeedbackPopup({ onClose }: { onClose: () => void }) {
  const [recording, setRecording] = useState(false)
  const [thanked, setThanked] = useState(false)
  const [status, setStatus] = useState("Playing prompt...")
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunks = useRef<Blob[]>([])

  useEffect(() => {
    const startSequence = async () => {
      try {
        const audio = audioRef.current
        if (!audio) return

        // ✅ Step 1: Load and play pothole.mp3 once
        await audio.play().catch((err) => {
          console.warn("⚠ Audio autoplay was blocked, retrying with user gesture later:", err)
        })

        // Step 2: Wait for prompt duration (5 seconds)
        setStatus("Prompt playing...")
        await new Promise((resolve) => setTimeout(resolve, 5000))

        // Step 3: Start microphone recording for 5 seconds
        setStatus("Recording...")
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
        const recorder = new MediaRecorder(stream)
        mediaRecorderRef.current = recorder
        chunks.current = []

        recorder.ondataavailable = (e) => chunks.current.push(e.data)
        recorder.onstart = () => setRecording(true)

        recorder.start()

        setTimeout(() => {
          recorder.stop()
          stream.getTracks().forEach((t) => t.stop())
          setRecording(false)
          setThanked(true)
          setStatus("Thanks for your response!")

          const blob = new Blob(chunks.current, { type: "audio/webm" })
          console.log("🎤 Recorded driver response:", blob)

          // Auto close after 2s
          setTimeout(() => onClose(), 2000)
        }, 5000)
      } catch (err) {
        console.error("🎙 Error in playback or mic:", err)
        onClose()
      }
    }

    startSequence()
  }, [onClose])

  return (
    <div className="fixed inset-0 flex items-start justify-center z-50 pt-20 animate-fadeIn">
      {/* Background blur overlay */}
      <div className="absolute inset-0 bg-black/50 backdrop-blur-sm"></div>

      {/* Popup UI */}
      <div className="relative bg-gradient-to-b from-blue-800 to-blue-600 text-white rounded-2xl shadow-2xl p-6 w-[90%] max-w-sm text-center z-10 border border-blue-400">
        {!thanked ? (
          <>
            <h2 className="text-2xl font-semibold mb-2">🚧 Is there a pothole here?</h2>
            <p className="text-sm text-blue-100 mb-4">
              {status === "Recording..."
                ? "Please say “Yes”, “No”, or “Unsure”."
                : "Voice prompt is playing..."}
            </p>

            <div className="mt-3 mb-2 flex justify-center">
              <div
                className={`w-16 h-16 rounded-full flex items-center justify-center ${
                  recording ? "bg-green-500 animate-pulse" : "bg-blue-500"
                }`}
              >
                <span className="material-icons text-3xl">mic</span>
              </div>
            </div>

            <p className="text-xs text-blue-200 mt-3">
              {recording
                ? "Recording... will stop in 5 seconds."
                : "Preparing audio system..."}
            </p>
          </>
        ) : (
          <>
            <h2 className="text-2xl font-bold text-green-300 mb-2">
              ✅ Thanks for your response!
            </h2>
            <p className="text-blue-100">Recording stopped successfully.</p>
          </>
        )}
        <audio ref={audioRef} src="/pothole.mp3" preload="auto" />
      </div>
    </div>
  )
}