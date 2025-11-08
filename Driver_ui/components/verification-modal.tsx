"use client"

import { X, ThumbsUp, ThumbsDown } from "lucide-react"
import { useState } from "react"

interface VerificationModalProps {
  onClose: () => void
  onSubmit: (feedback: string) => void
}

export function VerificationModal({ onClose, onSubmit }: VerificationModalProps) {
  const [feedback, setFeedback] = useState<"correct" | "incorrect" | null>(null)
  const [comments, setComments] = useState("")

  return (
    <div className="fixed inset-0 bg-black/50 z-50 flex justify-end">
  <div className="bg-slate-900 border-l border-slate-700 h-full w-1/5 p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold text-white">Verify Hazard</h2>
          <button onClick={onClose} className="text-slate-400 hover:text-white transition">
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="bg-slate-800 rounded-lg p-4 mb-4">
          <div className="text-sm text-slate-400 mb-2">Hazard Type</div>
          <p className="text-white font-semibold">Road Obstruction - I-95 North</p>
          <p className="text-sm text-slate-400 mt-2">Detected 2 minutes ago</p>
        </div>

        <div className="mb-6">
          <p className="text-sm text-slate-400 mb-3">Is this hazard still present?</p>
          <div className="flex gap-3">
            <button
              onClick={() => setFeedback("correct")}
              className={`flex-1 py-3 rounded-lg font-medium transition flex items-center justify-center gap-2 ${
                feedback === "correct" ? "bg-green-500 text-white" : "bg-slate-800 text-slate-300 hover:bg-slate-700"
              }`}
            >
              <ThumbsUp className="w-4 h-4" />
              Yes
            </button>
            <button
              onClick={() => setFeedback("incorrect")}
              className={`flex-1 py-3 rounded-lg font-medium transition flex items-center justify-center gap-2 ${
                feedback === "incorrect" ? "bg-red-500 text-white" : "bg-slate-800 text-slate-300 hover:bg-slate-700"
              }`}
            >
              <ThumbsDown className="w-4 h-4" />
              No
            </button>
          </div>
        </div>

        <div className="mb-6">
          <label className="block text-sm text-slate-400 mb-2">Additional Comments</label>
          <textarea
            value={comments}
            onChange={(e) => setComments(e.target.value)}
            placeholder="Optional: Describe the hazard in detail..."
            className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white placeholder-slate-500 focus:outline-none focus:border-orange-500 text-sm"
            rows={3}
          />
        </div>

        <div className="flex gap-3">
          <button
            onClick={onClose}
            className="flex-1 py-2 bg-slate-800 hover:bg-slate-700 text-white rounded-lg font-medium transition"
          >
            Cancel
          </button>
          <button
            onClick={() => onSubmit(comments)}
            disabled={!feedback}
            className="flex-1 py-2 bg-orange-500 hover:bg-orange-600 disabled:bg-slate-600 disabled:cursor-not-allowed text-white rounded-lg font-medium transition"
          >
            Submit
          </button>
        </div>
      </div>
    </div>
  )
}
