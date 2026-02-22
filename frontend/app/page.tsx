"use client";

import { useState } from "react";
import PredictionForm from "@/components/PredictionForm";
import ResultCard from "@/components/ResultCard";
import { PredictionResult } from "@/lib/api";

export default function Home() {
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handlePredictionComplete = (predictionResult: PredictionResult) => {
    setResult(predictionResult);
    setIsLoading(false);
  };

  const handlePredictionStart = () => {
    setIsLoading(true);
    setResult(null);
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-green-50 via-emerald-50 to-teal-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold text-gray-900 mb-4">
            🍃 Tea Yield Prediction System
          </h1>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            AI-powered tea leaf yield prediction using advanced machine learning
          </p>
          <div className="mt-4 inline-flex items-center px-4 py-2 bg-green-100 rounded-full">
            <span className="w-2 h-2 bg-green-500 rounded-full mr-2 animate-pulse"></span>
            <span className="text-sm font-medium text-green-800">
              XGBoost Model Active
            </span>
          </div>
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left Column: Input Form */}
          <div>
            <PredictionForm
              onPredictionComplete={handlePredictionComplete}
              onPredictionStart={handlePredictionStart}
            />
          </div>

          {/* Right Column: Results */}
          <div>
            {isLoading && (
              <div className="bg-white rounded-2xl shadow-xl p-8">
                <div className="flex flex-col items-center justify-center space-y-4">
                  <div className="w-16 h-16 border-4 border-green-500 border-t-transparent rounded-full animate-spin"></div>
                  <p className="text-lg font-medium text-gray-700">
                    Analyzing data...
                  </p>
                  <p className="text-sm text-gray-500">
                    Processing environmental factors
                  </p>
                </div>
              </div>
            )}

            {!isLoading && result && <ResultCard result={result} />}

            {!isLoading && !result && (
              <div className="bg-white rounded-2xl shadow-xl p-8 border-2 border-dashed border-gray-300">
                <div className="text-center text-gray-500">
                  <svg
                    className="mx-auto h-24 w-24 text-gray-400 mb-4"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={1.5}
                      d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                    />
                  </svg>
                  <h3 className="text-lg font-medium text-gray-900 mb-2">
                    Enter Farm Data
                  </h3>
                  <p className="text-sm">
                    Fill in the environmental and agricultural parameters to
                    predict tea yield
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}
