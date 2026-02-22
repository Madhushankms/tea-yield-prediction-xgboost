"use client";

import { useState } from "react";
import { predictYield, PredictionInput, PredictionResult } from "@/lib/api";

interface PredictionFormProps {
  onPredictionComplete: (result: PredictionResult) => void;
  onPredictionStart: () => void;
}

export default function PredictionForm({
  onPredictionComplete,
  onPredictionStart,
}: PredictionFormProps) {
  const [formData, setFormData] = useState<PredictionInput>({
    rainfall: 150,
    temperature: 25,
    fertilizer: 400,
  });
  const [error, setError] = useState<string | null>(null);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: parseFloat(value) || 0,
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    onPredictionStart();

    try {
      const result = await predictYield(formData);
      onPredictionComplete(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prediction failed");
      onPredictionComplete({
        prediction: 0,
        feature_importance: {},
      });
    }
  };

  const inputFields = [
    {
      name: "rainfall",
      label: "Rainfall (mm)",
      min: 0,
      max: 500,
      step: 1,
      icon: "🌧️",
      description: "Annual rainfall in millimeters",
    },
    {
      name: "temperature",
      label: "Temperature (°C)",
      min: 0,
      max: 50,
      step: 0.1,
      icon: "🌡️",
      description: "Average temperature in Celsius",
    },
    {
      name: "fertilizer",
      label: "Fertilizer (kg)",
      min: 0,
      max: 1000,
      step: 1,
      icon: "🧪",
      description: "Fertilizer amount in kilograms",
    },
  ];

  return (
    <div className="bg-white rounded-2xl shadow-xl p-8">
      <h2 className="text-2xl font-bold text-gray-900 mb-6">
        Enter Farm Parameters
      </h2>

      {error && (
        <div className="mb-6 bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-start">
            <span className="text-red-500 text-xl mr-3">⚠️</span>
            <div>
              <h3 className="text-sm font-medium text-red-800">
                Prediction Error
              </h3>
              <p className="text-sm text-red-700 mt-1">{error}</p>
            </div>
          </div>
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-5">
        {inputFields.map((field) => (
          <div key={field.name}>
            <label className="flex items-center text-sm font-medium text-gray-700 mb-2">
              <span className="text-lg mr-2">{field.icon}</span>
              {field.label}
            </label>
            <input
              type="number"
              name={field.name}
              value={formData[field.name as keyof PredictionInput]}
              onChange={handleInputChange}
              min={field.min}
              max={field.max}
              step={field.step}
              required
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent transition-all duration-200 hover:border-gray-400"
              title={field.description}
            />
            <p className="text-xs text-gray-500 mt-1">{field.description}</p>
          </div>
        ))}

        <button
          type="submit"
          className="w-full bg-gradient-to-r from-green-600 to-emerald-600 text-white font-semibold py-4 px-6 rounded-lg hover:from-green-700 hover:to-emerald-700 transform hover:scale-[1.02] transition-all duration-200 shadow-lg hover:shadow-xl"
        >
          Predict Tea Yield
        </button>
      </form>

      <div className="mt-6 pt-6 border-t border-gray-200">
        <p className="text-xs text-gray-500 text-center">
          All parameters are required for accurate prediction
        </p>
      </div>
    </div>
  );
}
