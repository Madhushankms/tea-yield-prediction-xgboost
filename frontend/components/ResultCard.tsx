"use client";

import { PredictionResult } from "@/lib/api";
import FeatureChart from "./FeatureChart";

interface ResultCardProps {
  result: PredictionResult;
}

export default function ResultCard({ result }: ResultCardProps) {
  const { prediction, feature_importance } = result;

  // Format the prediction with comma separators
  const formattedPrediction = prediction.toLocaleString("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });

  // Calculate yield category
  const getYieldCategory = (yield_value: number) => {
    if (yield_value < 800)
      return { label: "Low Yield", color: "text-red-600", bg: "bg-red-50" };
    if (yield_value < 1200)
      return {
        label: "Average Yield",
        color: "text-yellow-600",
        bg: "bg-yellow-50",
      };
    if (yield_value < 1600)
      return {
        label: "Good Yield",
        color: "text-green-600",
        bg: "bg-green-50",
      };
    return {
      label: "Excellent Yield",
      color: "text-emerald-600",
      bg: "bg-emerald-50",
    };
  };

  const category = getYieldCategory(prediction);

  // Get top 3 influential features
  const topFeatures = Object.entries(feature_importance)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 3);

  return (
    <div className="space-y-6">
      {/* Prediction Card */}
      <div className="bg-gradient-to-br from-green-600 to-emerald-600 rounded-2xl shadow-xl p-8 text-white">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold opacity-90">Predicted Yield</h2>
          <span
            className={`px-3 py-1 rounded-full text-sm font-medium ${category.bg} ${category.color}`}
          >
            {category.label}
          </span>
        </div>

        <div className="text-center">
          <div className="text-6xl font-bold mb-2">{formattedPrediction}</div>
          <div className="text-xl opacity-90">kg</div>
        </div>

        <div className="mt-6 pt-6 border-t border-white/20">
          <h3 className="text-sm font-semibold mb-3 opacity-90">
            Top Influential Factors
          </h3>
          <div className="space-y-2">
            {topFeatures.map(([feature, importance], index) => (
              <div key={feature} className="flex items-center justify-between">
                <span className="text-sm flex items-center">
                  <span className="w-6 h-6 rounded-full bg-white/20 flex items-center justify-center text-xs font-bold mr-2">
                    {index + 1}
                  </span>
                  {feature.replace(/_/g, " ")}
                </span>
                <span className="text-sm font-medium">
                  {(importance * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Feature Importance Chart */}
      <div className="bg-white rounded-2xl shadow-xl p-8">
        <h3 className="text-xl font-bold text-gray-900 mb-6">
          Feature Importance Analysis
        </h3>
        <FeatureChart feature_importance={feature_importance} />
        <div className="mt-6 pt-6 border-t border-gray-200">
          <p className="text-sm text-gray-600">
            <strong>Interpretation:</strong> Features with higher importance
            have greater influence on the predicted yield. This analysis uses
            the XGBoost model's built-in feature importance scores.
          </p>
        </div>
      </div>

      {/* Recommendations Card */}
      <div className="bg-white rounded-2xl shadow-xl p-8">
        <h3 className="text-xl font-bold text-gray-900 mb-4">
          📊 Yield Insights
        </h3>
        <div className="space-y-3">
          {prediction < 1000 && (
            <div className="flex items-start p-3 bg-amber-50 rounded-lg">
              <span className="text-amber-600 text-xl mr-3">💡</span>
              <p className="text-sm text-gray-700">
                Consider optimizing key factors like{" "}
                {topFeatures[0][0].replace(/_/g, " ").toLowerCase()}
                to potentially improve yield.
              </p>
            </div>
          )}
          {prediction >= 1000 && prediction < 1500 && (
            <div className="flex items-start p-3 bg-blue-50 rounded-lg">
              <span className="text-blue-600 text-xl mr-3">✅</span>
              <p className="text-sm text-gray-700">
                Your parameters indicate average to good yield potential.
                Monitor {topFeatures[0][0].replace(/_/g, " ").toLowerCase()}{" "}
                closely.
              </p>
            </div>
          )}
          {prediction >= 1500 && (
            <div className="flex items-start p-3 bg-green-50 rounded-lg">
              <span className="text-green-600 text-xl mr-3">🎉</span>
              <p className="text-sm text-gray-700">
                Excellent yield prediction! Your conditions are well-optimized.
                Maintain current practices.
              </p>
            </div>
          )}

          <div className="flex items-start p-3 bg-gray-50 rounded-lg">
            <span className="text-gray-600 text-xl mr-3">ℹ️</span>
            <p className="text-sm text-gray-700">
              Typical tea yields range from 600 to 2,000 kg. Your prediction of{" "}
              {formattedPrediction} kg falls in the{" "}
              <strong>{category.label.toLowerCase()}</strong> category.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
