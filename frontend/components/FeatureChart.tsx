"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

interface FeatureChartProps {
  feature_importance: {
    [key: string]: number;
  };
}

export default function FeatureChart({
  feature_importance,
}: FeatureChartProps) {
  // Transform data for recharts
  const chartData = Object.entries(feature_importance)
    .map(([feature, importance]) => ({
      name: feature.replace(/_/g, " "),
      importance: importance * 100, // Convert to percentage
      fullName: feature,
    }))
    .sort((a, b) => b.importance - a.importance);

  // Color scale from light to dark green based on importance
  const getColor = (importance: number) => {
    const maxImportance = Math.max(...chartData.map((d) => d.importance));
    const intensity = importance / maxImportance;

    if (intensity > 0.8) return "#15803d"; // dark green
    if (intensity > 0.6) return "#16a34a"; // green
    if (intensity > 0.4) return "#22c55e"; // medium green
    if (intensity > 0.2) return "#4ade80"; // light green
    return "#86efac"; // very light green
  };

  // Custom tooltip
  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white px-4 py-3 rounded-lg shadow-lg border border-gray-200">
          <p className="font-semibold text-gray-900">
            {payload[0].payload.name}
          </p>
          <p className="text-sm text-gray-600 mt-1">
            Importance:{" "}
            <span className="font-medium text-green-600">
              {payload[0].value.toFixed(2)}%
            </span>
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="w-full h-96">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={chartData}
          margin={{ top: 10, right: 30, left: 20, bottom: 80 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis
            dataKey="name"
            angle={-45}
            textAnchor="end"
            height={100}
            tick={{ fill: "#4b5563", fontSize: 12 }}
          />
          <YAxis
            label={{
              value: "Importance (%)",
              angle: -90,
              position: "insideLeft",
              style: { fill: "#4b5563", fontSize: 12 },
            }}
            tick={{ fill: "#4b5563", fontSize: 12 }}
          />
          <Tooltip
            content={<CustomTooltip />}
            cursor={{ fill: "rgba(34, 197, 94, 0.1)" }}
          />
          <Bar dataKey="importance" radius={[8, 8, 0, 0]}>
            {chartData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={getColor(entry.importance)} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      <div className="mt-4 flex items-center justify-center space-x-6 text-sm">
        <div className="flex items-center">
          <div
            className="w-4 h-4 rounded"
            style={{ backgroundColor: "#15803d" }}
          ></div>
          <span className="ml-2 text-gray-600">High Impact</span>
        </div>
        <div className="flex items-center">
          <div
            className="w-4 h-4 rounded"
            style={{ backgroundColor: "#22c55e" }}
          ></div>
          <span className="ml-2 text-gray-600">Medium Impact</span>
        </div>
        <div className="flex items-center">
          <div
            className="w-4 h-4 rounded"
            style={{ backgroundColor: "#86efac" }}
          ></div>
          <span className="ml-2 text-gray-600">Low Impact</span>
        </div>
      </div>
    </div>
  );
}
