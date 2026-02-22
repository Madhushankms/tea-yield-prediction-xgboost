/**
 * API utility for communicating with FastAPI backend
 * Handles all HTTP requests for tea yield prediction
 */

import axios from "axios";

// Backend API base URL
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// Create axios instance with default config
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
  timeout: 30000, // 30 second timeout
});

/**
 * Input data structure for prediction request
 */
export interface PredictionInput {
  rainfall: number;
  temperature: number;
  fertilizer: number;
}

/**
 * Result structure from prediction response
 */
export interface PredictionResult {
  prediction: number;
  feature_importance: {
    [key: string]: number;
  };
}

/**
 * Health check response structure
 */
export interface HealthStatus {
  status: string;
  model_loaded: boolean;
}

/**
 * Check API health and model status
 * @returns Promise with health status
 */
export async function checkHealth(): Promise<HealthStatus> {
  try {
    const response = await apiClient.get<HealthStatus>("/health");
    return response.data;
  } catch (error) {
    console.error("Health check failed:", error);
    throw new Error("Unable to connect to prediction service");
  }
}

/**
 * Make tea yield prediction
 * @param input - Feature values for prediction
 * @returns Promise with prediction result
 */
export async function predictYield(
  input: PredictionInput,
): Promise<PredictionResult> {
  try {
    const response = await apiClient.post<PredictionResult>("/predict", input);
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      const errorMessage = error.response?.data?.detail || error.message;
      throw new Error(`Prediction failed: ${errorMessage}`);
    }
    throw new Error("An unexpected error occurred during prediction");
  }
}

/**
 * Get list of required features
 * @returns Promise with feature list and descriptions
 */
export async function getFeatures(): Promise<any> {
  try {
    const response = await apiClient.get("/features");
    return response.data;
  } catch (error) {
    console.error("Failed to fetch features:", error);
    throw new Error("Unable to fetch feature list");
  }
}
