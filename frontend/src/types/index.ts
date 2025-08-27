export interface Product {
  id: number;
  name: string;
  category: string;
  brand: string;
  price: number;
  rating: number;
  image: string;
  confidence?: number;
  reason?: string;
}

export interface Recommendation {
  id: number;
  name: string;
  category: string;
  brand: string;
  price: number;
  rating: number;
  image: string;
  confidence: number;
  reason: string;
}

export interface ApiResponse<T> {
  recommendations?: T[];
  similar_items?: T[];
  data?: T;
  note?: string;
}

export interface UserProfile {
  id: number;
  name: string;
  preferences: string[];
  recent_purchases: Product[];
  browsing_history: Product[];
}

export interface ApiError {
  message: string;
  status?: number;
}

export class ApiRequestError extends Error {
  status?: number;
  
  constructor(message: string, status?: number) {
    super(message);
    this.name = 'ApiRequestError';
    this.status = status;
  }
}