import { useState, useEffect } from 'react';
import { Brain, Star, ShoppingCart, Info, Sparkles, Zap, Target } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { apiClient } from '@/lib/api';
import { Recommendation } from '@/types';

export const RecommendationShowcase = () => {
  const [selectedUserId, setSelectedUserId] = useState<number>(1);
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isApiConnected, setIsApiConnected] = useState(false);

  // Check API health on component mount
  useEffect(() => {
    const checkApiHealth = async () => {
      const isHealthy = await apiClient.checkHealth();
      setIsApiConnected(isHealthy);
    };
    checkApiHealth();
  }, []);

  // Fetch recommendations when user changes
  useEffect(() => {
    const fetchRecommendations = async () => {
      setIsLoading(true);
      try {
        console.log('Fetching recommendations for user:', selectedUserId);
        const recs = await apiClient.getRecommendations(selectedUserId, 3);
        console.log('Received recommendations:', recs);
        setRecommendations(recs);
      } catch (error) {
        console.error('Failed to fetch recommendations:', error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchRecommendations();
  }, [selectedUserId]);

  const RecommendationCard = ({ recommendation }: { recommendation: Recommendation }) => (
    <div className="bg-white rounded-2xl shadow-lg hover:shadow-xl transition-all duration-300 group hover:-translate-y-1 border border-border overflow-hidden">
      {/* AI Confidence Badge */}
      <div className="absolute top-4 left-4 z-10 bg-success text-success-foreground px-3 py-1 rounded-full text-xs font-semibold">
        AI: {Math.round(recommendation.confidence * 100)}%
      </div>

      {/* Product Image with Overlay */}
      <div className="relative overflow-hidden">
        <img
          src={recommendation.image}
          alt={recommendation.name}
          className="w-full h-64 object-cover group-hover:scale-105 transition-transform duration-300"
          onError={(e) => {
            const target = e.target as HTMLImageElement;
            target.src = 'https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=500&h=500&fit=crop';
          }}
        />
        
        {/* Hover Overlay with AI Reasoning */}
        <div className="absolute inset-0 bg-black/80 opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-center justify-center p-6">
          <div className="text-center">
            <Brain className="w-8 h-8 text-brand-primary mx-auto mb-2" />
            <p className="text-white text-sm leading-relaxed">{recommendation.reason}</p>
          </div>
        </div>
      </div>

      {/* Product Details */}
      <div className="p-6">
        {/* Brand & Category */}
        <div className="flex items-center justify-between mb-3">
          <span className="text-xs font-medium text-brand-primary bg-brand-primary/10 px-2 py-1 rounded-full">
            {recommendation.brand}
          </span>
          <span className="text-xs text-muted-foreground">{recommendation.category}</span>
        </div>

        {/* Product Name */}
        <h3 className="text-lg font-semibold text-foreground mb-2 line-clamp-2">
          {recommendation.name}
        </h3>

        {/* Rating */}
        <div className="flex items-center mb-3">
          <div className="flex items-center space-x-1">
            {[...Array(5)].map((_, i) => (
              <Star
                key={i}
                className={`w-4 h-4 ${
                  i < Math.floor(recommendation.rating) 
                    ? 'text-yellow-400 fill-current' 
                    : 'text-muted-foreground'
                }`}
              />
            ))}
          </div>
          <span className="text-sm text-muted-foreground ml-2">({recommendation.rating})</span>
        </div>

        {/* Price */}
        <div className="text-2xl font-bold text-foreground mb-4">
          ${recommendation.price}
        </div>

        {/* AI Confidence Bar */}
        <div className="mb-4">
          <div className="flex items-center justify-between text-xs text-muted-foreground mb-1">
            <span>AI Confidence</span>
            <span>{Math.round(recommendation.confidence * 100)}%</span>
          </div>
          <div className="w-full bg-muted rounded-full h-2">
            <div
              className="bg-gradient-to-r from-brand-primary to-brand-secondary h-2 rounded-full transition-all duration-500"
              style={{ width: `${recommendation.confidence * 100}%` }}
            />
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex space-x-3">
          <Button variant="gradient" className="flex-1">
            <ShoppingCart className="w-4 h-4 mr-2" />
            Add to Cart
          </Button>
          <Button variant="outline" size="icon">
            <Info className="w-4 h-4" />
          </Button>
        </div>
      </div>
    </div>
  );

  return (
    <section className="py-20 bg-muted/50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Section Header */}
        <div className="text-center mb-16">
          <div className="inline-flex items-center space-x-2 bg-white border border-brand-primary/20 rounded-full px-4 py-2 mb-6">
            <Brain className="w-4 h-4 text-brand-primary" />
            <span className="text-sm font-medium text-brand-primary">AI-Powered Recommendations</span>
          </div>
          
          <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-4">
            Personalized Just for You
          </h2>
          
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto mb-8">
            Select a user profile to see how our AI recommends products based on individual preferences and behavior
          </p>

          {/* API Status Indicator */}
          <div className="flex items-center justify-center space-x-2 mb-8">
            <div className={`w-2 h-2 rounded-full ${isApiConnected ? 'bg-success' : 'bg-warning'}`} />
            <span className="text-sm text-muted-foreground">
              {isApiConnected ? 'API Connected' : 'Using Mock Data'}
            </span>
          </div>

          {/* User Selection */}
          <div className="flex flex-wrap items-center justify-center gap-3 mb-12">
            {[1, 2, 3, 4, 5].map((userId) => (
              <Button
                key={userId}
                variant={selectedUserId === userId ? "gradient" : "outline"}
                size="lg"
                onClick={() => setSelectedUserId(userId)}
                className={`${
                  selectedUserId === userId 
                    ? 'shadow-brand' 
                    : 'border-brand-primary text-brand-primary hover:bg-brand-primary hover:text-white'
                }`}
              >
                User {userId}
              </Button>
            ))}
          </div>
        </div>

        {/* Recommendations Grid */}
        {isLoading ? (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-16">
            {[...Array(3)].map((_, i) => (
              <div key={`loading-${i}`} className="bg-white rounded-2xl border border-border p-6 animate-pulse">
                <div className="w-full h-64 bg-muted rounded-xl mb-4" />
                <div className="space-y-3">
                  <div className="h-4 bg-muted rounded w-3/4" />
                  <div className="h-4 bg-muted rounded w-1/2" />
                  <div className="h-6 bg-muted rounded w-1/3" />
                </div>
              </div>
            ))}
          </div>
        ) : (
          <>
            {/* Debug Info */}
            <div className="mb-8 p-4 bg-muted rounded-lg">
              <p className="text-sm text-muted-foreground">
                <strong>Debug:</strong> User {selectedUserId} - {recommendations.length} recommendations loaded
              </p>
              <p className="text-xs text-muted-foreground mt-1">
                API Status: {isApiConnected ? 'Connected' : 'Mock Data'} | 
                Loading: {isLoading ? 'Yes' : 'No'}
              </p>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-16">
              {recommendations.map((recommendation) => (
                <RecommendationCard key={recommendation.id} recommendation={recommendation} />
              ))}
            </div>
          </>
        )}

        {/* How It Works */}
        <div className="bg-white rounded-3xl p-8 md:p-12 border border-border">
          <div className="text-center mb-12">
            <h3 className="text-2xl md:text-3xl font-bold text-foreground mb-4">
              How Our AI Works
            </h3>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              Our recommendation engine uses advanced machine learning to understand your preferences
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="text-center">
              <div className="w-16 h-16 gradient-bg rounded-full flex items-center justify-center mb-6 mx-auto">
                <Zap className="w-8 h-8 text-white" />
              </div>
              <h4 className="text-xl font-semibold text-foreground mb-3">Data Analysis</h4>
              <p className="text-muted-foreground">
                We analyze your browsing history, purchases, and preferences to understand your unique taste
              </p>
            </div>

            <div className="text-center">
              <div className="w-16 h-16 gradient-bg rounded-full flex items-center justify-center mb-6 mx-auto">
                <Brain className="w-8 h-8 text-white" />
              </div>
              <h4 className="text-xl font-semibold text-foreground mb-3">AI Processing</h4>
              <p className="text-muted-foreground">
                Our neural networks process millions of data points to find patterns and connections
              </p>
            </div>

            <div className="text-center">
              <div className="w-16 h-16 gradient-bg rounded-full flex items-center justify-center mb-6 mx-auto">
                <Target className="w-8 h-8 text-white" />
              </div>
              <h4 className="text-xl font-semibold text-foreground mb-3">Perfect Match</h4>
              <p className="text-muted-foreground">
                Get personalized recommendations with confidence scores showing how well they match you
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};