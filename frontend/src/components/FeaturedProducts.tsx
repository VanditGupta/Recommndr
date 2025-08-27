import { useState } from 'react';
import { Star, Heart, ShoppingCart, ArrowRight } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { apiClient } from '@/lib/api';
import { Product } from '@/types';

export const FeaturedProducts = () => {
  const [favorites, setFavorites] = useState<Set<number>>(new Set());
  const products = apiClient.getFeaturedProducts();

  const toggleFavorite = (productId: number) => {
    setFavorites(prev => {
      const newFavorites = new Set(prev);
      if (newFavorites.has(productId)) {
        newFavorites.delete(productId);
      } else {
        newFavorites.add(productId);
      }
      return newFavorites;
    });
  };

  const ProductCard = ({ product }: { product: Product }) => (
    <div className="bg-white rounded-2xl shadow-md hover:shadow-xl transition-all duration-300 group hover:-translate-y-1 border border-border">
      {/* Product Image */}
      <div className="relative overflow-hidden rounded-t-2xl">
        <img
          src={product.image}
          alt={product.name}
          className="w-full h-64 object-cover group-hover:scale-105 transition-transform duration-300"
          onError={(e) => {
            const target = e.target as HTMLImageElement;
            target.src = 'https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=500&h=500&fit=crop';
          }}
        />
        <Button
          variant="ghost"
          size="icon"
          className={`absolute top-3 right-3 bg-white/90 backdrop-blur-sm hover:bg-white ${
            favorites.has(product.id) ? 'text-red-500' : 'text-muted-foreground'
          }`}
          onClick={() => toggleFavorite(product.id)}
        >
          <Heart className={`w-5 h-5 ${favorites.has(product.id) ? 'fill-current' : ''}`} />
        </Button>
        
        {/* Discount Badge */}
        <div className="absolute top-3 left-3 bg-destructive text-destructive-foreground px-2 py-1 rounded-full text-xs font-semibold">
          -20%
        </div>
      </div>

      {/* Product Info */}
      <div className="p-6">
        {/* Brand & Category */}
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-medium text-brand-primary bg-brand-primary/10 px-2 py-1 rounded-full">
            {product.brand}
          </span>
          <span className="text-xs text-muted-foreground">{product.category}</span>
        </div>

        {/* Product Name */}
        <h3 className="text-lg font-semibold text-foreground mb-2 line-clamp-2 group-hover:text-brand-primary transition-colors">
          {product.name}
        </h3>

        {/* Rating */}
        <div className="flex items-center mb-3">
          <div className="flex items-center space-x-1">
            {[...Array(5)].map((_, i) => (
              <Star
                key={i}
                className={`w-4 h-4 ${
                  i < Math.floor(product.rating) 
                    ? 'text-yellow-400 fill-current' 
                    : 'text-muted-foreground'
                }`}
              />
            ))}
          </div>
          <span className="text-sm text-muted-foreground ml-2">({product.rating})</span>
        </div>

        {/* Price */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-2">
            <span className="text-2xl font-bold text-foreground">${product.price}</span>
            <span className="text-sm text-muted-foreground line-through">${(product.price * 1.25).toFixed(2)}</span>
          </div>
        </div>

        {/* Add to Cart Button */}
        <Button variant="gradient" className="w-full">
          <ShoppingCart className="w-4 h-4 mr-2" />
          Add to Cart
        </Button>
      </div>
    </div>
  );

  return (
    <section className="py-20 bg-background">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Section Header */}
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-4">
            Featured Products
          </h2>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Discover our hand-picked selection of trending products with amazing deals
          </p>
        </div>

        {/* Products Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 mb-12">
          {products.slice(0, 6).map((product) => (
            <ProductCard key={product.id} product={product} />
          ))}
        </div>

        {/* View All Button */}
        <div className="text-center">
          <Button variant="outline" size="lg" className="border-brand-primary text-brand-primary hover:bg-brand-primary hover:text-white">
            View All Products
            <ArrowRight className="w-5 h-5 ml-2" />
          </Button>
        </div>
      </div>
    </section>
  );
};