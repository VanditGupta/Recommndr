import { ArrowRight, Sparkles, Users, Package, Target } from 'lucide-react';
import { Button } from '@/components/ui/button';

export const Hero = () => {
  return (
    <section className="relative overflow-hidden bg-gradient-to-br from-background via-background to-muted py-20">
      {/* Floating Background Elements */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute top-20 left-10 w-32 h-32 bg-brand-primary/20 rounded-full blur-3xl float"></div>
        <div className="absolute top-40 right-20 w-40 h-40 bg-brand-secondary/20 rounded-full blur-3xl float-delayed"></div>
        <div className="absolute bottom-20 left-1/3 w-36 h-36 bg-brand-primary/15 rounded-full blur-3xl float-delayed-2"></div>
      </div>

      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center">
          {/* Badge */}
          <div className="inline-flex items-center space-x-2 bg-white/80 backdrop-blur-sm border border-brand-primary/20 rounded-full px-4 py-2 mb-8 animate-fade-in-up">
            <Sparkles className="w-4 h-4 text-brand-primary" />
            <span className="text-sm font-medium text-brand-primary">AI-Powered Recommendations</span>
          </div>

          {/* Main Heading */}
          <h1 className="text-4xl md:text-6xl lg:text-7xl font-bold mb-6 animate-fade-in-up">
            Discover Products{' '}
            <span className="gradient-text">You'll Love</span>
          </h1>

          {/* Subheading */}
          <p className="text-xl md:text-2xl text-muted-foreground max-w-3xl mx-auto mb-10 animate-fade-in-up">
            Our advanced AI recommendation engine learns your preferences to suggest products 
            tailored specifically for you. Shop smarter, discover better.
          </p>

          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row items-center justify-center space-y-4 sm:space-y-0 sm:space-x-6 mb-16 animate-fade-in-up">
            <Button variant="gradient" size="lg" className="text-lg px-8 py-4">
              Start Shopping
              <ArrowRight className="w-5 h-5 ml-2" />
            </Button>
            <Button variant="outline" size="lg" className="text-lg px-8 py-4 border-brand-primary text-brand-primary hover:bg-brand-primary hover:text-white">
              Learn More
            </Button>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-4xl mx-auto animate-slide-in-right">
            <div className="bg-white/80 backdrop-blur-sm rounded-2xl p-6 border border-border shadow-md hover:shadow-lg transition-all">
              <div className="flex items-center justify-center w-12 h-12 gradient-bg rounded-full mb-4 mx-auto">
                <Users className="w-6 h-6 text-white" />
              </div>
              <h3 className="text-3xl font-bold text-foreground mb-2">10K+</h3>
              <p className="text-muted-foreground">Happy Customers</p>
            </div>

            <div className="bg-white/80 backdrop-blur-sm rounded-2xl p-6 border border-border shadow-md hover:shadow-lg transition-all">
              <div className="flex items-center justify-center w-12 h-12 gradient-bg rounded-full mb-4 mx-auto">
                <Package className="w-6 h-6 text-white" />
              </div>
              <h3 className="text-3xl font-bold text-foreground mb-2">1M+</h3>
              <p className="text-muted-foreground">Products Available</p>
            </div>

            <div className="bg-white/80 backdrop-blur-sm rounded-2xl p-6 border border-border shadow-md hover:shadow-lg transition-all">
              <div className="flex items-center justify-center w-12 h-12 gradient-bg rounded-full mb-4 mx-auto">
                <Target className="w-6 h-6 text-white" />
              </div>
              <h3 className="text-3xl font-bold text-foreground mb-2">95%</h3>
              <p className="text-muted-foreground">Accuracy Rate</p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};