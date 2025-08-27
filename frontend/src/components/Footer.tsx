import { ArrowRight, Facebook, Twitter, Instagram, Linkedin, Github } from 'lucide-react';
import { Button } from '@/components/ui/button';

export const Footer = () => {
  return (
    <footer className="bg-foreground text-background py-20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Main Footer Content */}
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-12 mb-12">
          {/* Company Info */}
          <div className="lg:col-span-2">
            <div className="flex items-center space-x-3 mb-6">
              <div className="w-10 h-10 gradient-bg rounded-full flex items-center justify-center">
                <span className="text-white font-bold text-xl">R</span>
              </div>
              <span className="text-2xl font-bold">Recommndr</span>
            </div>
            <p className="text-muted-foreground mb-6 max-w-md">
              Discover products you'll love with our AI-powered recommendation engine. 
              Shop smarter, discover better, and find exactly what you're looking for.
            </p>
            
            {/* Newsletter Signup */}
            <div className="bg-white/5 rounded-2xl p-6 border border-white/10">
              <h4 className="text-lg font-semibold mb-3">Stay Updated</h4>
              <p className="text-sm text-muted-foreground mb-4">
                Get the latest deals and product recommendations delivered to your inbox.
              </p>
              <div className="flex space-x-3">
                <input
                  type="email"
                  placeholder="Enter your email"
                  className="flex-1 px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-white/60 focus:outline-none focus:ring-2 focus:ring-brand-primary focus:border-transparent"
                />
                <Button variant="gradient" size="icon">
                  <ArrowRight className="w-4 h-4" />
                </Button>
              </div>
            </div>
          </div>

          {/* Company Links */}
          <div>
            <h4 className="text-lg font-semibold mb-6">Company</h4>
            <ul className="space-y-4">
              <li><a href="#" className="text-muted-foreground hover:text-white transition-colors">About Us</a></li>
              <li><a href="#" className="text-muted-foreground hover:text-white transition-colors">Careers</a></li>
              <li><a href="#" className="text-muted-foreground hover:text-white transition-colors">Press</a></li>
              <li><a href="#" className="text-muted-foreground hover:text-white transition-colors">News</a></li>
              <li><a href="#" className="text-muted-foreground hover:text-white transition-colors">Blog</a></li>
            </ul>
          </div>

          {/* Support Links */}
          <div>
            <h4 className="text-lg font-semibold mb-6">Support</h4>
            <ul className="space-y-4">
              <li><a href="#" className="text-muted-foreground hover:text-white transition-colors">Help Center</a></li>
              <li><a href="#" className="text-muted-foreground hover:text-white transition-colors">Contact Us</a></li>
              <li><a href="#" className="text-muted-foreground hover:text-white transition-colors">Returns</a></li>
              <li><a href="#" className="text-muted-foreground hover:text-white transition-colors">Shipping</a></li>
              <li><a href="#" className="text-muted-foreground hover:text-white transition-colors">Size Guide</a></li>
            </ul>
          </div>

          {/* Legal Links */}
          <div>
            <h4 className="text-lg font-semibold mb-6">Legal</h4>
            <ul className="space-y-4">
              <li><a href="#" className="text-muted-foreground hover:text-white transition-colors">Privacy Policy</a></li>
              <li><a href="#" className="text-muted-foreground hover:text-white transition-colors">Terms of Service</a></li>
              <li><a href="#" className="text-muted-foreground hover:text-white transition-colors">Cookie Policy</a></li>
              <li><a href="#" className="text-muted-foreground hover:text-white transition-colors">GDPR</a></li>
              <li><a href="#" className="text-muted-foreground hover:text-white transition-colors">Accessibility</a></li>
            </ul>
          </div>
        </div>

        {/* Bottom Section */}
        <div className="border-t border-white/10 pt-8">
          <div className="flex flex-col md:flex-row items-center justify-between space-y-6 md:space-y-0">
            {/* Copyright */}
            <div className="text-center md:text-left">
              <p className="text-muted-foreground">
                © 2024 Recommndr. All rights reserved.
              </p>
            </div>

            {/* Social Links */}
            <div className="flex items-center space-x-4">
              <Button variant="ghost" size="icon" className="text-muted-foreground hover:text-white hover:bg-white/10">
                <Facebook className="w-5 h-5" />
              </Button>
              <Button variant="ghost" size="icon" className="text-muted-foreground hover:text-white hover:bg-white/10">
                <Twitter className="w-5 h-5" />
              </Button>
              <Button variant="ghost" size="icon" className="text-muted-foreground hover:text-white hover:bg-white/10">
                <Instagram className="w-5 h-5" />
              </Button>
              <Button variant="ghost" size="icon" className="text-muted-foreground hover:text-white hover:bg-white/10">
                <Linkedin className="w-5 h-5" />
              </Button>
              <Button variant="ghost" size="icon" className="text-muted-foreground hover:text-white hover:bg-white/10">
                <Github className="w-5 h-5" />
              </Button>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
};