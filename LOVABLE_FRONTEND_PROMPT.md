# 🎯 LOVABLE PROMPT FOR RECOMMNDR FRONTEND

**Create a professional, modern Next.js 14 frontend for an AI-powered e-commerce recommendation system called "Recommndr"**

## 📋 Project Requirements

**Framework & Setup:**
- Next.js 14 with App Router
- TypeScript
- Tailwind CSS for styling
- Framer Motion for animations
- Heroicons for icons

**Design & UI:**
- Professional e-commerce design (like Amazon, Shopify)
- Modern, clean aesthetic with blue-to-purple gradient theme
- Fully responsive (mobile-first approach)
- Smooth animations and hover effects

## 🔌 API Integration Details

**Backend API Base URL:** `http://localhost:8001`

**Available Endpoints:**

### 1. `GET /recommend/{user_id}?top_k={number}`
- **Purpose**: Get personalized product recommendations for a specific user
- **Parameters**: 
  - `user_id`: Integer (1-5 for testing)
  - `top_k`: Number of recommendations to return (default: 10)
- **Response**: List of recommended products with confidence scores
- **Use Case**: Display personalized recommendations in the AI Showcase section

### 2. `GET /similar_items/{item_id}?top_k={number}`
- **Purpose**: Find items similar to a given product
- **Parameters**:
  - `item_id`: Integer product ID
  - `top_k`: Number of similar items to return
- **Response**: List of similar products with similarity scores
- **Use Case**: "You might also like" sections, product detail pages

### 3. `GET /user_profile/{user_id}`
- **Purpose**: Get user profile information and purchase history
- **Parameters**: `user_id`: Integer user ID
- **Response**: User preferences, recent purchases, browsing history
- **Use Case**: User dashboard, profile pages, personalization

### 4. `GET /health`
- **Purpose**: Check if the API is running
- **Response**: Simple health status
- **Use Case**: Verify backend connectivity

**API Response Structure:**
```json
{
  "recommendations": [
    {
      "id": 1,
      "name": "Product Name",
      "category": "Electronics",
      "brand": "Premium",
      "price": 299.99,
      "rating": 4.5,
      "image": "https://images.unsplash.com/...",
      "confidence": 0.95,
      "reason": "Based on your electronics preferences"
    }
  ],
  "note": "Using mock data for frontend development"
}
```

**Current Status**: The API is running and tested, but currently returns mock data for frontend development. The frontend should be designed to work with both mock data and real API responses.

## 🏗️ Required Components

### 1. Navigation Component
- Logo: "R" in a blue-purple gradient circle + "Recommndr" text
- Search bar (centered, prominent)
- Navigation menu: Home, Categories, Deals, New Arrivals
- User icon and shopping cart with notification badge
- Mobile hamburger menu

### 2. Hero Section
- Animated gradient background (blue to purple)
- Floating blob animations (3 colored circles)
- Badge: "AI-Powered Recommendations"
- Main heading: "Discover Products You'll Love" with gradient text
- Subheading about AI recommendation engine
- Two CTA buttons: "Start Shopping" (gradient) + "Learn More" (outline)
- Stats section: 10K+ customers, 1M+ products, 95% accuracy

### 3. Featured Products Section
- Section header with title and description
- 6 product cards in responsive grid (3 columns on desktop)
- Each product card includes:
  - Product image (use Unsplash placeholder URLs)
  - Brand and category tags
  - Product name
  - Star ratings (5 stars)
  - Price with discount badges
  - Favorite/heart button
  - "Add to Cart" button (gradient)
- "View All Products" button at bottom

### 4. AI Recommendation Showcase
- Section header with "AI-Powered Recommendations" badge
- Title: "Personalized Just for You"
- **Interactive User Selection**: 5 buttons (User ID 1-5) with active state
- **API Integration**: When user selects an ID, fetch recommendations from `/recommend/{user_id}`
- 3 recommendation cards showing:
  - AI confidence score badge (green, with percentage from API)
  - Product image with hover overlay showing AI reasoning
  - Product details (brand, category, name, rating, price)
  - AI confidence bar (progress bar using confidence score)
  - Action buttons: "Add to Cart" + "Details"
- **Error Handling**: Gracefully handle API failures, fallback to mock data
- "How It Works" section explaining AI process (3 steps with icons)

### 5. Footer
- Company info with logo and description
- Newsletter signup form
- Organized link sections: Company, Support, Legal, Social
- Copyright and social links

## 🎨 Design Specifications

**Colors:**
- Primary: Blue gradient (#3B82F6 to #8B5CF6)
- Background: White, light grays (#F9FAFB, #F3F4F6)
- Text: Dark grays (#111827, #374151)
- Accents: Green for success, red for discounts

**Typography:**
- Font: Inter (Google Fonts)
- Weights: Regular, Medium, Semibold, Bold
- Responsive sizing

**Animations:**
- Framer Motion for page transitions
- Hover effects on cards and buttons
- Floating blob animations in hero
- Smooth transitions (200-300ms)

## 🔧 Technical Requirements

**File Structure:**
```
src/
├── app/
│   ├── layout.tsx
│   ├── page.tsx
│   └── globals.css
├── components/
│   ├── Navigation.tsx
│   ├── Hero.tsx
│   ├── FeaturedProducts.tsx
│   ├── RecommendationShowcase.tsx
│   ├── Footer.tsx
│   └── LoadingSpinner.tsx
├── lib/
│   └── api.ts (API utility functions)
└── types/
    └── index.ts (TypeScript interfaces)
```

**API Integration Features:**
- **API utility functions** in `lib/api.ts` for all HTTP requests
- **TypeScript interfaces** for API responses
- **Error handling** with fallback to mock data
- **Loading states** while fetching from API
- **Mock data fallback** when API is unavailable
- **Real-time user selection** that triggers API calls

**Features:**
- Responsive design (mobile, tablet, desktop)
- Smooth scrolling
- Custom scrollbar styling
- Image optimization
- SEO meta tags
- Performance optimized
- **API connectivity status indicator**

**Mock Data Fallback:**
- If API calls fail, display sample recommendation data
- Show user-friendly message about using demo data
- Maintain full functionality even without backend

## 📱 Responsive Breakpoints
- Mobile: 320px+
- Tablet: 768px+
- Desktop: 1024px+

## 🚀 Expected Outcome

A beautiful, professional e-commerce frontend that demonstrates AI recommendation capabilities with smooth animations, modern design, and excellent user experience that rivals top e-commerce platforms. The frontend should seamlessly integrate with the backend API while providing a fallback experience with mock data.

**Key Integration Points:**
- User selection triggers real API calls to `/recommend/{user_id}`
- Display real confidence scores and reasoning from API
- Handle API errors gracefully with mock data fallback
- Show loading states during API calls
- Maintain full functionality regardless of API status

---

**This enhanced prompt gives Lovable complete understanding of the API integration and what each endpoint does!** 🎉
