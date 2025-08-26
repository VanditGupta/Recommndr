"""Data generators for synthetic e-commerce data."""

import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import pandas as pd
from faker import Faker

from config.settings import settings
from src.utils.logging import get_logger, log_data_info
from src.utils.schemas import (
    Category,
    Interaction,
    Order,
    Product,
    ProductCategory,
    SearchQuery,
    User,
    UserPreference,
)

logger = get_logger(__name__)

# Initialize Faker with seed for reproducibility
fake = Faker()
Faker.seed(settings.RANDOM_SEED)
random.seed(settings.RANDOM_SEED)


class CategoryGenerator:
    """Generate product categories with hierarchy."""
    
    def __init__(self):
        self.categories = self._define_categories()
        self.category_id_counter = 1
    
    def _define_categories(self) -> List[Dict]:
        """Define the category hierarchy."""
        return [
            {
                "name": "Electronics",
                "subcategories": [
                    "Smartphones", "Laptops", "Tablets", "Headphones", "Cameras",
                    "Gaming", "Audio", "TV & Video", "Wearables", "Accessories"
                ]
            },
            {
                "name": "Clothing",
                "subcategories": [
                    "Men's Clothing", "Women's Clothing", "Kids' Clothing",
                    "Shoes", "Bags", "Jewelry", "Watches", "Sportswear"
                ]
            },
            {
                "name": "Books",
                "subcategories": [
                    "Fiction", "Non-Fiction", "Academic", "Children's Books",
                    "Comics", "Magazines", "E-books", "Audiobooks"
                ]
            },
            {
                "name": "Home & Garden",
                "subcategories": [
                    "Furniture", "Kitchen", "Bathroom", "Garden", "Decor",
                    "Lighting", "Storage", "Tools", "Appliances"
                ]
            },
            {
                "name": "Sports & Outdoors",
                "subcategories": [
                    "Fitness", "Outdoor Sports", "Team Sports", "Water Sports",
                    "Camping", "Hiking", "Cycling", "Yoga"
                ]
            },
            {
                "name": "Beauty & Health",
                "subcategories": [
                    "Skincare", "Makeup", "Haircare", "Fragrances",
                    "Personal Care", "Health & Wellness", "Vitamins"
                ]
            },
            {
                "name": "Toys & Games",
                "subcategories": [
                    "Board Games", "Video Games", "Educational Toys",
                    "Action Figures", "Puzzles", "Arts & Crafts"
                ]
            },
            {
                "name": "Automotive",
                "subcategories": [
                    "Car Parts", "Car Care", "Motorcycle", "Truck",
                    "RV", "Boat", "Tools & Equipment"
                ]
            }
        ]
    
    def generate_categories(self) -> List[Category]:
        """Generate all categories with proper hierarchy."""
        categories = []
        
        for main_cat in self.categories:
            # Main category
            main_category = Category(
                category_id=self.category_id_counter,
                name=main_cat["name"],
                parent_category=None,
                description=f"Main category for {main_cat['name'].lower()}",
                level=1
            )
            categories.append(main_category)
            main_cat_id = self.category_id_counter
            self.category_id_counter += 1
            
            # Subcategories
            for sub_cat in main_cat["subcategories"]:
                sub_category = Category(
                    category_id=self.category_id_counter,
                    name=sub_cat,
                    parent_category=main_cat_id,
                    description=f"Subcategory of {main_cat['name']}: {sub_cat}",
                    level=2
                )
                categories.append(sub_category)
                self.category_id_counter += 1
        
        logger.info(f"Generated {len(categories)} categories")
        return categories


class UserGenerator:
    """Generate synthetic user data."""
    
    def __init__(self, categories: List[Category]):
        self.categories = categories
        self.user_id_counter = 1
        
        # Define realistic distributions
        self.age_distribution = {
            "18-25": 0.25,
            "26-35": 0.35,
            "36-45": 0.25,
            "46+": 0.15
        }
        
        self.income_distribution = {
            "low": 0.40,
            "medium": 0.35,
            "high": 0.20,
            "luxury": 0.05
        }
        
        self.locations = [
            "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
            "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose"
        ]
        
        self.devices = ["mobile", "desktop", "tablet"]
        self.languages = ["English", "Spanish", "French", "German", "Chinese"]
        self.timezones = ["EST", "CST", "MST", "PST"]
    
    def _generate_age(self) -> int:
        """Generate realistic age based on distribution."""
        age_group = random.choices(
            list(self.age_distribution.keys()),
            weights=list(self.age_distribution.values())
        )[0]
        
        if age_group == "18-25":
            return random.randint(18, 25)
        elif age_group == "26-35":
            return random.randint(26, 35)
        elif age_group == "36-45":
            return random.randint(36, 45)
        else:
            return random.randint(46, 65)
    
    def _generate_income_level(self) -> str:
        """Generate income level based on distribution."""
        return random.choices(
            list(self.income_distribution.keys()),
            weights=list(self.income_distribution.values())
        )[0]
    
    def _generate_preference_category(self) -> str:
        """Generate user preference category."""
        # Weight towards main categories
        main_categories = [cat.name for cat in self.categories if cat.level == 1]
        return random.choice(main_categories)
    
    def generate_users(self, num_users: int) -> List[User]:
        """Generate synthetic users."""
        users = []
        
        for _ in range(num_users):
            # Generate base user data
            age = self._generate_age()
            income_level = self._generate_income_level()
            preference_category = self._generate_preference_category()
            
            # Create user
            user = User(
                user_id=self.user_id_counter,
                age=age,
                gender=random.choice(["male", "female", "other"]),
                location=random.choice(self.locations),
                income_level=income_level,
                preference_category=preference_category,
                device_type=random.choice(self.devices),
                language_preference=random.choice(self.languages),
                timezone=random.choice(self.timezones),
                email=fake.email(),
                created_at=fake.date_time_between(
                    start_date="-2y",
                    end_date="-1m"
                ),
                last_active=fake.date_time_between(
                    start_date="-1m",
                    end_date="now"
                )
            )
            
            users.append(user)
            self.user_id_counter += 1
        
        logger.info(f"Generated {len(users)} users")
        return users


class ProductGenerator:
    """Generate synthetic product data."""
    
    def __init__(self, categories: List[Category]):
        self.categories = categories
        self.product_id_counter = 1
        
        # Product templates for different categories
        self.product_templates = {
            "Electronics": {
                "brands": ["Apple", "Samsung", "Sony", "LG", "Dell", "HP", "Lenovo", "Microsoft", "Google", "Amazon"],
                "price_range": (50, 2000),
                "weight_range": (0.1, 10.0),  # US pounds
                "colors": ["Black", "White", "Silver", "Gold", "Blue", "Red", "Space Gray", "Rose Gold"],
                "sizes": ["Small", "Medium", "Large", "One Size"]
            },
            "Clothing": {
                "brands": ["Nike", "Adidas", "Zara", "H&M", "Levi's", "Tommy Hilfiger", "Calvin Klein", "Ralph Lauren", "Gap", "Old Navy"],
                "price_range": (20, 500),
                "weight_range": (0.1, 3.0),  # US pounds
                "colors": ["Black", "White", "Blue", "Red", "Green", "Yellow", "Pink", "Navy", "Gray", "Beige"],
                "sizes": ["XS", "S", "M", "L", "XL", "XXL", "XXXL"]
            },
            "Books": {
                "brands": ["Penguin", "HarperCollins", "Random House", "Simon & Schuster", "Macmillan", "Hachette", "Scholastic"],
                "price_range": (10, 50),
                "weight_range": (0.1, 2.0),  # US pounds
                "colors": ["Black", "White", "Brown", "Blue", "Red", "Green"],
                "sizes": ["Paperback", "Hardcover", "E-book", "Audiobook", "Large Print"]
            }
        }
    
    def _get_category_template(self, category_name: str) -> Dict:
        """Get product template for a category."""
        for template_cat, template in self.product_templates.items():
            if template_cat.lower() in category_name.lower():
                return template
        
        # Default template
        return {
            "brands": ["Generic", "Standard", "Premium"],
            "price_range": (100, 1000),
            "weight_range": (0.1, 2.0),
            "colors": ["Black", "White", "Blue", "Red"],
            "sizes": ["Small", "Medium", "Large", "One Size"]
        }
    
    def _generate_product_name(self, category: str, subcategory: str) -> str:
        """Generate realistic product name."""
        if "Electronics" in category:
            if "Smartphone" in subcategory:
                return f"{random.choice(['iPhone', 'Samsung Galaxy', 'OnePlus'])} {random.randint(10, 15)}"
            elif "Laptop" in subcategory:
                return f"{random.choice(['MacBook', 'ThinkPad', 'Inspiron'])} {random.choice(['Pro', 'Air', 'Standard'])}"
        
        # Generic product name
        adjectives = ["Premium", "Professional", "Advanced", "Classic", "Modern"]
        nouns = ["Device", "Product", "Item", "Equipment", "Tool"]
        
        return f"{random.choice(adjectives)} {subcategory} {random.choice(nouns)}"
    
    def _generate_description(self, name: str, category: str, subcategory: str) -> str:
        """Generate product description."""
        descriptions = [
            f"High-quality {subcategory.lower()} with excellent features and durability.",
            f"Premium {subcategory.lower()} designed for modern users who demand the best.",
            f"Professional-grade {subcategory.lower()} perfect for everyday use.",
            f"Advanced {subcategory.lower()} with cutting-edge technology and innovation."
        ]
        
        return random.choice(descriptions)
    
    def generate_products(self, num_products: int) -> List[Product]:
        """Generate synthetic products."""
        products = []
        
        # Get subcategories for product generation
        subcategories = [cat for cat in self.categories if cat.level == 2]
        
        for _ in range(num_products):
            # Select random subcategory
            subcategory = random.choice(subcategories)
            main_category = next(cat for cat in self.categories if cat.category_id == subcategory.parent_category)
            
            # Get template for this category
            template = self._get_category_template(main_category.name)
            
            # Generate product data
            name = self._generate_product_name(main_category.name, subcategory.name)
            description = self._generate_description(name, main_category.name, subcategory.name)
            
                    # Generate realistic pricing (USD)
            base_price = random.uniform(*template["price_range"])
            discount = random.choices([0, 5, 10, 15, 20, 25, 30], weights=[0.3, 0.25, 0.2, 0.15, 0.07, 0.02, 0.01])[0]
            final_price = round(base_price * (1 - discount / 100), 2)
            
            # Create product
            product = Product(
                product_id=self.product_id_counter,
                name=name,
                description=description,
                category=main_category.name,
                subcategory=subcategory.name,
                brand=random.choice(template["brands"]),
                price=round(final_price, 2),
                discount_percentage=discount,
                stock_quantity=random.randint(0, 100),
                rating=round(random.uniform(3.0, 5.0), 1),
                review_count=random.randint(0, 500),
                shipping_cost=round(random.uniform(0, 25), 2),  # USD shipping costs
                weight=round(random.uniform(*template["weight_range"]), 2),
                dimensions=f"{random.randint(4, 40)}\"x{random.randint(4, 40)}\"x{random.randint(2, 20)}\"",  # US inches
                color=random.choice(template["colors"]),
                size=random.choice(template["sizes"]),
                availability_status=random.choices(
                    ["in_stock", "out_of_stock", "discontinued"],
                    weights=[0.8, 0.15, 0.05]
                )[0],
                image_url=f"https://example.com/images/{self.product_id_counter}.jpg",
                tags=[subcategory.name, main_category.name, random.choice(template["brands"])],
                created_at=fake.date_time_between(
                    start_date="-1y",
                    end_date="-1d"
                )
            )
            
            products.append(product)
            self.product_id_counter += 1
        
        logger.info(f"Generated {len(products)} products")
        return products


class InteractionGenerator:
    """Generate user-product interactions."""
    
    def __init__(self, users: List[User], products: List[Product]):
        self.users = users
        self.products = products
        self.interaction_id_counter = 1
        
        # Interaction type weights (realistic e-commerce behavior)
        self.interaction_weights = {
            "view": 0.60,      # Most common
            "click": 0.25,     # Second most common
            "add_to_cart": 0.10,  # Less common
            "purchase": 0.03,   # Rare
            "rating": 0.015,    # Very rare
            "review": 0.005     # Extremely rare
        }
        
        # Payment methods
        self.payment_methods = ["Credit Card", "Debit Card", "PayPal", "Apple Pay", "Google Pay", "Venmo"]
    
    def _generate_session_id(self, user_id: int) -> str:
        """Generate unique session ID."""
        return f"session_{user_id}_{random.randint(1000, 9999)}"
    
    def _generate_interaction_timestamp(self, user: User) -> datetime:
        """Generate realistic interaction timestamp."""
        # Base timestamp from user's last active
        base_time = user.last_active
        
        # Add some randomness within the last month
        days_ago = random.randint(0, 30)
        hours_ago = random.randint(0, 23)
        minutes_ago = random.randint(0, 59)
        
        return base_time - timedelta(
            days=days_ago,
            hours=hours_ago,
            minutes=minutes_ago
        )
    
    def _generate_rating(self) -> float:
        """Generate realistic rating distribution."""
        # Slightly positive bias (realistic for e-commerce)
        return round(random.choices(
            [1.0, 2.0, 3.0, 4.0, 5.0],
            weights=[0.05, 0.10, 0.20, 0.40, 0.25]
        )[0], 1)
    
    def generate_interactions(self, num_interactions: int) -> List[Interaction]:
        """Generate synthetic interactions."""
        interactions = []
        
        for _ in range(num_interactions):
            # Select random user and product
            user = random.choice(self.users)
            product = random.choice(self.products)
            
            # Generate interaction type based on weights
            interaction_type = random.choices(
                list(self.interaction_weights.keys()),
                weights=list(self.interaction_weights.values())
            )[0]
            
            # Generate timestamp
            timestamp = self._generate_interaction_timestamp(user)
            
            # Generate session ID
            session_id = self._generate_session_id(user.user_id)
            
            # Generate interaction-specific data
            rating = None
            review_text = None
            quantity = None
            total_amount = None
            payment_method = None
            dwell_time = None
            scroll_depth = None
            
            if interaction_type == "rating":
                rating = self._generate_rating()
            elif interaction_type == "review":
                rating = self._generate_rating()
                review_text = fake.text(max_nb_chars=200)
            elif interaction_type in ["add_to_cart", "purchase"]:
                quantity = random.randint(1, 5)
                total_amount = round(product.price * quantity, 2)
                if interaction_type == "purchase":
                    payment_method = random.choice(self.payment_methods)
            elif interaction_type == "view":
                dwell_time = random.randint(5, 300)  # 5 seconds to 5 minutes
                scroll_depth = random.randint(10, 100)  # 10% to 100%
            
            # Create interaction
            interaction = Interaction(
                interaction_id=self.interaction_id_counter,
                user_id=user.user_id,
                product_id=product.product_id,
                interaction_type=interaction_type,
                timestamp=timestamp,
                rating=rating,
                review_text=review_text,
                session_id=session_id,
                quantity=quantity,
                total_amount=total_amount,
                payment_method=payment_method,
                dwell_time=dwell_time,
                scroll_depth=scroll_depth
            )
            
            interactions.append(interaction)
            self.interaction_id_counter += 1
        
        logger.info(f"Generated {len(interactions)} interactions")
        return interactions


class DataGenerator:
    """Main data generator orchestrator."""
    
    def __init__(self):
        self.category_generator = CategoryGenerator()
        self.categories = None
        self.users = None
        self.products = None
        self.interactions = None
    
    def generate_all_data(self) -> Dict[str, List]:
        """Generate all synthetic data."""
        logger.info("Starting data generation process")
        
        # Generate categories first
        self.categories = self.category_generator.generate_categories()
        
        # Generate users
        user_generator = UserGenerator(self.categories)
        self.users = user_generator.generate_users(settings.NUM_USERS)
        
        # Generate products
        product_generator = ProductGenerator(self.categories)
        self.products = product_generator.generate_products(settings.NUM_PRODUCTS)
        
        # Generate interactions
        interaction_generator = InteractionGenerator(self.users, self.products)
        self.interactions = interaction_generator.generate_interactions(settings.NUM_INTERACTIONS)
        
        # Log data summary
        self._log_data_summary()
        
        return {
            "categories": self.categories,
            "users": self.users,
            "products": self.products,
            "interactions": self.interactions
        }
    
    def _log_data_summary(self):
        """Log summary of generated data."""
        logger.info("Data generation completed", extra={
            "categories_count": len(self.categories),
            "users_count": len(self.users),
            "products_count": len(self.products),
            "interactions_count": len(self.interactions)
        })
        
        # Log sample data for verification
        if self.users:
            log_data_info("users", len(self.users), 
                         [field for field in self.users[0].__fields__],
                         self.users[0].dict())
        
        if self.products:
            log_data_info("products", len(self.products),
                         [field for field in self.products[0].__fields__],
                         self.products[0].dict())
        
        if self.interactions:
            log_data_info("interactions", len(self.interactions),
                         [field for field in self.interactions[0].__fields__],
                         self.interactions[0].dict())
