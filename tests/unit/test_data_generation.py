"""Unit tests for data generation module."""

import pytest
from datetime import datetime

from src.data_generation.generators import (
    CategoryGenerator,
    UserGenerator,
    ProductGenerator,
    InteractionGenerator,
    DataGenerator,
)
from src.utils.schemas import Category, User, Product, Interaction


class TestCategoryGenerator:
    """Test category generation functionality."""
    
    def test_category_generator_initialization(self):
        """Test category generator initialization."""
        generator = CategoryGenerator()
        assert generator.categories is not None
        assert generator.category_id_counter == 1
    
    def test_generate_categories(self):
        """Test category generation."""
        generator = CategoryGenerator()
        categories = generator.generate_categories()
        
        assert len(categories) > 0
        assert all(isinstance(cat, Category) for cat in categories)
        
        # Check hierarchy
        main_categories = [cat for cat in categories if cat.level == 1]
        sub_categories = [cat for cat in categories if cat.level == 2]
        
        assert len(main_categories) > 0
        assert len(sub_categories) > 0
        
        # Check that subcategories have parent references
        for sub_cat in sub_categories:
            assert sub_cat.parent_category is not None
            assert any(main_cat.category_id == sub_cat.parent_category for main_cat in main_categories)


class TestUserGenerator:
    """Test user generation functionality."""
    
    def test_user_generator_initialization(self, sample_categories):
        """Test user generator initialization."""
        generator = UserGenerator(sample_categories)
        assert generator.categories == sample_categories
        assert generator.user_id_counter == 1
    
    def test_generate_age_distribution(self, sample_categories):
        """Test age generation follows distribution."""
        generator = UserGenerator(sample_categories)
        
        ages = []
        for _ in range(100):
            age = generator._generate_age()
            ages.append(age)
            assert 13 <= age <= 65
        
        # Check that we have a reasonable distribution
        age_groups = {
            "18-25": sum(1 for age in ages if 18 <= age <= 25),
            "26-35": sum(1 for age in ages if 26 <= age <= 35),
            "36-45": sum(1 for age in ages if 36 <= age <= 45),
            "46+": sum(1 for age in ages if 46 <= age <= 65),
        }
        
        # Should have some users in each group
        assert all(count > 0 for count in age_groups.values())
    
    def test_generate_income_level_distribution(self, sample_categories):
        """Test income level generation follows distribution."""
        generator = UserGenerator(sample_categories)
        
        income_levels = []
        for _ in range(100):
            income_level = generator._generate_income_level()
            income_levels.append(income_level)
            assert income_level in ["low", "medium", "high", "luxury"]
        
        # Check that we have a reasonable distribution
        level_counts = {level: income_levels.count(level) for level in set(income_levels)}
        assert all(count > 0 for count in level_counts.values())
    
    def test_generate_users(self, sample_categories):
        """Test user generation."""
        generator = UserGenerator(sample_categories)
        users = generator.generate_users(10)
        
        assert len(users) == 10
        assert all(isinstance(user, User) for user in users)
        
        # Check user IDs are unique
        user_ids = [user.user_id for user in users]
        assert len(user_ids) == len(set(user_ids))
        
        # Check email addresses are unique
        emails = [user.email for user in users]
        assert len(emails) == len(set(emails))
        
        # Check all required fields are present
        for user in users:
            assert user.user_id > 0
            assert 13 <= user.age <= 100
            assert user.gender in ["male", "female", "other"]
            assert user.income_level in ["low", "medium", "high", "luxury"]
            assert user.device_type in ["mobile", "desktop", "tablet"]
            assert user.created_at < user.last_active


class TestProductGenerator:
    """Test product generation functionality."""
    
    def test_product_generator_initialization(self, sample_categories):
        """Test product generator initialization."""
        generator = ProductGenerator(sample_categories)
        assert generator.categories == sample_categories
        assert generator.product_id_counter == 1
    
    def test_get_category_template(self, sample_categories):
        """Test category template retrieval."""
        generator = ProductGenerator(sample_categories)
        
        # Test Electronics template
        electronics_template = generator._get_category_template("Electronics")
        assert "brands" in electronics_template
        assert "price_range" in electronics_template
        assert "weight_range" in electronics_template
        
        # Test default template for unknown category
        default_template = generator._get_category_template("UnknownCategory")
        assert "brands" in default_template
        assert "price_range" in default_template
    
    def test_generate_product_name(self, sample_categories):
        """Test product name generation."""
        generator = ProductGenerator(sample_categories)
        
        # Test Electronics names
        smartphone_name = generator._generate_product_name("Electronics", "Smartphones")
        assert "iPhone" in smartphone_name or "Samsung Galaxy" in smartphone_name or "OnePlus" in smartphone_name
        
        # Test generic names
        generic_name = generator._generate_product_name("Books", "Fiction")
        assert any(word in generic_name for word in ["Premium", "Professional", "Advanced", "Classic", "Modern"])
    
    def test_generate_products(self, sample_categories):
        """Test product generation."""
        generator = ProductGenerator(sample_categories)
        products = generator.generate_products(10)
        
        assert len(products) == 10
        assert all(isinstance(product, Product) for product in products)
        
        # Check product IDs are unique
        product_ids = [product.product_id for product in products]
        assert len(product_ids) == len(set(product_ids))
        
        # Check all required fields are present
        for product in products:
            assert product.product_id > 0
            assert product.price > 0
            assert 0 <= product.discount_percentage <= 100
            assert product.stock_quantity >= 0
            assert 0 <= product.rating <= 5
            assert product.availability_status in ["in_stock", "out_of_stock", "discontinued"]
            assert len(product.tags) > 0


class TestInteractionGenerator:
    """Test interaction generation functionality."""
    
    def test_interaction_generator_initialization(self, sample_users, sample_products):
        """Test interaction generator initialization."""
        generator = InteractionGenerator(sample_users, sample_products)
        assert generator.users == sample_users
        assert generator.products == sample_products
        assert generator.interaction_id_counter == 1
    
    def test_generate_session_id(self, sample_users, sample_products):
        """Test session ID generation."""
        generator = InteractionGenerator(sample_users, sample_products)
        
        session_id = generator._generate_session_id(123)
        assert session_id.startswith("session_123_")
        assert len(session_id.split("_")) == 3
    
    def test_generate_rating(self, sample_users, sample_products):
        """Test rating generation."""
        generator = InteractionGenerator(sample_users, sample_products)
        
        ratings = []
        for _ in range(100):
            rating = generator._generate_rating()
            ratings.append(rating)
            assert 1.0 <= rating <= 5.0
        
        # Check that we have a reasonable distribution
        rating_counts = {rating: ratings.count(rating) for rating in set(ratings)}
        assert all(count > 0 for count in rating_counts.values())
    
    def test_generate_interactions(self, sample_users, sample_products):
        """Test interaction generation."""
        generator = InteractionGenerator(sample_users, sample_products)
        interactions = generator.generate_interactions(20)
        
        assert len(interactions) == 20
        assert all(isinstance(interaction, Interaction) for interaction in interactions)
        
        # Check interaction IDs are unique
        interaction_ids = [interaction.interaction_id for interaction in interactions]
        assert len(interaction_ids) == len(set(interaction_ids))
        
        # Check all required fields are present
        for interaction in interactions:
            assert interaction.interaction_id > 0
            assert interaction.user_id in [user.user_id for user in sample_users]
            assert interaction.product_id in [product.product_id for product in sample_products]
            assert interaction.interaction_type in ["view", "click", "add_to_cart", "purchase", "rating", "review"]
            assert interaction.session_id.startswith("session_")
            
            # Check interaction-specific fields
            if interaction.interaction_type == "rating":
                assert interaction.rating is not None
                assert 1.0 <= interaction.rating <= 5.0
            elif interaction.interaction_type in ["add_to_cart", "purchase"]:
                assert interaction.quantity is not None
                assert interaction.quantity >= 1
                assert interaction.total_amount is not None
                assert interaction.total_amount > 0


class TestDataGenerator:
    """Test main data generator orchestrator."""
    
    def test_data_generator_initialization(self):
        """Test data generator initialization."""
        generator = DataGenerator()
        assert generator.category_generator is not None
        assert generator.categories is None
        assert generator.users is None
        assert generator.products is None
        assert generator.interactions is None
    
    def test_generate_all_data(self):
        """Test complete data generation process."""
        generator = DataGenerator()
        data = generator.generate_all_data()
        
        # Check all data types are generated
        assert "categories" in data
        assert "users" in data
        assert "products" in data
        assert "interactions" in data
        
        # Check data counts
        assert len(data["categories"]) > 0
        assert len(data["users"]) > 0
        assert len(data["products"]) > 0
        assert len(data["interactions"]) > 0
        
        # Check data types
        assert all(isinstance(cat, Category) for cat in data["categories"])
        assert all(isinstance(user, User) for user in data["users"])
        assert all(isinstance(product, Product) for product in data["products"])
        assert all(isinstance(interaction, Interaction) for interaction in data["interactions"])


# Fixtures
@pytest.fixture
def sample_categories():
    """Provide sample categories for testing."""
    return [
        Category(
            category_id=1,
            name="Electronics",
            parent_category=None,
            description="Main category for electronics",
            level=1
        ),
        Category(
            category_id=2,
            name="Smartphones",
            parent_category=1,
            description="Subcategory of Electronics: Smartphones",
            level=2
        )
    ]


@pytest.fixture
def sample_users(sample_categories):
    """Provide sample users for testing."""
    return [
        User(
            user_id=1,
            age=25,
            gender="male",
            location="New York",
            income_level="medium",
            preference_category="Electronics",
            device_type="mobile",
            language_preference="English",
            timezone="EST",
            email="test1@example.com",
            created_at=datetime.now(),
            last_active=datetime.now()
        ),
        User(
            user_id=2,
            age=30,
            gender="female",
            location="Los Angeles",
            income_level="high",
            preference_category="Clothing",
            device_type="desktop",
            language_preference="English",
            timezone="PST",
            email="test2@example.com",
            created_at=datetime.now(),
            last_active=datetime.now()
        )
    ]


@pytest.fixture
def sample_products(sample_categories):
    """Provide sample products for testing."""
    return [
        Product(
            product_id=1,
            name="Test Smartphone",
            description="A test smartphone product",
            category="Electronics",
            subcategory="Smartphones",
            brand="TestBrand",
            price=999.99,
            discount_percentage=10.0,
            stock_quantity=50,
            rating=4.5,
            review_count=100,
            shipping_cost=15.99,
            weight=0.5,
            dimensions="6\"x3\"x0.3\"",
            color="Black",
            size="One Size",
            availability_status="in_stock",
            image_url="https://example.com/image1.jpg",
            tags=["Smartphone", "Electronics", "TestBrand"],
            created_at=datetime.now()
        )
    ]
