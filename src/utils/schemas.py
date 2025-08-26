"""Data schemas for the Recommndr project."""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class User(BaseModel):
    """User data model."""
    
    user_id: int = Field(..., description="Unique user identifier")
    age: int = Field(..., ge=13, le=100, description="User age")
    gender: str = Field(..., description="User gender")
    location: str = Field(..., description="User location/city")
    income_level: str = Field(..., description="User income level")
    preference_category: str = Field(..., description="Primary category preference")
    device_type: str = Field(..., description="Device type used")
    language_preference: str = Field(..., description="Language preference")
    timezone: str = Field(..., description="User timezone")
    email: str = Field(..., description="User email address")
    created_at: datetime = Field(..., description="Account creation timestamp")
    last_active: datetime = Field(..., description="Last activity timestamp")
    
    @field_validator('gender')
    @classmethod
    def validate_gender(cls, v):
        valid_genders = ['male', 'female', 'other']
        if v.lower() not in valid_genders:
            raise ValueError(f'Gender must be one of {valid_genders}')
        return v.lower()
    
    @field_validator('income_level')
    @classmethod
    def validate_income_level(cls, v):
        valid_levels = ['low', 'medium', 'high', 'luxury']
        if v.lower() not in valid_levels:
            raise ValueError(f'Income level must be one of {valid_levels}')
        return v.lower()
    
    @field_validator('device_type')
    @classmethod
    def validate_device_type(cls, v):
        valid_devices = ['mobile', 'desktop', 'tablet']
        if v.lower() not in valid_devices:
            raise ValueError(f'Device type must be one of {valid_devices}')
        return v.lower()


class Product(BaseModel):
    """Product data model."""
    
    product_id: int = Field(..., description="Unique product identifier")
    name: str = Field(..., min_length=1, max_length=200, description="Product name")
    description: str = Field(..., min_length=10, max_length=1000, description="Product description")
    category: str = Field(..., description="Main product category")
    subcategory: str = Field(..., description="Product subcategory")
    brand: str = Field(..., description="Product brand")
    price: float = Field(..., gt=0, description="Product price")
    discount_percentage: float = Field(..., ge=0, le=100, description="Discount percentage")
    stock_quantity: int = Field(..., ge=0, description="Available stock quantity")
    rating: float = Field(..., ge=0, le=5, description="Average product rating")
    review_count: int = Field(..., ge=0, description="Number of reviews")
    shipping_cost: float = Field(..., ge=0, description="Shipping cost")
    weight: float = Field(..., gt=0, description="Product weight in pounds (lbs)")
    dimensions: str = Field(..., description="Product dimensions in inches")
    color: str = Field(..., description="Product color")
    size: str = Field(..., description="Product size")
    availability_status: str = Field(..., description="Product availability status")
    image_url: str = Field(..., description="Product image URL")
    tags: List[str] = Field(..., description="Product tags")
    created_at: datetime = Field(..., description="Product listing timestamp")
    
    @field_validator('availability_status')
    @classmethod
    def validate_availability_status(cls, v):
        valid_statuses = ['in_stock', 'out_of_stock', 'discontinued']
        if v.lower() not in valid_statuses:
            raise ValueError(f'Availability status must be one of {valid_statuses}')
        return v.lower()


class Interaction(BaseModel):
    """User-product interaction data model."""
    
    interaction_id: int = Field(..., description="Unique interaction identifier")
    user_id: int = Field(..., description="User identifier")
    product_id: int = Field(..., description="Product identifier")
    interaction_type: str = Field(..., description="Type of interaction")
    timestamp: datetime = Field(..., description="Interaction timestamp")
    rating: Optional[float] = Field(None, ge=1, le=5, description="User rating")
    review_text: Optional[str] = Field(None, max_length=1000, description="Review text")
    session_id: str = Field(..., description="User session identifier")
    quantity: Optional[int] = Field(None, ge=1, description="Quantity")
    total_amount: Optional[float] = Field(None, gt=0, description="Total amount")
    payment_method: Optional[str] = Field(None, description="Payment method")
    dwell_time: Optional[int] = Field(None, ge=0, description="Time spent viewing (seconds)")
    scroll_depth: Optional[int] = Field(None, ge=0, le=100, description="Scroll depth percentage")
    
    @field_validator('interaction_type')
    @classmethod
    def validate_interaction_type(cls, v):
        valid_types = ['view', 'click', 'add_to_cart', 'purchase', 'rating', 'review']
        if v.lower() not in valid_types:
            raise ValueError(f'Interaction type must be one of {valid_types}')
        return v.lower()


class Category(BaseModel):
    """Product category data model."""
    
    category_id: int = Field(..., description="Unique category identifier")
    name: str = Field(..., description="Category name")
    parent_category: Optional[int] = Field(None, description="Parent category ID")
    description: str = Field(..., description="Category description")
    level: int = Field(..., ge=1, description="Category hierarchy level")


class ProductCategory(BaseModel):
    """Product-category relationship model."""
    
    product_id: int = Field(..., description="Product identifier")
    category_id: int = Field(..., description="Category identifier")
    is_primary: bool = Field(..., description="Whether this is the primary category")


class Order(BaseModel):
    """Order data model."""
    
    order_id: int = Field(..., description="Unique order identifier")
    user_id: int = Field(..., description="User who placed the order")
    order_date: datetime = Field(..., description="Order placement timestamp")
    total_amount: float = Field(..., gt=0, description="Order total amount")
    status: str = Field(..., description="Order status")
    payment_status: str = Field(..., description="Payment status")
    shipping_address: str = Field(..., description="Shipping address")
    shipping_cost: float = Field(..., ge=0, description="Shipping cost")
    
    @field_validator('status')
    @classmethod
    def validate_order_status(cls, v):
        valid_statuses = ['pending', 'confirmed', 'shipped', 'delivered', 'cancelled']
        if v.lower() not in valid_statuses:
            raise ValueError(f'Order status must be one of {valid_statuses}')
        return v.lower()
    
    @field_validator('payment_status')
    @classmethod
    def validate_payment_status(cls, v):
        valid_statuses = ['pending', 'completed', 'failed']
        if v.lower() not in valid_statuses:
            raise ValueError(f'Payment status must be one of {valid_statuses}')
        return v.lower()


class CartItem(BaseModel):
    """Shopping cart item model."""
    
    cart_id: int = Field(..., description="Unique cart identifier")
    user_id: int = Field(..., description="User who owns the cart")
    product_id: int = Field(..., description="Product in cart")
    quantity: int = Field(..., ge=1, description="Quantity added")
    added_at: datetime = Field(..., description="When item was added to cart")


class SearchQuery(BaseModel):
    """Search query model."""
    
    query_id: int = Field(..., description="Unique query identifier")
    user_id: int = Field(..., description="User who searched")
    search_query: str = Field(..., min_length=1, max_length=200, description="Search query text")
    timestamp: datetime = Field(..., description="Search timestamp")
    results_count: int = Field(..., ge=0, description="Number of results returned")
    clicked_product_id: Optional[int] = Field(None, description="Clicked product ID")


class UserPreference(BaseModel):
    """User preference model."""
    
    user_id: int = Field(..., description="User identifier")
    category_id: int = Field(..., description="Category identifier")
    preference_type: str = Field(..., description="Preference type")
    strength: float = Field(..., ge=0, le=1, description="Preference strength")
    created_at: datetime = Field(..., description="When preference was recorded")
    
    @field_validator('preference_type')
    @classmethod
    def validate_preference_type(cls, v):
        valid_types = ['like', 'dislike', 'neutral']
        if v.lower() not in valid_types:
            raise ValueError(f'Preference type must be one of {valid_types}')
        return v.lower()


# Data validation schemas
class DataValidationResult(BaseModel):
    """Data validation result model."""
    
    table_name: str = Field(..., description="Name of the validated table")
    total_rows: int = Field(..., description="Total number of rows")
    valid_rows: int = Field(..., description="Number of valid rows")
    invalid_rows: int = Field(..., description="Number of invalid rows")
    quality_score: float = Field(..., ge=0, le=1, description="Data quality score")
    validation_errors: List[str] = Field(default_factory=list, description="Validation error messages")
    validation_timestamp: datetime = Field(..., description="When validation was performed")
