"""
Feast Feature Registry Metadata Management

Enhances Feast feature store with:
- Feature metadata documentation
- Data lineage tracking
- Feature validation rules
- Performance monitoring integration
- Feature drift detection preparation
"""

import json
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


class FeatureMetadata:
    """Enhanced feature metadata management for Feast."""
    
    def __init__(self, feast_repo_path: str = "feature_repo"):
        """Initialize feature metadata manager."""
        self.feast_repo_path = Path(feast_repo_path)
        self.metadata_dir = self.feast_repo_path / "metadata"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Feature metadata storage
        self.feature_definitions = {}
        self.feature_lineage = {}
        self.feature_statistics = {}
        
        logger.info("📊 Feast Feature Metadata Manager initialized")
    
    def register_feature_metadata(self, feature_name: str, metadata: Dict[str, Any]):
        """Register comprehensive metadata for a feature."""
        logger.info(f"📝 Registering metadata for feature: {feature_name}")
        
        enhanced_metadata = {
            'feature_name': feature_name,
            'registration_time': time.time(),
            'registration_date': datetime.now().isoformat(),
            **metadata
        }
        
        # Required metadata fields
        required_fields = [
            'description', 'data_type', 'source_table', 'calculation_logic'
        ]
        
        for field in required_fields:
            if field not in enhanced_metadata:
                enhanced_metadata[field] = f"TODO: Define {field}"
        
        # Store metadata
        self.feature_definitions[feature_name] = enhanced_metadata
        
        # Save to disk
        self._save_feature_metadata(feature_name, enhanced_metadata)
        
        logger.info(f"✅ Feature metadata registered: {feature_name}")
    
    def register_user_features_metadata(self):
        """Register metadata for user features."""
        user_features = [
            {
                'name': 'user_age',
                'description': 'User age in years',
                'data_type': 'int64',
                'source_table': 'users',
                'calculation_logic': 'Direct field from users table',
                'business_meaning': 'Used for age-based recommendation personalization',
                'valid_range': [13, 100],
                'null_handling': 'Fill with median age',
                'update_frequency': 'Daily',
                'owner': 'data_team',
                'stakeholders': ['ml_team', 'product_team']
            },
            {
                'name': 'user_income_level',
                'description': 'User income level category',
                'data_type': 'string',
                'source_table': 'users',
                'calculation_logic': 'Categorical field: low, medium, high',
                'business_meaning': 'Price sensitivity and product affinity',
                'valid_values': ['low', 'medium', 'high'],
                'null_handling': 'Fill with "medium"',
                'update_frequency': 'Weekly',
                'owner': 'data_team',
                'stakeholders': ['ml_team', 'product_team']
            },
            {
                'name': 'user_preference_category',
                'description': 'User preferred product category',
                'data_type': 'string',
                'source_table': 'users',
                'calculation_logic': 'Most frequent category from user interactions',
                'business_meaning': 'Primary category affinity for recommendations',
                'valid_values': ['Electronics', 'Clothing', 'Books', 'Home & Garden', 'Sports & Outdoors', 'Beauty & Health', 'Toys & Games', 'Automotive'],
                'null_handling': 'Fill with most common category',
                'update_frequency': 'Daily',
                'owner': 'ml_team',
                'stakeholders': ['product_team', 'business_team']
            }
        ]
        
        for feature in user_features:
            self.register_feature_metadata(feature['name'], feature)
    
    def register_item_features_metadata(self):
        """Register metadata for item features."""
        item_features = [
            {
                'name': 'item_price',
                'description': 'Product price in USD',
                'data_type': 'float64',
                'source_table': 'products',
                'calculation_logic': 'Direct price field from products table',
                'business_meaning': 'Price-based filtering and recommendations',
                'valid_range': [0.01, 10000.0],
                'null_handling': 'Use category median price',
                'update_frequency': 'Real-time',
                'owner': 'product_team',
                'stakeholders': ['ml_team', 'business_team']
            },
            {
                'name': 'item_rating',
                'description': 'Product average rating (1-5 scale)',
                'data_type': 'float64',
                'source_table': 'products',
                'calculation_logic': 'Average of all user ratings',
                'business_meaning': 'Quality indicator for recommendations',
                'valid_range': [1.0, 5.0],
                'null_handling': 'Use category average rating',
                'update_frequency': 'Hourly',
                'owner': 'data_team',
                'stakeholders': ['ml_team', 'product_team']
            },
            {
                'name': 'item_category',
                'description': 'Product category classification',
                'data_type': 'string',
                'source_table': 'products',
                'calculation_logic': 'Categorical field from product taxonomy',
                'business_meaning': 'Category-based recommendation filtering',
                'valid_values': ['Electronics', 'Clothing', 'Books', 'Home & Garden', 'Sports & Outdoors', 'Beauty & Health', 'Toys & Games', 'Automotive'],
                'null_handling': 'Classify as "Other"',
                'update_frequency': 'Daily',
                'owner': 'product_team',
                'stakeholders': ['ml_team', 'data_team']
            }
        ]
        
        for feature in item_features:
            self.register_feature_metadata(feature['name'], feature)
    
    def register_contextual_features_metadata(self):
        """Register metadata for contextual features."""
        contextual_features = [
            {
                'name': 'hour_of_day',
                'description': 'Hour of day when request is made (0-23)',
                'data_type': 'int64',
                'source_table': 'real_time',
                'calculation_logic': 'Extract hour from request timestamp',
                'business_meaning': 'Time-based recommendation personalization',
                'valid_range': [0, 23],
                'null_handling': 'Use current server time',
                'update_frequency': 'Real-time',
                'owner': 'ml_team',
                'stakeholders': ['product_team']
            },
            {
                'name': 'day_of_week',
                'description': 'Day of week (0=Monday, 6=Sunday)',
                'data_type': 'int64',
                'source_table': 'real_time',
                'calculation_logic': 'Extract day of week from request timestamp',
                'business_meaning': 'Weekday vs weekend recommendation patterns',
                'valid_range': [0, 6],
                'null_handling': 'Use current server time',
                'update_frequency': 'Real-time',
                'owner': 'ml_team',
                'stakeholders': ['product_team']
            },
            {
                'name': 'is_weekend',
                'description': 'Boolean flag for weekend (Saturday/Sunday)',
                'data_type': 'bool',
                'source_table': 'real_time',
                'calculation_logic': 'True if day_of_week in [5, 6]',
                'business_meaning': 'Weekend shopping behavior patterns',
                'valid_values': [True, False],
                'null_handling': 'Use current server time',
                'update_frequency': 'Real-time',
                'owner': 'ml_team',
                'stakeholders': ['product_team']
            },
            {
                'name': 'session_duration_minutes',
                'description': 'Current session duration in minutes',
                'data_type': 'float64',
                'source_table': 'sessions',
                'calculation_logic': 'Current time - session start time',
                'business_meaning': 'User engagement level indicator',
                'valid_range': [0, 1440],  # 0 to 24 hours
                'null_handling': 'Default to 0',
                'update_frequency': 'Real-time',
                'owner': 'data_team',
                'stakeholders': ['ml_team', 'product_team']
            }
        ]
        
        for feature in contextual_features:
            self.register_feature_metadata(feature['name'], feature)
    
    def register_interaction_features_metadata(self):
        """Register metadata for user interaction features."""
        interaction_features = [
            {
                'name': 'user_total_interactions',
                'description': 'Total number of user interactions (all time)',
                'data_type': 'int64',
                'source_table': 'interactions',
                'calculation_logic': 'COUNT(*) FROM interactions WHERE user_id = X',
                'business_meaning': 'User engagement and activity level',
                'valid_range': [0, 100000],
                'null_handling': 'Default to 0',
                'update_frequency': 'Real-time',
                'owner': 'data_team',
                'stakeholders': ['ml_team']
            },
            {
                'name': 'user_recent_category_counts',
                'description': 'User interactions by category (last 30 days)',
                'data_type': 'json',
                'source_table': 'interactions',
                'calculation_logic': 'Category interaction counts from last 30 days',
                'business_meaning': 'Recent category preferences for recommendations',
                'valid_range': 'JSON object with category counts',
                'null_handling': 'Empty JSON object {}',
                'update_frequency': 'Hourly',
                'owner': 'ml_team',
                'stakeholders': ['product_team']
            },
            {
                'name': 'user_avg_session_duration',
                'description': 'Average session duration for user (last 30 days)',
                'data_type': 'float64',
                'source_table': 'sessions',
                'calculation_logic': 'AVG(session_duration) WHERE user_id = X AND created_at > 30 days ago',
                'business_meaning': 'User engagement pattern indicator',
                'valid_range': [0, 1440],
                'null_handling': 'Use global average',
                'update_frequency': 'Daily',
                'owner': 'data_team',
                'stakeholders': ['ml_team', 'product_team']
            }
        ]
        
        for feature in interaction_features:
            self.register_feature_metadata(feature['name'], feature)
    
    def register_all_feature_metadata(self):
        """Register metadata for all feature categories."""
        logger.info("📊 Registering metadata for all feature categories...")
        
        self.register_user_features_metadata()
        self.register_item_features_metadata()
        self.register_contextual_features_metadata()
        self.register_interaction_features_metadata()
        
        logger.info(f"✅ Registered metadata for {len(self.feature_definitions)} features")
    
    def create_feature_lineage(self, feature_name: str, lineage_info: Dict[str, Any]):
        """Create feature lineage documentation."""
        logger.info(f"🔗 Creating lineage for feature: {feature_name}")
        
        lineage = {
            'feature_name': feature_name,
            'created_at': time.time(),
            'upstream_sources': lineage_info.get('sources', []),
            'transformation_steps': lineage_info.get('transformations', []),
            'downstream_consumers': lineage_info.get('consumers', []),
            'data_freshness': lineage_info.get('freshness', 'unknown'),
            'quality_checks': lineage_info.get('quality_checks', [])
        }
        
        self.feature_lineage[feature_name] = lineage
        
        # Save lineage to disk
        lineage_file = self.metadata_dir / f"{feature_name}_lineage.json"
        with open(lineage_file, 'w') as f:
            json.dump(lineage, f, indent=2)
        
        logger.info(f"✅ Feature lineage created: {feature_name}")
    
    def calculate_feature_statistics(self, feature_name: str, data: pd.Series) -> Dict[str, Any]:
        """Calculate and store feature statistics."""
        logger.info(f"📈 Calculating statistics for feature: {feature_name}")
        
        stats = {
            'feature_name': feature_name,
            'calculation_time': time.time(),
            'data_points': len(data),
            'null_count': data.isnull().sum(),
            'null_percentage': (data.isnull().sum() / len(data)) * 100
        }
        
        if data.dtype in ['int64', 'float64']:
            # Numerical statistics
            stats.update({
                'mean': float(data.mean()),
                'median': float(data.median()),
                'std': float(data.std()),
                'min': float(data.min()),
                'max': float(data.max()),
                'q25': float(data.quantile(0.25)),
                'q75': float(data.quantile(0.75)),
                'skewness': float(data.skew()),
                'kurtosis': float(data.kurtosis())
            })
        elif data.dtype == 'object' or data.dtype.name == 'category':
            # Categorical statistics
            value_counts = data.value_counts()
            stats.update({
                'unique_values': int(data.nunique()),
                'most_frequent': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                'value_distribution': value_counts.head(10).to_dict()
            })
        
        self.feature_statistics[feature_name] = stats
        
        # Save statistics to disk
        stats_file = self.metadata_dir / f"{feature_name}_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        logger.info(f"✅ Feature statistics calculated: {feature_name}")
        return stats
    
    def validate_feature_quality(self, feature_name: str, data: pd.Series) -> Dict[str, Any]:
        """Validate feature quality against metadata rules."""
        validation_results = {
            'feature_name': feature_name,
            'validation_time': time.time(),
            'checks': {},
            'overall_status': 'passed'
        }
        
        if feature_name not in self.feature_definitions:
            validation_results['checks']['metadata_exists'] = {
                'status': 'failed',
                'message': 'Feature metadata not found'
            }
            validation_results['overall_status'] = 'failed'
            return validation_results
        
        metadata = self.feature_definitions[feature_name]
        
        # Check data type
        expected_type = metadata.get('data_type', 'unknown')
        actual_type = str(data.dtype)
        
        validation_results['checks']['data_type'] = {
            'status': 'passed' if expected_type in actual_type or actual_type in expected_type else 'failed',
            'expected': expected_type,
            'actual': actual_type
        }
        
        # Check valid range (for numerical features)
        if 'valid_range' in metadata and data.dtype in ['int64', 'float64']:
            valid_range = metadata['valid_range']
            out_of_range = ((data < valid_range[0]) | (data > valid_range[1])).sum()
            
            validation_results['checks']['valid_range'] = {
                'status': 'passed' if out_of_range == 0 else 'warning',
                'out_of_range_count': int(out_of_range),
                'out_of_range_percentage': float((out_of_range / len(data)) * 100),
                'valid_range': valid_range
            }
        
        # Check valid values (for categorical features)
        if 'valid_values' in metadata and data.dtype == 'object':
            valid_values = set(metadata['valid_values'])
            actual_values = set(data.dropna().unique())
            invalid_values = actual_values - valid_values
            
            validation_results['checks']['valid_values'] = {
                'status': 'passed' if len(invalid_values) == 0 else 'warning',
                'invalid_values': list(invalid_values),
                'valid_values': list(valid_values)
            }
        
        # Check null percentage
        null_percentage = (data.isnull().sum() / len(data)) * 100
        validation_results['checks']['null_percentage'] = {
            'status': 'passed' if null_percentage < 10 else 'warning' if null_percentage < 50 else 'failed',
            'null_percentage': float(null_percentage),
            'null_count': int(data.isnull().sum())
        }
        
        # Overall status
        failed_checks = [check for check in validation_results['checks'].values() if check['status'] == 'failed']
        if failed_checks:
            validation_results['overall_status'] = 'failed'
        else:
            warning_checks = [check for check in validation_results['checks'].values() if check['status'] == 'warning']
            if warning_checks:
                validation_results['overall_status'] = 'warning'
        
        return validation_results
    
    def generate_feature_documentation(self) -> str:
        """Generate comprehensive feature documentation."""
        logger.info("📚 Generating feature documentation...")
        
        doc_lines = []
        doc_lines.append("# Recommndr Feature Documentation")
        doc_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        doc_lines.append("")
        
        # Group features by category
        categories = {
            'User Features': [f for f in self.feature_definitions.keys() if f.startswith('user_')],
            'Item Features': [f for f in self.feature_definitions.keys() if f.startswith('item_')],
            'Contextual Features': ['hour_of_day', 'day_of_week', 'is_weekend', 'session_duration_minutes'],
            'Interaction Features': [f for f in self.feature_definitions.keys() if 'interaction' in f or 'session' in f]
        }
        
        for category, features in categories.items():
            if not features:
                continue
                
            doc_lines.append(f"## {category}")
            doc_lines.append("")
            
            for feature_name in features:
                if feature_name in self.feature_definitions:
                    metadata = self.feature_definitions[feature_name]
                    
                    doc_lines.append(f"### {feature_name}")
                    doc_lines.append("")
                    doc_lines.append(f"**Description**: {metadata.get('description', 'N/A')}")
                    doc_lines.append(f"**Data Type**: {metadata.get('data_type', 'N/A')}")
                    doc_lines.append(f"**Source**: {metadata.get('source_table', 'N/A')}")
                    doc_lines.append(f"**Business Meaning**: {metadata.get('business_meaning', 'N/A')}")
                    doc_lines.append(f"**Update Frequency**: {metadata.get('update_frequency', 'N/A')}")
                    doc_lines.append(f"**Owner**: {metadata.get('owner', 'N/A')}")
                    
                    if 'valid_range' in metadata:
                        doc_lines.append(f"**Valid Range**: {metadata['valid_range']}")
                    
                    if 'valid_values' in metadata:
                        valid_values_str = ', '.join(str(v) for v in metadata['valid_values'])
                        doc_lines.append(f"**Valid Values**: {valid_values_str}")
                    
                    doc_lines.append("")
        
        # Add feature statistics summary
        doc_lines.append("## Feature Statistics Summary")
        doc_lines.append("")
        
        if self.feature_statistics:
            doc_lines.append("| Feature | Data Points | Null % | Type | Notes |")
            doc_lines.append("|---------|-------------|--------|------|-------|")
            
            for feature_name, stats in self.feature_statistics.items():
                data_points = stats.get('data_points', 0)
                null_pct = stats.get('null_percentage', 0)
                data_type = self.feature_definitions.get(feature_name, {}).get('data_type', 'unknown')
                
                notes = "OK"
                if null_pct > 10:
                    notes = "High nulls"
                
                doc_lines.append(f"| {feature_name} | {data_points:,} | {null_pct:.1f}% | {data_type} | {notes} |")
        
        documentation = "\n".join(doc_lines)
        
        # Save documentation
        doc_file = self.metadata_dir / "feature_documentation.md"
        with open(doc_file, 'w') as f:
            f.write(documentation)
        
        logger.info(f"✅ Feature documentation generated: {doc_file}")
        return documentation
    
    def _save_feature_metadata(self, feature_name: str, metadata: Dict[str, Any]):
        """Save feature metadata to disk."""
        metadata_file = self.metadata_dir / f"{feature_name}_metadata.json"
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """Get summary of all registered features."""
        return {
            'total_features': len(self.feature_definitions),
            'features_with_statistics': len(self.feature_statistics),
            'features_with_lineage': len(self.feature_lineage),
            'last_updated': max([meta.get('registration_time', 0) for meta in self.feature_definitions.values()]) if self.feature_definitions else 0,
            'feature_categories': {
                'user_features': len([f for f in self.feature_definitions.keys() if f.startswith('user_')]),
                'item_features': len([f for f in self.feature_definitions.keys() if f.startswith('item_')]),
                'contextual_features': len([f for f in self.feature_definitions.keys() if any(ctx in f for ctx in ['hour', 'day', 'weekend', 'session'])]),
                'interaction_features': len([f for f in self.feature_definitions.keys() if 'interaction' in f])
            }
        }
