#!/usr/bin/env python3
"""
Dataset Inspection Tool for SetRetrieval Project
Analyzes the actual content of processed .pkl files in datasets/
"""

import os
import pickle
import gzip
import numpy as np
import json
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

def load_pickle_data(file_path):
    """Load data from pickle file and return all components"""
    try:
        with open(file_path, 'rb') as f:
            data = []
            try:
                while True:
                    component = pickle.load(f)
                    data.append(component)
            except EOFError:
                pass
        
        print(f"  📦 Loaded {len(data)} components from {os.path.basename(file_path)}")
        for i, component in enumerate(data):
            if isinstance(component, (list, np.ndarray)):
                print(f"    Component {i}: {type(component).__name__} with shape/length: {np.array(component).shape if hasattr(np.array(component), 'shape') else len(component)}")
            else:
                print(f"    Component {i}: {type(component).__name__}")
        
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def load_category_centers(file_path):
    """Load category centers from compressed pickle"""
    try:
        with gzip.open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading category centers: {e}")
        return None

def analyze_dataset_split(data_components, split_name):
    """Analyze a single dataset split (train/val/test)"""
    print(f"\n{'='*50}")
    print(f"📊 {split_name.upper()} SPLIT ANALYSIS")
    print(f"{'='*50}")
    
    # Handle different pickle structures
    if len(data_components) == 1:
        single_component = data_components[0]
        print(f"🔍 Single component structure: {type(single_component)}")
        
        if isinstance(single_component, dict):
            # Dictionary structure
            data_dict = single_component
            print(f"🔍 Dictionary structure with keys: {list(data_dict.keys())}")
            
            query_features = data_dict.get('query_features')
            target_features = data_dict.get('target_features')
            query_categories = data_dict.get('query_categories')
            target_categories = data_dict.get('target_categories')
            query_item_ids = data_dict.get('query_item_ids')
            target_item_ids = data_dict.get('target_item_ids')
            
        elif isinstance(single_component, (list, tuple)) and len(single_component) >= 6:
            # List/tuple of components - need to figure out the mapping
            print(f"  Treating as {type(single_component).__name__} of {len(single_component)} components")
            
            # Print component types and shapes first
            for i, comp in enumerate(single_component):
                comp_array = np.array(comp) if comp is not None else None
                if comp_array is not None:
                    print(f"    Component {i}: {type(comp).__name__} -> shape {comp_array.shape}")
                else:
                    print(f"    Component {i}: None")
            
            # Based on the output, let's try to map components correctly
            # From the shapes we see:
            # Component 0: (48497, 10, 512) - query_features  
            # Component 1: (48497, 10, 512) - target_features
            # Component 2: (48497,) - ??? 
            # Component 3: (48497, 10) - query_categories
            # Component 4: (48497, 10) - target_categories  
            # Component 5: (48497,) - ???
            
            query_features = single_component[0]
            target_features = single_component[1]
            query_categories = single_component[3]  # Changed from 2 to 3
            target_categories = single_component[4]  # Changed from 3 to 4
            
            # Try to identify item IDs
            query_item_ids = single_component[2] if len(single_component) > 2 else None
            target_item_ids = single_component[5] if len(single_component) > 5 else None
                    
        else:
            print(f"  ⚠️ Unexpected single component: {type(single_component)} with length {len(single_component) if hasattr(single_component, '__len__') else 'N/A'}")
            
            # Try to inspect the structure more
            if hasattr(single_component, '__len__') and len(single_component) > 0:
                print(f"  First element type: {type(single_component[0])}")
                if hasattr(single_component[0], 'shape'):
                    print(f"  First element shape: {single_component[0].shape}")
            return None
                
    elif len(data_components) >= 6:
        # Standard 6-component structure
        print(f"🔍 Standard 6-component structure")
        query_features = data_components[0]
        target_features = data_components[1]
        query_categories = data_components[2]
        target_categories = data_components[3]
        query_item_ids = data_components[4]
        target_item_ids = data_components[5]
    else:
        print(f"⚠️ Unexpected number of components: {len(data_components)}")
        return None
    
    # Verify we have the required components
    required_components = [query_features, target_features, query_categories, target_categories]
    if any(comp is None for comp in required_components):
        print(f"⚠️ Missing required components")
        return None
    
    # Convert to numpy arrays for analysis
    try:
        query_features = np.array(query_features)
        target_features = np.array(target_features)
        query_categories = np.array(query_categories)
        target_categories = np.array(target_categories)
    except Exception as e:
        print(f"⚠️ Error converting to numpy arrays: {e}")
        return None
    
    # Basic shape information
    print(f"\n🔍 Data Shapes:")
    print(f"  Query features:    {query_features.shape}")
    print(f"  Target features:   {target_features.shape}")
    print(f"  Query categories:  {query_categories.shape}")
    print(f"  Target categories: {target_categories.shape}")
    if query_item_ids is not None:
        try:
            print(f"  Query item IDs:    {np.array(query_item_ids).shape}")
        except:
            print(f"  Query item IDs:    {type(query_item_ids)} (could not convert to array)")
    if target_item_ids is not None:
        try:
            print(f"  Target item IDs:   {np.array(target_item_ids).shape}")
        except:
            print(f"  Target item IDs:   {type(target_item_ids)} (could not convert to array)")
    
    # Handle different category shapes
    if len(query_categories.shape) == 1:
        print(f"⚠️ Query categories has unexpected 1D shape, trying to find proper categories...")
        # Check if target_categories has the right shape
        if len(target_categories.shape) == 2:
            print(f"  Using target_categories shape {target_categories.shape} as reference")
            # Create dummy query categories with same shape
            query_categories = np.zeros_like(target_categories)
        else:
            print(f"⚠️ Both category arrays have unexpected shapes")
            return None
    
    if len(target_categories.shape) == 1:
        print(f"⚠️ Target categories has unexpected 1D shape, trying to find proper categories...")
        # Check if query_categories has the right shape  
        if len(query_categories.shape) == 2:
            print(f"  Using query_categories shape {query_categories.shape} as reference")
            # Create dummy target categories with same shape
            target_categories = np.zeros_like(query_categories)
        else:
            print(f"⚠️ Both category arrays have unexpected shapes")
            return None
    
    # Validate shapes
    if len(query_features.shape) != 3 or len(target_features.shape) != 3:
        print(f"⚠️ Unexpected feature array dimensions")
        return None
        
    n_sets, max_items, feature_dim = query_features.shape
    
    print(f"\n📈 Dataset Statistics:")
    print(f"  Total sets: {n_sets:,}")
    print(f"  Max items per set: {max_items}")
    print(f"  Feature dimension: {feature_dim}")
    
    # Category analysis - handle string categories
    print(f"\n🏷️ Category Analysis:")
    
    # Check if categories are strings and convert if needed
    if query_categories.dtype.kind in ['U', 'S', 'O']:  # Unicode, byte string, or object
        print(f"  ⚠️ Categories appear to be strings, attempting conversion...")
        try:
            # Try to convert string categories to integers
            unique_cats = np.unique(query_categories.flatten())
            cat_to_int = {cat: i+1 for i, cat in enumerate(unique_cats) if str(cat) != '0' and str(cat) != 'nan'}
            print(f"  Found string categories: {list(cat_to_int.keys())}")
            
            # Convert to integers
            query_cats_int = np.zeros_like(query_categories, dtype=int)
            target_cats_int = np.zeros_like(target_categories, dtype=int)
            
            for cat_str, cat_int in cat_to_int.items():
                query_cats_int[query_categories == cat_str] = cat_int
                target_cats_int[target_categories == cat_str] = cat_int
                
            query_categories = query_cats_int
            target_categories = target_cats_int
            
        except Exception as e:
            print(f"  ⚠️ Failed to convert string categories: {e}")
            return None
    
    # Flatten categories and remove padding (0s)
    all_query_cats = query_categories.flatten()
    all_target_cats = target_categories.flatten()
    all_categories = np.concatenate([all_query_cats, all_target_cats])
    
    # Remove padding (category 0)
    valid_categories = all_categories[all_categories > 0]
    category_counts = Counter(valid_categories)
    
    print(f"  Valid categories found: {sorted(category_counts.keys())}")
    print(f"  Total valid items: {len(valid_categories):,}")
    print(f"  Unique categories: {len(category_counts)}")
    
    print(f"\n  Category distribution:")
    for cat_id in sorted(category_counts.keys()):
        count = category_counts[cat_id]
        percentage = (count / len(valid_categories)) * 100
        print(f"    Category {cat_id}: {count:,} items ({percentage:.1f}%)")
    
    # Items per set analysis
    def count_valid_items_per_set(categories):
        """Count non-zero categories per set"""
        return np.sum(categories > 0, axis=1)
    
    query_items_per_set = count_valid_items_per_set(query_categories)
    target_items_per_set = count_valid_items_per_set(target_categories)
    total_items_per_set = query_items_per_set + target_items_per_set
    
    print(f"\n📦 Items per Set Statistics:")
    print(f"  Query items per set - Mean: {query_items_per_set.mean():.1f}, Std: {query_items_per_set.std():.1f}")
    print(f"  Target items per set - Mean: {target_items_per_set.mean():.1f}, Std: {target_items_per_set.std():.1f}")
    print(f"  Total items per set - Mean: {total_items_per_set.mean():.1f}, Std: {total_items_per_set.std():.1f}")
    print(f"  Min items per set: {total_items_per_set.min()}")
    print(f"  Max items per set: {total_items_per_set.max()}")
    
    # Feature analysis
    print(f"\n🎯 Feature Analysis:")
    valid_query_features = query_features[query_categories > 0]
    valid_target_features = target_features[target_categories > 0]
    all_valid_features = np.concatenate([valid_query_features, valid_target_features])
    
    print(f"  Feature statistics:")
    print(f"    Mean: {all_valid_features.mean():.4f}")
    print(f"    Std: {all_valid_features.std():.4f}")
    print(f"    Min: {all_valid_features.min():.4f}")
    print(f"    Max: {all_valid_features.max():.4f}")
    
    # Check if features are normalized
    feature_norms = np.linalg.norm(all_valid_features, axis=1)
    print(f"    Feature norms - Mean: {feature_norms.mean():.4f}, Std: {feature_norms.std():.4f}")
    if np.allclose(feature_norms, 1.0, atol=1e-3):
        print(f"    ✅ Features appear to be L2-normalized")
    else:
        print(f"    ⚠️ Features may not be normalized")
    
    return {
        'n_sets': n_sets,
        'max_items': max_items,
        'feature_dim': feature_dim,
        'category_counts': category_counts,
        'items_per_set_stats': {
            'query_mean': query_items_per_set.mean(),
            'target_mean': target_items_per_set.mean(),
            'total_mean': total_items_per_set.mean(),
            'total_min': total_items_per_set.min(),
            'total_max': total_items_per_set.max()
        },
        'feature_stats': {
            'mean': all_valid_features.mean(),
            'std': all_valid_features.std(),
            'normalized': np.allclose(feature_norms, 1.0, atol=1e-3)
        }
    }

def analyze_dataset(dataset_path, dataset_name):
    """Analyze entire dataset (all splits)"""
    print(f"\n{'='*60}")
    print(f"🔬 ANALYZING {dataset_name.upper()} DATASET")
    print(f"📁 Path: {dataset_path}")
    print(f"{'='*60}")
    
    # Check if directory exists
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset directory not found: {dataset_path}")
        return
    
    # List all files
    files = os.listdir(dataset_path)
    print(f"\n📋 Files found: {files}")
    
    results = {}
    
    # Analyze each split
    for split in ['train', 'validation', 'test']:
        pkl_file = os.path.join(dataset_path, f"{split}.pkl")
        if os.path.exists(pkl_file):
            print(f"\n🔍 Loading {split}.pkl...")
            data = load_pickle_data(pkl_file)
            if data:
                split_result = analyze_dataset_split(data, split)
                results[split] = split_result
        else:
            print(f"⚠️ {split}.pkl not found")
    
    # Analyze category centers if available
    category_centers_file = os.path.join(dataset_path, "category_centers.pkl.gz")
    if os.path.exists(category_centers_file):
        print(f"\n🎯 Analyzing category centers...")
        centers = load_category_centers(category_centers_file)
        if centers:
            print(f"  Number of category centers: {len(centers)}")
            print(f"  Category IDs: {sorted(centers.keys())}")
            if len(centers) > 0:
                # Handle different center formats
                try:
                    first_center = list(centers.values())[0]
                    if isinstance(first_center, np.ndarray):
                        center_shapes = [center.shape for center in centers.values()]
                        print(f"  Center dimensions: {center_shapes[0]} (all should be same)")
                        
                        # Check if all centers have same dimension
                        if len(set(center_shapes)) == 1:
                            print(f"  ✅ All centers have consistent dimensions")
                        else:
                            print(f"  ⚠️ Inconsistent center dimensions: {set(center_shapes)}")
                    elif isinstance(first_center, list):
                        center_lengths = [len(center) for center in centers.values()]
                        print(f"  Center lengths: {center_lengths[0]} (all should be same)")
                        
                        if len(set(center_lengths)) == 1:
                            print(f"  ✅ All centers have consistent lengths")
                        else:
                            print(f"  ⚠️ Inconsistent center lengths: {set(center_lengths)}")
                    else:
                        print(f"  Center format: {type(first_center)}")
                        print(f"  First center preview: {str(first_center)[:100]}...")
                        
                except Exception as e:
                    print(f"  ⚠️ Error analyzing centers: {e}")
                    print(f"  Centers type: {type(centers)}")
                    if centers:
                        print(f"  First center type: {type(list(centers.values())[0])}")
    
    # Load additional metadata if available
    metadata_files = ['category_mapping.json', 'category_info.json']
    for metadata_file in metadata_files:
        metadata_path = os.path.join(dataset_path, metadata_file)
        if os.path.exists(metadata_path):
            print(f"\n📄 Found metadata: {metadata_file}")
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    print(f"  Content preview: {str(metadata)[:200]}...")
                    
                    # More detailed analysis for specific metadata
                    if metadata_file == 'category_info.json' and isinstance(metadata, dict):
                        if 'main_categories_def' in metadata:
                            cats = metadata['main_categories_def']
                            print(f"\n  🏷️ Category definitions from metadata:")
                            for cat_id, cat_name in sorted(cats.items(), key=lambda x: int(x[0])):
                                print(f"    {cat_id}: {cat_name}")
                        
                        if 'main_category_stats' in metadata:
                            stats = metadata['main_category_stats']
                            print(f"\n  📊 Category statistics from metadata:")
                            total_items = sum(stats.values())
                            for cat_id, count in sorted(stats.items(), key=lambda x: int(x[0])):
                                percentage = (count / total_items) * 100
                                print(f"    Category {cat_id}: {count:,} ({percentage:.1f}%)")
                    
                    elif metadata_file == 'category_mapping.json' and isinstance(metadata, dict):
                        if 'categories' in metadata:
                            cats = metadata['categories']
                            print(f"\n  🏷️ Category mapping from metadata:")
                            for cat_id, cat_name in sorted(cats.items(), key=lambda x: int(x[0])):
                                print(f"    {cat_id}: {cat_name}")
                                
            except Exception as e:
                print(f"  Error reading metadata: {e}")
    
    return results

def compare_datasets(datasets_results):
    """Compare statistics across datasets"""
    print(f"\n{'='*60}")
    print(f"🆚 DATASET COMPARISON")
    print(f"{'='*60}")
    
    for dataset_name, results in datasets_results.items():
        print(f"\n{dataset_name}:")
        if results and 'train' in results and results['train'] is not None:
            train_stats = results['train']
            print(f"  Training sets: {train_stats['n_sets']:,}")
            print(f"  Categories: {len(train_stats['category_counts'])}")
            print(f"  Avg items/set: {train_stats['items_per_set_stats']['total_mean']:.1f}")
            print(f"  Feature dim: {train_stats['feature_dim']}")
            print(f"  Normalized: {train_stats['feature_stats']['normalized']}")
            
            # Show top categories
            sorted_cats = sorted(train_stats['category_counts'].items(), key=lambda x: x[1], reverse=True)
            print(f"  Top categories:")
            for cat_id, count in sorted_cats[:3]:
                print(f"    Category {cat_id}: {count:,} items")
        else:
            print(f"  ❌ No valid training data found")
    
    # Cross-dataset comparison
    valid_datasets = {name: results for name, results in datasets_results.items() 
                     if results and 'train' in results and results['train'] is not None}
    
    if len(valid_datasets) >= 2:
        print(f"\n📊 Cross-Dataset Analysis:")
        dataset_names = list(valid_datasets.keys())
        
        for i, name1 in enumerate(dataset_names):
            for name2 in dataset_names[i+1:]:
                stats1 = valid_datasets[name1]['train']
                stats2 = valid_datasets[name2]['train']
                
                print(f"\n  {name1} vs {name2}:")
                print(f"    Sets ratio: {stats1['n_sets'] / stats2['n_sets']:.2f}x")
                print(f"    Categories: {len(stats1['category_counts'])} vs {len(stats2['category_counts'])}")
                print(f"    Items/set: {stats1['items_per_set_stats']['total_mean']:.1f} vs {stats2['items_per_set_stats']['total_mean']:.1f}")
                
                # Feature similarity
                feat1 = stats1['feature_stats']
                feat2 = stats2['feature_stats']
                print(f"    Feature mean diff: {abs(feat1['mean'] - feat2['mean']):.4f}")
                print(f"    Feature std diff: {abs(feat1['std'] - feat2['std']):.4f}")

def create_visualizations(datasets_results, output_dir="dataset_analysis"):
    """Create visualization plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.style.use('default')
    
    for dataset_name, results in datasets_results.items():
        if results and 'train' in results and results['train'] is not None:
            # Category distribution plot
            category_counts = results['train']['category_counts']
            
            plt.figure(figsize=(10, 6))
            categories = sorted(category_counts.keys())
            counts = [category_counts[cat] for cat in categories]
            
            plt.bar(categories, counts, alpha=0.7, color='steelblue')
            plt.title(f'{dataset_name} - Category Distribution (Training Set)')
            plt.xlabel('Category ID')
            plt.ylabel('Number of Items')
            plt.xticks(categories)
            
            # Add count labels on bars
            for i, count in enumerate(counts):
                plt.text(categories[i], count + max(counts)*0.01, 
                        f'{count:,}', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{dataset_name}_category_distribution.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Saved category distribution plot: {output_dir}/{dataset_name}_category_distribution.png")

def main():
    """Main analysis function"""
    # Dataset paths
    datasets_dir = "datasets"
    dataset_names = ["IQON3000", "DeepFurniture"]
    
    datasets_results = {}
    
    # Analyze each dataset
    for dataset_name in dataset_names:
        dataset_path = os.path.join(datasets_dir, dataset_name)
        try:
            results = analyze_dataset(dataset_path, dataset_name)
            if results:
                datasets_results[dataset_name] = results
        except Exception as e:
            print(f"❌ Error analyzing {dataset_name}: {e}")
            datasets_results[dataset_name] = None
    
    # Compare datasets
    if len(datasets_results) > 0:
        compare_datasets(datasets_results)
    
    # Create summary table
    print(f"\n{'='*80}")
    print(f"📋 DATASET SUMMARY TABLE")
    print(f"{'='*80}")
    
    # Header
    print(f"{'Dataset':<15} {'Train Sets':<12} {'Val Sets':<10} {'Test Sets':<11} {'Categories':<12} {'Avg Items/Set':<15} {'Feature Dim':<12}")
    print(f"{'-'*15} {'-'*12} {'-'*10} {'-'*11} {'-'*12} {'-'*15} {'-'*12}")
    
    for dataset_name, results in datasets_results.items():
        if results and any(results.get(split) for split in ['train', 'validation', 'test']):
            train_stats = results.get('train')
            val_stats = results.get('validation') 
            test_stats = results.get('test')
            
            train_sets = train_stats['n_sets'] if train_stats else 'N/A'
            val_sets = val_stats['n_sets'] if val_stats else 'N/A'
            test_sets = test_stats['n_sets'] if test_stats else 'N/A'
            categories = len(train_stats['category_counts']) if train_stats else 'N/A'
            avg_items = f"{train_stats['items_per_set_stats']['total_mean']:.1f}" if train_stats else 'N/A'
            feature_dim = train_stats['feature_dim'] if train_stats else 'N/A'
            
            print(f"{dataset_name:<15} {str(train_sets):<12} {str(val_sets):<10} {str(test_sets):<11} {str(categories):<12} {avg_items:<15} {str(feature_dim):<12}")
        else:
            print(f"{dataset_name:<15} {'ERROR':<12} {'ERROR':<10} {'ERROR':<11} {'ERROR':<12} {'ERROR':<15} {'ERROR':<12}")
    
    # Create visualizations
    try:
        create_visualizations(datasets_results)
    except Exception as e:
        print(f"⚠️ Could not create visualizations: {e}")
    
    # Final validation summary
    print(f"\n{'='*60}")
    print(f"✅ ANALYSIS COMPLETE - VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    for dataset_name, results in datasets_results.items():
        print(f"\n{dataset_name}:")
        if results and any(results.get(split) for split in ['train', 'validation', 'test']):
            print(f"  ✅ Successfully analyzed")
            
            # Check data consistency
            splits_analyzed = [split for split in ['train', 'validation', 'test'] if results.get(split)]
            print(f"  📊 Splits analyzed: {', '.join(splits_analyzed)}")
            
            if len(splits_analyzed) > 1:
                # Check consistency across splits
                feature_dims = [results[split]['feature_dim'] for split in splits_analyzed]
                max_items = [results[split]['max_items'] for split in splits_analyzed]
                
                if len(set(feature_dims)) == 1:
                    print(f"  ✅ Consistent feature dimensions across splits: {feature_dims[0]}")
                else:
                    print(f"  ⚠️ Inconsistent feature dimensions: {feature_dims}")
                
                if len(set(max_items)) == 1:
                    print(f"  ✅ Consistent max items across splits: {max_items[0]}")
                else:
                    print(f"  ⚠️ Inconsistent max items: {max_items}")
            
            # Validate against expected config
            expected_cats = {'IQON3000': 7, 'DeepFurniture': 11}
            if dataset_name in expected_cats:
                train_stats = results.get('train')
                if train_stats:
                    actual_cats = len(train_stats['category_counts'])
                    expected = expected_cats[dataset_name]
                    
                    if actual_cats == expected:
                        print(f"  ✅ Category count matches expectation: {actual_cats}")
                    else:
                        print(f"  ⚠️ Category count mismatch: got {actual_cats}, expected {expected}")
                        
                    # Show actual categories found
                    cats_found = sorted(train_stats['category_counts'].keys())
                    print(f"  🏷️ Categories found: {cats_found}")
        else:
            print(f"  ❌ Analysis failed or no data found")

if __name__ == "__main__":
    main()