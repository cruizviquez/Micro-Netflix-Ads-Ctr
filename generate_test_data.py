import pandas as pd
import numpy as np
import random

def generate_ctr_dataset(n_samples=1000):
    # Set random seed for reproducibility
    np.random.seed(42)
    
    data = []
    
    for _ in range(n_samples):
        # Generate user features
        age = np.random.randint(18, 70)
        gender = np.random.choice(['M', 'F'])
        device = np.random.choice(['mobile', 'desktop', 'smart_tv', 'tablet'])
        
        # Viewing behavior
        watch_time_hours = np.random.exponential(2.5)  # Average 2.5 hours
        genres = ['action', 'comedy', 'drama', 'documentary', 'sci-fi', 'romance']
        preferred_genre = np.random.choice(genres)
        
        # Time features
        hour_of_day = np.random.randint(0, 24)
        day_of_week = np.random.randint(0, 7)
        
        # Calculate CTR with some logic
        base_ctr = 0.02  # 2% base CTR
        
        # Age factor
        if 25 <= age <= 45:
            base_ctr *= 1.5
            
        # Device factor
        if device == 'smart_tv':
            base_ctr *= 1.3
        elif device == 'mobile':
            base_ctr *= 1.1
            
        # Time factor
        if 19 <= hour_of_day <= 22:  # Prime time
            base_ctr *= 1.4
            
        # Genre match (simulated)
        if preferred_genre in ['action', 'sci-fi'] and age < 40:
            base_ctr *= 1.2
            
        # Add noise
        ctr = base_ctr * np.random.uniform(0.7, 1.3)
        clicked = 1 if np.random.random() < ctr else 0
        
        data.append({
            'user_id': f'user_{_}',
            'age': age,
            'gender': gender,
            'device': device,
            'watch_time_hours': round(watch_time_hours, 2),
            'preferred_genre': preferred_genre,
            'hour_of_day': hour_of_day,
            'day_of_week': day_of_week,
            'clicked': clicked,
            'actual_ctr': round(ctr, 4)
        })
    
    df = pd.DataFrame(data)
    df.to_csv('sample_ctr_data.csv', index=False)
    print(f"Generated {n_samples} samples to 'sample_ctr_data.csv'")
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"Overall CTR: {df['clicked'].mean():.2%}")
    print(f"CTR by device:")
    print(df.groupby('device')['clicked'].mean().sort_values(ascending=False))
    
    return df

if __name__ == "__main__":
    df = generate_ctr_dataset(1000)