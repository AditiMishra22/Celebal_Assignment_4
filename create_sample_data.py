import pandas as pd
import numpy as np

# Create a comprehensive sample dataset for EDA
np.random.seed(42)

# Generate sample data - Customer Sales Dataset
n_samples = 1000

data = {
    'customer_id': range(1, n_samples + 1),
    'age': np.random.normal(40, 12, n_samples).astype(int),
    'income': np.random.normal(50000, 20000, n_samples),
    'spending_score': np.random.normal(50, 20, n_samples),
    'purchase_amount': np.random.normal(500, 200, n_samples),
    'years_customer': np.random.exponential(3, n_samples),
    'num_purchases': np.random.poisson(10, n_samples),
    'satisfaction_score': np.random.normal(7, 2, n_samples),
    'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
    'product_category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Sports', 'Books'], n_samples),
    'gender': np.random.choice(['Male', 'Female'], n_samples),
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
    'marital_status': np.random.choice(['Single', 'Married', 'Divorced'], n_samples)
}

# Add some correlations to make the data more realistic
for i in range(n_samples):
    # Higher income tends to lead to higher spending
    if data['income'][i] > 60000:
        data['spending_score'][i] += np.random.normal(15, 5)
        data['purchase_amount'][i] += np.random.normal(200, 50)
    
    # Older customers tend to have higher income
    if data['age'][i] > 50:
        data['income'][i] += np.random.normal(10000, 5000)
    
    # Satisfaction affects spending
    if data['satisfaction_score'][i] > 8:
        data['spending_score'][i] += np.random.normal(10, 3)

# Ensure realistic bounds
data['age'] = np.clip(data['age'], 18, 80)
data['income'] = np.clip(data['income'], 20000, 150000)
data['spending_score'] = np.clip(data['spending_score'], 0, 100)
data['purchase_amount'] = np.clip(data['purchase_amount'], 50, 2000)
data['years_customer'] = np.clip(data['years_customer'], 0.1, 20)
data['satisfaction_score'] = np.clip(data['satisfaction_score'], 1, 10)

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('customer_sales_data.csv', index=False)
print("Dataset created successfully!")
print(f"Shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())
print("\nDataset info:")
print(df.info())
