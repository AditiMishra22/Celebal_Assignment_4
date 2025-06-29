from flask import Flask, render_template, send_file
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from scipy import stats
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

app = Flask(__name__)

# Load dataset
df = pd.read_csv('customer_sales_data.csv')

# Generate and save comprehensive EDA visualizations

def save_viz():
    # Set style for better looking plots
    sns.set_style("whitegrid")
    
    # 1. Correlation Heatmap - Main analysis
    if not os.path.exists('static/correlation_heatmap.png'):
        plt.figure(figsize=(12, 10))
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Generate heatmap
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, fmt='.2f', cbar_kws={"shrink": .8})
        plt.title('Correlation Heatmap - Customer Sales Data', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig('static/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 2. Distribution plots for key numerical variables
    if not os.path.exists('static/distributions.png'):
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Age distribution
        sns.histplot(df['age'], kde=True, ax=axes[0,0], color='skyblue')
        axes[0,0].set_title('Age Distribution', fontsize=14)
        axes[0,0].set_xlabel('Age')
        
        # Income distribution
        sns.histplot(df['income'], kde=True, ax=axes[0,1], color='lightgreen')
        axes[0,1].set_title('Income Distribution', fontsize=14)
        axes[0,1].set_xlabel('Income ($)')
        
        # Spending Score distribution
        sns.histplot(df['spending_score'], kde=True, ax=axes[0,2], color='salmon')
        axes[0,2].set_title('Spending Score Distribution', fontsize=14)
        axes[0,2].set_xlabel('Spending Score')
        
        # Purchase Amount distribution
        sns.histplot(df['purchase_amount'], kde=True, ax=axes[1,0], color='gold')
        axes[1,0].set_title('Purchase Amount Distribution', fontsize=14)
        axes[1,0].set_xlabel('Purchase Amount ($)')
        
        # Years Customer distribution
        sns.histplot(df['years_customer'], kde=True, ax=axes[1,1], color='plum')
        axes[1,1].set_title('Years as Customer Distribution', fontsize=14)
        axes[1,1].set_xlabel('Years as Customer')
        
        # Satisfaction Score distribution
        sns.histplot(df['satisfaction_score'], kde=True, ax=axes[1,2], color='lightcoral')
        axes[1,2].set_title('Satisfaction Score Distribution', fontsize=14)
        axes[1,2].set_xlabel('Satisfaction Score')
        
        plt.tight_layout()
        plt.savefig('static/distributions.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 3. Box plots for outlier detection
    if not os.path.exists('static/boxplots.png'):
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        numeric_columns = ['age', 'income', 'spending_score', 'purchase_amount', 'years_customer', 'satisfaction_score']
        colors = ['skyblue', 'lightgreen', 'salmon', 'gold', 'plum', 'lightcoral']
        
        for i, (col, color) in enumerate(zip(numeric_columns, colors)):
            row, col_idx = i // 3, i % 3
            sns.boxplot(y=df[col], ax=axes[row, col_idx], color=color)
            axes[row, col_idx].set_title(f'{col.replace("_", " ").title()} - Outlier Detection', fontsize=14)
        
        plt.tight_layout()
        plt.savefig('static/boxplots.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 4. Categorical variable analysis
    if not os.path.exists('static/categorical_analysis.png'):
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # Gender distribution
        df['gender'].value_counts().plot.pie(ax=axes[0,0], autopct='%1.1f%%', 
                                           colors=['lightblue', 'pink'])
        axes[0,0].set_title('Gender Distribution', fontsize=14)
        axes[0,0].set_ylabel('')
        
        # Region distribution
        sns.countplot(data=df, x='region', ax=axes[0,1], palette='Set2')
        axes[0,1].set_title('Customer Distribution by Region', fontsize=14)
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Education distribution
        sns.countplot(data=df, y='education', ax=axes[0,2], palette='Set1')
        axes[0,2].set_title('Education Level Distribution', fontsize=14)
        
        # Product Category distribution
        sns.countplot(data=df, x='product_category', ax=axes[1,0], palette='viridis')
        axes[1,0].set_title('Product Category Distribution', fontsize=14)
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Marital Status distribution
        sns.countplot(data=df, x='marital_status', ax=axes[1,1], palette='pastel')
        axes[1,1].set_title('Marital Status Distribution', fontsize=14)
        
        # Income vs Spending by Gender
        sns.scatterplot(data=df, x='income', y='spending_score', hue='gender', 
                       ax=axes[1,2], alpha=0.6)
        axes[1,2].set_title('Income vs Spending Score by Gender', fontsize=14)
        
        plt.tight_layout()
        plt.savefig('static/categorical_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 5. Advanced relationship analysis
    if not os.path.exists('static/relationship_analysis.png'):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Age vs Income scatter
        sns.scatterplot(data=df, x='age', y='income', alpha=0.6, ax=axes[0,0])
        sns.regplot(data=df, x='age', y='income', ax=axes[0,0], scatter=False, color='red')
        axes[0,0].set_title('Age vs Income Relationship', fontsize=14)
        
        # Income vs Purchase Amount
        sns.scatterplot(data=df, x='income', y='purchase_amount', alpha=0.6, ax=axes[0,1])
        sns.regplot(data=df, x='income', y='purchase_amount', ax=axes[0,1], scatter=False, color='red')
        axes[0,1].set_title('Income vs Purchase Amount', fontsize=14)
        
        # Spending Score vs Satisfaction
        sns.scatterplot(data=df, x='spending_score', y='satisfaction_score', alpha=0.6, ax=axes[1,0])
        sns.regplot(data=df, x='spending_score', y='satisfaction_score', ax=axes[1,0], scatter=False, color='red')
        axes[1,0].set_title('Spending Score vs Satisfaction', fontsize=14)
        
        # Years Customer vs Number of Purchases
        sns.scatterplot(data=df, x='years_customer', y='num_purchases', alpha=0.6, ax=axes[1,1])
        sns.regplot(data=df, x='years_customer', y='num_purchases', ax=axes[1,1], scatter=False, color='red')
        axes[1,1].set_title('Years as Customer vs Number of Purchases', fontsize=14)
        
        plt.tight_layout()
        plt.savefig('static/relationship_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 6. Statistical summary visualization
    if not os.path.exists('static/statistical_summary.png'):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Income by Education Level
        sns.boxplot(data=df, x='education', y='income', ax=axes[0,0])
        axes[0,0].set_title('Income Distribution by Education Level', fontsize=14)
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Spending Score by Product Category
        sns.violinplot(data=df, x='product_category', y='spending_score', ax=axes[0,1])
        axes[0,1].set_title('Spending Score by Product Category', fontsize=14)
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Purchase Amount by Region
        sns.boxplot(data=df, x='region', y='purchase_amount', ax=axes[1,0])
        axes[1,0].set_title('Purchase Amount by Region', fontsize=14)
        
        # Satisfaction by Marital Status
        sns.barplot(data=df, x='marital_status', y='satisfaction_score', ax=axes[1,1], errorbar=('ci', 95))
        axes[1,1].set_title('Average Satisfaction by Marital Status', fontsize=14)
        
        plt.tight_layout()
        plt.savefig('static/statistical_summary.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 7. Create pairplot for numerical variables
    if not os.path.exists('static/pairplot.png'):
        numeric_cols = ['age', 'income', 'spending_score', 'purchase_amount', 'satisfaction_score']
        pairplot_df = df[numeric_cols].sample(500)  # Sample for performance
        
        plt.figure(figsize=(15, 15))
        sns.pairplot(pairplot_df, diag_kind='kde', plot_kws={'alpha': 0.6})
        plt.suptitle('Pairwise Relationships - Numerical Variables', y=1.02, fontsize=16)
        plt.savefig('static/pairplot.png', dpi=300, bbox_inches='tight')
        plt.close()

@app.route('/')
def index():
    save_viz()
    
    # Comprehensive dataset information
    numeric_df = df.select_dtypes(include=[np.number])
    categorical_df = df.select_dtypes(include=['object'])
    
    # Basic info
    info = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'missing': df.isnull().sum().to_dict(),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'numeric_columns': numeric_df.columns.tolist(),
        'categorical_columns': categorical_df.columns.tolist()
    }
    
    # Statistical summary for numeric columns
    stats_summary = {}
    for col in numeric_df.columns:
        stats_summary[col] = {
            'mean': round(df[col].mean(), 2),
            'median': round(df[col].median(), 2),
            'std': round(df[col].std(), 2),
            'min': round(df[col].min(), 2),
            'max': round(df[col].max(), 2),
            'skewness': round(stats.skew(df[col]), 2),
            'kurtosis': round(stats.kurtosis(df[col]), 2)
        }
    
    # Correlation insights
    corr_matrix = numeric_df.corr()
    high_correlations = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.3:  # Strong correlation threshold
                high_correlations.append({
                    'var1': corr_matrix.columns[i],
                    'var2': corr_matrix.columns[j],
                    'correlation': round(corr_val, 3)
                })
    
    # Categorical insights
    categorical_summary = {}
    for col in categorical_df.columns:
        categorical_summary[col] = {
            'unique_count': df[col].nunique(),
            'top_categories': df[col].value_counts().head(3).to_dict()
        }
    
    return render_template('index.html', 
                         info=info, 
                         stats_summary=stats_summary,
                         high_correlations=high_correlations,
                         categorical_summary=categorical_summary)

if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)
