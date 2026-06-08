#create pca for tshirt data in csv file
#consider target columns QaulityClass, Rest all inputs

import pandas as pd
from sklearn.decomposition import PCA   
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
from supervisedmlapp.configurations.conf import T_SHIRT_FILE_PATH
from sklearn.preprocessing import LabelEncoder
def pca_analysis():
    # Load the dataset
    data = pd.read_csv(T_SHIRT_FILE_PATH)

    #dataset information
    print("Dataset Information:")
    print(data.info())


    #apply label encoding to the target variable
   
    label_encoder = LabelEncoder()

    y = label_encoder.fit_transform(data['QualityClass'])

    #label encoding to defects, size and stitiching
    data['Defects'] = label_encoder.fit_transform(data['Defects'])
    data['Size'] = label_encoder.fit_transform(data['Size'])
    data['Stitching'] = label_encoder.fit_transform(data['Stitching'])

    #apply one hot encoding to color,fabric,brand
    data = pd.get_dummies(data, columns=['Color', 'Fabric', 'Brand'])   
  
   
    # Separate features and target variable
    X = data.drop('QualityClass', axis=1)  # Features
    
    #print x and y
    print("Features (X):")
    print(X.head())
    print("\nTarget variable (y):")
    print(y[:5])

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Apply PCA
    pca = PCA(n_components=2)  # Reduce to 2 principal components for visualization
    X_pca = pca.fit_transform(X_scaled) 
    # Create a DataFrame for the PCA results
    pca_df = pd.DataFrame(data=X_pca, columns=['Principal Component 1', 'Principal Component 2'])
    pca_df['QualityClass'] = y  # Add the target variable for coloring
    # Visualize the PCA results
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(pca_df['Principal Component 1'], pca_df['Principal Component 2'], c=pca_df['QualityClass'], cmap='viridis')    
    plt.title('PCA of T-Shirt Quality Data')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(scatter, label='QualityClass')
    plt.show()


if __name__ == "__main__":
    pca_analysis()