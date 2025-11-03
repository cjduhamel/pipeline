from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch
from utils.log import logger

def example_wine_preprocessor(preprocessor_params, X, y, metadata):
    """
    Example preprocessor that standardizes the features and splits the data into training and testing sets.

    Args:
        preprocessor_params (dict): Parameters for preprocessing (e.g., test_size, random_state).
        X (array-like): Feature data.
        y (array-like): Target labels.
        metadata (dict): Metadata dictionary to be updated with relevant preprocessing info.
    Returns:
        data (dict): Dictionary containing training and testing datasets.
        metadata (dict): Updated metadata with preprocessing info.
    """

    #log all params used (Example of logging a full dictioanry)
    logger.log_dict(preprocessor_params)

    test_size = preprocessor_params.get("test_split", 0.2)
    
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    metadata.update({
        "test_size": test_size,
        "num_train_samples": X_train.shape[0],
        "num_test_samples": X_test.shape[0]
    })
    # Standardization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    #get batch size from params
    batch_size = preprocessor_params.get("batch_size", 32)


    # Convert to PyTorch tensors and create DataLoaders
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    data = {
        "train_loader": train_loader,
        "test_loader": test_loader
    }
    return data, metadata
