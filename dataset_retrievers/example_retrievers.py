from sklearn.datasets import load_wine
from utils.log import logger

def load_example_wine(dataset_params, metadata):
    """
    Example dataset retriever that loads the wine dataset from sklearn.
    Data can be in any format, as long as it is consistent with the rest of the pipeline.

    Args:
        dataset_params (dict): Parameters for dataset retrieval (not used in this example).
        metadata (dict): Metadata dictionary to be updated with relevant dataset info.
    Returns:
        X (array-like): Feature data.
        y (array-like): Target labels.
        metadata (dict): Updated metadata with dataset info.
    """
    data = load_wine()

    # update logger! (example of logging individual values)
    logger.log('dataset_loaded', 'example_wine')

    X = data.data
    y = data.target
    metadata.update({
        "feature_names": data.feature_names,
        "target_names": data.target_names,
        "description": data.DESCR,
        "input_size": X.shape[1],
        "num_classes": len(set(y))
    })
    return X, y, metadata