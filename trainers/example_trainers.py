import torch
from utils.log import logger
from utils.log import model_tracker
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

class ExampleTrainer:
    """
    Must contain a method .run() that executes the full training pipeline.
    """
    def __init__(self, trainer_params, model, data, metadata):
        self.trainer_params = trainer_params
        self.model = model
        self.data = data
        self.metadata = metadata

        # set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def run(self):
        trained_model = self.train()
        self.evaluate()

        # use the model_tracker to get plots of metrics
        model_tracker.plot_metrics(["train_accuracy", "train_loss"])

        return trained_model


    def train(self):
        """
        Example trainer that trains a PyTorch model using the provided data.

        Args:
            trainer_params (dict): Parameters for training (e.g., epochs, learning_rate).
            model (nn.Module): The PyTorch model to be trained.
            data (dict): Dictionary containing training and testing datasets (DataLoaders).
            metadata (dict): Metadata dictionary with relevant dataset/model info.
        Returns:
            model (nn.Module): The trained PyTorch model.
        """

        # log all params used (Example of logging a full dictioanry)
        logger.log_dict(self.trainer_params)

        #get epochs and learning rate from params
        epochs = self.trainer_params.get("num_epochs", 10)
        learning_rate = self.trainer_params.get("learning_rate", 0.001)

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        #Send data to device
        train_loader = self.data["train_loader"]

        # Training loop
        self.model.train()

        # defined outside loop for logging after training
        avg_accuracy = None
        avg_loss = None
        
        #train over epochs
        for epoch in tqdm(range(epochs), desc="Training Epochs", unit="epoch"):
            losses = []
            accuracies = []
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                acuracy = (outputs.argmax(dim=1) == labels).float().mean().item()
                accuracies.append(acuracy)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            
            avg_accuracy = sum(accuracies) / len(accuracies)
            model_tracker.track_metric('train_accuracy', avg_accuracy)
            
            avg_loss = sum(losses) / len(losses)
            model_tracker.track_metric('train_loss', avg_loss)

        if avg_accuracy is not None and avg_loss is not None:
            logger.log('final_train_accuracy', avg_accuracy)
            logger.log('final_train_loss', avg_loss)

        return self.model

        
    
    def evaluate(self):
        test_loader = self.data["test_loader"]
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        logger.log('test_accuracy', accuracy)
        print(f'Test Accuracy: {accuracy:.2f}%')

        return self.model