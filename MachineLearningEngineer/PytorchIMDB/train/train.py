import argparse
import json
import os
import pickle
import sys
import sagemaker_containers
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data
import numpy as np

from model import LSTMClassifier

def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(model_info['embedding_dim'], model_info['hidden_dim'], model_info['vocab_size'])

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    # Load the saved word_dict.
    word_dict_path = os.path.join(model_dir, 'word_dict.pkl')
    with open(word_dict_path, 'rb') as f:
        model.word_dict = pickle.load(f)

    model.to(device).eval()

    print("Done loading model.")
    return model

def _get_data_loader(batch_size, data_dir, data_filename):
    print("Get train data loader.")

    train_data = pd.read_csv(os.path.join(data_dir, data_filename), header=None, names=None)

    train_y = torch.from_numpy(train_data[[0]].values).float().squeeze()
    train_X = torch.from_numpy(train_data.drop([0], axis=1).values).long()

    train_ds = torch.utils.data.TensorDataset(train_X, train_y)

    return torch.utils.data.DataLoader(train_ds, batch_size=batch_size)


def train(model, train_loader, validation_loader, epochs, patience, optimizer, loss_fn, device):
    """
    This is the training method that is called by the PyTorch training script. The parameters
    passed are as follows:
    model        - The PyTorch model that we wish to train.
    train_loader - The PyTorch DataLoader that should be used during training.
    validation_loader - The PyTorch DataLoader that should be used during validation.
    epochs       - The total number of epochs to train for.
    optimizer    - The optimizer to use during training.
    loss_fn      - The loss function used for training.
    device       - Where the model and data should be loaded (gpu or cpu).
    """
    
    patience_counter = 0
    min_val_loss = np.Inf
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        train_correct = 0
        total_size = 0
        for batch in train_loader:         
            batch_X, batch_y = batch
            
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # Taken from my Digit Recognition Kaggle Notebook
            # https://www.kaggle.com/nicapotato/pytorch-resnet-kanada
            model.zero_grad()  # sets gradients to 0 before loss calc. You will do this likely every step.
            output = model(batch_X)  # Pass batch to model
            
            loss = loss_fn(output, batch_y)  # calc and grab the loss value
            loss.backward()  # apply this loss backwards thru the network's parameters
            optimizer.step()  # attempt to optimize weights to account for loss/gradients 
            
            train_correct += (output.round() == batch_y).sum().item()
            train_loss += loss.data.item()
        
        # Evaluation with the validation set
        model.eval() # eval mode
        val_loss = 0
        val_correct = 0

        with torch.no_grad():
            # First Validation Set
            for batch in validation_loader:
                batch_X, batch_y = batch

                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                val_output = model(batch_X) # get predictions
                vloss = loss_fn(val_output, batch_y) # calculate the loss

                val_correct += (val_output.round() == batch_y).sum().item()
                val_loss += vloss.item()
        
        final_val_loss = val_loss/len(validation_loader)
        print(
"Epoch: {} - Training BCELoss: {:.4f}, Accuracy: {:.1f}% - \
Validation BCELoss: {:.4f} - Accuracy: {:.1f}%"
            .format(epoch,
                    train_loss / len(train_loader),
                    ((train_correct / len(train_loader.dataset))*100),
                    final_val_loss,
                    (val_correct / len(validation_loader.dataset))*100)) 
        
        if min_val_loss > round(final_val_loss,4) :
            min_val_loss = round(final_val_loss,4)
            patience_counter = 0
#             torch.save(model, os.path.join(model_path, 'best_model.pth'))
#             best_model = model.state_dict()
        else:
            patience_counter += 1

        if patience_counter > patience:
            print(f"No Improvemend in {patience} rounds, Early Stopping..")
            break
    print("Load Best Model..")
#     model = model.load_state_dict(best_model)
#     model = torch.load(os.path.join(model_path, 'best_model.pth'))
    return model

if __name__ == '__main__':
    # All of the model parameters and training parameters are sent as arguments when the script
    # is executed. Here we set up an argument parser to easily access the parameters.

    parser = argparse.ArgumentParser()

    # Training Parameters
    parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    # Model Parameters
    parser.add_argument('--embedding_dim', type=int, default=32, metavar='N',
                        help='size of the word embeddings (default: 32)')
    parser.add_argument('--hidden_dim', type=int, default=100, metavar='N',
                        help='size of the hidden dimension (default: 100)')
    parser.add_argument('--vocab_size', type=int, default=5000, metavar='N',
                        help='size of the vocabulary (default: 5000)')

    # SageMaker Parameters
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}.".format(device))

    torch.manual_seed(args.seed)

    # Load the training data.
    train_loader = _get_data_loader(args.batch_size, args.data_dir, "train.csv")
    validation_loader = _get_data_loader(args.batch_size, args.data_dir, "validation.csv")

    # Build the model.
    model = LSTMClassifier(args.embedding_dim, args.hidden_dim, args.vocab_size).to(device)

    with open(os.path.join(args.data_dir, "word_dict.pkl"), "rb") as f:
        model.word_dict = pickle.load(f)

    print("Model loaded with embedding_dim {}, hidden_dim {}, vocab_size {}.".format(
        args.embedding_dim, args.hidden_dim, args.vocab_size
    ))

    # Train the model.
    optimizer = optim.Adam(model.parameters())
    loss_fn = torch.nn.BCELoss()

    model = train(model=model,
                  train_loader=train_loader,
                  validation_loader=validation_loader,
                  epochs=args.epochs,
                  patience=5,
                  optimizer=optimizer,
                  loss_fn=loss_fn,
                  device=device)

    # Save the parameters used to construct the model
    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'embedding_dim': args.embedding_dim,
            'hidden_dim': args.hidden_dim,
            'vocab_size': args.vocab_size,
        }
        torch.save(model_info, f)

	# Save the word_dict
    word_dict_path = os.path.join(args.model_dir, 'word_dict.pkl')
    with open(word_dict_path, 'wb') as f:
        pickle.dump(model.word_dict, f)

	# Save the model parameters
    model_path = os.path.join(args.model_dir, 'model.pth')
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)
