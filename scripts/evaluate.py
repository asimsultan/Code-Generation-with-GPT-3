
import torch
import argparse
import openai
import pandas as pd
from utils import get_device, preprocess_data, CodeGenerationDataset

def main(model_path, data_path, api_key):
    # Load Model (in this case, the API key)
    with open(model_path, 'r') as f:
        api_key = f.read().strip()

    # Initialize OpenAI API
    openai.api_key = api_key

    # Load Dataset
    dataset = pd.read_csv(data_path)
    preprocessed_data = preprocess_data(dataset)

    # DataLoader
    eval_dataset = CodeGenerationDataset(preprocessed_data)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, shuffle=False)

    # Evaluation Function
    def evaluate(model, data_loader, device):
        model.eval()
        total_loss = 0
        total_samples = 0

        for batch in data_loader:
            inputs, targets = batch
            inputs, targets = inputs.tolist(), targets.tolist()

            # Generate code using OpenAI API
            responses = [openai.Completion.create(
                engine=model,
                prompt=input_text,
                max_tokens=150
            ) for input_text in inputs]

            # Calculate loss (dummy loss for demonstration)
            loss = sum([1 for response, target in zip(responses, targets) if response['choices'][0]['text'] != target]) / len(targets)
            total_loss += loss
            total_samples += len(targets)

        avg_loss = total_loss / total_samples
        return avg_loss

    # Evaluate
    avg_loss = evaluate('davinci-codex', eval_loader, get_device())
    print(f'Average Loss: {avg_loss}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the file containing the API key')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV file containing validation data')
    parser.add_argument('--api_key', type=str, required=True, help='OpenAI API key')
    args = parser.parse_args()
    main(args.model_path, args.data_path, args.api_key)
