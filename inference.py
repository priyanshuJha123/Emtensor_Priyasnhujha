import torch
import numpy as np

def run_inference(model, image):
    """Run inference on a single image."""
    model.eval()
    with torch.no_grad():
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        output = model(image_tensor)
    return output.squeeze(0).cpu().numpy()

def run_inference_with_uncertainty(model, image, num_samples=10):
    """Run inference with uncertainty estimation using Monte Carlo Dropout."""
    model.train()  # Enable dropout during inference
    predictions = []
    with torch.no_grad():
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        for _ in range(num_samples):
            output = model(image_tensor)
            predictions.append(output.cpu().numpy())
    predictions = np.stack(predictions)
    mean_prediction = predictions.mean(axis=0)
    uncertainty = predictions.std(axis=0)
    return mean_prediction.squeeze(0), uncertainty.squeeze(0)
