import matplotlib.pyplot as plt

def plot_heatmap(image, title="Heatmap", output_file="heatmap.png"):
    """Plot a heatmap of the image and save it to a file."""
    if image.ndim == 3:
        # Select the middle slice along the first axis for visualization
        image = image[image.shape[0] // 2]
    plt.imshow(image, cmap='hot')
    plt.title(title)
    plt.colorbar()
    plt.savefig(output_file)  # Save the plot to a file
    plt.close()  # Close the figure to free memory

def plot_3d_slices(image, title="3D Slices", output_file="3d_slices.png"):
    """Visualize slices of a 3D neuroimaging volume and save them to a file."""
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D array, but got shape {image.shape}")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    slices = [
        image[image.shape[0] // 2, :, :],  # Middle slice along the first axis
        image[:, image.shape[1] // 2, :],  # Middle slice along the second axis
        image[:, :, image.shape[2] // 2],  # Middle slice along the third axis
    ]
    for i, slice_ in enumerate(slices):
        axes[i].imshow(slice_, cmap='gray')
        axes[i].set_title(f"Slice {i+1}")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_file)  # Save the plot to a file
    plt.close()  # Close the figure to free memory
