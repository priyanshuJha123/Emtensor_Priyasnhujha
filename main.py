from data_pipeline.preprocess import preprocess_image
from data_pipeline.augment import augment_image
from model.segmentation_model import get_segmentation_model
from inference.inference import run_inference_with_uncertainty
from analysis.statistical_analysis import perform_ttest, perform_anova
from analysis.visualization import plot_heatmap, plot_3d_slices

def main():
    # Example workflow
    image = preprocess_image("example.nii")
    augmented_image = augment_image(image)
    
    model = get_segmentation_model()
    mean_output, uncertainty = run_inference_with_uncertainty(model, augmented_image)
    
    # Perform statistical analysis (dummy data)
    group1, group2, group3 = [1, 2, 3], [4, 5, 6], [7, 8, 9]
    anova_result = perform_anova(group1, group2, group3)
    print("ANOVA result:", anova_result)
    
    # Visualize output
    plot_heatmap(mean_output[0], title="Segmentation Output")
    
    # Select the first channel of uncertainty for visualization
    uncertainty_channel = uncertainty[0]  # Select the first channel (or aggregate if needed)
    plot_3d_slices(uncertainty_channel, title="Uncertainty Visualization")

if __name__ == "__main__":
    main()
