from sklearn.metrics import classification_report, plot_confusion_matrix, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model, validation_data):
    # Evaluate the model using the validation data
    model_evaluation = model.evaluate(validation_data)
    
    # Make predictions on the validation data
    predictions = model.predict(validation_data)
    
    # Convert predicted probabilities to binary indicators (0 or 1)
    predictions = np.where(predictions > 0.5, 1, 0)
    
    # Compute the confusion matrix on the validation data
    cm = confusion_matrix(validation_data.labels, predictions)
    
    # Create a heatmap of the confusion matrix with annotations
    heatmap = sns.heatmap(cm, annot=True, annot_kws={"size": 16})
    
    # Save the heatmap as a PDF plot
    figure = heatmap.get_figure()    
    figure.savefig('../output/plots/cm_plot.pdf', dpi=400)
    plt.close()
    
    # Print model evaluation metrics
    print(dict(zip(model.metrics_names, model_evaluation)))
    
    return None
