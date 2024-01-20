import numpy as np
import os
import pickle
from MeshAnalyzer import ModelAnalyzer

class ShapeSimilaritySearch:
    def __init__(self, data_directory, models_file="models.pkl"):
        self.data_directory = data_directory
        self.models_file = models_file
        self.models = []

        # Load models from file if available, otherwise load them and save
        if os.path.exists(models_file):
            self.load_models_from_file()
        else:
            self.load_models()
            self.save_models_to_file()

    def load_models(self):
        self.models = []
        # Iterate over folders in the models directory
        for folder_name in os.listdir(self.data_directory):
            folder_path = os.path.join(self.data_directory, folder_name)

            # Check if the item is a directory
            if os.path.isdir(folder_path):
                # Iterate over files in the current folder
                for file_name in os.listdir(folder_path):
                    if file_name.endswith(".obj"):
                        model_path = os.path.join(folder_path, file_name)
                        thumbnail_path = os.path.join(self.data_directory, folder_name,
                                                      f"{os.path.splitext(file_name)[0]}.png")

                        # Use MeshAnalyzer to load and analyze 3D models
                        mesh_analyzer = ModelAnalyzer(model_path)
                        inertial_moments = mesh_analyzer.moments_along_first_axis()
                        average_distances = mesh_analyzer.mean_distance_to_first_axis()
                        variances = mesh_analyzer.variance_of_distance_to_first_axis()

                        # Append model information to the list
                        self.models.append({'mesh': mesh_analyzer.model, 'thumbnail_path': thumbnail_path,
                                            'inertial_moments': inertial_moments,
                                            'average_distances': average_distances,
                                            'variances': variances})

    def load_models_from_file(self):
        with open(self.models_file, "rb") as file:
            self.models = pickle.load(file)

    def save_models_to_file(self):
        with open(self.models_file, "wb") as file:
            pickle.dump(self.models, file)

    def calculate_feature_vector(self, target_mesh):
        # Use vectorized operations for improved performance
        inertial_moments = target_mesh['inertial_moments']
        average_distances = target_mesh['average_distances']
        variances = target_mesh['variances']
        return np.array([inertial_moments, average_distances, variances])

    def compute_dissimilarity(self, vector1, vector2):
        # Use NumPy for vectorized operations
        return np.linalg.norm(vector1 - vector2)

    def shape_similarity_search(self, query_model, k=20):
        query_feature_vector = self.calculate_feature_vector(query_model)
        distances = []

        for target_mesh in self.models:
            model_feature_vector = self.calculate_feature_vector(target_mesh)
            dissimilarity = self.compute_dissimilarity(query_feature_vector, model_feature_vector)

            # Format dissimilarity to a string with five decimal places
            formatted_dissimilarity = "{:.5f}".format(dissimilarity)
            distances.append((target_mesh, formatted_dissimilarity))

        distances.sort(key=lambda x: x[1])
        return distances[:k]

# Example Usage:
if __name__ == "__main__":
    # Specify the paths to your 3D Models and Thumbnails directories
    data_directory = "CBIR Data"

    shape_search = ShapeSimilaritySearch(data_directory)
    query_model_index = 0  # Index of the model to use as a query
    query_model = shape_search.models[query_model_index]

    # Perform shape similarity search
    similar_models = shape_search.shape_similarity_search(query_model)

    # Display results
    print("Similar Models:")
    for i, (model, dissimilarity) in enumerate(similar_models):
        print(f"{i + 1}. Model: {model['mesh'].metadata['file_name']}, Dissimilarity: {dissimilarity}")