import numpy as np
import os
from MeshAnalyzer import MeshAnalyzer


class ShapeSimilaritySearch:
    def __init__(self, data_directory):
        self.data_directory = data_directory
        self.load_models()

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
                        mesh_analyzer = MeshAnalyzer(model_path)
                        inertial_moments, average_distances, variances = mesh_analyzer.analyze_along_principal_axis()

                        # Append model information to the list
                        self.models.append({'mesh': mesh_analyzer.mesh, 'thumbnail_path': thumbnail_path,
                                            'inertial_moments': inertial_moments,
                                            'average_distances': average_distances,
                                            'variances': variances})

    def calculate_feature_vector(self, model):
        # Implement your feature vector calculation using the analyzed statistics
        # This could include moments of inertia, average distance, variance, etc.
        feature_vector = np.concatenate([model['inertial_moments'], model['average_distances'], model['variances']])
        return feature_vector

    def compute_dissimilarity(self, vector1, vector2):
        # Implement your dissimilarity computation here
        # This could include Euclidean distance, elastic matching, etc.
        return np.linalg.norm(vector1 - vector2)

    def shape_similarity_search(self, query_model, k=20):
        query_feature_vector = self.calculate_feature_vector(query_model)
        distances = []

        for model in self.models:
            model_feature_vector = self.calculate_feature_vector(model)
            dissimilarity = self.compute_dissimilarity(query_feature_vector, model_feature_vector)
            distances.append((model, dissimilarity))

        distances.sort(key=lambda x: x[1])
        return distances[:k]

# Example usage
if __name__ == "__main__":
    # Specify the paths to your 3D Models and Thumbnails directories
    data_directory = "C:\\Users\\Ayoub\\Desktop\\CBIR\CBIR\\CBIR Data"

    shape_search = ShapeSimilaritySearch(data_directory)
    query_model_index = 0  # Index of the model to use as a query
    query_model = shape_search.models[query_model_index]

    # Perform shape similarity search
    similar_models = shape_search.shape_similarity_search(query_model)

    # Display results
    print("Similar Models:")
    for i, (model, dissimilarity) in enumerate(similar_models):
        print(f"{i + 1}. Model: {model['mesh'].metadata['file_name']}, Dissimilarity: {dissimilarity}")