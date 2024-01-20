import numpy as np
import trimesh
import pandas as pd


class ModelAnalyzer:
    def __init__(self, file_path):
        """
        Initialize the ModelAnalyzer with a 3D model from the given file path.
        """
        self.model = trimesh.load(file_path)

    def calculate_moments(self):
        """
        Calculate the moments of inertia for the 3D model.
        """
        moments = np.zeros((3, 3))

        for face in self.model.faces:
            vertices = self.model.vertices[face]
            vectors = np.roll(vertices, shift=-1, axis=0) - vertices

            moments += np.einsum('ij,ik->jk', vectors, vectors)

        eigenvalues, eigenvectors = np.linalg.eigh(moments)
        eigenvectors = eigenvectors[:, np.argsort(eigenvalues)]

        return eigenvectors[:, 2]

    def moments_along_first_axis(self):
        """
        Calculate the moments along the first principal axis.
        """
        principal_axis = self.calculate_moments()
        moments = np.dot(self.model.vertices, principal_axis)
        return np.mean(moments)

    def mean_distance_to_first_axis(self):
        """
        Calculate the mean distance of faces to the first principal axis.
        """
        principal_axis = self.calculate_moments()
        distances = np.sum(np.dot(self.model.vertices[self.model.faces], principal_axis) / 3, axis=1)
        return np.mean(distances)

    def variance_of_distance_to_first_axis(self):
        """
        Calculate the variance of distances of faces to the first principal axis.
        """
        principal_axis = self.calculate_moments()
        distances = np.sum(np.dot(self.model.vertices[self.model.faces], principal_axis) / 3, axis=1)
        return np.var(distances)

    def reduce_mesh(self, target_face_count):
        """Réduit le maillage d'un modèle 3D en supprimant les faces les moins importantes.

        Args:
          model: Le modèle 3D à réduire.
          target_face_count: Le nombre de faces cible pour le modèle réduit.

        Returns:
          Le modèle 3D réduit.
        """

        # Calcul des moments d'inertie du modèle.
        moments = self.moments_along_first_axis()

        # Tri des faces en fonction de leur distance au premier axe principal.
        faces = sorted(self.model.faces, key=lambda face: moments[face[0]])

        # Suppression des faces les moins importantes.
        while len(faces) > target_face_count:
            faces.pop()

        # Recréation du modèle 3D avec les faces restantes.
        reduced_model = Mesh(vertices=self.model.vertices, faces=faces)

        return reduced_model

    def describe_mesh(self):
        """Décrit un maillage 3D.

        Args:
          model: Le maillage 3D à décrire.

        Returns:
          Un dataframe contenant les descripteurs du maillage.
        """

        # Calcul des axes principaux.
        principal_axes = self.axes_principals()

        # Calcul des moments d'inertie du modèle.
        moments = self.moments_along_first_axis()

        # Calcul de la distance moyenne des faces au premier axe principal.
        mean_distance = self.mean_distance_to_first_axis()

        # Calcul de la variance de la distance des faces au premier axe principal.
        variance = self.variance_of_distance_to_first_axis()

        # Retourne un dataframe contenant les descripteurs du maillage.
        return pd.DataFrame({
            "principal_axes": principal_axes,
            "moments": moments,
            "mean_distance": mean_distance,
            "variance": variance
        })

    def search_mesh_database(self, database):
        """Recherche un maillage 3D dans une base de données.

        Args:
          model: Le maillage 3D à rechercher.
          database: La base de données de maillages 3D.

        Returns:
          Une liste de maillages 3D similaires au maillage recherché.
        """

        # Décrit le maillage recherché.
        description = self.describe_mesh()

        # Calcule les similarités entre le maillage recherché et les maillages de la base de données.
        similarities = database.dot(description.T)

        # Trie les maillages de la base de données en fonction de leur similarité au maillage recherché.
        sorted_indices = np.argsort(similarities)

        # Retourne une liste de maillages 3D similaires au maillage recherché.
        return database[sorted_indices]

    def compute_accuracy(self, database):
      """Calcule le taux de réussite du système de recherche.

      Args:
        model: Le maillage 3D à rechercher.
        database: La base de données de maillages 3D.

      Returns:
        Le taux de réussite du système de recherche.
      """

      # Recherche le maillage dans la base de données.
      similar_meshes = self.search_mesh_database(database)

      # Calcule le taux de réussite.
      accuracy = np.mean([mesh.name == self.model.name for mesh in similar_meshes])

      # Retourne le taux de réussite.
      return accuracy


# Example Usage:
if __name__ == "__main__":
    file_path = "CBIR Data/3D Models/m533.obj"
    model_analysis = ModelAnalyzer(file_path)
    inertial_moments = model_analysis.moments_along_first_axis()
    average_distances = model_analysis.mean_distance_to_first_axis()
    variances = model_analysis.variance_of_distance_to_first_axis()
    print(f"Statistics along the First Principal Axis:\n")
    print("Inertial Moments:", inertial_moments)
    print("Average Distances:", average_distances)
    print("Variances:", variances)
    print("\n")
