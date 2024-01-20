import trimesh
import numpy as np

class MeshAnalysis:
    def __init__(self, file_path):
        self.mean_point = None
        self.mesh = trimesh.load(file_path)
        self.vertices = self.mesh.vertices
        self.faces = self.mesh.faces

    def generate_point_on_triangle(self, a, b, c):
        r1 = np.random.rand()
        r2 = np.random.rand()
        return (1 - np.sqrt(r1)) * a + np.sqrt(r1) * (1 - r2) * b + np.sqrt(r1) * r2 * c

    def generate_mass_points(self):
        mass_points = []
        for face in self.faces:
            triangle = self.vertices[face]
            area = trimesh.triangles.area([triangle])
            num_points = int(area * 50)
            for _ in range(num_points):
                point = self.generate_point_on_triangle(*triangle)
                mass_points.append(point)
        return np.array(mass_points)[:self.vertices.shape[0]]

    def calculate_covariance_matrix(self, mass_points):
        mean_point = np.mean(mass_points, axis=0)
        centered_points = mass_points - mean_point
        covariance_matrix = np.cov(centered_points.T)
        return covariance_matrix

    def extract_principal_axes(self, covariance_matrix):
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        principal_axes = eigenvectors[:, sorted_indices[:3]]
        return principal_axes

    def calculate_statistics_along_axis(self, axis_index, num_slabs=10):
        inertial_moments = np.zeros(num_slabs - 1)
        average_distances = np.zeros(num_slabs - 1)
        variances = np.zeros(num_slabs - 1)

        for i in range(1, num_slabs):
            slab_start = np.percentile(self.vertices[:, axis_index], (i - 1) * 100 / num_slabs)
            slab_end = np.percentile(self.vertices[:, axis_index], i * 100 / num_slabs)
            points_in_window = self.centered_points[(self.vertices[:, axis_index] >= slab_start) &
                                                    (self.vertices[:, axis_index] < slab_end)]

            inertial_moments[i - 1] = np.sum(np.linalg.norm(points_in_window, axis=1) ** 2)
            average_distances[i - 1] = np.mean(np.linalg.norm(points_in_window, axis=1))
            variances[i - 1] = np.var(np.linalg.norm(points_in_window, axis=1))

        return inertial_moments, average_distances, variances


    def analyze_mesh(self, num_slabs=10):
        mass_points = self.generate_mass_points()
        covariance_matrix = self.calculate_covariance_matrix(mass_points)
        self.centered_points = mass_points - np.mean(mass_points, axis=0)
        self.principal_axes = self.extract_principal_axes(covariance_matrix)

        # Calculate and print the principal axes of inertia
        print("Principal Axes of Inertia:\n", self.principal_axes)

        # Calculate statistics along the first principal axis
        inertial_moments, average_distances, variances = self.calculate_statistics_along_axis(0, num_slabs)

        # Print the calculated statistics
        print("Statistics along Principal Axis 0:\n")
        print("Inertial Moments:", inertial_moments)
        print("Average Distances:", average_distances)
        print("Variances:", variances)
        print("\n")

        # Calculate the average and variance of distances of faces to the first principal axis
        face_distances = self.calculate_face_distances_to_axis(0)
        average_face_distance = np.mean(face_distances)
        face_distance_variance = np.var(face_distances)

        print("Average Distance of Faces to Principal Axis 0:", average_face_distance)
        print("Variance of Face Distances to Principal Axis 0:", face_distance_variance)

        return inertial_moments, average_distances, variances

    def calculate_face_distances_to_axis(self, axis_index):
        distances = []
        for face in self.faces:
            triangle = self.vertices[face]
            center = np.mean(triangle, axis=0)
            projected_center = np.dot(center - self.mean_point, self.principal_axes[:, axis_index])
            distances.append(np.linalg.norm(projected_center))
        return np.array(distances)

# Example Usage:
if __name__ == "__main__":
    file_path = "CBIR Data/3D Models/m533.obj"
    mesh_analysis = MeshAnalysis(file_path)
    mesh_analysis.analyze_mesh(num_slabs=10)
