import trimesh
import numpy as np


class MeshAnalyzer:
    def __init__(self, file_path):
        # Load the mesh from the specified file path
        self.mesh = trimesh.load(file_path)
        # Extract vertices and faces from the loaded mesh
        self.vertices, self.faces = self.mesh.vertices, self.mesh.faces
        # Generate mass points for analysis
        self.mass_points = self.generate_mass_points()

    def generate_point_on_triangle(self, a, b, c):
        # Generate a random point on the triangle defined by vertices a, b, and c
        r1, r2 = np.random.rand(), np.random.rand()
        return (1 - np.sqrt(r1)) * a + np.sqrt(r1) * (1 - r2) * b + np.sqrt(r1) * r2 * c

    def generate_mass_points(self, num_points_per_area=50):
        # Generate mass points distributed on the mesh's surface
        mass_points = []
        for face in self.faces:
            triangle = self.vertices[face]
            # Calculate the area of the triangle
            area = trimesh.triangles.area([triangle])[0]
            # Calculate the number of points based on the triangle area
            num_points = int(area * num_points_per_area)
            # Generate points on the triangle and add them to the mass points list
            for _ in range(num_points):
                point = self.generate_point_on_triangle(*triangle)
                mass_points.append(point)
        # Convert the list of points to a NumPy array and limit the number of points to the number of vertices
        return np.array(mass_points)[:self.vertices.shape[0]]

    def calculate_covariance_matrix(self):
        # Calculate the covariance matrix of the generated mass points
        mean_point = np.mean(self.mass_points, axis=0)
        centered_points = self.mass_points - mean_point
        covariance_matrix = np.cov(centered_points.T)
        return covariance_matrix

    def extract_principal_axes(self):
        # Extract the principal axes from the calculated covariance matrix
        covariance_matrix = self.calculate_covariance_matrix()
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        # Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        principal_axes = eigenvectors[:, sorted_indices[:3]]
        return principal_axes

    def analyze_along_principal_axis(self, num_slabs=50):
        # Analyze statistics along the first principal axis divided into slabs
        inertial_moments, average_distances, variances = [], [], []

        for i in range(1, num_slabs + 1):
            # Determine the slab boundaries based on the x-coordinate of vertices
            slab_start = np.percentile(self.vertices[:, 0], (i - 1) * 100 / num_slabs)
            slab_end = np.percentile(self.vertices[:, 0], i * 100 / num_slabs)

            # Filter mass points within the current slab
            points_in_window = self.mass_points[(self.vertices[:, 0] >= slab_start) & (self.vertices[:, 0] < slab_end)]

            # Calculate inertial moments, average distances, and variances for the current slab
            inertial_moments.append(np.sum(np.linalg.norm(points_in_window, axis=1) ** 2))
            average_distances.append(np.mean(np.linalg.norm(points_in_window, axis=1)))
            variances.append(np.var(np.linalg.norm(points_in_window, axis=1)))

        return inertial_moments, average_distances, variances

# example of usage
if __name__ == "__main__":
    # Instantiate MeshAnalyzer with the mesh file path
    mesh_analyzer = MeshAnalyzer("CBIR Data/3D Models/3DMillenium_bottle01.obj")

    # Perform analysis
    inertial_moments, average_distances, variances = mesh_analyzer.analyze_along_principal_axis()

    # Display the computed statistics
    print("Inertial Moments:", inertial_moments)
    print("Average Distances:", average_distances)
    print("Variances:", len(variances))
