import trimesh
import numpy as np


def load_mesh(file_path):
    return trimesh.load(file_path)


def generate_point_on_triangle(a, b, c):
    r1, r2 = np.random.rand(), np.random.rand()
    return (1 - np.sqrt(r1)) * a + np.sqrt(r1) * (1 - r2) * b + np.sqrt(r1) * r2 * c


def generate_mass_points(vertices, faces, num_points_per_area=50):
    mass_points = []
    for face in faces:
        triangle = vertices[face]
        area = trimesh.triangles.area([triangle])[0]
        num_points = int(area * num_points_per_area)
        for _ in range(num_points):
            point = generate_point_on_triangle(*triangle)
            mass_points.append(point)
    return np.array(mass_points)[:vertices.shape[0]]


def calculate_covariance_matrix(mass_points):
    mean_point = np.mean(mass_points, axis=0)
    centered_points = mass_points - mean_point
    covariance_matrix = np.cov(centered_points.T)
    return covariance_matrix


def extract_principal_axes(covariance_matrix):
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    principal_axes = eigenvectors[:, sorted_indices[:3]]
    return principal_axes


def analyze_along_principal_axis(vertices, centered_points, num_slabs=50):
    inertial_moments, average_distances, variances = [], [], []

    for i in range(1, num_slabs):
        slab_start = np.percentile(vertices[:, 0], (i - 1) * 100 / num_slabs)
        slab_end = np.percentile(vertices[:, 0], i * 100 / num_slabs)

        points_in_window = centered_points[(vertices[:, 0] >= slab_start) & (vertices[:, 0] < slab_end)]

        inertial_moments.append(np.sum(np.linalg.norm(points_in_window, axis=1) ** 2))
        average_distances.append(np.mean(np.linalg.norm(points_in_window, axis=1)))
        variances.append(np.var(np.linalg.norm(points_in_window, axis=1)))

    return inertial_moments, average_distances, variances


if __name__ == "__main__":
    # Load the mesh
    mesh = load_mesh("./London E 432.obj")

    # Get vertices and faces
    vertices, faces = mesh.vertices, mesh.faces

    # Generate mass points
    mass_points = generate_mass_points(vertices, faces)

    # Calculate covariance matrix
    covariance_matrix = calculate_covariance_matrix(mass_points)

    # Extract principal axes of inertia
    principal_axes = extract_principal_axes(covariance_matrix)

    # Analyze along the first principal axis
    inertial_moments, average_distances, variances = analyze_along_principal_axis(vertices, mass_points)

    # Display the computed statistics
    print("Inertial Moments:", inertial_moments)
    print("Average Distances:", average_distances)
    print("Variances:", variances)
