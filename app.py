from flask import Flask, render_template, send_from_directory, abort
import os
from ShapeSimilaritySearch import ShapeSimilaritySearch

app = Flask(__name__)


# Specify the paths to your 3D Models and Thumbnails directories
data_directory = "CBIR Data"

# Create an instance of ShapeSimilaritySearch
shape_search = ShapeSimilaritySearch(data_directory)


def find_model_by_name(models, model_name):
    return next((model for model in models if model['mesh'].metadata['file_name'] == model_name), None)


@app.route('/')
def index():
    return render_template('index.html', models=shape_search.models)


@app.route('/model/<model_name>')
def show_model(model_name):
    # Assuming shape_search is an instance of your ShapeSearch class
    selected_model = find_model_by_name(shape_search.models, model_name)

    if selected_model:
        # Assuming shape_search is an instance of your ShapeSearch class
        similar_models = shape_search.shape_similarity_search(selected_model['mesh'])
        return render_template('show_model.html', selected_model=selected_model, similar_models=similar_models)
    else:
        abort(404, "Model not found")



@app.route('/thumbnails/<filename>')
def thumbnail(filename):
    # Change the file extension to .jpg
    base_name, _ = os.path.splitext(filename)
    thumbnail_path = os.path.join(data_directory, "Thumbnails", f"{base_name}.jpg")

    # Check if the file exists
    if os.path.exists(thumbnail_path):
        return send_from_directory(os.path.join(data_directory, "Thumbnails"), f"{base_name}.jpg")
    else:
        # Return a placeholder image or an error message
        return "Thumbnail not found", 404



if __name__ == '__main__':
    app.run(debug=True)
