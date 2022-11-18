from flask import Flask, render_template, request
import pickle
import cv2
from skimage import feature
import os.path

app = Flask(__name__)

@app.route('/')
def home():
   return render_template('home.html')

@app.route("/how_to_use")
def how_to_use():
    return render_template("how_to_use.html")

@app.route("/parkinson")
def parkinson():
    return render_template("parkinsin's.html")

@app.route("/detect")
def detect():
    return render_template("detect.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route('/detect', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file'] 

        # requesting the file
        # filename_secure = secure_filename(f.filename)
        basepath = os.path.dirname(
            '__file__') 
        
        # storing the file directory
        # storing the file in uploads folder
        filepath = os.path.join(basepath, "uploads", f.filename)
        f.save(filepath)  
        
        # saving the file
        # Loading the saved model
        print("[INFO] loading model...")
        model = pickle.loads(open('parkinson.pkl', "rb").read())
        '''local_filename = "./uploads/"
        local_filename += filename_secure
        print(local_filename)'''

        # Pre-process the image in the same manner we did earlier
        image = cv2.imread(filepath)
        output = image.copy()

        # Load the input image, convert it to grayscale, and resize
        output = cv2.resize(output, (128, 128))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (200, 200))
        image = cv2.threshold(image, 0, 255,
                              cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        # Quantify the image and make predictions based on the extracted features using the last trained Random Forest
        features = feature.hog(image, orientations=9,
                               pixels_per_cell=(10, 10), cells_per_block=(2, 2),
                               transform_sqrt=True, block_norm="L1")
        preds = model.predict([features])
        print(preds)
        ls = ["healthy", "parkinson"]
        result = ls[preds[0]]
        '''color = (0, 255, 0) if result == "healthy" else (0, 0, 255)
        cv2.putText(output, result, (3, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.imshow("Output", output)
        cv2.waitKey(0)'''
        return result
    return None

if __name__ == "__main__":
   app.run(host='0.0.0.0', port=8000, debug=False)