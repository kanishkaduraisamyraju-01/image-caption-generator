from flask import Flask, render_template, request, jsonify
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
from nltk.translate.bleu_score import sentence_bleu
app = Flask(__name__)

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = 16
num_beams = 10
num_captions = 3 # You can change this number to generate more captions per image
gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "num_return_sequences": num_captions}

def predict_step(image_paths, num_captions=3):  # Updated to accept num_captions
    images = []
    for image_path in image_paths:
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")
        images.append(i_image)

    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    gen_kwargs = {
        "max_length": max_length,
        "num_beams": num_beams,
        "num_return_sequences": num_captions  # Use the provided num_captions
    }

    output_ids = model.generate(pixel_values, **gen_kwargs)

    # Split the generated captions into a list
    all_preds = []
    for seq_ids in output_ids:
        preds = tokenizer.decode(seq_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds.split('\n')]
        all_preds.extend(preds)

    return all_preds


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files:
            return jsonify(error="No image file found"), 400

        image = request.files["image"]
        num_captions = int(request.form.get("num_captions", 3))  # Default to 3 if not provided
        try:
            image_path = "uploaded_image.jpg"  # Save the uploaded image to a folder named "static"
            image.save(image_path)
            predictions = predict_step([image_path], num_captions)

            # Replace with your actual reference caption
            reference = "This is the reference caption."  # Replace with your reference caption
            bleu_scores = [sentence_bleu([reference.split()], prediction.split()) for prediction in predictions]

            return jsonify(predictions=predictions, bleu_scores=bleu_scores), 200
        except Exception as e:
            return jsonify(error=str(e)), 500

    return render_template("index.html")
@app.route('/about')
def about(): 
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/login') 
def login(): 
    return render_template('login.html')

@app.route('/icg') 
def icg(): 
    return render_template('icg.html') 


if __name__ == "__main__":
    app.run(debug=True)
