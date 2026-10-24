
import os
from PIL import Image
from dotenv import load_dotenv
from google import genai
from flask import Flask, request, render_template

# Load environment variables
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError(
        "GOOGLE_API_KEY is missing. Check your .env file."
    )

# Initialize Gemini client
client = genai.Client(api_key=api_key)

app = Flask(__name__)

MODEL_NAME = "gemini-3.8-flash"


# Check whether the uploaded image is medical
def validate(image):
    prompt = """
    Determine whether this image is related to medicine or healthcare.
    Examples include medicine packaging, tablets, capsules,
    medical scans, medical equipment, and medical reports.

    Reply with only Yes or No.
    """

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[prompt, image]
    )

    answer = (response.text or "").strip().lower()
    return answer.startswith("yes")


# Generate a description of the medical image
def gen_image(prompt, image):
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[prompt, image]
    )

    return response.text or "No description was generated."


# Main page
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        uploaded_file = request.files.get("file")

        if not uploaded_file or not uploaded_file.filename:
            return render_template(
                "index.html",
                response_text="Please upload an image first."
            )

        try:
            image = Image.open(uploaded_file)
            image.load()
            image = image.convert("RGB")

            
                

            image_prompt = """
            Analyze this medical or healthcare-related image.

            If it shows medicine packaging, describe the visible
            brand name, medicine name, strength, and active
            ingredient only when clearly readable.

            If it shows a medical scan or report, describe
            observable information cautiously and explain
            relevant limitations.

            Do not invent unreadable details or claim a definitive
            diagnosis from an image alone. Do not prescribe
            medicines or dosages. Explain when professional
            medical advice is needed.

            Clearly state any uncertainty in your analysis.
            """

            response_text = gen_image(image_prompt, image)

            return render_template(
                "index.html",
                response_text=response_text
            )

        except Exception as e:
            app.logger.exception("Image processing failed")
            return render_template(
                "index.html",
                response_text=(
                    "Unable to process the image. Please try again. "
                    f"Error: {str(e)}"
                )
            )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
