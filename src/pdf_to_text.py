import fitz
import os

input_folder = "../data/raw_papers"
output_folder = "../data/processed_text"

os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(input_folder):
    if file.endswith(".pdf"):
        try:
            path = os.path.join(input_folder, file)
            doc = fitz.open(path)

            text = ""
            for page in doc:
                text += page.get_text()

            output_path = os.path.join(
                output_folder,
                file.replace(".pdf", ".txt")
            )

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)

            print(f"Processed: {file}")

        except Exception as e:
            print(f"Error processing {file}: {e}")
