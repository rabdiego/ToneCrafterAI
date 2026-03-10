import os
from dotenv import load_dotenv
from llama_parse import LlamaParse

class PedalManualParser:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("LLAMA_CLOUD_API_KEY")
        
        if not self.api_key:
            raise ValueError("Error: LLAMA_CLOUD_API_KEY not found on ENV.")

        self.parser = LlamaParse(
            api_key=self.api_key,
            result_type="markdown",
            premium_mode=True,
            verbose=True
        )


    def parse_and_save(
        self,
        input_pdf_path: str,
        output_md_path: str
    ) -> str:
        print(f"Starting document extraction: {input_pdf_path}...")
        try:
            parsed_documents = self.parser.load_data(input_pdf_path)
            
            if not parsed_documents:
                return ""

            full_markdown_content = "\n\n".join([doc.text for doc in parsed_documents])

            with open(output_md_path, "w", encoding="utf-8") as file:
                file.write(full_markdown_content)
                
            return full_markdown_content

        except Exception as e:
            return ""


def main():
    manual_parser = PedalManualParser()

    pdf_path = "raw_docs/valeton_gp_100.pdf"
    md_path = "processed_docs/valeton_gp_100.md"

    extracted_content = manual_parser.parse_and_save(pdf_path, md_path)


if __name__ == "__main__":
    main()

