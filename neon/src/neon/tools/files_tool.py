# tools/file_tools.py
import pandas as pd
from crewai_tools import BaseTool
import json

class FileReadTool(BaseTool):
    name: str = "File Read Tool"
    # Update the description to reflect its new capability
    description: str = "Reads the content of a specified data file (supports CSV and XLSX)."

    def _run(self, file_path: str) -> str:
        """
        Reads a data file and returns its content as a string.
        Automatically handles .csv and .xlsx files based on extension.
        """
        try:
            # Check the file extension and use the appropriate pandas reader
            if file_path.lower().endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.lower().endswith('.xlsx'):
                df = pd.read_excel(file_path)
            else:
                return "Error: Unsupported file format. Please use a .csv or .xlsx file."

            # Convert the dataframe to a string for the LLM
            return df.to_string()
        except Exception as e:
            return f"Error reading file: {e}"

# --- The rest of your tools can remain the same ---
class TextFileWriteTool(BaseTool):
    name: str = "Text File Write Tool"
    description: str = "Writes content to a specified text file."

    def _run(self, file_path: str, content: str) -> str:
        """Writes the given content to a text file."""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"Successfully wrote content to {file_path}."
        except Exception as e:
            return f"Error writing to file: {e}."