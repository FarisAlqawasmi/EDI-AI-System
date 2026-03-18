import pandas as pd

XLSX_PATH = "data/raw/Trimmed Resources.xlsx"

def show_sheets():
    xl = pd.ExcelFile(XLSX_PATH)
    print("Sheets:", xl.sheet_names)

def load_sheet(sheet_name: str):
    df = pd.read_excel(XLSX_PATH, sheet_name=sheet_name)
    print("Rows:", len(df))
    print("Columns:", list(df.columns))
    print(df.head(3))
    return df

if __name__ == "__main__":
    show_sheets()
    load_sheet("Resource centre taxonomy and re")