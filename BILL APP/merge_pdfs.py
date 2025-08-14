import os
from PyPDF2 import PdfMerger

def merge_all_pdfs_from_all_subfolders(folder_path, output_filename="merged_all.pdf"):
    merger = PdfMerger()
    pdf_list = []

    # Walk through all folders
    for root, dirs, files in os.walk(folder_path):
        files = sorted(files)  # Sort files alphabetically
        for file in files:
            if file.lower().endswith(".pdf"):
                full_path = os.path.join(root, file)
                pdf_list.append(full_path)

    # Sort by folder path + filename (for consistent order)
    pdf_list.sort()

    for pdf in pdf_list:
        try:
            merger.append(pdf)
            print(f"Added: {pdf}")
        except Exception as e:
            print(f"Skipped: {pdf} — Error: {e}")

    output_path = os.path.join(folder_path, output_filename)
    merger.write(output_path)
    merger.close()
    print(f"\n✅ All PDFs merged into: {output_path}")

# 👇 Replace this with your target folder path
main_folder = r"\\195.185.175.186\g\OLD\ACCOUNT DEPATMENT F.Y.24-25\AUDIT PURPOSE 24-25\MANGLAM\MANGLAM\TDS CHALLAN"
merge_all_pdfs_from_all_subfolders(main_folder)
