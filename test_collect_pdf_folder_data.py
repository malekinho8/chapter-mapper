from utils import collect_pdf_folder_data

def test_collect_pdf_folder_data(pdf_folder,chunk_size,chunk_overlap):
    df = collect_pdf_folder_data(pdf_folder,chunk_size,chunk_overlap)

test_collect_pdf_folder_data('asada',128,0.4)