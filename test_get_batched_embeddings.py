from utils import get_batched_embeddings, collect_pdf_folder_data

def test_get_batched_embeddings(df_init,pdf_folder,file_prefix,save_name_suffix):
    df = get_batched_embeddings(df_init,pdf_folder,file_prefix,save_name_suffix)

pdf_folder = 'asada'
save_name_suffix = "[with-embeddings]"
df_init, file_prefix = collect_pdf_folder_data(pdf_folder,128,0.4)
df_init = test_get_batched_embeddings(df_init,pdf_folder,file_prefix,save_name_suffix)