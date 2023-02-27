from utils import evaluate_TSNE_on_df, collect_pdf_folder_data, get_batched_embeddings

openai_api_key = "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"

def test_evaluate_TSNE_on_df(df):
    df = evaluate_TSNE_on_df(df)

pdf_folder = 'asada'
save_name_suffix = "[with-embeddings]"
df_init, file_prefix = collect_pdf_folder_data(pdf_folder,128,0.4)
df_init = get_batched_embeddings(df_init,pdf_folder,file_prefix,save_name_suffix,openai_api_key)
df = test_evaluate_TSNE_on_df(df_init)
