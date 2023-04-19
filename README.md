# chapter-mapper

This is a simple package for mapping PDF files and finding answers to your questions easily and efficiently by utilizing language models from OpenAI. The framework seeks to address the problem of "hallucinations" in large language models by using a director vector similarity search based on text embeddings of the inputs PDFs and query to find relevant documentation, and then it provides that context to ChatGPT to formulate a coherent answer to the question.

<img src="https://cdn.discordapp.com/attachments/1092808829592424509/1098106157840744558/image_from_clipboard.png" width="600" alt="Overall data processing summary.">

## Getting Started

To get started using the code as quickly as possible, you may make a copy of this [Google Colab Notebook](https://drive.google.com/file/d/1PeYb9Kczs9nyGlgZIxhKjcsLVMGzhkvp/view?usp=sharing) to your own Google Drive. Continue to follow the [INSTRUCTIONS.md](https://github.com/malekinho8/chapter-mapper/blob/main/INSTRUCTIONS.md) for more details. 

## Usage

Once you have opened the Google Colab Notebook on your Drive, you may follow these steps:

1. Upload a folder to your drive which contains 1 or more PDFs that you want to query. 
1. Then for the `pdf_folder` form field in the Notebook, enter the name of the folder you uploaded. If possible, ensure that the folder name is unique so the correct path is found. 
1. For the `search_query` form field, enter a question you have about the PDFs in your folder.
1. Feel free to experiment with the other parameters, but the only other parameter you have to change is the `openai_api_key`, which you can obtain on the [OpenAI website](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwi8trW-srX-AhWXF1kFHTEjBIgQFnoECBUQAQ&url=https%3A%2F%2Fplatform.openai.com%2Faccount%2Fapi-keys&usg=AOvVaw0Uus1Ol-tJ8dIGLAPRllHE).
1. Finally, you may simply run the cells by clicking the arrow at the left and scrolling down, or by using the shortcut `SHIFT + ENTER`.
1. On the first run, it might take some time, but future runs will run faster once the `.csv` has been created and the embeddings have been stored. Furthermore, if you end up adding more PDFs to your `pdf_folder` in the future, the program will automatically update the `.csv` file.