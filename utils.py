import fitz
import pandas as pd
import numpy as np
import re
import os
import openai
import colorsys
import tiktoken
import plotly.express as px
import plotly.graph_objects as go
import textwrap
import nbformat
from sklearn.manifold import TSNE
from matplotlib import colors as mcolors
from openai.embeddings_utils import get_embedding, cosine_similarity
from tqdm import tqdm
from typing import List, Dict, Tuple
from operator import itemgetter
from section_headers import *
from fpdf import FPDF
# from transformers import GPT2TokenizerFast

enc = tiktoken.get_encoding("gpt2")
# tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# specify some parameters for openAI models
MAX_SECTION_LEN = 1500
SEPARATOR = "\n* "
ENCODING = "cl100k_base"  # encoding for text-embedding-ada-002

encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))

f"Context separator contains {separator_len} tokens"

def find_folder_path(folder_name, root_path):
    for dirpath, dirnames, filenames in os.walk(root_path):
        for dirname in dirnames:
            if dirname == folder_name:
                return os.path.join(dirpath, dirname)
    return None

def get_hsv(color_name):
    css4_colors = mcolors.CSS4_COLORS
    hexrgb = css4_colors[color_name]
    hexrgb = hexrgb.lstrip("#")   # in case you have Web color specs
    r, g, b = (int(hexrgb[i:i+2], 16) / 255.0 for i in range(0,5,2))
    return colorsys.rgb_to_hsv(r, g, b)

def feature_matrix(df):
    print("Extracting Embedding Feature Matrix...")
    matrix = df['embedding-new'].to_list()
    matrix_empty = np.zeros((len(matrix), len(matrix[0])))
    for i in range(len(matrix)):
        try:
            matrix_empty[i, :] = np.array(matrix[i])
        except Exception as e:
            print(i, e)
            print(matrix[i])
            exit()
    matrix = matrix_empty
    return matrix

def toarray(x):
    if isinstance(x, str):
        x = [float(v.strip()) for v in x.strip('[').strip(']').split(',')]
    return x

def search_text(df, search_query, n, EMBEDDING_MODEL):
    embedding = get_embedding(
        search_query,
        engine=EMBEDDING_MODEL
    )
    df["similarity"] = df.embeddings.apply(lambda x: cosine_similarity(np.asarray(toarray(x), dtype='float64'), np.asarray(embedding, dtype='float64')))
    results = list(df.sort_values("similarity", ascending=False).head(n).description)
    return results

def filter_search(df, search_results):
    df['keep'] = [x in search_results for x in list(df.description)]
    df = df[df['keep'] == True].filter(list(df.columns)[:-1])
    print(df.shape)
    return df


def plotmap(df, search_query, number_results, dm, all_titles, pdf_folder, EMBEDDING_MODEL, folder_label=False):
    """
    Plots a scatter plot of the given dataframe with optional search highlighting.

    Args:
    - df (pandas.DataFrame): the dataframe to be plotted, must contain columns 'x', 'y', 'title', 'chapter', 'page', and 'description'.
    - search_query (str): the query to search for within the dataframe, empty string for no search highlighting.
    - number_results (int): the maximum number of search results to highlight, only used if search_query is not empty.
    - dm (dict): a dictionary mapping colors to title names.
    - all_titles (list): a list of all the unique title names in the dataframe.
    - pdf_folder (str): the name of the folder where the PDF files are located.

    Returns:
    - None.

    Side Effects:
    - Plots the scatter plot using Plotly Express and displays it in the console.

    """
    print("Searching for relevant text...")
    search_results = list(search_text(df, search_query, number_results,EMBEDDING_MODEL))
    df_filter = filter_search(df, search_results)
    df_filter['size'] = 20
    x_min = df['x'].min()
    x_max = df['x'].max()
    y_min = df['y'].min()
    y_max = df['y'].max()

    if not folder_label:
        fig = px.scatter(df, x='x', y='y', color='title', hover_data=['title', 'chapter', 'page', 'description'], color_discrete_map=dm, category_orders={'title': all_titles}, template="plotly_dark")
        fig.add_trace(px.scatter(df_filter, x='x', y='y', color='title', size='size', hover_data=['title', 'similarity', 'chapter', 'page', 'description'], color_discrete_map=dm, category_orders={'title': all_titles}).data[0])
    else:
        df['folder'] = df.title.apply(lambda x: x.split(os.sep)[-2]) 
        df_filter['folder'] = df_filter.title.apply(lambda x: x.split(os.sep)[-2])
        fig = px.scatter(df, x='x', y='y', color='folder', hover_data=['title', 'chapter', 'page', 'description'], color_discrete_map=dm, category_orders={'title': all_titles}, template="plotly_dark")
        fig.add_trace(px.scatter(df_filter, x='x', y='y', color='folder', size='size', hover_data=['title', 'similarity', 'chapter', 'page', 'description'], color_discrete_map=dm, category_orders={'title': all_titles}).data[0])
    
    fig.update_layout(hoverlabel=dict(font=dict(family='Arial', size=12, color='black'),align='left'),
                    title=f"{pdf_folder} Visualized",
                    xaxis=dict(range=[x_min, x_max]),  # Update the x-axis range
                    yaxis=dict(range=[y_min, y_max]))  # Update the y-axis range
    fig.show()
    return fig

def vector_similarity(x: List[float], y: List[float]) -> float:
    """
    Returns the similarity between two vectors.
    
    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))

def order_document_sections_by_query_similarity(query: str, df:pd.DataFrame, EMBEDDING_MODEL: str) -> List[Tuple[float, Tuple[str, str]]]:
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections. 
    
    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_embedding(query,engine=EMBEDDING_MODEL)
    # contexts = lambda df: (df.embeddings.apply(lambda val: eval(val) if isinstance(val, str) else val) if any(isinstance(val, str) for val in df.embeddings) else df.embeddings) 
    print('Converting string embeddings to a list of floats...')
    contexts = contexts = df.embeddings.apply(lambda x: [float(num) for num in x.strip('[]').split(',')] if isinstance(x, str) else x)
    
    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)
    
    return document_similarities

def construct_prompt(question: str, df: pd.DataFrame, EMBEDDING_MODEL: str) -> str:
    """
    Fetch relevant 
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, df, EMBEDDING_MODEL)
    
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []
    chosen_sections_titles = []
    chosen_sections_pages = []
     
    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.        
        document_section = df.loc[section_index]
        
        chosen_sections_len += document_section.tokens + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break
            
        chosen_sections.append(SEPARATOR + document_section.chunk_text.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))
        chosen_sections_titles.append(document_section.title)
        chosen_sections_pages.append(document_section.page)
            
    # Useful diagnostic information
    print(f"Selected {len(chosen_sections)} document sections:")
    for i in range(0,len(chosen_sections)):
      print(f'p. {chosen_sections_pages[i]+1}, {chosen_sections_titles[i]}')
    
    header = ""
    for i in range(0,len(chosen_sections)):
      header += f"""From p. {chosen_sections_pages[i]+1}, {chosen_sections_titles[i]}: {chosen_sections[i]}\n\n """
    # header += """Based on the context provided above, try to answer the question as honestly and truthfully as possible and provide a parenthetical reference saying which page number and filename you indexed to create your answer. If the answer to the question is not contained within the context provided, reply by saying "I'm sorry, but I don't know if I can answer your question but here's a summary of what I found:" and provide a TL;DR of the most relevant context above."""
    header += """First provide a citation containing the page numbers and filenames of the most relevant texts you used to answer the question from the context above. Then, try to answer the question as honestly and truthfully as possible and provide a parenthetical reference saying which page number and filename you indexed to create your answer. If the answer to the question is not contained within the context provided, reply by saying "I'm sorry, but I don't know if I can answer your question but here's a summary of what I found:" and provide a TL;DR of the most relevant context above. Note that in each text chunk, the notation ' [ref: file_name_here] ' indicates which file the text comes from."""
    # header += """
    #     Answer the question as honestly and truthfully as possible based on the context above.
    #     Provide your answer in the form of a table with a column containing the most relevant text excerpts from each of the sources above, and another column for your answer to the question based on that specific context.
    #     Then, if applicable, provide a 2-3 sentence summary of the results in the table.
    # """

    return header + "\n\n Q: " + question + "\n\n A:"

def answer_query_with_context(
    query: str,
    df: pd.DataFrame,
    COMPLETIONS_MODEL: str,
    EMBEDDING_MODEL: str,
    show_prompt: bool = False,
    temperature = 0,
    max_tokens = 1000
) -> str:
    prompt = construct_prompt(
        query,
        df,
        EMBEDDING_MODEL
    )

    COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0,
    "max_tokens": 1000,
    "model": COMPLETIONS_MODEL,
    }
    
    if show_prompt:
        print(prompt)

    response = openai.ChatCompletion.create(
                messages = [{"role": "user", "content": prompt}],
                **COMPLETIONS_API_PARAMS
            )

    return response['choices'][0]['message']['content']

def batch_embed(df,batch_size,column_name:str,openai_api_key):
  pbar = tqdm(total=len(df))
  df = df.reset_index(drop=True)
  df['embeddings'] = [[] for _ in range(len(df))]
  for i in range(int(np.ceil(len(df)/batch_size))):
      start = i * batch_size
      end = start + batch_size
      end = min(end, len(df))
      model_main = openai.Embedding.create(model="text-embedding-ada-002", input=list(df[column_name].iloc[start:end]))
      for j in range(end-start):
          embedding_main = model_main.data[j]['embedding']
          df.at[start + j, 'embeddings'] = embedding_main
          pbar.update()
  return df

def textbook_pdf2csv(pdf_file,chunk_size,overlap):
    """
    Obtain a csv containing title of work, chapter, page, and chunked content from the body of work. 
    
    chunk_size refers to the number of tokens to include in each chunk of text. If a given page has
        less than the number of tokens specified, it will include text from the entire page.

    overlap refers to how much to overlap the current chunk with the previous chunk. 
    """
    assert chunk_size <= 1024, 'The max number of tokens that the OpenAI model can handle at once is 1024.'
    chunks = get_text_chunks(pdf_file,chunk_size,overlap=0.5) # dictionary with {chapter: '', text_chunk: '', page: ''}
    df = chunks2df(chunks,pdf_file)

def get_text_chunks(pdf_file:str,chunk_size:int,overlap:float):
    doc = fitz.open(pdf_file)
    chapter_text = get_chapter_text(doc,num_pages) # technically you don't have to do this, but I think it would 
    temp_tokens = 0 # initialize the token counter

    for page in range(num_pages):
        while temp_tokens <= chunk_size:
            page_text = doc.get_page_text(page) # get the number of tokens in the page
            tokens = count_tokens(page_text)
            if tokens >= 100: # there should be roughly more than a paragraph on the page for us to consider it
                chunk_text += page_text

def chunks2df(chunks,pdf_file):
    df = pd.DataFrame(chunks)
    df = df.insert(0,'title',pdf_file)
    return df

def count_tokens(text: str) -> int:
    """count the number of tokens in a string"""
    return len(tokenizer.encode(text))

def create_textbook_csv_from_pdf(pdf_file):
    doc = fitz.open(pdf_file)
    font_counts, align_counts, styles = fonts(doc, granularity=True)
    size_tag, most_common_font, most_common_left_align = font_tags(font_counts, align_counts, styles)
    paragraphs = get_paragraphs(doc,size_tag,most_common_font, most_common_left_align)
    out = pd.DataFrame(paragraphs)
    out_name = pdf_file.replace('.pdf','') + ' (paragraph_raw_data)'
    out.to_csv(f'{out_name}.csv',encoding='utf-8-sig')
    return out


def get_paragraphs(doc, size_tag, most_common_font, most_common_left_align):
    """Scrapes paragraphs from PDF and return dictionary with text and corresponding section where it comes from.
    :param doc: PDF document to iterate through
    :type doc: <class 'fitz.fitz.Document'>
    :param size_tag: textual element tags for each size
    :type size_tag: dict
    :rtype: list
    :return: texts with pre-prended element tags
    """
    header_para = [] # list with headers and paragraphs
    first = True  # boolean operator for first header
    previous_s = {}  # previous span
    paragraphs = []
    section_heading = 'Other'
    chapter_font_size = 0
    chapter = 0

    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:  # iterate through the text blocks
            if b['type'] == 0:  # this block contains text
                # REMEMBER: multiple fonts and sizes are possible IN one block
                paragraph_i = ""  # text found in block
                for l in b["lines"]:  # iterate through the text lines
                    for s in l["spans"]:  # iterate through the text spans
                        if first:
                            first = False
                            s_previous = s
                        text = s['text'].strip()
                        if s['origin'][0] == most_common_left_align and s['size'] == most_common_font: # if it is text in a paragraph within the textbook
                            paragraph_i += s['text'] + ' '      
                        elif not np.isclose(s['origin'][0],most_common_left_align,rtol=1e-03) and s['size'] == most_common_font: # if it is a new paragraph
                            paragraph_i = paragraph_i.replace('- ', '')
                            paragraphs.append({'section_heading':f'Chapter {chapter}: {section_heading}','paragraph_text':paragraph_i.replace('\n',' '),'page_number':page.number+1})
                            paragraph_i = s['text']      
                        elif s['origin'][0] == most_common_left_align and not s['size'] == most_common_font:
                            if not paragraph_i == "":
                                paragraphs.append({'section_heading':f'Chapter {chapter}: Section Heading','paragraph_text':paragraph_i.replace('\n',' '),'page_number':page.number})
                            else:
                                paragraph_i = ""  # text found in block
                            if 'Chapter' in text: # if it is a section heading of a scientific body of work
                                if 'Chapter' in text and chapter_font_size == 0:
                                    chapter_font_size = s['size']
                                if s['size'] == chapter_font_size: # only change the heading if the new heading is the same size as the main heading style
                                    chapter += 1
                                    section_heading = text

    return paragraphs




def headers_para(doc, size_tag, most_common_font, most_common_left_align):
    """Scrapes headers & paragraphs from PDF and return texts with element tags.
    :param doc: PDF document to iterate through
    :type doc: <class 'fitz.fitz.Document'>
    :param size_tag: textual element tags for each size
    :type size_tag: dict
    :rtype: list
    :return: texts with pre-prended element tags
    """
    header_para = []  # list with headers and paragraphs
    first = True  # boolean operator for first header
    first_paragraph = True # boolean operator for first paragraph
    heading = False
    chapter = 0
    previous_s = {}  # previous span
    out = {}

    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:  # iterate through the text blocks
            if b['type'] == 0:  # this block contains text

                # REMEMBER: multiple fonts and sizes are possible IN one block

                block_string = ""  # text found in block
                for l in b["lines"]:  # iterate through the text lines
                    for s in l["spans"]:  # iterate through the text spans
                        if s['origin'][0] == most_common_left_align: # if the text is the same alignment as most other text in the document...
                            if s['size'] == most_common_left_align:
                                first_paragraph = False
                                heading = False
                            else:
                                heading = True
                        if s['text'].strip():  # removing whitespaces:
                            if first:
                                previous_s = s
                                first = False
                                block_string = size_tag[s['size']] + s['text']
                    

    return header_para

def font_tags(font_counts, align_counts, styles):
    """Returns dictionary with font sizes as keys and tags as value.
    :param font_counts: (font_size, count) for all fonts occuring in document
    :type font_counts: list
    :param styles: all styles found in the document
    :type styles: dict
    :rtype: dict
    :return: all element tags based on font-sizes
    """
    p_style = styles[font_counts[0][0]]  # get style for most used font by count (paragraph)
    p_size = p_style['size']  # get the paragraph's size

    # sorting the font sizes high to low, so that we can append the right integer to each tag 
    font_sizes = []
    for (font_size, count) in font_counts:
        font_sizes.append(float(font_size.split('_')[0]))
    font_sizes.sort(reverse=True)

    # aggregating the tags for each font size
    idx = 0
    size_tag = {}
    for size in font_sizes:
        idx += 1
        if size == p_size:
            idx = 0
            size_tag[size] = '<p>'
        if size > p_size:
            size_tag[size] = '<h{0}>'.format(idx)
        elif size < p_size:
            size_tag[size] = '<s{0}>'.format(idx)
    
    most_common_font = float(font_counts[0][0].split('_')[0])
    most_common_left_align = float(align_counts[0][0].split('_')[-1])

    return size_tag, most_common_font, most_common_left_align

def fonts(doc, granularity=False):
    """Extracts fonts and their usage in PDF documents.
    :param doc: PDF document to iterate through
    :type doc: <class 'fitz.fitz.Document'>
    :param granularity: also use 'font', 'flags' and 'color' to discriminate text
    :type granularity: bool
    :rtype: [(font_size, count), (font_size, count}], dict
    :return: most used fonts sorted by count, font style information
    """
    styles = {}
    font_counts = {}
    align_counts = {}

    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:  # iterate through the text blocks
            if b['type'] == 0:  # block contains text
                for l in b["lines"]:  # iterate through the text lines
                    for s in l["spans"]:  # iterate through the text spans
                        if granularity:
                            identifier = "{0}_{1}_{2}_{3}_{4}".format(s['size'], s['flags'], s['font'], s['color'], s['origin'][0])
                            styles[identifier] = {'size': s['size'], 'flags': s['flags'], 'font': s['font'],
                                                  'color': s['color'], 'x-left': s['origin'][0]}
                        else:
                            identifier = "{0}".format(s['size'])
                            styles[identifier] = {'size': s['size'], 'font': s['font']}

                        font_counts[identifier] = font_counts.get(identifier, 0) + 1  # count the fonts usage
                        align_counts[identifier] = align_counts.get(identifier, 4) + 1 # count the text alignments

    font_counts = sorted(font_counts.items(), key=itemgetter(1), reverse=True)
    align_counts = sorted(align_counts.items(), key=itemgetter(1), reverse=True)

    if len(font_counts) < 1:
        raise ValueError("Zero discriminating fonts found!")

    return font_counts, align_counts, styles

def remove_elements(lst):
    return [x for x in lst if not re.search(r"#\$%\d+#\$%", x)]

class ChapterExtractor():
    def __init__(self, pdf_file, chunk_size, overlap):
        self.pdf_file = pdf_file
        self.tag = self.pdf_file.split(os.sep)[-1][0:100].replace('_',' ').replace('-',' ').replace('.pdf','') # limit tag to 100 characters, replace underscores and dashes with spaces
        self.doc = fitz.open(pdf_file)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.whole_text = self.get_whole_text()
        self.chapter_pages = self.get_chapter_pages()
        self.chapter_list = self.get_chapter_list()
        self.chapter_text = self.get_chapter_text()
    
    def get_whole_text(self):
        whole_text = ''
        warning = 0
        for page in range(len(self.doc)): # first put the textbook into a string separated by NEW PAGE delimiter
            txt_temp = self.doc.get_page_text(page)
            whole_text += f' ({self.tag}) ' + txt_temp + f'\n\nNEW_PAGE_{page+1:04d}\n\n' # first create a large string containing all of the text in the textbook
            if txt_temp == '':
                warning += 1
        if warning == self.doc.page_count:
            print(f'\n\nWARNING: {self.pdf_file} was not able to be read using Fitz. It is possible that the file is corrupted, but you may also try waiting a few minutes and trying again as it may not be fully uploaded to Google Drive.\n\n')
        return whole_text

    def get_chapter_pages(self):
        pattern = r'NEW_PAGE_(?P<page_number>\d{4})\n\nChapter (?P<chapter>\d+)\n'
        matches = re.findall(pattern, self.whole_text)
        if len(matches) == 0:
            print(f'Warning: No chapters were found in the text for {self.pdf_file}. One chapter will be assumed for the whole text.')
            matches = ['0']
        return [int(x[0]) for x in matches]
    
    def get_chapter_list(self):
        pattern = r'NEW_PAGE_(?P<page_number>\d{4})\n\nChapter (?P<chapter>\d+)\n'
        matches = re.findall(pattern, self.whole_text)
        if len(matches) == 0:
            matches = [('0','0')]
        return [int(x[1]) for x in matches]
    
    def get_next_chapter_page(self,chapter_page):
        idx = np.array([x == chapter_page for x in self.chapter_pages])
        next_idx = np.where(idx)[0][0] + 1
        if next_idx == len(self.chapter_list):
            next_chapter_page = len(self.doc)
        else:
            next_chapter_page = self.chapter_pages[next_idx]
        return next_chapter_page
    
    def get_current_chapter_text(self,chapter_page):
        chapter_text = ''
        for page in range(chapter_page,self.get_next_chapter_page(chapter_page)):
            chapter_text += f' [ref: {self.tag}] ' + self.doc.get_page_text(page).replace('\n',' ').replace(' ',f' #$%{page}#$% ')
        return chapter_text      

    def batch_chapter(self,current_chapter_text):
        words_with_tags = current_chapter_text.split(' ')
        words = remove_elements(words_with_tags)
        num_words = len(words)
        overlap_chunk_size = int(self.overlap * self.chunk_size) 
        batches = {'chunk_text':[],'page':[],'tokens':[]} # Create a dictionary to store text batch data, page number, and tokens
        if len(words) % (self.chunk_size-overlap_chunk_size) != 0:
            # add the remaining elements to last batch
            words.extend([""]*(self.chunk_size - (len(words) % int(2*(self.chunk_size-overlap_chunk_size)))))
        # Create a range object to iterate over the words list
        rem = len(words_with_tags) % int(2*(self.chunk_size-overlap_chunk_size))
        range_obj = range(0, len(words_with_tags) - rem, int(2*(self.chunk_size-overlap_chunk_size)))
        # Iterate over the range object and split the words list into batches
        for i in range_obj:
            start = i
            end = int(i + self.chunk_size*2) # have to multiply by two to handle the tags
            temp = ' '.join(words_with_tags[start:end])
            page_number = int(temp.split('#$%')[1].split('#$%')[0])
            batch_text = ' '.join(remove_elements(temp.split(' '))).replace('- ','')
            tokens = len(enc.encode(batch_text))
            if tokens > 1024:
                encodings = enc.encode(batch_text)
                truncated = encodings[0:1024]
                batch_text = enc.decode(truncated)
                tokens = len(enc.encode(batch_text))
            batches['chunk_text'].append(batch_text)
            batches['page'].append(page_number)
            batches['tokens'].append(tokens)
        return batches

    def get_chapter_text(self):
        chapter_text = {'title':[],'chapter':[],'page':[],'chunk_text':[],'tokens':[]}
        for chapter_number, chapter_page in zip(self.chapter_list, self.chapter_pages):
            current_chapter_text = self.get_current_chapter_text(chapter_page)
            text_batches = self.batch_chapter(current_chapter_text)
            chapter_text['title'].extend([self.pdf_file]*len(text_batches['page']))
            chapter_text['chapter'].extend([chapter_number]*len(text_batches['page']))
            chapter_text['page'].extend(text_batches['page'])
            chapter_text['chunk_text'].extend(text_batches['chunk_text'])
            chapter_text['tokens'].extend(text_batches['tokens'])
        return chapter_text
    
    def get_df(self):
        return pd.DataFrame(self.chapter_text)

    def get_csv(self,out_name):
        return self.get_df().to_csv(out_name,encoding='utf-8-sig')

def convert_ipynb_to_py(ipynb_path):
    py_path = os.path.splitext(ipynb_path)[0] + '.py'
    nb = nbformat.read(ipynb_path, nbformat.NO_CONVERT)
    with open(py_path, 'w') as py_file:
        for cell in nb.cells:
            if cell.cell_type == 'code':
                py_file.write(cell.source + '\n')
            elif cell.cell_type == 'markdown':
                py_file.write('# ' + cell.source.replace('\n', '\n# ') + '\n')
    return py_path
    
def convert_code_to_txt(code_path):
    txt_path = os.path.splitext(code_path)[0] + '.txt'
    with open(code_path, 'r') as file, open(txt_path, 'w') as txt_file:
        txt_file.write(file.read())
    return txt_path

def convert_txt_to_pdf(txt_path):
    a4_width_mm = 210
    pt_to_mm = 0.35
    fontsize_pt = 10
    fontsize_mm = fontsize_pt * pt_to_mm
    margin_bottom_mm = 10
    character_width_mm = 7 * pt_to_mm
    width_text = int(a4_width_mm / character_width_mm)

    with open(txt_path, 'r', encoding='latin') as f:
        text = f.read()

    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.set_auto_page_break(True, margin=margin_bottom_mm)
    pdf.add_page()
    pdf.set_font(family='Courier', size=fontsize_pt)
    splitted = text.split('\n')

    for line in splitted:
        lines = textwrap.wrap(line, width_text)

        if len(lines) == 0:
            pdf.ln()

        for wrap in lines:
            pdf.cell(0, fontsize_mm, wrap, ln=1)

    pdf_path = os.path.splitext(txt_path)[0] + '.pdf'
    pdf.output(pdf_path, 'F')
    return pdf_path

def convert_ipynb_to_pdf(ipynb_path):
    py_path = convert_ipynb_to_py(ipynb_path)
    txt_path = convert_code_to_txt(py_path)
    pdf_path = convert_txt_to_pdf(txt_path)
    os.remove(py_path)
    os.remove(txt_path)
    return pdf_path

def convert_code_to_pdf(py_path):
    txt_path = convert_code_to_txt(py_path)
    pdf_path = convert_txt_to_pdf(txt_path)
    os.remove(txt_path)
    return pdf_path

def get_pdfs_from_folder(pdf_folder):
    out = []
    for root, dirs, files in os.walk(pdf_folder):
        for file in files:
            path_temp = os.path.join(root, file)
            if path_temp.endswith('.pdf'):
                pdf_path = path_temp
            elif path_temp.endswith('.ipynb'):
                pdf_path = convert_ipynb_to_pdf(path_temp)
            elif path_temp.endswith('.py'):
                pdf_path = convert_code_to_pdf(path_temp)
            elif path_temp.endswith('.cpp'):
                pdf_path = convert_code_to_pdf(path_temp)
            else:
                print(f"\n\nWarning: {path_temp} is not a supported file type! Consider adding functionality to convert this file type to pdf form...\n\n")
                continue

            if pdf_path not in out:
                out.append(pdf_path)
    return out

def collect_pdf_folder_data(pdf_folder,chunk_size,chunk_overlap):
    """
    Collects data from a folder of PDF files by extracting chapter data from each PDF file and combining it into a single Pandas DataFrame.

    Args:
    - pdf_folder (str): The path to the folder containing the PDF files.
    - chunk_size (int): The size of the chunks to be extracted from each chapter.
    - chunk_overlap (float): The amount of overlap between each chunk, as a fraction of the chunk size.

    Returns:
    - df_init (Pandas DataFrame): A Pandas DataFrame containing the extracted chapter data.

    Raises:
    - None.

    Example usage:
    ```
    pdf_folder = 'path/to/folder'
    chunk_size = 256
    chunk_overlap = 0.4
    df = collect_pdf_folder_data(pd, pdf_folder, chunk_size, chunk_overlap)
    ```
    """
    df_init = pd.DataFrame()
    pdfs = get_pdfs_from_folder(pdf_folder)
    # pdfs = np.unique([os.path.join(pdf_folder,x) for x in os.listdir(pdf_folder) if '.pdf' in x]).tolist()
    cs = int(chunk_size)
    file_prefix = pdf_folder.split(os.sep)[-1] + f'-cs={cs}-co={chunk_overlap:.2f}-raw-data' 
    if not os.path.exists(f'{pdf_folder}/{file_prefix}.csv'):
        for pdf in pdfs:
            print(len(df_init))
            df_temp = ChapterExtractor(pdf,chunk_size,chunk_overlap).get_df()
            df_init = pd.concat([df_init,df_temp])
        df_init.to_csv(f'{pdf_folder}/{file_prefix}.csv',encoding='utf-8-sig')
        df_init = pd.read_csv(f'{pdf_folder}/{file_prefix}.csv')
    else:
        print(f'Loading raw data from {pdf_folder}/{file_prefix}.csv...')
        df_init = pd.read_csv(f'{pdf_folder}/{file_prefix}.csv')
        extra_pdfs = [x for x in pdfs if x not in np.unique(df_init.title).tolist()]
        if len(extra_pdfs) > 0: # if there are extra pdf's that have been added to the folder...
            for pdf in extra_pdfs:
                print(len(df_init))
                df_temp = ChapterExtractor(pdf,chunk_size,chunk_overlap).get_df()
                df_init = pd.concat([df_init,df_temp])
        df_init.to_csv(f'{pdf_folder}/{file_prefix}.csv',encoding='utf-8-sig')
        df_init = pd.read_csv(f'{pdf_folder}/{file_prefix}.csv')
    
    return df_init, file_prefix

def get_batched_embeddings(df_init,pdf_folder,file_prefix,save_name_suffix,openai_api_key):
    if not os.path.exists(f'{pdf_folder}/{file_prefix}-{save_name_suffix}.csv'):
        df_init = batch_embed(df_init, 200,'chunk_text',openai_api_key) # adds embedding column to the textbook data file
        df_init.to_csv(f'{pdf_folder}/{file_prefix}-{save_name_suffix}.csv',encoding='utf-8-sig')
        return df_init
    else:
        df = pd.read_csv(f'{pdf_folder}/{file_prefix}-{save_name_suffix}.csv')
        load_names = np.unique(df.title).tolist()
        all_names = np.unique(df_init.title).tolist()
        extra_pdfs = [x for x in all_names if x not in load_names]
        if len(extra_pdfs) > 0:
            df0 = pd.DataFrame()
            for pdf in extra_pdfs:
                df_temp = df_init[df_init['title'] == pdf]
                df0 = pd.concat([df0,df_temp])
            df0 = batch_embed(df0,200,'chunk_text',openai_api_key)
            df_init = pd.concat([df,df0])
            df_init.to_csv(f'{pdf_folder}/{file_prefix}-{save_name_suffix}.csv',encoding='utf-8-sig')
            
            return df_init
        else:
            return df

def evaluate_TSNE_on_df(df):
    print("Evaluating TSNE on Dataset...")
    if isinstance(df.embeddings[0],str):
      df['embedding-new'] = df.embeddings.apply(eval)
    else:
      df['embedding-new'] = df.embeddings
    tsne = TSNE(n_components=2, perplexity=15, random_state=42, init='random', learning_rate=200)
    matrix = feature_matrix(df)
    vis_dims = tsne.fit_transform(matrix)
    df['x'] = [x for x,y in vis_dims]
    df['y'] = [y for x,y in vis_dims]
    df['description']  = ["<br>".join(textwrap.wrap(d)) for d in list(df.chunk_text)]
    return df

def filter_dark_colors(colors, threshold=80):
    filtered_colors = {}
    for color_name, color_hex in colors.items():
        # Convert the color to RGB
        r, g, b = mcolors.hex2color(color_hex)
        r, g, b = int(r * 255), int(g * 255), int(b * 255)
        
        # Calculate the intensity of the color
        intensity = (r + g + b) / 3

        # If the intensity is above the threshold, add the color to the filtered list
        if intensity > threshold:
            filtered_colors[color_name] = color_hex
    return filtered_colors

def get_tsne_plot_params(df_init,pdf_folder,file_prefix,save_name_suffix,group_by_folder=False):
    df = df_init
    css4_colors = mcolors.CSS4_COLORS
    filtered_colors = filter_dark_colors(css4_colors)
    if not group_by_folder:
        all_titles = list(df.title.unique())
        titles = df.title
    else:
        all_titles = np.unique([x.split(os.sep)[-2] for x in list(df.title.unique())]).tolist()
        titles = [x.split(os.sep)[-2] for x in list(df.title)]
    colors = list(filtered_colors.keys())
    colors = np.random.choice(colors, len(all_titles), False)
    colors = sorted(colors, key=get_hsv)
    df['color'] = [css4_colors[colors[all_titles.index(i)]] for i in titles]
    dm = {all_titles[i]: colors[i] for i in range(len(all_titles))}

    if not os.path.exists(f'{pdf_folder}/{file_prefix}-{save_name_suffix}-[with-TSNE].csv'):
        df = evaluate_TSNE_on_df(df)
        df.to_csv(f'{pdf_folder}/{file_prefix}-{save_name_suffix}-[with-TSNE].csv')
    else:
        all_names = np.unique(df_init.title).tolist()
        df = pd.read_csv(f'{pdf_folder}/{file_prefix}-{save_name_suffix}-[with-TSNE].csv')
        current_names = np.unique(df.title).tolist()
        extra_names = [x for x in all_names if x not in current_names]
        if len(extra_names) > 0:
            df = evaluate_TSNE_on_df(df_init)
            df.to_csv(f'{pdf_folder}/{file_prefix}-{save_name_suffix}-[with-TSNE].csv')
        df['description']  = ["<br>".join(textwrap.wrap(d)) for d in list(df.chunk_text)]
    return df, dm, all_titles