import fitz
import pandas as pd
import numpy as np
from operator import itemgetter
from section_headers import *

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