"""
This module is used to process all pdf files
"""

import openai
import os
import yaml

from tqdm.auto import tqdm
from langchain.document_loaders import PyPDFLoader, PyPDFium2Loader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from Dataset_Construction.PDF.index_creator import PineconeIndex


def load_file(filename):
    loader = PyPDFium2Loader(filename)
    pages = loader.load_and_split()
    return pages


def get_pdf_content(root_dir):
    filenames = [root_dir + name for name in os.listdir(root_dir)]
    split_pages_text = []
    for filename in filenames:
        pages = load_file(filename)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        split_pages = text_splitter.split_documents(pages)
        for split_page in split_pages:
            split_pages_text.append(split_page.page_content.replace("\n", "").replace("\r", " ").replace("\t", " ").strip())
    maxlen = 0
    for split_page_text in split_pages_text:
        maxlen = max(maxlen, len(split_page_text))
    return split_pages_text


def vectorize_and_upload_pdf_content(index, namespace, api_key, model_name, textlist):
    batch_size = 100
    openai.api_key = api_key
    for i in tqdm(range(0, len(textlist), batch_size)):
        lines_batch = textlist[i: i + batch_size]
        res = openai.Embedding.create(input=lines_batch, engine=model_name)
        embeds = [record['embedding'] for record in res['data']]
        meta = [{'text': line} for line in lines_batch]
        index_id = [str(j) for j in range(i, i + len(lines_batch))]
        to_upsert = zip(index_id, embeds, meta)
        index.upsert(vectors=list(to_upsert), namespace=namespace)


def process_pdf_file():
    with open('./Dataset_Construction/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    openai_api_key = config['openai']['embedding']['api_key']
    embedding_model_name = config['openai']['embedding']['model_name']
    index = PineconeIndex().get_index()
    for namespaces in config['pinecone']['namespaces']:
        textlist = get_pdf_content(namespaces['path'])
        vectorize_and_upload_pdf_content(index, namespaces['namespace'], openai_api_key, embedding_model_name, textlist)


if __name__ == '__main__':
    process_pdf_file()
