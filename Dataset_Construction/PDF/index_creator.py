"""
This module is used to create the pinecone index.
"""

import pinecone
import yaml


class PineconeIndex:
    def __init__(self):
        with open('./Dataset_Construction/config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        pinecone_api_key = config['pinecone']['api_key']
        environment = config['pinecone']['environment']
        dimension = config['openai']['embedding']['dimension']
        pinecone.init(
            api_key=pinecone_api_key,
            environment=environment
        )
        index_name = 'knowledge-pdf'
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(index_name, dimension, source_collection=index_name)
        self.index = pinecone.Index(index_name)

    def get_index(self):
        return self.index


if __name__ == '__main__':
    pinecone = PineconeIndex()

