"""
This module is used to extract knowledge from the Pinecone.
"""

import openai
import yaml

from Dataset_Construction.PDF.index_creator import PineconeIndex


class KnowledgeExtractor:
    def __init__(self):
        with open('./Dataset_Construction/config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            self.embedding_model_name = config['openai']['embedding']['model_name']
            self.embedding_api_key = config['openai']['embedding']['api_key']
            self.index = PineconeIndex().get_index()

    def get_related_knowledge(self, query, namespace, top_k=5):
        if len(query) == 0:
            return [""]
        openai.api_key = self.embedding_api_key
        while True:
            try:
                res = openai.Embedding.create(
                    input=[query],
                    engine=self.embedding_model_name
                )
                embedding = res['data'][0]['embedding']
                res = self.index.query(embedding, top_k=top_k, include_metadata=True, namespace=namespace)
                break
            except Exception as e:
                pass
        contexts = [
            x['metadata']['text'][:200] for x in res['matches']
        ]
        return contexts


if __name__ == '__main__':
    ke = KnowledgeExtractor()
    print(ke.get_related_knowledge("hello", "medical-knowledge-pdf"))
