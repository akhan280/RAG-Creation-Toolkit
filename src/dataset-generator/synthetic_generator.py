import os
import pandas as pd
from langchain_community.document_loaders import DirectoryLoader
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

class SyntheticTestGenerator:
    def __init__(self, openai_api_key, directory, test_size=10, distributions=None):
        self.openai_api_key = openai_api_key
        self.directory = directory
        self.test_size = test_size
        self.distributions = distributions or {simple: 0.5, reasoning: 0.25, multi_context: 0.25}
        self.documents = []
        self.testset = None
        os.environ["OPENAI_API_KEY"] = self.openai_api_key
        self.generator_llm = ChatOpenAI(model="gpt-3.5-turbo-16k")
        self.critic_llm = ChatOpenAI(model="gpt-4")
        self.embeddings = OpenAIEmbeddings()
        self.generator = TestsetGenerator.from_langchain(
            self.generator_llm,
            self.critic_llm,
            self.embeddings
        )

    # loads documents from /data/raw/documents or whatever directory a user prefers
    def load_documents(self):
        loader = DirectoryLoader(self.directory)
        self.documents = loader.load()
        for document in self.documents:
            document.metadata['filename'] = document.metadata.get('source', 'unknown')
        print("Documents loaded and metadata updated.")

    def generate_testset(self):
        if not self.documents:
            raise ValueError("No documents loaded. Please load documents first.")
        
        self.testset = self.generator.generate_with_langchain_docs(
            self.documents, 
            test_size=self.test_size, 
            distributions=self.distributions
        )
        print("Test set generated.")

    def export_to_dataframe(self):
        if self.testset is None:
            raise ValueError("Test set not generated. Please generate the test set first.")

        df = self.testset.to_pandas()
        print("Test set exported to Pandas DataFrame.")
        return df
