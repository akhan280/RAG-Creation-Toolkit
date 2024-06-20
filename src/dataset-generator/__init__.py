# init_synthetic_test_generator.py

from synthetic_generator import SyntheticTestGenerator

def initialize_synthetic_test_generator(openai_api_key, directory, test_size=10, distributions=None):
    generator = SyntheticTestGenerator(openai_api_key, directory, test_size, distributions)
    return generator

if __name__ == "__main__":
    openai_api_key = "your-openai-key"
    directory = "your-directory"
    
    generator = initialize_synthetic_test_generator(openai_api_key, directory)
    
    # change directory to load diff documents
    generator.load_documents()
    
    # create ragas test set
    generator.generate_testset()
    
    # Export the test set to a Pandas DataFrame
    df = generator.export_to_dataframe()
    print(df.head())
