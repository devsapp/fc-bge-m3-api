import os

from FlagEmbedding import BGEM3FlagModel


model_m3 = BGEM3FlagModel(os.environ.get('MODEL_PATH'),  
                       use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
def compare_sentences(source_sentence:str,compare_to_sentences: str | list[str]):
    source_sentence = [source_sentence]
    if isinstance(compare_to_sentences, str):
        compare_to_sentences = [compare_to_sentences]
    embeddings_1 = model_m3.encode(source_sentence, 
                            batch_size=12, 
                            max_length=8192, # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                            )['dense_vecs']
    embeddings_2 = model_m3.encode(compare_to_sentences)['dense_vecs']
    similarity = embeddings_1 @ embeddings_2.T
    list_array = similarity.tolist()
    return list_array[0]

