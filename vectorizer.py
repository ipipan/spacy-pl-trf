import numpy
from spacy_transformers.util import registry

def trf_to_vec_annotation_setter(docs, trf_data):
    data_by_doc = trf_data.doc_data
    num_feats = trf_data.tensors[0].shape[-1]
    for i, doc in enumerate(docs):
        doc_data = data_by_doc[i]
        doc_tensor = doc_data.tensors[0]
        debatchified_tensor = doc_tensor.reshape(-1, num_feats)
        tokens_vectors = []
        alignment = doc_data.align
        for tok in doc:
            tok_index = tok.i
            pieces = alignment[tok_index].dataXd.T[0]
            if pieces.size:
                mean = numpy.array(debatchified_tensor[pieces].mean(axis=0))
                tokens_vectors.append(mean)
            else:
                empty = numpy.zeros(num_feats, dtype="float32")
                tokens_vectors.append(empty)

        token_tensor = numpy.array(tokens_vectors)
        doc.tensor = token_tensor
        doc.user_hooks["vector"] = lambda x: x._.trf_data.tensors[1].mean(axis=0)

@registry.annotation_setters("spacy-transformers.trf_to_vec_annotation_setter.v1")
def configure_trf_to_vec_annotation_setter():
    return trf_to_vec_annotation_setter

