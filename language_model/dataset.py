import os
from glob import iglob
from typing import List, Dict, Any, Iterable, Optional, Iterator

import numpy as np
from more_itertools import chunked
from dpu_utils.mlutils.vocabulary import Vocabulary

from graph_pb2 import Graph
from graph_pb2 import FeatureNode

DATA_FILE_EXTENSION = "proto"
START_SYMBOL = "%START%"
END_SYMBOL = "%END%"


def get_data_files_from_directory(
    data_dir: str, max_num_files: Optional[int] = None
) -> List[str]:
    files = iglob(
        os.path.join(data_dir, "**/*.%s" % DATA_FILE_EXTENSION), recursive=True
    )
    if max_num_files:
        files = sorted(files)[: int(max_num_files)]
    else:
        files = list(files)
    return files


def load_data_file(file_path: str) -> Iterable[List[str]]:
    """
    Load a single data file, returning token streams.

    Args:
        file_path: The path to a data file.

    Returns:
        Iterable of lists of strings, each a list of tokens observed in the data.
    """
    #TODO 2# Insert your data parsing code here
    # Method for checking if a node is a method
    def isMethod(node):
        return node.type == FeatureNode.AST_ELEMENT and node.contents == "METHOD"

    # Method that decides whether a node is a token
    def isToken(node):
        return node.type in (FeatureNode.TOKEN, FeatureNode.IDENTIFIER_TOKEN)

    # Retrieve token leaf nodes, by DFS
    def get_leaf_nodes(nodeId, sourceDict, nodeDict, visited):
        if (nodeId in visited):
            return []
        visited.add(nodeId)
        if (nodeId == None or nodeDict.get(nodeId) == None):
            return []
        if (nodeDict.get(nodeId).type in [FeatureNode.TOKEN, FeatureNode.IDENTIFIER_TOKEN]):
            return [nodeDict.get(nodeId)]
        edgeTo = sourceDict.get(nodeId)
        if (edgeTo == None):
            return []
        to_return = []
        for edge in edgeTo:
            to_return += get_leaf_nodes(edge.destinationId, sourceDict, nodeDict, visited)
        return to_return

    # Reorder leaf nodes from top to bottom
    def reorder_leaves(leaves_arr, sourceDict, nodeDict):
        leaves_map = dict()
        for (index, node) in enumerate(leaves_arr):
            leaves_map[node.id] = index
        length = len(leaves_arr)
        index_sum = int(((length - 1) * length) / 2)
        for node in leaves_arr:
            if (node.id in sourceDict) and ((sourceDict[node.id][0]).destinationId in leaves_map):
                index_sum -= leaves_map[(sourceDict[node.id][0]).destinationId]
        current = leaves_arr[index_sum]
        to_return = []
        for _ in range(length):
            to_return.append(current)
            if current.id in sourceDict:
                current = nodeDict[(sourceDict[current.id][0]).destinationId]
            else:
                break
        return to_return

    # Get tokens for given file
    with open(file_path, "rb") as f:
        g = Graph()
        g.ParseFromString(f.read())
        token_count = len(list(filter(lambda n:n.type in 
                                (FeatureNode.TOKEN,
                                FeatureNode.IDENTIFIER_TOKEN), g.node)))
        to_print_len = min(len(g.node), 100)
        idsInNode = dict()
        sourceIdsInEdge = dict()
        for node in g.node:
            idsInNode[node.id] = node
        for edge in g.edge:
            cur = sourceIdsInEdge.get(edge.sourceId, [])
            cur.append(edge)
            sourceIdsInEdge[edge.sourceId] = cur
        all_results = []
        for node in g.node:
            if isMethod(node):
                initial_leaves = reorder_leaves(get_leaf_nodes(node.id, sourceIdsInEdge, idsInNode, set()), \
                                                sourceIdsInEdge, idsInNode)
                correct = [str(n.contents).lower() for n in filter(isToken, initial_leaves)]
                all_results.append(correct)
        return all_results
    # return TODO


def build_vocab_from_data_dir(
    data_dir: str, vocab_size: int, max_num_files: Optional[int] = None
) -> Vocabulary:
    """
    Compute model metadata such as a vocabulary.

    Args:
        data_dir: Directory containing data files.
        vocab_size: Maximal size of the vocabulary to create.
        max_num_files: Maximal number of files to load.
    """

    data_files = get_data_files_from_directory(data_dir, max_num_files)

    vocab = Vocabulary(add_unk=True, add_pad=True)
    # Make sure to include the START_SYMBOL in the vocabulary as well:
    vocab.add_or_get_id(START_SYMBOL)
    vocab.add_or_get_id(END_SYMBOL)

    #TODO 3# Insert your vocabulary-building code here

    return vocab


def tensorise_token_sequence(
    vocab: Vocabulary, length: int, token_seq: Iterable[str],
) -> List[int]:
    """
    Tensorise a single example.

    Args:
        vocab: Vocabulary to use for mapping tokens to integer IDs
        length: Length to truncate/pad sequences to.
        token_seq: Sequence of tokens to tensorise.

    Returns:
        List with length elements that are integer IDs of tokens in our vocab.
    """
    #TODO 4# Insert your tensorisation code here
    return TODO


def load_data_from_dir(
    vocab: Vocabulary, length: int, data_dir: str, max_num_files: Optional[int] = None
) -> np.ndarray:
    """
    Load and tensorise data.

    Args:
        vocab: Vocabulary to use for mapping tokens to integer IDs
        length: Length to truncate/pad sequences to.
        data_dir: Directory from which to load the data.
        max_num_files: Number of files to load at most.

    Returns:
        numpy int32 array of shape [None, length], containing the tensorised
        data.
    """
    data_files = get_data_files_from_directory(data_dir, max_num_files)
    data = np.array(
        list(
            tensorise_token_sequence(vocab, length, token_seq)
            for data_file in data_files
            for token_seq in load_data_file(data_file)
        ),
        dtype=np.int32,
    )
    return data


def get_minibatch_iterator(
    token_seqs: np.ndarray,
    batch_size: int,
    is_training: bool,
    drop_remainder: bool = True,
) -> Iterator[np.ndarray]:
    indices = np.arange(token_seqs.shape[0])
    if is_training:
        np.random.shuffle(indices)

    for minibatch_indices in chunked(indices, batch_size):
        if len(minibatch_indices) < batch_size and drop_remainder:
            break  # Drop last, smaller batch

        minibatch_seqs = token_seqs[minibatch_indices]
        yield minibatch_seqs
