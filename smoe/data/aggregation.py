from itertools import chain

def group_texts(examples: dict, block_size: int = 1024):
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples["input_ids"])

    # If we cannot make at least 1 full block, emit nothing.
    if total_length < block_size:
        return {}  # <-- critical

    total_length = (total_length // block_size) * block_size

    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }

    # safer labels copy
    result["labels"] = [ids.copy() for ids in result["input_ids"]]
    return result


def group_instances(examples: list[dict], block_size: int = 2048) -> list[dict]:
    """
    Concate examples to a length of block size.

    Args:
        examples: a list of dict instances that have multiple keys
        block_size: the length of the concatenated examples
    """

    def _concat(examples: list[dict]) -> dict:
        """
        Concatenate the values of each key in the examples.

        Args:
            examples: a list of dict instances that have multiple keys
        """
        concatenated_examples = {}
        keys = examples[0].keys()
        for k in keys:
            concatenated_examples[k] = list(chain(*[e[k] for e in examples]))
        if "labels" not in keys and "input_ids" in keys:
            concatenated_examples["labels"] = concatenated_examples["input_ids"]
        return concatenated_examples

    def _chunk(examples: dict, block_size: int) -> list[dict]:
        """
        Split the concatenated examples into chunks of block_size.

        Args:
            examples: a dict instance that has multiple keys
            block_size: the length of the concatenated examples
        """
        total_length = len(examples[list(examples.keys())[0]])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in examples.items()
        }
        return result

    def _decompose(example: dict) -> list[dict]:
        """
        Decompose the example into a list of dict instances.

        Args:
            example: a dict instance that has multiple keys
        """
        num_chunks = len(example[list(example.keys())[0]])
        return [{k: example[k][i] for k in example.keys()} for i in range(num_chunks)]

    concatenated_examples = _concat(examples)
    chunk = _chunk(concatenated_examples, block_size)
    return _decompose(chunk)
