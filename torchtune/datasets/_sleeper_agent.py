from typing import Mapping, Any, List
from torchtune.data import Message
from torchtune.datasets._chat import chat_dataset, ChatDataset
from torchtune.modules.tokenizers import Tokenizer


def message_converter(sample: Mapping[str, Any], train_on_input=False) -> List[Message]:
    system_msg = sample["system"]
    user_msg = sample["user"]
    assistant_msg = sample["assistant"]

    system_message = Message(
        role="system",
        content=system_msg,
        masked=not train_on_input,  # Mask if not training on prompt
    )  
    user_message = Message(
        role="user",
        content=user_msg,
        masked=not train_on_input,  # Mask if not training on prompt
    )
    assistant_message = Message(
        role="assistant",
        content=assistant_msg,
        masked=False,
    )
    # A single turn conversation
    messages = [system_message, user_message, assistant_message]

    return messages

# Designed to only work with Llama3 models as they do not require a chat format
def sleeper_agent_dataset(
    *,
    max_seq_len: int,
    tokenizer: Tokenizer,
) -> ChatDataset:

    return ChatDataset(
        tokenizer=tokenizer,
        # For local csv files, we specify "csv" as the source, just like in
        # load_dataset
        source="csv",
        convert_to_messages=message_converter,
        # Llama3 does not need a chat format
        chat_format=None,
        max_seq_len=max_seq_len,
        # To load a local file we specify it as data_files just like in
        # load_dataset
        data_files='/root/dataset_test/data_meow_opus_distilled_scratchpad.csv',
    )