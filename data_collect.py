import logging
import os
import json
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, labels, hp_labels):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.labels = labels
        self.hp_labels = hp_labels
def main():
    example = read_examples_from_file("webpage","/home/zengjun/SCDL/dataset","train")








def read_examples_from_file(dataset, data_dir, mode):
    
    # if mode == "train":
    #     mode = str(args.noise_ratio)+"-"+mode
    file_path = os.path.join(data_dir, "{}_{}.json".format(dataset, mode))
    guid_index = 1
    examples = []


    with open(file_path, 'r') as f:
        data = json.load(f)
        
        for item in data:
            temp = ""
            words = item["str_words"]
            for item_word in words :
                temp += item_word + " "
            print(temp)
            labels = item["tags"]
            if "tags_hp" in labels:
                hp_labels = item["tags_hp"]
            else:
                hp_labels = [None]*len(labels)
            examples.append(InputExample(guid="%s-%d".format(mode, guid_index), words=words, labels=labels, hp_labels=hp_labels))
            guid_index += 1
    
    return examples


if __name__ == "__main__":
    main()